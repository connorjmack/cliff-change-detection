#!/usr/bin/env python3
"""
plot_delmar_dashboard_final_v8.py

Generates a 4x1 "Master Dashboard" for Del Mar Coastal Erosion (25cm Resolution).

Corrected Logic:
  1. Volume Calculation: Uses the user's EXACT 'calculate_volume_bounds' function
     from 'plot_cumulative_all_in_one.py'.
  2. Panel 1: Variable width bars, no labels.
  3. Panel 4: White -> Flipped Viridis colormap.

Usage:
    python3 plot_delmar_dashboard_final_v8.py
"""

import os
import re
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from datetime import datetime, timedelta

# --- Configuration ---
LOCATION = 'DelMar'
RESOLUTION = '25cm'
RES_VAL = 0.25
FILE_TAG = '25cm'

# Colors
COLOR_VOL = '#E76F51'       # Burnt Orange
COLOR_CUM = '#2A9D8F'       # Teal
COLOR_COUNT = '#264653'     # Dark Blue
COLOR_SHADING = '#E9ECEF'   # Light Grey for Winter
DPI = 300

# ==============================================================================
# 1. PATHS & HELPERS
# ==============================================================================

def get_base_dir():
    if platform.system() == 'Darwin':
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

def parse_dates(folder_name):
    match = re.search(r'(\d{8})_to_(\d{8})', folder_name)
    if match:
        d1 = datetime.strptime(match.group(1), '%Y%m%d')
        d2 = datetime.strptime(match.group(2), '%Y%m%d')
        return d1, d2
    return None, None

def add_winter_shading(ax, start_date, end_date):
    """Adds shaded bands for Winter (Oct 1 - Mar 31)."""
    start_year = start_date.year - 1
    end_year = end_date.year + 1
    
    for year in range(start_year, end_year):
        winter_start = datetime(year, 10, 1)
        winter_end = datetime(year + 1, 3, 31)
        
        if winter_end < start_date or winter_start > end_date:
            continue
            
        ax.axvspan(winter_start, winter_end, color=COLOR_SHADING, alpha=0.5, zorder=0, linewidth=0)

# ==============================================================================
# 2. DATA LOADING LOGIC (USER REFERENCE IMPLEMENTATION)
# ==============================================================================

def calculate_volume_bounds(grid_path, unc_path, res_val):
    """
    Calculates Volume, Lower Bound, and Upper Bound using cell-by-cell logic.
    Formula: Sum(Area * (Distance +/- Uncertainty))
    
    (Copied verbatim from plot_cumulative_all_in_one.py)
    """
    cell_area = res_val * res_val
    
    if not os.path.exists(grid_path):
        return None, None, None
    
    try:
        # Load Main Grid
        df_grid = pd.read_csv(grid_path, index_col=0)
        df_grid = df_grid.apply(pd.to_numeric, errors='coerce').fillna(0)
    except Exception as e:
        print(f"    [Error] Reading grid {os.path.basename(grid_path)}: {e}")
        return None, None, None

    # Load Uncertainty Grid
    if unc_path and os.path.exists(unc_path):
        try:
            df_unc = pd.read_csv(unc_path, index_col=0)
            df_unc = df_unc.apply(pd.to_numeric, errors='coerce').fillna(0)
            df_unc = df_unc.reindex_like(df_grid).fillna(0)
        except Exception as e:
            df_unc = pd.DataFrame(0, index=df_grid.index, columns=df_grid.columns)
    else:
        df_unc = pd.DataFrame(0, index=df_grid.index, columns=df_grid.columns)

    # Cell-by-cell calculation
    grid_lower = df_grid - df_unc
    grid_upper = df_grid + df_unc
    
    vol_main  = df_grid.values.sum() * cell_area
    vol_lower = grid_lower.values.sum() * cell_area
    vol_upper = grid_upper.values.sum() * cell_area
    
    return vol_main, vol_lower, vol_upper, df_grid

def load_cluster_count(cluster_path):
    if not os.path.exists(cluster_path): return 0
    try:
        df = pd.read_csv(cluster_path, index_col=0)
        vals = df.values.flatten()
        unique_ids = np.unique(vals[~np.isnan(vals)])
        count = len(unique_ids[unique_ids != 0])
        return count
    except:
        return 0

def clean_and_snap_grid(df, resolution_val):
    """Prepares grid for Panel 4 (Cliff Face)"""
    cleaned_cols = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)
    try:
        col_floats = cleaned_cols.astype(float)
        scale = 1.0 / resolution_val
        new_cols = (col_floats * scale).round().astype(int)
        df.columns = new_cols
        df.index = df.index.astype(int)
        return df
    except:
        return None

def collect_dashboard_data(base_dir):
    print(f"Collecting data for {LOCATION} ({RESOLUTION})...")
    erosion_dir = os.path.join(base_dir, 'results', LOCATION, 'erosion')
    
    if not os.path.isdir(erosion_dir):
        print("Error: Erosion directory not found.")
        return [], None

    intervals = sorted([d for d in os.listdir(erosion_dir) 
                       if os.path.isdir(os.path.join(erosion_dir, d))])
    
    stats_list = []
    cumulative_grid = None
    
    for interval in intervals:
        d1, d2 = parse_dates(interval)
        if not d1: continue
        
        folder = os.path.join(erosion_dir, interval)
        grid_file = os.path.join(folder, f"{interval}_ero_grid_{FILE_TAG}_filled.csv")
        unc_file = os.path.join(folder, f"{interval}_ero_uncertainty_{FILE_TAG}.csv")
        clus_file = os.path.join(folder, f"{interval}_ero_clusters_{FILE_TAG}_filled.csv")
        
        if not os.path.exists(grid_file): continue
        
        # 1. Calculate Volumes using EXACT User Function
        # Note: calculate_volume_bounds returns (main, lower, upper)
        # We modified it slightly to return df_grid as 4th output for the spatial map
        # But wait, the user's code only returns 3 values.
        # I will strictly call the user logic for volumes, then reload/use df_grid for the map.
        
        vol_main, vol_low, vol_high, df_grid = calculate_volume_bounds(grid_file, unc_file, RES_VAL)
        
        if vol_main is None: continue

        # 2. Accumulate Spatial Grid (Panel 4)
        if df_grid is not None:
            spatial_df = clean_and_snap_grid(df_grid.copy(), RES_VAL)
            if spatial_df is not None:
                if cumulative_grid is None:
                    cumulative_grid = spatial_df.fillna(0.0)
                else:
                    cumulative_grid = cumulative_grid.add(spatial_df.fillna(0.0), fill_value=0)
        
        # 3. Event Count
        n_events = load_cluster_count(clus_file)
        
        days = (d2 - d1).days
        if days < 1: days = 1
        
        stats_list.append({
            'start': d1,
            'end': d2,
            'mid': d1 + (d2 - d1)/2,
            'days': days,
            'volume': vol_main,
            'vol_lower': vol_low,
            'vol_upper': vol_high,
            'events': n_events
        })
        
        print(f"  {interval}: {vol_main:.1f} m³ (Bounds: {vol_low:.1f} to {vol_high:.1f})")
            
    return stats_list, cumulative_grid

# ==============================================================================
# 3. PLOTTING
# ==============================================================================

def plot_dashboard(stats, cum_grid, out_dir):
    if not stats: return

    stats.sort(key=lambda x: x['end'])

    starts = [s['start'] for s in stats]
    ends = [s['end'] for s in stats]
    mids = [s['mid'] for s in stats]
    widths = [s['days'] for s in stats]
    
    vols = [s['volume'] for s in stats]
    
    # Error Bar Calculation
    # Error bars for Panel 1: Distance from Main Volume to Bounds
    err_lower = [s['volume'] - s['vol_lower'] for s in stats]
    err_upper = [s['vol_upper'] - s['volume'] for s in stats]
    
    # Safety check for plotting: ensure errors aren't negative (though logic ensures they shouldn't be)
    err_lower = [max(0, e) for e in err_lower]
    err_upper = [max(0, e) for e in err_upper]
    yerr = [err_lower, err_upper]
    
    events = [s['events'] for s in stats]
    
    # Cumulative Calculation
    cum_vol = np.cumsum(vols)
    cum_lower = np.cumsum([s['vol_lower'] for s in stats])
    cum_upper = np.cumsum([s['vol_upper'] for s in stats])
    
    dates_cum = ends
    
    # --- SETUP FIGURE ---
    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 1, 1.5, 0.15], hspace=0.45)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3])
    cbar_ax = fig.add_subplot(gs[4])

    global_start = min(starts)
    global_end = max(ends)

    # ==================== PANEL 1: EROSION VOLUME (Variable Width) ====================
    add_winter_shading(ax1, global_start, global_end)
    
    ax1.bar(starts, vols, width=widths, align='edge', 
            color=COLOR_VOL, edgecolor='white', alpha=0.9, label='Erosion Volume')
    
    # Error Bars
    ax1.errorbar(mids, vols, yerr=yerr, fmt='none', ecolor='black', capsize=2, alpha=0.5, elinewidth=1)
    
    ax1.set_ylabel(r"Erosion Vol ($m^3$)", fontsize=12, fontweight='bold')
    ax1.set_title(f"A. Erosion Volume per Interval", loc='left', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # Force 0 to align
    ax1.axhline(0, color='black', linewidth=0.5)

    # ==================== PANEL 2: CUMULATIVE VOLUME ====================
    add_winter_shading(ax2, global_start, global_end)
    
    ax2.fill_between(dates_cum, cum_lower, cum_upper,
                     color=COLOR_CUM, alpha=0.3, label='Uncertainty Envelope')
    
    ax2.plot(dates_cum, cum_vol, color=COLOR_CUM, marker='o', markersize=4, linewidth=2, label='Cumulative Volume')
    
    ax2.set_ylabel(r"Cumulative ($m^3$)", fontsize=12, fontweight='bold')
    ax2.set_title(f"B. Cumulative Erosion Volume", loc='left', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    # Final Stat Box
    total_v = cum_vol[-1]
    total_range_str = f"{cum_lower[-1]:,.0f} – {cum_upper[-1]:,.0f}"
    ax2.text(0.02, 0.85, f"Total: {total_v:,.0f} m³\nRange: {total_range_str} m³", 
             transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round'), fontsize=10)

    # ==================== PANEL 3: EVENT FREQUENCY ====================
    add_winter_shading(ax3, global_start, global_end)
    
    ax3.bar(starts, events, width=widths, align='edge', 
            color=COLOR_COUNT, alpha=0.8, edgecolor='white')
    
    ax3.set_ylabel("Events (Count)", fontsize=12, fontweight='bold')
    ax3.set_title(f"C. Event Frequency (Cluster Count)", loc='left', fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.5)
    ax3.set_ylim(bottom=0)
    
    # X-Axis Formatting
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')

    # ==================== PANEL 4: SPATIAL HEATMAP ====================
    if cum_grid is not None:
        plot_df = cum_grid.T 
        x_poly = plot_df.columns.astype(int)
        y_elev = plot_df.index.astype(int)
        max_elev_m = len(y_elev) * RES_VAL
        extent = [x_poly.min(), x_poly.max(), 0, max_elev_m]
        
        # --- Custom Colormap: White -> Flipped Viridis ---
        # Viridis_r: Yellow(0) -> Purple(1)
        # We want White(0) -> Yellow -> Purple
        viridis_r = cm.get_cmap('viridis_r', 256)
        newcolors = viridis_r(np.linspace(0, 1, 256))
        newcolors[0, :] = np.array([1, 1, 1, 1]) # Set 0 to White
        
        cmap = LinearSegmentedColormap.from_list("WhiteViridisR", newcolors)

        vmax = np.percentile(plot_df.values, 99.5)
        
        im = ax4.imshow(plot_df.values, origin='lower', extent=extent, 
                        cmap=cmap, aspect='auto', interpolation='none',
                        vmin=0, vmax=vmax)
        
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Cumulative Erosion Depth (m)', fontsize=12)
        
        ax4.set_title(f"D. Spatial Distribution (Cliff Face View)", loc='left', fontsize=14, fontweight='bold')
        ax4.set_xlabel("Alongshore Location (Polygon ID)", fontsize=12, fontweight='bold')
        ax4.set_ylabel("Elevation (m)", fontsize=12, fontweight='bold')
        ax4.invert_xaxis()
    else:
        ax4.text(0.5, 0.5, "Grid Data Not Available", ha='center', va='center')
        cbar_ax.axis('off')

    # Final Layout
    plt.suptitle(f"{LOCATION} Coastal Erosion Dashboard ({RESOLUTION})\n{global_start.strftime('%Y-%m')} to {global_end.strftime('%Y-%m')}", 
                 fontsize=18, fontweight='bold', y=0.99)
    
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{LOCATION}_Dashboard_{RESOLUTION}_final_v8.png")
    
    plt.savefig(out_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Dashboard saved to: {out_file}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    base_dir = get_base_dir()
    out_dir = os.path.join(base_dir, "figures", "dashboards")
    
    print(f"--- Generating Dashboard for {LOCATION} ---")
    
    # 1. Collect Data
    stats, cum_grid = collect_dashboard_data(base_dir)
    
    if not stats:
        print("No data found. Exiting.")
        return
        
    # 2. Plot
    plot_dashboard(stats, cum_grid, out_dir)

if __name__ == "__main__":
    main()