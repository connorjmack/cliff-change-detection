#!/usr/bin/env python3
"""
plot_location_dashboard_v11_final.py

Generates a 5x1 "Master Dashboard" for Coastal Erosion Analysis (25cm Resolution).

UPDATES IN V8/V9/V10:
  1. Font sizes increased.
  2. Missing Survey Report restored.
  3. Rate Line is BLACK, Axis labels are BLACK.
  4. Rate is 3-survey rolling average.
  5. Panel B: Removed 'x' markers.
  6. Panel C: Year labels enlarged.
  7. Panel C & D Spacing: Increased.
  8. Panel D Legend: Horizontal, Top Right, No Box.

UPDATES IN V11 (Current):
  1. Panels C & D Color: Orange-Red ('OrRd') mapped to volume.
  2. Panel E Color: Magma ('magma_r') for cumulative heatmap (distinct from bubbles).
  3. Panel D: Added 'Alongshore Location (m)' label and ticks.
  4. Panel C: Added date label to bottom.
"""

import os
import re
import platform
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from datetime import datetime, timedelta

# --- Configuration ---
RESOLUTION = '25cm'
RES_VAL = 0.25
FILE_TAG = '25cm'

# --- COLORS ---
COLOR_MAIN = '#495057'       # Dark Grey (Primary Bars, Lines)
COLOR_SHADING = '#E9ECEF'    # Light Grey for Winter bands
COLOR_RATE = '#000000'       # Black for Rate Line

# --- COLORMAPS ---
# Panels C & D (Bubbles): Orange-Red
CMAP_BUBBLES = 'OrRd'

# Panel E (Heatmap): Magma (Reversed so lighter is 0, darker is deep erosion)
# Or we can use 'magma_r' directly if 0 should be light.
# Let's use 'magma_r' + White for 0 to match previous style.
CMAP_HEATMAP = 'magma_r'

DPI = 300

# --- FONT SIZES ---
FS_TITLE = 18
FS_LABEL = 14
FS_TICK  = 12
FS_LEGEND = 12

# Available locations in processing order
LOCATIONS_ALL = ['DelMar', 'Torrey', 'Solana', 'Encinitas', 'SanElijo']

# ==============================================================================
# 1. PATHS & HELPERS
# ==============================================================================

def get_base_dir():
    if platform.system() == 'Darwin':
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

def get_output_dir(base_dir):
    return os.path.join(base_dir, "figures", "dashboards")

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

def get_custom_cmap(name, vmax=6.0):
    """Creates a custom colormap with White at 0."""
    base_cmap = cm.get_cmap(name, 256)
    newcolors = base_cmap(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 1]) # Force 0 to be White
    return LinearSegmentedColormap.from_list(f"White_{name}", newcolors), Normalize(vmin=0, vmax=vmax)

# ==============================================================================
# 2. DATA LOADING LOGIC
# ==============================================================================

def load_uncertainty_stats(unc_path):
    if not unc_path or not os.path.exists(unc_path): return 0.0
    try:
        df = pd.read_csv(unc_path)
        unc_columns = [col for col in df.columns if col != 'Polygon_ID' and 'Uncertainty' in col]
        if not unc_columns: unc_values = df.iloc[:, 1:].values.flatten()
        else: unc_values = df[unc_columns].values.flatten()
        unc_values = unc_values[~np.isnan(unc_values)]
        unc_values = unc_values[unc_values > 0]
        return np.mean(unc_values) if len(unc_values) > 0 else 0.0
    except:
        return 0.0

def calculate_volume_bounds_properly(grid_path, unc_path, res_val):
    cell_area = res_val * res_val
    if not os.path.exists(grid_path): return None, None, None, None
    try:
        mean_uncertainty = load_uncertainty_stats(unc_path)
        df_grid = pd.read_csv(grid_path, index_col=0)
        df_grid = df_grid.apply(pd.to_numeric, errors='coerce').fillna(0)
        distances = df_grid.values
        erosion_mask = distances > 0.01
        vol_main = distances.sum() * cell_area
        distances_lower = distances.copy()
        distances_lower[erosion_mask] = np.maximum(distances[erosion_mask] - mean_uncertainty, 0)
        vol_lower = distances_lower.sum() * cell_area
        distances_upper = distances.copy()
        distances_upper[erosion_mask] = distances[erosion_mask] + mean_uncertainty
        vol_upper = distances_upper.sum() * cell_area
        return vol_main, vol_lower, vol_upper, df_grid
    except:
        return None, None, None, None

def extract_cluster_events(cluster_path, grid_path, res_val, date_mid):
    if not os.path.exists(cluster_path) or not os.path.exists(grid_path): return []
    try:
        df_c = pd.read_csv(cluster_path, index_col=0).fillna(0)
        df_g = pd.read_csv(grid_path, index_col=0).fillna(0)
        df_c.columns = [c.split('_')[-1] for c in df_c.columns]
        df_g.columns = [c.split('_')[-1] for c in df_g.columns]
        common_index = df_c.index.intersection(df_g.index)
        common_cols = df_c.columns.intersection(df_g.columns)
        if len(common_cols) == 0: return []
        df_c = df_c.loc[common_index, common_cols]
        df_g = df_g.loc[common_index, common_cols]
        try: z_values = [float(re.findall(r"[-+]?\d*\.\d+|\d+", c)[0]) for c in df_c.columns]
        except: z_values = np.arange(len(df_c.columns)) * res_val
        z_values = np.array(z_values)
        try: x_values = df_c.index.astype(int).values
        except: x_values = np.arange(len(df_c.index))

        c_vals = df_c.values
        g_vals = df_g.values
        cell_area = res_val * res_val
        unique_ids = np.unique(c_vals)
        unique_ids = unique_ids[unique_ids != 0] 
        events = []
        for uid in unique_ids:
            mask = (c_vals == uid)
            dists = g_vals[mask]
            vol = np.sum(np.abs(dists)) * cell_area
            rows, cols = np.where(mask)
            z_cells = z_values[cols]
            z_w = np.average(z_cells, weights=np.abs(dists))
            x_cells = x_values[rows]
            x_w_idx = np.average(x_cells, weights=np.abs(dists))
            events.append({'date': date_mid, 'volume': vol, 'x_idx': x_w_idx, 'z': z_w})
        return events
    except Exception as e:
        print(f"    [Error] extracting clusters: {e}")
        return []

def clean_and_snap_grid(df, resolution_val):
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

def collect_dashboard_data(base_dir, location):
    print(f"\nCollecting data for {location} ({RESOLUTION})...")
    erosion_dir = os.path.join(base_dir, 'results', location, 'erosion')
    if not os.path.isdir(erosion_dir): return [], None, []

    intervals = sorted([d for d in os.listdir(erosion_dir) if os.path.isdir(os.path.join(erosion_dir, d))])
    stats_list = []
    cumulative_grid = None
    all_events = [] 
    missing_surveys = []
    
    for interval in intervals:
        d1, d2 = parse_dates(interval)
        if not d1: continue
        folder = os.path.join(erosion_dir, interval)
        grid_file = os.path.join(folder, f"{interval}_ero_grid_{FILE_TAG}_filled.csv")
        unc_file = os.path.join(folder, f"{interval}_ero_uncertainty_{FILE_TAG}.csv")
        clus_file = os.path.join(folder, f"{interval}_ero_clusters_{FILE_TAG}_filled.csv")
        if not os.path.exists(grid_file): continue
        vol_main, vol_low, vol_high, df_grid = calculate_volume_bounds_properly(grid_file, unc_file, RES_VAL)
        if vol_main is None: continue
        if vol_main == 0:
            missing_surveys.append({'Location': location, 'Interval': interval, 'Date': d2.strftime('%Y-%m-%d'), 'Volume': 0, 'Reason': 'Zero Volume'})

        if df_grid is not None:
            spatial_df = clean_and_snap_grid(df_grid.copy(), RES_VAL)
            if spatial_df is not None:
                if cumulative_grid is None: cumulative_grid = spatial_df.fillna(0.0)
                else: cumulative_grid = cumulative_grid.add(spatial_df.fillna(0.0), fill_value=0)
        
        date_mid = d1 + (d2 - d1)/2
        events = extract_cluster_events(clus_file, grid_file, RES_VAL, date_mid)
        all_events.extend(events)
        days = (d2 - d1).days
        if days < 1: days = 1
        stats_list.append({'start': d1, 'end': d2, 'mid': date_mid, 'days': days, 'volume': vol_main, 'vol_lower': vol_low, 'vol_upper': vol_high})
        print(f"  ✓ {interval}: Vol={vol_main:.1f} m3 | {len(events)} events")
            
    return stats_list, cumulative_grid, all_events, missing_surveys

# ==============================================================================
# 3. PLOTTING
# ==============================================================================

def plot_dashboard(stats, cum_grid, all_events, out_dir, location):
    if not stats: return
    stats.sort(key=lambda x: x['end'])

    # Rolling Rate
    df_stats = pd.DataFrame(stats)
    df_stats['raw_rate'] = df_stats['volume'] / df_stats['days']
    df_stats['smooth_rate'] = df_stats['raw_rate'].rolling(window=3, center=True, min_periods=1).mean()
    
    starts, ends, mids, widths = df_stats['start'], df_stats['end'], df_stats['mid'], df_stats['days']
    vols, rates = df_stats['volume'], df_stats['smooth_rate']
    yerr = [[max(0, s['volume'] - s['vol_lower']) for s in stats], [max(0, s['vol_upper'] - s['volume']) for s in stats]]
    
    cum_vol_raw = np.cumsum(vols)
    cum_lower = np.cumsum([s['vol_lower'] for s in stats])
    cum_upper = np.cumsum([s['vol_upper'] for s in stats])

    # Event Data & Colors
    df_events = pd.DataFrame(all_events)
    if not df_events.empty:
        df_events['x_m'] = df_events['x_idx'] * RES_VAL
    df_large = df_events[df_events['volume'] > 5].copy()
    
    # --- COLOR SETUP ---
    VMAX_BUBBLE = 50 
    VMAX_GRID = 6.0  
    
    # Orange-Red for Bubbles
    cmap_bubbles, norm_bubbles = get_custom_cmap(CMAP_BUBBLES, vmax=VMAX_BUBBLE)
    # Magma for Heatmap
    cmap_grid, norm_grid = get_custom_cmap(CMAP_HEATMAP, vmax=VMAX_GRID)

    BUBBLE_SCALE = 3.0 

    # --- FIGURE SETUP ---
    fig = plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(6, 1, height_ratios=[1, 1, 1.5, 2.0, 1.2, 0.15], hspace=0.5)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4], sharex=ax4)
    cbar_ax = fig.add_subplot(gs[5])

    # Time Bounds
    global_start = min(starts)
    global_end = max(ends)
    date_buffer = timedelta(days=15)
    ax1.set_xlim(global_start - date_buffer, global_end + date_buffer)

    # Spatial Bounds
    if cum_grid is not None:
        plot_df = cum_grid.T 
        x_indices = plot_df.columns.astype(int)
        x_meters = x_indices * RES_VAL
        y_elev = plot_df.index.astype(int)
        max_elev_m = len(y_elev) * RES_VAL
        ax4.set_xlim(x_meters.max(), x_meters.min()) 
        ax5.set_xlim(x_meters.max(), x_meters.min())
        ax4.set_ylim(0, max_elev_m)
        ax5.set_ylim(0, max_elev_m)

    # ==================== PANEL A ====================
    add_winter_shading(ax1, global_start, global_end)
    ax1.bar(starts, vols, width=widths, align='edge', color=COLOR_MAIN, edgecolor='white', alpha=0.9)
    ax1.errorbar(mids, vols, yerr=yerr, fmt='none', ecolor='black', capsize=2, alpha=0.5, elinewidth=1)
    ax1.set_ylabel(r"Total Vol ($m^3$)", fontsize=FS_LABEL, fontweight='bold', color='black')
    ax1.set_title(f"A. Total Erosion Volume per Interval", loc='left', fontsize=FS_TITLE, fontweight='bold', color='black')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.tick_params(axis='both', labelsize=FS_TICK, labelcolor='black')

    # ==================== PANEL B ====================
    add_winter_shading(ax2, global_start, global_end)
    ax2.fill_between(ends, cum_lower, cum_upper, color=COLOR_MAIN, alpha=0.3)
    ax2.plot(ends, cum_vol_raw, color=COLOR_MAIN, marker='o', linestyle='-', linewidth=2, markersize=4)
    ax2.set_ylabel(r"Volume ($m^3$)", fontsize=FS_LABEL, fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=FS_TICK)
    ax2.set_title(f"B. Cumulative Erosion & Daily Rate (3-Survey Rolling Avg)", loc='left', fontsize=FS_TITLE, fontweight='bold', color='black')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.tick_params(axis='x', labelsize=FS_TICK, labelcolor='black')

    ax2_twin = ax2.twinx()
    ax2_twin.plot(ends, rates, color=COLOR_RATE, marker=None, linestyle='--', linewidth=1.5)
    ax2_twin.set_ylabel(r"Rate ($m^3/day$)", fontsize=FS_LABEL, fontweight='bold', color='black')
    ax2_twin.tick_params(axis='y', labelcolor='black', labelsize=FS_TICK)

    # ==================== PANEL C: COLORED BUBBLES ====================
    add_winter_shading(ax3, global_start, global_end)
    if not df_large.empty:
        sc3 = ax3.scatter(
            df_large['date'], 
            df_large['x_m'], 
            s=df_large['volume'] * BUBBLE_SCALE, 
            c=df_large['volume'],  # Orange-Red
            cmap=cmap_bubbles,
            norm=norm_bubbles,
            alpha=0.7, 
            edgecolors='black', 
            linewidth=0.3
        )
    
    ax3.set_ylabel("Location (m)", fontsize=FS_LABEL, fontweight='bold', color='black')
    ax3.set_xlabel("Survey Date", fontsize=FS_LABEL, fontweight='bold', color='black') # Added Date Label
    ax3.set_title(f"C. Spatio-Temporal Distribution of Large Events (>5 $m^3$)", loc='left', fontsize=FS_TITLE, fontweight='bold', color='black')
    ax3.grid(True, linestyle=':', alpha=0.5)
    ax3.tick_params(axis='x', labelsize=FS_TICK + 4, labelcolor='black')
    ax3.tick_params(axis='y', labelsize=FS_TICK, labelcolor='black')

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=True, rotation=0, ha='center')

    # ==================== PANEL D: COLORED BUBBLES ====================
    if not df_large.empty:
        # Dummy Legend Handles (Dark Grey)
        sizes = [10, 100, 500]
        points = [plt.scatter([], [], s=s*BUBBLE_SCALE, c='#495057', alpha=0.7, edgecolors='black') for s in sizes]
        labels = [f'{s} $m^3$' for s in sizes]
        
        ax4.legend(points, labels, scatterpoints=1, frameon=False, title="Event Vol", 
                   loc='upper right', ncol=3, columnspacing=1.5, handletextpad=0.5,
                   fontsize=FS_LEGEND, title_fontsize=FS_LEGEND)

        # Main Scatter (Orange-Red)
        sc4 = ax4.scatter(
            df_large['x_m'], 
            df_large['z'], 
            s=df_large['volume'] * BUBBLE_SCALE, 
            c=df_large['volume'], 
            cmap=cmap_bubbles,
            norm=norm_bubbles,
            alpha=0.7, 
            edgecolors='black', 
            linewidth=0.3
        )
        
        # Colorbar for Bubbles
        box = ax4.get_position()
        cbar_ax_d = fig.add_axes([box.x1 + 0.01, box.y0, 0.01, box.height])
        cb_d = plt.colorbar(sc4, cax=cbar_ax_d, orientation='vertical')
        cb_d.set_label(r'Event Volume ($m^3$)', fontsize=FS_LABEL, color='black')
        cb_d.ax.tick_params(labelsize=FS_TICK, labelcolor='black')

    ax4.set_ylabel("Elevation (m)", fontsize=FS_LABEL, fontweight='bold', color='black')
    ax4.set_xlabel("Alongshore Location (m)", fontsize=FS_LABEL, fontweight='bold', color='black') # Added Location Label
    ax4.set_title(f"D. Spatial Distribution of Large Events (>5 $m^3$)", loc='left', fontsize=FS_TITLE, fontweight='bold', color='black')
    ax4.grid(True, linestyle=':', alpha=0.5)
    ax4.tick_params(axis='both', labelsize=FS_TICK, labelcolor='black')
    # Removed set_visible(False) for xticklabels to ensure location ticks show up
    # plt.setp(ax4.get_xticklabels(), visible=False) 

    # ==================== PANEL E ====================
    if cum_grid is not None:
        matrix = plot_df.values
        extent = [x_meters.min(), x_meters.max(), 0, max_elev_m]
        # Magma Heatmap
        im = ax5.imshow(matrix, origin='lower', extent=extent, cmap=cmap_grid, norm=norm_grid, aspect='auto', interpolation='none')
        
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Cumulative Erosion Depth (m)', fontsize=FS_LABEL, color='black')
        cbar.ax.tick_params(labelsize=FS_TICK, labelcolor='black')
        
        ax5.set_title(f"E. Cumulative Cliff Activity Index (Cliff Facing View)", loc='left', fontsize=FS_TITLE, fontweight='bold', color='black')
        ax5.set_xlabel("Alongshore Location (m)", fontsize=FS_LABEL, fontweight='bold', color='black')
        ax5.set_ylabel("Elevation (m)", fontsize=FS_LABEL, fontweight='bold', color='black')
        ax5.tick_params(axis='both', labelsize=FS_TICK, labelcolor='black')
    else:
        ax5.text(0.5, 0.5, "Data Not Available", ha='center', va='center')
        cbar_ax.axis('off')

    plt.suptitle(f"{location} Erosion Dashboard ({RESOLUTION})", fontsize=24, fontweight='bold', y=0.99, color='black')
    
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{location}_Dashboard_{RESOLUTION}_v11.png")
    plt.savefig(out_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {out_file}")
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate V11 Dashboard")
    parser.add_argument('--location', type=str, default='all', help=f"Available: {', '.join(LOCATIONS_ALL)}")
    args = parser.parse_args()
    
    base_dir = get_base_dir()
    out_dir = get_output_dir(base_dir)
    locations = LOCATIONS_ALL if args.location.lower() == 'all' else [args.location]
    all_missing_surveys = []

    print(f"--- Processing {len(locations)} locations ---")
    for loc in locations:
        print(f"\nProcessing: {loc}")
        stats, cum_grid, all_events, missing = collect_dashboard_data(base_dir, loc)
        if missing: all_missing_surveys.extend(missing)
        if stats and all_events: plot_dashboard(stats, cum_grid, all_events, out_dir, loc)
        else: print(f"  [Warning] No data found for {loc}")

    if all_missing_surveys:
        suffix = "ALL" if args.location.lower() == 'all' else args.location
        log_path = os.path.join(out_dir, f"missing_surveys_log_{suffix}.csv")
        df_missing = pd.DataFrame(all_missing_surveys)
        df_missing.to_csv(log_path, index=False)
        print(f"\n[INFO] Logged {len(all_missing_surveys)} missing/blank surveys to: {log_path}")
    else:
        print("\n[INFO] No blank surveys found.")

if __name__ == "__main__":
    main()