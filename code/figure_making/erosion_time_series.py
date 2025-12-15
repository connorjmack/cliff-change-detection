#!/usr/bin/env python3
"""
plot_cumulative_all_in_one.py

Generates a SINGLE cumulative erosion plot per location containing 6 curves:
  - 3 Resolutions (10cm, 25cm, 1m)
  - 2 Types (Filled, Cleaned)

Features:
  - Cumulative uncertainty bounds (shaded bands) for EVERY curve.
  - "Cool" color palette (Blues, Teals, Purples).
  - Cell-by-cell uncertainty calculation logic preserved.

Usage:
    python3 plot_cumulative_all_in_one.py --location SanElijo
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import re
import platform
import sys

# --- Configuration ---

# Define "Cool" Color Palette by Resolution
# We will use these base colors for the 'Filled' lines.
# 'Cleaned' lines will use the same color but dashed/lighter.
RES_COLORS = {
    '10cm': '#2E86AB',  # Strong Blue
    '25cm': '#06D6A0',  # Teal/Green
    '1m':   '#8338EC'   # Purple
}

RESOLUTIONS = [
    {'label': '10cm', 'val': 0.10, 'file_tag': '10cm'},
    {'label': '25cm', 'val': 0.25, 'file_tag': '25cm'},
    {'label': '1m',   'val': 1.00, 'file_tag': '100cm'}
]

TYPES = [
    {'name': 'Filled',  'suffix': 'filled',  'linestyle': '-',  'marker': 'o', 'alpha': 1.0},
    {'name': 'Cleaned', 'suffix': 'cleaned', 'linestyle': '--', 'marker': 'x', 'alpha': 0.7}
]

LOCATIONS_ALL = ['DelMar', 'Solana', 'SanElijo', 'Encinitas', 'Torrey', 'Blacks']

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

def calculate_volume_bounds(grid_path, unc_path, res_val):
    """
    Calculates Volume, Lower Bound, and Upper Bound using cell-by-cell logic.
    Formula: Sum(Area * (Distance +/- Uncertainty))
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
    
    return vol_main, vol_lower, vol_upper

def collect_location_data(location, base_dir):
    """Reads all CSVs and returns a structured dictionary."""
    print(f"\nCollecting data for: {location}")
    erosion_dir = os.path.join(base_dir, 'results', location, 'erosion')
    if not os.path.isdir(erosion_dir):
        print(f"  [Skipping] No erosion directory found at {erosion_dir}")
        return None

    intervals = sorted([d for d in os.listdir(erosion_dir) if os.path.isdir(os.path.join(erosion_dir, d))])
    
    # Structure: location_data[res][type] = list of dicts
    location_data = {r['label']: {t['name']: [] for t in TYPES} for r in RESOLUTIONS}
    
    for interval in intervals:
        start_date, end_date = parse_dates(interval)
        if not start_date: continue
        plot_date = end_date
        folder_path = os.path.join(erosion_dir, interval)
        files = os.listdir(folder_path)
        
        for res in RESOLUTIONS:
            unc_pattern = f"_ero_uncertainty_{res['file_tag']}.csv"
            unc_file = next((f for f in files if unc_pattern in f), None)
            unc_path = os.path.join(folder_path, unc_file) if unc_file else None
            
            for dtype in TYPES:
                grid_suffix = f"_{res['file_tag']}_{dtype['suffix']}.csv"
                grid_file = next((f for f in files if grid_suffix in f and 'grid' in f), None)
                
                if grid_file:
                    grid_path = os.path.join(folder_path, grid_file)
                    v_main, v_low, v_high = calculate_volume_bounds(grid_path, unc_path, res['val'])
                    
                    if v_main is not None:
                        location_data[res['label']][dtype['name']].append({
                            'date': plot_date, 'vol': v_main, 'lower': v_low, 'upper': v_high
                        })
    return location_data

def get_cumulative_series(series_data):
    """Sorts by date and calculates cumulative sums."""
    if not series_data: return None, None, None, None
    series_data.sort(key=lambda x: x['date'])
    dates = [x['date'] for x in series_data]
    cum_vol = np.cumsum([x['vol'] for x in series_data])
    cum_lower = np.cumsum([x['lower'] for x in series_data])
    cum_upper = np.cumsum([x['upper'] for x in series_data])
    return dates, cum_vol, cum_lower, cum_upper

def plot_single_panel_all(location, location_data, out_dir):
    """
    Plots all 6 curves (3 Res x 2 Types) on a single axes with cumulative error bands.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Track max Y to set limits nicely if needed, or let mpl handle it
    
    # Iterate through Resolutions first to group colors
    for res in RESOLUTIONS:
        res_label = res['label']
        base_color = RES_COLORS[res_label]
        
        for dtype in TYPES:
            type_name = dtype['name']
            series_data = location_data[res_label][type_name]
            
            dates, c_vol, c_low, c_high = get_cumulative_series(series_data)
            
            if dates is None or len(dates) == 0:
                continue
            
            # Label for Legend: e.g. "10cm Filled"
            label_str = f"{res_label} {type_name}"
            
            # Plot Main Line
            ax.plot(dates, c_vol, 
                    label=label_str, 
                    color=base_color, 
                    linestyle=dtype['linestyle'],
                    marker=dtype['marker'], 
                    markersize=5, 
                    linewidth=2,
                    alpha=dtype['alpha'])
            
            # Plot Uncertainty Band (Cumulative)
            # Use lighter alpha for Cleaned so Filled stands out, or uniform
            band_alpha = 0.15 if type_name == 'Filled' else 0.05
            
            ax.fill_between(dates, c_low, c_high, 
                            color=base_color, 
                            alpha=band_alpha, 
                            linewidth=0) # No edge on band

    # Formatting
    ax.set_title(f"{location}: Cumulative Erosion (All Resolutions)", fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel(r"Cumulative Volume ($m^3$) $\pm$ Uncertainty", fontsize=14)
    ax.set_xlabel("Survey Date", fontsize=14)
    
    # Grid styling
    ax.grid(True, which='major', linestyle='--', alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)
    ax.minorticks_on()
    
    # Date Axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()
    
    # Legend
    # Move legend outside or to best corner
    ax.legend(title="Resolution & Type", fontsize=11, title_fontsize=12, loc='upper left')

    plt.tight_layout()
    
    out_filename = f"{location}_cumulative_combined.png"
    out_path = os.path.join(out_dir, out_filename)
    plt.savefig(out_path, dpi=300)
    print(f"  ✓ Saved Combined Plot: {out_filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot Single-Panel Cumulative Erosion with 6 Curves")
    parser.add_argument('--location', required=True, help="Location name or 'all'")
    args = parser.parse_args()
    
    base_dir = get_base_dir()
    out_dir = os.path.join(base_dir, "figures", "erosion_time_series_combined")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"--- Cumulative Erosion Plotter (All-in-One) ---")
    print(f"Output Directory: {out_dir}")
    
    if args.location.lower() == 'all':
        locs = LOCATIONS_ALL
    else:
        locs = [args.location]
        
    for loc in locs:
        # 1. Collect Data
        loc_data = collect_location_data(loc, base_dir)
        
        if loc_data:
            # 2. Check for data existence
            has_data = any(len(loc_data[r['label']][t['name']]) > 0 for r in RESOLUTIONS for t in TYPES)
            if has_data:
                # 3. Plot
                plot_single_panel_all(loc, loc_data, out_dir)
            else:
                print(f"  No valid data found for {loc}.")
        
    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()