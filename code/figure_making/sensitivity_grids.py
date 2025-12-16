#!/usr/bin/env python3
"""
plot_grid_sensitivity_dashboard.py

Purpose:
    Generates a 4-panel dashboard comparing erosion results across
    different Grid Resolutions (10cm, 25cm, 1m).
    Follows the aesthetic and logic of existing manuscript figures.

Usage:
    python3 code/figure_making/plot_grid_sensitivity_dashboard.py --location DelMar
"""

import os
import re
import platform
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates

# ==============================================================================
# 1. CONFIGURATION & AESTHETICS
# ==============================================================================

# Define the resolutions to test
# Format: (Label for legend, Resolution Value (m), File Tag used in filename)
RESOLUTIONS_TO_TEST = [
    ("10cm (High Res)", 0.10, "10cm"),
    ("25cm (Medium Res)", 0.25, "25cm"),
    ("1m (Coarse Res)",   1.00, "100cm")
]

# Color Palette specifically for resolutions, distinct from erosion/deposition
RES_COLORS = {
    "10cm (High Res)": "#2ca02c",  # Green
    "25cm (Medium Res)": "#ff7f0e",  # Orange
    "1m (Coarse Res)":   "#1f77b4"   # Blue
}

sns.set_theme(style="whitegrid", font_scale=1.1)
DASH_STYLE = (2, 2) # For reference lines

# ==============================================================================
# 2. HELPER FUNCTIONS (Reused from pipeline logic)
# ==============================================================================

def get_base_dir():
    """Determine system root path based on OS."""
    if platform.system() == 'Darwin':
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

def parse_dates_from_folder(folder_name):
    """Extract datetime objects from folder naming convention."""
    match = re.search(r'(\d{8})_to_(\d{8})', folder_name)
    if match:
        d1 = datetime.strptime(match.group(1), '%Y%m%d')
        d2 = datetime.strptime(match.group(2), '%Y%m%d')
        return d1, d2
    return None, None

def calculate_main_volume(grid_path, res_val):
    """Calculates total volume sum from a grid file."""
    cell_area = res_val * res_val
    if not os.path.exists(grid_path): return None
    try:
        df_grid = pd.read_csv(grid_path, index_col=0)
        # Ensure numeric and fill NaNs (though filled grids shouldn't have them)
        df_grid = df_grid.apply(pd.to_numeric, errors='coerce').fillna(0)
        distances = df_grid.values
        # Sum distances * cell area
        vol_main = distances.sum() * cell_area
        return vol_main
    except Exception:
        return None

# ==============================================================================
# 3. DATA LOADING ENGINE
# ==============================================================================

def load_dashboard_data(location):
    base_dir = get_base_dir()
    erosion_dir = os.path.join(base_dir, 'results', location, 'erosion')
    
    if not os.path.exists(erosion_dir):
        print(f"[ERROR] Directory not found: {erosion_dir}")
        return None

    # Get sorted survey interval folders
    intervals = sorted([d for d in os.listdir(erosion_dir) 
                       if os.path.isdir(os.path.join(erosion_dir, d))])
    
    print(f"--- Loading data from {len(intervals)} intervals for {location} ---")
    
    # Structure to hold all data: List of dictionaries for DataFrame conversion
    all_data_records = []

    for interval in intervals:
        d1, d2 = parse_dates_from_folder(interval)
        if not d1: continue
        
        # Date for plotting (using end date of interval)
        plot_date = d2
        folder_path = os.path.join(erosion_dir, interval)
        
        # For this specific interval, try to load ALL resolution files
        interval_record = {'Date': plot_date, 'IntervalStr': interval}
        
        data_found_for_interval = False
        for res_label, res_val, file_tag in RESOLUTIONS_TO_TEST:
            # Construct path based on pipeline naming convention
            grid_file = os.path.join(folder_path, f"{interval}_ero_grid_{file_tag}_filled.csv")
            
            vol = calculate_main_volume(grid_file, res_val)
            
            if vol is not None:
                # Store volume for this resolution
                interval_record[res_label] = vol
                data_found_for_interval = True
            else:
                # Mark missing data if a specific resolution file didn't exist for this date
                interval_record[res_label] = np.nan

        if data_found_for_interval:
            all_data_records.append(interval_record)

    # Create master DataFrame and sort by date
    df_master = pd.DataFrame(all_data_records).sort_values('Date').reset_index(drop=True)
    
    # Calculate Cumulative Volumes
    for res_label, _, _ in RESOLUTIONS_TO_TEST:
        if res_label in df_master.columns:
            # fillna(0) ensures cumulative sum doesn't break on missing intervals, 
            # though ideally all resolutions have the same intervals.
            df_master[f'{res_label}_Cumulative'] = df_master[res_label].fillna(0).cumsum()

    return df_master

# ==============================================================================
# 4. DASHBOARD PLOTTING
# ==============================================================================

def plot_sensitivity_dashboard(df, location):
    if df is None or df.empty: return

    # Setup the 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{location} Cliff Erosion: Grid Resolution Sensitivity Analysis", 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Flatten axes array for easy iteration
    ax_cumulative = axes[0, 0]
    ax_top_events = axes[0, 1]
    ax_scatter   = axes[1, 0]
    ax_ecdf      = axes[1, 1]

    # ==========================================================================
    # Panel A: Cumulative Erosion Time Series
    # ==========================================================================
    print("Plotting Panel A: Cumulative Time Series...")
    for res_label, _, _ in RESOLUTIONS_TO_TEST:
        cum_col = f'{res_label}_Cumulative'
        if cum_col in df.columns:
            sns.lineplot(data=df, x='Date', y=cum_col, ax=ax_cumulative, 
                         label=res_label, color=RES_COLORS[res_label], linewidth=3)
    
    ax_cumulative.set_title("A. Cumulative Erosion Volume Over Time", fontweight='bold')
    ax_cumulative.set_ylabel("Cumulative Volume ($m^3$)", fontweight='bold')
    ax_cumulative.set_xlabel("")
    ax_cumulative.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate(ax=ax_cumulative)
    ax_cumulative.legend(title="Grid Resolution")
    ax_cumulative.grid(True, which='major', linestyle='--')

    # ==========================================================================
    # Panel B: Top N Largest Events Comparison
    # ==========================================================================
    print("Plotting Panel B: Top Events Barplot...")
    N_TOP = 15
    top_events_data = []
    
    for res_label, _, _ in RESOLUTIONS_TO_TEST:
        if res_label in df.columns:
            # Get all individual survey volumes for this resolution, drop NaNs
            vols = df[res_label].dropna().sort_values(ascending=False).head(N_TOP).reset_index(drop=True)
            for rank, vol in enumerate(vols):
                top_events_data.append({
                    'Resolution': res_label,
                    'Rank': rank + 1,
                    'Volume': vol
                })
                
    df_top = pd.DataFrame(top_events_data)
    
    sns.barplot(data=df_top, x='Rank', y='Volume', hue='Resolution', 
                palette=RES_COLORS, ax=ax_top_events)
    
    ax_top_events.set_title(f"B. Magnitudes of Top {N_TOP} Largest Events", fontweight='bold')
    ax_top_events.set_ylabel("Event Volume ($m^3$)", fontweight='bold')
    ax_top_events.set_xlabel("Event Rank (Largest to Smallest)")
    ax_top_events.legend(title="Grid Resolution")

    # ==========================================================================
    # Panel C: Per-Survey Volume Scatter (vs. 10cm Baseline)
    # ==========================================================================
    print("Plotting Panel C: Comparison Scatter Plot...")
    baseline_label = RESOLUTIONS_TO_TEST[0][0] # Assuming 10cm is first
    
    # Find max value for plotting 1:1 line
    max_vol = df[[r[0] for r in RESOLUTIONS_TO_TEST]].max().max()
    
    # Plot 1:1 reference line
    ax_scatter.plot([0, max_vol], [0, max_vol], 'k--', linewidth=1.5, label="1:1 Line")

    for res_label, _, _ in RESOLUTIONS_TO_TEST:
        if res_label == baseline_label: continue # Don't plot 10cm vs 10cm
        
        sns.scatterplot(data=df, x=baseline_label, y=res_label, ax=ax_scatter,
                        color=RES_COLORS[res_label], label=res_label, s=80, alpha=0.7)
        
    ax_scatter.set_title(f"C. Individual Survey Volume vs. {baseline_label} Baseline", fontweight='bold')
    ax_scatter.set_xlabel(f"{baseline_label} Volume ($m^3$)", fontweight='bold')
    ax_scatter.set_ylabel("Comparison Resolution Volume ($m^3$)", fontweight='bold')
    ax_scatter.legend()
    ax_scatter.set_aspect('equal', adjustable='box') # Make axes square for better comparison

    # ==========================================================================
    # Panel D: ECDF of All Event Magnitudes
    # ==========================================================================
    print("Plotting Panel D: ECDF...")
    for res_label, _, _ in RESOLUTIONS_TO_TEST:
        if res_label in df.columns:
            # Get all non-zero, non-NaN volumes
            vols = df[res_label].dropna()
            vols = vols[vols > 0]
            
            sns.ecdfplot(data=vols, ax=ax_ecdf, label=res_label, 
                         color=RES_COLORS[res_label], linewidth=3)

    ax_ecdf.set_title("D. Empirical Cumulative Distribution of Event Volumes", fontweight='bold')
    ax_ecdf.set_xlabel("Log10 Event Volume ($m^3$)", fontweight='bold')
    ax_ecdf.set_ylabel("Cumulative Probability", fontweight='bold')
    ax_ecdf.set_xscale('log') # Crucial for seeing distribution across magnitudes
    ax_ecdf.legend(title="Grid Resolution", loc="lower right")
    ax_ecdf.grid(True, which='minor', linestyle=':', alpha=0.5)

    # ==========================================================================
    # Final Layout Adjustments and Saving
    # ==========================================================================
    plt.tight_layout()
    # Adjust for main suptitle
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.2)
    
    # Save paths
    base_dir = get_base_dir()
    out_dir = os.path.join(base_dir, "figures", "sensitivity")
    os.makedirs(out_dir, exist_ok=True)
    out_path_png = os.path.join(out_dir, f"{location}_Grid_Sensitivity_Dashboard.png")
    out_path_svg = os.path.join(out_dir, f"{location}_Grid_Sensitivity_Dashboard.svg")
    
    plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
    # plt.savefig(out_path_svg, bbox_inches='tight') # Uncomment for vector graphics
    print(f"\n[SUCCESS] Dashboard saved to: {out_path_png}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default="DelMar", help="Location to analyze (e.g., DelMar, TorreyPines)")
    args = parser.parse_args()
    
    # 1. Load and structure data
    df_master = load_dashboard_data(args.location)
    
    # 2. Generate Dashboard
    plot_sensitivity_dashboard(df_master, args.location)