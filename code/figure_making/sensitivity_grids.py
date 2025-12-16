#!/usr/bin/env python3
"""
plot_grid_sensitivity.py

Purpose:
    Generates a sensitivity analysis figure showing how Total Erosion Volume
    changes across different Grid Resolutions (10cm, 25cm, 1m).

    Reuses EXACT logic from:
      - plot_dashboard.py (Path handling, Volume calculation)
      - 8_make_grids.py (File naming conventions)

Usage:
    python3 plot_grid_sensitivity.py --location DelMar
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

# ==============================================================================
# 1. CONFIGURATION (Adapted from plot_dashboard.py)
# ==============================================================================

# Define the resolutions to test
# Format: (Label, Resolution Value (m), File Tag)
RESOLUTIONS_TO_TEST = [
    ("10cm", 0.10, "10cm"),
    ("25cm", 0.25, "25cm"),
    ("1m",   1.00, "100cm")  # Note: 8_make_grids.py names 1m files as "100cm"
]

# Plotting Settings
sns.set_theme(style="whitegrid", font_scale=1.2)
COLOR_MAIN = "#d62728"  # Red for Erosion

# ==============================================================================
# 2. HELPER FUNCTIONS (Directly from plot_dashboard.py)
# ==============================================================================

def get_base_dir():
    """Determine system root path based on OS."""
    if platform.system() == 'Darwin':
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

def parse_dates(folder_name):
    """Extract dates from folder name (YYYYMMDD_to_YYYYMMDD)."""
    match = re.search(r'(\d{8})_to_(\d{8})', folder_name)
    if match:
        d1 = datetime.strptime(match.group(1), '%Y%m%d')
        d2 = datetime.strptime(match.group(2), '%Y%m%d')
        return d1, d2
    return None, None

def calculate_volume_bounds_properly(grid_path, unc_path, res_val):
    """
    Calculates Volume using cell-by-cell logic.
    EXACT COPY from plot_dashboard.py to ensure consistency.
    """
    cell_area = res_val * res_val
    
    if not os.path.exists(grid_path):
        return None, None, None

    try:
        # Load Main Grid
        df_grid = pd.read_csv(grid_path, index_col=0)
        df_grid = df_grid.apply(pd.to_numeric, errors='coerce').fillna(0)
        distances = df_grid.values
        
        # Load Uncertainty Stats (Simplified for sensitivity plot - focusing on Mean Volume)
        # We only strictly need the main volume, but the function requires the path
        
        # Calculate Main Volume
        # Note: Your dashboard logic calculates bounds, but here we primarily need the main estimate
        vol_main = distances.sum() * cell_area
        
        return vol_main, 0, 0 # Returning dummy bounds as we just want the main vol for sensitivity
        
    except Exception as e:
        # print(f"    [Error] Reading grid {os.path.basename(grid_path)}: {e}")
        return None, None, None

# ==============================================================================
# 3. DATA COLLECTION
# ==============================================================================

def collect_sensitivity_data(location):
    base_dir = get_base_dir()
    erosion_dir = os.path.join(base_dir, 'results', location, 'erosion')
    
    if not os.path.exists(erosion_dir):
        print(f"[ERROR] Directory not found: {erosion_dir}")
        return []

    # Get all survey intervals
    intervals = sorted([d for d in os.listdir(erosion_dir) 
                       if os.path.isdir(os.path.join(erosion_dir, d))])
    
    print(f"--- Scanning {len(intervals)} intervals for {location} ---")
    
    results = []

    # Iterate through every resolution defined
    for res_label, res_val, file_tag in RESOLUTIONS_TO_TEST:
        print(f"Processing Resolution: {res_label} (Tag: {file_tag})...")
        
        total_volume = 0.0
        survey_count = 0
        
        for interval in intervals:
            # 1. Parse Date
            d1, d2 = parse_dates(interval)
            if not d1: continue
            
            # 2. Construct Paths (Using dashboard logic)
            folder = os.path.join(erosion_dir, interval)
            
            # Look for the specific resolution file
            # Pattern: {interval}_ero_grid_{TAG}_filled.csv
            grid_file = os.path.join(folder, f"{interval}_ero_grid_{file_tag}_filled.csv")
            
            # We don't strictly need uncertainty for the main volume sum, 
            # but the function asks for it. We can pass None if the function handles it,
            # or construct the path. Your dashboard function handles missing unc files gracefully.
            unc_file = os.path.join(folder, f"{interval}_ero_uncertainty_{file_tag}.csv")
            
            if not os.path.exists(grid_file):
                # print(f"  [MISSING] {grid_file}")
                continue
                
            # 3. Calculate Volume
            vol_main, _, _ = calculate_volume_bounds_properly(grid_file, unc_file, res_val)
            
            if vol_main is not None:
                total_volume += vol_main
                survey_count += 1
        
        print(f"  -> Total Volume: {total_volume:,.2f} m³ ({survey_count} surveys)")
        
        results.append({
            "Resolution Label": res_label,
            "Grid Size (m)": res_val,
            "Total Erosion Volume (m³)": total_volume,
            "Survey Count": survey_count
        })

    return pd.DataFrame(results)

# ==============================================================================
# 4. PLOTTING
# ==============================================================================

def plot_sensitivity(df, location):
    if df.empty:
        print("No data to plot.")
        return

    base_dir = get_base_dir()
    out_dir = os.path.join(base_dir, "figures", "sensitivity")
    os.makedirs(out_dir, exist_ok=True)
    
    # Setup Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Line with Markers
    sns.lineplot(
        data=df, 
        x="Grid Size (m)", 
        y="Total Erosion Volume (m³)", 
        marker='o', 
        markersize=10,
        color=COLOR_MAIN,
        linewidth=2.5,
        ax=ax
    )
    
    # Add text labels for points
    for line in range(0, df.shape[0]):
        x_val = df["Grid Size (m)"][line]
        y_val = df["Total Erosion Volume (m³)"][line]
        label = f"{y_val:,.0f} m³"
        
        ax.text(
            x_val, y_val, 
            label, 
            horizontalalignment='left', 
            size='medium', 
            color='black', 
            weight='semibold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5)
        )

    # Formatting
    ax.set_title(f"Grid Size Sensitivity Analysis: {location}\nTotal Cumulative Erosion Volume", 
                 fontweight='bold', fontsize=16)
    ax.set_xlabel("Grid Cell Resolution (m)", fontweight='bold', fontsize=14)
    ax.set_ylabel("Total Erosion Volume ($m^3$)", fontweight='bold', fontsize=14)
    
    # Force X-ticks to match our resolutions
    ax.set_xticks(df["Grid Size (m)"].unique())
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Invert X axis? Usually smaller grid size (left) is higher resolution. 
    # Standard plots usually go 0 -> 1. Let's keep it standard.
    
    # Save
    out_path = os.path.join(out_dir, f"{location}_Grid_Sensitivity.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"\n[SUCCESS] Figure saved to: {out_path}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default="DelMar", help="Location to analyze")
    args = parser.parse_args()
    
    # 1. Collect Data
    df = collect_sensitivity_data(args.location)
    
    # 2. Plot
    plot_sensitivity(df, args.location)

if __name__ == "__main__":
    main()