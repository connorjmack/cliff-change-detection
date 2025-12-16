#!/usr/bin/env python3
"""
plot_grid_sensitivity.py

Visualizes the "Cliff Activity Index" (Cumulative Erosion) across 
three different grid resolutions (1m, 25cm, 10cm) to assess sensitivity.

Features:
- Uses 'magma_r' colormap (White -> Orange -> Purple -> Black)
- Supports full cliff view or specific zoomed regions.
- Aligns all plots to the same color scale (0 to 6m) for fair comparison.

Usage:
    python3 plot_grid_sensitivity.py --location DelMar
    python3 plot_grid_sensitivity.py --location DelMar --zoom 1450 1600
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from matplotlib.colors import LinearSegmentedColormap, Normalize
from datetime import datetime

# --- Configuration ---
RESOLUTIONS = ['1m', '25cm', '10cm']
VMAX_EROSION = 6.0  # Saturation point in meters (Matches your screenshot colorbar)

# --- Styling ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 12

def get_resolution_value(res_str):
    if 'cm' in res_str:
        return float(res_str.replace('cm', '')) / 100.0
    elif 'm' in res_str:
        return float(res_str.replace('m', ''))
    return 0.1

def normalize_resolution_string(res):
    """Matches file naming convention (1m -> 100cm)"""
    if res == '1m': return '100cm'
    return res

def get_custom_magma_cmap():
    """
    Creates the 'White -> Orange -> Purple -> Black' colormap 
    seen in your screenshot.
    """
    # Get standard magma_r (reversed)
    # magma_r goes: White/Yellow -> Red -> Purple -> Black
    magma = cm.get_cmap('magma_r', 256)
    newcolors = magma(np.linspace(0, 1, 256))
    
    # Force the very first color (0 value) to be strictly white/off-white
    # to represent "No Change" clearly.
    newcolors[0, :] = np.array([1, 1, 0.95, 1]) 
    
    return LinearSegmentedColormap.from_list("WhiteMagma", newcolors)

def load_and_sum_grids(base_dir, location, resolution):
    """
    Loads all erosion grids for a resolution and sums them up.
    Returns the final cumulative DataFrame.
    """
    # 1. Setup Paths
    file_res = normalize_resolution_string(resolution)
    res_val = get_resolution_value(resolution)
    data_dir = os.path.join(base_dir, 'results', location, 'erosion')
    
    if not os.path.isdir(data_dir):
        print(f"  [Error] Directory not found: {data_dir}")
        return None

    # 2. Find Files
    cumulative_df = None
    count = 0
    
    # Sorted by date folder
    for date_folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, date_folder)
        if not os.path.isdir(folder_path): continue
        
        # Look for filled grid first, then cleaned
        target_file = None
        patterns = [f"_ero_grid_{file_res}_filled.csv", f"grid_{file_res}_filled.csv"]
        
        for f in os.listdir(folder_path):
            if any(p in f for p in patterns) and f.endswith('.csv'):
                target_file = os.path.join(folder_path, f)
                break
        
        if not target_file: continue

        # 3. Load & Process
        try:
            df = pd.read_csv(target_file, index_col=0)
            
            # Clean Headers (Remove 'M3C2_', 'm') -> Convert to float
            cleaned_cols = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)
            col_floats = cleaned_cols.astype(float)
            
            # Snap to integer indices based on resolution
            # e.g. 0.10m / 0.10 = Index 1
            new_cols = (col_floats / res_val).round().astype(int)
            df.columns = new_cols
            df.index = df.index.astype(int) # Polygon IDs
            
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            
            if cumulative_df is None:
                cumulative_df = df
            else:
                cumulative_df = cumulative_df.add(df, fill_value=0)
            count += 1
            
        except Exception as e:
            print(f"    Skipping {date_folder}: {e}")
            continue

    print(f"  {resolution}: Summed {count} surveys.")
    return cumulative_df

def plot_panel(ax, df, resolution, zoom_range=None):
    """
    Plots a single resolution panel using the custom Magma colormap.
    """
    res_val = get_resolution_value(resolution)
    
    # Transpose: X=Alongshore, Y=Elevation
    plot_df = df.T
    plot_df.sort_index(axis=0, inplace=True) # Sort elevations 0->Top
    plot_df.sort_index(axis=1, inplace=True) # Sort Alongshore Left->Right
    
    # Setup Extent [Xmin, Xmax, Ymin, Ymax]
    # We use the raw indices * resolution to get Meters
    x_min = plot_df.columns.min()
    x_max = plot_df.columns.max()
    y_max = plot_df.index.max() * res_val
    
    extent = [x_min, x_max, 0, y_max]
    
    # Setup Colormap
    cmap = get_custom_magma_cmap()
    norm = Normalize(vmin=0, vmax=VMAX_EROSION)
    
    # Plot Image
    im = ax.imshow(plot_df.values, origin='lower', extent=extent, 
                   cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    
    # Styling
    ax.set_title(f"{resolution} Resolution", fontsize=14, fontweight='bold', pad=10)
    ax.set_ylabel("Elevation (m)", fontsize=12)
    ax.grid(False)
    
    # If Zoom is requested, apply limits
    if zoom_range:
        ax.set_xlim(zoom_range[1], zoom_range[0]) # Inverted for cliff view (North->South)
    else:
        ax.invert_xaxis() # Default cliff view
        
    return im

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', required=True, help="Survey Location (e.g. DelMar)")
    parser.add_argument('--zoom', nargs=2, type=int, help="Zoom limits (e.g. 1450 1600)")
    args = parser.parse_args()

    # Detect OS path
    import platform
    if platform.system() == 'Darwin':
         base_dir = '/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs'
    else:
         base_dir = '/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs'

    # Setup Output
    out_dir = os.path.join(base_dir, 'figures', 'grid_sensitivity')
    os.makedirs(out_dir, exist_ok=True)
    
    # Setup Figure (3 rows, 1 col)
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    print(f"--- Generating Grid Sensitivity Plot for {args.location} ---")
    
    mappable = None
    
    for i, res in enumerate(RESOLUTIONS):
        df = load_and_sum_grids(base_dir, args.location, res)
        if df is not None:
            mappable = plot_panel(axes[i], df, res, args.zoom)
        else:
            axes[i].text(0.5, 0.5, f"No Data for {res}", ha='center', va='center')

    # Common X Label
    axes[-1].set_xlabel("Alongshore Location (m)", fontsize=12, fontweight='bold')
    
    # Add Shared Colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02]) # [left, bottom, width, height]
    if mappable:
        cb = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
        cb.set_label("Cumulative Erosion Depth (m)", fontsize=12, fontweight='bold')
        cb.ax.xaxis.set_ticks_position('bottom')

    # Title
    zoom_str = f"(Zoom: {args.zoom[0]}-{args.zoom[1]}m)" if args.zoom else "(Full Extent)"
    fig.suptitle(f"{args.location}: Grid Resolution Sensitivity Analysis {zoom_str}", 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Save
    suffix = "zoomed" if args.zoom else "full"
    out_path = os.path.join(out_dir, f"{args.location}_Sensitivity_{suffix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out_path}")

if __name__ == '__main__':
    main()