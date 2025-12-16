#!/usr/bin/env python3
"""
plot_spatial_sensitivity.py

Purpose:
    Visualizes the SPATIAL differences between grid resolutions.
    Fixes coordinate mapping bugs to ensure 'Zoom' views align correctly.

Usage:
    python3 code/figure_making/plot_spatial_sensitivity.py --location DelMar
"""

import os
import re
import platform
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# (Label, Resolution Value, File Tag)
RESOLUTIONS = [
    ("1m",   1.00, "100cm"),
    ("25cm", 0.25, "25cm"),
    ("10cm", 0.10, "10cm")
]

# Visuals
CMAP_NAME = 'magma_r'
VMAX_CUMULATIVE = 5.0  # Saturation point (meters) for heatmap

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def get_base_dir():
    if platform.system() == 'Darwin':
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

def get_custom_cmap(name, vmax):
    """Creates a colormap that forces 0 values to be White."""
    base = cm.get_cmap(name, 256)
    colors = base(np.linspace(0, 1, 256))
    colors[0, :] = [1, 1, 1, 1] # White
    return LinearSegmentedColormap.from_list(f"White_{name}", colors), Normalize(vmin=0, vmax=vmax)

def clean_and_snap_grid(df, resolution_val):
    """
    Converts DataFrame headers (Elev) and Index (Alongshore) to float/int.
    CRITICAL: Sorts the data to ensure imshow works correctly.
    """
    # Clean Columns (Elevation)
    cleaned_cols = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)
    try:
        col_floats = cleaned_cols.astype(float)
        # Snap to grid index (0.1 -> 1, 0.2 -> 2)
        scale = 1.0 / resolution_val
        new_cols = (col_floats * scale).round().astype(int)
        df.columns = new_cols
    except:
        return None

    # Clean Index (Alongshore) - usually already int from CSV
    try:
        df.index = df.index.astype(int)
    except:
        return None

    # SORTING IS CRITICAL FOR IMSHOW
    df = df.sort_index(axis=0) # Sort Polygons (Rows)
    df = df.sort_index(axis=1) # Sort Elevations (Cols)
    
    return df

def load_cumulative_grid(location, res_val, file_tag):
    base_dir = get_base_dir()
    erosion_dir = os.path.join(base_dir, 'results', location, 'erosion')
    
    if not os.path.exists(erosion_dir): return None

    cumulative_df = None
    intervals = sorted([d for d in os.listdir(erosion_dir) if os.path.isdir(os.path.join(erosion_dir, d))])
    
    print(f"  [{file_tag}] Scanning {len(intervals)} intervals...")
    
    for interval in intervals:
        grid_path = os.path.join(erosion_dir, interval, f"{interval}_ero_grid_{file_tag}_filled.csv")
        
        if os.path.exists(grid_path):
            try:
                # Load sparse, fill 0
                df = pd.read_csv(grid_path, index_col=0).fillna(0)
                df_clean = clean_and_snap_grid(df, res_val)
                
                if df_clean is not None:
                    if cumulative_df is None:
                        cumulative_df = df_clean
                    else:
                        # Add using fill_value=0 to handle expanding grid sizes
                        cumulative_df = cumulative_df.add(df_clean, fill_value=0)
            except:
                pass
                
    return cumulative_df

# ==============================================================================
# 3. PLOTTING LOGIC
# ==============================================================================

def plot_spatial_comparison(grids, location, out_dir):
    """
    Generates BOTH the Full View and the Zoom View in one run.
    """
    cmap, norm = get_custom_cmap(CMAP_NAME, VMAX_CUMULATIVE)
    
    # 1. SETUP FIGURE 1: FULL COASTLINE
    fig1, axes1 = plt.subplots(3, 1, figsize=(18, 10), sharex=True, constrained_layout=True)
    
    # Calculate global X bounds from the 10cm grid (ground truth)
    ref_grid = grids.get("10cm")['data']
    # Physical extent: Max Polygon ID * Resolution
    global_x_max_m = ref_grid.index.max() * 0.10
    
    print("  Generating Full Coastline Plot...")
    
    for ax, (label, res, _) in zip(axes1, RESOLUTIONS):
        if label not in grids: continue
        
        df = grids[label]['data']
        # Transpose: X=Alongshore (cols), Y=Elevation (rows) for imshow
        # Original df: Index=Alongshore, Cols=Elevation
        # We need array to be (Elevation, Alongshore)
        matrix = df.T.values 
        
        # Extent [Left, Right, Bottom, Top]
        # X starts at 0, goes to (MaxIndex * Resolution)
        x_max_local = df.index.max() * res
        y_max_local = df.columns.max() * res
        
        extent = [0, x_max_local, 0, y_max_local]
        
        im = ax.imshow(matrix, origin='lower', extent=extent, 
                       cmap=cmap, norm=norm, aspect='auto', interpolation='none')
        
        # Calculate Volume
        vol = matrix.sum() * (res**2)
        ax.text(0.01, 0.9, f"Vol: {vol:,.0f} m³", transform=ax.transAxes, 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.9))
        
        ax.set_title(f"{label} Resolution", fontweight='bold')
        ax.set_ylabel("Elevation (m)")
        
        # Invert X axis to match standard coastal view (North Left -> South Right or vice versa)
        # Assuming Polygon 0 is North. Let's flip it so North is Left? 
        # Standard imshow puts 0 on Left. 
        # Let's align them all to the global max.
        ax.set_xlim(global_x_max_m, 0) # Flip: Max on Left, 0 on Right
        ax.set_ylim(0, 30)

    fig1.colorbar(im, ax=axes1, label="Cumulative Erosion (m)", location='right', fraction=0.02)
    fig1.suptitle(f"{location}: Grid Resolution Sensitivity (Full Coastline)", fontsize=16, fontweight='bold')
    
    path1 = os.path.join(out_dir, f"{location}_Spatial_Full.png")
    plt.savefig(path1, dpi=200)
    print(f"  Saved: {path1}")
    plt.close(fig1)

    # 2. SETUP FIGURE 2: ZOOM ON LARGEST EVENT
    print("  Generating Zoom Plot...")
    
    # Find hotspot in 10cm grid
    # Sum alongshore (sum across columns of df which are elevation)
    # df index is Alongshore ID.
    profile = ref_grid.sum(axis=1)
    
    # Find index of peak erosion
    peak_idx = profile.idxmax()
    peak_loc_m = peak_idx * 0.10
    
    # Define Window (+/- 30m)
    zoom_radius = 30 
    x_zoom_min = peak_loc_m - zoom_radius
    x_zoom_max = peak_loc_m + zoom_radius
    
    print(f"    Zooming on event at {peak_loc_m:.1f}m (Window: {x_zoom_min:.1f}m - {x_zoom_max:.1f}m)")

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharey=True, constrained_layout=True)
    
    for ax, (label, res, _) in zip(axes2, RESOLUTIONS):
        if label not in grids: continue
        
        df = grids[label]['data']
        matrix = df.T.values
        
        x_max_local = df.index.max() * res
        y_max_local = df.columns.max() * res
        extent = [0, x_max_local, 0, y_max_local]
        
        # Use 'nearest' interpolation to show the pixels clearly
        im = ax.imshow(matrix, origin='lower', extent=extent, 
                       cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')
        
        ax.set_title(f"{label} Grid", fontweight='bold')
        ax.set_xlabel("Alongshore (m)")
        
        # Set Zoom Limits
        # IMPORTANT: Match the flip from Figure 1
        ax.set_xlim(x_zoom_max, x_zoom_min) 
        ax.set_ylim(0, 25)
        
        # Add Gridlines to show cell size
        ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    axes2[0].set_ylabel("Elevation (m)", fontweight='bold')
    
    cbar2 = fig2.colorbar(im, ax=axes2, label="Cumulative Erosion (m)", location='bottom', fraction=0.05, pad=0.1)
    fig2.suptitle(f"{location}: Detail View of Largest Erosion Event", fontsize=16, fontweight='bold')
    
    path2 = os.path.join(out_dir, f"{location}_Spatial_Zoom.png")
    plt.savefig(path2, dpi=200)
    print(f"  Saved: {path2}")
    plt.close(fig2)

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default="DelMar")
    args = parser.parse_args()
    
    base_dir = get_base_dir()
    out_dir = os.path.join(base_dir, "figures", "sensitivity")
    os.makedirs(out_dir, exist_ok=True)

    # Load Data
    grids = {}
    for label, val, tag in RESOLUTIONS:
        df = load_cumulative_grid(args.location, val, tag)
        if df is not None:
            grids[label] = {'data': df, 'res': val}
    
    if not grids:
        print("No data found.")
        return

    plot_spatial_comparison(grids, args.location, out_dir)

if __name__ == "__main__":
    main()