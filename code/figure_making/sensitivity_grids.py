#!/usr/bin/env python3
"""
plot_spatial_sensitivity.py

Purpose:
    Visualizes SPATIAL differences between grid resolutions.
    Correctly aligns grids with different polygon counts by converting to physical meters.

    Features:
    1. Full Coastline Stack (1m, 25cm, 10cm aligned).
    2. Zoom View of the largest erosion event.
    3. Robust file finding (matches patterns instead of exact names).

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

# (Label, Resolution Value (m), File Tag)
RESOLUTIONS = [
    ("1m",   1.00, "100cm"),
    ("25cm", 0.25, "25cm"),
    ("10cm", 0.10, "10cm")
]

# Visuals
CMAP_NAME = 'magma_r'
VMAX_CUMULATIVE = 5.0  # Saturation point (meters)

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
    base = cm.get_cmap(name, 256)
    colors = base(np.linspace(0, 1, 256))
    colors[0, :] = [1, 1, 1, 1] # White for 0
    return LinearSegmentedColormap.from_list(f"White_{name}", colors), Normalize(vmin=0, vmax=vmax)

def clean_and_snap_grid(df, resolution_val):
    """
    Cleans headers and snaps to integer indices.
    Returns transposed DataFrame (Index=Elevation, Cols=Alongshore) if needed,
    but here we keep (Index=Alongshore, Cols=Elevation) and transpose later.
    """
    # Clean Columns (Elevation)
    cleaned_cols = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)
    try:
        col_floats = cleaned_cols.astype(float)
        # Snap to grid index
        scale = 1.0 / resolution_val
        new_cols = (col_floats * scale).round().astype(int)
        df.columns = new_cols
    except:
        return None

    # Clean Index (Polygon ID)
    try:
        df.index = df.index.astype(int)
    except:
        return None

    return df

def find_file_fuzzy(folder, pattern):
    """Finds a file in folder matching the pattern (like cum_erosion.py)."""
    if not os.path.exists(folder): return None
    files = os.listdir(folder)
    matches = [f for f in files if pattern in f and f.endswith('.csv')]
    if matches:
        return os.path.join(folder, matches[0]) # Return first match
    return None

def load_cumulative_grid(location, res_val, file_tag):
    base_dir = get_base_dir()
    erosion_dir = os.path.join(base_dir, 'results', location, 'erosion')
    
    if not os.path.exists(erosion_dir): return None

    cumulative_df = None
    intervals = sorted([d for d in os.listdir(erosion_dir) if os.path.isdir(os.path.join(erosion_dir, d))])
    
    print(f"  [{file_tag}] Scanning {len(intervals)} intervals...")
    
    pattern = f"grid_{file_tag}_filled.csv" # Look for filled grids
    
    for interval in intervals:
        folder_path = os.path.join(erosion_dir, interval)
        grid_path = find_file_fuzzy(folder_path, pattern)
        
        if grid_path:
            try:
                # Load sparse, fill 0
                df = pd.read_csv(grid_path, index_col=0).fillna(0)
                df_clean = clean_and_snap_grid(df, res_val)
                
                if df_clean is not None:
                    if cumulative_df is None:
                        cumulative_df = df_clean.fillna(0)
                    else:
                        cumulative_df = cumulative_df.add(df_clean.fillna(0), fill_value=0)
            except:
                pass
                
    if cumulative_df is not None:
        # Sort indices to ensure spatial continuity
        cumulative_df.sort_index(axis=0, inplace=True)
        cumulative_df.sort_index(axis=1, inplace=True)
        
    return cumulative_df

# ==============================================================================
# 3. PLOTTING LOGIC
# ==============================================================================

def plot_spatial_comparison(grids, location, out_dir):
    cmap, norm = get_custom_cmap(CMAP_NAME, VMAX_CUMULATIVE)
    
    # --- CALCULATE GLOBAL EXTENT (METERS) ---
    # We use the 10cm grid to define the true physical length.
    ref_grid = grids.get("10cm") or list(grids.values())[0]
    ref_df = ref_grid['data']
    ref_res = ref_grid['res']
    
    # Calculate Max Physical Meter (PolygonID * Resolution)
    # If 10cm: MaxIndex 23000 * 0.1 = 2300m
    # If 1m:   MaxIndex 2300 * 1.0 = 2300m
    global_x_max = ref_df.index.max() * ref_res
    
    # 1. FULL COASTLINE STACK
    fig1, axes1 = plt.subplots(3, 1, figsize=(18, 10), sharex=True, constrained_layout=True)
    
    print("  Generating Full Coastline Plot...")

    for ax, (label, res, _) in zip(axes1, RESOLUTIONS):
        if label not in grids: 
            ax.text(0.5, 0.5, "Data Missing", ha='center'); continue
        
        df = grids[label]['data']
        
        # TRANSPOSE FOR PLOTTING
        # Original: Index=Alongshore, Cols=Elevation
        # Plotting: X=Alongshore, Y=Elevation
        # imshow expects (Rows, Cols) -> (Elevation, Alongshore) -> df.T
        plot_df = df.T 
        
        # Calculate Physical Extents
        x_min_m = plot_df.columns.min() * res
        x_max_m = plot_df.columns.max() * res
        y_min_m = plot_df.index.min() * res
        y_max_m = plot_df.index.max() * res
        
        # Standard Imshow Extent: [left, right, bottom, top]
        extent = [x_min_m, x_max_m, y_min_m, y_max_m]
        
        im = ax.imshow(plot_df.values, origin='lower', extent=extent, 
                       cmap=cmap, norm=norm, aspect='auto', interpolation='none')
        
        # Stats Label
        vol = plot_df.values.sum() * (res**2)
        ax.text(0.01, 0.85, f"Vol: {vol:,.0f} m³", transform=ax.transAxes, 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.9))
        
        ax.set_title(f"{label} Resolution", fontweight='bold')
        ax.set_ylabel("Elevation (m)")
        
        # ALIGN X-AXIS: Flip to standard coastal view (Max -> 0)
        ax.set_xlim(global_x_max, 0) 
        ax.set_ylim(0, 30) # Standard cliff height

    fig1.colorbar(im, ax=axes1, label="Cumulative Erosion (m)", location='right', fraction=0.02)
    fig1.suptitle(f"{location}: Grid Resolution Sensitivity (Full Coastline)", fontsize=16, fontweight='bold')
    
    path1 = os.path.join(out_dir, f"{location}_Spatial_Full.png")
    plt.savefig(path1, dpi=200)
    print(f"  Saved: {path1}")
    plt.close(fig1)

    # 2. ZOOM VIEW
    print("  Generating Zoom Plot...")
    
    # Auto-detect hotspot from 10cm grid
    if "10cm" in grids:
        df_10 = grids["10cm"]['data']
        # Sum alongshore (df index) to find peak
        profile = df_10.sum(axis=1) # Sums all elevation bins for each polygon
        peak_poly_id = profile.idxmax()
        peak_loc_m = peak_poly_id * 0.10
        
        # Define 60m window
        zoom_r = 30 
        z_min_m = peak_loc_m - zoom_r
        z_max_m = peak_loc_m + zoom_r
        
        print(f"    Zooming at {peak_loc_m:.1f}m ({z_min_m:.0f}m - {z_max_m:.0f}m)")

        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharey=True, constrained_layout=True)
        
        for ax, (label, res, _) in zip(axes2, RESOLUTIONS):
            if label not in grids: continue
            
            df = grids[label]['data']
            plot_df = df.T
            
            x_min_m = plot_df.columns.min() * res
            x_max_m = plot_df.columns.max() * res
            y_max_m = plot_df.index.max() * res
            
            extent = [x_min_m, x_max_m, 0, y_max_m]
            
            # 'nearest' interpolation shows the pixels
            im = ax.imshow(plot_df.values, origin='lower', extent=extent, 
                           cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')
            
            ax.set_title(f"{label} Grid", fontweight='bold')
            ax.set_xlabel("Alongshore (m)")
            
            # Apply Zoom Limits (Match Full View Flip: Max -> Min)
            ax.set_xlim(z_max_m, z_min_m) 
            ax.set_ylim(0, 25)
            
            # Add gridlines to emphasize cell size
            ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        axes2[0].set_ylabel("Elevation (m)", fontweight='bold')
        cbar2 = fig2.colorbar(im, ax=axes2, label="Cumulative Erosion (m)", location='bottom', fraction=0.05, pad=0.1)
        fig2.suptitle(f"{location}: Detail View of Largest Erosion Event", fontsize=16, fontweight='bold')
        
        path2 = os.path.join(out_dir, f"{location}_Spatial_Zoom.png")
        plt.savefig(path2, dpi=200)
        print(f"  Saved: {path2}")
        plt.close(fig2)
    else:
        print("[WARN] 10cm grid missing, cannot auto-zoom.")

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
        print("No data found. Check directories.")
        return

    plot_spatial_comparison(grids, args.location, out_dir)

if __name__ == "__main__":
    main()