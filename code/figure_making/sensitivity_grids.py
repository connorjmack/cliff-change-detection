#!/usr/bin/env python3
"""
plot_spatial_sensitivity_v4.py

Purpose:
    Visualizes SPATIAL differences between grid resolutions (1m, 25cm, 10cm).
    
    FIXES:
    1. Fixed ValueError: "The truth value of a DataFrame is ambiguous" by removing 'or' logic.
    2. Fixed MatplotlibDeprecationWarning by using 'matplotlib.colormaps'.
    3. Retains correct Axis Scaling (X=PolygonID, Y=Meters) from V3.

Usage:
    python3 code/figure_making/plot_spatial_sensitivity_v4.py --location DelMar
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
from datetime import datetime

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

RESOLUTIONS = [
    ("1m",   1.00, "100cm"),
    ("25cm", 0.25, "25cm"),
    ("10cm", 0.10, "10cm")
]

CMAP_NAME = 'magma_r'
VMAX_CUMULATIVE = 6.0 

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def get_base_dir():
    if platform.system() == 'Darwin':
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

def normalize_resolution_for_files(resolution):
    if resolution == '1m': return '100cm'
    elif resolution == '100cm': return '100cm'
    else: return resolution

def get_custom_cmap(name, vmax):
    # Fix for Matplotlib Deprecation Warning
    try:
        base = matplotlib.colormaps[name]
    except:
        base = plt.get_cmap(name)
        
    colors = base(np.linspace(0, 1, 256))
    colors[0, :] = [1, 1, 1, 1] # White for 0
    return LinearSegmentedColormap.from_list(f"White_{name}", colors), Normalize(vmin=0, vmax=vmax)

def clean_and_snap_grid(df, resolution_val):
    """
    Standard cleaning: Strips headers to floats, Index to ints.
    """
    # Columns (Elevation)
    cleaned_cols = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)
    try:
        col_floats = cleaned_cols.astype(float)
        # Snap to grid index
        scale = 1.0 / resolution_val
        new_cols = (col_floats * scale).round().astype(int)
        df.columns = new_cols
    except:
        return None

    # Index (Polygon ID)
    try:
        df.index = df.index.astype(int)
    except:
        return None

    # SORT (Crucial for imshow)
    df = df.sort_index(axis=0) # Sort Polygons
    df = df.sort_index(axis=1) # Sort Elevations
    return df

def find_grid_files(base_dir, location, file_tag):
    """Locates files using fuzzy matching logic from cum_erosion.py"""
    erosion_dir = os.path.join(base_dir, 'results', location, 'erosion')
    if not os.path.exists(erosion_dir): return []

    grid_files = []
    
    # Try filled first, then cleaned
    patterns = [f"grid_{file_tag}_filled.csv", f"grid_{file_tag}_cleaned.csv"]

    for date_folder in sorted(os.listdir(erosion_dir)):
        folder_path = os.path.join(erosion_dir, date_folder)
        if not os.path.isdir(folder_path): continue
            
        files_in_folder = os.listdir(folder_path)
        found_file = None
        
        for pattern in patterns:
            match = [f for f in files_in_folder if pattern in f and f.endswith('.csv')]
            if match:
                found_file = os.path.join(folder_path, match[0])
                break
        
        if found_file:
            grid_files.append(found_file)
            
    return grid_files

def calculate_cumulative_data(files, res_val):
    """Sums up the grids."""
    if not files: return None
    
    # print(f"    Summing {len(files)} grids...")
    cumulative_df = None
    
    for f in files:
        try:
            df = pd.read_csv(f, index_col=0).fillna(0)
            df_clean = clean_and_snap_grid(df, res_val)
            
            if df_clean is not None:
                if cumulative_df is None:
                    cumulative_df = df_clean.fillna(0)
                else:
                    cumulative_df = cumulative_df.add(df_clean.fillna(0), fill_value=0)
        except:
            pass
            
    return cumulative_df

# ==============================================================================
# 3. PLOTTING LOGIC
# ==============================================================================

def plot_spatial_comparison(grids, location, out_dir):
    cmap, norm = get_custom_cmap(CMAP_NAME, VMAX_CUMULATIVE)
    
    # --- 1. SETUP FIGURE 1: FULL COASTLINE ---
    fig1, axes1 = plt.subplots(3, 1, figsize=(18, 12), sharex=True, constrained_layout=True)
    
    # Use 10cm grid to find global X limits (Polygon IDs)
    # FIX: Explicit check instead of 'or'
    if "10cm" in grids:
        ref_df = grids["10cm"]
    else:
        ref_df = list(grids.values())[0]

    global_min_id = ref_df.index.min()
    global_max_id = ref_df.index.max()
    
    print(f"  Plotting extent: Polygon IDs {global_min_id} to {global_max_id}")

    for ax, (label, res, _) in zip(axes1, RESOLUTIONS):
        if label not in grids: 
            ax.text(0.5,0.5,"N/A", ha='center')
            continue
        
        df = grids[label]
        
        # TRANSPOSE: Rows=Elevation, Cols=PolygonID
        plot_df = df.T 
        
        # --- EXTENT CALCULATION ---
        # X Axis = Raw Polygon IDs (Do NOT multiply by res)
        x_start = plot_df.columns.min()
        x_end   = plot_df.columns.max()
        
        # Y Axis = Physical Elevation (Index * Res)
        y_start = plot_df.index.min() * res
        y_end   = plot_df.index.max() * res
        
        extent = [x_start, x_end, y_start, y_end]
        
        im = ax.imshow(plot_df.values, origin='lower', extent=extent, 
                       cmap=cmap, norm=norm, aspect='auto', interpolation='none')
        
        # Vol Label
        vol = plot_df.values.sum() * (res**2)
        ax.text(0.01, 0.9, f"Vol: {vol:,.0f} m³", transform=ax.transAxes, 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.9))
        
        ax.set_title(f"{label} Resolution", fontweight='bold')
        ax.set_ylabel("Elevation (m)")
        
        # Align X axis (Inverted for standard view)
        ax.set_xlim(global_max_id, global_min_id) 
        ax.set_ylim(0, 30)

    axes1[-1].set_xlabel("Polygon ID (Alongshore Index)", fontweight='bold')
    fig1.colorbar(im, ax=axes1, label="Cumulative Erosion (m)", location='right', fraction=0.02)
    fig1.suptitle(f"{location}: Grid Resolution Sensitivity (Full Coastline)", fontsize=16, fontweight='bold')
    
    path1 = os.path.join(out_dir, f"{location}_Spatial_Full.png")
    plt.savefig(path1, dpi=200)
    print(f"  Saved: {path1}")
    plt.close(fig1)

    # --- 2. SETUP FIGURE 2: ZOOM ON LARGEST EVENT ---
    print("  Generating Zoom Plot...")
    
    # Auto-detect hotspot using 10cm grid
    if "10cm" in grids:
        df_10 = grids["10cm"]
        # Sum columns (Elevations) to get total erosion per Polygon ID
        # Index is Polygon ID
        profile = df_10.sum(axis=1)
        peak_id = profile.idxmax()
        
        # Define Window (+/- 25 Polygons)
        zoom_w = 25
        z_min = peak_id - zoom_w
        z_max = peak_id + zoom_w
        
        print(f"    Zooming at Polygon ID {peak_id} ({z_min}-{z_max})")

        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharey=True, constrained_layout=True)
        
        for ax, (label, res, _) in zip(axes2, RESOLUTIONS):
            if label not in grids: continue
            
            df = grids[label]
            plot_df = df.T
            
            # Recalculate extent for this grid
            x_s = plot_df.columns.min()
            x_e = plot_df.columns.max()
            y_s = plot_df.index.min() * res
            y_e = plot_df.index.max() * res
            extent = [x_s, x_e, y_s, y_e]
            
            # Use 'nearest' to show pixelation
            im = ax.imshow(plot_df.values, origin='lower', extent=extent, 
                           cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
            
            ax.set_title(f"{label} Grid", fontweight='bold')
            ax.set_xlabel("Polygon ID")
            
            # Apply Zoom Limits (Match Full View Flip)
            ax.set_xlim(z_max, z_min) 
            ax.set_ylim(0, 25)
            
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
    print(f"--- Loading Grids for {args.location} ---")
    
    for label, val, tag in RESOLUTIONS:
        files = find_grid_files(base_dir, args.location, tag)
        if files:
            df = calculate_cumulative_data(files, val)
            if df is not None:
                grids[label] = df
    
    if not grids:
        print("No data found.")
        return

    plot_spatial_comparison(grids, args.location, out_dir)

if __name__ == "__main__":
    main()