#!/usr/bin/env python3
"""
plot_spatial_sensitivity.py

Purpose:
    Visualizes the SPATIAL differences between grid resolutions.
    
    CORRECTIONS BASED ON CUM_EROSION.PY:
    1. Extent Logic: Uses (min, max) indices directly from data, not assuming 0.
    2. Orientation: Transposes data (df.T) so X=Alongshore, Y=Elevation.
    3. Scaling: Multiplies Index by Resolution to get real-world Meters.

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
VMAX_CUMULATIVE = 5.0  # Saturation point (meters) for heatmap

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# ==============================================================================
# 2. HELPER FUNCTIONS (Matched to cum_erosion.py)
# ==============================================================================

def get_base_dir():
    if platform.system() == 'Darwin':
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

def get_custom_cmap(name, vmax):
    base = cm.get_cmap(name, 256)
    colors = base(np.linspace(0, 1, 256))
    colors[0, :] = [1, 1, 1, 1] # White background for 0
    return LinearSegmentedColormap.from_list(f"White_{name}", colors), Normalize(vmin=0, vmax=vmax)

def clean_and_snap_grid(df, resolution_val):
    """
    Identical logic to cum_erosion.py to ensure grids line up.
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

    # Clean Index (Polygon ID / Alongshore)
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
                        cumulative_df = cumulative_df.add(df_clean, fill_value=0)
            except:
                pass
                
    return cumulative_df

# ==============================================================================
# 3. PLOTTING LOGIC
# ==============================================================================

def plot_spatial_comparison(grids, location, out_dir):
    cmap, norm = get_custom_cmap(CMAP_NAME, VMAX_CUMULATIVE)
    
    # --- PRE-CALCULATE GLOBAL BOUNDS (Meters) ---
    # We use the 10cm grid to define the "True" physical extent of the coastline.
    # Note: Polygon IDs map to physical meters via the Resolution.
    # ID 100 @ 10cm = 10m. ID 100 @ 1m = 100m.
    ref_grid = grids.get("10cm") or list(grids.values())[0]
    ref_data = ref_grid['data']
    ref_res  = ref_grid['res']
    
    # Calculate physical meters for the reference grid
    # min_id, max_id = ref_data.index.min(), ref_data.index.max()
    # global_x_min = min_id * ref_res
    # global_x_max = max_id * ref_res
    
    # 1. SETUP FIGURE 1: FULL COASTLINE
    fig1, axes1 = plt.subplots(3, 1, figsize=(18, 12), sharex=True, constrained_layout=True)
    
    print("  Generating Full Coastline Plot...")
    
    # Get global max X from the 10cm grid for consistent plotting
    if "10cm" in grids:
        g_df = grids["10cm"]['data']
        # Physical extent is Max Index * Resolution
        global_x_limit = g_df.index.max() * 0.10
    else:
        # Fallback
        global_x_limit = 2000 

    for ax, (label, res, _) in zip(axes1, RESOLUTIONS):
        if label not in grids: 
            ax.text(0.5,0.5,"N/A", ha='center')
            continue
        
        df = grids[label]['data']
        
        # --- MATCHING LOGIC FROM CUM_EROSION.PY ---
        # 1. Transpose: df.T means Rows=Elevation, Cols=PolygonID
        plot_df = df.T 
        
        # 2. Calculate Extents based on real-world meters
        # Cols (X) = PolygonID * Resolution
        x_start = plot_df.columns.min() * res
        x_end   = plot_df.columns.max() * res
        
        # Rows (Y) = Elev Index * Resolution
        y_start = plot_df.index.min() * res
        y_end   = plot_df.index.max() * res
        
        # Standard Imshow Extent: [left, right, bottom, top]
        extent = [x_start, x_end, y_start, y_end]
        
        im = ax.imshow(plot_df.values, origin='lower', extent=extent, 
                       cmap=cmap, norm=norm, aspect='auto', interpolation='none')
        
        # Vol Label
        vol = plot_df.values.sum() * (res**2)
        ax.text(0.01, 0.9, f"Vol: {vol:,.0f} m³", transform=ax.transAxes, 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.9))
        
        ax.set_title(f"{label} Resolution", fontweight='bold')
        ax.set_ylabel("Elevation (m)")
        
        # Align X axis to match the global extent (North-South orientation)
        # Assuming we want to display from North (High ID? Low ID?)
        # Usually coastal plots flip X. Let's flip Max -> Min.
        ax.set_xlim(global_x_limit, 0) 
        ax.set_ylim(0, 30)

    fig1.colorbar(im, ax=axes1, label="Cumulative Erosion (m)", location='right', fraction=0.02)
    fig1.suptitle(f"{location}: Grid Resolution Sensitivity (Full Coastline)", fontsize=16, fontweight='bold')
    
    path1 = os.path.join(out_dir, f"{location}_Spatial_Full.png")
    plt.savefig(path1, dpi=200)
    print(f"  Saved: {path1}")
    plt.close(fig1)

    # 2. SETUP FIGURE 2: ZOOM ON LARGEST EVENT
    print("  Generating Zoom Plot...")
    
    # Find peak using 10cm reference
    if "10cm" in grids:
        df_10 = grids["10cm"]['data']
        # Sum alongshore columns (which are elevation bins) to get total erosion per meter
        # df index is PolygonID.
        profile = df_10.sum(axis=1) 
        peak_idx = profile.idxmax()
        peak_loc_m = peak_idx * 0.10
        
        zoom_r = 30 # meters
        x_min_z = peak_loc_m - zoom_r
        x_max_z = peak_loc_m + zoom_r
        
        print(f"    Zooming at {peak_loc_m:.1f}m ({x_min_z}-{x_max_z}m)")

        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharey=True, constrained_layout=True)
        
        for ax, (label, res, _) in zip(axes2, RESOLUTIONS):
            if label not in grids: continue
            
            df = grids[label]['data']
            plot_df = df.T
            
            x_s = plot_df.columns.min() * res
            x_e = plot_df.columns.max() * res
            y_s = plot_df.index.min() * res
            y_e = plot_df.index.max() * res
            
            extent = [x_s, x_e, y_s, y_e]
            
            # Use 'nearest' to show pixelation clearly
            im = ax.imshow(plot_df.values, origin='lower', extent=extent, 
                           cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')
            
            ax.set_title(f"{label} Grid", fontweight='bold')
            ax.set_xlabel("Alongshore (m)")
            
            # Apply Zoom Limits (Match flip from Fig 1)
            ax.set_xlim(x_max_z, x_min_z) 
            ax.set_ylim(0, 25)
            
            # Gridlines
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
        print("No data found.")
        return

    plot_spatial_comparison(grids, args.location, out_dir)

if __name__ == "__main__":
    main()