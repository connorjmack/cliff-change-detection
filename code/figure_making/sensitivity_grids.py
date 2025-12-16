#!/usr/bin/env python3
"""
plot_spatial_sensitivity.py

Purpose:
    Visualizes the SPATIAL differences between grid resolutions (10cm, 25cm, 1m).
    Generates two figures:
      1. Full Coastline Comparison (Stacked Heatmaps)
      2. Zoomed-in View of the Largest Erosion Event (Side-by-Side)

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
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

RESOLUTIONS = [
    ("1m",   1.00, "100cm"),
    ("25cm", 0.25, "25cm"),
    ("10cm", 0.10, "10cm")
]

# Visual Settings
CMAP_NAME = 'magma_r' # White -> Yellow -> Red/Black
VMAX_CUMULATIVE = 6.0 # Max depth (m) for colorbar saturation

# Font Sizes
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 12

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
    base_cmap = cm.get_cmap(name, 256)
    newcolors = base_cmap(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 1]) # 0 = White
    return LinearSegmentedColormap.from_list(f"White_{name}", newcolors), Normalize(vmin=0, vmax=vmax)

def clean_and_snap_grid(df, resolution_val):
    """Parses columns (elevation) and index (alongshore ID) to numeric."""
    # Remove text like 'M3C2_' or 'm'
    cleaned_cols = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)
    try:
        col_floats = cleaned_cols.astype(float)
        # Snap to resolution grid (e.g. 0.1, 0.2 -> index 1, 2)
        scale = 1.0 / resolution_val
        new_cols = (col_floats * scale).round().astype(int)
        df.columns = new_cols
        df.index = df.index.astype(int)
        return df
    except:
        return None

def load_cumulative_grid(location, res_val, file_tag):
    base_dir = get_base_dir()
    erosion_dir = os.path.join(base_dir, 'results', location, 'erosion')
    
    if not os.path.exists(erosion_dir):
        print(f"[ERROR] Missing dir: {erosion_dir}")
        return None

    # Find all filled grids for this resolution
    cumulative_df = None
    
    # Sort folders by date
    intervals = sorted([d for d in os.listdir(erosion_dir) if os.path.isdir(os.path.join(erosion_dir, d))])
    
    print(f"  Accumulating {file_tag} grids from {len(intervals)} surveys...")
    
    for interval in intervals:
        grid_path = os.path.join(erosion_dir, interval, f"{interval}_ero_grid_{file_tag}_filled.csv")
        
        if os.path.exists(grid_path):
            try:
                df = pd.read_csv(grid_path, index_col=0).fillna(0)
                # Clean header/index structure
                df_clean = clean_and_snap_grid(df, res_val)
                
                if df_clean is not None:
                    if cumulative_df is None:
                        cumulative_df = df_clean
                    else:
                        # Align and add (handles changing grid sizes if any)
                        cumulative_df = cumulative_df.add(df_clean, fill_value=0)
            except:
                pass

    return cumulative_df

# ==============================================================================
# 3. PLOTTING: FULL COASTLINE STACK
# ==============================================================================

def plot_full_coastline(grids_dict, location, out_dir):
    """Plots 3 stacked heatmaps (1m, 25cm, 10cm) of the entire coast."""
    if not grids_dict: return

    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True, constrained_layout=True)
    cmap, norm = get_custom_cmap(CMAP_NAME, VMAX_CUMULATIVE)

    # Determine global X limits (meters)
    # We use the 10cm grid (highest res) to define the master extent if available
    master_ref = grids_dict.get("10cm") or list(grids_dict.values())[0]
    master_df_T = master_ref['data'].T
    x_min_global = master_df_T.columns.min() * master_ref['res']
    x_max_global = master_df_T.columns.max() * master_ref['res']

    for ax, (label, res_val, _) in zip(axes, RESOLUTIONS):
        if label not in grids_dict:
            ax.text(0.5, 0.5, "Data Not Available", ha='center', va='center')
            continue

        df = grids_dict[label]['data']
        # Transpose so X=Alongshore, Y=Elevation
        plot_df = df.T 
        
        # Convert indices to physical units (meters)
        x_indices = plot_df.columns.astype(float)
        y_indices = plot_df.index.astype(float)
        
        x_meters = x_indices * res_val
        y_meters = y_indices * res_val
        
        # Calculate Total Volume for Label
        total_vol = df.sum().sum() * (res_val ** 2)
        
        # Extent = [left, right, bottom, top]
        # Invert X axis (North to South usually) -> max to min
        extent = [x_meters.max(), x_meters.min(), y_meters.min(), y_meters.max()]
        
        im = ax.imshow(plot_df.values, origin='lower', extent=extent, 
                       cmap=cmap, norm=norm, aspect='auto', interpolation='none')
        
        # Annotation Box
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        ax.text(0.01, 0.9, f"Volume: {total_vol:,.0f} $m^3$", transform=ax.transAxes, 
                fontsize=12, fontweight='bold', bbox=props)
        
        ax.set_title(f"{label} Resolution", fontweight='bold', loc='center')
        ax.set_ylabel("Elevation (m)")
        
        # Force X limits to align visual comparison
        ax.set_xlim(x_max_global, x_min_global)
        ax.set_ylim(0, 30) # Standard cliff height

    # Shared Colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
    cbar.set_label("Cumulative Erosion (m)", fontsize=12, fontweight='bold')
    
    axes[-1].set_xlabel("Alongshore Location (m)", fontsize=12, fontweight='bold')
    
    fig.suptitle(f"{location}: Spatial Resolution Sensitivity (Full Coastline)", 
                 fontsize=18, fontweight='bold')
    
    save_path = os.path.join(out_dir, f"{location}_Spatial_Sensitivity_Full.png")
    plt.savefig(save_path, dpi=300)
    print(f"[SUCCESS] Full coastline figure saved: {save_path}")

# ==============================================================================
# 4. PLOTTING: ZOOM ON LARGEST EVENT
# ==============================================================================

def find_largest_event_bbox(df_10cm, res_val):
    """Finds the bounding box of the single largest erosion hotspot in the 10cm grid."""
    # Simple hotspot detection: Find column (alongshore index) with max cumulative erosion sum
    alongshore_sums = df_10cm.sum(axis=1) # Sum vertical column
    max_idx = alongshore_sums.idxmax()
    
    # Define a window around it (e.g., +/- 25 meters)
    window_m = 25.0 
    window_cells = int(window_m / res_val)
    
    center_idx = int(max_idx)
    start_idx = max(0, center_idx - window_cells)
    end_idx = center_idx + window_cells
    
    return start_idx, end_idx, window_m

def plot_event_zoom(grids_dict, location, out_dir):
    """Plots a zoomed-in view of the largest event across 3 resolutions."""
    if "10cm" not in grids_dict:
        print("[WARN] 10cm grid needed for hotspot detection. Skipping zoom plot.")
        return

    # 1. Find Hotspot using 10cm grid
    df_10 = grids_dict["10cm"]['data']
    res_10 = 0.10
    
    # Get index range in 10cm grid units
    s_idx, e_idx, width_m = find_largest_event_bbox(df_10, res_10)
    
    # Convert to physical coordinates (Meters)
    # Note: Indices might be reversed in plotting, but physical meters are absolute
    x_center_m = ((s_idx + e_idx) / 2) * res_10
    x_min_m = x_center_m - width_m
    x_max_m = x_center_m + width_m

    print(f"  Zooming on event at ~{x_center_m:.0f}m alongshore...")

    # 2. Setup Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, constrained_layout=True)
    cmap, norm = get_custom_cmap(CMAP_NAME, VMAX_CUMULATIVE)

    for ax, (label, res_val, _) in zip(axes, RESOLUTIONS):
        if label not in grids_dict: continue
        
        df = grids_dict[label]['data']
        plot_df = df.T # X=Alongshore, Y=Elevation
        
        x_indices = plot_df.columns.astype(float)
        y_indices = plot_df.index.astype(float)
        x_meters = x_indices * res_val
        y_meters = y_indices * res_val
        
        # Extent [right, left, bottom, top] for inverted X axis
        extent = [x_meters.max(), x_meters.min(), y_meters.min(), y_meters.max()]
        
        im = ax.imshow(plot_df.values, origin='lower', extent=extent,
                       cmap=cmap, norm=norm, aspect='equal', interpolation='nearest') # 'nearest' shows pixelation!
        
        ax.set_title(f"{label} Grid", fontweight='bold', fontsize=14)
        ax.set_xlabel("Alongshore (m)")
        
        # Zoom In
        ax.set_xlim(x_max_m, x_min_m) # Inverted: Max -> Min
        ax.set_ylim(0, 25)
        
        # Add grid lines to emphasize pixel size
        # Only visible if zoomed in enough, helps visualization
        ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    axes[0].set_ylabel("Elevation (m)", fontweight='bold', fontsize=12)
    
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)
    cbar.set_label("Cumulative Erosion (m)", fontsize=12)

    fig.suptitle(f"{location}: Detail View of Largest Erosion Event", fontsize=16, fontweight='bold')
    
    save_path = os.path.join(out_dir, f"{location}_Spatial_Sensitivity_Zoom.png")
    plt.savefig(save_path, dpi=300)
    print(f"[SUCCESS] Zoom figure saved: {save_path}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default="DelMar")
    args = parser.parse_args()
    
    # 1. Load Data
    grids = {}
    print(f"--- Loading Grids for {args.location} ---")
    
    for label, val, tag in RESOLUTIONS:
        df = load_cumulative_grid(args.location, val, tag)
        if df is not None:
            grids[label] = {'data': df, 'res': val}
    
    if not grids:
        print("[ERROR] No data found.")
        return

    # 2. Setup Output
    base_dir = get_base_dir()
    out_dir = os.path.join(base_dir, "figures", "sensitivity")
    os.makedirs(out_dir, exist_ok=True)

    # 3. Generate Plots
    plot_full_coastline(grids, args.location, out_dir)
    plot_event_zoom(grids, args.location, out_dir)

if __name__ == "__main__":
    main()