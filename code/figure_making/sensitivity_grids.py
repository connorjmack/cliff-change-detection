#!/usr/bin/env python3
"""
plot_grid_sensitivity_final_v2.py

Visualizes "Cliff Activity Index" (Cumulative Erosion) across different grid resolutions.
CORRECTLY scales axes so that 1m, 25cm, and 10cm grids align spatially.
FIXES layout to prevent colorbar overlapping x-axis labels.
RESTORES Y-Axis scaling (divides by 100 if units are detected as cm).

Usage:
    python3 plot_grid_sensitivity_final_v2.py --location DelMar
    python3 plot_grid_sensitivity_final_v2.py --location DelMar --zoom 1450 1550
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
import re

# --- Configuration ---
RESOLUTIONS = ['1m', '25cm', '10cm']
VMAX_EROSION = 6.0  # Saturation point in meters

# --- Styling ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 12

def get_resolution_value(res_str):
    """Returns resolution in meters (float)."""
    if 'cm' in res_str:
        return float(res_str.replace('cm', '')) / 100.0
    elif 'm' in res_str:
        return float(res_str.replace('m', ''))
    return 0.1

def normalize_resolution_string(res):
    """Matches file naming convention (1m -> 100cm)."""
    if res == '1m': return '100cm'
    return res

def get_custom_magma_cmap():
    """
    Creates 'magma_r' but with a PURE WHITE background for 0.
    """
    # Get standard magma_r (reversed)
    magma = cm.get_cmap('magma_r', 256)
    newcolors = magma(np.linspace(0, 1, 256))
    
    # FORCE ZERO TO PURE WHITE [R, G, B, Alpha]
    newcolors[0, :] = np.array([1, 1, 1, 1]) 
    
    return LinearSegmentedColormap.from_list("WhiteMagma", newcolors)

def load_and_sum_grids(base_dir, location, resolution):
    """
    Loads/sums grids and returns DataFrame AND max physical dimensions.
    """
    file_res = normalize_resolution_string(resolution)
    res_val = get_resolution_value(resolution)
    data_dir = os.path.join(base_dir, 'results', location, 'erosion')
    
    if not os.path.isdir(data_dir):
        return None

    cumulative_df = None
    
    # Iterate all survey folders
    for date_folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, date_folder)
        if not os.path.isdir(folder_path): continue
        
        # Find file
        patterns = [f"_ero_grid_{file_res}_filled.csv", f"grid_{file_res}_filled.csv"]
        target_file = None
        for f in os.listdir(folder_path):
            if any(p in f for p in patterns) and f.endswith('.csv'):
                target_file = os.path.join(folder_path, f)
                break
        
        if not target_file: continue

        try:
            # Load
            df = pd.read_csv(target_file, index_col=0)
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            
            # Initialize cumulative df with same shape if needed
            if cumulative_df is None:
                cumulative_df = df
            else:
                # Align indices to ensure we are adding the same polygons
                cumulative_df = cumulative_df.add(df, fill_value=0)
                
        except:
            continue

    if cumulative_df is None:
        return None

    # --- CLEANING AND SCALING ---
    # 1. Clean Column Headers (Elevation)
    # Remove characters, keep numbers. e.g. "M3C2_0.10m" -> 0.10
    cleaned_cols = cumulative_df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)
    cumulative_df.columns = cleaned_cols.astype(float) # Columns are now Elevation (m)
    
    # 2. Clean Index (Alongshore Position)
    # The index is the Polygon ID. We need to convert this to Meters for plotting.
    # We assume Polygon 0 = 0m, Polygon 1 = 1 * res_val, etc.
    cumulative_df.index = cumulative_df.index.astype(int)
    
    return cumulative_df

def plot_panel(ax, df, resolution, global_max_x, zoom_range=None):
    """
    Plots a single resolution panel, scaled to physical meters.
    """
    res_val = get_resolution_value(resolution)
    
    # Transpose so X=Alongshore, Y=Elevation
    plot_df = df.T
    plot_df.sort_index(axis=0, inplace=True) # Sort Y (Elevation)
    plot_df.sort_index(axis=1, inplace=True) # Sort X (Alongshore)
    
    # --- CALCULATE PHYSICAL EXTENT (METERS) ---
    # X-Axis: Index * Resolution
    x_indices = plot_df.columns.values
    x_min_m = x_indices.min() * res_val
    x_max_m = x_indices.max() * res_val
    
    # Y-Axis: Check if scaling is needed
    y_min_m = plot_df.index.min()
    y_max_m = plot_df.index.max()
    
    # --- RESTORED LOGIC: Auto-scale Y axis if > 100 (assume cm -> m) ---
    if y_max_m > 100:
        y_min_m /= 100.0
        y_max_m /= 100.0
    
    extent = [x_min_m, x_max_m, y_min_m, y_max_m]
    
    # --- PLOTTING ---
    cmap = get_custom_magma_cmap()
    norm = Normalize(vmin=0, vmax=VMAX_EROSION)
    
    im = ax.imshow(plot_df.values, origin='lower', extent=extent, 
                   cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    
    # Styling
    ax.set_title(f"{resolution} Resolution", fontsize=14, fontweight='bold')
    ax.set_ylabel("Elevation (m)", fontsize=12)
    ax.grid(False)
    
    # --- LOCK X-AXIS ---
    # This ensures 1m (coarse) and 10cm (fine) share the exact same physical width
    if zoom_range:
        ax.set_xlim(zoom_range[1], zoom_range[0]) # Inverted X for Cliff View
    else:
        ax.set_xlim(global_max_x, 0) # Inverted X, 0 to Max
        
    return im

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', required=True, help="Survey Location (e.g. DelMar)")
    parser.add_argument('--zoom', nargs=2, type=int, help="Zoom limits (e.g. 1450 1600)")
    args = parser.parse_args()

    import platform
    if platform.system() == 'Darwin':
         base_dir = '/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs'
    else:
         base_dir = '/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs'

    out_dir = os.path.join(base_dir, 'figures', 'grid_sensitivity')
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"--- Generating Grid Sensitivity Plot for {args.location} ---")
    
    # 1. Load Data first to find Global Max X (Physical Distance)
    data_store = {}
    global_max_dist = 0
    
    for res in RESOLUTIONS:
        print(f"  Loading {res}...")
        df = load_and_sum_grids(base_dir, args.location, res)
        if df is not None:
            data_store[res] = df
            
            # Calculate physical length of this grid
            res_val = get_resolution_value(res)
            max_dist = df.index.max() * res_val
            if max_dist > global_max_dist:
                global_max_dist = max_dist
    
    if not data_store:
        print("No data found!")
        return

    print(f"  Global Extent: {global_max_dist:.1f} meters")

    # 2. Setup Figure
    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True, sharey=True)
    
    # --- LAYOUT ADJUSTMENT: Reserve bottom space for colorbar ---
    # bottom=0.15 lifts the lowest plot up, creating whitespace for the colorbar
    plt.subplots_adjust(hspace=0.3, bottom=0.15) 
    
    mappable = None
    
    # 3. Plot Each
    for i, res in enumerate(RESOLUTIONS):
        if res in data_store:
            mappable = plot_panel(axes[i], data_store[res], res, global_max_dist, args.zoom)
        else:
            axes[i].text(0.5, 0.5, f"No Data for {res}", ha='center', va='center')
            axes[i].set_title(f"{res} Resolution")

    # 4. Final Formatting
    axes[-1].set_xlabel("Alongshore Location (m)", fontsize=14, fontweight='bold')
    
    # Shared Colorbar
    # Placed at bottom=0.06, width=0.7, height=0.025
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.025]) 
    if mappable:
        cb = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
        cb.set_label("Cumulative Erosion Depth (m)", fontsize=14, fontweight='bold', labelpad=10)
        cb.ax.tick_params(labelsize=12)

    zoom_str = f"(Zoom: {args.zoom[0]}-{args.zoom[1]}m)" if args.zoom else "(Full Extent)"
    fig.suptitle(f"{args.location}: Grid Resolution Sensitivity Analysis {zoom_str}", 
                 fontsize=20, fontweight='bold', y=0.96)
    
    suffix = "zoomed" if args.zoom else "full"
    out_path = os.path.join(out_dir, f"{args.location}_Sensitivity_{suffix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out_path}")

if __name__ == '__main__':
    main()