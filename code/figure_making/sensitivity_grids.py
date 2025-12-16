#!/usr/bin/env python3
"""
plot_spatial_sensitivity_final.py

Purpose:
    Visualizes SPATIAL differences between grid resolutions.
    
    FIXES:
    1. EXPLICIT file pattern matching based on your provided path structure.
    2. VERBOSE logging: Prints exactly how many files are found and loaded.
    3. ROBUST data cleaning: Drops bad columns instead of failing the whole file.
    4. CORRECT Extents: Maps Polygon IDs (X) and Elevation (Y) accurately.

Usage:
    python3 code/figure_making/plot_spatial_sensitivity_final.py --location DelMar
"""

import os
import glob
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

CMAP_NAME = 'magma_r'
VMAX_CUMULATIVE = 6.0 

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# ==============================================================================
# 2. DATA LOADING ENGINE (The Fix)
# ==============================================================================

def get_base_dir():
    if platform.system() == 'Darwin':
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

def clean_and_snap_grid(df, resolution_val):
    """
    Parses headers (M3C2_0.10m -> 0.10) and sets correct index/columns.
    """
    # 1. Sanitize Columns
    # Remove any column that doesn't look like an elevation bin (optional safety)
    # For now, we assume all cols are bins as per your pipeline.
    
    # Clean Headers: Remove characters to get numbers
    cleaned_cols = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)
    
    try:
        # Force empty strings to NaN then drop
        cleaned_cols = pd.to_numeric(cleaned_cols, errors='coerce')
        
        # Check if we lost columns
        if np.isnan(cleaned_cols).all():
            print("    [WARN] All columns failed to parse as numbers. Check CSV headers.")
            return None
            
        # Snap to grid index (0.1 -> 1, 0.2 -> 2)
        scale = 1.0 / resolution_val
        new_cols = (cleaned_cols * scale).round().astype(int)
        
        df.columns = new_cols
        
        # Drop any columns that became NaN during parsing
        df = df.loc[:, ~df.columns.isna()]
        
    except Exception as e:
        print(f"    [WARN] Column parsing error: {e}")
        return None

    # 2. Sanitize Index (Polygon ID)
    try:
        df.index = df.index.astype(int)
    except:
        print("    [WARN] Index parsing error (PolygonID not int).")
        return None

    # 3. Sort (Critical for plotting)
    df = df.sort_index(axis=0) # Sort Polygons (Rows)
    df = df.sort_index(axis=1) # Sort Elevations (Cols)
    
    return df

def load_cumulative_grid(location, res_val, file_tag):
    base_dir = get_base_dir()
    erosion_dir = os.path.join(base_dir, 'results', location, 'erosion')
    
    if not os.path.exists(erosion_dir):
        print(f"[ERROR] Directory missing: {erosion_dir}")
        return None

    cumulative_df = None
    count = 0
    
    # Iterate over date folders
    # Structure: results/DelMar/erosion/YYYYMMDD_to_YYYYMMDD/
    subdirs = sorted([d for d in os.listdir(erosion_dir) if os.path.isdir(os.path.join(erosion_dir, d))])
    
    print(f"[{file_tag}] Searching {len(subdirs)} survey intervals...")

    for folder in subdirs:
        folder_path = os.path.join(erosion_dir, folder)
        
        # EXPLICIT PATTERN MATCHING based on your provided example
        # Pattern: *ero_grid_10cm_filled.csv
        search_pattern = os.path.join(folder_path, f"*ero_grid_{file_tag}_filled.csv")
        matches = glob.glob(search_pattern)
        
        if not matches:
            # Fallback: try without 'ero_' prefix if finding generic grids
            search_pattern = os.path.join(folder_path, f"*grid_{file_tag}_filled.csv")
            matches = glob.glob(search_pattern)
            
        if matches:
            grid_path = matches[0] # Take the first match
            try:
                # Load
                df = pd.read_csv(grid_path, index_col=0).fillna(0)
                
                # Clean
                df_clean = clean_and_snap_grid(df, res_val)
                
                if df_clean is not None:
                    if cumulative_df is None:
                        cumulative_df = df_clean
                    else:
                        cumulative_df = cumulative_df.add(df_clean, fill_value=0)
                    count += 1
            except Exception as e:
                print(f"    [ERR] Failed to load {os.path.basename(grid_path)}: {e}")
    
    print(f"  -> Successfully aggregated {count} grids.")
    return cumulative_df

# ==============================================================================
# 3. PLOTTING LOGIC
# ==============================================================================

def get_custom_cmap(name, vmax):
    try:
        base = plt.get_cmap(name)
    except:
        import matplotlib.cm as cm
        base = cm.get_cmap(name)
        
    colors = base(np.linspace(0, 1, 256))
    colors[0, :] = [1, 1, 1, 1] # White for 0
    return LinearSegmentedColormap.from_list(f"White_{name}", colors), Normalize(vmin=0, vmax=vmax)

def plot_spatial_comparison(grids, location, out_dir):
    cmap, norm = get_custom_cmap(CMAP_NAME, VMAX_CUMULATIVE)
    
    # --- 1. FULL COASTLINE ---
    print("\n[PLOTTING] Generating Full Coastline View...")
    fig1, axes1 = plt.subplots(3, 1, figsize=(18, 12), sharex=True, constrained_layout=True)
    
    # Establish Global X-Limits (Polygon IDs) from the 10cm grid
    if "10cm" in grids:
        ref_df = grids["10cm"]
    else:
        ref_df = list(grids.values())[0] # Fallback

    global_min_id = ref_df.index.min()
    global_max_id = ref_df.index.max()
    print(f"  Global Extent: Polygon IDs {global_min_id} to {global_max_id}")

    for ax, (label, res, _) in zip(axes1, RESOLUTIONS):
        if label not in grids:
            ax.text(0.5, 0.5, "Data Missing", ha='center')
            continue
        
        df = grids[label] # Already aggregated
        
        # TRANSPOSE: Y=Elevation, X=PolygonID
        # Original df: Index=PolygonID, Cols=ElevIndex
        plot_df = df.T 
        
        # Extent Calculation
        # X: Polygon IDs (Unitless/1m)
        x_start = plot_df.columns.min()
        x_end   = plot_df.columns.max()
        
        # Y: Elevation (Meters) = Index * Resolution
        y_start = plot_df.index.min() * res
        y_end   = plot_df.index.max() * res
        
        extent = [x_start, x_end, y_start, y_end]
        
        # Plot
        im = ax.imshow(plot_df.values, origin='lower', extent=extent, 
                       cmap=cmap, norm=norm, aspect='auto', interpolation='none')
        
        # Stats
        vol = plot_df.values.sum() * (res**2)
        ax.text(0.01, 0.85, f"Vol: {vol:,.0f} m³", transform=ax.transAxes, 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.9))
        
        ax.set_title(f"{label} Grid", fontweight='bold')
        ax.set_ylabel("Elevation (m)")
        
        # Set Consistent Limits
        ax.set_xlim(global_max_id, global_min_id) # Invert X
        ax.set_ylim(0, 30) 

    axes1[-1].set_xlabel("Alongshore Location (Polygon ID)", fontweight='bold')
    fig1.colorbar(im, ax=axes1, label="Cumulative Erosion (m)", location='right', fraction=0.02)
    fig1.suptitle(f"{location}: Grid Resolution Sensitivity (Full Coastline)", fontsize=16, fontweight='bold')
    
    p1 = os.path.join(out_dir, f"{location}_Spatial_Full.png")
    plt.savefig(p1, dpi=200)
    print(f"  Saved: {p1}")
    plt.close(fig1)

    # --- 2. ZOOM VIEW ---
    print("\n[PLOTTING] Generating Zoom View...")
    
    if "10cm" in grids:
        # Find Hotspot
        df_10 = grids["10cm"]
        # Sum rows (axis=1) to find which Polygon ID has most total erosion
        profile = df_10.sum(axis=1)
        peak_id = profile.idxmax()
        
        zoom_w = 40 # +/- 40 polygons
        z_min = peak_id - zoom_w
        z_max = peak_id + zoom_w
        
        print(f"  Zooming on Polygon ID {peak_id} (Range: {z_min}-{z_max})")

        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharey=True, constrained_layout=True)
        
        for ax, (label, res, _) in zip(axes2, RESOLUTIONS):
            if label not in grids: continue
            
            df = grids[label]
            plot_df = df.T
            
            # Recalc extent for this specific grid
            x_s = plot_df.columns.min()
            x_e = plot_df.columns.max()
            y_s = plot_df.index.min() * res
            y_e = plot_df.index.max() * res
            extent = [x_s, x_e, y_s, y_e]
            
            # Nearest neighbor interpolation to show pixelation
            im = ax.imshow(plot_df.values, origin='lower', extent=extent, 
                           cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')
            
            ax.set_title(f"{label} Grid", fontweight='bold')
            ax.set_xlabel("Polygon ID")
            
            # Apply Zoom
            ax.set_xlim(z_max, z_min) 
            ax.set_ylim(0, 25)
            
            ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        axes2[0].set_ylabel("Elevation (m)", fontweight='bold')
        cbar2 = fig2.colorbar(im, ax=axes2, label="Cumulative Erosion (m)", location='bottom', fraction=0.05, pad=0.1)
        fig2.suptitle(f"{location}: Detail View of Largest Erosion Event", fontsize=16, fontweight='bold')
        
        p2 = os.path.join(out_dir, f"{location}_Spatial_Zoom.png")
        plt.savefig(p2, dpi=200)
        print(f"  Saved: {p2}")
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

    grids = {}
    print(f"--- Loading Grids for {args.location} ---")
    
    for label, val, tag in RESOLUTIONS:
        df = load_cumulative_grid(args.location, val, tag)
        if df is not None and not df.empty:
            grids[label] = df
        else:
            print(f"  [WARN] No data loaded for {label}")
    
    if not grids:
        print("[ERROR] No data found for any resolution.")
        return

    plot_spatial_comparison(grids, args.location, out_dir)

if __name__ == "__main__":
    main()