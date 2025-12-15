#!/usr/bin/env python3
"""
plot_cumulative_cliff_face_rotated.py

Calculates cumulative erosion/deposition for CLIFF FACE GRIDS.
Creates 3x1 figures showing 1m, 25cm, and 10cm resolutions for each location.

CONFIGURATION:
- X-axis: Alongshore Location (Polygon ID)
- Y-axis: Elevation on Face (m)
- Output: .../figures/cumulative_erosion/

Usage:
    python3 plot_cumulative_cliff_face_rotated.py --erosion
    python3 plot_cumulative_cliff_face_rotated.py --deposition
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime

# Plotting Style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

TARGET_LOCATIONS = ['DelMar', 'Solana', 'SanElijo', 'Encinitas', 'Torrey']
RESOLUTIONS = ['1m', '25cm', '10cm']

def normalize_resolution_for_files(resolution):
    """
    Normalizes resolution string to match file naming convention.
    8_make_grids.py names files with cm labels: 10cm, 25cm, 100cm
    This function converts '1m' -> '100cm' for file searching.
    """
    if resolution == '1m':
        return '100cm'
    elif resolution == '100cm':
        return '100cm'
    else:
        return resolution  # '10cm' and '25cm' stay as is

def clean_and_snap_grid(df, resolution_val):
    """
    1. Strips text from headers (e.g. 'M3C2_0.10m' -> 0.10).
    2. Snaps Elevation columns to integers.
    """
    # 1. Clean Headers: Remove 'M3C2_', 'm', etc to get raw numbers
    cleaned_cols = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)
    
    # 2. Convert to float
    try:
        col_floats = cleaned_cols.astype(float)
    except ValueError as e:
        print(f"    [Error] Could not convert headers to elevation numbers: {e}")
        return None

    # 3. Snap Columns (Elevation) to Integer Grid indices
    scale = 1.0 / resolution_val
    new_cols = (col_floats * scale).round().astype(int)
    df.columns = new_cols
    
    # Ensure index is integer (Polygon ID)
    df.index = df.index.astype(int)
    
    return df

def get_resolution_value(res_str):
    """Convert resolution string to numeric value in meters."""
    if 'cm' in res_str:
        return float(res_str.replace('cm', '')) / 100.0
    elif 'm' in res_str:
        return float(res_str.replace('m', ''))
    return 0.1 # Default

def load_grid_dataframe(filepath, res_val):
    if not os.path.exists(filepath): return None
    try:
        df = pd.read_csv(filepath, index_col=0)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = clean_and_snap_grid(df, res_val)
        return df
    except Exception as e:
        print(f"    Error reading {os.path.basename(filepath)}: {e}")
        return None

def parse_date_from_folder(folder_name):
    match = re.search(r'(\d{8})', folder_name)
    if match: return datetime.strptime(match.group(1), '%Y%m%d')
    match = re.search(r'(\d{4})_(\d{2})_(\d{2})', folder_name)
    if match: return datetime.strptime(f"{match.group(1)}{match.group(2)}{match.group(3)}", '%Y%m%d')
    return None

def find_grid_files(base_dir, location, resolution, data_type, use_filled=True):
    """
    Find grid files for a specific resolution.
    Normalizes resolution to match file naming (1m -> 100cm).
    """
    type_dir = os.path.join(base_dir, 'results', location, data_type)
    if not os.path.isdir(type_dir): return []

    # Normalize resolution for file searching (1m -> 100cm)
    file_resolution = normalize_resolution_for_files(resolution)

    grid_files = []
    
    if data_type == 'erosion':
        patterns = [f"_ero_grid_{file_resolution}_filled.csv", f"grid_{file_resolution}_filled.csv"] if use_filled else \
                   [f"_ero_grid_{file_resolution}_cleaned.csv", f"grid_{file_resolution}_cleaned.csv"]
    else:
        patterns = [f"_dep_grid_{file_resolution}_filled.csv", f"dep_grid_{file_resolution}_filled.csv"] if use_filled else \
                   [f"_dep_grid_{file_resolution}_cleaned.csv", f"dep_grid_{file_resolution}_cleaned.csv"]

    for date_folder in sorted(os.listdir(type_dir)):
        folder_path = os.path.join(type_dir, date_folder)
        if not os.path.isdir(folder_path): continue
            
        files_in_folder = os.listdir(folder_path)
        grid_file = None
        for pattern in patterns:
            match = [f for f in files_in_folder if pattern in f and f.endswith('.csv')]
            if match:
                grid_file = os.path.join(folder_path, match[0])
                break
        
        if grid_file:
            d = parse_date_from_folder(date_folder)
            if d: grid_files.append((d, grid_file))
            
    grid_files.sort(key=lambda x: x[0])
    return grid_files

def calculate_final_cumulative(grid_files, res_val):
    if not grid_files: return None, None, None, []

    print(f"    Summing {len(grid_files)} surveys...")
    
    start_date, first_path = grid_files[0]
    cumulative_df = load_grid_dataframe(first_path, res_val)
    
    if cumulative_df is None: return None, None, None, []
    
    # Baseline shape (Polygons x Height Bins)
    baseline_shape = cumulative_df.shape
    print(f"      Baseline: {start_date.strftime('%Y-%m-%d')} | Shape: {baseline_shape}")
    
    cumulative_df = cumulative_df.fillna(0.0)
    end_date = start_date
    qc_flags = []

    for i, (date, filepath) in enumerate(grid_files[1:]):
        filename = os.path.basename(filepath)
        current_df = load_grid_dataframe(filepath, res_val)
        
        if current_df is None: continue
        
        # STRICT SIZE CHECK
        if current_df.shape != baseline_shape:
            msg = (f"      [QC FLAG] SKIPPED: {filename} (Shape {current_df.shape} != {baseline_shape})")
            print(msg)
            qc_flags.append(msg)
            continue

        current_df = current_df.fillna(0.0)
        end_date = date
        cumulative_df = cumulative_df.add(current_df, fill_value=0)

    cumulative_df.sort_index(axis=0, inplace=True) # Sort Polygon IDs
    cumulative_df.sort_index(axis=1, inplace=True) # Sort Elevations

    return cumulative_df, start_date, end_date, qc_flags

def plot_single_resolution(ax, df, resolution, res_val, data_type, start_date, end_date):
    """
    Plots a single resolution on the given axis.
    FIXED: Y-axis now correctly shows elevation in meters (0-30m) for all resolutions.
    """
    # Volume calculation
    cell_area = res_val * res_val  # Horizontal area of each polygon (m²)
    total_volume = df.sum().sum() * cell_area  # Volume in m³
    
    # TRANSPOSE THE DATA FRAME for plotting
    plot_df = df.T
    
    # X Axis (Polygon IDs)
    x_coords = plot_df.columns.astype(float).values 
    
    # Y Axis - EXPLICIT ELEVATION RANGE
    # The bin indices in plot_df.index represent different things for each resolution:
    # - 1m: indices 0-30 → elevations 0-30m
    # - 25cm: indices 0-120 → elevations 0-30m  
    # - 10cm: indices 0-300 → elevations 0-30m
    
    # Get the number of vertical bins
    n_bins = len(plot_df.index)
    
    # Calculate actual elevation range (should be 0 to ~30m for all)
    max_elevation = n_bins * res_val
    
    # Extent [Xmin, Xmax, Ymin, Ymax] - EXPLICITLY SET Y to 0-max_elevation
    extent = [x_coords.min(), x_coords.max(), 
              0, max_elevation]  # Y always starts at 0
    
    vals = plot_df.values.flatten()
    vals = vals[vals != 0]
    if len(vals) > 0:
        vmin, vmax = np.percentile(vals, [2, 98])
    else:
        vmin, vmax = 0, 1
        
    cmap = 'Reds' if data_type == 'erosion' else 'Blues'
    
    im = ax.imshow(plot_df.values, origin='lower', extent=extent, 
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', 
                   interpolation='none')
    
    cbar = plt.colorbar(im, ax=ax, label=f'Cumulative (m)', fraction=0.02, pad=0.02)
    
    # Title for this subplot
    ax.set_title(f'{resolution} Resolution', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Polygon ID (Alongshore Index)')
    ax.set_ylabel('Elevation (m)')
    ax.invert_xaxis()
    
    # Add volume annotation
    ax.text(0.01, 0.95, f"Volume: {total_volume:.1f} m³", transform=ax.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    return im

def plot_all_resolutions(location, data_type, base_dir, use_filled, output_path, dpi=300):
    """
    Creates a 3x1 figure with all three resolutions for a single location.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    all_qc_flags = []
    date_range_str = None
    any_data = False
    
    for i, resolution in enumerate(RESOLUTIONS):
        print(f"  Processing {resolution}...")
        res_val = get_resolution_value(resolution)
        
        files = find_grid_files(base_dir, location, resolution, data_type, use_filled)
        
        if not files:
            print(f"    No files found for {resolution}")
            axes[i].text(0.5, 0.5, f'No data available\n({resolution})', 
                        ha='center', va='center', fontsize=14, transform=axes[i].transAxes)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            continue
        
        final_df, s_date, e_date, flags = calculate_final_cumulative(files, res_val)
        
        if flags:
            all_qc_flags.extend(flags)
        
        if final_df is not None:
            any_data = True
            plot_single_resolution(axes[i], final_df, resolution, res_val, data_type, s_date, e_date)
            
            # Use the date range from the first successful plot
            if date_range_str is None:
                date_range_str = f"{s_date.strftime('%Y-%m-%d')} to {e_date.strftime('%Y-%m-%d')}"
        else:
            axes[i].text(0.5, 0.5, f'Processing failed\n({resolution})', 
                        ha='center', va='center', fontsize=14, transform=axes[i].transAxes)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
    
    # Overall title
    if date_range_str:
        fig.suptitle(f"{location}: Cumulative {data_type.title()}\n{date_range_str}", 
                     fontsize=16, fontweight='bold', y=0.995)
    else:
        fig.suptitle(f"{location}: Cumulative {data_type.title()}", 
                     fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    
    if any_data:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        print(f"  ✗ No data to plot for {location}")
    
    plt.close()
    
    return all_qc_flags

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--erosion', action='store_true')
    parser.add_argument('--deposition', action='store_true')
    parser.add_argument('--use_original', action='store_true')
    args = parser.parse_args()

    import platform
    if platform.system() == 'Darwin':
         base_dir = '/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs'
    else:
         base_dir = '/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs'

    types = []
    if args.erosion: types.append('erosion')
    if args.deposition: types.append('deposition')
    if not types: types = ['erosion']

    out_dir = os.path.join(base_dir, 'figures', 'cumulative_erosion')
    os.makedirs(out_dir, exist_ok=True)
    
    all_qc_flags = {}

    print(f"--- Cliff Face Cumulative Plotter (All Resolutions) ---")
    print(f"Resolutions: {', '.join(RESOLUTIONS)}")
    print(f"Output Dir: {out_dir}")
    
    for dtype in types:
        print(f"\n{'='*60}")
        print(f"Processing {dtype.upper()}")
        print(f"{'='*60}")
        
        for loc in TARGET_LOCATIONS:
            print(f"\n{loc}:")
            
            out_name = f"{loc}_cumulative_{dtype}_all_resolutions.png"
            output_path = os.path.join(out_dir, out_name)
            
            flags = plot_all_resolutions(loc, dtype, base_dir, not args.use_original, output_path)
            
            if flags:
                all_qc_flags[loc] = flags

    if all_qc_flags:
        print("\n" + "="*60)
        print("QC REPORT: SKIPPED FILES")
        print("="*60)
        for loc, flags in all_qc_flags.items():
            print(f"\n{loc}:")
            for f in flags:
                print(f"  {f}")

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()