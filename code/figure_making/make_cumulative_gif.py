#!/usr/bin/env python3
"""
make_cumulative_grid_gif.py

Creates cumulative erosion/deposition GIF animations from time-series grid data.
Each frame adds the current survey's grid to the cumulative total.
Now generates both HIGH-RES and LOW-RES versions.

Usage:
    python3 make_cumulative_grid_gif.py <location> --resolution <10cm|25cm|1m> [options]

Example:
    python3 make_cumulative_grid_gif.py SanElijo --resolution 10cm --erosion
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from datetime import datetime
import re
from pathlib import Path

# Publication-quality style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


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


def get_resolution_value(res_str):
    """Convert resolution string to numeric value in meters."""
    if 'cm' in res_str:
        return float(res_str.replace('cm', '')) / 100.0
    elif 'm' in res_str:
        return float(res_str.replace('m', ''))
    return 0.1  # Default


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


def load_grid_dataframe(filepath, res_val):
    """Load grid CSV and return cleaned DataFrame."""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, index_col=0)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = clean_and_snap_grid(df, res_val)
        return df
    except Exception as e:
        print(f"    Error reading {os.path.basename(filepath)}: {e}")
        return None


def parse_date_from_folder(folder_name):
    """
    Extract date from folder name in format YYYYMMDD or YYYY_MM_DD.
    Returns datetime object or None if parsing fails.
    """
    # Try YYYYMMDD format
    match = re.search(r'(\d{8})', folder_name)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d')
        except:
            pass
    
    # Try YYYY_MM_DD format
    match = re.search(r'(\d{4})_(\d{2})_(\d{2})', folder_name)
    if match:
        try:
            return datetime.strptime(f"{match.group(1)}{match.group(2)}{match.group(3)}", '%Y%m%d')
        except:
            pass
    
    return None


def find_grid_files(base_dir, location, resolution, data_type='erosion', use_filled=True):
    """
    Find all grid CSV files for the given location, resolution, and type.
    Returns list of tuples: (date, date_folder, filepath)
    """
    type_dir = os.path.join(base_dir, 'results', location, data_type)
    
    if not os.path.isdir(type_dir):
        return []
    
    grid_files = []
    
    # Normalize resolution for file searching (1m -> 100cm)
    file_resolution = normalize_resolution_for_files(resolution)
    
    # Patterns to search for (in priority order)
    if data_type == 'erosion':
        if use_filled:
            patterns = [
                f"_ero_grid_{file_resolution}_filled.csv",
                f"grid_{file_resolution}_filled.csv",
                f"_ero_grid_{file_resolution}_cleaned.csv",
                f"grid_{file_resolution}_cleaned.csv",
            ]
        else:
            patterns = [
                f"_ero_grid_{file_resolution}_cleaned.csv",
                f"grid_{file_resolution}_cleaned.csv",
                f"_ero_grid_{file_resolution}.csv",
                f"grid_{file_resolution}.csv",
            ]
    else:  # deposition
        if use_filled:
            patterns = [
                f"_dep_grid_{file_resolution}_filled.csv",
                f"dep_grid_{file_resolution}_filled.csv",
                f"_dep_grid_{file_resolution}_cleaned.csv",
                f"dep_grid_{file_resolution}_cleaned.csv",
            ]
        else:
            patterns = [
                f"_dep_grid_{file_resolution}_cleaned.csv",
                f"dep_grid_{file_resolution}_cleaned.csv",
                f"_dep_grid_{file_resolution}.csv",
                f"dep_grid_{file_resolution}.csv",
            ]
    
    for date_folder in os.listdir(type_dir):
        folder_path = os.path.join(type_dir, date_folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Try to find the first matching file pattern
        grid_file = None
        files_in_folder = os.listdir(folder_path)
        
        for pattern in patterns:
            matching_files = [f for f in files_in_folder if pattern in f and f.endswith('.csv')]
            if matching_files:
                grid_file = os.path.join(folder_path, matching_files[0])
                break
        
        if grid_file:
            date = parse_date_from_folder(date_folder)
            if date:
                grid_files.append((date, date_folder, grid_file))
    
    # Sort by date
    grid_files.sort(key=lambda x: x[0])
    
    return grid_files


def create_cumulative_gif(grid_files, output_path_base, data_type='erosion', 
                         fps=2, figsize=(15, 6), resolution='10cm'):
    """
    Create cumulative GIF animation from grid files.
    Creates BOTH high-res and low-res versions.
    Uses the same plotting logic as plot_cumulative_cliff_face_rotated.py
    
    Current settings:
    - HIGH-RES: 150 DPI (2250 x 900 pixels) - publication quality
    - LOW-RES:  75 DPI (1125 x 450 pixels) - web/preview quality
    """
    if not grid_files:
        print(f"No grid files found to create GIF")
        return False
    
    print(f"\nCreating cumulative {data_type} GIF...")
    print(f"Found {len(grid_files)} surveys")
    
    # Get resolution value
    res_val = get_resolution_value(resolution)
    
    # Load all grids as DataFrames
    grids = []
    dates = []
    date_labels = []
    
    for date, date_folder, filepath in grid_files:
        print(f"  Loading: {date_folder}")
        df = load_grid_dataframe(filepath, res_val)
        
        if df is None:
            print(f"    Warning: Could not load {filepath}")
            continue
        
        grids.append(df)
        dates.append(date)
        date_labels.append(date.strftime('%Y-%m-%d'))
    
    if not grids:
        print("No valid grids loaded")
        return False
    
    # Calculate cumulative grids
    cumulative_grids = []
    cumulative_sum = grids[0].fillna(0.0).copy()
    cumulative_grids.append(cumulative_sum.copy())
    
    for i in range(1, len(grids)):
        grid_filled = grids[i].fillna(0.0)
        cumulative_sum = cumulative_sum.add(grid_filled, fill_value=0)
        cumulative_grids.append(cumulative_sum.copy())
    
    # Calculate volumes (CORRECTED)
    cell_area = res_val * res_val
    for i, cum_grid in enumerate(cumulative_grids):
        volume = cum_grid.sum().sum() * cell_area
        print(f"  {date_labels[i]}: Cumulative volume = {volume:.2f} m³")
    
    # Determine colorbar limits using percentiles
    all_values = pd.concat(cumulative_grids).values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    all_values = all_values[all_values != 0]
    
    if len(all_values) == 0:
        print("No valid data values found")
        return False
    
    vmin, vmax = np.percentile(all_values, [2, 98])
    
    # Adjust colormap based on data type
    if data_type == 'erosion':
        cmap = 'Reds'
        label = 'Cumulative (m)'
    else:
        cmap = 'Blues'
        label = 'Cumulative (m)'
    
    # ========================================================================
    # CREATE BOTH HIGH-RES AND LOW-RES VERSIONS
    # ========================================================================
    
    versions = [
        {'suffix': '_highres', 'dpi': 150, 'label': 'HIGH-RES'},
        {'suffix': '_preview', 'dpi': 75, 'label': 'LOW-RES'}
    ]
    
    for version in versions:
        print(f"\n  Generating {version['label']} version (DPI={version['dpi']})...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # TRANSPOSE THE DATAFRAME (like in the reference code)
        plot_df = cumulative_grids[0].T
        
        # X Axis (Polygon IDs - columns after transpose)
        x_coords = plot_df.columns.astype(float).values
        
        # Y Axis (Elevation in meters - index after transpose)
        y_coords = plot_df.index.astype(float).values * res_val
        
        # Extent [Xmin, Xmax, Ymin, Ymax]
        extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
        
        # Initial plot
        im = ax.imshow(plot_df.values, origin='lower', extent=extent, 
                       cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', 
                       interpolation='none')
        
        cbar = plt.colorbar(im, ax=ax, label=label, fraction=0.02, pad=0.02)
        
        # Title with date
        title = ax.set_title(f'{resolution} Resolution', 
                             fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Polygon ID (Alongshore Index)', fontsize=12)
        ax.set_ylabel('Elevation (m)', fontsize=12)
        ax.invert_xaxis()  # Higher polygon IDs on the left
        
        # Add volume and date text
        volume_text = ax.text(0.01, 0.95, '', transform=ax.transAxes,
                             verticalalignment='top', fontsize=11,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Animation update function
        def update(frame):
            # Transpose current cumulative grid
            plot_df_frame = cumulative_grids[frame].T
            im.set_data(plot_df_frame.values)
            
            # Calculate volume
            volume = cumulative_grids[frame].sum().sum() * cell_area
            
            # Update text
            volume_text.set_text(f'Volume: {volume:.1f} m³\n{date_labels[frame]}')
            
            return im, title, volume_text
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(grids), 
                            interval=1000/fps, blit=True, repeat=True)
        
        # Generate output path
        output_path = output_path_base.replace('.gif', f'{version["suffix"]}.gif')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=version['dpi'])
        
        plt.close(fig)
        
        # Get file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        pixel_width = int(figsize[0] * version['dpi'])
        pixel_height = int(figsize[1] * version['dpi'])
        
        print(f"    ✓ Saved: {os.path.basename(output_path)}")
        print(f"      Resolution: {pixel_width} x {pixel_height} pixels")
        print(f"      File size: {file_size_mb:.2f} MB")
    
    print(f"\n  ✓ Both versions saved successfully!")
    print(f"    Total frames: {len(grids)}")
    print(f"    FPS: {fps}")
    print(f"    Duration: {len(grids)/fps:.1f} seconds")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Create cumulative erosion/deposition GIF from time-series grids (HIGH-RES + LOW-RES)'
    )
    parser.add_argument('location', help='Site location (e.g., SanElijo, Encinitas)')
    parser.add_argument('--resolution', choices=['10cm', '25cm', '1m'], default='10cm',
                       help='Grid resolution (default: 10cm)')
    parser.add_argument('--erosion', action='store_true',
                       help='Process erosion data (default if neither specified)')
    parser.add_argument('--deposition', action='store_true',
                       help='Process deposition data')
    parser.add_argument('--use_original', action='store_true',
                       help='Use original grids instead of cleaned/filled versions')
    parser.add_argument('--fps', type=float, default=2.0,
                       help='Frames per second for GIF (default: 2.0)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[15, 6],
                       help='Figure size in inches (default: 15 6)')
    
    args = parser.parse_args()
    
    # Determine base directory (Mac vs Linux)
    import platform
    if platform.system() == 'Darwin':
        base_dir = '/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs'
    else:
        base_dir = '/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs'
    
    # Determine what to process
    process_erosion = args.erosion or not (args.erosion or args.deposition)
    process_deposition = args.deposition or not (args.erosion or args.deposition)
    
    print(f"\n{'='*80}")
    print(f"CUMULATIVE GRID GIF GENERATOR (HIGH-RES + LOW-RES)")
    print(f"{'='*80}")
    print(f"Location:     {args.location}")
    print(f"Resolution:   {args.resolution}")
    print(f"Base Dir:     {base_dir}")
    print(f"Use Original: {args.use_original}")
    print(f"FPS:          {args.fps}")
    print(f"Output formats:")
    print(f"  HIGH-RES:   150 DPI ({int(args.figsize[0]*150)} x {int(args.figsize[1]*150)} pixels)")
    print(f"  LOW-RES:    75 DPI ({int(args.figsize[0]*75)} x {int(args.figsize[1]*75)} pixels)")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_base = os.path.join(base_dir, 'figures', 'erosion_gifs', args.location)
    
    success_count = 0
    
    # Process erosion
    if process_erosion:
        print("Processing EROSION data...")
        grid_files = find_grid_files(base_dir, args.location, args.resolution, 
                                     'erosion', use_filled=not args.use_original)
        
        if grid_files:
            output_path = os.path.join(output_base, 
                                      f'cumulative_erosion_{args.resolution}.gif')
            if create_cumulative_gif(grid_files, output_path, 'erosion', 
                                   args.fps, tuple(args.figsize), args.resolution):
                success_count += 2  # Count both versions
        else:
            print("  No erosion grid files found")
    
    # Process deposition
    if process_deposition:
        print("\nProcessing DEPOSITION data...")
        grid_files = find_grid_files(base_dir, args.location, args.resolution,
                                     'deposition', use_filled=not args.use_original)
        
        if grid_files:
            output_path = os.path.join(output_base,
                                      f'cumulative_deposition_{args.resolution}.gif')
            if create_cumulative_gif(grid_files, output_path, 'deposition',
                                   args.fps, tuple(args.figsize), args.resolution):
                success_count += 2  # Count both versions
        else:
            print("  No deposition grid files found")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully created {success_count} GIF file(s)")
    print(f"  ({success_count//2} datasets × 2 versions each)")
    print(f"Output directory: {output_base}")
    print(f"\nFile naming:")
    print(f"  *_highres.gif  - High resolution (150 DPI) for publication")
    print(f"  *_preview.gif  - Low resolution (75 DPI) for quick viewing")
    print(f"{'='*80}\n")
    

if __name__ == '__main__':
    main()