#!/usr/bin/env python3
"""
cliff_top_testing.py

Visualizes cumulative erosion at 25cm resolution,
overlaid with the visually digitized cliff top cutoff.

Usage:
    python3 cliff_top_testing.py --location DelMar
    python3 cliff_top_testing.py --location SanElijo
    python3 cliff_top_testing.py --all
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform
import re
from datetime import datetime
from PIL import Image

# Import shared config from digitize_cliff_top (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from digitize_cliff_top import (
    find_image_for_location,
    HEIGHTS,
)

# Plotting Style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# --- PATHS ---
def get_base_path():
    if platform.system() == "Darwin":
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

BASE_PATH = get_base_path()

# --- GRID LOADING LOGIC ---
def get_resolution_value(res_str):
    if 'cm' in res_str: return float(res_str.replace('cm', '')) / 100.0
    elif 'm' in res_str: return float(res_str.replace('m', ''))
    return 0.1

def clean_and_snap_grid(df, resolution_val):
    cleaned_cols = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)
    try:
        col_floats = cleaned_cols.astype(float)
    except ValueError:
        return None
    scale = 1.0 / resolution_val
    new_cols = (col_floats * scale).round().astype(int)
    df.columns = new_cols
    df.index = df.index.astype(int)
    return df

def load_grid_dataframe(filepath, res_val):
    if not os.path.exists(filepath): return None
    try:
        df = pd.read_csv(filepath, index_col=0)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = clean_and_snap_grid(df, res_val)
        return df
    except Exception as e:
        print(f"Error reading {os.path.basename(filepath)}: {e}")
        return None

def parse_date_from_folder(folder_name):
    match = re.search(r'(\d{8})', folder_name)
    if match: return datetime.strptime(match.group(1), '%Y%m%d')
    match = re.search(r'(\d{4})_(\d{2})_(\d{2})', folder_name)
    if match: return datetime.strptime(f"{match.group(1)}{match.group(2)}{match.group(3)}", '%Y%m%d')
    return None

def find_grid_files(base_dir, location, resolution, data_type, use_filled=True):
    """Find grid files for a location/resolution."""
    type_dir = os.path.join(base_dir, 'results', location, data_type)
    if not os.path.isdir(type_dir):
        return []

    grid_files = []
    suffix = "_filled.csv" if use_filled else ".csv"
    patterns = [f"_ero_grid_{resolution}{suffix}", f"_dep_grid_{resolution}{suffix}", f"grid_{resolution}{suffix}"]

    for date_folder in sorted(os.listdir(type_dir)):
        folder_path = os.path.join(type_dir, date_folder)
        if not os.path.isdir(folder_path):
            continue

        res_subdir = os.path.join(folder_path, resolution)
        if os.path.isdir(res_subdir):
            search_dir = res_subdir
            files_in_folder = os.listdir(res_subdir)
        else:
            search_dir = folder_path
            files_in_folder = os.listdir(folder_path)

        grid_file = None
        for pattern in patterns:
            matches = [f for f in files_in_folder if pattern in f and f.endswith('.csv')]
            if matches:
                grid_file = os.path.join(search_dir, matches[0])
                break

        if grid_file:
            d = parse_date_from_folder(date_folder)
            if d:
                grid_files.append((d, grid_file))

    grid_files.sort(key=lambda x: x[0])
    return grid_files

def calculate_final_cumulative(grid_files, res_val):
    if not grid_files: return None
    print(f"    Summing {len(grid_files)} surveys...")
    start_path = grid_files[0][1]
    cumulative_df = load_grid_dataframe(start_path, res_val)
    if cumulative_df is None: return None

    baseline_shape = cumulative_df.shape
    cumulative_df = cumulative_df.fillna(0.0)

    for i, (date, filepath) in enumerate(grid_files[1:]):
        current_df = load_grid_dataframe(filepath, res_val)
        if current_df is None: continue
        if current_df.shape != baseline_shape: continue
        cumulative_df = cumulative_df.add(current_df.fillna(0.0), fill_value=0)

    cumulative_df.sort_index(axis=0, inplace=True)
    cumulative_df.sort_index(axis=1, inplace=True)
    return cumulative_df


# --- PLOTTING ---
def process_location(location):
    """Generate cliff top testing visualization for a single location."""
    cutoff_dir = os.path.join(BASE_PATH, "utilities", "cliff_top_cutoffs", location)
    output_fig = os.path.join(cutoff_dir, f"{location}_Visual_CliffTop_test.png")
    os.makedirs(cutoff_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"--- {location}: Plotting Visual Check ---")
    print(f"Output: {output_fig}")

    resolution = '25cm'
    res_val = get_resolution_value(resolution)

    # 1. Load cumulative erosion grid
    print(f"\nProcessing {resolution}...")
    grid_files = find_grid_files(BASE_PATH, location, resolution, 'erosion', use_filled=True)
    if not grid_files:
        print(f"    [SKIP] No grid files found for {resolution}")
        return None

    print(f"    Found {len(grid_files)} grid files")
    cumulative_df = calculate_final_cumulative(grid_files, res_val)
    if cumulative_df is None:
        print(f"    [ERROR] Failed to load grid data")
        return None

    # 2. Load cutoff CSV
    cutoff_path = os.path.join(cutoff_dir, f"{location}_Visual_CliffTop_{resolution}.csv")
    cutoff_df = None
    if os.path.exists(cutoff_path):
        print(f"    Loading cutoff: {cutoff_path}")
        cutoff_df = pd.read_csv(cutoff_path)
    else:
        print(f"    [WARNING] Cutoff file not found: {cutoff_path}")

    # 3. Compute axis extents from the grid data itself
    plot_df = cumulative_df.T
    polygon_ids = plot_df.columns.astype(float).values
    x_coords_meters = polygon_ids * res_val

    n_bins = len(plot_df.index)
    max_elevation = n_bins * res_val
    extent = [x_coords_meters.min(), x_coords_meters.max(), 0, max_elevation]

    print(f"    Grid extent: X=[{extent[0]:.1f}, {extent[1]:.1f}], Y=[0, {max_elevation:.1f}]")

    # 4. Match figure dimensions to source cliff top image
    try:
        source_path = find_image_for_location(location)
        img = Image.open(source_path)
        img_w, img_h = img.size
        dpi = 150
        fig_w = img_w / dpi
        fig_h = img_h / dpi
        print(f"    Source image: {img_w}x{img_h} px -> figure {fig_w:.1f}x{fig_h:.1f} in")
    except FileNotFoundError:
        print(f"    [WARNING] Source image not found, using default dimensions")
        fig_w, fig_h, dpi = 15, 4, 150

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # 5. Plot heatmap
    vals = plot_df.values.flatten()
    vals = vals[vals > 0]
    if len(vals) > 0:
        vmin, vmax = np.percentile(vals, [2, 98])
    else:
        vmin, vmax = 0, 1

    im = ax.imshow(plot_df.values, origin='lower', extent=extent,
                   cmap='Reds', vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Cumulative Erosion (m)')

    # 6. Overlay cutoff line
    if cutoff_df is not None:
        cutoff_df = cutoff_df.sort_values('Polygon_ID')
        cutoff_meters = cutoff_df['Polygon_ID'].values * res_val

        mask = (cutoff_meters >= x_coords_meters.min()) & (cutoff_meters <= x_coords_meters.max())
        subset_meters = cutoff_meters[mask]
        subset_z = cutoff_df['CliffTop_Z'].values[mask]

        ax.plot(subset_meters, subset_z,
                color='blue', linewidth=1.5, linestyle='--', label='Visual Cutoff')
        ax.legend(loc='upper right', fontsize='small')

    # 7. Formatting - use grid data extent, NOT HEIGHTS
    ax.set_title(f"{resolution} Resolution", fontsize=12, fontweight='bold')
    ax.set_xlabel("Alongshore Position (m)")
    ax.set_ylabel("Elevation (m)")
    ax.invert_xaxis()
    ax.set_ylim(0, max_elevation)

    fig.suptitle(f"{location}: Cumulative Erosion & Visual Cutoff Check",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(output_fig, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Done. Saved to {output_fig}")

    return output_fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize cumulative erosion with cliff top cutoffs."
    )
    parser.add_argument('--location', type=str, default=None,
                        help='Location name (e.g., DelMar, SanElijo)')
    parser.add_argument('--all', action='store_true',
                        help='Process all locations')

    args = parser.parse_args()

    if args.all:
        locations = list(HEIGHTS.keys())
        print(f"Processing all locations: {locations}")
        for loc in locations:
            try:
                process_location(loc)
            except Exception as e:
                print(f"[ERROR] {loc}: {e}")
    elif args.location:
        if args.location not in HEIGHTS:
            print(f"Unknown location '{args.location}'. Available: {list(HEIGHTS.keys())}")
            return
        process_location(args.location)
    else:
        parser.print_help()
        print(f"\nAvailable locations: {list(HEIGHTS.keys())}")


if __name__ == "__main__":
    main()
