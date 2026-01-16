#!/usr/bin/env python3
"""
cum_raw_vis.py

Visualizes cumulative erosion and deposition grids over time.

Plots cumulative change grids for each location/resolution/type combination
as a single heatmap (similar to dashboard Panel E).

Usage:
    python3 cum_raw_vis.py --location SanElijo --resolution 25cm --type erosion
    python3 cum_raw_vis.py --location SanElijo  # All resolutions, both types
    python3 cum_raw_vis.py --all  # All locations, all resolutions, both types
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from datetime import datetime

# Detect OS and set base paths
if sys.platform == 'darwin':
    BASE = '/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs'
else:
    BASE = '/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs'

RESULTS_DIR = os.path.join(BASE, 'results')
FIGURES_DIR = os.path.join(BASE, 'figures', 'raw_grid_vis')

LOCATIONS = ['DelMar', 'SanElijo', 'Solana', 'Encinitas', 'Torrey', 'Blacks']
RESOLUTIONS = ['10cm', '25cm', '1m']
TYPES = ['erosion', 'deposition']

# Font sizes
FS_TITLE = 18
FS_LABEL = 14
FS_TICK = 12


def get_resolution_value(resolution):
    """Convert resolution string to numeric value."""
    if resolution == '10cm':
        return 0.10
    elif resolution == '25cm':
        return 0.25
    elif resolution == '1m':
        return 1.0
    return 0.25


def get_custom_cmap(name, vmax=6.0):
    """Creates a custom colormap with White at 0 (matching dashboard style)."""
    base_cmap = cm.get_cmap(name, 256)
    newcolors = base_cmap(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 1])  # Force 0 to be White
    return LinearSegmentedColormap.from_list(f"White_{name}", newcolors), Normalize(vmin=0, vmax=vmax)


def clean_and_snap_grid(df, resolution_val):
    """Clean column names and snap to resolution grid (matching dashboard)."""
    cleaned_cols = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)
    try:
        col_floats = cleaned_cols.astype(float)
        scale = 1.0 / resolution_val
        new_cols = (col_floats * scale).round().astype(int)
        df.columns = new_cols
        df.index = df.index.astype(int)
        return df
    except:
        return None


def parse_date_from_folder(folder_name):
    """
    Parse start date from folder name like '20180209_to_20180216'.
    Returns datetime object for the END date (second date).
    """
    try:
        date_str = folder_name.split('_to_')[-1]
        return datetime.strptime(date_str, '%Y%m%d')
    except:
        return None


def load_grid_csv(csv_path):
    """
    Load grid CSV file.
    Returns pandas DataFrame.
    """
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path, index_col=0, na_values=['', 'nan', 'NaN', 'NULL'])
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def get_sorted_survey_pairs(location, change_type):
    """
    Get chronologically sorted list of survey pair folders.

    Returns list of tuples: (folder_name, date)
    """
    type_dir = os.path.join(RESULTS_DIR, location, change_type)

    if not os.path.isdir(type_dir):
        return []

    # Get all date folders
    date_folders = [d for d in os.listdir(type_dir)
                    if os.path.isdir(os.path.join(type_dir, d))
                    and '_to_' in d]

    # Parse dates and sort
    folder_data = []
    for folder in date_folders:
        date = parse_date_from_folder(folder)
        if date:
            folder_data.append((folder, date))

    folder_data.sort(key=lambda x: x[1])

    return [(folder, date) for folder, date in folder_data]


def compute_cumulative_grid(location, resolution, change_type):
    """
    Compute cumulative grid across all time steps.
    Matches dashboard Panel E logic.

    Returns:
        cumulative_grid: pandas DataFrame (cleaned and snapped)
        n_surveys: number of surveys processed
    """
    survey_pairs = get_sorted_survey_pairs(location, change_type)

    if not survey_pairs:
        return None, 0

    res_val = get_resolution_value(resolution)
    cumulative_grid = None
    n_surveys = 0

    prefix = 'ero' if change_type == 'erosion' else 'dep'

    for folder, date in survey_pairs:
        # Construct path to grid CSV
        grid_path = os.path.join(
            RESULTS_DIR, location, change_type, folder, resolution,
            f"{folder}_{prefix}_grid_{resolution}.csv"
        )

        df_grid = load_grid_csv(grid_path)

        if df_grid is None:
            continue

        # Clean and snap the grid (matching dashboard)
        spatial_df = clean_and_snap_grid(df_grid.copy(), res_val)

        if spatial_df is not None:
            if cumulative_grid is None:
                cumulative_grid = spatial_df.fillna(0.0)
            else:
                cumulative_grid = cumulative_grid.add(spatial_df.fillna(0.0), fill_value=0)
            n_surveys += 1

    return cumulative_grid, n_surveys


def plot_cumulative_grid(cumulative_grid, location, resolution, change_type,
                        n_surveys, output_path):
    """
    Create single heatmap visualization of cumulative grid.
    Matches dashboard Panel E style.
    """
    if cumulative_grid is None or cumulative_grid.empty:
        print(f"No data to plot for {location} {resolution} {change_type}")
        return

    res_val = get_resolution_value(resolution)

    # Transpose for plotting (matching dashboard)
    plot_df = cumulative_grid.T

    # Get spatial coordinates
    x_indices = plot_df.columns.astype(int)
    x_meters = x_indices * res_val
    y_elev = plot_df.index.astype(int)
    max_elev_m = len(y_elev) * res_val

    # Get matrix values
    matrix = plot_df.values

    # Set up extent
    extent = [x_meters.min(), x_meters.max(), 0, max_elev_m]

    # Choose colormap and vmax based on type
    if change_type == 'erosion':
        cmap_name = 'magma_r'
        vmax = 6.0
        cbar_label = 'Cumulative Erosion Depth (m)'
    else:
        cmap_name = 'viridis_r'
        vmax = 3.0
        cbar_label = 'Cumulative Deposition Depth (m)'

    # Get custom colormap with white at 0
    cmap, norm = get_custom_cmap(cmap_name, vmax=vmax)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot heatmap
    im = ax.imshow(matrix, origin='lower', extent=extent,
                   cmap=cmap, norm=norm, aspect='auto',
                   interpolation='none')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                       pad=0.08, fraction=0.05, aspect=50)
    cbar.set_label(cbar_label, fontsize=FS_LABEL, color='black', fontweight='bold')
    cbar.ax.tick_params(labelsize=FS_TICK, labelcolor='black')

    # Labels and title
    ax.set_xlabel("Alongshore Location (m)", fontsize=FS_LABEL,
                 fontweight='bold', color='black')
    ax.set_ylabel("Elevation (m)", fontsize=FS_LABEL,
                 fontweight='bold', color='black')

    title = f'{location} - Cumulative {change_type.capitalize()} - {resolution}'
    title += f'\n({n_surveys} survey pairs)'
    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', color='black', pad=20)

    # Reverse x-axis to match dashboard (cliff facing view)
    ax.set_xlim(x_meters.max(), x_meters.min())
    ax.set_ylim(0, max_elev_m)

    # Grid and ticks
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.tick_params(axis='both', labelsize=FS_TICK, labelcolor='black')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")


def process_combination(location, resolution, change_type):
    """
    Process a single location/resolution/type combination.
    """
    print(f"\nProcessing {location} - {resolution} - {change_type}...")

    # Compute cumulative grid
    cumulative_grid, n_surveys = compute_cumulative_grid(
        location, resolution, change_type
    )

    if cumulative_grid is None or n_surveys == 0:
        print(f"  No data found for {location} {resolution} {change_type}")
        return

    # Create output directory
    output_dir = os.path.join(FIGURES_DIR, location)
    os.makedirs(output_dir, exist_ok=True)

    # Create output filename
    output_filename = f"{location}_{resolution}_{change_type}_cumulative.png"
    output_path = os.path.join(output_dir, output_filename)

    # Plot
    plot_cumulative_grid(cumulative_grid, location, resolution, change_type,
                        n_surveys, output_path)

    print(f"  Processed {n_surveys} survey pairs")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize cumulative erosion/deposition grids',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 cum_raw_vis.py --location SanElijo --resolution 25cm --type erosion
  python3 cum_raw_vis.py --location SanElijo
  python3 cum_raw_vis.py --all
        """
    )

    parser.add_argument('--location', type=str, choices=LOCATIONS,
                       help='Location to process (if not --all)')
    parser.add_argument('--resolution', type=str, choices=RESOLUTIONS,
                       help='Resolution to process (default: all)')
    parser.add_argument('--type', type=str, choices=TYPES,
                       help='Type to process: erosion or deposition (default: both)')
    parser.add_argument('--all', action='store_true',
                       help='Process all locations')

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.location:
        parser.error("Must specify --location or --all")

    # Determine what to process
    locations = LOCATIONS if args.all else [args.location]
    resolutions = RESOLUTIONS if not args.resolution else [args.resolution]
    types = TYPES if not args.type else [args.type]

    # Create base output directory
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Process all combinations
    total = len(locations) * len(resolutions) * len(types)
    count = 0

    print(f"Processing {total} combinations...")
    print(f"Locations: {locations}")
    print(f"Resolutions: {resolutions}")
    print(f"Types: {types}")
    print(f"Output directory: {FIGURES_DIR}")

    for location in locations:
        for resolution in resolutions:
            for change_type in types:
                count += 1
                print(f"\n[{count}/{total}]", end=' ')
                process_combination(location, resolution, change_type)

    print(f"\n\nDone! Figures saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
