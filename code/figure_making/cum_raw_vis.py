#!/usr/bin/env python3
"""
cum_raw_vis.py

Visualizes cumulative erosion and deposition grids over time.

Plots cumulative change grids for each location/resolution/type combination
and saves to figures/raw_grid_vis/<Location>/.

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
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
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
    Returns numpy array of grid values (rows=polygons, cols=elevations).
    """
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path, index_col=0, na_values=['', 'nan', 'NaN', 'NULL'])
        # Replace NaN with 0 for cumulative calculation
        return df.fillna(0).values
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def get_sorted_survey_pairs(location, change_type):
    """
    Get chronologically sorted list of survey pair folders.

    Returns list of tuples: (folder_name, date, grid_csv_path)
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


def compute_cumulative_grids(location, resolution, change_type):
    """
    Compute cumulative grid values over time.

    Returns:
        dates: list of datetime objects
        cumulative_grids: list of numpy arrays (cumulative sum)
        folder_names: list of folder names
    """
    survey_pairs = get_sorted_survey_pairs(location, change_type)

    if not survey_pairs:
        return None, None, None

    dates = []
    cumulative_grids = []
    folder_names = []
    cumulative_sum = None

    prefix = 'ero' if change_type == 'erosion' else 'dep'

    for folder, date in survey_pairs:
        # Construct path to grid CSV
        grid_path = os.path.join(
            RESULTS_DIR, location, change_type, folder, resolution,
            f"{folder}_{prefix}_grid_{resolution}.csv"
        )

        grid = load_grid_csv(grid_path)

        if grid is None:
            continue

        # Initialize cumulative sum on first valid grid
        if cumulative_sum is None:
            cumulative_sum = np.zeros_like(grid)

        # Add current grid to cumulative (using absolute values for magnitude)
        cumulative_sum = cumulative_sum + np.abs(grid)

        dates.append(date)
        cumulative_grids.append(cumulative_sum.copy())
        folder_names.append(folder)

    if not dates:
        return None, None, None

    return dates, cumulative_grids, folder_names


def plot_cumulative_grid(dates, cumulative_grids, folder_names, location,
                         resolution, change_type, output_path):
    """
    Create visualization of cumulative grid evolution.

    Creates a multi-panel figure showing grid state at key time points.
    """
    n_grids = len(cumulative_grids)

    if n_grids == 0:
        print(f"No data to plot for {location} {resolution} {change_type}")
        return

    # Determine how many subplots to show (max 12, evenly spaced)
    n_plots = min(12, n_grids)
    indices = np.linspace(0, n_grids - 1, n_plots, dtype=int)

    # Determine grid layout
    if n_plots <= 3:
        nrows, ncols = 1, n_plots
    elif n_plots <= 6:
        nrows, ncols = 2, 3
    elif n_plots <= 9:
        nrows, ncols = 3, 3
    else:
        nrows, ncols = 3, 4

    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Set colormap
    cmap = 'Reds' if change_type == 'erosion' else 'Blues'

    # Get global vmin/vmax for consistent colorbar
    all_data = np.concatenate([g.flatten() for g in cumulative_grids])
    vmin, vmax = 0, np.nanpercentile(all_data[all_data > 0], 99.5) if np.any(all_data > 0) else 1

    # Plot each selected time point
    for i, idx in enumerate(indices):
        ax = axes[i]
        grid = cumulative_grids[idx]
        date = dates[idx]

        im = ax.imshow(grid, aspect='auto', cmap=cmap,
                      vmin=vmin, vmax=vmax, interpolation='nearest')

        ax.set_title(f"{date.strftime('%Y-%m-%d')}\n({folder_names[idx]})",
                    fontsize=9)
        ax.set_xlabel('Elevation Bin', fontsize=8)
        ax.set_ylabel('Polygon ID', fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal',
                       pad=0.05, fraction=0.03, aspect=50)
    cbar.set_label(f'Cumulative {change_type.capitalize()} (m)', fontsize=10)

    # Overall title
    fig.suptitle(f'{location} - Cumulative {change_type.capitalize()} - {resolution}',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def process_combination(location, resolution, change_type):
    """
    Process a single location/resolution/type combination.
    """
    print(f"\nProcessing {location} - {resolution} - {change_type}...")

    # Compute cumulative grids
    dates, cumulative_grids, folder_names = compute_cumulative_grids(
        location, resolution, change_type
    )

    if dates is None or len(dates) == 0:
        print(f"  No data found for {location} {resolution} {change_type}")
        return

    # Create output directory
    output_dir = os.path.join(FIGURES_DIR, location)
    os.makedirs(output_dir, exist_ok=True)

    # Create output filename
    output_filename = f"{location}_{resolution}_{change_type}_cumulative.png"
    output_path = os.path.join(output_dir, output_filename)

    # Plot
    plot_cumulative_grid(dates, cumulative_grids, folder_names,
                        location, resolution, change_type, output_path)

    print(f"  Processed {len(dates)} survey pairs")


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
