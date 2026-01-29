#!/usr/bin/env python3
"""
plot_cube_cumulative.py

Plots cumulative erosion and deposition from NPZ data cubes.
Used to verify that the NPZ cubes were built correctly by comparing
against the CSV-based cumulative plots.

Usage:
    python3 plot_cube_cumulative.py --location SanElijo
    python3 plot_cube_cumulative.py --all
    python3 plot_cube_cumulative.py --location DelMar --save
"""

import argparse
import os
import platform
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# --- Configuration ---

CELL_SIZE = 0.25  # meters (vertical bin size, matches make_event_lists.py)

LOCATIONS_ALL = ['DelMar', 'Solana', 'SanElijo', 'Encinitas', 'Torrey']

COLORS = {
    'erosion': '#D62828',      # Red
    'deposition': '#2E86AB',   # Blue
}


def get_base_dir():
    """Get base directory based on platform."""
    if platform.system() == 'Darwin':
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"


def get_cube_path(location):
    """Get path to NPZ cube file, checking both server and local repo."""
    base_dir = get_base_dir()

    # Check server location first
    server_path = os.path.join(base_dir, 'results', 'data_cubes', f'{location}_cube.npz')
    if os.path.exists(server_path):
        return server_path

    # Check local repo results folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    local_path = os.path.join(repo_root, 'results', 'data_cubes', f'{location}_cube.npz')
    if os.path.exists(local_path):
        return local_path

    return None


def load_cube(npz_path):
    """
    Load NPZ cube and return data dictionary.

    Returns:
        dict with keys: erosion, deposition, alongshore_m, elevation_m, dates, date_strings
    """
    data = np.load(npz_path, allow_pickle=True)

    result = {
        'alongshore_m': data['alongshore_m'],
        'elevation_m': data['elevation_m'],
        'dates': data['dates'],
        'date_strings': data['date_strings'],
    }

    if 'erosion' in data.files:
        result['erosion'] = data['erosion']
    if 'deposition' in data.files:
        result['deposition'] = data['deposition']

    data.close()
    return result


def compute_cell_width(alongshore_m):
    """Compute alongshore cell width from position array."""
    valid = alongshore_m[~np.isnan(alongshore_m)]
    if len(valid) > 1:
        sorted_pos = np.sort(valid)
        spacings = np.diff(sorted_pos)
        return np.median(spacings)
    return CELL_SIZE  # fallback


def compute_cumulative_volume(cube_3d, cell_area):
    """
    Compute cumulative volume over time from 3D cube.

    Args:
        cube_3d: 3D array (alongshore, elevation, time)
        cell_area: Area of each cell in m^2

    Returns:
        1D array of cumulative volumes for each time step
    """
    n_time = cube_3d.shape[2]
    volumes = np.zeros(n_time)

    for t in range(n_time):
        slice_2d = cube_3d[:, :, t]
        # Sum absolute values (M3C2 distances) * cell_area = volume
        valid = slice_2d[~np.isnan(slice_2d)]
        volumes[t] = np.sum(np.abs(valid)) * cell_area

    return np.cumsum(volumes)


def ordinal_to_date(ordinal_arr):
    """Convert array of ordinal integers to datetime objects."""
    dates = []
    for o in ordinal_arr:
        if o > 0:
            dates.append(datetime.fromordinal(int(o)))
        else:
            dates.append(None)
    return dates


def plot_location(location, cube_data, out_dir=None, save=False):
    """
    Create cumulative plot for a single location.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Compute cell area
    cell_width = compute_cell_width(cube_data['alongshore_m'])
    cell_area = cell_width * CELL_SIZE

    # Convert dates
    dates = ordinal_to_date(cube_data['dates'])
    valid_date_mask = [d is not None for d in dates]
    plot_dates = [d for d in dates if d is not None]

    has_data = False

    # Plot erosion
    if 'erosion' in cube_data:
        erosion = cube_data['erosion']
        cum_ero = compute_cumulative_volume(erosion, cell_area)
        cum_ero_valid = cum_ero[valid_date_mask]

        ax.plot(plot_dates, cum_ero_valid,
                label=f'Erosion (from NPZ)',
                color=COLORS['erosion'],
                linewidth=2,
                marker='o',
                markersize=3,
                alpha=0.8)
        has_data = True

        # Print summary stats
        print(f"  Erosion:")
        print(f"    Shape: {erosion.shape}")
        print(f"    Non-NaN cells: {np.sum(~np.isnan(erosion)):,}")
        print(f"    Final cumulative volume: {cum_ero_valid[-1]:,.1f} m^3")

    # Plot deposition
    if 'deposition' in cube_data:
        deposition = cube_data['deposition']
        cum_dep = compute_cumulative_volume(deposition, cell_area)
        cum_dep_valid = cum_dep[valid_date_mask]

        ax.plot(plot_dates, cum_dep_valid,
                label=f'Deposition (from NPZ)',
                color=COLORS['deposition'],
                linewidth=2,
                marker='s',
                markersize=3,
                alpha=0.8)
        has_data = True

        print(f"  Deposition:")
        print(f"    Shape: {deposition.shape}")
        print(f"    Non-NaN cells: {np.sum(~np.isnan(deposition)):,}")
        print(f"    Final cumulative volume: {cum_dep_valid[-1]:,.1f} m^3")

    if not has_data:
        print(f"  [WARNING] No erosion or deposition data found")
        plt.close()
        return

    # Formatting
    ax.set_title(f"{location}: Cumulative Volume from NPZ Data Cube",
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel(r"Cumulative Volume ($m^3$)", fontsize=13)
    ax.set_xlabel("Date", fontsize=13)

    ax.grid(True, which='major', linestyle='--', alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)
    ax.minorticks_on()

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()

    ax.legend(fontsize=11, loc='upper left')

    # Add info text
    info_text = (
        f"Cell size: {cell_width:.2f}m x {CELL_SIZE}m\n"
        f"Time steps: {len(plot_dates)}\n"
        f"Alongshore bins: {len(cube_data['alongshore_m'])}\n"
        f"Elevation bins: {len(cube_data['elevation_m'])}"
    )
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
            fontsize=10, family='monospace', va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save and out_dir:
        out_path = os.path.join(out_dir, f'{location}_cube_cumulative.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {out_path}")
        plt.close()
    else:
        plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot cumulative erosion/deposition from NPZ data cubes"
    )
    parser.add_argument('--location', '-l', help="Location name (e.g., SanElijo)")
    parser.add_argument('--all', action='store_true', help="Process all locations")
    parser.add_argument('--save', '-s', action='store_true', help="Save figures instead of displaying")
    parser.add_argument('--out-dir', '-o', help="Output directory for saved figures")
    args = parser.parse_args()

    if not args.location and not args.all:
        parser.error("Must specify --location or --all")

    if args.all and args.location:
        parser.error("Cannot specify both --all and --location")

    locations = LOCATIONS_ALL if args.all else [args.location]

    # Set up output directory
    if args.save:
        if args.out_dir:
            out_dir = args.out_dir
        else:
            base_dir = get_base_dir()
            out_dir = os.path.join(base_dir, 'figures', 'cube_validation')
        os.makedirs(out_dir, exist_ok=True)
        print(f"Output directory: {out_dir}")
    else:
        out_dir = None

    print(f"\n{'='*60}")
    print("NPZ CUBE CUMULATIVE VOLUME VERIFICATION")
    print(f"{'='*60}\n")

    for location in locations:
        print(f"\n--- {location} ---")

        cube_path = get_cube_path(location)
        if not cube_path:
            print(f"  [ERROR] No NPZ cube found for {location}")
            continue

        print(f"  Loading: {cube_path}")

        try:
            cube_data = load_cube(cube_path)
        except Exception as e:
            print(f"  [ERROR] Failed to load cube: {e}")
            continue

        plot_location(location, cube_data, out_dir, save=args.save)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
