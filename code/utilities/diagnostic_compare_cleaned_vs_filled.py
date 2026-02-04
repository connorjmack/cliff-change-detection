#!/usr/bin/env python3
"""
diagnostic_compare_cleaned_vs_filled.py

Compares cleaned (unfilled) vs filled grids for the top N largest erosion events.
Creates side-by-side figures to diagnose hole filling artifacts.

Reads directly from server grid CSV files:
- *_grid_25cm.csv (unfilled - step 7 output)
- *_grid_25cm_filled.csv (filled - step 8 output)

Usage:
    python3 diagnostic_compare_cleaned_vs_filled.py --location DelMar --top 5
    python3 diagnostic_compare_cleaned_vs_filled.py --all --top 3
"""

import os
import sys
import platform
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colormaps

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# === Configuration ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
EVENT_LISTS_DIR = os.path.join(LOCAL_RESULTS_DIR, 'event_lists', 'erosion')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'figures', 'testing')

# Server paths
if platform.system() == "Darwin":
    SERVER_BASE = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
else:
    SERVER_BASE = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

RESOLUTION = "25cm"
ALL_LOCATIONS = ['DelMar', 'Solana', 'Encinitas', 'SanElijo', 'Torrey']


def get_server_results_dir():
    """Get server results directory and check if mounted."""
    server_results = os.path.join(SERVER_BASE, 'results')
    if not os.path.isdir(server_results):
        return None
    return server_results


def find_shapefile(location: str) -> str:
    """Locate shapefile for a location."""
    util_dir = os.path.join(SERVER_BASE, 'utilities', 'shape_files')
    if not os.path.isdir(util_dir):
        return None

    candidates = [
        d for d in os.listdir(util_dir)
        if d.lower().startswith(location.lower())
           and 'polygon' in d.lower()
           and f'at{RESOLUTION}'.lower() in d.lower()
           and os.path.isdir(os.path.join(util_dir, d))
    ]

    if not candidates:
        return None

    fld = candidates[0]
    shp_path = os.path.join(util_dir, fld, fld + '.shp')
    return shp_path if os.path.isfile(shp_path) else None


def load_polygon_alongshore(shapefile_path: str) -> dict:
    """Load polygon centroids and compute alongshore positions."""
    gdf = gpd.read_file(shapefile_path)
    gdf['Polygon_ID'] = gdf.index
    centroids = gdf.geometry.centroid

    coords = {int(pid): (c.x, c.y) for pid, c in zip(gdf['Polygon_ID'], centroids)}

    # Use minimum Y as reference
    all_y = [c[1] for c in coords.values()]
    min_y = min(all_y)

    alongshore = {pid: coord[1] - min_y for pid, coord in coords.items()}
    return alongshore


def parse_elevation_from_header(header: str) -> float:
    """Extract elevation from header like 'M3C2_0.25m' -> 0.25"""
    try:
        parts = header.split('_')
        val_str = parts[-1].replace('m', '')
        return float(val_str)
    except Exception:
        return None


def load_grid_csv(filepath: str) -> tuple:
    """Load grid CSV file, returns (grid_array, row_ids, elevations)."""
    if not os.path.exists(filepath):
        return None, None, None

    df = pd.read_csv(filepath, index_col=0, header=0,
                     na_values=['', 'nan', 'NaN', 'NULL'])

    row_ids = [int(r) for r in df.index.tolist()]
    headers = df.columns.tolist()
    elevations = np.array([parse_elevation_from_header(h) for h in headers])

    return df.values, row_ids, elevations


def event_dates_to_folder(start_date: str, end_date: str) -> str:
    """Convert event dates to grid folder name format."""
    start = pd.to_datetime(start_date).strftime('%Y%m%d')
    end = pd.to_datetime(end_date).strftime('%Y%m%d')
    return f"{start}_to_{end}"


def find_grid_files(server_results: str, location: str, date_folder: str) -> tuple:
    """
    Find the unfilled and filled grid CSV files for a specific event.

    Returns:
        (unfilled_path, filled_path) - either can be None if not found
    """
    res_folder = os.path.join(server_results, location, 'erosion',
                              date_folder, RESOLUTION)

    if not os.path.isdir(res_folder):
        return None, None

    unfilled_path = None
    filled_path = None

    for f in os.listdir(res_folder):
        if f.endswith(f'_grid_{RESOLUTION}.csv') and '_filled' not in f:
            unfilled_path = os.path.join(res_folder, f)
        elif f.endswith(f'_grid_{RESOLUTION}_filled.csv'):
            filled_path = os.path.join(res_folder, f)

    return unfilled_path, filled_path


def grids_to_2d_arrays(grid_data, row_ids, elevations, alongshore_positions):
    """
    Convert grid data to 2D arrays indexed by (alongshore, elevation).

    Returns:
        grid_2d: 2D array (n_alongshore, n_elevation)
        alongshore_m: 1D array of alongshore positions (sorted)
        elevation_m: 1D array of elevation values
    """
    # Map row IDs to alongshore positions
    alongshore_vals = [alongshore_positions.get(pid, np.nan) for pid in row_ids]

    # Create DataFrame for easier reindexing
    df = pd.DataFrame(grid_data, index=alongshore_vals, columns=elevations)

    # Sort by alongshore position
    df = df.sort_index()

    alongshore_m = df.index.values
    elevation_m = elevations

    return df.values, alongshore_m, elevation_m


def get_zoom_extent(event: pd.Series, x_pad_m: float = 15,
                    y_pad_bottom_m: float = 8, y_pad_top_m: float = 3) -> dict:
    """Calculate zoom bounds from event coordinates."""
    x_min = event['alongshore_start_m']
    x_max = event['alongshore_end_m']

    elev_centroid = event['elevation']
    height = event['height']

    y_min = max(0, elev_centroid - height / 2 - y_pad_bottom_m)
    y_max = elev_centroid + height / 2 + y_pad_top_m

    return {
        'x_min': x_min - x_pad_m,
        'x_max': x_max + x_pad_m,
        'y_min': y_min,
        'y_max': y_max
    }


def get_diverging_cmap(vmax: float = 5.0):
    """Create a diverging colormap for erosion visualization."""
    cmap = colormaps.get_cmap('RdBu_r').resampled(256)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    return cmap, norm


def plot_comparison(unfilled_slice, filled_slice, alongshore, elevation,
                    event, location, event_idx, output_path):
    """
    Create side-by-side comparison of unfilled vs filled grids.

    Args:
        unfilled_slice: 2D array (alongshore, elevation) - cleaned but not filled
        filled_slice: 2D array (alongshore, elevation) - after hole filling
        alongshore: 1D array of alongshore positions
        elevation: 1D array of elevation values
        event: Event row from events DataFrame
        location: Location name
        event_idx: Event rank (1-indexed)
        output_path: Path to save figure
    """
    # Get zoom extent
    extent = get_zoom_extent(event)

    # Find indices for zoom
    x_mask = (alongshore >= extent['x_min']) & (alongshore <= extent['x_max'])
    y_mask = (elevation >= extent['y_min']) & (elevation <= extent['y_max'])

    if not np.any(x_mask) or not np.any(y_mask):
        print(f"  Warning: No data in zoom extent for event {event_idx}")
        return False

    x_indices = np.where(x_mask)[0]
    y_indices = np.where(y_mask)[0]
    x_idx_min, x_idx_max = x_indices.min(), x_indices.max() + 1
    y_idx_min, y_idx_max = y_indices.min(), y_indices.max() + 1

    # Extract zoomed regions
    unfilled_zoom = unfilled_slice[x_idx_min:x_idx_max, y_idx_min:y_idx_max].T
    filled_zoom = filled_slice[x_idx_min:x_idx_max, y_idx_min:y_idx_max].T
    alongshore_zoom = alongshore[x_idx_min:x_idx_max]
    elevation_zoom = elevation[y_idx_min:y_idx_max]

    # Compute difference (filled - unfilled)
    diff_zoom = filled_zoom - unfilled_zoom

    # Replace NaN with 0 for display
    unfilled_zoom = np.nan_to_num(unfilled_zoom, nan=0.0)
    filled_zoom = np.nan_to_num(filled_zoom, nan=0.0)
    diff_zoom = np.nan_to_num(diff_zoom, nan=0.0)

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    cmap, norm = get_diverging_cmap(vmax=5.0)

    # Panel 1: Unfilled (Cleaned)
    ax1 = axes[0]
    im1 = ax1.imshow(unfilled_zoom, origin='lower', aspect='auto',
                     interpolation='nearest', cmap=cmap, norm=norm)
    ax1.set_title('CLEANED (Unfilled)', fontsize=12, fontweight='bold')

    # Panel 2: Filled
    ax2 = axes[1]
    im2 = ax2.imshow(filled_zoom, origin='lower', aspect='auto',
                     interpolation='nearest', cmap=cmap, norm=norm)
    ax2.set_title('FILLED', fontsize=12, fontweight='bold')

    # Panel 3: Difference (what was added by filling)
    ax3 = axes[2]
    diff_cmap = colormaps.get_cmap('Greens').resampled(256)
    diff_norm = Normalize(vmin=0, vmax=2.0)
    im3 = ax3.imshow(np.abs(diff_zoom), origin='lower', aspect='auto',
                     interpolation='nearest', cmap=diff_cmap, norm=diff_norm)
    ax3.set_title('DIFFERENCE (|filled - cleaned|)', fontsize=12, fontweight='bold')

    # Add axis labels and crosshairs to all panels
    for ax in axes:
        n_y, n_x = unfilled_zoom.shape

        # X-axis labels
        n_xticks = min(6, n_x) if n_x > 1 else 1
        if n_x > 1:
            xtick_idx = np.linspace(0, n_x - 1, n_xticks, dtype=int)
            ax.set_xticks(xtick_idx)
            ax.set_xticklabels([f'{alongshore_zoom[i]:.0f}' for i in xtick_idx])

        # Y-axis labels
        n_yticks = min(6, n_y) if n_y > 1 else 1
        if n_y > 1:
            ytick_idx = np.linspace(0, n_y - 1, n_yticks, dtype=int)
            ax.set_yticks(ytick_idx)
            ax.set_yticklabels([f'{elevation_zoom[i]:.1f}' for i in ytick_idx])

        # Reverse x-axis (cliff-facing view)
        ax.invert_xaxis()

        # Add crosshairs at event centroid
        if len(alongshore_zoom) > 0 and len(elevation_zoom) > 0:
            x_cross = np.argmin(np.abs(alongshore_zoom - event['alongshore_centroid_m']))
            y_cross = np.argmin(np.abs(elevation_zoom - event['elevation']))
            ax.axhline(y_cross, color='#666666', linestyle='--', linewidth=1.0, alpha=0.5)
            ax.axvline(x_cross, color='#666666', linestyle='--', linewidth=1.0, alpha=0.5)

        ax.set_xlabel("Alongshore (m)", fontsize=10)
        ax.set_ylabel("Elevation (m)", fontsize=10)

    # Add colorbars
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='M3C2 (m)')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='M3C2 (m)')
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='|Diff| (m)')

    # Overall title
    title = (f"{location} - Event #{event_idx} | "
             f"Vol={event['volume']:.2f} m³, Elev={event['elevation']:.1f} m\n"
             f"{event['start_date']} to {event['end_date']}")
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Print stats
    unfilled_nonzero = np.sum(unfilled_zoom != 0)
    filled_nonzero = np.sum(filled_zoom != 0)
    diff_nonzero = np.sum(diff_zoom != 0)

    print(f"  Saved: {os.path.basename(output_path)}")
    print(f"    Cleaned cells: {unfilled_nonzero}, Filled cells: {filled_nonzero}, "
          f"Diff cells: {diff_nonzero} ({diff_nonzero / max(1, unfilled_nonzero) * 100:.1f}% added)")

    return True


def process_location(location: str, top_n: int = 5):
    """Process a single location and create comparison figures."""
    print(f"\n{'='*60}")
    print(f"Processing: {location}")
    print(f"{'='*60}")

    # Check server connectivity
    server_results = get_server_results_dir()
    if server_results is None:
        print(f"  ERROR: Server not mounted at {SERVER_BASE}")
        print(f"  Please mount the server and try again.")
        return

    print(f"  Server: {server_results}")

    # Load shapefile for alongshore positions
    shp_path = find_shapefile(location)
    if shp_path is None:
        print(f"  ERROR: Shapefile not found for {location}")
        return

    alongshore_positions = load_polygon_alongshore(shp_path)
    print(f"  Loaded {len(alongshore_positions)} polygon positions")

    # Load event list
    event_csv = os.path.join(EVENT_LISTS_DIR, f"{location}_events.csv")
    if not os.path.exists(event_csv):
        print(f"  Error: Event list not found: {event_csv}")
        return

    events_df = pd.read_csv(event_csv)
    events_df = events_df.sort_values('volume', ascending=False).reset_index(drop=True)
    print(f"  Loaded {len(events_df)} events")

    # Process top N events
    n_events = min(top_n, len(events_df))
    print(f"  Processing top {n_events} events by volume...")

    for i in range(n_events):
        event = events_df.iloc[i]
        date_folder = event_dates_to_folder(event['start_date'], event['end_date'])

        print(f"\n  Event {i+1}: {date_folder} (Vol={event['volume']:.2f} m³)")

        # Find grid files
        unfilled_path, filled_path = find_grid_files(server_results, location, date_folder)

        if unfilled_path is None:
            print(f"    WARNING: Unfilled grid not found for {date_folder}")
            continue
        if filled_path is None:
            print(f"    WARNING: Filled grid not found for {date_folder}")
            continue

        print(f"    Unfilled: {os.path.basename(unfilled_path)}")
        print(f"    Filled: {os.path.basename(filled_path)}")

        # Load grids
        unfilled_grid, unfilled_rows, unfilled_elevs = load_grid_csv(unfilled_path)
        filled_grid, filled_rows, filled_elevs = load_grid_csv(filled_path)

        if unfilled_grid is None or filled_grid is None:
            print(f"    WARNING: Could not load grids")
            continue

        # Convert to 2D arrays with consistent indexing
        unfilled_2d, alongshore, elevation = grids_to_2d_arrays(
            unfilled_grid, unfilled_rows, unfilled_elevs, alongshore_positions)
        filled_2d, _, _ = grids_to_2d_arrays(
            filled_grid, filled_rows, filled_elevs, alongshore_positions)

        # Create comparison figure
        output_path = os.path.join(
            OUTPUT_DIR, f"{location}_event{i+1}_cleaned_vs_filled.png")

        plot_comparison(
            unfilled_2d, filled_2d, alongshore, elevation,
            event, location, i+1, output_path
        )


def main():
    parser = argparse.ArgumentParser(
        description='Compare cleaned vs filled grids for top erosion events')
    parser.add_argument('--location', type=str,
                        help='Location to process (e.g., DelMar, SanElijo)')
    parser.add_argument('--all', action='store_true',
                        help='Process all locations')
    parser.add_argument('--top', type=int, default=5,
                        help='Number of top events to compare (default: 5)')
    args = parser.parse_args()

    if not args.all and not args.location:
        parser.error("Must specify either --location or --all")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.all:
        locations = ALL_LOCATIONS
    else:
        locations = [args.location]

    for location in locations:
        process_location(location, args.top)

    print(f"\n{'='*60}")
    print(f"Done! Figures saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
