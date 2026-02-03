#!/usr/bin/env python3
"""
Diagnostic script to plot the largest filled erosion events.
Helps verify that hole filling is working correctly.

Usage:
    python3 diagnostic_plot_filled.py
    python3 diagnostic_plot_filled.py --location SanElijo --top 10
"""
import os
import platform
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Configuration
RESOLUTION = "25cm"
CELL_SIZE = 0.25


def get_base_dir():
    if platform.system() == "Darwin":
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"


def get_repo_root():
    """Get the repository root directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(script_dir))


def load_grid_csv(filepath):
    """Load a grid CSV file."""
    if not os.path.exists(filepath):
        return None, None, None
    df = pd.read_csv(filepath, index_col=0, header=0,
                     na_values=['', 'nan', 'NaN', 'NULL'])
    header_labels = df.columns.tolist()
    row_labels = [int(r) for r in df.index.tolist()]
    return header_labels, row_labels, df.values


def parse_elevation_from_header(header):
    """Extract elevation value from header like 'M3C2_0.25m' -> 0.25"""
    try:
        parts = header.split('_')
        val_str = parts[-1].replace('m', '')
        return float(val_str)
    except:
        return None


def load_top_events_from_csv(location, top_n, base_dir):
    """Load top N events from the event list CSV, sorted by volume."""
    repo_root = get_repo_root()
    csv_path = os.path.join(repo_root, 'results', 'event_lists', 'erosion', f'{location}_events.csv')

    if not os.path.exists(csv_path):
        print(f"Event CSV not found: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    df = df.sort_values('volume', ascending=False).head(top_n)

    events = []
    for _, row in df.iterrows():
        # Build date string from start_date and end_date
        start_str = str(row['start_date']).replace('-', '')
        end_str = str(row['end_date']).replace('-', '')
        date_str = f"{start_str}_to_{end_str}"

        # Build path to grid files
        res_folder = os.path.join(base_dir, 'results', location, 'erosion', date_str, RESOLUTION)

        grid_file_filled = None
        grid_file_original = None
        if os.path.isdir(res_folder):
            for f in os.listdir(res_folder):
                if f.endswith(f'_grid_{RESOLUTION}_filled.csv'):
                    grid_file_filled = os.path.join(res_folder, f)
                elif f.endswith(f'_grid_{RESOLUTION}.csv') and '_filled' not in f:
                    grid_file_original = os.path.join(res_folder, f)

        events.append({
            'date': date_str,
            'volume': row['volume'],
            'elevation': row['elevation'],
            'width': row['width'],
            'height': row['height'],
            'alongshore_start_m': row['alongshore_start_m'],
            'alongshore_end_m': row['alongshore_end_m'],
            'alongshore_centroid_m': row['alongshore_centroid_m'],
            'grid_file_filled': grid_file_filled,
            'grid_file_original': grid_file_original
        })

    return events


def load_alongshore_mapping(location, base_dir):
    """Load shapefile and compute alongshore positions for each polygon."""
    util_dir = os.path.join(base_dir, 'utilities', 'shape_files')

    # Find shapefile folder
    candidates = [d for d in os.listdir(util_dir)
                  if d.lower().startswith(location.lower())
                  and 'polygon' in d.lower()
                  and f'at{RESOLUTION}'.lower() in d.lower()]

    if not candidates:
        return None

    shp_folder = candidates[0]
    shp_path = os.path.join(util_dir, shp_folder, shp_folder + '.shp')

    if not os.path.exists(shp_path):
        return None

    gdf = gpd.read_file(shp_path)
    gdf['Polygon_ID'] = gdf.index
    centroids = gdf.geometry.centroid

    # Get Y coordinates (alongshore)
    coords = {int(pid): c.y for pid, c in zip(gdf['Polygon_ID'], centroids)}

    # Compute alongshore distance from min Y
    min_y = min(coords.values())
    alongshore = {pid: y - min_y for pid, y in coords.items()}

    return alongshore


def get_event_bounds(event, rows, elevations, alongshore_map, x_pad_m=10, y_pad_m=3):
    """
    Get grid index bounds from event's physical coordinates.

    Uses event's alongshore_start/end_m and elevation/height to find
    the corresponding row and column indices in the grid.
    """
    # Map polygon IDs to alongshore positions
    row_alongshore = np.array([alongshore_map.get(pid, np.nan) for pid in rows])

    # Find row indices that fall within event's alongshore range
    x_min = event['alongshore_start_m'] - x_pad_m
    x_max = event['alongshore_end_m'] + x_pad_m

    row_mask = (row_alongshore >= x_min) & (row_alongshore <= x_max)
    if not np.any(row_mask):
        # Fallback: find closest rows to centroid
        centroid = event['alongshore_centroid_m']
        distances = np.abs(row_alongshore - centroid)
        closest_idx = np.argmin(distances)
        row_min = max(0, closest_idx - 50)
        row_max = min(len(rows) - 1, closest_idx + 50)
    else:
        row_indices = np.where(row_mask)[0]
        row_min, row_max = row_indices.min(), row_indices.max()

    # Find column indices for elevation range
    elev_centroid = event['elevation']
    height = event['height']
    y_min = max(0, elev_centroid - height / 2 - y_pad_m)
    y_max = elev_centroid + height / 2 + y_pad_m

    col_mask = (elevations >= y_min) & (elevations <= y_max)
    if not np.any(col_mask):
        # Fallback
        col_min, col_max = 0, len(elevations) - 1
    else:
        col_indices = np.where(col_mask)[0]
        col_min, col_max = col_indices.min(), col_indices.max()

    return row_min, row_max, col_min, col_max


def plot_grid(grid_file, ax, title, bounds=None):
    """Plot a single grid on the given axes, zoomed to bounds."""
    if grid_file is None:
        ax.text(0.5, 0.5, "Grid file not found",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return None, None, None, None

    headers, rows, grid = load_grid_csv(grid_file)

    if grid is None:
        ax.text(0.5, 0.5, "Failed to load", ha='center', va='center')
        ax.set_title(title)
        return None, None, None, None

    # Parse elevations
    elevations = np.array([parse_elevation_from_header(h) for h in headers])

    # Replace 0 with NaN for visualization
    plot_data = grid.copy().astype(float)
    plot_data[plot_data == 0] = np.nan

    # Get bounds if not provided
    if bounds is None:
        bounds = get_data_bounds(plot_data, padding_frac=0.15)

    if bounds is None:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        ax.set_title(title)
        return None, None, None, None

    row_min, row_max, col_min, col_max = bounds

    # Crop to bounds
    plot_cropped = plot_data[row_min:row_max+1, col_min:col_max+1]
    rows_cropped = [rows[i] for i in range(row_min, row_max+1)]
    elevations_cropped = elevations[col_min:col_max+1]

    # Count non-empty cells in full grid
    n_cells = np.sum(~np.isnan(plot_data))

    # Plot
    im = ax.imshow(plot_cropped.T, aspect='auto', origin='lower',
                   cmap='magma_r', vmin=0, vmax=2,
                   interpolation='nearest')

    # Labels
    n_rows_c, n_cols_c = plot_cropped.shape

    # Y-axis: elevations
    n_yticks = min(6, n_cols_c)
    if n_cols_c > 1:
        ytick_idx = np.linspace(0, n_cols_c - 1, n_yticks, dtype=int)
        ax.set_yticks(ytick_idx)
        ax.set_yticklabels([f'{elevations_cropped[i]:.1f}' for i in ytick_idx])

    # X-axis: polygon IDs
    n_xticks = min(6, n_rows_c)
    if n_rows_c > 1:
        xtick_idx = np.linspace(0, n_rows_c - 1, n_xticks, dtype=int)
        ax.set_xticks(xtick_idx)
        ax.set_xticklabels([str(rows_cropped[i]) for i in xtick_idx], rotation=45)

    ax.set_xlabel("Polygon ID (alongshore)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"{title}\n({n_cells} cells)")

    return im, elevations, rows, bounds


def plot_event_comparison(event, output_path, alongshore_map):
    """Create a 2-panel comparison figure for an event (original vs filled)."""
    # Load filled grid to get dimensions and compute bounds
    bounds = None

    if event['grid_file_filled'] is not None:
        headers, rows, grid_filled = load_grid_csv(event['grid_file_filled'])
        if grid_filled is not None and alongshore_map is not None:
            elevations = np.array([parse_elevation_from_header(h) for h in headers])
            bounds = get_event_bounds(event, rows, elevations, alongshore_map,
                                      x_pad_m=10, y_pad_m=3)

    # Create figure with space for colorbar at bottom
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    # Plot original (non-filled) with shared bounds
    im1, _, _, _ = plot_grid(event['grid_file_original'], axes[0], "Original (before filling)", bounds=bounds)

    # Plot filled with shared bounds
    im2, _, _, _ = plot_grid(event['grid_file_filled'], axes[1], "Filled (after filling)", bounds=bounds)

    # Add horizontal colorbar at bottom
    im = im2 if im2 is not None else im1
    if im is not None:
        cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='|M3C2| (m)')

    fig.suptitle(f"{event['date']}\nVolume={event['volume']:.2f} m³, Elevation={event['elevation']:.1f}m",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.subplots_adjust(bottom=0.18, top=0.88, wspace=0.25)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Plot largest filled erosion events')
    parser.add_argument('--location', default='DelMar', help='Location name')
    parser.add_argument('--top', type=int, default=5, help='Number of events to plot')
    parser.add_argument('--output', default=None, help='Output file path')
    args = parser.parse_args()

    base_dir = get_base_dir()

    print(f"Loading top {args.top} events from {args.location} event CSV...")
    events = load_top_events_from_csv(args.location, args.top, base_dir)

    if not events:
        print("No events found!")
        return

    print(f"\nTop {args.top} by volume:")
    for i, e in enumerate(events):
        print(f"  {i+1}. {e['date']}: {e['volume']:.2f} m³ (elev={e['elevation']:.1f}m, width={e['width']:.1f}m)")

    # Filter out events with missing filled grid files
    valid_events = [e for e in events if e['grid_file_filled'] is not None]
    if len(valid_events) < len(events):
        print(f"\nWarning: {len(events) - len(valid_events)} events have missing filled grid files")

    if not valid_events:
        print("No valid events with grid files found!")
        return

    # Load alongshore mapping for proper zoom
    print(f"\nLoading shapefile for alongshore mapping...")
    alongshore_map = load_alongshore_mapping(args.location, base_dir)
    if alongshore_map is None:
        print("Warning: Could not load shapefile - zoom may not work correctly")

    # Create individual comparison figures
    repo_root = get_repo_root()
    output_dir = os.path.join(repo_root, 'figures', 'testing')

    print(f"\nGenerating comparison figures...")
    for i, event in enumerate(valid_events):
        output_path = os.path.join(output_dir, f'{args.location}_event{i+1}_{event["date"]}_comparison.png')
        plot_event_comparison(event, output_path, alongshore_map)
        print(f"  {i+1}. Saved: {os.path.basename(output_path)}")

    print(f"\nDone! {len(valid_events)} figures saved to {output_dir}")


if __name__ == '__main__':
    main()
