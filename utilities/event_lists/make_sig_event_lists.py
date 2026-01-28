#!/usr/bin/env python3
"""
make_sig_event_lists.py

Filters event list CSVs to keep only significant erosion events.

Significance criteria:
  - Volume > 5 m³
  - Elevation > 5 m (originated above 5m elevation)

Filters erosion/ and combined/ subdirectories only (not deposition/).
Outputs files with thresholds encoded in filename: <location>_vol_<V>_elv_<E>.csv

Optional NPZ export:
  --make-npz creates filtered 3D data cubes (alongshore × elevation × time).
  Only includes cells belonging to clusters that meet the significance criteria.
  Output: results/data_cubes/<location>_vol_<V>_elv_<E>_cube.npz

Usage:
    python3 make_sig_event_lists.py
    python3 make_sig_event_lists.py --min_volume 10 --min_elevation 3
    python3 make_sig_event_lists.py --input_dir /path/to/event_lists
    python3 make_sig_event_lists.py --make-npz
"""
import os
import platform
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_MIN_VOLUME = 5.0      # m³
DEFAULT_MIN_ELEVATION = 5.0   # m

ALL_LOCATIONS = ['DelMar', 'Solana', 'Encinitas', 'SanElijo', 'Torrey']

RESOLUTION = "25cm"
CELL_SIZE = 0.25  # meters (vertical bin size)


def get_base_dir():
    """Get base directory based on platform."""
    if platform.system() == "Darwin":
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"


# ============================================================================
# SHAPEFILE AND GRID UTILITIES
# ============================================================================

def find_shapefile(base_util, location, poly_res):
    """Locate the shapefile for a given location and polygon resolution."""
    sf_root = os.path.join(base_util, 'shape_files')
    if not os.path.isdir(sf_root):
        raise FileNotFoundError(f"Shape files directory not found: {sf_root}")

    candidates = [
        d for d in os.listdir(sf_root)
        if d.lower().startswith(location.lower())
           and 'polygon' in d.lower()
           and f'at{poly_res}'.lower() in d.lower()
           and os.path.isdir(os.path.join(sf_root, d))
    ]

    if not candidates:
        raise FileNotFoundError(f"No shapefile folder found for {location} at {poly_res}")

    fld = candidates[0]
    shp_path = os.path.join(sf_root, fld, fld + '.shp')
    if not os.path.isfile(shp_path):
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")
    return shp_path


def load_polygon_alongshore(shapefile_path):
    """Load polygon centroids and compute alongshore positions."""
    gdf = gpd.read_file(shapefile_path)
    gdf['Polygon_ID'] = gdf.index
    centroids = gdf.geometry.centroid

    coords = {int(pid): (c.x, c.y) for pid, c in zip(gdf['Polygon_ID'], centroids)}
    all_y = [c[1] for c in coords.values()]
    min_y = min(all_y)
    alongshore = {pid: coord[1] - min_y for pid, coord in coords.items()}

    sorted_positions = sorted(alongshore.values())
    if len(sorted_positions) > 1:
        spacings = np.diff(sorted_positions)
        cell_width = np.median(spacings)
    else:
        cell_width = CELL_SIZE

    return alongshore, cell_width


def parse_elevation_from_header(header):
    """Extract elevation value from header like 'M3C2_0.25m' -> 0.25"""
    try:
        parts = header.split('_')
        val_str = parts[-1].replace('m', '')
        return float(val_str)
    except Exception:
        return None


def load_grid_csv(filepath):
    """Load a grid CSV file."""
    if not os.path.exists(filepath):
        return None, None, None

    df = pd.read_csv(filepath, index_col=0, header=0,
                     na_values=['', 'nan', 'NaN', 'NULL'])
    header_labels = df.columns.tolist()
    row_labels = [int(r) for r in df.index.tolist()]
    return header_labels, row_labels, df.values


# ============================================================================
# FILTERING
# ============================================================================

def filter_significant_events(df, min_volume, min_elevation):
    """
    Filter events to keep only significant ones.

    Args:
        df: DataFrame with event data
        min_volume: Minimum volume threshold (m³)
        min_elevation: Minimum elevation threshold (m)

    Returns:
        Filtered DataFrame
    """
    mask = (df['volume'] > min_volume) & (df['elevation'] > min_elevation)
    return df[mask].copy()


def make_output_filename(location, min_volume, min_elevation):
    """
    Generate output filename with threshold values encoded.

    Args:
        location: Location name (e.g., 'SanElijo')
        min_volume: Minimum volume threshold
        min_elevation: Minimum elevation threshold

    Returns:
        Filename like 'SanElijo_vol_5_elv_5.csv'
    """
    # Format as integers if whole numbers, otherwise keep decimals
    vol_str = str(int(min_volume)) if min_volume == int(min_volume) else str(min_volume)
    elv_str = str(int(min_elevation)) if min_elevation == int(min_elevation) else str(min_elevation)
    return f"{location}_vol_{vol_str}_elv_{elv_str}.csv"


def process_directory(subdir_path, min_volume, min_elevation):
    """
    Process all *_events.csv files in a directory.

    Args:
        subdir_path: Path to erosion/ or combined/ subdirectory
        min_volume: Minimum volume threshold
        min_elevation: Minimum elevation threshold

    Returns:
        Tuple of (total_events, significant_events) counts
    """
    total = 0
    significant = 0

    if not os.path.isdir(subdir_path):
        return total, significant

    for filename in os.listdir(subdir_path):
        if filename.endswith('_events.csv') and '_vol_' not in filename:
            input_path = os.path.join(subdir_path, filename)
            location = filename.replace('_events.csv', '')
            output_filename = make_output_filename(location, min_volume, min_elevation)
            output_path = os.path.join(subdir_path, output_filename)

            try:
                df = pd.read_csv(input_path)
                total += len(df)

                df_sig = filter_significant_events(df, min_volume, min_elevation)
                significant += len(df_sig)

                df_sig.to_csv(output_path, index=False)

                print(f"    {location}: {len(df)} -> {len(df_sig)} events ({output_filename})")

            except Exception as e:
                print(f"    [ERROR] {filename}: {e}")

    return total, significant


# ============================================================================
# NPZ CUBE BUILDING WITH SIGNIFICANCE FILTERING
# ============================================================================

def compute_cluster_stats(grid, clusters, elevations, cell_width):
    """
    Compute volume and elevation centroid for each cluster.

    Returns:
        dict: cluster_id -> {'volume': float, 'elevation': float}
    """
    cell_area = cell_width * CELL_SIZE

    valid_mask = ~np.isnan(clusters) & (clusters != 0)
    unique_clusters = np.unique(clusters[valid_mask])

    stats = {}
    for cluster_id in unique_clusters:
        cluster_id = int(cluster_id)
        cluster_mask = clusters == cluster_id

        m3c2_values = grid[cluster_mask]
        valid_values = m3c2_values[~np.isnan(m3c2_values)]

        if len(valid_values) == 0:
            continue

        volume = np.sum(np.abs(valid_values)) * cell_area

        # Get elevation for each cell in cluster
        row_indices, col_indices = np.where(cluster_mask)
        cell_elevations = elevations[col_indices]

        # Volume-weighted elevation centroid
        weights = np.abs(valid_values)
        valid_elev_mask = ~np.isnan(m3c2_values)
        if np.sum(weights) > 0:
            elevation_centroid = np.average(
                cell_elevations[valid_elev_mask],
                weights=weights
            )
        else:
            elevation_centroid = np.mean(cell_elevations)

        stats[cluster_id] = {
            'volume': volume,
            'elevation': elevation_centroid
        }

    return stats


def build_filtered_data_cube(location, base_dir, event_type, min_volume, min_elevation):
    """
    Build a 3D data cube with significance filtering applied.

    Only includes cells belonging to clusters that meet:
      - volume > min_volume
      - elevation > min_elevation

    Returns:
        cube: 3D numpy array (alongshore, elevation, time)
        alongshore_m: 1D array of alongshore positions
        elevation_m: 1D array of elevation values
        dates: List of mid-dates
        date_strings: List of date folder names
        filter_stats: Dict with filtering statistics
    """
    results_dir = os.path.join(base_dir, 'results', location)
    util_dir = os.path.join(base_dir, 'utilities')

    type_dir = os.path.join(results_dir, event_type)
    if not os.path.isdir(type_dir):
        print(f"  [WARNING] {event_type}/ directory not found")
        return None, None, None, None, None, None

    # Load shapefile for alongshore positions
    try:
        shp_path = find_shapefile(util_dir, location, RESOLUTION)
        alongshore_positions, cell_width = load_polygon_alongshore(shp_path)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return None, None, None, None, None, None

    # Collect all date folders
    dates = sorted([d for d in os.listdir(type_dir)
                    if os.path.isdir(os.path.join(type_dir, d))])

    if not dates:
        print(f"  [WARNING] No date folders found in {event_type}/")
        return None, None, None, None, None, None

    # First pass: collect all unique polygon IDs and elevations
    all_polygon_ids = set()
    all_elevations = set()
    valid_dates = []
    data_cache = {}  # Store grid, cluster, and filter info

    total_clusters = 0
    sig_clusters = 0

    for date_str in dates:
        res_folder = os.path.join(type_dir, date_str, RESOLUTION)
        if not os.path.isdir(res_folder):
            continue

        # Find grid and cluster files
        grid_file = None
        cluster_file = None

        for f in os.listdir(res_folder):
            if f.endswith(f'_grid_{RESOLUTION}_filled.csv'):
                grid_file = os.path.join(res_folder, f)
            elif f.endswith(f'_clusters_{RESOLUTION}_filled.csv'):
                cluster_file = os.path.join(res_folder, f)

        if not grid_file or not cluster_file:
            continue

        headers_g, rows_g, grid = load_grid_csv(grid_file)
        headers_c, rows_c, clusters = load_grid_csv(cluster_file)

        if grid is None or clusters is None:
            continue

        # Parse elevations
        elevations = np.array([parse_elevation_from_header(h) for h in headers_g])

        # Compute cluster stats and identify significant clusters
        cluster_stats = compute_cluster_stats(grid, clusters, elevations, cell_width)
        total_clusters += len(cluster_stats)

        sig_cluster_ids = set()
        for cid, stats in cluster_stats.items():
            if stats['volume'] > min_volume and stats['elevation'] > min_elevation:
                sig_cluster_ids.add(cid)
        sig_clusters += len(sig_cluster_ids)

        all_polygon_ids.update(rows_g)
        all_elevations.update([e for e in elevations if e is not None])
        valid_dates.append(date_str)
        data_cache[date_str] = {
            'headers': headers_g,
            'rows': rows_g,
            'grid': grid,
            'clusters': clusters,
            'sig_cluster_ids': sig_cluster_ids
        }

    if not valid_dates:
        print(f"  [WARNING] No valid grid files found for {event_type}")
        return None, None, None, None, None, None

    # Create sorted axes
    sorted_polygon_ids = sorted(all_polygon_ids)
    sorted_elevations = sorted(all_elevations)

    alongshore_m = np.array([alongshore_positions.get(pid, np.nan)
                             for pid in sorted_polygon_ids])
    elevation_m = np.array(sorted_elevations)

    # Create index mappings
    pid_to_idx = {pid: i for i, pid in enumerate(sorted_polygon_ids)}
    elv_to_idx = {elv: i for i, elv in enumerate(sorted_elevations)}

    # Parse mid-dates
    mid_dates = []
    for date_str in valid_dates:
        parts = date_str.split('_to_')
        if len(parts) == 2:
            try:
                start = datetime.strptime(parts[0], '%Y%m%d').date()
                end = datetime.strptime(parts[1], '%Y%m%d').date()
                mid = start + (end - start) / 2
                mid_dates.append(mid)
            except ValueError:
                mid_dates.append(None)
        else:
            mid_dates.append(None)

    # Build 3D cube with filtering
    n_alongshore = len(sorted_polygon_ids)
    n_elevation = len(sorted_elevations)
    n_time = len(valid_dates)

    cube = np.full((n_alongshore, n_elevation, n_time), np.nan, dtype=np.float32)

    for t_idx, date_str in enumerate(valid_dates):
        cache = data_cache[date_str]
        headers = cache['headers']
        rows = cache['rows']
        grid = cache['grid']
        clusters = cache['clusters']
        sig_cluster_ids = cache['sig_cluster_ids']

        for row_idx, pid in enumerate(rows):
            if pid not in pid_to_idx:
                continue
            as_idx = pid_to_idx[pid]

            for col_idx, header in enumerate(headers):
                elv = parse_elevation_from_header(header)
                if elv is None or elv not in elv_to_idx:
                    continue
                el_idx = elv_to_idx[elv]

                # Only include if cell belongs to a significant cluster
                cluster_id = clusters[row_idx, col_idx]
                if np.isnan(cluster_id) or int(cluster_id) not in sig_cluster_ids:
                    continue

                value = grid[row_idx, col_idx]
                if not np.isnan(value):
                    cube[as_idx, el_idx, t_idx] = value

    filter_stats = {
        'total_clusters': total_clusters,
        'significant_clusters': sig_clusters
    }

    return cube, alongshore_m, elevation_m, mid_dates, valid_dates, filter_stats


def make_npz_filename(location, min_volume, min_elevation):
    """Generate NPZ filename with threshold values encoded."""
    vol_str = str(int(min_volume)) if min_volume == int(min_volume) else str(min_volume)
    elv_str = str(int(min_elevation)) if min_elevation == int(min_elevation) else str(min_elevation)
    return f"{location}_vol_{vol_str}_elv_{elv_str}_cube.npz"


def save_filtered_cubes(input_dir, npz_output_dir, base_dir, min_volume, min_elevation):
    """
    Build and save filtered 3D data cubes for all locations.

    Args:
        input_dir: Directory containing event_lists (for finding locations)
        npz_output_dir: Output directory for NPZ files
        base_dir: Base directory for source data
        min_volume: Minimum volume threshold
        min_elevation: Minimum elevation threshold
    """
    # Find all locations from erosion directory
    erosion_dir = os.path.join(input_dir, 'erosion')
    if not os.path.isdir(erosion_dir):
        print(f"[ERROR] erosion/ directory not found: {erosion_dir}")
        return

    # Get locations from existing event files
    locations = []
    for f in os.listdir(erosion_dir):
        if f.endswith('_events.csv'):
            loc = f.replace('_events.csv', '')
            if loc in ALL_LOCATIONS:
                locations.append(loc)

    if not locations:
        print("[WARNING] No location event files found")
        return

    print(f"\nBuilding filtered data cubes for: {', '.join(locations)}")

    for location in sorted(locations):
        print(f"\n--- {location} ---")

        # Build erosion cube
        result = build_filtered_data_cube(
            location, base_dir, 'erosion',
            min_volume, min_elevation
        )

        if result[0] is not None:
            cube, alongshore, elevation, dates, date_strs, stats = result

            data = {
                'erosion': cube,
                'alongshore_m': alongshore,
                'elevation_m': elevation,
                'dates': np.array([d.toordinal() if d else 0 for d in dates]),
                'date_strings': np.array(date_strs, dtype=object),
                'min_volume': min_volume,
                'min_elevation': min_elevation
            }

            npz_filename = make_npz_filename(location, min_volume, min_elevation)
            npz_path = os.path.join(npz_output_dir, npz_filename)
            np.savez_compressed(npz_path, **data)

            print(f"  Erosion cube: {cube.shape} (alongshore, elevation, time)")
            print(f"    Clusters: {stats['significant_clusters']}/{stats['total_clusters']} significant")
            print(f"    Non-NaN values: {np.sum(~np.isnan(cube)):,}")
            print(f"  Saved: {npz_path}")
        else:
            print(f"  [WARNING] Could not build erosion cube for {location}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter event lists to keep only significant erosion events'
    )
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory (default: results/event_lists/)')
    parser.add_argument('--min_volume', type=float, default=DEFAULT_MIN_VOLUME,
                        help=f'Minimum volume threshold in m³ (default: {DEFAULT_MIN_VOLUME})')
    parser.add_argument('--min_elevation', type=float, default=DEFAULT_MIN_ELEVATION,
                        help=f'Minimum elevation threshold in m (default: {DEFAULT_MIN_ELEVATION})')
    parser.add_argument('--make-npz', action='store_true',
                        help='Create filtered 3D data cube NPZ files')
    args = parser.parse_args()

    # Set input directory (default: repo's results/event_lists/)
    if args.input_dir:
        input_dir = args.input_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(script_dir))  # up from utilities/event_lists/
        input_dir = os.path.join(repo_root, 'results', 'event_lists')

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory not found: {input_dir}")
        return 1

    # Set up NPZ output directory (separate from CSV output)
    npz_output_dir = os.path.join(os.path.dirname(input_dir), 'data_cubes')

    print(f"\n{'='*60}")
    print(f"SIGNIFICANT EVENT FILTERING")
    print(f"Input: {input_dir}")
    print(f"Criteria:")
    print(f"  - Volume > {args.min_volume} m³")
    print(f"  - Elevation > {args.min_elevation} m")
    if args.make_npz:
        print(f"Mode: NPZ cube generation")
        print(f"NPZ Output: {npz_output_dir}")
        os.makedirs(npz_output_dir, exist_ok=True)
    print(f"{'='*60}\n")

    if args.make_npz:
        # Build and save filtered data cubes
        base_dir = get_base_dir()
        save_filtered_cubes(input_dir, npz_output_dir, base_dir, args.min_volume, args.min_elevation)
    else:
        # Filter CSV files
        total_all = 0
        sig_all = 0

        # Process erosion/
        erosion_dir = os.path.join(input_dir, 'erosion')
        if os.path.isdir(erosion_dir):
            print("Processing erosion/...")
            total, sig = process_directory(erosion_dir, args.min_volume, args.min_elevation)
            total_all += total
            sig_all += sig
        else:
            print(f"[WARNING] erosion/ directory not found: {erosion_dir}")

        # Process combined/
        combined_dir = os.path.join(input_dir, 'combined')
        if os.path.isdir(combined_dir):
            print("\nProcessing combined/...")
            total, sig = process_directory(combined_dir, args.min_volume, args.min_elevation)
            total_all += total
            sig_all += sig
        else:
            print(f"[WARNING] combined/ directory not found: {combined_dir}")

        print(f"\n{'='*60}")
        print(f"Summary: {sig_all} significant events from {total_all} total")
        print(f"{'='*60}\n")

    return 0


if __name__ == '__main__':
    exit(main())
