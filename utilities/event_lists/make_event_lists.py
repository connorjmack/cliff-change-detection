#!/usr/bin/env python3
"""
make_event_lists.py

Generates event list CSVs from 25cm gridded results.
Each row represents a single erosion or deposition cluster event.

Output columns match the legacy format:
  - mid_date, start_date, end_date: Temporal info from survey pair
  - volume: Total volume change (m³)
  - elevation: Volume-weighted elevation centroid (m)
  - alongshore_centroid_m: Alongshore position of centroid (m)
  - alongshore_start_m, alongshore_end_m: Alongshore extent
  - mop_centroid: MOP coordinate of centroid (e.g., 567.005)
  - mop_start, mop_end: MOP coordinate extent
  - width: Alongshore extent (m)
  - height: Vertical extent (m)
  - vol_unc: Volume uncertainty (m³)
  - month: Calendar month

Optional NPZ export:
  --make-npz additionally creates a 3D data cube (alongshore × elevation × time) from filled grids.
  Output: results/data_cubes/<location>_cube.npz with keys:
    - 'erosion': 3D array (alongshore, elevation, time) of M3C2 values
    - 'deposition': 3D array (alongshore, elevation, time) of M3C2 values
    - 'alongshore_m': 1D array of alongshore positions (meters)
    - 'elevation_m': 1D array of elevation values (meters)
    - 'dates': 1D array of mid-dates (as ordinal integers for numpy compatibility)
    - 'date_strings': 1D array of date strings (YYYYMMDD_to_YYYYMMDD)

Usage:
    python3 make_event_lists.py SanElijo              # CSVs only
    python3 make_event_lists.py --all                 # CSVs for all locations
    python3 make_event_lists.py SanElijo --erosion    # Erosion CSVs only
    python3 make_event_lists.py SanElijo --deposition # Deposition CSVs only
    python3 make_event_lists.py --all --make-npz      # CSVs + NPZ cubes for all locations
"""
import os
import platform
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

RESOLUTION = "25cm"
CELL_SIZE = 0.25  # meters (vertical bin size)

ALL_LOCATIONS = ['DelMar', 'Solana', 'Encinitas', 'SanElijo', 'Torrey']


def get_base_dir():
    """Get base directory based on platform."""
    if platform.system() == "Darwin":
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"


# ============================================================================
# SHAPEFILE AND POLYGON UTILITIES
# ============================================================================

def find_shapefile(base_util, location, poly_res):
    """
    Locate the shapefile for a given location and polygon resolution.
    """
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
        available = [d for d in os.listdir(sf_root) if os.path.isdir(os.path.join(sf_root, d))]
        raise FileNotFoundError(
            f"No shapefile folder found for location='{location}' resolution='{poly_res}'.\n"
            f"Searched in: {sf_root}\n"
            f"Available folders: {available}"
        )

    fld = candidates[0]
    shp_path = os.path.join(sf_root, fld, fld + '.shp')
    if not os.path.isfile(shp_path):
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")
    return shp_path


def load_polygon_alongshore(shapefile_path):
    """
    Load polygon centroids and compute alongshore positions.

    Returns:
        dict: Polygon_ID -> alongshore_position_m
        dict: Polygon_ID -> MOP_ID (if available, else None)
        float: Alongshore cell width (spacing between adjacent polygons)
    """
    gdf = gpd.read_file(shapefile_path)
    gdf['Polygon_ID'] = gdf.index
    centroids = gdf.geometry.centroid

    # Get UTM coordinates
    coords = {int(pid): (c.x, c.y) for pid, c in zip(gdf['Polygon_ID'], centroids)}

    # Compute alongshore distance from reference point
    # Use minimum Y coordinate as reference (assuming coast runs roughly N-S)
    all_y = [c[1] for c in coords.values()]
    min_y = min(all_y)

    alongshore = {pid: coord[1] - min_y for pid, coord in coords.items()}

    # Load MOP_ID if available
    mop_coords = {}
    if 'MOP_ID' in gdf.columns:
        for pid, mop_id in zip(gdf['Polygon_ID'], gdf['MOP_ID']):
            try:
                mop_coords[int(pid)] = float(mop_id)
            except (ValueError, TypeError):
                mop_coords[int(pid)] = np.nan
    else:
        # MOP_ID not available
        mop_coords = {int(pid): np.nan for pid in gdf['Polygon_ID']}

    # Estimate cell width from polygon spacing
    sorted_positions = sorted(alongshore.values())
    if len(sorted_positions) > 1:
        spacings = np.diff(sorted_positions)
        cell_width = np.median(spacings)
    else:
        cell_width = CELL_SIZE  # fallback

    return alongshore, mop_coords, cell_width


def parse_elevation_from_header(header):
    """Extract elevation value from header like 'M3C2_0.25m' -> 0.25"""
    try:
        parts = header.split('_')
        val_str = parts[-1].replace('m', '')
        return float(val_str)
    except Exception:
        return None


# ============================================================================
# CSV LOADING
# ============================================================================

def load_grid_csv(filepath):
    """Load a grid CSV file."""
    if not os.path.exists(filepath):
        return None, None, None

    df = pd.read_csv(filepath, index_col=0, header=0,
                     na_values=['', 'nan', 'NaN', 'NULL'])
    header_labels = df.columns.tolist()
    row_labels = [int(r) for r in df.index.tolist()]
    return header_labels, row_labels, df.values


def load_stats_npz(filepath):
    """Load stats NPZ file for uncertainty."""
    if not os.path.exists(filepath):
        return None

    data = np.load(filepath)
    return {
        'uncertainty': data['uncertainty'],
        'counts': data['counts'],
        'variance': data['variance'],
        'polygon_ids': data['polygon_ids'],
        'elevation_labels': data['elevation_labels']
    }


# ============================================================================
# EVENT EXTRACTION
# ============================================================================

def extract_events_from_grids(grid_csv_path, cluster_csv_path, stats_npz_path,
                               alongshore_positions, mop_positions, cell_width,
                               date_str, event_type):
    """
    Extract individual cluster events from grid data.

    Args:
        grid_csv_path: Path to *_grid_25cm_filled.csv
        cluster_csv_path: Path to *_clusters_25cm_filled.csv
        stats_npz_path: Path to *_stats_25cm.npz
        alongshore_positions: Dict mapping Polygon_ID -> alongshore_m
        mop_positions: Dict mapping Polygon_ID -> MOP coordinate (float)
        cell_width: Alongshore width of each polygon cell (m)
        date_str: Survey pair date string (DATE1_to_DATE2)
        event_type: 'erosion' or 'deposition'

    Returns:
        List of event dictionaries
    """
    # Load data
    header_g, rows_g, grid = load_grid_csv(grid_csv_path)
    header_c, rows_c, clusters = load_grid_csv(cluster_csv_path)
    stats = load_stats_npz(stats_npz_path)

    if grid is None or clusters is None:
        return []

    # Parse elevations from headers
    elevations = np.array([parse_elevation_from_header(h) for h in header_g])

    # Parse dates from date_str (format: YYYYMMDD_to_YYYYMMDD)
    parts = date_str.split('_to_')
    if len(parts) != 2:
        print(f"  [WARNING] Cannot parse date string: {date_str}")
        return []

    try:
        start_date = datetime.strptime(parts[0], '%Y%m%d').date()
        end_date = datetime.strptime(parts[1], '%Y%m%d').date()
        mid_date = start_date + (end_date - start_date) / 2
    except ValueError as e:
        print(f"  [WARNING] Date parsing error for {date_str}: {e}")
        return []

    # Cell area for volume calculation
    cell_area = cell_width * CELL_SIZE  # alongshore * vertical

    # Get uncertainty grid if available
    unc_grid = stats['uncertainty'] if stats else None

    # Find unique cluster IDs (excluding 0 and NaN)
    valid_mask = ~np.isnan(clusters) & (clusters != 0)
    unique_clusters = np.unique(clusters[valid_mask])

    events = []

    for cluster_id in unique_clusters:
        cluster_id = int(cluster_id)

        # Get all cells belonging to this cluster
        cluster_mask = clusters == cluster_id

        # Get cell indices
        row_indices, col_indices = np.where(cluster_mask)

        if len(row_indices) == 0:
            continue

        # Extract values for this cluster
        m3c2_values = grid[cluster_mask]
        valid_values = m3c2_values[~np.isnan(m3c2_values)]

        if len(valid_values) == 0:
            continue

        # Volume: sum of |M3C2| * cell_area
        volume = np.sum(np.abs(valid_values)) * cell_area

        # Volume uncertainty
        if unc_grid is not None:
            unc_values = unc_grid[cluster_mask]
            valid_unc = unc_values[~np.isnan(unc_values)]
            if len(valid_unc) > 0:
                # RMS uncertainty propagated through sum
                vol_unc = np.sqrt(np.sum(valid_unc**2)) * cell_area
            else:
                vol_unc = np.nan
        else:
            vol_unc = np.nan

        # Elevation: volume-weighted centroid
        cell_elevations = elevations[col_indices]
        weights = np.abs(valid_values)
        if np.sum(weights) > 0:
            elevation_centroid = np.average(cell_elevations[~np.isnan(m3c2_values)],
                                            weights=weights)
        else:
            elevation_centroid = np.mean(cell_elevations)

        # Height: vertical extent
        height = elevations[col_indices].max() - elevations[col_indices].min() + CELL_SIZE

        # Alongshore positions
        polygon_ids = [rows_g[r] for r in row_indices]
        alongshore_vals = [alongshore_positions.get(pid, np.nan) for pid in polygon_ids]
        alongshore_vals = [v for v in alongshore_vals if not np.isnan(v)]

        if len(alongshore_vals) == 0:
            continue

        alongshore_start = min(alongshore_vals)
        alongshore_end = max(alongshore_vals) + cell_width

        # Alongshore centroid (volume-weighted)
        cell_alongshore = np.array([alongshore_positions.get(rows_g[r], np.nan)
                                    for r in row_indices])
        valid_alongshore_mask = ~np.isnan(cell_alongshore) & ~np.isnan(m3c2_values)
        if np.sum(valid_alongshore_mask) > 0:
            alongshore_centroid = np.average(
                cell_alongshore[valid_alongshore_mask],
                weights=np.abs(m3c2_values[valid_alongshore_mask])
            )
        else:
            alongshore_centroid = (alongshore_start + alongshore_end) / 2

        width = alongshore_end - alongshore_start

        # MOP coordinates
        mop_vals = [mop_positions.get(pid, np.nan) for pid in polygon_ids]
        mop_vals_valid = [v for v in mop_vals if not np.isnan(v)]

        if len(mop_vals_valid) > 0:
            mop_start = min(mop_vals_valid)
            mop_end = max(mop_vals_valid)

            # MOP centroid (volume-weighted)
            cell_mop = np.array([mop_positions.get(rows_g[r], np.nan)
                                 for r in row_indices])
            valid_mop_mask = ~np.isnan(cell_mop) & ~np.isnan(m3c2_values)
            if np.sum(valid_mop_mask) > 0:
                mop_centroid = np.average(
                    cell_mop[valid_mop_mask],
                    weights=np.abs(m3c2_values[valid_mop_mask])
                )
            else:
                mop_centroid = (mop_start + mop_end) / 2
        else:
            mop_start = np.nan
            mop_end = np.nan
            mop_centroid = np.nan

        event = {
            'mid_date': mid_date,
            'start_date': start_date,
            'end_date': end_date,
            'volume': volume,
            'elevation': elevation_centroid,
            'alongshore_centroid_m': alongshore_centroid,
            'alongshore_start_m': alongshore_start,
            'alongshore_end_m': alongshore_end,
            'mop_centroid': mop_centroid,
            'mop_start': mop_start,
            'mop_end': mop_end,
            'width': width,
            'height': height,
            'vol_unc': vol_unc,
            'month': mid_date.month
        }
        events.append(event)

    return events


# ============================================================================
# NPZ CUBE BUILDING
# ============================================================================

def build_data_cube(location, base_dir, event_type='erosion'):
    """
    Build a 3D data cube from filled grid CSVs.

    Args:
        location: Location name (e.g., 'SanElijo')
        base_dir: Base directory path
        event_type: 'erosion' or 'deposition'

    Returns:
        cube: 3D numpy array (alongshore, elevation, time)
        alongshore_m: 1D array of alongshore positions
        elevation_m: 1D array of elevation values
        dates: List of mid-dates
        date_strings: List of date folder names
    """
    results_dir = os.path.join(base_dir, 'results', location)
    util_dir = os.path.join(base_dir, 'utilities')

    type_dir = os.path.join(results_dir, event_type)
    if not os.path.isdir(type_dir):
        print(f"  [WARNING] {event_type}/ directory not found")
        return None, None, None, None, None

    # Load shapefile for alongshore positions
    try:
        shp_path = find_shapefile(util_dir, location, RESOLUTION)
        alongshore_positions, _, _ = load_polygon_alongshore(shp_path)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return None, None, None, None, None

    # Collect all date folders and their grid data
    dates = sorted([d for d in os.listdir(type_dir)
                    if os.path.isdir(os.path.join(type_dir, d))])

    if not dates:
        print(f"  [WARNING] No date folders found in {event_type}/")
        return None, None, None, None, None

    # First pass: collect all unique polygon IDs and elevations
    all_polygon_ids = set()
    all_elevations = set()
    valid_dates = []
    grid_data_cache = {}

    for date_str in dates:
        res_folder = os.path.join(type_dir, date_str, RESOLUTION)
        if not os.path.isdir(res_folder):
            continue

        # Find grid file
        grid_file = None
        for f in os.listdir(res_folder):
            if f.endswith(f'_grid_{RESOLUTION}_filled.csv'):
                grid_file = os.path.join(res_folder, f)
                break

        if not grid_file:
            continue

        headers, rows, grid = load_grid_csv(grid_file)
        if grid is None:
            continue

        # Parse elevations from headers
        elevations = [parse_elevation_from_header(h) for h in headers]
        elevations = [e for e in elevations if e is not None]

        all_polygon_ids.update(rows)
        all_elevations.update(elevations)
        valid_dates.append(date_str)
        grid_data_cache[date_str] = (headers, rows, grid)

    if not valid_dates:
        print(f"  [WARNING] No valid grid files found for {event_type}")
        return None, None, None, None, None

    # Create sorted axes
    sorted_polygon_ids = sorted(all_polygon_ids)
    sorted_elevations = sorted(all_elevations)

    # Map polygon IDs to alongshore positions
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

    # Build 3D cube (alongshore, elevation, time)
    n_alongshore = len(sorted_polygon_ids)
    n_elevation = len(sorted_elevations)
    n_time = len(valid_dates)

    cube = np.full((n_alongshore, n_elevation, n_time), np.nan, dtype=np.float32)

    for t_idx, date_str in enumerate(valid_dates):
        headers, rows, grid = grid_data_cache[date_str]

        for row_idx, pid in enumerate(rows):
            if pid not in pid_to_idx:
                continue
            as_idx = pid_to_idx[pid]

            for col_idx, header in enumerate(headers):
                elv = parse_elevation_from_header(header)
                if elv is None or elv not in elv_to_idx:
                    continue
                el_idx = elv_to_idx[elv]

                value = grid[row_idx, col_idx]
                if not np.isnan(value):
                    cube[as_idx, el_idx, t_idx] = value

    return cube, alongshore_m, elevation_m, mid_dates, valid_dates


def _align_cube_to_dates(cube, source_dates, target_dates):
    """
    Re-index a 3D cube's time axis to align with a target date list.

    Dates present in target but missing from source get NaN slices.

    Args:
        cube: 3D numpy array (alongshore, elevation, time)
        source_dates: List of date strings matching cube's current time axis
        target_dates: List of date strings for the desired time axis

    Returns:
        New 3D numpy array aligned to target_dates
    """
    if cube is None:
        return None

    n_along, n_elev, _ = cube.shape
    n_time = len(target_dates)
    aligned = np.full((n_along, n_elev, n_time), np.nan, dtype=np.float32)

    source_to_idx = {d: i for i, d in enumerate(source_dates)}
    for t_new, date_str in enumerate(target_dates):
        if date_str in source_to_idx:
            aligned[:, :, t_new] = cube[:, :, source_to_idx[date_str]]

    return aligned


def save_data_cube_npz(location, output_dir, base_dir, process_erosion=True, process_deposition=True):
    """
    Build and save 3D data cubes to NPZ file.

    Both erosion and deposition cubes are aligned to a unified time axis
    (the sorted union of their date folders) so that a single date_strings
    index can safely address both cubes.

    Args:
        location: Location name
        output_dir: Output directory
        base_dir: Base directory for input data
        process_erosion: Include erosion data
        process_deposition: Include deposition data

    Returns:
        Path to saved NPZ file, or None if no data
    """
    print(f"  Building data cubes for {location}...")

    # Build both cubes independently
    ero_cube = ero_along = ero_elev = ero_dates = ero_date_strs = None
    dep_cube = dep_along = dep_elev = dep_dates = dep_date_strs = None

    if process_erosion:
        result = build_data_cube(location, base_dir, 'erosion')
        if result[0] is not None:
            ero_cube, ero_along, ero_elev, ero_dates, ero_date_strs = result
            print(f"    Erosion cube: {ero_cube.shape} (alongshore, elevation, time)")
            print(f"      Non-NaN values: {np.sum(~np.isnan(ero_cube)):,}")

    if process_deposition:
        result = build_data_cube(location, base_dir, 'deposition')
        if result[0] is not None:
            dep_cube, dep_along, dep_elev, dep_dates, dep_date_strs = result
            print(f"    Deposition cube: {dep_cube.shape} (alongshore, elevation, time)")
            print(f"      Non-NaN values: {np.sum(~np.isnan(dep_cube)):,}")

    if ero_cube is None and dep_cube is None:
        print(f"  [WARNING] No data cubes built for {location}")
        return None

    # Build unified time axis from the sorted union of both date lists
    if ero_date_strs and dep_date_strs:
        unified_dates = sorted(set(ero_date_strs) | set(dep_date_strs))
        if len(unified_dates) != len(ero_date_strs) or len(unified_dates) != len(dep_date_strs):
            n_ero_only = len(set(ero_date_strs) - set(dep_date_strs))
            n_dep_only = len(set(dep_date_strs) - set(ero_date_strs))
            print(f"    [INFO] Date mismatch: erosion={len(ero_date_strs)}, "
                  f"deposition={len(dep_date_strs)}, unified={len(unified_dates)} "
                  f"({n_ero_only} erosion-only, {n_dep_only} deposition-only)")
    elif ero_date_strs:
        unified_dates = list(ero_date_strs)
    else:
        unified_dates = list(dep_date_strs)

    # Align both cubes to the unified time axis
    if ero_cube is not None:
        ero_cube = _align_cube_to_dates(ero_cube, ero_date_strs, unified_dates)
    if dep_cube is not None:
        dep_cube = _align_cube_to_dates(dep_cube, dep_date_strs, unified_dates)

    # Compute mid-dates for the unified date list
    mid_dates_ordinal = []
    for date_str in unified_dates:
        parts = date_str.split('_to_')
        if len(parts) == 2:
            try:
                start = datetime.strptime(parts[0], '%Y%m%d').date()
                end = datetime.strptime(parts[1], '%Y%m%d').date()
                mid = start + (end - start) / 2
                mid_dates_ordinal.append(mid.toordinal())
            except ValueError:
                mid_dates_ordinal.append(0)
        else:
            mid_dates_ordinal.append(0)

    # Use spatial axes from whichever cube is available (prefer erosion)
    alongshore = ero_along if ero_along is not None else dep_along
    elevation = ero_elev if ero_elev is not None else dep_elev

    # Assemble output
    data = {
        'alongshore_m': alongshore,
        'elevation_m': elevation,
        'dates': np.array(mid_dates_ordinal),
        'date_strings': np.array(unified_dates, dtype=object),
    }
    if ero_cube is not None:
        data['erosion'] = ero_cube
    if dep_cube is not None:
        data['deposition'] = dep_cube

    # Save to NPZ
    npz_path = os.path.join(output_dir, f'{location}_cube.npz')
    np.savez_compressed(npz_path, **data)
    print(f"  Saved: {npz_path} (unified time steps: {len(unified_dates)})")

    return npz_path


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_location(location, base_dir, process_erosion=True, process_deposition=True):
    """
    Process a single location and extract all events.

    Returns:
        erosion_events: List of erosion event dicts
        deposition_events: List of deposition event dicts
    """
    results_dir = os.path.join(base_dir, 'results', location)
    util_dir = os.path.join(base_dir, 'utilities')

    if not os.path.isdir(results_dir):
        print(f"[WARNING] Results directory not found: {results_dir}")
        return [], []

    # Load shapefile for alongshore positions
    try:
        shp_path = find_shapefile(util_dir, location, RESOLUTION)
        alongshore_positions, mop_positions, cell_width = load_polygon_alongshore(shp_path)
        has_mop = any(not np.isnan(v) for v in mop_positions.values())
        mop_status = "with MOP_ID" if has_mop else "no MOP_ID"
        print(f"  Loaded {len(alongshore_positions)} polygons, cell width: {cell_width:.2f}m ({mop_status})")
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return [], []

    erosion_events = []
    deposition_events = []

    # Process erosion
    if process_erosion:
        ero_dir = os.path.join(results_dir, 'erosion')
        if os.path.isdir(ero_dir):
            dates = sorted([d for d in os.listdir(ero_dir)
                           if os.path.isdir(os.path.join(ero_dir, d))])
            print(f"  Processing {len(dates)} erosion date folders...")

            for date_str in dates:
                res_folder = os.path.join(ero_dir, date_str, RESOLUTION)
                if not os.path.isdir(res_folder):
                    continue

                # Find the filled grid files
                grid_file = None
                cluster_file = None
                stats_file = None

                for f in os.listdir(res_folder):
                    if f.endswith(f'_grid_{RESOLUTION}_filled.csv'):
                        grid_file = os.path.join(res_folder, f)
                    elif f.endswith(f'_clusters_{RESOLUTION}_filled.csv'):
                        cluster_file = os.path.join(res_folder, f)
                    elif f.endswith(f'_stats_{RESOLUTION}.npz'):
                        stats_file = os.path.join(res_folder, f)

                if grid_file and cluster_file:
                    events = extract_events_from_grids(
                        grid_file, cluster_file, stats_file,
                        alongshore_positions, mop_positions, cell_width,
                        date_str, 'erosion'
                    )
                    erosion_events.extend(events)

    # Process deposition
    if process_deposition:
        dep_dir = os.path.join(results_dir, 'deposition')
        if os.path.isdir(dep_dir):
            dates = sorted([d for d in os.listdir(dep_dir)
                           if os.path.isdir(os.path.join(dep_dir, d))])
            print(f"  Processing {len(dates)} deposition date folders...")

            for date_str in dates:
                res_folder = os.path.join(dep_dir, date_str, RESOLUTION)
                if not os.path.isdir(res_folder):
                    continue

                grid_file = None
                cluster_file = None
                stats_file = None

                for f in os.listdir(res_folder):
                    if f.endswith(f'_grid_{RESOLUTION}_filled.csv'):
                        grid_file = os.path.join(res_folder, f)
                    elif f.endswith(f'_clusters_{RESOLUTION}_filled.csv'):
                        cluster_file = os.path.join(res_folder, f)
                    elif f.endswith(f'_stats_{RESOLUTION}.npz'):
                        stats_file = os.path.join(res_folder, f)

                if grid_file and cluster_file:
                    events = extract_events_from_grids(
                        grid_file, cluster_file, stats_file,
                        alongshore_positions, mop_positions, cell_width,
                        date_str, 'deposition'
                    )
                    deposition_events.extend(events)

    return erosion_events, deposition_events


def main():
    parser = argparse.ArgumentParser(
        description='Generate event list CSVs from 25cm gridded results'
    )
    parser.add_argument('location', nargs='?', help='Location name (e.g., SanElijo)')
    parser.add_argument('--all', action='store_true', help='Process all locations')
    parser.add_argument('--erosion', action='store_true', help='Process only erosion')
    parser.add_argument('--deposition', action='store_true', help='Process only deposition')
    parser.add_argument('--make-npz', action='store_true',
                        help='Create 3D data cube NPZ file (alongshore × elevation × time)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: results/event_lists/)')
    args = parser.parse_args()

    # Validate arguments
    if args.all and args.location:
        parser.error("Cannot specify both --all and a specific location")
    if not args.all and not args.location:
        parser.error("Must specify either a location or --all")

    # Determine what to process
    process_erosion = args.erosion or not (args.erosion or args.deposition)
    process_deposition = args.deposition or not (args.erosion or args.deposition)

    # Get locations to process
    locations = ALL_LOCATIONS if args.all else [args.location]

    base_dir = get_base_dir()

    # Set output directory (default: repo's results/event_lists/)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(script_dir))  # up from utilities/event_lists/
        output_dir = os.path.join(repo_root, 'results', 'event_lists')

    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories (only needed for CSV output, not NPZ)
    erosion_dir = os.path.join(output_dir, 'erosion')
    deposition_dir = os.path.join(output_dir, 'deposition')
    combined_dir = os.path.join(output_dir, 'combined')

    if not args.make_npz:
        os.makedirs(erosion_dir, exist_ok=True)
        os.makedirs(deposition_dir, exist_ok=True)
        os.makedirs(combined_dir, exist_ok=True)

    # Set up NPZ output directory (separate from CSV output)
    npz_output_dir = os.path.join(os.path.dirname(output_dir), 'data_cubes')
    if args.make_npz:
        os.makedirs(npz_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EVENT LIST GENERATION")
    print(f"Resolution: {RESOLUTION}")
    print(f"Output: {output_dir}")
    print(f"  - erosion/")
    print(f"  - deposition/")
    print(f"  - combined/")
    if args.make_npz:
        print(f"  + NPZ cubes: {npz_output_dir}")
    print(f"{'='*60}\n")

    for location in locations:
        print(f"\n--- {location} ---")

        # Always generate event list CSVs
        erosion_events, deposition_events = process_location(
            location, base_dir,
            process_erosion=process_erosion,
            process_deposition=process_deposition
        )

        # Save erosion events
        if erosion_events:
            df_ero = pd.DataFrame(erosion_events).sort_values('mid_date')
            ero_path = os.path.join(erosion_dir, f'{location}_events.csv')
            df_ero.to_csv(ero_path, index=False)
            print(f"  Erosion: {len(df_ero)} events -> {ero_path}")
        else:
            print(f"  Erosion: No events found")

        # Save deposition events
        if deposition_events:
            df_dep = pd.DataFrame(deposition_events).sort_values('mid_date')
            dep_path = os.path.join(deposition_dir, f'{location}_events.csv')
            df_dep.to_csv(dep_path, index=False)
            print(f"  Deposition: {len(df_dep)} events -> {dep_path}")
        else:
            print(f"  Deposition: No events found")

        # Save combined events
        all_events = erosion_events + deposition_events
        if all_events:
            df_all = pd.DataFrame(all_events).sort_values('mid_date')
            combined_path = os.path.join(combined_dir, f'{location}_events.csv')
            df_all.to_csv(combined_path, index=False)
            print(f"  Combined: {len(df_all)} events -> {combined_path}")
            print(f"    Date range: {df_all['start_date'].min()} to {df_all['end_date'].max()}")
            print(f"    Volume range: {df_all['volume'].min():.2f} to {df_all['volume'].max():.2f} m³")
        else:
            print(f"  No events found for {location}")

        # Additionally generate NPZ cubes if requested
        if args.make_npz:
            save_data_cube_npz(
                location, npz_output_dir, base_dir,
                process_erosion=process_erosion,
                process_deposition=process_deposition
            )

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
