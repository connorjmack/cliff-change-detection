#!/usr/bin/env python3
"""
grid_loader.py

Utilities for loading and processing grid data for the Event QC tool.
"""

import os
import re
import glob
import platform
import numpy as np
import pandas as pd
from datetime import datetime
from functools import lru_cache


# OS Detection and Path Setup (matching existing codebase pattern)
if platform.system() == "Darwin":
    DEFAULT_RESULTS_DIR = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs/results"
else:
    DEFAULT_RESULTS_DIR = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs/results"


def get_resolution_value(resolution: str) -> float:
    """Convert resolution string to numeric value in meters."""
    if resolution == '10cm':
        return 0.10
    elif resolution == '25cm':
        return 0.25
    elif resolution == '1m':
        return 1.0
    return 0.25


def event_dates_to_folder(start_date: str, end_date: str) -> str:
    """
    Convert event dates to grid folder name format.

    Args:
        start_date: Date string like '2017-10-04'
        end_date: Date string like '2018-01-31'

    Returns:
        Folder name like '20171004_to_20180131'
    """
    start = pd.to_datetime(start_date).strftime('%Y%m%d')
    end = pd.to_datetime(end_date).strftime('%Y%m%d')
    return f"{start}_to_{end}"


def infer_location_from_filename(csv_path: str) -> str:
    """
    Infer location name from event CSV filename.

    Args:
        csv_path: Path like '/path/to/SanElijo_events.csv' or '/path/to/DelMar_vol_10_elv_3.csv'
                  Also handles QC files like 'DelMar_vol_10_elv_3_qc_20260128_172029.csv'

    Returns:
        Location string like 'SanElijo' or 'DelMar'
    """
    basename = os.path.basename(csv_path)
    # Remove .csv extension
    name = basename.replace('.csv', '')

    # Strip QC suffix pattern: _qc_YYYYMMDD_HHMMSS and anything after
    # This handles files saved from the QC tool
    name = re.sub(r'_qc_\d{8}_\d{6}.*$', '', name)

    # Remove common suffixes in order (longer/more specific first to avoid partial matches)
    suffixes_to_remove = [
        '_dep_events_sig', '_events_sig',
        '_dep_events', '_events',
    ]
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    # Remove filter suffixes like _vol_10_elv_3 or _vol_5_elv_5
    # Pattern: _vol_<number>_elv_<number> at end of string
    name = re.sub(r'_vol_\d+_elv_\d+$', '', name)

    return name


def infer_event_type_from_filename(csv_path: str) -> str:
    """
    Infer event type (erosion/deposition) from event CSV filename.

    Args:
        csv_path: Path like '/path/to/SanElijo_dep_events.csv'

    Returns:
        'erosion' or 'deposition'
    """
    basename = os.path.basename(csv_path).lower()
    if 'dep' in basename:
        return 'deposition'
    return 'erosion'


def find_grid_file(results_dir: str, location: str, event_type: str,
                   date_folder: str, resolution: str) -> str:
    """
    Locate the filled grid CSV for a specific survey pair.

    Args:
        results_dir: Base results directory
        location: Location name (e.g., 'SanElijo')
        event_type: 'erosion' or 'deposition'
        date_folder: Folder name like '20171004_to_20180131'
        resolution: Resolution string like '25cm'

    Returns:
        Path to grid CSV or None if not found
    """
    prefix = 'ero' if event_type == 'erosion' else 'dep'
    grid_dir = os.path.join(results_dir, location, event_type, date_folder, resolution)

    if not os.path.isdir(grid_dir):
        return None

    # Try multiple patterns (filled preferred, then unfilled)
    patterns = [
        f"{date_folder}_{prefix}_grid_{resolution}_filled.csv",
        f"*_{prefix}_grid_{resolution}_filled.csv",
        f"{date_folder}_{prefix}_grid_{resolution}.csv",
        f"*_{prefix}_grid_{resolution}.csv",
    ]

    for pattern in patterns:
        matches = glob.glob(os.path.join(grid_dir, pattern))
        if matches:
            return matches[0]

    return None


def clean_and_snap_grid(df: pd.DataFrame, resolution_val: float) -> pd.DataFrame:
    """
    Clean column names and keep coordinates in physical meters.

    Args:
        df: Raw grid DataFrame
        resolution_val: Resolution in meters (used for reference)

    Returns:
        Cleaned DataFrame with float column values (elevation in meters)
    """
    # Remove letters and underscores from column names (e.g., 'M3C2_0.25m' -> '0.25')
    cleaned_cols = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)

    try:
        col_floats = cleaned_cols.astype(float)
        # Keep columns as physical meters (not grid indices)
        df.columns = col_floats
        # Keep row index as-is (alongshore positions, typically in meters)
        df.index = df.index.astype(float)
        return df
    except Exception:
        return None


def load_and_prepare_grid(grid_path: str, resolution_m: float) -> pd.DataFrame:
    """
    Load grid CSV and prepare for visualization.

    Args:
        grid_path: Path to grid CSV file
        resolution_m: Resolution in meters

    Returns:
        Prepared DataFrame or None if loading fails
    """
    if not os.path.exists(grid_path):
        return None

    try:
        df = pd.read_csv(grid_path, index_col=0, na_values=['', 'nan', 'NaN', 'NULL'])
        cleaned = clean_and_snap_grid(df.copy(), resolution_m)
        if cleaned is not None:
            return cleaned.fillna(0.0)
        return None
    except Exception as e:
        print(f"Error loading grid {grid_path}: {e}")
        return None


def get_zoom_extent(event: pd.Series, resolution_m: float, padding: float = 0.3,
                    x_pad_m: float = None, y_pad_bottom_m: float = None,
                    y_pad_top_m: float = None) -> dict:
    """
    Calculate zoom bounds from event coordinates.

    Args:
        event: Event row from events DataFrame
        resolution_m: Resolution in meters
        padding: Fractional padding to add around event (default 30%)
        x_pad_m: Override horizontal padding in meters (applied to both sides)
        y_pad_bottom_m: Override bottom padding in meters
        y_pad_top_m: Override top padding in meters

    Returns:
        Dict with x_min, x_max, y_min, y_max in physical units (meters)
    """
    # Alongshore bounds (X-axis)
    x_min = event['alongshore_start_m']
    x_max = event['alongshore_end_m']
    x_range = x_max - x_min

    if x_pad_m is not None:
        x_pad = x_pad_m
    else:
        x_pad = max(5, x_range * padding)

    # Elevation bounds (Y-axis)
    elev_centroid = event['elevation']
    height = event['height']

    if y_pad_bottom_m is not None:
        y_min = max(0, elev_centroid - height / 2 - y_pad_bottom_m)
    else:
        y_min = max(0, elev_centroid - height / 2 - height * padding)

    if y_pad_top_m is not None:
        y_max = elev_centroid + height / 2 + y_pad_top_m
    else:
        y_max = elev_centroid + height / 2 + height * padding

    return {
        'x_min': x_min - x_pad,
        'x_max': x_max + x_pad,
        'y_min': y_min,
        'y_max': y_max
    }


def load_event_csv(csv_path: str) -> pd.DataFrame:
    """
    Load event list CSV file.

    Args:
        csv_path: Path to event CSV file

    Returns:
        Events DataFrame or None if loading fails
    """
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        # Ensure date columns are parsed
        for col in ['start_date', 'end_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
        return df
    except Exception as e:
        print(f"Error loading event CSV {csv_path}: {e}")
        return None


def scan_event_csvs(directory: str, recursive: bool = True) -> list:
    """
    Scan directory for all CSV files.

    Args:
        directory: Directory to scan
        recursive: If True, also scan subdirectories

    Returns:
        List of CSV file paths
    """
    if not os.path.isdir(directory):
        return []

    if recursive:
        # Scan directory and all subdirectories
        files = glob.glob(os.path.join(directory, '**', '*.csv'), recursive=True)
    else:
        # Scan only the specified directory
        files = glob.glob(os.path.join(directory, '*.csv'))

    return sorted(set(files))


# ============================================================================
# NPZ DATA CUBE UTILITIES
# ============================================================================

def csv_path_to_npz_path(csv_path: str, data_cubes_dir: str = None) -> str:
    """
    Convert an event CSV path to its corresponding NPZ cube path.

    Always maps to the base location cube, regardless of filtering suffixes.

    Mapping examples:
        - results/event_lists/erosion/DelMar_events.csv -> results/data_cubes/DelMar_cube.npz
        - results/event_lists/erosion/SanElijo_events.csv -> results/data_cubes/SanElijo_cube.npz
        - results/event_lists/combined/DelMar_vol_10_elv_3.csv -> results/data_cubes/DelMar_cube.npz

    Args:
        csv_path: Path to event CSV file
        data_cubes_dir: Directory containing NPZ cubes (if None, infers from csv_path)

    Returns:
        Path to corresponding NPZ file
    """
    # Use infer_location_from_filename to get the base location name
    # This handles all suffix patterns: _events, _dep_events, _vol_X_elv_Y, etc.
    location = infer_location_from_filename(csv_path)
    npz_name = f"{location}_cube.npz"

    # Determine data_cubes directory
    if data_cubes_dir is None:
        csv_dir = os.path.dirname(os.path.abspath(csv_path))

        # Walk up to find 'results' directory
        current = csv_dir
        while current and current != '/':
            if os.path.basename(current) == 'results':
                # Found results dir - data_cubes is a subdirectory
                data_cubes_dir = os.path.join(current, 'data_cubes')
                break
            parent = os.path.dirname(current)
            if parent == current:  # Reached root
                break
            current = parent

        # Fallback: look for results/data_cubes relative to project root
        if data_cubes_dir is None:
            # Try to find project root by looking for common markers
            current = csv_dir
            while current and current != '/':
                potential = os.path.join(current, 'results', 'data_cubes')
                if os.path.isdir(potential):
                    data_cubes_dir = potential
                    break
                parent = os.path.dirname(current)
                if parent == current:
                    break
                current = parent

    return os.path.join(data_cubes_dir, npz_name) if data_cubes_dir else None


def load_npz_cube(npz_path: str) -> dict:
    """
    Load a 3D data cube from NPZ file.

    Args:
        npz_path: Path to NPZ file

    Returns:
        Dict with keys:
            - 'erosion': 3D array (alongshore, elevation, time) or None
            - 'deposition': 3D array (alongshore, elevation, time) or None
            - 'alongshore_m': 1D array of alongshore positions (m)
            - 'elevation_m': 1D array of elevation values (m)
            - 'date_strings': 1D array of date folder names (YYYYMMDD_to_YYYYMMDD)
            - 'dates': 1D array of ordinal dates

        Returns None if file doesn't exist or fails to load.
    """
    if not os.path.exists(npz_path):
        return None

    try:
        data = np.load(npz_path, allow_pickle=True)

        result = {
            'erosion': data.get('erosion'),
            'deposition': data.get('deposition'),
            'alongshore_m': data['alongshore_m'],
            'elevation_m': data['elevation_m'],
            'date_strings': data['date_strings'],
            'dates': data.get('dates'),
        }

        # Handle case where date_strings might be object array
        if result['date_strings'] is not None:
            result['date_strings'] = [str(s) for s in result['date_strings']]

        return result

    except Exception as e:
        print(f"Error loading NPZ {npz_path}: {e}")
        return None


def extract_grid_slice_from_cube(cube_data: dict, event: pd.Series,
                                  event_type: str = 'erosion') -> pd.DataFrame:
    """
    Extract a 2D grid slice from the 3D cube for a specific event.

    The event's date range is matched to the cube's date_strings to find
    the correct time index.

    Args:
        cube_data: Dict from load_npz_cube()
        event: Event row from events DataFrame (must have start_date, end_date)
        event_type: 'erosion' or 'deposition'

    Returns:
        DataFrame with:
            - Index: alongshore positions (m), sorted ascending
            - Columns: elevation values (m), sorted ascending
            - Values: M3C2 change values (reordered to match sorted coordinates)

        Returns None if matching slice not found.
    """
    if cube_data is None:
        return None

    # Get the appropriate 3D cube
    cube_3d = cube_data.get(event_type)
    if cube_3d is None:
        return None

    # Build date folder string from event dates
    date_folder = event_dates_to_folder(event['start_date'], event['end_date'])

    # Find matching time index
    date_strings = cube_data.get('date_strings', [])
    time_idx = None
    for i, ds in enumerate(date_strings):
        if ds == date_folder:
            time_idx = i
            break

    if time_idx is None:
        return None

    # Extract 2D slice: (alongshore, elevation) at time_idx
    slice_2d = cube_3d[:, :, time_idx]

    # Get coordinates
    alongshore = np.array(cube_data['alongshore_m'])
    elevation = np.array(cube_data['elevation_m'])

    # CRITICAL: Sort alongshore and reorder data to match
    # imshow requires data to be on a regular sorted grid
    along_sort_idx = np.argsort(alongshore)
    alongshore_sorted = alongshore[along_sort_idx]
    slice_2d_sorted = slice_2d[along_sort_idx, :]

    # DataFrame: rows=alongshore (sorted), columns=elevation
    df = pd.DataFrame(
        slice_2d_sorted,
        index=alongshore_sorted,
        columns=elevation
    )

    # Replace NaN with 0 for display
    df = df.fillna(0.0)

    return df


def extract_both_slices_from_cube(cube_data: dict, event: pd.Series) -> tuple:
    """
    Extract both erosion and deposition 2D slices from the 3D cube for a specific event.

    Args:
        cube_data: Dict from load_npz_cube()
        event: Event row from events DataFrame (must have start_date, end_date)

    Returns:
        Tuple of (erosion_df, deposition_df, alongshore_sorted, elevation)
        DataFrames have:
            - Index: alongshore positions (m), sorted ascending
            - Columns: elevation values (m)
            - Values: M3C2 change values

        Returns (None, None, None, None) if date not found.
    """
    if cube_data is None:
        return None, None, None, None

    # Build date folder string from event dates
    date_folder = event_dates_to_folder(event['start_date'], event['end_date'])

    # Find matching time index
    date_strings = cube_data.get('date_strings', [])
    time_idx = None
    for i, ds in enumerate(date_strings):
        if ds == date_folder:
            time_idx = i
            break

    if time_idx is None:
        return None, None, None, None

    # Get coordinates and sort indices
    alongshore = np.array(cube_data['alongshore_m'])
    elevation = np.array(cube_data['elevation_m'])
    along_sort_idx = np.argsort(alongshore)
    alongshore_sorted = alongshore[along_sort_idx]

    # Extract erosion slice
    erosion_df = None
    erosion_cube = cube_data.get('erosion')
    if erosion_cube is not None:
        slice_2d = erosion_cube[:, :, time_idx]
        slice_sorted = slice_2d[along_sort_idx, :]
        erosion_df = pd.DataFrame(slice_sorted, index=alongshore_sorted, columns=elevation)

    # Extract deposition slice
    deposition_df = None
    deposition_cube = cube_data.get('deposition')
    if deposition_cube is not None:
        slice_2d = deposition_cube[:, :, time_idx]
        slice_sorted = slice_2d[along_sort_idx, :]
        deposition_df = pd.DataFrame(slice_sorted, index=alongshore_sorted, columns=elevation)

    return erosion_df, deposition_df, alongshore_sorted, elevation


def find_npz_for_csv(csv_path: str) -> str:
    """
    Find the NPZ file corresponding to a CSV file.

    Searches in the sibling data_cubes directory.

    Args:
        csv_path: Path to event CSV file

    Returns:
        Path to NPZ file if found, None otherwise
    """
    npz_path = csv_path_to_npz_path(csv_path)
    if npz_path and os.path.exists(npz_path):
        return npz_path
    return None
