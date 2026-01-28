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
        csv_path: Path like '/path/to/SanElijo_events.csv'

    Returns:
        Location string like 'SanElijo'
    """
    basename = os.path.basename(csv_path)
    # Remove common suffixes
    name = basename.replace('_events_sig.csv', '').replace('_events.csv', '')
    name = name.replace('_dep_events_sig.csv', '').replace('_dep_events.csv', '')
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


def get_zoom_extent(event: pd.Series, resolution_m: float, padding: float = 0.3) -> dict:
    """
    Calculate zoom bounds from event coordinates.

    Args:
        event: Event row from events DataFrame
        resolution_m: Resolution in meters
        padding: Fractional padding to add around event (default 30%)

    Returns:
        Dict with x_min, x_max, y_min, y_max in physical units (meters)
    """
    # Alongshore bounds (X-axis)
    x_min = event['alongshore_start_m']
    x_max = event['alongshore_end_m']
    x_range = x_max - x_min
    x_pad = max(5, x_range * padding)

    # Elevation bounds (Y-axis)
    elev_centroid = event['elevation']
    height = event['height']
    y_min = max(0, elev_centroid - height / 2 - height * padding)
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

    Mapping examples:
        - results/event_lists/erosion/SanElijo_events.csv -> results/data_cubes/SanElijo_events_cube.npz
        - results/event_lists/combined/SanElijo_vol_5_elv_3.csv -> results/data_cubes/SanElijo_vol_5_elv_3_cube.npz

    Args:
        csv_path: Path to event CSV file
        data_cubes_dir: Directory containing NPZ cubes (if None, infers from csv_path)

    Returns:
        Path to corresponding NPZ file
    """
    basename = os.path.basename(csv_path)

    # Mapping: CSV name with _cube suffix
    # SanElijo_events.csv -> SanElijo_events_cube.npz
    npz_name = basename.replace('.csv', '_cube.npz')

    # Determine data_cubes directory
    if data_cubes_dir is None:
        # Infer from csv_path: results/event_lists/<subdir>/file.csv -> results/data_cubes/
        csv_dir = os.path.dirname(csv_path)

        # Walk up to find 'event_lists' directory
        current = csv_dir
        while current:
            if os.path.basename(current) == 'event_lists':
                # Found it - data_cubes is a sibling
                results_dir = os.path.dirname(current)
                data_cubes_dir = os.path.join(results_dir, 'data_cubes')
                break
            parent = os.path.dirname(current)
            if parent == current:  # Reached root
                break
            current = parent

        # Fallback: assume data_cubes is sibling to csv directory's grandparent
        if data_cubes_dir is None:
            results_dir = os.path.dirname(os.path.dirname(csv_dir))
            data_cubes_dir = os.path.join(results_dir, 'data_cubes')

    return os.path.join(data_cubes_dir, npz_name)


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
            - Index: alongshore positions (m)
            - Columns: elevation values (m)
            - Values: M3C2 change values

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

    # Create DataFrame with proper coordinates
    alongshore = cube_data['alongshore_m']
    elevation = cube_data['elevation_m']

    # DataFrame: rows=alongshore, columns=elevation
    df = pd.DataFrame(
        slice_2d,
        index=alongshore,
        columns=elevation
    )

    # Replace NaN with 0 for display
    df = df.fillna(0.0)

    return df


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
    if os.path.exists(npz_path):
        return npz_path
    return None
