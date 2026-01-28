#!/usr/bin/env python3
"""
grid_loader.py

Utilities for loading and processing grid data for the Event QC tool.
"""

import os
import re
import glob
import platform
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
    Clean column names and snap to resolution grid.
    Matches existing dashboard pattern.

    Args:
        df: Raw grid DataFrame
        resolution_val: Resolution in meters

    Returns:
        Cleaned DataFrame with integer column indices
    """
    # Remove letters and underscores from column names (e.g., 'M3C2_0.25m' -> '0.25')
    cleaned_cols = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)

    try:
        col_floats = cleaned_cols.astype(float)
        # Convert to grid indices
        scale = 1.0 / resolution_val
        new_cols = (col_floats * scale).round().astype(int)
        df.columns = new_cols
        df.index = df.index.astype(int)
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


def scan_event_csvs(directory: str) -> list:
    """
    Scan directory for event CSV files.

    Args:
        directory: Directory to scan

    Returns:
        List of CSV file paths
    """
    if not os.path.isdir(directory):
        return []

    patterns = ['*_events.csv', '*_events_sig.csv', '*_dep_events.csv', '*_dep_events_sig.csv']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern)))

    return sorted(set(files))
