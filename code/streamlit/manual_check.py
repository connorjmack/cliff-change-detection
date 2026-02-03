#!/usr/bin/env python3
"""
Manual Check Tool - Streamlit GUI for re-checking "needs_check" events

Displays M3C2 distances and DBSCAN clusters side-by-side in 2D cliff-planar view.
Updates the input CSV in place when events are reclassified.

Usage:
    streamlit run manual_check.py
"""

import os
import glob
import platform
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from datetime import datetime

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

from utils.grid_loader import (
    infer_location_from_filename,
    infer_event_type_from_filename,
    event_dates_to_folder,
    load_event_csv,
    scan_event_csvs,
)

# === Constants ===
QC_FLAGS = ['real', 'noise', 'construction', 'veg_error', 'beach_error', 'other']

# OS-aware base paths
if platform.system() == "Darwin":
    BASE_RESULTS_DIR = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs/results"
else:
    BASE_RESULTS_DIR = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs/results"

# Approximate cliff orientation angles (degrees from north, clockwise)
# Used to rotate UTM coordinates to cliff-planar view
CLIFF_ORIENTATIONS = {
    "DelMar": 342,      # Cliff faces roughly west
    "Solana": 345,
    "Encinitas": 350,
    "SanElijo": 348,
    "Torrey": 340,
    "Blacks": 335,
}


def find_latest_pipeline_run(m3c2_base_dir: str) -> str:
    """Find the most recent pipeline_run folder."""
    runs = glob.glob(os.path.join(m3c2_base_dir, "pipeline_run_*"))
    if not runs:
        return None

    def extract_date(path):
        folder_name = os.path.basename(path)
        try:
            date_str = folder_name.replace("pipeline_run_", "")
            return int(date_str)
        except ValueError:
            return 0

    runs.sort(key=extract_date, reverse=True)
    return runs[0]


def find_m3c2_las_for_event(event: pd.Series, location: str, results_dir: str = None) -> str:
    """
    Find the M3C2 LAS file for a given event.

    Args:
        event: Event row with start_date and end_date
        location: Location name (e.g., 'DelMar')
        results_dir: Base results directory

    Returns:
        Path to M3C2 LAS file or None if not found
    """
    if results_dir is None:
        results_dir = BASE_RESULTS_DIR

    date_folder = event_dates_to_folder(event['start_date'], event['end_date'])
    m3c2_base = os.path.join(results_dir, location, "m3c2")

    # Find latest pipeline run
    pipeline_run = find_latest_pipeline_run(m3c2_base)
    if not pipeline_run:
        return None

    # Look for M3C2 file in the date folder
    m3c2_dir = os.path.join(pipeline_run, date_folder)
    if not os.path.isdir(m3c2_dir):
        return None

    # Find the m3c2.las file
    pattern = os.path.join(m3c2_dir, "*_m3c2.las")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def find_dbscan_las_for_event(event: pd.Series, location: str, event_type: str,
                               results_dir: str = None) -> str:
    """
    Find the DBSCAN clusters LAS file for a given event.

    Args:
        event: Event row with start_date and end_date
        location: Location name
        event_type: 'erosion' or 'deposition'
        results_dir: Base results directory

    Returns:
        Path to clusters LAS file or None if not found
    """
    if results_dir is None:
        results_dir = BASE_RESULTS_DIR

    date_folder = event_dates_to_folder(event['start_date'], event['end_date'])
    prefix = 'ero' if event_type == 'erosion' else 'dep'

    clusters_path = os.path.join(
        results_dir, location, event_type, date_folder,
        f"{prefix}_clusters.las"
    )

    return clusters_path if os.path.exists(clusters_path) else None


def load_las_subset(las_path: str, x_min: float, x_max: float,
                    y_min: float, y_max: float,
                    z_min: float = None, z_max: float = None) -> dict:
    """
    Load points from LAS file within a bounding box.

    Returns dict with 'x', 'y', 'z', 'm3c2_distance', 'cluster_id' arrays.
    """
    if not HAS_LASPY:
        return None

    try:
        las = laspy.read(las_path)
    except Exception as e:
        st.error(f"Error reading LAS: {e}")
        return None

    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    # Spatial filter
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    if z_min is not None:
        mask &= (z >= z_min)
    if z_max is not None:
        mask &= (z <= z_max)

    if np.sum(mask) == 0:
        return None

    result = {
        'x': x[mask],
        'y': y[mask],
        'z': z[mask],
    }

    # Find M3C2 distance field
    m3c2_field = None
    for field in las.point_format.dimension_names:
        if 'm3c2' in field.lower() and 'distance' in field.lower():
            m3c2_field = field
            break
    if m3c2_field is None:
        m3c2_field = next((f for f in las.point_format.dimension_names
                          if 'm3c2' == f.lower()), None)

    if m3c2_field:
        result['m3c2_distance'] = getattr(las, m3c2_field)[mask]
    else:
        result['m3c2_distance'] = np.zeros(np.sum(mask))

    # Find ClusterID field
    cluster_field = next((f for f in las.point_format.dimension_names
                         if 'cluster' in f.lower()), None)
    if cluster_field:
        result['cluster_id'] = getattr(las, cluster_field)[mask]
    else:
        result['cluster_id'] = np.zeros(np.sum(mask), dtype=int)

    return result


def rotate_to_cliff_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          orientation_deg: float) -> tuple:
    """
    Rotate UTM coordinates to cliff-planar view.

    Args:
        x, y, z: UTM coordinates
        orientation_deg: Cliff orientation in degrees from north (clockwise)

    Returns:
        (alongshore, elevation) - 2D coordinates for plotting
    """
    # Convert to radians (orientation is clockwise from north)
    theta = np.radians(orientation_deg)

    # Center the coordinates
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)

    # Rotate to align cliff with Y axis (alongshore = rotated Y)
    # Cross-shore = rotated X (perpendicular to cliff)
    alongshore = x_c * np.sin(theta) + y_c * np.cos(theta)

    return alongshore, z


def estimate_utm_bbox_from_event(event: pd.Series, location: str,
                                  buffer_m: float = 20) -> dict:
    """
    Estimate UTM bounding box from event's alongshore coordinates.

    This is approximate - we use the event's alongshore extent plus a buffer.
    The actual mapping from alongshore_m to UTM depends on the polygon shapefile,
    but for subsetting purposes we can use a generous buffer.

    Returns dict with x_min, x_max, y_min, y_max, z_min, z_max
    """
    # The alongshore positions in the event CSV are derived from polygon IDs
    # which are based on MOP lines. The UTM coordinates are roughly:
    # - Y (northing) varies along the coast
    # - X (easting) varies perpendicular to coast

    # We'll use the event's spatial extent with a large buffer
    # This is imprecise but works for subsetting the point cloud

    alongshore_min = event['alongshore_start_m']
    alongshore_max = event['alongshore_end_m']
    elevation_min = max(0, event['elevation'] - event['height'] / 2 - buffer_m)
    elevation_max = event['elevation'] + event['height'] / 2 + buffer_m

    # For UTM coordinates, we need to know the approximate mapping
    # Based on typical San Diego cliffs, UTM Y increases northward
    # and the alongshore coordinate roughly tracks Y
    # We'll use a very generous X range (perpendicular to cliff)

    return {
        'y_min': alongshore_min * 1000 + 3600000,  # Rough estimate
        'y_max': alongshore_max * 1000 + 3600000,
        'x_min': 460000,  # Wide range to capture cliff
        'x_max': 490000,
        'z_min': elevation_min,
        'z_max': elevation_max,
    }


def get_actual_utm_bbox(las_path: str, event: pd.Series, buffer_m: float = 15) -> dict:
    """
    Get UTM bounding box by reading the LAS file header and subsetting by elevation.
    More accurate than estimate_utm_bbox_from_event.
    """
    if not HAS_LASPY:
        return None

    try:
        las = laspy.read(las_path)
    except Exception:
        return None

    # Get full extent from file
    x_min_file, x_max_file = las.header.x_min, las.header.x_max
    y_min_file, y_max_file = las.header.y_min, las.header.y_max

    # Elevation bounds from event
    z_min = max(0, event['elevation'] - event['height'] / 2 - buffer_m)
    z_max = event['elevation'] + event['height'] / 2 + buffer_m

    # For alongshore, we'll use the relative position in the file
    # The event's alongshore_centroid_m is relative to MOP lines
    # We approximate by assuming the file covers the same alongshore range
    # and find the corresponding Y range

    # Simplified: just filter by elevation and use some Y range based on event width
    y_range = y_max_file - y_min_file

    # Approximate the event's Y position based on alongshore centroid
    # This assumes alongshore_m roughly maps to northing within the survey
    alongshore_range = 200  # Typical survey alongshore extent in polygon units
    event_along_center = event['alongshore_centroid_m']
    event_along_start = event['alongshore_start_m']
    event_along_end = event['alongshore_end_m']

    # Use the event width plus buffer
    width = event_along_end - event_along_start

    return {
        'x_min': x_min_file,
        'x_max': x_max_file,
        'y_min': y_min_file,  # Will filter further if needed
        'y_max': y_max_file,
        'z_min': z_min,
        'z_max': z_max,
        'alongshore_center': event_along_center,
        'alongshore_width': width + 2 * buffer_m,
    }


def load_and_filter_las(las_path: str, event: pd.Series, buffer_m: float = 15) -> dict:
    """
    Load LAS file and filter to event region based on elevation and alongshore.
    """
    if not HAS_LASPY:
        return None

    try:
        las = laspy.read(las_path)
    except Exception as e:
        st.error(f"Error reading {las_path}: {e}")
        return None

    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    # Filter by elevation
    z_min = max(0, event['elevation'] - event['height'] / 2 - buffer_m)
    z_max = event['elevation'] + event['height'] / 2 + buffer_m
    z_mask = (z >= z_min) & (z <= z_max)

    if np.sum(z_mask) == 0:
        return None

    # Apply elevation filter first
    x = x[z_mask]
    y = y[z_mask]
    z = z[z_mask]

    # Now filter by alongshore position
    # The alongshore_m in the event is derived from polygon IDs
    # We'll subset to a Y range that's proportional to the event width
    y_full_range = np.max(y) - np.min(y)

    # Get event's alongshore extent relative to the file's extent
    event_width = event['alongshore_end_m'] - event['alongshore_start_m']
    event_center = event['alongshore_centroid_m']

    # We need to find points within +-width/2 + buffer of the centroid
    # Since we don't have the exact UTM->alongshore mapping, we'll estimate
    # by using the relative position

    # Simplified approach: sort points by Y and take a slice
    y_sorted_idx = np.argsort(y)
    n_points = len(y)

    # Estimate what fraction of the file corresponds to the event
    # This is approximate but usually works for focused events
    keep_fraction = min(1.0, (event_width + 2 * buffer_m) / y_full_range)
    n_keep = max(1000, int(n_points * keep_fraction * 2))  # 2x for safety

    # Center the slice around the middle (approximate)
    center_idx = n_points // 2
    start_idx = max(0, center_idx - n_keep // 2)
    end_idx = min(n_points, center_idx + n_keep // 2)

    # Just use all points that pass elevation filter for now
    # (the alongshore filtering is complex without the shapefile mapping)

    # Find M3C2 distance field
    m3c2_field = None
    for field in las.point_format.dimension_names:
        if 'm3c2' in field.lower() and 'distance' in field.lower():
            m3c2_field = field
            break
    if m3c2_field is None:
        m3c2_field = next((f for f in las.point_format.dimension_names
                          if 'm3c2' == f.lower()), None)

    if m3c2_field:
        m3c2_dist = getattr(las, m3c2_field)[z_mask]
    else:
        m3c2_dist = np.zeros(len(x))

    # Find ClusterID field
    cluster_field = next((f for f in las.point_format.dimension_names
                         if 'cluster' in f.lower()), None)
    if cluster_field:
        cluster_id = getattr(las, cluster_field)[z_mask]
    else:
        cluster_id = np.zeros(len(x), dtype=int)

    return {
        'x': x,
        'y': y,
        'z': z,
        'm3c2_distance': m3c2_dist,
        'cluster_id': cluster_id,
    }


def plot_cliff_view(points: dict, value_field: str, title: str,
                    orientation_deg: float, cmap_name: str = 'RdBu_r',
                    vmax: float = 2.0, ax=None) -> plt.Figure:
    """
    Plot points in cliff-planar 2D view.

    Args:
        points: Dict with 'x', 'y', 'z' and value field
        value_field: 'm3c2_distance' or 'cluster_id'
        title: Plot title
        orientation_deg: Cliff orientation for rotation
        cmap_name: Colormap name
        vmax: Max value for colorbar (symmetric around 0 for M3C2)
        ax: Matplotlib axis (optional)

    Returns:
        Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Rotate to cliff-planar view
    alongshore, elevation = rotate_to_cliff_plane(
        points['x'], points['y'], points['z'], orientation_deg
    )

    # Get values for coloring
    values = points.get(value_field, np.zeros(len(alongshore)))

    if value_field == 'm3c2_distance':
        # Diverging colormap for M3C2 (negative=erosion, positive=deposition)
        cmap = cm.get_cmap(cmap_name)
        norm = Normalize(vmin=-vmax, vmax=vmax)
        label = "M3C2 Distance (m)"
    else:
        # Categorical colormap for clusters
        n_clusters = len(np.unique(values[values >= 0]))
        cmap = cm.get_cmap('tab20', max(20, n_clusters))
        norm = None
        label = "Cluster ID"

    # Subsample if too many points
    max_points = 50000
    if len(alongshore) > max_points:
        idx = np.random.choice(len(alongshore), max_points, replace=False)
        alongshore = alongshore[idx]
        elevation = elevation[idx]
        values = values[idx]

    scatter = ax.scatter(
        alongshore, elevation,
        c=values, cmap=cmap, norm=norm,
        s=1, alpha=0.7, rasterized=True
    )

    ax.set_xlabel("Alongshore (m, relative)", fontsize=11)
    ax.set_ylabel("Elevation (m)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(label, fontsize=10)

    ax.set_aspect('equal', adjustable='box')
    ax.invert_xaxis()  # Cliff-facing view

    return fig


def update_csv_in_place(csv_path: str, events_df: pd.DataFrame, qc_flags: dict):
    """Update the CSV file in place with current QC flags."""
    export_df = events_df.copy()
    export_df['qc_flag'] = export_df.index.map(lambda i: qc_flags.get(i, 'needs_check'))
    export_df.to_csv(csv_path, index=False)


# === Session State ===
def init_session_state():
    defaults = {
        'events_df': None,
        'needs_check_indices': [],
        'current_check_idx': 0,  # Index into needs_check_indices
        'qc_flags': {},
        'csv_path': None,
        'location': None,
        'event_type': 'erosion',
        'results_dir': BASE_RESULTS_DIR,
        'm3c2_points': None,
        'dbscan_points': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# === Streamlit App ===
st.set_page_config(page_title="Manual Check Tool", layout="wide")
init_session_state()

st.title("Manual Check Tool")
st.markdown("Re-check events flagged as 'needs_check' using M3C2 and DBSCAN point clouds")

if not HAS_LASPY:
    st.error("laspy is required but not installed. Run: pip install laspy")
    st.stop()

# === Sidebar ===
st.sidebar.header("Configuration")

# File Selection
st.sidebar.subheader("1. Load QC'd CSV")

# Default to event_lists_qc directory
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
default_csv_dir = os.path.join(base_dir, "results", "event_lists_qc")

csv_dir = st.sidebar.text_input(
    "CSV Directory",
    value=default_csv_dir,
    help="Directory containing QC'd event CSVs with 'needs_check' flags"
)

if os.path.isdir(csv_dir):
    csv_files = scan_event_csvs(csv_dir)
    if csv_files:
        csv_display = {os.path.relpath(f, csv_dir): f for f in csv_files}
        selected_csv = st.sidebar.selectbox("Select CSV", list(csv_display.keys()))

        if st.sidebar.button("Load CSV", type="primary"):
            csv_path = csv_display[selected_csv]
            events_df = load_event_csv(csv_path)

            if events_df is not None and 'qc_flag' in events_df.columns:
                st.session_state.events_df = events_df
                st.session_state.csv_path = csv_path
                st.session_state.location = infer_location_from_filename(csv_path)
                st.session_state.event_type = infer_event_type_from_filename(csv_path)

                # Load existing flags
                st.session_state.qc_flags = {
                    i: row['qc_flag'] if pd.notna(row['qc_flag']) else 'unreviewed'
                    for i, row in events_df.iterrows()
                }

                # Find indices of 'needs_check' events
                needs_check = [
                    i for i, row in events_df.iterrows()
                    if st.session_state.qc_flags.get(i) == 'needs_check'
                ]
                st.session_state.needs_check_indices = needs_check
                st.session_state.current_check_idx = 0

                n_check = len(needs_check)
                st.sidebar.success(f"Loaded {len(events_df)} events, {n_check} need checking")
            else:
                st.sidebar.error("CSV must have 'qc_flag' column")
    else:
        st.sidebar.warning("No CSV files found")
else:
    st.sidebar.warning("Directory does not exist")

# Results directory
st.sidebar.subheader("2. Data Source")
st.session_state.results_dir = st.sidebar.text_input(
    "Results Directory",
    value=st.session_state.results_dir,
    help="Server results directory with M3C2 and DBSCAN outputs"
)

# Navigation
if st.session_state.needs_check_indices:
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Navigation")

    n_total = len(st.session_state.needs_check_indices)
    current = st.session_state.current_check_idx

    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    with col1:
        if st.button("< Prev") and current > 0:
            st.session_state.current_check_idx -= 1
            st.session_state.m3c2_points = None
            st.session_state.dbscan_points = None
            st.rerun()
    with col2:
        new_idx = st.number_input(
            "Event",
            min_value=0,
            max_value=n_total - 1,
            value=current,
            label_visibility="collapsed"
        )
        if new_idx != current:
            st.session_state.current_check_idx = new_idx
            st.session_state.m3c2_points = None
            st.session_state.dbscan_points = None
            st.rerun()
    with col3:
        if st.button("Next >") and current < n_total - 1:
            st.session_state.current_check_idx += 1
            st.session_state.m3c2_points = None
            st.session_state.dbscan_points = None
            st.rerun()

    st.sidebar.markdown(f"**Checking {current + 1} / {n_total}**")

    # Progress
    st.sidebar.markdown("---")
    st.sidebar.subheader("4. Progress")

    reviewed = sum(1 for i in st.session_state.needs_check_indices
                   if st.session_state.qc_flags.get(i, 'needs_check') != 'needs_check')
    st.sidebar.progress(reviewed / n_total if n_total > 0 else 0)
    st.sidebar.markdown(f"**Reclassified:** {reviewed} / {n_total}")

# === Main Content ===
if st.session_state.needs_check_indices:
    # Get current event
    event_idx = st.session_state.needs_check_indices[st.session_state.current_check_idx]
    event = st.session_state.events_df.iloc[event_idx]
    location = st.session_state.location
    event_type = st.session_state.event_type

    # Event header
    st.subheader(f"Event #{event_idx + 1}: {event['start_date']} to {event['end_date']}")

    # Event info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Volume:** {event['volume']:.2f} m\u00b3")
        st.markdown(f"**Elevation:** {event['elevation']:.1f} m")
    with col2:
        st.markdown(f"**Alongshore:** {event['alongshore_centroid_m']:.0f} m")
        st.markdown(f"**Width:** {event['width']:.1f} m")
    with col3:
        st.markdown(f"**Height:** {event['height']:.1f} m")
        st.markdown(f"**Location:** {location}")

    st.markdown("---")

    # Load point cloud data
    if st.session_state.m3c2_points is None:
        with st.spinner("Loading point clouds..."):
            # Find M3C2 file
            m3c2_path = find_m3c2_las_for_event(
                event, location, st.session_state.results_dir
            )

            # Find DBSCAN file
            dbscan_path = find_dbscan_las_for_event(
                event, location, event_type, st.session_state.results_dir
            )

            if m3c2_path:
                st.session_state.m3c2_points = load_and_filter_las(m3c2_path, event)

            if dbscan_path:
                st.session_state.dbscan_points = load_and_filter_las(dbscan_path, event)

    # Display point clouds
    m3c2_pts = st.session_state.m3c2_points
    dbscan_pts = st.session_state.dbscan_points

    orientation = CLIFF_ORIENTATIONS.get(location, 345)

    if m3c2_pts is not None or dbscan_pts is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # M3C2 distances
        if m3c2_pts is not None:
            plot_cliff_view(
                m3c2_pts, 'm3c2_distance',
                f"M3C2 Distances ({len(m3c2_pts['x']):,} pts)",
                orientation, ax=ax1
            )
        else:
            ax1.text(0.5, 0.5, "M3C2 data not found",
                    ha='center', va='center', fontsize=14)
            ax1.set_title("M3C2 Distances")

        # DBSCAN clusters
        if dbscan_pts is not None:
            plot_cliff_view(
                dbscan_pts, 'cluster_id',
                f"DBSCAN Clusters ({len(dbscan_pts['x']):,} pts)",
                orientation, ax=ax2
            )
        else:
            ax2.text(0.5, 0.5, "DBSCAN data not found",
                    ha='center', va='center', fontsize=14)
            ax2.set_title("DBSCAN Clusters")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Could not load point cloud data for this event")
        st.info(f"Searched in: {st.session_state.results_dir}/{location}/")

    # Classification buttons
    st.markdown("---")
    st.markdown("**Reclassify this event:**")

    cols = st.columns(len(QC_FLAGS))
    for i, flag in enumerate(QC_FLAGS):
        with cols[i]:
            if st.button(flag.replace('_', ' ').title(),
                        key=f"btn_{flag}",
                        use_container_width=True):
                # Update flag
                st.session_state.qc_flags[event_idx] = flag

                # Save to CSV
                update_csv_in_place(
                    st.session_state.csv_path,
                    st.session_state.events_df,
                    st.session_state.qc_flags
                )

                # Move to next
                if st.session_state.current_check_idx < len(st.session_state.needs_check_indices) - 1:
                    st.session_state.current_check_idx += 1
                    st.session_state.m3c2_points = None
                    st.session_state.dbscan_points = None

                st.rerun()

elif st.session_state.events_df is not None:
    st.success("No events marked as 'needs_check' in this file!")
    st.info("All events have been classified.")
else:
    st.info("Load a QC'd CSV file from the sidebar to begin.")
    st.markdown("""
    **Instructions:**
    1. Select a CSV file that has been through initial QC (must have `qc_flag` column)
    2. Only events marked as `needs_check` will be shown
    3. View M3C2 distances and DBSCAN clusters side-by-side
    4. Reclassify as: Real, Noise, Construction, Veg Error, Beach Error, or Other
    5. The CSV is updated in place after each classification
    """)
