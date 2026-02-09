#!/usr/bin/env python3
"""
Manual Check Tool - Streamlit GUI for re-checking "needs_check" events

Displays DBSCAN clusters in 2D cliff-planar view.
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

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

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
    BASE_DIR = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
else:
    BASE_DIR = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

BASE_RESULTS_DIR = os.path.join(BASE_DIR, "results")
BASE_UTILITIES_DIR = os.path.join(BASE_DIR, "utilities")


@st.cache_data
def find_shapefile(location: str, resolution: str = "1m") -> str:
    """Locate the shapefile for a given location and resolution."""
    sf_root = os.path.join(BASE_UTILITIES_DIR, 'shape_files')
    if not os.path.isdir(sf_root):
        return None

    candidates = [
        d for d in os.listdir(sf_root)
        if d.lower().startswith(location.lower())
           and 'polygon' in d.lower()
           and f'at{resolution}'.lower() in d.lower()
           and os.path.isdir(os.path.join(sf_root, d))
    ]

    if not candidates:
        return None

    fld = candidates[0]
    shp_path = os.path.join(sf_root, fld, fld + '.shp')
    return shp_path if os.path.isfile(shp_path) else None


@st.cache_data
def load_shapefile_data(shp_path: str):
    """
    Load shapefile and compute cliff orientation for coordinate rotation.
    Also computes alongshore position mapping (Polygon_ID -> meters).

    Returns dict with: gdf, angle_rad, ref_point, alongshore_map, cell_width
    """
    if not HAS_GEOPANDAS or not shp_path:
        return None

    try:
        gdf = gpd.read_file(shp_path)

        # Use 0-based index as Polygon_ID
        gdf = gdf.reset_index(drop=True)
        gdf["Polygon_ID"] = gdf.index

        # Compute cliff orientation from polygon centroids
        centroids = gdf.geometry.centroid
        cx = centroids.x.values
        cy = centroids.y.values

        # Compute alongshore position for each polygon (in meters from min Y)
        # This matches how make_event_lists.py computes alongshore values
        min_y = cy.min()
        alongshore_m = cy - min_y  # Physical distance in meters
        gdf['alongshore_m'] = alongshore_m

        # Create mapping: Polygon_ID -> alongshore_m
        alongshore_map = dict(zip(gdf['Polygon_ID'], alongshore_m))

        # Estimate cell width from polygon spacing
        sorted_positions = np.sort(alongshore_m)
        if len(sorted_positions) > 1:
            spacings = np.diff(sorted_positions)
            cell_width = np.median(spacings)
        else:
            cell_width = 0.1  # fallback

        # Use first and last polygon centroids to define cliff direction
        # (direction of increasing Polygon_ID = alongshore direction)
        if len(cx) > 1:
            dx = cx[-1] - cx[0]
            dy = cy[-1] - cy[0]
            angle_rad = np.arctan2(dy, dx)
        else:
            angle_rad = 0

        # Reference point: first polygon centroid
        ref_x = cx[0]
        ref_y = cy[0]

        return {
            'gdf': gdf,
            'angle_rad': angle_rad,
            'ref_point': (ref_x, ref_y),
            'alongshore_map': alongshore_map,
            'cell_width': cell_width,
        }

    except Exception as e:
        st.error(f"Error loading shapefile: {e}")
        return None


def rotate_to_cliff_coords(x, y, angle_rad, ref_point):
    """
    Rotate UTM coordinates to cliff-aligned coordinates.

    Returns (alongshore, cross_shore) in meters where:
    - alongshore: distance along the cliff (parallel to cliff face)
    - cross_shore: distance perpendicular to cliff
    """
    ref_x, ref_y = ref_point

    # Translate to origin
    dx = x - ref_x
    dy = y - ref_y

    # Rotate by -angle to align cliff direction with X-axis
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)

    alongshore = dx * cos_a + dy * sin_a
    cross_shore = -dx * sin_a + dy * cos_a

    return alongshore, cross_shore


def assign_polygon_ids(points: dict, gdf) -> tuple:
    """
    Assign Polygon_ID and alongshore_m to each point via spatial join.

    Returns:
        tuple: (polygon_ids, alongshore_m) arrays
               - polygon_ids: integer index of polygon (NaN for points outside)
               - alongshore_m: physical alongshore distance in meters (NaN for points outside)
    """
    if not HAS_GEOPANDAS or gdf is None:
        return None, None

    # Ensure alongshore_m column exists
    if 'alongshore_m' not in gdf.columns:
        centroids = gdf.geometry.centroid
        min_y = centroids.y.min()
        gdf = gdf.copy()
        gdf['alongshore_m'] = centroids.y - min_y

    # Create point geometries with explicit index
    points_gdf = gpd.GeoDataFrame(
        {'point_idx': np.arange(len(points['x']))},
        geometry=gpd.points_from_xy(points['x'], points['y']),
        crs=gdf.crs
    )

    # Spatial join - include both Polygon_ID and alongshore_m
    joined = gpd.sjoin(points_gdf, gdf[['Polygon_ID', 'alongshore_m', 'geometry']],
                       how='left', predicate='within')

    # Remove duplicates caused by overlapping polygons (keep first match only)
    joined = joined[~joined.index.duplicated(keep='first')]

    # Ensure output arrays match input length
    n_pts = len(points['x'])
    result_pids = np.full(n_pts, np.nan)
    result_along = np.full(n_pts, np.nan)

    result_pids[joined.index] = joined['Polygon_ID'].values
    result_along[joined.index] = joined['alongshore_m'].values

    return result_pids, result_along


def assign_polygon_ids_for_event(points: dict, gdf, event: pd.Series,
                                  buffer_m: float = 10.0,
                                  progress_bar=None) -> tuple:
    """
    Optimized polygon ID assignment - only spatial joins points near the event.

    1. Gets UTM bounding box from polygons in the event's alongshore range (meters)
    2. Pre-filters points using fast numpy comparisons
    3. Spatial joins only the filtered subset
    4. Returns full arrays with NaN for points outside event region

    This is much faster than joining millions of points.

    NOTE: Event CSV stores alongshore_start_m and alongshore_end_m as PHYSICAL
    DISTANCES in meters (computed from UTM Y - min_Y), NOT Polygon_ID indices.
    We must filter using gdf['alongshore_m'], not gdf['Polygon_ID'].

    Returns:
        tuple: (polygon_ids, alongshore_m) arrays
    """
    if not HAS_GEOPANDAS or gdf is None:
        return None, None

    n_pts = len(points['x'])
    result_pid = np.full(n_pts, np.nan)
    result_along = np.full(n_pts, np.nan)

    if progress_bar:
        progress_bar.progress(96, text="Computing event bounding box...")

    # Get event's alongshore range (PHYSICAL METERS, not Polygon_ID!)
    along_min = event['alongshore_start_m'] - buffer_m
    along_max = event['alongshore_end_m'] + buffer_m
    z_min = max(0, event['elevation'] - event['height'] / 2 - buffer_m)
    z_max = event['elevation'] + event['height'] / 2 + buffer_m

    # Get polygons in the event's alongshore range using PHYSICAL distance
    # The 'alongshore_m' column was computed as (UTM_Y - min_Y) in load_shapefile_data()
    if 'alongshore_m' not in gdf.columns:
        # Fallback: compute it now (shouldn't happen with updated load_shapefile_data)
        centroids = gdf.geometry.centroid
        min_y = centroids.y.min()
        gdf = gdf.copy()
        gdf['alongshore_m'] = centroids.y - min_y

    event_polygons = gdf[(gdf['alongshore_m'] >= along_min) &
                         (gdf['alongshore_m'] <= along_max)]

    if len(event_polygons) == 0:
        return result_pid, result_along

    # Get UTM bounding box from those polygons
    bounds = event_polygons.total_bounds  # [minx, miny, maxx, maxy]
    utm_x_min, utm_y_min, utm_x_max, utm_y_max = bounds

    # Add buffer to UTM bounds
    utm_x_min -= buffer_m
    utm_x_max += buffer_m
    utm_y_min -= buffer_m
    utm_y_max += buffer_m

    if progress_bar:
        progress_bar.progress(97, text="Pre-filtering points by bounding box...")

    # Pre-filter points using fast numpy comparisons
    x, y, z = points['x'], points['y'], points['z']
    in_bbox = ((x >= utm_x_min) & (x <= utm_x_max) &
               (y >= utm_y_min) & (y <= utm_y_max) &
               (z >= z_min) & (z <= z_max))

    bbox_indices = np.where(in_bbox)[0]
    n_bbox = len(bbox_indices)

    if n_bbox == 0:
        return result_pid, result_along

    if progress_bar:
        progress_bar.progress(98, text=f"Spatial join on {n_bbox:,} points (filtered from {n_pts:,})...")

    # Create GeoDataFrame only for filtered points
    filtered_gdf = gpd.GeoDataFrame(
        {'orig_idx': bbox_indices},
        geometry=gpd.points_from_xy(x[bbox_indices], y[bbox_indices]),
        crs=gdf.crs
    )

    # Spatial join only on filtered subset - include alongshore_m
    joined = gpd.sjoin(filtered_gdf, event_polygons[['Polygon_ID', 'alongshore_m', 'geometry']],
                       how='left', predicate='within')

    if progress_bar:
        progress_bar.progress(99, text="Mapping polygon IDs and alongshore...")

    # Map results back to full arrays (vectorized)
    valid_mask = joined['Polygon_ID'].notna()
    valid_indices = joined.loc[valid_mask, 'orig_idx'].values.astype(int)
    valid_pids = joined.loc[valid_mask, 'Polygon_ID'].values
    valid_along = joined.loc[valid_mask, 'alongshore_m'].values

    result_pid[valid_indices] = valid_pids
    result_along[valid_indices] = valid_along

    return result_pid, result_along


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
    """Find the M3C2 LAS file for a given event."""
    if results_dir is None:
        results_dir = BASE_RESULTS_DIR

    date_folder = event_dates_to_folder(event['start_date'], event['end_date'])
    m3c2_base = os.path.join(results_dir, location, "m3c2")

    pipeline_run = find_latest_pipeline_run(m3c2_base)
    if not pipeline_run:
        return None

    m3c2_dir = os.path.join(pipeline_run, date_folder)
    if not os.path.isdir(m3c2_dir):
        return None

    pattern = os.path.join(m3c2_dir, "*_m3c2.las")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def find_dbscan_las_for_event(event: pd.Series, location: str, event_type: str,
                               results_dir: str = None) -> str:
    """Find the DBSCAN clusters LAS file for a given event."""
    if results_dir is None:
        results_dir = BASE_RESULTS_DIR

    date_folder = event_dates_to_folder(event['start_date'], event['end_date'])
    prefix = 'ero' if event_type == 'erosion' else 'dep'

    clusters_path = os.path.join(
        results_dir, location, event_type, date_folder,
        f"{prefix}_clusters.las"
    )

    return clusters_path if os.path.exists(clusters_path) else None


def find_both_dbscan_files(event: pd.Series, location: str,
                           results_dir: str = None) -> dict:
    """
    Find both erosion and deposition DBSCAN files for the same time period.
    Returns dict with 'erosion' and 'deposition' paths (or None if not found).
    """
    if results_dir is None:
        results_dir = BASE_RESULTS_DIR

    date_folder = event_dates_to_folder(event['start_date'], event['end_date'])

    paths = {}
    for etype, prefix in [('erosion', 'ero'), ('deposition', 'dep')]:
        clusters_path = os.path.join(
            results_dir, location, etype, date_folder,
            f"{prefix}_clusters.las"
        )
        paths[etype] = clusters_path if os.path.exists(clusters_path) else None

    return paths


def load_las_simple(las_path: str) -> dict:
    """
    Load entire LAS file into memory (simple approach).

    Returns dict with x, y, z, m3c2_distance, cluster_id arrays.
    """
    if not HAS_LASPY:
        return None

    try:
        las = laspy.read(las_path)
    except Exception as e:
        st.error(f"Error reading {las_path}: {e}")
        return None

    result = {
        'x': np.array(las.x),
        'y': np.array(las.y),
        'z': np.array(las.z),
    }

    # Find M3C2 distance field
    dim_names = [d.name for d in las.point_format.dimensions]
    m3c2_field = None
    for field in dim_names:
        if 'm3c2' in field.lower() and 'distance' in field.lower():
            m3c2_field = field
            break
    if m3c2_field is None:
        m3c2_field = next((f for f in dim_names if 'm3c2' == f.lower()), None)

    if m3c2_field:
        result['m3c2_distance'] = np.array(getattr(las, m3c2_field))
    else:
        result['m3c2_distance'] = np.zeros(len(result['x']))

    # Find ClusterID field
    cluster_field = next((f for f in dim_names if 'cluster' in f.lower()), None)
    if cluster_field:
        result['cluster_id'] = np.array(getattr(las, cluster_field))
    else:
        result['cluster_id'] = np.zeros(len(result['x']), dtype=int)

    return result


def load_las_with_progress(las_path: str, progress_bar=None) -> dict:
    """
    Load LAS file with progress reporting using chunked reading.

    Returns dict with x, y, z, m3c2_distance, cluster_id arrays.
    """
    if not HAS_LASPY:
        return None

    try:
        # Get file info first
        file_size_mb = os.path.getsize(las_path) / (1024 * 1024)
        if progress_bar:
            progress_bar.progress(0, text=f"Opening file ({file_size_mb:.1f} MB)...")

        # Open file and get total points
        with laspy.open(las_path) as las_file:
            total_points = las_file.header.point_count

            if progress_bar:
                progress_bar.progress(5, text=f"Reading {total_points:,} points...")

            # Read in chunks
            chunk_size = 1_000_000  # 1M points per chunk
            x_chunks, y_chunks, z_chunks = [], [], []
            m3c2_chunks, cluster_chunks = [], []

            m3c2_field = None
            cluster_field = None
            points_read = 0

            for chunk in las_file.chunk_iterator(chunk_size):
                # First chunk: identify field names
                if m3c2_field is None:
                    dim_names = [d.name for d in chunk.point_format.dimensions]
                    for field in dim_names:
                        if 'm3c2' in field.lower() and 'distance' in field.lower():
                            m3c2_field = field
                            break
                    if m3c2_field is None:
                        m3c2_field = next((f for f in dim_names if 'm3c2' == f.lower()), None)
                    cluster_field = next((f for f in dim_names if 'cluster' in f.lower()), None)

                # Extract coordinates
                x_chunks.append(np.array(chunk.x))
                y_chunks.append(np.array(chunk.y))
                z_chunks.append(np.array(chunk.z))

                # Extract M3C2 distance
                if m3c2_field:
                    m3c2_chunks.append(np.array(getattr(chunk, m3c2_field)))
                else:
                    m3c2_chunks.append(np.zeros(len(chunk)))

                # Extract cluster ID
                if cluster_field:
                    cluster_chunks.append(np.array(getattr(chunk, cluster_field)))
                else:
                    cluster_chunks.append(np.zeros(len(chunk), dtype=int))

                points_read += len(chunk)
                if progress_bar:
                    pct = min(5 + int(85 * points_read / total_points), 90)
                    progress_bar.progress(pct, text=f"Read {points_read:,} / {total_points:,} points...")

        if progress_bar:
            progress_bar.progress(92, text="Concatenating arrays...")

        result = {
            'x': np.concatenate(x_chunks),
            'y': np.concatenate(y_chunks),
            'z': np.concatenate(z_chunks),
            'm3c2_distance': np.concatenate(m3c2_chunks),
            'cluster_id': np.concatenate(cluster_chunks),
        }

        if progress_bar:
            progress_bar.progress(100, text=f"Loaded {len(result['x']):,} points")

        return result

    except Exception as e:
        st.error(f"Error reading {las_path}: {e}")
        return None


def plot_dbscan_view(points: dict, event: pd.Series,
                      alongshore_m: np.ndarray = None,
                      alongshore_coords: np.ndarray = None,
                      ax=None, buffer_m: float = 5) -> plt.Figure:
    """
    Plot DBSCAN clusters in 2D cliff-facing view (rotated alongshore vs elevation).

    Uses alongshore_m (physical distance) to filter to event region,
    plots using rotated coordinates.

    Args:
        points: dict with x, y, z, m3c2_distance arrays
        event: pandas Series with event info
        alongshore_m: physical alongshore distance in meters for each point (for filtering)
        alongshore_coords: rotated alongshore coordinate for each point (for plotting)
        buffer_m: buffer around event extent in meters
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = ax.figure

    z = points['z']
    m3c2 = points['m3c2_distance']
    n_pts_total = len(z)

    # Use rotated alongshore for plotting
    if alongshore_coords is not None:
        alongshore = alongshore_coords
    else:
        # Fallback: use Y coordinate
        alongshore = points['y'] - np.nanmean(points['y'])

    # Filter to event extent using alongshore_m (physical meters from event CSV)
    if event is not None and alongshore_m is not None:
        along_min_filter = event['alongshore_start_m'] - buffer_m
        along_max_filter = event['alongshore_end_m'] + buffer_m
        z_min = max(0, event['elevation'] - event['height'] / 2 - buffer_m)
        z_max = event['elevation'] + event['height'] / 2 + buffer_m

        # Filter using physical alongshore distance (meters)
        valid_along = ~np.isnan(alongshore_m)
        in_region = valid_along & \
                    (alongshore_m >= along_min_filter) & \
                    (alongshore_m <= along_max_filter) & \
                    (z >= z_min) & (z <= z_max)

        # Extract filtered points using rotated coords for plotting
        alongshore_plot = alongshore[in_region]
        z_plot = z[in_region]
        m3c2_plot = m3c2[in_region]
        n_pts = len(alongshore_plot)
    else:
        alongshore_plot = alongshore
        z_plot = z
        m3c2_plot = m3c2
        n_pts = n_pts_total

    if n_pts == 0:
        ax.text(0.5, 0.5, "No points in event region",
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title("DBSCAN Clusters")
        return fig

    # Use M3C2 distance for coloring with diverging colormap
    cmap = cm.get_cmap('RdBu_r')
    norm = Normalize(vmin=-2.0, vmax=2.0)

    # Subsample if too many points
    max_points = 100000
    if n_pts > max_points:
        idx = np.random.choice(n_pts, max_points, replace=False)
        alongshore_plot = alongshore_plot[idx]
        z_plot = z_plot[idx]
        m3c2_plot = m3c2_plot[idx]

    # Larger points with black edge for visibility
    scatter = ax.scatter(
        alongshore_plot, z_plot,
        c=m3c2_plot, cmap=cmap, norm=norm,
        s=25, alpha=0.9, edgecolors='black', linewidths=0.5, rasterized=True
    )

    ax.set_xlabel("Alongshore (m, rotated)", fontsize=11)
    ax.set_ylabel("Elevation (m)", fontsize=11)
    ax.set_title(f"DBSCAN Clusters ({n_pts:,} of {n_pts_total:,} pts) - Cliff-Facing View",
                 fontsize=12, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("M3C2 Distance (m)", fontsize=10)

    # Set view limits based on filtered points' bounding box (ensures all points visible)
    if n_pts > 0:
        # Get bounding box of filtered points
        x_min, x_max = alongshore_plot.min(), alongshore_plot.max()
        z_min_pts, z_max_pts = z_plot.min(), z_plot.max()

        # Calculate range with small minimum to avoid divide-by-zero
        x_range = max(x_max - x_min, 1)
        z_range = max(z_max_pts - z_min_pts, 1)

        # Add padding: 10% of range, minimum 1m
        x_pad = max(x_range * 0.1, 1)
        z_pad = max(z_range * 0.1, 1)

        # Set limits with padding (X inverted for cliff-facing view)
        ax.set_xlim(x_max + x_pad, x_min - x_pad)
        ax.set_ylim(z_min_pts - z_pad, z_max_pts + z_pad)

        # Draw crosshair at actual centroid of filtered points
        centroid_along = np.median(alongshore_plot)
        centroid_elev = np.median(z_plot)  # Use actual point median
        event_elev = event['elevation'] if event is not None else centroid_elev

        # Crosshair at actual point centroid
        ax.axhline(centroid_elev, color='lime', linestyle='--',
                   linewidth=1.5, alpha=0.7)
        ax.axvline(centroid_along, color='lime', linestyle='--',
                   linewidth=1.5, alpha=0.7)
        ax.plot(centroid_along, centroid_elev, 'g+', markersize=15, markeredgewidth=2,
                label=f"Points centroid: {centroid_elev:.1f}m")

        # Also show event's recorded elevation if different
        if event is not None and abs(event_elev - centroid_elev) > 0.5:
            ax.axhline(event_elev, color='orange', linestyle=':',
                       linewidth=1.5, alpha=0.7, label=f"Event CSV elev: {event_elev:.1f}m")

        ax.legend(loc='upper right', fontsize=9)

    return fig


def plot_m3c2_view(points: dict, event: pd.Series,
                    alongshore_m: np.ndarray = None,
                    alongshore_coords: np.ndarray = None,
                    view_extent: dict = None,
                    ax=None, buffer_m: float = 5) -> plt.Figure:
    """
    Plot M3C2 point cloud in 2D cliff-facing view (rotated alongshore vs elevation).

    Args:
        points: dict with x, y, z, m3c2_distance arrays
        event: pandas Series with event info
        alongshore_m: physical alongshore distance in meters for each point (for filtering)
        alongshore_coords: rotated alongshore coordinate for each point (for plotting)
        view_extent: dict with x_min, x_max, z_min, z_max to match DBSCAN view
        buffer_m: buffer around event extent in meters
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = ax.figure

    z = points['z']
    m3c2 = points['m3c2_distance']
    n_pts_total = len(z)

    # Use rotated alongshore for plotting
    if alongshore_coords is not None:
        alongshore = alongshore_coords
    else:
        # Fallback: use Y coordinate
        alongshore = points['y'] - np.nanmean(points['y'])

    # Filter to event extent using alongshore_m (physical meters from event CSV)
    if event is not None and alongshore_m is not None:
        along_min_filter = event['alongshore_start_m'] - buffer_m
        along_max_filter = event['alongshore_end_m'] + buffer_m
        z_min = max(0, event['elevation'] - event['height'] / 2 - buffer_m)
        z_max = event['elevation'] + event['height'] / 2 + buffer_m

        # Filter using physical alongshore distance (meters)
        valid_along = ~np.isnan(alongshore_m)
        in_region = valid_along & \
                    (alongshore_m >= along_min_filter) & \
                    (alongshore_m <= along_max_filter) & \
                    (z >= z_min) & (z <= z_max)

        # Extract filtered points using rotated coords for plotting
        alongshore_plot = alongshore[in_region]
        z_plot = z[in_region]
        m3c2_plot = m3c2[in_region]
        n_pts = len(alongshore_plot)
    else:
        alongshore_plot = alongshore
        z_plot = z
        m3c2_plot = m3c2
        n_pts = n_pts_total

    if n_pts == 0:
        ax.text(0.5, 0.5, "No M3C2 points in event region",
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title("M3C2 Point Cloud")
        return fig

    # Use M3C2 distance for coloring with diverging colormap
    cmap = cm.get_cmap('RdBu_r')
    norm = Normalize(vmin=-2.0, vmax=2.0)

    # Subsample if too many points
    max_points = 100000
    if n_pts > max_points:
        idx = np.random.choice(n_pts, max_points, replace=False)
        alongshore_plot = alongshore_plot[idx]
        z_plot = z_plot[idx]
        m3c2_plot = m3c2_plot[idx]

    # Smaller points for M3C2 (many more points) but still visible
    scatter = ax.scatter(
        alongshore_plot, z_plot,
        c=m3c2_plot, cmap=cmap, norm=norm,
        s=4, alpha=0.7, rasterized=True
    )

    ax.set_xlabel("Alongshore (m, rotated)", fontsize=11)
    ax.set_ylabel("Elevation (m)", fontsize=11)
    ax.set_title(f"M3C2 ({n_pts:,} of {n_pts_total:,} pts) - Cliff-Facing View",
                 fontsize=12, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("M3C2 Distance (m)", fontsize=10)

    # Use view_extent if provided (to match DBSCAN zoom), otherwise compute from points
    if view_extent is not None:
        ax.set_xlim(view_extent['x_max'], view_extent['x_min'])  # X inverted for cliff-facing
        ax.set_ylim(view_extent['z_min'], view_extent['z_max'])
    elif n_pts > 0:
        # Compute from filtered points
        x_min, x_max = alongshore_plot.min(), alongshore_plot.max()
        z_min_pts, z_max_pts = z_plot.min(), z_plot.max()
        x_range = max(x_max - x_min, 1)
        z_range = max(z_max_pts - z_min_pts, 1)
        x_pad = max(x_range * 0.1, 1)
        z_pad = max(z_range * 0.1, 1)
        ax.set_xlim(x_max + x_pad, x_min - x_pad)
        ax.set_ylim(z_min_pts - z_pad, z_max_pts + z_pad)

    # Draw crosshair at actual centroid of filtered points
    if n_pts > 0:
        centroid_along = np.median(alongshore_plot)
        centroid_elev = np.median(z_plot)  # Use actual point median
        event_elev = event['elevation'] if event is not None else centroid_elev

        # Crosshair at actual point centroid
        ax.axhline(centroid_elev, color='lime', linestyle='--',
                   linewidth=1.5, alpha=0.7)
        ax.axvline(centroid_along, color='lime', linestyle='--',
                   linewidth=1.5, alpha=0.7)
        ax.plot(centroid_along, centroid_elev, 'g+', markersize=15, markeredgewidth=2,
                label=f"Points centroid: {centroid_elev:.1f}m")

        # Also show event's recorded elevation if different
        if event is not None and abs(event_elev - centroid_elev) > 0.5:
            ax.axhline(event_elev, color='orange', linestyle=':',
                       linewidth=1.5, alpha=0.7, label=f"Event CSV elev: {event_elev:.1f}m")

        ax.legend(loc='upper right', fontsize=9)

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
        'current_check_idx': 0,
        'qc_flags': {},
        'csv_path': None,
        'location': None,
        'event_type': 'erosion',
        'results_dir': BASE_RESULTS_DIR,
        'dbscan_points': None,
        'polygon_ids': None,
        'alongshore_m': None,  # Physical alongshore distance in meters (for filtering)
        'alongshore_coords': None,
        'cross_shore_coords': None,
        'shapefile_data': None,
        'shp_location': None,
        'm3c2_points': None,
        'm3c2_polygon_ids': None,
        'm3c2_alongshore_m': None,  # Physical alongshore distance in meters (for filtering)
        'm3c2_alongshore_coords': None,
        'm3c2_cross_shore_coords': None,
        'show_m3c2': False,
        'dbscan_view_extent': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# === Streamlit App ===
st.set_page_config(page_title="Manual Check Tool", layout="wide")
init_session_state()

st.title("Manual Check Tool")
st.markdown("Re-check events flagged as 'needs_check' using DBSCAN point clouds")

if not HAS_LASPY:
    st.error("laspy is required but not installed. Run: pip install laspy")
    st.stop()

# === Sidebar ===
st.sidebar.header("Configuration")

# File Selection
st.sidebar.subheader("1. Load QC'd CSV")

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
default_csv_dir = os.path.join(base_dir, "results", "event_lists_qc")

csv_dir = st.sidebar.text_input(
    "CSV Directory",
    value=default_csv_dir,
    help="Directory containing QC'd event CSVs"
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
                st.session_state.dbscan_points = None

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
    help="Server results directory with DBSCAN outputs"
)

# Navigation
if st.session_state.needs_check_indices:
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Navigation")

    n_total = len(st.session_state.needs_check_indices)
    current = st.session_state.current_check_idx

    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    def reset_point_data():
        """Reset all point cloud data when changing events."""
        st.session_state.dbscan_points = None
        st.session_state.polygon_ids = None
        st.session_state.alongshore_m = None
        st.session_state.alongshore_coords = None
        st.session_state.cross_shore_coords = None
        st.session_state.m3c2_points = None
        st.session_state.m3c2_polygon_ids = None
        st.session_state.m3c2_alongshore_m = None
        st.session_state.m3c2_alongshore_coords = None
        st.session_state.m3c2_cross_shore_coords = None
        st.session_state.show_m3c2 = False
        st.session_state.dbscan_view_extent = None

    with col1:
        if st.button("< Prev") and current > 0:
            st.session_state.current_check_idx -= 1
            reset_point_data()
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
            reset_point_data()
            st.rerun()
    with col3:
        if st.button("Next >") and current < n_total - 1:
            st.session_state.current_check_idx += 1
            reset_point_data()
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
        st.markdown(f"**Volume:** {event['volume']:.2f} m³")
        st.markdown(f"**Elevation:** {event['elevation']:.1f} m")
    with col2:
        st.markdown(f"**Alongshore:** {event['alongshore_centroid_m']:.0f} m")
        st.markdown(f"**Width:** {event['width']:.1f} m")
    with col3:
        st.markdown(f"**Height:** {event['height']:.1f} m")
        st.markdown(f"**Location:** {location}")

    st.markdown("---")

    # Load shapefile for coordinate mapping (cached)
    if 'shapefile_data' not in st.session_state or st.session_state.get('shp_location') != location:
        shp_path = find_shapefile(location, "1m")
        if shp_path:
            shp_data = load_shapefile_data(shp_path)
            st.session_state.shapefile_data = shp_data
            st.session_state.shp_location = location
        else:
            st.session_state.shapefile_data = None
            st.warning(f"Shapefile not found for {location}")

    shp_data = st.session_state.get('shapefile_data')

    # Debug: Show rotation diagnostics
    if shp_data is not None:
        with st.expander("Rotation Diagnostics", expanded=False):
            gdf = shp_data['gdf']
            angle_rad = shp_data['angle_rad']
            ref_point = shp_data['ref_point']
            angle_deg = np.degrees(angle_rad)

            # Get first and last polygon info
            centroids = gdf.geometry.centroid
            first_centroid = (centroids.iloc[0].x, centroids.iloc[0].y)
            last_centroid = (centroids.iloc[-1].x, centroids.iloc[-1].y)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Shapefile Info:**")
                st.markdown(f"- Polygons: {len(gdf)}")
                st.markdown(f"- First centroid (UTM): ({first_centroid[0]:.1f}, {first_centroid[1]:.1f})")
                st.markdown(f"- Last centroid (UTM): ({last_centroid[0]:.1f}, {last_centroid[1]:.1f})")
            with col2:
                st.markdown("**Rotation:**")
                st.markdown(f"- Cliff angle: {angle_deg:.1f}° from East")
                st.markdown(f"- Reference point: ({ref_point[0]:.1f}, {ref_point[1]:.1f})")

            # Show event's alongshore range (PHYSICAL METERS, not Polygon_ID!)
            along_min = event['alongshore_start_m']
            along_max = event['alongshore_end_m']

            # Use alongshore_m for filtering (physical distance), NOT Polygon_ID
            event_polygons = gdf[(gdf['alongshore_m'] >= along_min) & (gdf['alongshore_m'] <= along_max)]

            st.markdown(f"**Event Alongshore Range:** {along_min:.1f}m - {along_max:.1f}m (physical distance)")
            st.markdown(f"**Matching Polygons:** {len(event_polygons)} polygons")

            if len(event_polygons) > 0:
                bounds = event_polygons.total_bounds
                pid_min = event_polygons['Polygon_ID'].min()
                pid_max = event_polygons['Polygon_ID'].max()
                st.markdown(f"**Polygon_ID Range:** {pid_min} - {pid_max} (0-based indices)")
                st.markdown(f"**UTM Bounds:** X=[{bounds[0]:.1f}, {bounds[2]:.1f}], Y=[{bounds[1]:.1f}, {bounds[3]:.1f}]")

                # Find polygon closest to event center
                center_along = event['alongshore_centroid_m']
                closest_idx = (gdf['alongshore_m'] - center_along).abs().idxmin()
                center_polygon = gdf.loc[closest_idx]
                center_geom = center_polygon.geometry.centroid
                rot_along, rot_cross = rotate_to_cliff_coords(
                    center_geom.x, center_geom.y, angle_rad, ref_point
                )
                st.markdown(f"**Event center (alongshore={center_along:.1f}m):**")
                st.markdown(f"- Closest Polygon_ID: {center_polygon['Polygon_ID']}")
                st.markdown(f"- UTM: ({center_geom.x:.1f}, {center_geom.y:.1f})")
                st.markdown(f"- Rotated: alongshore={rot_along:.1f}m, cross-shore={rot_cross:.1f}m")

    # Load DBSCAN files (both erosion and deposition for context)
    if st.session_state.dbscan_points is None:
        dbscan_paths = find_both_dbscan_files(
            event, location, st.session_state.results_dir
        )

        # Load both erosion and deposition if available
        all_points = {'x': [], 'y': [], 'z': [], 'm3c2_distance': [], 'cluster_id': []}
        loaded_files = []

        with st.spinner("Loading DBSCAN clusters..."):
            for etype, path in dbscan_paths.items():
                if path:
                    pts = load_las_simple(path)
                    if pts:
                        for key in all_points:
                            all_points[key].append(pts[key])
                        loaded_files.append(f"{etype}: {len(pts['x']):,} pts")

        if loaded_files:
            # Combine all points
            combined = {
                key: np.concatenate(all_points[key]) if all_points[key] else np.array([])
                for key in all_points
            }
            st.session_state.dbscan_points = combined
            n_pts = len(combined['x'])
            st.success(f"Loaded {n_pts:,} points ({', '.join(loaded_files)})")

            if shp_data is not None:
                with st.spinner("Computing cliff-facing coordinates..."):
                    # Get Polygon_IDs and alongshore_m (physical meters) for filtering
                    polygon_ids, alongshore_m = assign_polygon_ids(
                        st.session_state.dbscan_points, shp_data['gdf']
                    )
                    st.session_state.polygon_ids = polygon_ids
                    st.session_state.alongshore_m = alongshore_m

                    # Rotate UTM to cliff-aligned coordinates for plotting
                    alongshore, cross_shore = rotate_to_cliff_coords(
                        st.session_state.dbscan_points['x'],
                        st.session_state.dbscan_points['y'],
                        shp_data['angle_rad'],
                        shp_data['ref_point']
                    )
                    st.session_state.alongshore_coords = alongshore
                    st.session_state.cross_shore_coords = cross_shore

                valid_count = np.sum(~np.isnan(polygon_ids))
                st.caption(f"Mapped {valid_count:,} of {n_pts:,} points")
            else:
                st.session_state.polygon_ids = None
                st.session_state.alongshore_coords = None
                st.session_state.cross_shore_coords = None
        else:
            st.warning("No DBSCAN cluster files found for this date range")

    # Display DBSCAN plot
    dbscan_pts = st.session_state.dbscan_points
    alongshore_m = st.session_state.get('alongshore_m')  # Physical meters for filtering
    alongshore_coords = st.session_state.get('alongshore_coords')  # Rotated coords for plotting

    if dbscan_pts is not None:
        fig = plot_dbscan_view(dbscan_pts, event, alongshore_m, alongshore_coords)

        # Capture view extent from the DBSCAN plot for M3C2 to match
        ax = fig.axes[0]
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        # Note: x_lim is inverted for cliff-facing view (x_lim[0] > x_lim[1])
        st.session_state.dbscan_view_extent = {
            'x_min': min(x_lim),
            'x_max': max(x_lim),
            'z_min': y_lim[0],
            'z_max': y_lim[1],
        }

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Could not load point cloud data")
        st.info(f"Searched in: {st.session_state.results_dir}/{location}/{event_type}/")

    # === M3C2 Section ===
    st.markdown("---")

    # Load M3C2 button
    if st.button("Load M3C2 Point Cloud", type="secondary", use_container_width=False):
        st.session_state.show_m3c2 = True
        st.session_state.m3c2_points = None  # Force reload
        st.rerun()

    # Display M3C2 panel if toggled
    if st.session_state.show_m3c2:
        # Load M3C2 data if not already loaded
        if st.session_state.m3c2_points is None:
            m3c2_path = find_m3c2_las_for_event(event, location, st.session_state.results_dir)

            if m3c2_path:
                st.caption(f"Loading: {os.path.basename(m3c2_path)}")
                progress_bar = st.progress(0, text="Initializing...")

                m3c2_pts = load_las_with_progress(m3c2_path, progress_bar)

                if m3c2_pts:
                    st.session_state.m3c2_points = m3c2_pts
                    n_pts = len(m3c2_pts['x'])

                    # Compute rotated coordinates and polygon IDs for M3C2
                    if shp_data is not None:
                        # Use optimized spatial join - only joins points near event
                        # Returns both polygon_ids and alongshore_m (physical meters)
                        m3c2_polygon_ids, m3c2_alongshore_m = assign_polygon_ids_for_event(
                            m3c2_pts, shp_data['gdf'], event,
                            buffer_m=10.0, progress_bar=progress_bar
                        )
                        st.session_state.m3c2_polygon_ids = m3c2_polygon_ids
                        st.session_state.m3c2_alongshore_m = m3c2_alongshore_m

                        # Rotate coordinates (vectorized, fast) for plotting
                        m3c2_alongshore_coords, m3c2_cross_shore = rotate_to_cliff_coords(
                            m3c2_pts['x'],
                            m3c2_pts['y'],
                            shp_data['angle_rad'],
                            shp_data['ref_point']
                        )
                        st.session_state.m3c2_alongshore_coords = m3c2_alongshore_coords
                        st.session_state.m3c2_cross_shore_coords = m3c2_cross_shore

                        # Count how many points were mapped
                        n_mapped = np.sum(~np.isnan(m3c2_alongshore_m))
                        progress_bar.progress(100, text=f"Done - {n_mapped:,} points in event region")
                        st.success(f"Loaded M3C2: {n_pts:,} total, {n_mapped:,} in event region")
                    else:
                        progress_bar.progress(100, text=f"Done - {n_pts:,} points loaded")
                        st.success(f"Loaded M3C2: {n_pts:,} points")
                else:
                    st.error("Failed to load M3C2 file")
            else:
                st.warning("M3C2 file not found for this event")
                st.caption(f"Searched in: {st.session_state.results_dir}/{location}/m3c2/pipeline_run_*/")

        # Plot M3C2 if loaded
        m3c2_pts = st.session_state.m3c2_points
        m3c2_alongshore_m = st.session_state.get('m3c2_alongshore_m')  # Physical meters for filtering
        m3c2_alongshore_coords = st.session_state.get('m3c2_alongshore_coords')  # Rotated for plotting
        view_extent = st.session_state.get('dbscan_view_extent')

        if m3c2_pts is not None:
            st.markdown("**M3C2 Point Cloud (zoomed to DBSCAN extent)**")
            fig_m3c2 = plot_m3c2_view(
                m3c2_pts, event,
                m3c2_alongshore_m, m3c2_alongshore_coords,
                view_extent=view_extent
            )
            st.pyplot(fig_m3c2)
            plt.close(fig_m3c2)

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

                # Move to next and reset all point data
                if st.session_state.current_check_idx < len(st.session_state.needs_check_indices) - 1:
                    st.session_state.current_check_idx += 1
                    st.session_state.dbscan_points = None
                    st.session_state.polygon_ids = None
                    st.session_state.alongshore_m = None
                    st.session_state.alongshore_coords = None
                    st.session_state.cross_shore_coords = None
                    st.session_state.m3c2_points = None
                    st.session_state.m3c2_polygon_ids = None
                    st.session_state.m3c2_alongshore_m = None
                    st.session_state.m3c2_alongshore_coords = None
                    st.session_state.m3c2_cross_shore_coords = None
                    st.session_state.show_m3c2 = False
                    st.session_state.dbscan_view_extent = None

                st.rerun()

elif st.session_state.events_df is not None:
    st.success("No events marked as 'needs_check' in this file!")
else:
    st.info("Load a QC'd CSV file from the sidebar to begin.")
    st.markdown("""
    **Instructions:**
    1. Select a CSV file that has been through initial QC (must have `qc_flag` column)
    2. Only events marked as `needs_check` will be shown
    3. View DBSCAN clusters colored by M3C2 distance
    4. Reclassify as: Real, Noise, Construction, Veg Error, Beach Error, or Other
    5. The CSV is updated in place after each classification
    """)
