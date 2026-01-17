#!/usr/bin/env python3
"""
8_clean_fill_grids.py

Combined parallel processing: cleaning + hole filling for erosion/deposition grids.
Uses physical coordinates (polygon centroids) for spatial proximity calculations.

Outputs only _filled.csv files (not _cleaned.csv intermediate files).
For erosion: applies cleaning + hole filling + morphological cleanup
For deposition: applies cleaning only (no hole filling)

Arguments:
    location          Name of the study site folder (e.g. SanElijo, Encinitas)
    --all             Process all locations (DelMar, Solana, Encinitas, SanElijo, Torrey, Blacks)
    --resolution      Grid resolution: '10cm', '25cm', or '1m' (default: 10cm)
    --erosion         Process only erosion clusters/grids
    --deposition      Process only deposition clusters/grids
    --threshold       Min non-zero cells to keep cluster (auto-scales if not provided)
    --min_volume      Min cluster volume (m³) for hole filling (default: 2.0)
    --elevation_scale Weight for elevation vs horizontal distance (default: 1.0)
    --cleanup_size    Min connected component size to keep (default: 3)
    --testing         Print actions without writing files
    --replace         Overwrite existing outputs
    --skip_filling    Skip hole filling step, only do cleaning (erosion only)
"""
import os
import platform
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import geopandas as gpd
import time
from datetime import datetime
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, cKDTree
from scipy.ndimage import binary_opening, label
from shapely.geometry import Point, Polygon
import alphashape

# ============================================================================
# RESOLUTION-DEPENDENT PARAMETERS
# ============================================================================

def get_resolution_params(resolution):
    """
    Get resolution-dependent parameters.
    Scales alpha and buffer_dist for the vector method.
    """
    params = {
        '10cm': {
            'cell_size': 0.10,
            'cell_area': 0.01,
            'default_threshold': 25,
            'buffer_bins': 20,       # 2m vertical check
            'min_points': 15,
            'alpha': 0.1,            # Alpha for alphashape
            'buffer_dist': 0.2       # Buffer distance in meters
        },
        '25cm': {
            'cell_size': 0.25,
            'cell_area': 0.0625,
            'default_threshold': 4,
            'buffer_bins': 8,
            'min_points': 8,
            'alpha': 0.04,
            'buffer_dist': 0.5
        },
        '1m': {
            'cell_size': 1.0,
            'cell_area': 1.0,
            'default_threshold': 1,
            'buffer_bins': 2,
            'min_points': 3,
            'alpha': 0.01,
            'buffer_dist': 2.0
        }
    }

    return params.get(resolution, params['10cm'])


# ============================================================================
# OPTIMIZED I/O FUNCTIONS (Pandas)
# ============================================================================

def load_csv_data(filepath):
    """Fast loads CSV using Pandas."""
    try:
        df = pd.read_csv(filepath, index_col=0, header=0, na_values=['', 'nan', 'NaN', 'NULL'])
        header_labels = df.columns.tolist()
        row_labels = df.index.astype(str).tolist()
        return header_labels, row_labels, df.values
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        raise e


def save_csv_data(filepath, data, header_labels, row_labels, testing, replace):
    """Fast saves data to CSV using Pandas."""
    if testing:
        print(f"[TEST] Would save: {filepath}", flush=True)
        return

    if os.path.exists(filepath) and not replace:
        print(f"[SKIP] File exists: {filepath}", flush=True)
        return

    try:
        df = pd.DataFrame(data, columns=header_labels, index=row_labels)
        df.to_csv(filepath, na_rep='', float_format='%.6g')
    except Exception as e:
        print(f"Error saving {filepath}: {e}")


# ============================================================================
# SHAPEFILE LOADING FUNCTIONS
# ============================================================================

def find_shapefile(base_util, location, poly_res):
    """
    Locate the shapefile for a given location and polygon resolution.

    Expected naming pattern: {Location}Polygon[e]s{MOP1}to{MOP2}at{resolution}
    Example: SanElijoPolygones683to708at25cm
    """
    sf_root = os.path.join(base_util, 'shape_files')
    if not os.path.isdir(sf_root):
        raise FileNotFoundError(f"Shape files directory not found: {sf_root}")

    # Pattern: location name + "Polygon" (may have 'e' or 's') + MOP range + "at" + resolution
    # Case-insensitive matching
    candidates = [
        d for d in os.listdir(sf_root)
        if d.lower().startswith(location.lower())
           and 'polygon' in d.lower()
           and f'at{poly_res}'.lower() in d.lower()
           and os.path.isdir(os.path.join(sf_root, d))
    ]

    if not candidates:
        # List available folders for debugging
        available = [d for d in os.listdir(sf_root) if os.path.isdir(os.path.join(sf_root, d))]
        raise FileNotFoundError(
            f"No shapefile folder found for location='{location}' resolution='{poly_res}'.\n"
            f"Searched in: {sf_root}\n"
            f"Available folders: {available}"
        )

    # If multiple matches, prefer exact match
    if len(candidates) > 1:
        print(f"Warning: Multiple shapefile folders found: {candidates}")
        print(f"Using first match: {candidates[0]}")

    fld = candidates[0]
    shp_path = os.path.join(sf_root, fld, fld + '.shp')
    if not os.path.isfile(shp_path):
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")
    return shp_path


def load_polygon_centroids(shapefile_path):
    """
    Load polygon centroids from shapefile.
    Returns dict mapping Polygon_ID -> (UTM_X, UTM_Y)
    """
    gdf = gpd.read_file(shapefile_path)
    gdf['Polygon_ID'] = gdf.index
    centroids = gdf.geometry.centroid
    return {int(pid): (c.x, c.y) for pid, c in zip(gdf['Polygon_ID'], centroids)}


def parse_elevation_from_header(header):
    """
    Extracts elevation value from header like 'M3C2_0.10m' -> 0.10
    """
    try:
        parts = header.split('_')
        val_str = parts[-1]
        val_str = val_str.replace('m', '')
        return float(val_str)
    except:
        return None


# ============================================================================
# CLEANING FUNCTIONS
# ============================================================================

def resolution_to_label(resolution):
    """Convert resolution string to label matching 7_make_grids.py"""
    res_map = {'10cm': '10cm', '25cm': '25cm', '1m': '1m'}
    return res_map.get(resolution, resolution)


def process_files_in_folder(date_folder, file_type, threshold, resolution):
    """Load and filter grid/cluster files by threshold."""
    label = resolution_to_label(resolution)

    # Navigate into resolution subdirectory
    resolution_folder = os.path.join(date_folder, label)
    if not os.path.isdir(resolution_folder):
        return None

    files = os.listdir(resolution_folder)

    # New file naming pattern includes date prefix: DATE1_to_DATE2_ero_grid_10cm.csv
    clust_pattern = f"clusters_{label}.csv"
    grid_pattern = f"grid_{label}.csv"

    if file_type == 'erosion':
        cls = [f for f in files if clust_pattern in f and 'dep_' not in f]
        grd = [f for f in files if grid_pattern in f and 'dep_' not in f]
    elif file_type == 'deposition':
        cls = [f for f in files if f"dep_{clust_pattern}" in f]
        grd = [f for f in files if f"dep_{grid_pattern}" in f]
    else:
        return None

    if not cls or not grd:
        return None

    cpath = os.path.join(resolution_folder, cls[0])
    gpath = os.path.join(resolution_folder, grd[0])

    try:
        h_c, r_c, clusters = load_csv_data(cpath)
        h_g, r_g, grid = load_csv_data(gpath)
    except Exception as e:
        print(f"Load error in {date_folder}: {e}")
        return None

    # --- THRESHOLD FILTERING ---
    nonz = clusters[(~np.isnan(clusters)) & (clusters != 0)]
    total_clusters = 0
    if nonz.size:
        ids, cnt = np.unique(nonz, return_counts=True)
        total_clusters = len(ids)
        valid = ids[cnt >= threshold]
    else:
        valid = np.array([])

    kept_count = len(valid)
    removed_threshold = total_clusters - kept_count

    mask = np.isin(clusters, valid)
    filt_c = np.where(mask, clusters, 0)
    filt_g = np.where(mask, grid, 0)

    return {
        'header_c': h_c, 'rows_c': r_c, 'orig_c': clusters,
        'header_g': h_g, 'rows_g': r_g, 'orig_g': grid,
        'filt_c': filt_c, 'filt_g': filt_g,
        'cfile': cpath, 'gfile': gpath,
        'stats': {
            'total_clusters': total_clusters,
            'removed_threshold': removed_threshold,
            'kept_threshold': kept_count
        }
    }


def footprint_check(dep_res, ero_grid, buffer_bins):
    """Check deposition clusters have erosion foundation."""
    dep = dep_res['filt_c'].copy()
    dgr = dep_res['filt_g'].copy()
    removed_count = 0
    unique_ids = np.unique(dep[dep != 0])

    for cid in unique_ids:
        idx = np.where(dep == cid)
        amin, amax = idx[0].min(), idx[0].max()
        bottom = idx[1].min()
        amin_b = max(0, amin - buffer_bins)
        amax_b = min(ero_grid.shape[0], amax + buffer_bins + 1)
        foot = ero_grid[amin_b:amax_b, bottom:ero_grid.shape[1]]

        if not np.any(np.nan_to_num(foot) > 0):
            dep[idx] = 0
            dgr[idx] = 0
            removed_count += 1

    return dep, dgr, removed_count


# ============================================================================
# MORPHOLOGICAL CLEANUP
# ============================================================================

def morphological_cleanup(grid, clusters, min_component_size=3):
    """
    Remove isolated cells and small fragments using morphological operations.
    Applied AFTER hole filling.

    Args:
        grid: M3C2 distance grid
        clusters: Cluster ID grid
        min_component_size: Minimum number of cells to keep a component (default: 3)

    Returns:
        grid_cleaned, clusters_cleaned
    """
    # Create binary mask of non-zero cells
    mask = (grid != 0) & ~np.isnan(grid)

    # Skip if nothing to clean
    if not np.any(mask):
        return grid, clusters

    # Opening removes small isolated features (3x3 structuring element)
    structure = np.ones((3, 3))
    mask_cleaned = binary_opening(mask, structure=structure)

    # Remove small connected components
    labeled, num_features = label(mask_cleaned)
    for i in range(1, num_features + 1):
        component_mask = labeled == i
        if np.sum(component_mask) < min_component_size:
            mask_cleaned[component_mask] = False

    # Apply cleanup mask
    grid_cleaned = np.where(mask_cleaned, grid, 0)
    clusters_cleaned = np.where(mask_cleaned, clusters, 0)

    cells_removed = np.sum(mask) - np.sum(mask_cleaned)
    if cells_removed > 0:
        print(f"    [Cleanup] Removed {cells_removed} isolated cells")

    return grid_cleaned, clusters_cleaned


# ============================================================================
# ALPHASHAPE / VECTOR HOLE FILLING WITH PHYSICAL COORDINATES
# ============================================================================

def calculate_cluster_volumes(grid, clusters_grid, cell_area):
    """Calculate volumes for all clusters from the grid."""
    cluster_volumes = {}
    if clusters_grid is None:
        return cluster_volumes

    valid_original = (grid != 0) & ~np.isnan(grid)
    valid_clusters = (clusters_grid != 0) & ~np.isnan(clusters_grid)
    unique_clusters = np.unique(clusters_grid[valid_clusters])
    unique_clusters = unique_clusters[unique_clusters > 0]

    for cluster_id in unique_clusters:
        cluster_mask = (clusters_grid == cluster_id) & valid_original
        if np.sum(cluster_mask) > 0:
            cluster_values = grid[cluster_mask]
            cluster_volumes[cluster_id] = np.sum(np.abs(cluster_values)) * cell_area
        else:
            cluster_volumes[cluster_id] = 0.0
    return cluster_volumes


def create_cluster_boundary(cluster_points, alpha=0.1, buffer_distance=0.2):
    """Create a boundary polygon around cluster points using alpha shape."""
    if len(cluster_points) < 3:
        return None
    try:
        alpha_shape = alphashape.alphashape(cluster_points, alpha)
        if alpha_shape is None or alpha_shape.geom_type not in ['Polygon', 'MultiPolygon']:
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]
            alpha_shape = Polygon(hull_points)
        return alpha_shape.buffer(buffer_distance)
    except:
        try:
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]
            polygon = Polygon(hull_points)
            return polygon.buffer(buffer_distance)
        except:
            return None


def grid_to_physical_coords(row_idx, col_idx, row_labels, col_elevations,
                            polygon_centroids, elevation_scale=1.0):
    """
    Convert grid indices to physical (X, Y, Z) coordinates.

    Args:
        row_idx: Row index in grid
        col_idx: Column index in grid
        row_labels: List of Polygon_ID strings
        col_elevations: Array of elevation values for each column
        polygon_centroids: Dict mapping Polygon_ID -> (X, Y)
        elevation_scale: Weight for Z relative to X/Y (default: 1.0 = equal)

    Returns:
        (X, Y, Z_scaled) tuple
    """
    polygon_id = int(row_labels[row_idx])
    elevation = col_elevations[col_idx]

    if polygon_id in polygon_centroids:
        x, y = polygon_centroids[polygon_id]
    else:
        # Fallback: use row index as proxy (shouldn't happen with valid data)
        x, y = float(row_idx), 0.0

    return (x, y, elevation * elevation_scale)


def assign_holes_to_clusters_physical(grid, clusters_grid, hole_mask, row_labels,
                                       col_elevations, polygon_centroids, elevation_scale=1.0):
    """
    Assign holes to nearest cluster using physical coordinates.
    Uses cKDTree for efficient nearest-neighbor search.

    Returns:
        hole_to_cluster: 2D array mapping each hole to its nearest cluster ID
    """
    # Get valid cluster cell positions
    valid_clusters_mask = ~hole_mask & (clusters_grid != 0) & ~np.isnan(clusters_grid)

    if not np.any(valid_clusters_mask):
        return np.zeros(grid.shape, dtype=int)

    valid_rows, valid_cols = np.where(valid_clusters_mask)

    # Convert valid cluster cells to physical coordinates
    valid_physical = np.array([
        grid_to_physical_coords(r, c, row_labels, col_elevations,
                                polygon_centroids, elevation_scale)
        for r, c in zip(valid_rows, valid_cols)
    ])

    # Build KD-tree from valid cluster positions
    tree = cKDTree(valid_physical)

    # Get hole positions
    hole_rows, hole_cols = np.where(hole_mask)

    if len(hole_rows) == 0:
        return np.zeros(grid.shape, dtype=int)

    # Convert holes to physical coordinates
    hole_physical = np.array([
        grid_to_physical_coords(r, c, row_labels, col_elevations,
                                polygon_centroids, elevation_scale)
        for r, c in zip(hole_rows, hole_cols)
    ])

    # Find nearest cluster cell for each hole
    _, nearest_indices = tree.query(hole_physical)

    # Map holes to cluster IDs
    hole_to_cluster = np.zeros(grid.shape, dtype=int)
    for i, (hr, hc) in enumerate(zip(hole_rows, hole_cols)):
        nearest_row = valid_rows[nearest_indices[i]]
        nearest_col = valid_cols[nearest_indices[i]]
        hole_to_cluster[hr, hc] = int(clusters_grid[nearest_row, nearest_col])

    return hole_to_cluster


def interpolate_within_cluster_physical(grid_data, cluster_data, cluster_id,
                                         hole_positions, row_labels, col_elevations,
                                         polygon_centroids, alpha, buffer_dist,
                                         elevation_scale=1.0):
    """
    Interpolate erosion values within a cluster boundary using physical coordinates.
    """
    cluster_mask = (cluster_data == cluster_id) & (grid_data != 0) & ~np.isnan(grid_data)
    if np.sum(cluster_mask) < 3:
        return np.full(len(hole_positions), np.nan)

    cluster_rows, cluster_cols = np.where(cluster_mask)
    cluster_values = grid_data[cluster_mask]

    # Convert cluster positions to physical coordinates
    cluster_physical = np.array([
        grid_to_physical_coords(r, c, row_labels, col_elevations,
                                polygon_centroids, elevation_scale)
        for r, c in zip(cluster_rows, cluster_cols)
    ])

    # Create boundary in physical space (using X, Y only for 2D boundary)
    cluster_xy = cluster_physical[:, :2]  # Only X, Y for boundary
    boundary = create_cluster_boundary(cluster_xy, alpha, buffer_dist)

    if boundary is None:
        return np.full(len(hole_positions), np.nan)

    # Convert hole positions to physical coordinates
    hole_physical = np.array([
        grid_to_physical_coords(r, c, row_labels, col_elevations,
                                polygon_centroids, elevation_scale)
        for r, c in hole_positions
    ])
    hole_xy = hole_physical[:, :2]

    # Check which holes are within this cluster's XY boundary
    within_boundary = np.array([boundary.contains(Point(xy)) for xy in hole_xy])

    if not np.any(within_boundary):
        return np.full(len(hole_positions), np.nan)

    interpolated = np.full(len(hole_positions), np.nan)

    # Interpolate using full 3D physical coordinates
    if np.sum(within_boundary) > 0:
        target_subset = hole_physical[within_boundary]
        try:
            interp_values = griddata(cluster_physical, cluster_values, target_subset,
                                   method='linear', fill_value=np.nan)
            interpolated[within_boundary] = interp_values
        except:
            pass

    return interpolated


def fill_holes_alphashape(grid, clusters_grid, hole_mask, cluster_volumes,
                          min_cluster_volume, alpha, buffer_dist,
                          row_labels, col_elevations, polygon_centroids,
                          elevation_scale=1.0):
    """
    Orchestrates the vector-based hole filling using physical coordinates.
    """
    filled_grid = grid.copy()
    filled_clusters_grid = clusters_grid.copy()

    holes_filled = 0
    filled_sum = 0
    clusters_skipped = 0
    clusters_processed = 0

    hole_positions = np.column_stack(np.where(hole_mask))
    if len(hole_positions) == 0:
        return filled_grid, filled_clusters_grid, 0, 0, 0, 0

    valid_clusters_mask = ~hole_mask & (clusters_grid != 0) & ~np.isnan(clusters_grid)
    if not np.any(valid_clusters_mask):
        return filled_grid, filled_clusters_grid, 0, 0, 0, 0

    # Assign holes to nearest cluster using PHYSICAL coordinates
    hole_to_cluster = assign_holes_to_clusters_physical(
        grid, clusters_grid, hole_mask, row_labels, col_elevations,
        polygon_centroids, elevation_scale
    )

    unique_clusters = np.unique(clusters_grid[valid_clusters_mask])
    unique_clusters = unique_clusters[unique_clusters > 0]

    for cluster_id in unique_clusters:
        cluster_volume = cluster_volumes.get(cluster_id, 0.0)

        if cluster_volume < min_cluster_volume:
            clusters_skipped += 1
            continue

        clusters_processed += 1

        # Find holes assigned to this cluster
        holes_in_cluster = hole_mask & (hole_to_cluster == cluster_id)
        if not np.any(holes_in_cluster):
            continue

        hole_indices = np.where(holes_in_cluster)
        hole_coords = np.column_stack(hole_indices)

        # Interpolate within cluster using physical coordinates
        interpolated = interpolate_within_cluster_physical(
            grid, clusters_grid, cluster_id, hole_coords,
            row_labels, col_elevations, polygon_centroids,
            alpha, buffer_dist, elevation_scale
        )

        valid_interp = ~np.isnan(interpolated)

        if np.any(valid_interp):
            filled_grid[hole_indices[0][valid_interp], hole_indices[1][valid_interp]] = interpolated[valid_interp]
            filled_clusters_grid[hole_indices[0][valid_interp], hole_indices[1][valid_interp]] = cluster_id

            holes_filled += np.sum(valid_interp)
            filled_sum += np.sum(interpolated[valid_interp])

    return filled_grid, filled_clusters_grid, holes_filled, filled_sum, clusters_processed, clusters_skipped


# ============================================================================
# WORKER FUNCTION
# ============================================================================

def worker(task):
    (date_folder, ftype, threshold, resolution, testing, replace,
     skip_cleaning, skip_filling, min_volume, res_params,
     polygon_centroids, elevation_scale, cleanup_size) = task

    folder_name = os.path.basename(date_folder)

    # ========== STEP 1: CLEANING ==========
    if not skip_cleaning:
        res = process_files_in_folder(date_folder, ftype, threshold, resolution)
        if res is None: return None

        removed_footprint = 0
        if ftype == 'deposition':
            ero_res = process_files_in_folder(date_folder, 'erosion', threshold, resolution)
            if ero_res:
                dc, dg, rm_count = footprint_check(res, ero_res['orig_g'], res_params['buffer_bins'])
                res['filt_c'], res['filt_g'] = dc, dg
                removed_footprint = rm_count

        # Skip saving _cleaned.csv - will only save final _filled.csv output
        cleaning_stats = res['stats']

        cleaned_grid = res['filt_g']
        cleaned_clusters = res['filt_c']
        header_g, rows_g = res['header_g'], res['rows_g']
        header_c, rows_c = res['header_c'], res['rows_c']
        gfile = res['gfile']
        cfile = res['cfile']
    else:
        # --skip_cleaning option is no longer supported since _cleaned.csv files are not saved
        # Users should run the full pipeline (cleaning is fast)
        print(f"[WARNING] --skip_cleaning is no longer supported. Intermediate _cleaned.csv files are not saved.")
        print(f"[WARNING] Skipping {folder_name} - run without --skip_cleaning flag.")
        return None

    # ========== STEP 2: ALPHASHAPE HOLE FILLING (EROSION ONLY) ==========
    if not skip_filling and ftype == 'erosion':
        # Parse column elevations
        col_elevations = np.array([parse_elevation_from_header(h) for h in header_g])

        original_volume = np.nansum(np.abs(cleaned_grid[cleaned_grid != 0])) * res_params['cell_area']
        hole_mask = (cleaned_grid == 0) & ~np.isnan(cleaned_grid)
        cluster_volumes = calculate_cluster_volumes(cleaned_grid, cleaned_clusters, res_params['cell_area'])

        filled_grid, filled_clusters, holes_filled, filled_sum, clusters_processed, clusters_skipped = \
            fill_holes_alphashape(
                cleaned_grid, cleaned_clusters, hole_mask, cluster_volumes,
                min_volume, res_params['alpha'], res_params['buffer_dist'],
                rows_g, col_elevations, polygon_centroids, elevation_scale
            )

        # ========== STEP 3: MORPHOLOGICAL CLEANUP ==========
        filled_grid, filled_clusters = morphological_cleanup(
            filled_grid, filled_clusters, min_component_size=cleanup_size
        )

        filled_volume = np.nansum(np.abs(filled_grid[filled_grid != 0])) * res_params['cell_area']
        volume_change = filled_volume - original_volume
        fill_percentage = (holes_filled / np.sum(hole_mask) * 100) if np.sum(hole_mask) > 0 else 0

        # Save Filled Grid
        g_filled = gfile.replace('.csv', '_filled.csv')
        save_csv_data(g_filled, filled_grid, header_g, rows_g, testing, replace)

        # Save Filled Clusters
        c_filled = cfile.replace('.csv', '_filled.csv')
        save_csv_data(c_filled, filled_clusters, header_c, rows_c, testing, replace)

        filling_stats = {
            'original_volume': original_volume, 'filled_volume': filled_volume,
            'volume_change': volume_change, 'holes_filled': holes_filled,
            'fill_percentage': fill_percentage, 'clusters_processed': clusters_processed,
            'clusters_skipped': clusters_skipped
        }
    else:
        # For deposition (or when filling is skipped), save cleaned results as _filled.csv
        g_filled = gfile.replace('.csv', '_filled.csv')
        save_csv_data(g_filled, cleaned_grid, header_g, rows_g, testing, replace)

        c_filled = cfile.replace('.csv', '_filled.csv')
        save_csv_data(c_filled, cleaned_clusters, header_c, rows_c, testing, replace)

        filling_stats = None

    return {
        'survey': folder_name, 'type': ftype,
        'cleaning': cleaning_stats,
        'filling': filling_stats
    }


# ============================================================================
# REPORTING
# ============================================================================

def generate_combined_report(location, resolution, threshold, min_volume, stats_list,
                             base_dir, skip_filling):
    report_dir = os.path.join(base_dir, 'validation', 'hole_filling', 'reports', location)
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"combined_report_{resolution}_{timestamp}.txt")

    with open(report_path, 'w') as f:
        f.write(f"CLEANING + HOLE FILLING (Physical Coordinates)\n")
        f.write(f"=" * 80 + "\n")
        f.write(f"Location: {location}\nResolution: {resolution}\n")
        f.write(f"Method: Vector (Alphashape) with Physical Coordinates\n")
        f.write(f"Output: _filled.csv files only (no intermediate _cleaned.csv)\n")

        if not stats_list:
            f.write("No files processed.\n")
            return

        # Cleaning is always performed
        f.write(f"\nCLEANING SUMMARY\n" + "-" * 30 + "\n")
        total_removed = sum(s['cleaning'].get('removed_threshold', 0) for s in stats_list)
        f.write(f"Total Clusters Removed by Threshold: {total_removed}\n")

        if not skip_filling:
            filled_stats = [s for s in stats_list if s['filling'] is not None]
            if filled_stats:
                f.write(f"\nHOLE FILLING SUMMARY (Erosion only)\n" + "-" * 30 + "\n")
                f.write(f"Total Vol Change: {sum(s['filling']['volume_change'] for s in filled_stats):.4f} m³\n")
                f.write(f"Total Holes Filled: {sum(s['filling']['holes_filled'] for s in filled_stats)}\n")

        f.write(f"\nDETAILED LOG\n" + "-" * 90 + "\n")
        f.write(f"{'Survey':<25} {'Type':<8} {'Removed':<10} {'Holes':<7} {'Vol∆':<10} {'Fill%':<7}\n")
        for s in sorted(stats_list, key=lambda x: (x['survey'], x['type'])):
            removed = s['cleaning'].get('removed_threshold', 0)
            if s['filling']:
                f.write(f"{s['survey']:<25} {s['type']:<8} {removed:<10} {s['filling']['holes_filled']:<7} "
                       f"{s['filling']['volume_change']:<10.3f} {s['filling']['fill_percentage']:<7.1f}\n")
            else:
                f.write(f"{s['survey']:<25} {s['type']:<8} {removed:<10} {'N/A':<7} {'N/A':<10} {'N/A':<7}\n")

    print(f"\n✓ Report generated: {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Cleaning + hole filling with physical coordinates')
    parser.add_argument('location', nargs='?', help='Site folder under results/')
    parser.add_argument('--all', action='store_true', help='Process all locations')
    parser.add_argument('--resolution', choices=['10cm', '25cm', '1m'], default='10cm')
    parser.add_argument('--erosion', action='store_true')
    parser.add_argument('--deposition', action='store_true')
    parser.add_argument('--threshold', type=int, default=None)
    parser.add_argument('--min_volume', type=float, default=2.0)
    parser.add_argument('--elevation_scale', type=float, default=1.0,
                        help='Weight for elevation vs horizontal distance (default: 1.0)')
    parser.add_argument('--cleanup_size', type=int, default=3,
                        help='Min connected component size to keep (default: 3)')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--skip_filling', action='store_true',
                        help='Skip hole filling (erosion only), only apply cleaning')
    args = parser.parse_args()

    # Validate arguments
    if args.all and args.location:
        parser.error("Cannot specify both --all and a specific location")
    if not args.all and not args.location:
        parser.error("Must specify either a location or --all")

    # Define all locations
    all_locations = ['DelMar', 'Solana', 'Encinitas', 'SanElijo', 'Torrey', 'Blacks']

    # Determine which locations to process
    if args.all:
        locations = all_locations
        print(f"\n{'='*80}\nProcessing ALL locations\n{'='*80}")
    else:
        locations = [args.location]

    # Process each location
    for location in locations:
        process_location(location, args)


def process_location(location, args):
    """Process a single location with the given arguments."""
    # Set skip_cleaning to False (always do cleaning since we don't save intermediate files)
    skip_cleaning = False

    res_params = get_resolution_params(args.resolution)
    threshold = args.threshold if args.threshold is not None else res_params['default_threshold']

    process_erosion = args.erosion or not (args.erosion or args.deposition)
    process_deposition = args.deposition or not (args.erosion or args.deposition)

    print(f"\n{'='*80}\nCLEANING + HOLE FILLING (Physical Coordinates)\n{'='*80}")
    print(f"Location: {location}")
    print(f"Resolution: {args.resolution}")
    print(f"Elevation Scale: {args.elevation_scale}")
    print(f"Cleanup Size: {args.cleanup_size}")

    system = platform.system()
    base_dir = '/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs' if system == 'Darwin' else '/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs'
    results_dir = os.path.join(base_dir, 'results', location)
    util_dir = os.path.join(base_dir, 'utilities')

    # Check if results directory exists
    if not os.path.isdir(results_dir):
        print(f"[WARNING] Results directory not found: {results_dir}")
        print(f"[WARNING] Skipping location: {location}")
        return

    # LOAD SHAPEFILE AND POLYGON CENTROIDS
    try:
        shp_path = find_shapefile(util_dir, location, args.resolution)
        print(f"Loading shapefile: {shp_path}")
        polygon_centroids = load_polygon_centroids(shp_path)
        print(f"Loaded {len(polygon_centroids)} polygon centroids")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print(f"[WARNING] Cannot proceed without shapefile for physical coordinates.")
        print(f"[WARNING] Skipping location: {location}")
        return

    tasks = []
    if process_erosion:
        ero_dir = os.path.join(results_dir, 'erosion')
        if os.path.isdir(ero_dir):
            for d in os.listdir(ero_dir):
                path = os.path.join(ero_dir, d)
                if os.path.isdir(path):
                    tasks.append((path, 'erosion', threshold, args.resolution,
                                args.testing, args.replace, skip_cleaning,
                                args.skip_filling, args.min_volume, res_params,
                                polygon_centroids, args.elevation_scale, args.cleanup_size))

    if process_deposition:
        dep_dir = os.path.join(results_dir, 'deposition')
        if os.path.isdir(dep_dir):
            for d in os.listdir(dep_dir):
                path = os.path.join(dep_dir, d)
                if os.path.isdir(path):
                    tasks.append((path, 'deposition', threshold, args.resolution,
                                args.testing, args.replace, skip_cleaning,
                                True, args.min_volume, res_params,
                                polygon_centroids, args.elevation_scale, args.cleanup_size))

    if not tasks:
        print(f"[WARNING] No tasks found for location: {location}")
        return

    workers = max(1, multiprocessing.cpu_count() // 4)
    print(f"Launching {len(tasks)} tasks on {workers} workers...\n")

    start_time = time.time()
    with multiprocessing.Pool(workers) as pool:
        stats_results = [r for r in pool.map(worker, tasks) if r is not None]

    generate_combined_report(location, args.resolution, threshold,
                           args.min_volume, stats_results, base_dir,
                           args.skip_filling)

    print(f"Processing complete for {location} in {time.time() - start_time:.2f} s")

if __name__ == '__main__':
    main()
