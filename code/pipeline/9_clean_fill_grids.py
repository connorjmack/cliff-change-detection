#!/usr/bin/env python3
"""
combined_clean_fill_alphashape.py

Combined parallel processing: cleaning + hole filling for erosion/deposition grids.
Architecture: Optimized Pandas I/O (from File 9)
Algorithm:    Alphashape/Vector Hole Filling + Visual Cliff Top Cutoff

Arguments:
    location          Name of the study site folder (e.g. SanElijo, Encinitas)
    --resolution      Grid resolution: '10cm', '25cm', or '1m' (default: 10cm)
    --erosion         Process only erosion clusters/grids
    --deposition      Process only deposition clusters/grids
    --threshold       Min non-zero cells to keep cluster (auto-scales if not provided)
    --min_volume      Min cluster volume (m³) for hole filling (default: 2.0)
    --testing         Print actions without writing files
    --replace         Overwrite existing outputs
    --skip_cleaning   Skip cleaning step, only do hole filling
    --skip_filling    Skip hole filling step, only do cleaning
"""
import os
import platform
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import time
from datetime import datetime
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from scipy.ndimage import distance_transform_edt
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
            'alpha': 0.1,            # Alpha for alphashape (File 2 default)
            'buffer_dist': 0.2       # Buffer distance in meters (File 2 default)
        },
        '25cm': {
            'cell_size': 0.25,
            'cell_area': 0.0625,
            'default_threshold': 4,
            'buffer_bins': 8,
            'min_points': 8,
            'alpha': 0.04,           # Roughly 0.1 * (10/25)
            'buffer_dist': 0.5
        },
        '1m': {
            'cell_size': 1.0,
            'cell_area': 1.0,
            'default_threshold': 1,
            'buffer_bins': 2,
            'min_points': 3,
            'alpha': 0.01,           # Roughly 0.1 * (10/100)
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
# NEW: VERTICAL CUTOFF LOGIC (FIXED PARSER)
# ============================================================================

def load_cutoff_dataframe(base_dir, location, resolution):
    """
    Loads the visual cliff top cutoff file. 
    Tries multiple potential paths.
    """
    filename = f"{location}_Visual_CliffTop_{resolution}.csv"
    
    paths = [
        os.path.join(base_dir, "utilities", "cliff_top_cutoffs", filename),
        os.path.join(base_dir, "utilities", "cliff_top_cutoffs", "computer_vision", location, filename),
        os.path.join("utilities", "cliff_top_cutoffs", filename)
    ]
    
    cutoff_path = None
    for p in paths:
        if os.path.exists(p):
            cutoff_path = p
            break
            
    if not cutoff_path:
        print(f"[WARNING] No visual cutoff file found for {location} {resolution}. Skipping vertical slice.")
        return None
        
    print(f"Loaded Cutoff File: {cutoff_path}")
    try:
        df = pd.read_csv(cutoff_path)
        if 'Polygon_ID' in df.columns:
            df.set_index('Polygon_ID', inplace=True)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load cutoff file: {e}")
        return None

def parse_elevation_from_header(header):
    """
    FIXED PARSER: Correctly extracts '0.10' from 'M3C2_0.10m'.
    Old version stripped letters but kept ALL digits, turning 'M3C2_0.10m' -> 320.10
    """
    try:
        # 1. Split by underscore. Usually ['M3C2', '0.10m'] or ['ClusterID', '20.50m']
        parts = header.split('_')
        
        # 2. Take the last part ('0.10m')
        val_str = parts[-1]
        
        # 3. Remove 'm' or other suffix
        val_str = val_str.replace('m', '')
        
        return float(val_str)
    except:
        return None

def apply_vertical_cutoff(grid, clusters, header_labels, row_labels, cutoff_df):
    """
    Zeros out any data where Elevation > CliffTop_Z for that polygon.
    """
    if cutoff_df is None:
        return grid, clusters, 0

    # 1. Parse Column Headers to Elevation Floats (USING FIXED PARSER)
    try:
        col_elevs = [parse_elevation_from_header(h) for h in header_labels]
        # Validate parsing
        if any(x is None for x in col_elevs):
            print(f"[WARNING] Header parsing failed for some columns (e.g. {header_labels[0]}). Skipping cutoff.")
            return grid, clusters, 0
        col_elevs = np.array(col_elevs)
    except:
        print("[WARNING] Exception during header parsing. Skipping cutoff.")
        return grid, clusters, 0

    # 2. Align Cutoff Zs to Grid Rows
    try:
        # Convert row labels (PolygonIDs) to int/float to match cutoff index
        row_ids = np.array(row_labels).astype(float).astype(int)
    except:
        print("[WARNING] Could not parse row labels as IDs. Skipping cutoff.")
        return grid, clusters, 0

    # Get cutoff Z for each row (aligned)
    # reindex handles missing polygons by putting NaN
    # We fill NaNs with 9999 so we don't accidentally cut valid data if the lookup fails
    aligned_cutoffs = cutoff_df.reindex(row_ids)['CliffTop_Z'].values
    aligned_cutoffs = np.nan_to_num(aligned_cutoffs, nan=9999.0)

    # 3. Create 2D Mask (Rows x Cols)
    # Mask = True where Column_Elevation > Row_Cutoff
    # Broadcasting: (1, Cols) > (Rows, 1)
    mask_above = col_elevs[None, :] > aligned_cutoffs[:, None]
    
    # 4. Apply Mask
    cells_removed = np.sum((grid != 0) & mask_above & ~np.isnan(grid))
    
    # Zero out data above the line
    grid[mask_above] = 0
    clusters[mask_above] = 0
    
    return grid, clusters, cells_removed

# ============================================================================
# CLEANING FUNCTIONS
# ============================================================================

def resolution_to_label(resolution):
    """Convert resolution string to cm label matching 8_make_grids.py"""
    res_map = {'10cm': '10cm', '25cm': '25cm', '1m': '100cm'}
    return res_map.get(resolution, resolution)


def process_files_in_folder(date_folder, file_type, threshold, resolution, cutoff_df=None):
    files = os.listdir(date_folder)
    label = resolution_to_label(resolution)
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
    
    cpath = os.path.join(date_folder, cls[0])
    gpath = os.path.join(date_folder, grd[0])
    
    try:
        h_c, r_c, clusters = load_csv_data(cpath)
        h_g, r_g, grid = load_csv_data(gpath)
    except Exception as e:
        print(f"Load error in {date_folder}: {e}")
        return None

    # --- APPLY VERTICAL CUTOFF ---
    cutoff_removed_count = 0
    if cutoff_df is not None:
        grid, clusters, cutoff_removed_count = apply_vertical_cutoff(
            grid, clusters, h_g, r_c, cutoff_df
        )

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
            'kept_threshold': kept_count,
            'cutoff_removed_cells': cutoff_removed_count
        }
    }


def footprint_check(dep_res, ero_grid, buffer_bins):
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
# ALPHASHAPE / VECTOR HOLE FILLING
# ============================================================================

def calculate_cluster_volumes(grid, clusters_grid, cell_area):
    """Calculate volumes for all clusters from the grid."""
    cluster_volumes = {}
    if clusters_grid is None:
        return cluster_volumes
    
    valid_original = (grid != 0) & ~np.isnan(grid)
    valid_clusters = (clusters_grid != 0) & ~np.isnan(clusters_grid)
    unique_clusters = np.unique(clusters_grid[valid_clusters])
    unique_clusters = unique_clusters[unique_clusters > 0] # Ignore 0 and -1
    
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


def interpolate_within_cluster(grid_data, cluster_data, cluster_id, hole_positions, alpha, buffer_dist):
    """Interpolate erosion values within a specific cluster boundary."""
    cluster_mask = (cluster_data == cluster_id) & (grid_data != 0) & ~np.isnan(grid_data)
    if np.sum(cluster_mask) < 3:
        return np.full(len(hole_positions), np.nan)
    
    cluster_positions = np.column_stack(np.where(cluster_mask))
    cluster_values = grid_data[cluster_mask]
    
    # Create boundary using specific alpha/buffer
    boundary = create_cluster_boundary(cluster_positions, alpha, buffer_dist)
    
    if boundary is None:
        return np.full(len(hole_positions), np.nan)
    
    # Check which holes are within this cluster's boundary
    within_boundary = np.array([boundary.contains(Point(pos)) for pos in hole_positions])
    
    if not np.any(within_boundary):
        return np.full(len(hole_positions), np.nan)
    
    interpolated = np.full(len(hole_positions), np.nan)
    
    if np.sum(within_boundary) > 0:
        target_subset = hole_positions[within_boundary]
        try:
            interp_values = griddata(cluster_positions, cluster_values, target_subset, 
                                   method='linear', fill_value=np.nan)
            interpolated[within_boundary] = interp_values
        except:
            pass # linear failed
            
    return interpolated


def fill_holes_alphashape(grid, clusters_grid, hole_mask, cluster_volumes,
                          min_cluster_volume, alpha, buffer_dist):
    """
    Orchestrates the vector-based hole filling.
    Now correctly updates and returns the filled CLUSTER grid as well.
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

    # === Assign holes to nearest cluster ===
    _, nearest_indices = distance_transform_edt(~valid_clusters_mask, return_indices=True)
    hole_to_cluster = np.zeros(grid.shape, dtype=int)
    hole_to_cluster[hole_mask] = clusters_grid[nearest_indices[0][hole_mask], nearest_indices[1][hole_mask]]
    
    unique_clusters = np.unique(clusters_grid[valid_clusters_mask])
    unique_clusters = unique_clusters[unique_clusters > 0] # Ignore 0 and -1

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
        
        # Interpolate within cluster (uses alpha-shape boundary)
        interpolated = interpolate_within_cluster(
            grid, clusters_grid, cluster_id, hole_coords, alpha, buffer_dist
        )
        
        valid_interp = ~np.isnan(interpolated)
        
        if np.any(valid_interp):
            # Fill the elevation grid
            filled_grid[hole_indices[0][valid_interp], hole_indices[1][valid_interp]] = interpolated[valid_interp]
            # Fill the cluster ID grid
            filled_clusters_grid[hole_indices[0][valid_interp], hole_indices[1][valid_interp]] = cluster_id
            
            holes_filled += np.sum(valid_interp)
            filled_sum += np.sum(interpolated[valid_interp])
            
    return filled_grid, filled_clusters_grid, holes_filled, filled_sum, clusters_processed, clusters_skipped


# ============================================================================
# WORKER FUNCTION
# ============================================================================

def worker(task):
    (date_folder, ftype, threshold, resolution, testing, replace, 
     skip_cleaning, skip_filling, min_volume, res_params, cutoff_df) = task
    
    folder_name = os.path.basename(date_folder)
    
    # ========== STEP 1: CLEANING + CUTOFF ==========
    if not skip_cleaning:
        # Pass cutoff_df here
        res = process_files_in_folder(date_folder, ftype, threshold, resolution, cutoff_df)
        if res is None: return None

        removed_footprint = 0
        if ftype == 'deposition':
            # Deposition also gets cutoff checked (redundant but safe)
            ero_res = process_files_in_folder(date_folder, 'erosion', threshold, resolution, cutoff_df)
            if ero_res:
                dc, dg, rm_count = footprint_check(res, ero_res['orig_g'], res_params['buffer_bins'])
                res['filt_c'], res['filt_g'] = dc, dg
                removed_footprint = rm_count

        c_out = res['cfile'].replace('.csv', '_cleaned.csv')
        g_out = res['gfile'].replace('.csv', '_cleaned.csv')
        save_csv_data(c_out, res['filt_c'], res['header_c'], res['rows_c'], testing, replace)
        save_csv_data(g_out, res['filt_g'], res['header_g'], res['rows_g'], testing, replace)

        cleaning_stats = res['stats']
        final_clean_count = cleaning_stats['kept_threshold'] - removed_footprint
        
        cleaned_grid = res['filt_g']
        cleaned_clusters = res['filt_c']
        header_g, rows_g = res['header_g'], res['rows_g']
        header_c, rows_c = res['header_c'], res['rows_c']
    else:
        # Load existing cleaned
        files = os.listdir(date_folder)
        grd = [f for f in files if f"grid_{resolution}_cleaned.csv" in f and (ftype == 'erosion' or 'dep_' in f)]
        cls = [f for f in files if f"clusters_{resolution}_cleaned.csv" in f and (ftype == 'erosion' or 'dep_' in f)]
        
        if not cls or not grd: return None
        try:
            header_g, rows_g, cleaned_grid = load_csv_data(os.path.join(date_folder, grd[0]))
            header_c, rows_c, cleaned_clusters = load_csv_data(os.path.join(date_folder, cls[0]))
            cleaning_stats = {'total_clusters': 0, 'removed_threshold': 0, 'kept_threshold': 0, 'cutoff_removed_cells': 0}
            removed_footprint = 0
            final_clean_count = 0
        except: return None
    
    # ========== STEP 2: ALPHASHAPE HOLE FILLING (EROSION ONLY) ==========
    if not skip_filling and ftype == 'erosion':
        original_volume = np.nansum(np.abs(cleaned_grid[cleaned_grid != 0])) * res_params['cell_area']
        hole_mask = (cleaned_grid == 0) & ~np.isnan(cleaned_grid)
        cluster_volumes = calculate_cluster_volumes(cleaned_grid, cleaned_clusters, res_params['cell_area'])
        
        filled_grid, filled_clusters, holes_filled, filled_sum, clusters_processed, clusters_skipped = \
            fill_holes_alphashape(
                cleaned_grid, cleaned_clusters, hole_mask, cluster_volumes,
                min_volume, res_params['alpha'], res_params['buffer_dist']
            )
        
        filled_volume = np.nansum(np.abs(filled_grid[filled_grid != 0])) * res_params['cell_area']
        volume_change = filled_volume - original_volume
        fill_percentage = (holes_filled / np.sum(hole_mask) * 100) if np.sum(hole_mask) > 0 else 0
        
        # Save Filled Grid
        g_filled = res['gfile'].replace('.csv', '_filled.csv') if not skip_cleaning else \
                   os.path.join(date_folder, grd[0].replace('_cleaned.csv', '_filled.csv'))
        save_csv_data(g_filled, filled_grid, header_g, rows_g, testing, replace)
        
        # Save Filled Clusters
        c_filled = res['cfile'].replace('.csv', '_filled.csv') if not skip_cleaning else \
                   os.path.join(date_folder, cls[0].replace('_cleaned.csv', '_filled.csv'))
        save_csv_data(c_filled, filled_clusters, header_c, rows_c, testing, replace)
        
        filling_stats = {
            'original_volume': original_volume, 'filled_volume': filled_volume,
            'volume_change': volume_change, 'holes_filled': holes_filled,
            'fill_percentage': fill_percentage, 'clusters_processed': clusters_processed,
            'clusters_skipped': clusters_skipped
        }
    else:
        filling_stats = None
    
    return {
        'survey': folder_name, 'type': ftype,
        'cleaning': cleaning_stats,
        'filling': filling_stats
    }


# ============================================================================
# REPORTING
# ============================================================================

def generate_combined_report(location, resolution, threshold, min_volume, stats_list, base_dir, skip_cleaning, skip_filling):
    report_dir = os.path.join(base_dir, 'validation', 'hole_filling', 'reports', location)
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"combined_report_alphashape_{resolution}_{timestamp}.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"COMBINED CLEANING + HOLE FILLING + VISUAL CUTOFF\n")
        f.write(f"=" * 80 + "\n")
        f.write(f"Location: {location}\nResolution: {resolution}\n")
        f.write(f"Method: Vector (Alphashape)\n")
        
        if not stats_list:
            f.write("No files processed.\n")
            return

        if not skip_cleaning:
            f.write(f"\nCLEANING SUMMARY (Visual Cutoff Applied)\n" + "-" * 30 + "\n")
            total_cutoff_cells = sum(s['cleaning'].get('cutoff_removed_cells', 0) for s in stats_list)
            f.write(f"Total Cells Removed by Visual Cutoff: {total_cutoff_cells}\n")

        if not skip_filling:
            filled_stats = [s for s in stats_list if s['filling'] is not None]
            if filled_stats:
                f.write(f"\nHOLE FILLING SUMMARY\n" + "-" * 30 + "\n")
                f.write(f"Total Vol Change: {sum(s['filling']['volume_change'] for s in filled_stats):.4f} m³\n")
                f.write(f"Total Holes Filled: {sum(s['filling']['holes_filled'] for s in filled_stats)}\n")
        
        f.write(f"\nDETAILED LOG\n" + "-" * 90 + "\n")
        f.write(f"{'Survey':<25} {'Type':<8} {'CutoffCells':<12} {'Holes':<7} {'Vol∆':<10} {'Fill%':<7}\n")
        for s in sorted(stats_list, key=lambda x: (x['survey'], x['type'])):
            cutoff_cells = s['cleaning'].get('cutoff_removed_cells', 0) if not skip_cleaning else 'N/A'
            if s['filling']:
                f.write(f"{s['survey']:<25} {s['type']:<8} {cutoff_cells:<12} {s['filling']['holes_filled']:<7} "
                       f"{s['filling']['volume_change']:<10.3f} {s['filling']['fill_percentage']:<7.1f}\n")
            else:
                f.write(f"{s['survey']:<25} {s['type']:<8} {cutoff_cells:<12} {'N/A':<7} {'N/A':<10} {'N/A':<7}\n")

    print(f"\n✓ Report generated: {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Combined cleaning + alphashape hole filling + visual cutoff')
    parser.add_argument('location', help='Site folder under results/')
    parser.add_argument('--resolution', choices=['10cm', '25cm', '1m'], default='10cm')
    parser.add_argument('--erosion', action='store_true')
    parser.add_argument('--deposition', action='store_true')
    parser.add_argument('--threshold', type=int, default=None)
    parser.add_argument('--min_volume', type=float, default=2.0)
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--skip_cleaning', action='store_true')
    parser.add_argument('--skip_filling', action='store_true')
    args = parser.parse_args()

    res_params = get_resolution_params(args.resolution)
    if args.threshold is None: args.threshold = res_params['default_threshold']
    
    process_erosion = args.erosion or not (args.erosion or args.deposition)
    process_deposition = args.deposition or not (args.erosion or args.deposition)
    
    print(f"\n{'='*80}\nCLEANING + VISUAL CUTOFF + ALPHASHAPE FILLING\n{'='*80}")
    print(f"Location: {args.location}")
    print(f"Resolution: {args.resolution}")

    system = platform.system()
    base_dir = '/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs' if system == 'Darwin' else '/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs'
    results_dir = os.path.join(base_dir, 'results', args.location)
    
    # LOAD CUTOFF DATAFRAME ONCE
    cutoff_df = load_cutoff_dataframe(base_dir, args.location, args.resolution)
    
    tasks = []
    if process_erosion:
        ero_dir = os.path.join(results_dir, 'erosion')
        if os.path.isdir(ero_dir):
            for d in os.listdir(ero_dir):
                path = os.path.join(ero_dir, d)
                if os.path.isdir(path):
                    tasks.append((path, 'erosion', args.threshold, args.resolution, 
                                args.testing, args.replace, args.skip_cleaning, 
                                args.skip_filling, args.min_volume, res_params, cutoff_df))
    
    if process_deposition:
        dep_dir = os.path.join(results_dir, 'deposition')
        if os.path.isdir(dep_dir):
            for d in os.listdir(dep_dir):
                path = os.path.join(dep_dir, d)
                if os.path.isdir(path):
                    tasks.append((path, 'deposition', args.threshold, args.resolution,
                                args.testing, args.replace, args.skip_cleaning,
                                True, args.min_volume, res_params, cutoff_df))

    workers = max(1, multiprocessing.cpu_count() // 4)
    print(f"Launching {len(tasks)} tasks on {workers} workers...\n")
    
    start_time = time.time()
    with multiprocessing.Pool(workers) as pool:
        stats_results = [r for r in pool.map(worker, tasks) if r is not None]
    
    generate_combined_report(args.location, args.resolution, args.threshold, 
                           args.min_volume, stats_results, base_dir,
                           args.skip_cleaning, args.skip_filling)
    
    print(f"Processing complete in {time.time() - start_time:.2f} s")

if __name__ == '__main__':
    main()