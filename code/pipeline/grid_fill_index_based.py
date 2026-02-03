#!/usr/bin/env python3
"""
grid_fill_index_based.py

New hole-filling algorithm that works entirely in grid index space.
Avoids UTM coordinate conversions that cause stripe artifacts.

Algorithm:
1. For each cluster, find its bounding box in grid space
2. Identify holes (zero cells) within cluster boundaries
3. Use morphological operations to define fillable regions
4. Interpolate values using scipy.ndimage or griddata on indices

Usage:
    python3 grid_fill_index_based.py --location DelMar --top 5 --test
    python3 grid_fill_index_based.py --location DelMar --apply
"""

import os
import sys
import platform
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colormaps
from scipy.ndimage import binary_dilation, binary_erosion, label, binary_fill_holes, distance_transform_edt, binary_closing, binary_opening, gaussian_filter, convolve
from scipy.spatial import ConvexHull, Delaunay

# === Configuration ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
EVENT_LISTS_DIR = os.path.join(LOCAL_RESULTS_DIR, 'event_lists', 'erosion')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'figures', 'testing')

if platform.system() == "Darwin":
    SERVER_BASE = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
else:
    SERVER_BASE = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

RESOLUTION = "25cm"
CELL_SIZE = 0.25


# ============================================================================
# GRID I/O
# ============================================================================

def load_grid_csv(filepath: str) -> tuple:
    """Load grid CSV file."""
    if not os.path.exists(filepath):
        return None, None, None

    df = pd.read_csv(filepath, index_col=0, header=0,
                     na_values=['', 'nan', 'NaN', 'NULL'])

    row_ids = [int(r) for r in df.index.tolist()]
    headers = df.columns.tolist()

    return df.values, row_ids, headers


def parse_elevation_from_header(header: str) -> float:
    """Extract elevation from header like 'M3C2_0.25m' -> 0.25"""
    try:
        parts = header.split('_')
        val_str = parts[-1].replace('m', '')
        return float(val_str)
    except Exception:
        return None


# ============================================================================
# NEW INDEX-BASED HOLE FILLING
# ============================================================================

def get_cluster_mask(clusters: np.ndarray, cluster_id: int) -> np.ndarray:
    """Get binary mask for a specific cluster."""
    return clusters == cluster_id


def get_cluster_boundary_mask(cluster_mask: np.ndarray, dilation_iterations: int = 2) -> np.ndarray:
    """
    Create a boundary region around cluster using morphological dilation.
    This defines where holes can be filled.
    """
    # Dilate the cluster mask to include nearby holes
    dilated = binary_dilation(cluster_mask, iterations=dilation_iterations)
    return dilated


def get_convex_hull_mask(cluster_mask: np.ndarray) -> np.ndarray:
    """
    Create convex hull mask around cluster points.
    Holes within this convex hull are candidates for filling.
    """
    if not np.any(cluster_mask):
        return cluster_mask

    # Get points in cluster
    rows, cols = np.where(cluster_mask)
    points = np.column_stack([rows, cols])

    if len(points) < 3:
        return binary_dilation(cluster_mask, iterations=2)

    try:
        # Compute convex hull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # Create Delaunay triangulation to test point containment
        delaunay = Delaunay(hull_points)

        # Create output mask
        hull_mask = np.zeros_like(cluster_mask, dtype=bool)

        # Get bounding box
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        # Test each point in bounding box
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                if delaunay.find_simplex([r, c]) >= 0:
                    hull_mask[r, c] = True

        return hull_mask

    except Exception:
        # Fallback to dilation + fill holes
        dilated = binary_dilation(cluster_mask, iterations=2)
        return binary_fill_holes(dilated)


def get_alpha_shape_mask(cluster_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create alpha shape (concave hull) mask around cluster points.
    Tighter fit than convex hull - follows concave boundaries.

    Args:
        cluster_mask: Binary mask of cluster cells
        alpha: Controls tightness (higher = tighter, 0 = convex hull)
               Typical values: 0.1-2.0
    """
    if not np.any(cluster_mask):
        return cluster_mask

    rows, cols = np.where(cluster_mask)
    points = np.column_stack([rows, cols])

    if len(points) < 4:
        return binary_dilation(cluster_mask, iterations=1)

    try:
        # Use alphashape library if available
        import alphashape
        from shapely.geometry import Point

        # Compute alpha shape
        alpha_shape = alphashape.alphashape(points, alpha)

        if alpha_shape is None or alpha_shape.is_empty:
            return binary_fill_holes(cluster_mask)

        # Create mask from alpha shape
        hull_mask = np.zeros_like(cluster_mask, dtype=bool)

        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                if alpha_shape.contains(Point(r, c)):
                    hull_mask[r, c] = True

        return hull_mask

    except ImportError:
        # Fallback if alphashape not installed
        print("    [Note: alphashape not installed, using morphological approach]")
        return get_morphological_fill_mask(cluster_mask, dilation=1)
    except Exception:
        return binary_fill_holes(cluster_mask)


def get_morphological_fill_mask(cluster_mask: np.ndarray, dilation: int = 1,
                                 smooth_close: int = 0, smooth_open: int = 0) -> np.ndarray:
    """
    Create fill mask using morphological operations.
    Tight fitting - only fills internal holes and small gaps.

    Args:
        cluster_mask: Binary mask of cluster cells
        dilation: Number of dilation iterations (1-3 typical)
        smooth_close: Iterations of binary_closing for smoothing (0 = none)
        smooth_open: Iterations of binary_opening before closing (0 = none)
    """
    if not np.any(cluster_mask):
        return cluster_mask

    # First fill internal holes (only fills holes completely surrounded)
    filled = binary_fill_holes(cluster_mask)

    # Then dilate slightly to catch edge gaps
    if dilation > 0:
        filled = binary_dilation(filled, iterations=dilation)

    # Apply smoothing: opening removes small protrusions
    if smooth_open > 0:
        filled = binary_opening(filled, iterations=smooth_open)

    # Apply smoothing: closing fills small gaps and smooths boundary
    if smooth_close > 0:
        filled = binary_closing(filled, iterations=smooth_close)

    return filled


def get_rowwise_fill_mask(cluster_mask: np.ndarray, neighbor_rows: int = 2) -> np.ndarray:
    """
    Create fill mask by looking at each row's valid elevation range,
    expanded by neighboring rows. This handles vertical stripes better
    than morphological operations.

    For cliff data (rows=alongshore, cols=elevation), this fills gaps
    where a row has no data but neighboring rows do.

    Args:
        cluster_mask: Binary mask of cluster cells (rows=alongshore, cols=elevation)
        neighbor_rows: How many neighboring rows to consider for elevation range

    Returns:
        Fill mask where True = can be filled
    """
    if not np.any(cluster_mask):
        return cluster_mask

    n_rows, n_cols = cluster_mask.shape
    fill_mask = np.zeros_like(cluster_mask, dtype=bool)

    # For each row, determine fillable elevation range from neighbors
    for row in range(n_rows):
        # Get elevation range from this row and neighbors
        row_min = max(0, row - neighbor_rows)
        row_max = min(n_rows, row + neighbor_rows + 1)

        neighbor_data = cluster_mask[row_min:row_max, :]

        if not np.any(neighbor_data):
            continue

        # Find elevation (column) range that has data in neighborhood
        cols_with_data = np.any(neighbor_data, axis=0)
        if not np.any(cols_with_data):
            continue

        col_indices = np.where(cols_with_data)[0]
        col_min, col_max = col_indices.min(), col_indices.max()

        # Mark this row's elevation range as fillable
        fill_mask[row, col_min:col_max+1] = True

    return fill_mask


def get_tight_boundary_mask(cluster_mask: np.ndarray, method: str = 'rowwise',
                            alpha: float = 0.5, dilation: int = 1,
                            neighbor_rows: int = 2) -> np.ndarray:
    """
    Get fill boundary mask using specified method.

    Args:
        cluster_mask: Binary mask of cluster cells
        method: 'convex', 'alpha', 'morphological', or 'rowwise'
        alpha: Alpha parameter for alpha shapes (lower = tighter)
        dilation: Dilation iterations for morphological method
        neighbor_rows: Neighbor rows for rowwise method

    Returns:
        Binary mask defining where holes can be filled
    """
    if method == 'convex':
        return get_convex_hull_mask(cluster_mask)
    elif method == 'alpha':
        return get_alpha_shape_mask(cluster_mask, alpha=alpha)
    elif method == 'morphological':
        return get_morphological_fill_mask(cluster_mask, dilation=dilation)
    elif method == 'rowwise':
        return get_rowwise_fill_mask(cluster_mask, neighbor_rows=neighbor_rows)
    else:
        # Default to rowwise (best for cliff data)
        return get_rowwise_fill_mask(cluster_mask, neighbor_rows=neighbor_rows)


def interpolate_holes_in_cluster(grid: np.ndarray, clusters: np.ndarray,
                                  cluster_id: int, min_neighbors: int = 3,
                                  hull_method: str = 'rowwise',
                                  alpha: float = 0.5, dilation: int = 1,
                                  neighbor_rows: int = 2) -> np.ndarray:
    """
    Fill holes within a cluster using distance transform nearest-neighbor.

    This method is GUARANTEED to fill every cell inside the boundary by
    using scipy.ndimage.distance_transform_edt to find the nearest cell
    with data for every empty cell.

    Args:
        grid: M3C2 values grid (n_rows, n_cols)
        clusters: Cluster ID grid (n_rows, n_cols)
        cluster_id: ID of cluster to fill
        min_neighbors: (unused, kept for API compatibility)
        hull_method: 'convex', 'alpha', 'morphological', or 'rowwise'
        alpha: Alpha parameter for alpha shapes
        dilation: Dilation iterations for morphological method
        neighbor_rows: Neighbor rows for rowwise method

    Returns:
        filled_grid: Grid with holes filled for this cluster
    """
    filled_grid = grid.copy()

    # Get cluster mask and its boundary
    cluster_mask = get_cluster_mask(clusters, cluster_id)

    if np.sum(cluster_mask) < 4:
        return filled_grid

    # Get boundary mask - defines region where holes can be filled
    hull_mask = get_tight_boundary_mask(cluster_mask, method=hull_method,
                                        alpha=alpha, dilation=dilation,
                                        neighbor_rows=neighbor_rows)

    # Valid data points (non-zero, non-nan within cluster)
    has_data = (grid != 0) & ~np.isnan(grid)
    valid_in_cluster = cluster_mask & has_data

    if np.sum(valid_in_cluster) < 4:
        return filled_grid

    # Holes are: within hull boundary, but currently have no data
    hole_mask = hull_mask & ~has_data

    if not np.any(hole_mask):
        return filled_grid

    # Use distance transform to find nearest cell with data for EVERY position
    # distance_transform_edt with return_indices=True gives us the indices
    # of the nearest non-zero cell for every cell in the grid
    # We need to invert the mask: True where there IS data (so distance=0 there)
    _, nearest_indices = distance_transform_edt(~has_data, return_indices=True)

    # For each hole cell inside the boundary, copy value from nearest data cell
    hole_rows, hole_cols = np.where(hole_mask)

    for i in range(len(hole_rows)):
        r, c = hole_rows[i], hole_cols[i]
        # Get indices of nearest cell that has data
        nearest_r = nearest_indices[0, r, c]
        nearest_c = nearest_indices[1, r, c]
        # Copy the value
        filled_grid[r, c] = grid[nearest_r, nearest_c]

    return filled_grid


def fill_grid_index_based(grid: np.ndarray, clusters: np.ndarray,
                          min_cluster_cells: int = 10,
                          min_cluster_volume: float = 1.0,
                          hull_method: str = 'rowwise',
                          alpha: float = 0.5,
                          dilation: int = 1,
                          neighbor_rows: int = 2) -> tuple:
    """
    Main function: Fill holes in all clusters using index-based interpolation.

    This function:
    1. Processes each cluster individually using the specified hull method
    2. Performs a GLOBAL cleanup pass to fill ANY remaining empty cells
       within the overall data envelope (row-wise min to max)

    Args:
        grid: M3C2 values grid (n_rows, n_cols)
        clusters: Cluster ID grid (n_rows, n_cols)
        min_cluster_cells: Skip clusters smaller than this
        min_cluster_volume: Skip clusters with volume less than this (m³)
        hull_method: 'convex', 'alpha', or 'morphological'
        alpha: Alpha parameter for alpha shapes (lower = tighter)
        dilation: Dilation iterations for morphological method

    Returns:
        filled_grid: Grid with holes filled
        filled_clusters: Updated cluster grid
        stats: Dictionary with filling statistics
    """
    filled_grid = grid.copy()
    filled_clusters = clusters.copy()

    # Find unique cluster IDs
    valid_mask = ~np.isnan(clusters) & (clusters != 0)
    unique_clusters = np.unique(clusters[valid_mask])

    stats = {
        'total_clusters': len(unique_clusters),
        'clusters_processed': 0,
        'clusters_skipped': 0,
        'holes_filled': 0,
        'original_nonzero': np.sum((grid != 0) & ~np.isnan(grid)),
        'hull_method': hull_method,
    }

    cell_area = CELL_SIZE * CELL_SIZE  # Assuming square cells for volume

    for cluster_id in unique_clusters:
        cluster_id = int(cluster_id)
        cluster_mask = clusters == cluster_id

        # Check minimum size
        cluster_cells = np.sum(cluster_mask & (grid != 0) & ~np.isnan(grid))
        if cluster_cells < min_cluster_cells:
            stats['clusters_skipped'] += 1
            continue

        # Check minimum volume
        cluster_values = grid[cluster_mask & (grid != 0) & ~np.isnan(grid)]
        if len(cluster_values) > 0:
            cluster_volume = np.sum(np.abs(cluster_values)) * cell_area
            if cluster_volume < min_cluster_volume:
                stats['clusters_skipped'] += 1
                continue

        # Fill holes for this cluster
        before_nonzero = np.sum((filled_grid != 0) & ~np.isnan(filled_grid))
        filled_grid = interpolate_holes_in_cluster(
            filled_grid, clusters, cluster_id,
            hull_method=hull_method, alpha=alpha, dilation=dilation,
            neighbor_rows=neighbor_rows
        )
        after_nonzero = np.sum((filled_grid != 0) & ~np.isnan(filled_grid))

        holes_filled_this_cluster = after_nonzero - before_nonzero
        stats['holes_filled'] += holes_filled_this_cluster

        # Update cluster assignments for filled holes
        if holes_filled_this_cluster > 0:
            hull_mask = get_tight_boundary_mask(cluster_mask, method=hull_method,
                                                alpha=alpha, dilation=dilation,
                                                neighbor_rows=neighbor_rows)
            new_data_mask = hull_mask & (filled_grid != 0) & ~np.isnan(filled_grid) & (filled_clusters == 0)
            filled_clusters[new_data_mask] = cluster_id

        stats['clusters_processed'] += 1

    stats['final_nonzero'] = np.sum((filled_grid != 0) & ~np.isnan(filled_grid))
    stats['percent_increase'] = (stats['final_nonzero'] - stats['original_nonzero']) / max(1, stats['original_nonzero']) * 100

    return filled_grid, filled_clusters, stats


# ============================================================================
# TESTING AND VISUALIZATION
# ============================================================================

def event_dates_to_folder(start_date: str, end_date: str) -> str:
    """Convert event dates to folder name."""
    start = pd.to_datetime(start_date).strftime('%Y%m%d')
    end = pd.to_datetime(end_date).strftime('%Y%m%d')
    return f"{start}_to_{end}"


def find_grid_files(server_results: str, location: str, date_folder: str) -> tuple:
    """Find unfilled and filled grid CSV files."""
    res_folder = os.path.join(server_results, location, 'erosion',
                              date_folder, RESOLUTION)

    if not os.path.isdir(res_folder):
        return None, None, None

    unfilled_path = None
    filled_path = None
    clusters_path = None

    for f in os.listdir(res_folder):
        if f.endswith(f'_grid_{RESOLUTION}.csv') and '_filled' not in f:
            unfilled_path = os.path.join(res_folder, f)
        elif f.endswith(f'_grid_{RESOLUTION}_filled.csv'):
            filled_path = os.path.join(res_folder, f)
        elif f.endswith(f'_clusters_{RESOLUTION}.csv') and '_filled' not in f:
            clusters_path = os.path.join(res_folder, f)

    return unfilled_path, filled_path, clusters_path


def get_zoom_indices(event, alongshore_m, elevation_m, x_pad_m=15, y_pad_bottom=8, y_pad_top=3):
    """Get zoom indices for an event."""
    x_min = event['alongshore_start_m'] - x_pad_m
    x_max = event['alongshore_end_m'] + x_pad_m
    y_min = max(0, event['elevation'] - event['height']/2 - y_pad_bottom)
    y_max = event['elevation'] + event['height']/2 + y_pad_top

    x_mask = (alongshore_m >= x_min) & (alongshore_m <= x_max)
    y_mask = (elevation_m >= y_min) & (elevation_m <= y_max)

    if not np.any(x_mask) or not np.any(y_mask):
        return None, None, None, None

    x_idx = np.where(x_mask)[0]
    y_idx = np.where(y_mask)[0]

    return x_idx.min(), x_idx.max() + 1, y_idx.min(), y_idx.max() + 1


def compute_cluster_hulls(clusters: np.ndarray, grid: np.ndarray, min_cells: int = 10,
                          hull_method: str = 'rowwise', alpha: float = 0.5,
                          dilation: int = 1, neighbor_rows: int = 2) -> dict:
    """
    Compute boundary masks for all significant clusters.

    Returns:
        dict mapping cluster_id -> hull_mask (boolean array)
    """
    hulls = {}
    valid_mask = ~np.isnan(clusters) & (clusters != 0)
    unique_clusters = np.unique(clusters[valid_mask])

    for cluster_id in unique_clusters:
        cluster_id = int(cluster_id)
        cluster_mask = (clusters == cluster_id) & (grid != 0) & ~np.isnan(grid)

        if np.sum(cluster_mask) < min_cells:
            continue

        hull_mask = get_tight_boundary_mask(cluster_mask, method=hull_method,
                                            alpha=alpha, dilation=dilation,
                                            neighbor_rows=neighbor_rows)
        hulls[cluster_id] = hull_mask

    return hulls


def plot_boundary_comparison(grid, clusters, alongshore_m, elevation_m,
                              event, location, event_idx, output_path,
                              dilation: int = 1, smooth_sigma: float = 2.0):
    """
    Create 2x2 panel comparison of boundary smoothing methods:
    1. Original (no smoothing)
    2. Opening + Closing
    3. Gaussian blur + threshold
    4. Circular convolution + threshold
    """
    # Get zoom extent
    zoom = get_zoom_indices(event, alongshore_m, elevation_m)
    if zoom[0] is None:
        print(f"    WARNING: Could not compute zoom for event {event_idx}")
        return False

    x_min, x_max, y_min, y_max = zoom

    # Extract zoomed grid (transpose for imshow: elevation on y-axis)
    grid_zoom = grid[x_min:x_max, y_min:y_max].T

    alongshore_zoom = alongshore_m[x_min:x_max]
    elevation_zoom = elevation_m[y_min:y_max]

    # Replace NaN with 0
    grid_zoom = np.nan_to_num(grid_zoom, nan=0.0)

    # Compute boundary masks with different smoothing
    valid_mask = ~np.isnan(clusters) & (clusters != 0)
    unique_clusters = np.unique(clusters[valid_mask])

    def get_base_mask():
        """Get the base fill mask (original morphological)."""
        combined_mask = np.zeros_like(grid, dtype=bool)
        for cluster_id in unique_clusters:
            cluster_id = int(cluster_id)
            cluster_mask = (clusters == cluster_id) & (grid != 0) & ~np.isnan(grid)
            if np.sum(cluster_mask) < 8:
                continue
            hull_mask = get_morphological_fill_mask(cluster_mask, dilation=dilation,
                                                     smooth_close=0, smooth_open=0)
            combined_mask |= hull_mask
        return combined_mask

    def mask_to_boundary(mask_zoom):
        """Convert a mask to boundary overlay."""
        if not np.any(mask_zoom):
            return np.zeros_like(mask_zoom, dtype=float)
        interior = binary_erosion(mask_zoom, iterations=1)
        boundary = mask_zoom & ~interior
        return boundary.astype(float)

    # Get base mask on full grid
    base_mask = get_base_mask()

    # Method 1: Original (no smoothing)
    original_zoom = base_mask[x_min:x_max, y_min:y_max].T
    boundary_original = mask_to_boundary(original_zoom)

    # Method 2: Opening + Closing (morphological)
    open_close_mask = binary_closing(binary_opening(base_mask, iterations=3), iterations=3)
    open_close_zoom = open_close_mask[x_min:x_max, y_min:y_max].T
    boundary_open_close = mask_to_boundary(open_close_zoom)

    # Method 3: Gaussian blur + threshold
    gaussian_blur = gaussian_filter(base_mask.astype(float), sigma=smooth_sigma)
    gaussian_mask = gaussian_blur > 0.5
    gaussian_zoom = gaussian_mask[x_min:x_max, y_min:y_max].T
    boundary_gaussian = mask_to_boundary(gaussian_zoom)

    # Method 4: Circular convolution + threshold
    # Create circular kernel
    kernel_radius = int(np.ceil(smooth_sigma * 2))
    y_k, x_k = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]
    circular_kernel = (x_k**2 + y_k**2 <= kernel_radius**2).astype(float)
    circular_kernel /= circular_kernel.sum()  # Normalize

    conv_blur = convolve(base_mask.astype(float), circular_kernel)
    conv_mask = conv_blur > 0.5
    conv_zoom = conv_mask[x_min:x_max, y_min:y_max].T
    boundary_conv = mask_to_boundary(conv_zoom)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    cmap = colormaps.get_cmap('RdBu_r').resampled(256)
    norm = Normalize(vmin=-5, vmax=5)

    titles = [
        'ORIGINAL (no smoothing)',
        'OPENING + CLOSING (iter=3)',
        f'GAUSSIAN BLUR (σ={smooth_sigma})',
        f'CIRCULAR CONVOLUTION (r={kernel_radius})'
    ]
    boundaries = [boundary_original, boundary_open_close, boundary_gaussian, boundary_conv]
    colors = ['lime', 'cyan', 'magenta', 'yellow']

    for ax, title, boundary, color in zip(axes, titles, boundaries, colors):
        # Show grid data
        im = ax.imshow(grid_zoom, origin='lower', aspect='auto',
                       interpolation='nearest', cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Overlay boundary
        if np.any(boundary):
            ax.contour(boundary, levels=[0.5], colors=[color],
                      linewidths=2.5, linestyles='-', alpha=0.9)

        # Axis labels
        n_y, n_x = grid_zoom.shape
        n_xticks = min(5, n_x)
        if n_x > 1:
            xtick_idx = np.linspace(0, n_x - 1, n_xticks, dtype=int)
            ax.set_xticks(xtick_idx)
            ax.set_xticklabels([f'{alongshore_zoom[i]:.0f}' for i in xtick_idx])

        n_yticks = min(5, n_y)
        if n_y > 1:
            ytick_idx = np.linspace(0, n_y - 1, n_yticks, dtype=int)
            ax.set_yticks(ytick_idx)
            ax.set_yticklabels([f'{elevation_zoom[i]:.1f}' for i in ytick_idx])

        ax.invert_xaxis()
        ax.set_xlabel("Alongshore (m)", fontsize=9)
        ax.set_ylabel("Elevation (m)", fontsize=9)

        # Add crosshairs
        if len(alongshore_zoom) > 0 and len(elevation_zoom) > 0:
            x_cross = np.argmin(np.abs(alongshore_zoom - event['alongshore_centroid_m']))
            y_cross = np.argmin(np.abs(elevation_zoom - event['elevation']))
            ax.axhline(y_cross, color='#666666', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(x_cross, color='#666666', linestyle='--', linewidth=1, alpha=0.5)

    # Title
    title = (f"{location} - Event #{event_idx} | Vol={event['volume']:.2f} m³\n"
             f"{event['start_date']} to {event['end_date']} | Boundary Smoothing Comparison")
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Adjust layout for colorbar at bottom
    plt.subplots_adjust(bottom=0.08, top=0.90, wspace=0.25, hspace=0.25)

    # Add horizontal colorbar at the bottom center
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('M3C2 (m)', fontsize=10)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {os.path.basename(output_path)}")
    return True


def iterative_diffusion_fill(grid: np.ndarray, boundary_mask: np.ndarray,
                              max_iterations: int = 1000) -> np.ndarray:
    """
    Fill empty cells inside boundary using iterative diffusion.

    For each empty cell inside the boundary, replace with mean of valid neighbors.
    Only considers neighbors that are also inside the boundary.
    Iterates until all empty cells inside boundary are filled.
    Falls back to nearest-neighbor for any remaining disconnected cells.

    Args:
        grid: Data grid with empty cells (0 or NaN)
        boundary_mask: Boolean mask defining fill region
        max_iterations: Safety limit to prevent infinite loops

    Returns:
        Filled grid
    """
    filled = grid.copy()

    # Track which cells have valid data
    has_data = (filled != 0) & ~np.isnan(filled)

    # Cells to fill: inside boundary but no data
    to_fill = boundary_mask & ~has_data

    n_rows, n_cols = grid.shape

    for iteration in range(max_iterations):
        if not np.any(to_fill):
            break

        # Find cells that can be filled this iteration (have at least 1 valid neighbor)
        new_values = np.zeros_like(filled)
        can_fill = np.zeros_like(to_fill)

        # Get indices of cells to fill
        fill_rows, fill_cols = np.where(to_fill)

        for idx in range(len(fill_rows)):
            r, c = fill_rows[idx], fill_cols[idx]

            # Get 4-connected neighbors
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                # Check bounds
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    # Only use neighbors inside boundary AND with valid data
                    if boundary_mask[nr, nc] and has_data[nr, nc]:
                        neighbors.append(filled[nr, nc])

            if len(neighbors) > 0:
                new_values[r, c] = np.mean(neighbors)
                can_fill[r, c] = True

        # Apply new values
        if not np.any(can_fill):
            # No progress made - use nearest-neighbor fallback for remaining cells
            remaining = np.sum(to_fill)
            if remaining > 0:
                # Use distance transform to find nearest cell with data
                _, nearest_indices = distance_transform_edt(~has_data, return_indices=True)

                fill_rows_remaining, fill_cols_remaining = np.where(to_fill)
                for idx in range(len(fill_rows_remaining)):
                    r, c = fill_rows_remaining[idx], fill_cols_remaining[idx]
                    nearest_r = nearest_indices[0, r, c]
                    nearest_c = nearest_indices[1, r, c]
                    filled[r, c] = grid[nearest_r, nearest_c]

                print(f"    Fallback: {remaining} cells filled via nearest-neighbor")
            break

        filled[can_fill] = new_values[can_fill]
        has_data[can_fill] = True
        to_fill[can_fill] = False

    return filled


def plot_fill_comparison(grid, clusters, alongshore_m, elevation_m,
                         event, location, event_idx, output_path,
                         dilation: int = 1, radius: int = 4):
    """
    Create 2-panel comparison:
    1. Boundary + shaded interior
    2. Results after diffusion filling
    """
    # Get zoom extent
    zoom = get_zoom_indices(event, alongshore_m, elevation_m)
    if zoom[0] is None:
        print(f"    WARNING: Could not compute zoom for event {event_idx}")
        return False

    x_min, x_max, y_min, y_max = zoom

    # Extract zoomed grid (transpose for imshow: elevation on y-axis)
    grid_zoom = grid[x_min:x_max, y_min:y_max].T

    alongshore_zoom = alongshore_m[x_min:x_max]
    elevation_zoom = elevation_m[y_min:y_max]

    # Replace NaN with 0 for display
    grid_zoom_display = np.nan_to_num(grid_zoom, nan=0.0)

    # Compute boundary mask
    valid_mask = ~np.isnan(clusters) & (clusters != 0)
    unique_clusters = np.unique(clusters[valid_mask])

    def get_base_mask():
        """Get the base fill mask (original morphological)."""
        combined_mask = np.zeros_like(grid, dtype=bool)
        for cluster_id in unique_clusters:
            cluster_id = int(cluster_id)
            cluster_mask = (clusters == cluster_id) & (grid != 0) & ~np.isnan(grid)
            if np.sum(cluster_mask) < 8:
                continue
            hull_mask = get_morphological_fill_mask(cluster_mask, dilation=dilation,
                                                     smooth_close=0, smooth_open=0)
            combined_mask |= hull_mask
        return combined_mask

    def apply_circular_convolution(base_mask, r):
        """Apply circular convolution with given radius."""
        y_k, x_k = np.ogrid[-r:r+1, -r:r+1]
        circular_kernel = (x_k**2 + y_k**2 <= r**2).astype(float)
        circular_kernel /= circular_kernel.sum()
        conv_blur = convolve(base_mask.astype(float), circular_kernel)
        return conv_blur > 0.5

    # Get smoothed boundary mask on full grid
    base_mask = get_base_mask()
    boundary_mask = apply_circular_convolution(base_mask, radius)

    # Apply diffusion filling on full grid
    filled_grid = iterative_diffusion_fill(grid, boundary_mask)

    # Extract zoomed versions
    boundary_zoom = boundary_mask[x_min:x_max, y_min:y_max].T
    filled_zoom = filled_grid[x_min:x_max, y_min:y_max].T
    filled_zoom_display = np.nan_to_num(filled_zoom, nan=0.0)

    # Create 2-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    cmap = colormaps.get_cmap('RdBu_r').resampled(256)
    norm = Normalize(vmin=-5, vmax=5)

    # Panel 1: Original data + boundary + shading
    ax1 = axes[0]
    im1 = ax1.imshow(grid_zoom_display, origin='lower', aspect='auto',
                     interpolation='nearest', cmap=cmap, norm=norm)
    ax1.set_title('ORIGINAL + BOUNDARY (r=4)', fontsize=11, fontweight='bold')

    # Add shaded fill
    if np.any(boundary_zoom):
        overlay = np.zeros((*boundary_zoom.shape, 4))
        overlay[boundary_zoom, :3] = [0, 1, 1]  # Cyan
        overlay[boundary_zoom, 3] = 0.25  # 25% opacity
        ax1.imshow(overlay, origin='lower', aspect='auto', interpolation='nearest')
        # Draw boundary line
        ax1.contour(boundary_zoom.astype(float), levels=[0.5], colors=['cyan'],
                   linewidths=2.5, linestyles='-', alpha=0.9)

    # Panel 2: Filled data
    ax2 = axes[1]
    im2 = ax2.imshow(filled_zoom_display, origin='lower', aspect='auto',
                     interpolation='nearest', cmap=cmap, norm=norm)
    ax2.set_title('AFTER DIFFUSION FILL', fontsize=11, fontweight='bold')

    # Also show boundary on filled panel for reference
    if np.any(boundary_zoom):
        ax2.contour(boundary_zoom.astype(float), levels=[0.5], colors=['cyan'],
                   linewidths=1.5, linestyles='--', alpha=0.7)

    # Format both axes
    for ax in axes:
        n_y, n_x = grid_zoom.shape
        n_xticks = min(5, n_x)
        if n_x > 1:
            xtick_idx = np.linspace(0, n_x - 1, n_xticks, dtype=int)
            ax.set_xticks(xtick_idx)
            ax.set_xticklabels([f'{alongshore_zoom[i]:.0f}' for i in xtick_idx])

        n_yticks = min(5, n_y)
        if n_y > 1:
            ytick_idx = np.linspace(0, n_y - 1, n_yticks, dtype=int)
            ax.set_yticks(ytick_idx)
            ax.set_yticklabels([f'{elevation_zoom[i]:.1f}' for i in ytick_idx])

        ax.invert_xaxis()
        ax.set_xlabel("Alongshore (m)", fontsize=9)
        ax.set_ylabel("Elevation (m)", fontsize=9)

        # Add crosshairs
        if len(alongshore_zoom) > 0 and len(elevation_zoom) > 0:
            x_cross = np.argmin(np.abs(alongshore_zoom - event['alongshore_centroid_m']))
            y_cross = np.argmin(np.abs(elevation_zoom - event['elevation']))
            ax.axhline(y_cross, color='#666666', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(x_cross, color='#666666', linestyle='--', linewidth=1, alpha=0.5)

    # Title
    title = (f"{location} - Event #{event_idx} | Vol={event['volume']:.2f} m³\n"
             f"{event['start_date']} to {event['end_date']} | Diffusion Fill Test")
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Adjust layout for colorbar at bottom
    plt.subplots_adjust(bottom=0.12, top=0.88, wspace=0.20)

    # Add horizontal colorbar at the bottom center
    cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.025])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('M3C2 (m)', fontsize=10)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Print stats
    original_nonzero = np.sum(grid_zoom_display != 0)
    filled_nonzero = np.sum(filled_zoom_display != 0)
    boundary_cells = np.sum(boundary_zoom)

    print(f"  Saved: {os.path.basename(output_path)}")
    print(f"    Original: {original_nonzero} cells, Filled: {filled_nonzero} cells, Boundary: {boundary_cells} cells")
    print(f"    Fill rate: {filled_nonzero / max(1, boundary_cells) * 100:.1f}% of boundary filled")

    return True


def test_diffusion_fill(location: str, top_n: int = 10, start_idx: int = 0,
                        dilation: int = 1, radius: int = 4):
    """Test diffusion filling on events.

    Args:
        location: Site name
        top_n: Number of events to process
        start_idx: Starting index (0-based, after sorting by volume)
        dilation: Dilation for base mask
        radius: Convolution radius for smoothing
    """
    print(f"\n{'='*70}")
    print(f"Testing Diffusion Fill: {location}")
    print(f"Events {start_idx+1} to {start_idx+top_n} (by volume)")
    print(f"Dilation: {dilation}, Convolution radius: {radius}")
    print(f"{'='*70}")

    # Check server
    server_results = os.path.join(SERVER_BASE, 'results')
    if not os.path.isdir(server_results):
        print(f"ERROR: Server not mounted at {SERVER_BASE}")
        return

    # Load alongshore positions
    alongshore_positions = load_polygon_alongshore(location)
    if alongshore_positions is None:
        print(f"ERROR: Could not load shapefile for {location}")
        return

    print(f"  Loaded {len(alongshore_positions)} polygon positions")

    # Load events
    event_csv = os.path.join(EVENT_LISTS_DIR, f"{location}_events.csv")
    if not os.path.exists(event_csv):
        print(f"ERROR: Event list not found: {event_csv}")
        return

    events_df = pd.read_csv(event_csv)
    events_df = events_df.sort_values('volume', ascending=False).reset_index(drop=True)
    print(f"  Loaded {len(events_df)} events")

    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, 'diffusion_fill')
    os.makedirs(output_dir, exist_ok=True)

    end_idx = min(start_idx + top_n, len(events_df))
    n_events = end_idx - start_idx
    print(f"  Processing events {start_idx+1} to {end_idx}...\n")

    for i in range(start_idx, end_idx):
        event = events_df.iloc[i]
        date_folder = event_dates_to_folder(event['start_date'], event['end_date'])

        print(f"Event {i+1}: {date_folder} (Vol={event['volume']:.2f} m³)")

        # Find files
        unfilled_path, _, clusters_path = find_grid_files(
            server_results, location, date_folder)

        if unfilled_path is None or clusters_path is None:
            print(f"  WARNING: Missing grid files for {date_folder}")
            continue

        # Load grids
        unfilled_grid, unfilled_rows, unfilled_headers = load_grid_csv(unfilled_path)
        clusters_grid, clusters_rows, _ = load_grid_csv(clusters_path)

        if unfilled_grid is None or clusters_grid is None:
            print(f"  WARNING: Could not load grids")
            continue

        # Convert to sorted arrays for plotting
        grid_sorted, alongshore_m, elevation_m = grids_to_sorted_arrays(
            unfilled_grid, unfilled_rows, unfilled_headers, alongshore_positions)
        clusters_sorted, _, _ = grids_to_sorted_arrays(
            clusters_grid, clusters_rows, unfilled_headers, alongshore_positions)

        # Create fill comparison figure
        output_path = os.path.join(output_dir, f"{location}_event{i+1}_diffusion_fill.png")

        plot_fill_comparison(
            grid_sorted, clusters_sorted, alongshore_m, elevation_m,
            event, location, i+1, output_path,
            dilation=dilation, radius=radius
        )
        print()

    print(f"{'='*70}")
    print(f"Done! Figures saved to: {output_dir}")
    print(f"{'='*70}")


def plot_convolution_radius_comparison(grid, clusters, alongshore_m, elevation_m,
                                        event, location, event_idx, output_path,
                                        dilation: int = 1, radii: list = [1, 2, 4, 6]):
    """
    Create 2x2 panel comparison of circular convolution with different radii.
    """
    # Get zoom extent
    zoom = get_zoom_indices(event, alongshore_m, elevation_m)
    if zoom[0] is None:
        print(f"    WARNING: Could not compute zoom for event {event_idx}")
        return False

    x_min, x_max, y_min, y_max = zoom

    # Extract zoomed grid (transpose for imshow: elevation on y-axis)
    grid_zoom = grid[x_min:x_max, y_min:y_max].T

    alongshore_zoom = alongshore_m[x_min:x_max]
    elevation_zoom = elevation_m[y_min:y_max]

    # Replace NaN with 0
    grid_zoom = np.nan_to_num(grid_zoom, nan=0.0)

    # Compute boundary masks
    valid_mask = ~np.isnan(clusters) & (clusters != 0)
    unique_clusters = np.unique(clusters[valid_mask])

    def get_base_mask():
        """Get the base fill mask (original morphological)."""
        combined_mask = np.zeros_like(grid, dtype=bool)
        for cluster_id in unique_clusters:
            cluster_id = int(cluster_id)
            cluster_mask = (clusters == cluster_id) & (grid != 0) & ~np.isnan(grid)
            if np.sum(cluster_mask) < 8:
                continue
            hull_mask = get_morphological_fill_mask(cluster_mask, dilation=dilation,
                                                     smooth_close=0, smooth_open=0)
            combined_mask |= hull_mask
        return combined_mask

    def apply_circular_convolution(base_mask, radius):
        """Apply circular convolution with given radius."""
        y_k, x_k = np.ogrid[-radius:radius+1, -radius:radius+1]
        circular_kernel = (x_k**2 + y_k**2 <= radius**2).astype(float)
        circular_kernel /= circular_kernel.sum()  # Normalize

        conv_blur = convolve(base_mask.astype(float), circular_kernel)
        return conv_blur > 0.5

    # Get base mask on full grid
    base_mask = get_base_mask()

    # Compute masks for each radius (contour the mask directly, not a boundary ring)
    masks = []
    for r in radii:
        conv_mask = apply_circular_convolution(base_mask, r)
        conv_zoom = conv_mask[x_min:x_max, y_min:y_max].T
        masks.append(conv_zoom.astype(float))

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    cmap = colormaps.get_cmap('RdBu_r').resampled(256)
    norm = Normalize(vmin=-5, vmax=5)

    colors = ['lime', 'cyan', 'magenta', 'yellow']

    for ax, r, mask, color in zip(axes, radii, masks, colors):
        # Show grid data
        im = ax.imshow(grid_zoom, origin='lower', aspect='auto',
                       interpolation='nearest', cmap=cmap, norm=norm)
        ax.set_title(f'CIRCULAR CONVOLUTION (r={r})', fontsize=11, fontweight='bold')

        # Overlay shaded fill to show area inside boundary
        if np.any(mask):
            # Create RGBA overlay - semi-transparent fill
            overlay = np.zeros((*mask.shape, 4))
            # Convert color name to RGB
            from matplotlib.colors import to_rgb
            rgb = to_rgb(color)
            overlay[mask > 0.5, :3] = rgb
            overlay[mask > 0.5, 3] = 0.3  # 30% opacity for fill
            ax.imshow(overlay, origin='lower', aspect='auto', interpolation='nearest')

            # Draw boundary line on top
            ax.contour(mask, levels=[0.5], colors=[color],
                      linewidths=2.5, linestyles='-', alpha=0.9)

        # Axis labels
        n_y, n_x = grid_zoom.shape
        n_xticks = min(5, n_x)
        if n_x > 1:
            xtick_idx = np.linspace(0, n_x - 1, n_xticks, dtype=int)
            ax.set_xticks(xtick_idx)
            ax.set_xticklabels([f'{alongshore_zoom[i]:.0f}' for i in xtick_idx])

        n_yticks = min(5, n_y)
        if n_y > 1:
            ytick_idx = np.linspace(0, n_y - 1, n_yticks, dtype=int)
            ax.set_yticks(ytick_idx)
            ax.set_yticklabels([f'{elevation_zoom[i]:.1f}' for i in ytick_idx])

        ax.invert_xaxis()
        ax.set_xlabel("Alongshore (m)", fontsize=9)
        ax.set_ylabel("Elevation (m)", fontsize=9)

        # Add crosshairs
        if len(alongshore_zoom) > 0 and len(elevation_zoom) > 0:
            x_cross = np.argmin(np.abs(alongshore_zoom - event['alongshore_centroid_m']))
            y_cross = np.argmin(np.abs(elevation_zoom - event['elevation']))
            ax.axhline(y_cross, color='#666666', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(x_cross, color='#666666', linestyle='--', linewidth=1, alpha=0.5)

    # Title
    title = (f"{location} - Event #{event_idx} | Vol={event['volume']:.2f} m³\n"
             f"{event['start_date']} to {event['end_date']} | Circular Convolution Radius Comparison")
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Adjust layout for colorbar at bottom
    plt.subplots_adjust(bottom=0.08, top=0.90, wspace=0.25, hspace=0.25)

    # Add horizontal colorbar at the bottom center
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('M3C2 (m)', fontsize=10)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {os.path.basename(output_path)}")
    return True


def test_convolution_radii(location: str, top_n: int = 5, dilation: int = 1,
                           radii: list = [1, 2, 4, 6]):
    """Test circular convolution with different radii on top events."""
    print(f"\n{'='*70}")
    print(f"Testing Convolution Radii: {location}")
    print(f"Dilation: {dilation}, Radii: {radii}")
    print(f"{'='*70}")

    # Check server
    server_results = os.path.join(SERVER_BASE, 'results')
    if not os.path.isdir(server_results):
        print(f"ERROR: Server not mounted at {SERVER_BASE}")
        return

    # Load alongshore positions
    alongshore_positions = load_polygon_alongshore(location)
    if alongshore_positions is None:
        print(f"ERROR: Could not load shapefile for {location}")
        return

    print(f"  Loaded {len(alongshore_positions)} polygon positions")

    # Load events
    event_csv = os.path.join(EVENT_LISTS_DIR, f"{location}_events.csv")
    if not os.path.exists(event_csv):
        print(f"ERROR: Event list not found: {event_csv}")
        return

    events_df = pd.read_csv(event_csv)
    events_df = events_df.sort_values('volume', ascending=False).reset_index(drop=True)
    print(f"  Loaded {len(events_df)} events")

    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, 'boundary')
    os.makedirs(output_dir, exist_ok=True)

    n_events = min(top_n, len(events_df))
    print(f"  Processing top {n_events} events...\n")

    for i in range(n_events):
        event = events_df.iloc[i]
        date_folder = event_dates_to_folder(event['start_date'], event['end_date'])

        print(f"Event {i+1}: {date_folder} (Vol={event['volume']:.2f} m³)")

        # Find files
        unfilled_path, _, clusters_path = find_grid_files(
            server_results, location, date_folder)

        if unfilled_path is None or clusters_path is None:
            print(f"  WARNING: Missing grid files for {date_folder}")
            continue

        # Load grids
        unfilled_grid, unfilled_rows, unfilled_headers = load_grid_csv(unfilled_path)
        clusters_grid, clusters_rows, _ = load_grid_csv(clusters_path)

        if unfilled_grid is None or clusters_grid is None:
            print(f"  WARNING: Could not load grids")
            continue

        # Convert to sorted arrays for plotting
        grid_sorted, alongshore_m, elevation_m = grids_to_sorted_arrays(
            unfilled_grid, unfilled_rows, unfilled_headers, alongshore_positions)
        clusters_sorted, _, _ = grids_to_sorted_arrays(
            clusters_grid, clusters_rows, unfilled_headers, alongshore_positions)

        # Create convolution radius comparison figure
        output_path = os.path.join(output_dir, f"{location}_event{i+1}_conv_radii.png")

        plot_convolution_radius_comparison(
            grid_sorted, clusters_sorted, alongshore_m, elevation_m,
            event, location, i+1, output_path,
            dilation=dilation, radii=radii
        )
        print()

    print(f"{'='*70}")
    print(f"Done! Figures saved to: {output_dir}")
    print(f"{'='*70}")


def test_boundary_smoothing(location: str, top_n: int = 5, dilation: int = 1,
                            smooth_sigma: float = 2.0):
    """Test boundary smoothing on top events."""
    print(f"\n{'='*70}")
    print(f"Testing Boundary Smoothing: {location}")
    print(f"Dilation: {dilation}, Smooth sigma: {smooth_sigma}")
    print(f"{'='*70}")

    # Check server
    server_results = os.path.join(SERVER_BASE, 'results')
    if not os.path.isdir(server_results):
        print(f"ERROR: Server not mounted at {SERVER_BASE}")
        return

    # Load alongshore positions
    alongshore_positions = load_polygon_alongshore(location)
    if alongshore_positions is None:
        print(f"ERROR: Could not load shapefile for {location}")
        return

    print(f"  Loaded {len(alongshore_positions)} polygon positions")

    # Load events
    event_csv = os.path.join(EVENT_LISTS_DIR, f"{location}_events.csv")
    if not os.path.exists(event_csv):
        print(f"ERROR: Event list not found: {event_csv}")
        return

    events_df = pd.read_csv(event_csv)
    events_df = events_df.sort_values('volume', ascending=False).reset_index(drop=True)
    print(f"  Loaded {len(events_df)} events")

    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, 'boundary')
    os.makedirs(output_dir, exist_ok=True)

    n_events = min(top_n, len(events_df))
    print(f"  Processing top {n_events} events...\n")

    for i in range(n_events):
        event = events_df.iloc[i]
        date_folder = event_dates_to_folder(event['start_date'], event['end_date'])

        print(f"Event {i+1}: {date_folder} (Vol={event['volume']:.2f} m³)")

        # Find files
        unfilled_path, _, clusters_path = find_grid_files(
            server_results, location, date_folder)

        if unfilled_path is None or clusters_path is None:
            print(f"  WARNING: Missing grid files for {date_folder}")
            continue

        # Load grids
        unfilled_grid, unfilled_rows, unfilled_headers = load_grid_csv(unfilled_path)
        clusters_grid, clusters_rows, _ = load_grid_csv(clusters_path)

        if unfilled_grid is None or clusters_grid is None:
            print(f"  WARNING: Could not load grids")
            continue

        # Convert to sorted arrays for plotting
        grid_sorted, alongshore_m, elevation_m = grids_to_sorted_arrays(
            unfilled_grid, unfilled_rows, unfilled_headers, alongshore_positions)
        clusters_sorted, _, _ = grids_to_sorted_arrays(
            clusters_grid, clusters_rows, unfilled_headers, alongshore_positions)

        # Create boundary comparison figure
        output_path = os.path.join(output_dir, f"{location}_event{i+1}_boundary_smoothing.png")

        plot_boundary_comparison(
            grid_sorted, clusters_sorted, alongshore_m, elevation_m,
            event, location, i+1, output_path,
            dilation=dilation, smooth_sigma=smooth_sigma
        )
        print()

    print(f"{'='*70}")
    print(f"Done! Figures saved to: {output_dir}")
    print(f"{'='*70}")


def plot_three_way_comparison(cleaned, old_filled, new_filled, alongshore_m, elevation_m,
                               event, location, event_idx, output_path,
                               clusters=None, show_hulls=True,
                               hull_method='rowwise', alpha=0.5, dilation=1,
                               neighbor_rows=2):
    """
    Create 3-panel comparison: Cleaned vs Old Filled vs New Filled.
    Optionally overlay boundary masks.
    """
    # Get zoom extent
    zoom = get_zoom_indices(event, alongshore_m, elevation_m)
    if zoom[0] is None:
        print(f"    WARNING: Could not compute zoom for event {event_idx}")
        return False

    x_min, x_max, y_min, y_max = zoom

    # Extract zoomed regions (transpose for imshow: elevation on y-axis)
    cleaned_zoom = cleaned[x_min:x_max, y_min:y_max].T
    old_zoom = old_filled[x_min:x_max, y_min:y_max].T
    new_zoom = new_filled[x_min:x_max, y_min:y_max].T

    alongshore_zoom = alongshore_m[x_min:x_max]
    elevation_zoom = elevation_m[y_min:y_max]

    # Replace NaN with 0
    cleaned_zoom = np.nan_to_num(cleaned_zoom, nan=0.0)
    old_zoom = np.nan_to_num(old_zoom, nan=0.0)
    new_zoom = np.nan_to_num(new_zoom, nan=0.0)

    # Compute hull masks if clusters provided
    hull_overlay = None
    if show_hulls and clusters is not None:
        # Compute hulls on full grid using specified method
        hulls = compute_cluster_hulls(clusters, cleaned, min_cells=8,
                                      hull_method=hull_method, alpha=alpha,
                                      dilation=dilation, neighbor_rows=neighbor_rows)

        # Create combined hull boundary overlay for the zoomed region
        hull_overlay = np.zeros_like(cleaned_zoom, dtype=float)
        clusters_zoom = clusters[x_min:x_max, y_min:y_max].T

        for cluster_id, hull_mask in hulls.items():
            # Extract zoomed hull
            hull_zoom = hull_mask[x_min:x_max, y_min:y_max].T

            # Get boundary (edge) of hull using gradient
            if np.any(hull_zoom):
                # Create boundary by finding edges
                from scipy.ndimage import binary_erosion
                interior = binary_erosion(hull_zoom, iterations=1)
                boundary = hull_zoom & ~interior
                hull_overlay[boundary] = 1.0

    # Create figure with space for horizontal colorbar at bottom
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    cmap = colormaps.get_cmap('RdBu_r').resampled(256)
    norm = Normalize(vmin=-5, vmax=5)

    titles = ['CLEANED (Step 7)', 'OLD FILLED (UTM-based)', 'NEW FILLED (Index-based)']
    data = [cleaned_zoom, old_zoom, new_zoom]

    for ax, title, d in zip(axes, titles, data):
        im = ax.imshow(d, origin='lower', aspect='auto',
                       interpolation='nearest', cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Overlay hull boundaries on all panels
        if hull_overlay is not None:
            # Plot hull boundaries as green contour lines
            ax.contour(hull_overlay, levels=[0.5], colors=['lime'],
                      linewidths=1.5, linestyles='-', alpha=0.8)

        # Axis labels
        n_y, n_x = d.shape
        n_xticks = min(6, n_x)
        if n_x > 1:
            xtick_idx = np.linspace(0, n_x - 1, n_xticks, dtype=int)
            ax.set_xticks(xtick_idx)
            ax.set_xticklabels([f'{alongshore_zoom[i]:.0f}' for i in xtick_idx])

        n_yticks = min(6, n_y)
        if n_y > 1:
            ytick_idx = np.linspace(0, n_y - 1, n_yticks, dtype=int)
            ax.set_yticks(ytick_idx)
            ax.set_yticklabels([f'{elevation_zoom[i]:.1f}' for i in ytick_idx])

        ax.invert_xaxis()
        ax.set_xlabel("Alongshore (m)", fontsize=10)
        ax.set_ylabel("Elevation (m)", fontsize=10)

        # Add crosshairs
        if len(alongshore_zoom) > 0 and len(elevation_zoom) > 0:
            x_cross = np.argmin(np.abs(alongshore_zoom - event['alongshore_centroid_m']))
            y_cross = np.argmin(np.abs(elevation_zoom - event['elevation']))
            ax.axhline(y_cross, color='#666666', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(x_cross, color='#666666', linestyle='--', linewidth=1, alpha=0.5)

    # Add legend for fill boundary on third panel
    if hull_overlay is not None:
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='lime', linewidth=2, label='Fill Boundary')]
        axes[2].legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Title
    title = (f"{location} - Event #{event_idx} | Vol={event['volume']:.2f} m³\n"
             f"{event['start_date']} to {event['end_date']}")
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Adjust layout to make room for colorbar at bottom
    plt.subplots_adjust(bottom=0.15, top=0.88, wspace=0.25)

    # Add horizontal colorbar at the bottom center
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.025])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('M3C2 (m)', fontsize=11)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Print stats
    cleaned_nz = np.sum(cleaned_zoom != 0)
    old_nz = np.sum(old_zoom != 0)
    new_nz = np.sum(new_zoom != 0)

    print(f"  Saved: {os.path.basename(output_path)}")
    print(f"    Cleaned: {cleaned_nz} cells, Old filled: {old_nz} cells, New filled: {new_nz} cells")
    print(f"    Old increase: {(old_nz - cleaned_nz) / max(1, cleaned_nz) * 100:.1f}%, "
          f"New increase: {(new_nz - cleaned_nz) / max(1, cleaned_nz) * 100:.1f}%")

    return True


def load_polygon_alongshore(location: str) -> dict:
    """Load polygon alongshore positions from shapefile."""
    import geopandas as gpd

    util_dir = os.path.join(SERVER_BASE, 'utilities', 'shape_files')

    candidates = [
        d for d in os.listdir(util_dir)
        if d.lower().startswith(location.lower())
           and 'polygon' in d.lower()
           and f'at{RESOLUTION}'.lower() in d.lower()
           and os.path.isdir(os.path.join(util_dir, d))
    ]

    if not candidates:
        return None

    fld = candidates[0]
    shp_path = os.path.join(util_dir, fld, fld + '.shp')

    if not os.path.isfile(shp_path):
        return None

    gdf = gpd.read_file(shp_path)
    gdf['Polygon_ID'] = gdf.index
    centroids = gdf.geometry.centroid

    coords = {int(pid): (c.x, c.y) for pid, c in zip(gdf['Polygon_ID'], centroids)}
    all_y = [c[1] for c in coords.values()]
    min_y = min(all_y)

    return {pid: coord[1] - min_y for pid, coord in coords.items()}


def grids_to_sorted_arrays(grid_data, row_ids, headers, alongshore_positions):
    """Convert grid to sorted 2D array with coordinate arrays."""
    elevations = np.array([parse_elevation_from_header(h) for h in headers])
    alongshore_vals = np.array([alongshore_positions.get(pid, np.nan) for pid in row_ids])

    # Sort by alongshore
    sort_idx = np.argsort(alongshore_vals)

    sorted_grid = grid_data[sort_idx, :]
    sorted_alongshore = alongshore_vals[sort_idx]

    return sorted_grid, sorted_alongshore, elevations


def test_on_events(location: str, top_n: int = 5, hull_method: str = 'rowwise',
                   alpha: float = 0.5, dilation: int = 1, neighbor_rows: int = 2):
    """Test the new filling algorithm on top events."""
    print(f"\n{'='*70}")
    print(f"Testing Index-Based Hole Filling: {location}")
    method_info = f"Hull method: {hull_method}"
    if hull_method == 'alpha':
        method_info += f" (alpha={alpha})"
    elif hull_method == 'morphological':
        method_info += f" (dilation={dilation})"
    elif hull_method == 'rowwise':
        method_info += f" (neighbor_rows={neighbor_rows})"
    print(method_info)
    print(f"{'='*70}")

    # Check server
    server_results = os.path.join(SERVER_BASE, 'results')
    if not os.path.isdir(server_results):
        print(f"ERROR: Server not mounted at {SERVER_BASE}")
        return

    # Load alongshore positions
    alongshore_positions = load_polygon_alongshore(location)
    if alongshore_positions is None:
        print(f"ERROR: Could not load shapefile for {location}")
        return

    print(f"  Loaded {len(alongshore_positions)} polygon positions")

    # Load events
    event_csv = os.path.join(EVENT_LISTS_DIR, f"{location}_events.csv")
    if not os.path.exists(event_csv):
        print(f"ERROR: Event list not found: {event_csv}")
        return

    events_df = pd.read_csv(event_csv)
    events_df = events_df.sort_values('volume', ascending=False).reset_index(drop=True)
    print(f"  Loaded {len(events_df)} events")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_events = min(top_n, len(events_df))
    print(f"  Processing top {n_events} events...\n")

    for i in range(n_events):
        event = events_df.iloc[i]
        date_folder = event_dates_to_folder(event['start_date'], event['end_date'])

        print(f"Event {i+1}: {date_folder} (Vol={event['volume']:.2f} m³)")

        # Find files
        unfilled_path, filled_path, clusters_path = find_grid_files(
            server_results, location, date_folder)

        if unfilled_path is None or clusters_path is None:
            print(f"  WARNING: Missing grid files for {date_folder}")
            continue

        # Load grids
        unfilled_grid, unfilled_rows, unfilled_headers = load_grid_csv(unfilled_path)
        clusters_grid, clusters_rows, _ = load_grid_csv(clusters_path)

        if unfilled_grid is None or clusters_grid is None:
            print(f"  WARNING: Could not load grids")
            continue

        # Load old filled for comparison
        if filled_path:
            old_filled_grid, _, _ = load_grid_csv(filled_path)
        else:
            old_filled_grid = unfilled_grid.copy()

        # Apply new filling algorithm
        print(f"  Applying index-based filling ({hull_method})...")
        new_filled_grid, new_filled_clusters, stats = fill_grid_index_based(
            unfilled_grid, clusters_grid,
            min_cluster_cells=8,
            min_cluster_volume=1.0,
            hull_method=hull_method,
            alpha=alpha,
            dilation=dilation,
            neighbor_rows=neighbor_rows
        )

        print(f"    Clusters processed: {stats['clusters_processed']}, skipped: {stats['clusters_skipped']}")
        print(f"    Holes filled: {stats['holes_filled']}, increase: {stats['percent_increase']:.1f}%")

        # Convert to sorted arrays for plotting
        cleaned_sorted, alongshore_m, elevation_m = grids_to_sorted_arrays(
            unfilled_grid, unfilled_rows, unfilled_headers, alongshore_positions)
        old_filled_sorted, _, _ = grids_to_sorted_arrays(
            old_filled_grid, unfilled_rows, unfilled_headers, alongshore_positions)
        new_filled_sorted, _, _ = grids_to_sorted_arrays(
            new_filled_grid, unfilled_rows, unfilled_headers, alongshore_positions)
        clusters_sorted, _, _ = grids_to_sorted_arrays(
            clusters_grid, clusters_rows, unfilled_headers, alongshore_positions)

        # Create comparison figure with hull overlay
        output_path = os.path.join(OUTPUT_DIR, f"{location}_event{i+1}_{hull_method}_fill.png")

        plot_three_way_comparison(
            cleaned_sorted, old_filled_sorted, new_filled_sorted,
            alongshore_m, elevation_m, event, location, i+1, output_path,
            clusters=clusters_sorted, show_hulls=True,
            hull_method=hull_method, alpha=alpha, dilation=dilation,
            neighbor_rows=neighbor_rows
        )
        print()

    print(f"{'='*70}")
    print(f"Done! Figures saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Index-based grid hole filling (no UTM coordinates)')
    parser.add_argument('--location', type=str, default='DelMar',
                        help='Location to process')
    parser.add_argument('--top', type=int, default=5,
                        help='Number of events to test')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting event index (0-based, after sorting by volume)')
    parser.add_argument('--test', action='store_true',
                        help='Run test mode with visualization')
    parser.add_argument('--test-boundary', action='store_true',
                        help='Test boundary smoothing options')
    parser.add_argument('--test-radii', action='store_true',
                        help='Test circular convolution with different radii')
    parser.add_argument('--test-fill', action='store_true',
                        help='Test diffusion fill approach')
    parser.add_argument('--hull-method', type=str, default='rowwise',
                        choices=['convex', 'alpha', 'morphological', 'rowwise'],
                        help='Boundary method: rowwise (best for cliffs), morphological, alpha, convex')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha parameter for alpha shapes (lower = tighter)')
    parser.add_argument('--dilation', type=int, default=1,
                        help='Dilation iterations for morphological method (1-3)')
    parser.add_argument('--neighbor-rows', type=int, default=2,
                        help='Neighbor rows for rowwise method (1-5)')
    parser.add_argument('--smooth', type=float, default=2.0,
                        help='Smoothing sigma for Gaussian/convolution (used with --test-boundary)')
    args = parser.parse_args()

    if args.test_fill:
        test_diffusion_fill(args.location, args.top,
                            start_idx=args.start,
                            dilation=args.dilation,
                            radius=4)
    elif args.test_radii:
        test_convolution_radii(args.location, args.top,
                               dilation=args.dilation,
                               radii=[1, 2, 4, 6])
    elif args.test_boundary:
        test_boundary_smoothing(args.location, args.top,
                                dilation=args.dilation,
                                smooth_sigma=args.smooth)
    elif args.test:
        test_on_events(args.location, args.top,
                      hull_method=args.hull_method,
                      alpha=args.alpha,
                      dilation=args.dilation,
                      neighbor_rows=args.neighbor_rows)
    else:
        print("Use --test flag to run comparison tests")
        print("\nExamples:")
        print("  Rowwise (best):         python3 grid_fill_index_based.py --test --hull-method rowwise --neighbor-rows 2")
        print("  Morphological (tight):  python3 grid_fill_index_based.py --test --hull-method morphological --dilation 1")
        print("  Convex hull (loose):    python3 grid_fill_index_based.py --test --hull-method convex")
        print("  Boundary smoothing:     python3 grid_fill_index_based.py --test-boundary --smooth 2")


if __name__ == '__main__':
    main()
