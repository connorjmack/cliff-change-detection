#!/usr/bin/env python3
"""
digitize_cliff_top.py

Extracts a manually drawn cliff-top line from an annotated PNG image,
converts pixel coordinates to real-world coordinates, and outputs
CSV files at multiple resolutions for use in the pipeline.

The input image should have:
- X-axis: Alongshore Location (m)
- Y-axis: Elevation (m)
- A colored line drawn to mark the cliff top

Usage:
    # Auto-detect image, read axis extents from grid data:
    python3 digitize_cliff_top.py --location DelMar --line_color green --x_inverted

    # Specify axis extents manually (read from the image tick labels):
    python3 digitize_cliff_top.py --location DelMar \
        --x_min 0 --x_max 2800 --y_max 48 --line_color green --x_inverted

    # Specify plot area bounds manually (pixels):
    python3 digitize_cliff_top.py --location DelMar --line_color green --x_inverted \
        --plot_left 100 --plot_right 1800 --plot_top 50 --plot_bottom 400
"""

import os
import re
import argparse
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


# --- CONFIGURATION ---
RESOLUTIONS = ['1m', '25cm', '10cm']

# Elevation cutoffs per location (from 7_make_grids.py)
HEIGHTS = {
    'DelMar': 30,
    'SanElijo': 40,
    'Solana': 50,
    'Encinitas': 50,
    'Torrey': 75,
    'Blacks': 100
}

# Color definitions (RGB ranges for line detection)
COLOR_RANGES = {
    'green': {'min': (0, 150, 0), 'max': (100, 255, 100)},
    'blue': {'min': (0, 0, 150), 'max': (100, 100, 255)},
    'red': {'min': (150, 0, 0), 'max': (255, 100, 100)},
    'cyan': {'min': (0, 150, 150), 'max': (100, 255, 255)},
    'magenta': {'min': (150, 0, 150), 'max': (255, 100, 255)},
    'yellow': {'min': (150, 150, 0), 'max': (255, 255, 100)},
    'white': {'min': (240, 240, 240), 'max': (255, 255, 255)},
    'black': {'min': (0, 0, 0), 'max': (30, 30, 30)},
}


def _resolution_to_meters(resolution):
    """Convert resolution string to meters."""
    mapping = {'1m': 1.0, '100cm': 1.0, '25cm': 0.25, '10cm': 0.10}
    if resolution not in mapping:
        raise ValueError(f"Unknown resolution: {resolution}")
    return mapping[resolution]


def get_base_path():
    """Return base path based on OS."""
    if platform.system() == "Darwin":
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"


def get_output_dir(location):
    """Return output directory for cliff top cutoff CSVs."""
    base = get_base_path()
    return os.path.join(base, "utilities", "cliff_top_cutoffs", location)


def find_image_for_location(location):
    """
    Auto-detect the cliff top image for a given location.

    Expected naming pattern: {location}_clifftop.png (lowercase)
    Location: code/utilities/cliff_top_cropping/figures/manual_lines/
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "figures", "manual_lines")

    patterns = [
        f"{location.lower()}_clifftop.png",
        f"{location}_clifftop.png",
        f"{location.lower()}_cliftop.png",
        f"{location}_CliffTop.png",
    ]

    for pattern in patterns:
        candidate = os.path.join(figures_dir, pattern)
        if os.path.exists(candidate):
            return candidate

    available = []
    if os.path.isdir(figures_dir):
        available = [f for f in os.listdir(figures_dir) if f.endswith('.png')]

    raise FileNotFoundError(
        f"Could not find cliff top image for '{location}'.\n"
        f"Searched in: {figures_dir}\n"
        f"Available images:\n  " + "\n  ".join(available) if available else "(none)"
    )


def find_shapefile(location, resolution='1m'):
    """Locate the shapefile for a given location and resolution."""
    base = get_base_path()
    sf_root = os.path.join(base, 'utilities', 'shape_files')

    if not os.path.isdir(sf_root):
        raise FileNotFoundError(f"Shape files directory not found: {sf_root}")

    candidates = [
        d for d in os.listdir(sf_root)
        if d.lower().startswith(location.lower())
           and 'polygon' in d.lower()
           and f'at{resolution}'.lower() in d.lower()
           and os.path.isdir(os.path.join(sf_root, d))
    ]

    if not candidates:
        available = [d for d in os.listdir(sf_root) if os.path.isdir(os.path.join(sf_root, d))]
        raise FileNotFoundError(
            f"No shapefile folder for location='{location}' resolution='{resolution}'.\n"
            f"Searched in: {sf_root}\n"
            f"Available folders: {available}"
        )

    if len(candidates) > 1:
        print(f"Warning: Multiple shapefile folders found: {candidates}, using {candidates[0]}")

    fld = candidates[0]
    shp_path = os.path.join(sf_root, fld, fld + '.shp')

    if not os.path.isfile(shp_path):
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    return shp_path


def get_n_meters_from_shapefile(location):
    """
    Get the number of alongshore meters from the 1m shapefile.

    The number of features in the 1m shapefile equals the number of
    alongshore meters in the study area.
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError("geopandas is required for auto-detecting extent from shapefile")

    shp_path = find_shapefile(location, '1m')
    print(f"  Reading shapefile: {shp_path}")

    gdf = gpd.read_file(shp_path)
    n_meters = len(gdf)
    print(f"  Alongshore meters from shapefile: {n_meters}")

    return n_meters


def get_image_extent_from_grids(location):
    """
    Get the exact axis extent used by cliff_top_testing.py to create the image.

    Reads one 25cm erosion grid file and computes the extent from the
    polygon IDs (rows) and elevation bins (columns), matching exactly how
    cliff_top_testing.py creates its images.

    Returns:
        (x_min, x_max, y_min, y_max) in meters, or None if grids unavailable
    """
    base = get_base_path()
    resolution = '25cm'
    res_val = 0.25

    erosion_dir = os.path.join(base, 'results', location, 'erosion')
    if not os.path.isdir(erosion_dir):
        return None

    # Find first valid grid file (same search as cliff_top_testing.py)
    for date_folder in sorted(os.listdir(erosion_dir)):
        folder_path = os.path.join(erosion_dir, date_folder)
        if not os.path.isdir(folder_path):
            continue

        # Check resolution subdirectory, then the folder itself
        for search_dir in [os.path.join(folder_path, resolution), folder_path]:
            if not os.path.isdir(search_dir):
                continue
            for f in os.listdir(search_dir):
                if not (f.endswith('.csv') and 'grid' in f.lower()):
                    continue
                filepath = os.path.join(search_dir, f)
                try:
                    df = pd.read_csv(filepath, index_col=0)
                    df = df.apply(pd.to_numeric, errors='coerce')

                    # Polygon IDs are the row index (alongshore positions)
                    polygon_ids = df.index.astype(float).values

                    # Columns are elevation bin headers like "M3C2_0.25m"
                    # Strip letters/underscores to get the float values
                    cleaned_cols = df.columns.astype(str).str.replace(
                        r'[a-zA-Z_]', '', regex=True)
                    col_floats = cleaned_cols.astype(float)
                    # Scale to bin indices (same as clean_and_snap_grid)
                    scale = 1.0 / res_val
                    bin_indices = (col_floats * scale).round().astype(int)
                    n_bins = len(bin_indices)

                    # Compute extent exactly as testing script does
                    x_coords = polygon_ids * res_val
                    x_min = float(x_coords.min())
                    x_max = float(x_coords.max())
                    y_min = 0.0
                    y_max = float(n_bins * res_val)

                    return (x_min, x_max, y_min, y_max)
                except Exception as e:
                    print(f"  Warning: Could not read grid {f}: {e}")
                    continue

    return None


def detect_line_pixels(image_array, color_name):
    """
    Detect pixels matching the specified color.

    Args:
        image_array: RGB numpy array (H, W, 3)
        color_name: Name of color to detect (e.g., 'green', 'blue')

    Returns:
        Binary mask where True = line pixel
    """
    if color_name not in COLOR_RANGES:
        raise ValueError(f"Unknown color '{color_name}'. Available: {list(COLOR_RANGES.keys())}")

    color_range = COLOR_RANGES[color_name]
    min_rgb = np.array(color_range['min'])
    max_rgb = np.array(color_range['max'])

    mask = np.all((image_array >= min_rgb) & (image_array <= max_rgb), axis=2)
    return mask


def extract_line_coordinates(mask, method='top', plot_bounds=None):
    """
    Extract line coordinates from binary mask.

    For each pixel column within the plot area, finds the line pixel
    using the specified method.

    Args:
        mask: Binary mask (H, W) where True = line pixel
        method: 'top' (highest y), 'bottom' (lowest y), 'center' (centroid)
        plot_bounds: Optional (left, right, top, bottom) pixel bounds of plot area

    Returns:
        Arrays of (x_pixels, y_pixels) for the line
    """
    h, w = mask.shape
    x_pixels = []
    y_pixels = []

    if plot_bounds:
        col_start, col_end = plot_bounds[0], plot_bounds[1]
        row_start, row_end = plot_bounds[2], plot_bounds[3]
    else:
        col_start, col_end = 0, w
        row_start, row_end = 0, h

    for x in range(col_start, col_end):
        col = mask[row_start:row_end, x]
        indices = np.where(col)[0]

        if len(indices) == 0:
            continue

        if method == 'top':
            y = indices.min() + row_start
        elif method == 'bottom':
            y = indices.max() + row_start
        else:  # center
            y = indices.mean() + row_start

        x_pixels.append(x)
        y_pixels.append(y)

    return np.array(x_pixels), np.array(y_pixels)


def pixels_to_data_coords(x_pixels, y_pixels, x_min, x_max, y_min, y_max,
                          x_inverted=False, plot_bounds=None, image_shape=None):
    """
    Convert pixel coordinates to data coordinates.

    Args:
        x_pixels, y_pixels: Pixel coordinates
        x_min, x_max: Data extent in x (alongshore meters)
        y_min, y_max: Data extent in y (elevation meters)
        x_inverted: If True, x-axis runs right-to-left (decreasing)
        plot_bounds: (left, right, top, bottom) pixel bounds of plot area
        image_shape: (height, width) - used only if plot_bounds is None

    Returns:
        x_data, y_data: Arrays of data coordinates
    """
    if plot_bounds:
        px_left, px_right, px_top, px_bottom = plot_bounds
    else:
        h, w = image_shape
        px_left, px_right, px_top, px_bottom = 0, w, 0, h

    plot_width = px_right - px_left
    plot_height = px_bottom - px_top

    # Normalize pixel positions to [0, 1] within plot area
    x_norm = (x_pixels - px_left) / plot_width
    y_norm = (y_pixels - px_top) / plot_height

    # Convert to data coordinates
    if x_inverted:
        x_data = x_max - x_norm * (x_max - x_min)
    else:
        x_data = x_min + x_norm * (x_max - x_min)

    # Y is inverted in image coords (0 at top)
    y_data = y_max - y_norm * (y_max - y_min)

    return x_data, y_data


def smooth_line(x_data, y_data, window=5):
    """Apply simple moving average smoothing to the line."""
    if len(y_data) < window:
        return x_data, y_data

    kernel = np.ones(window) / window
    y_smooth = np.convolve(y_data, kernel, mode='valid')

    trim = (len(y_data) - len(y_smooth)) // 2
    x_smooth = x_data[trim:trim + len(y_smooth)]

    return x_smooth, y_smooth


def resample_line(x_data, y_data, resolution, n_meters):
    """
    Resample line to produce one elevation value per polygon ID.

    For each polygon ID at the given resolution, computes the corresponding
    alongshore meter position and interpolates the green line's elevation.

    Args:
        x_data: Alongshore positions in meters (from green line extraction)
        y_data: Elevation values in meters (from green line extraction)
        resolution: Target resolution string ('1m', '25cm', '10cm')
        n_meters: Total number of alongshore meters (from shapefile).

    Returns:
        DataFrame with Polygon_ID and CliffTop_Z columns
    """
    res_m = _resolution_to_meters(resolution)

    n_polygons = int(n_meters / res_m)
    all_ids = np.arange(n_polygons)
    meter_positions = all_ids * res_m

    # Sort extracted data by alongshore position for interpolation
    sort_idx = np.argsort(x_data)
    x_sorted = x_data[sort_idx]
    y_sorted = y_data[sort_idx]

    # np.interp clamps to boundary values outside the extracted range
    elevations = np.interp(meter_positions, x_sorted, y_sorted)

    return pd.DataFrame({'Polygon_ID': all_ids, 'CliffTop_Z': elevations})


def _group_indices(indices, min_gap=15):
    """
    Group consecutive indices, splitting at gaps > min_gap pixels.

    Returns list of lists, where each inner list is a group of adjacent indices.
    """
    if len(indices) == 0:
        return []
    groups = [[indices[0]]]
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] > min_gap:
            groups.append([])
        groups[-1].append(indices[i])
    return groups


def detect_plot_bounds(image_array, debug=False):
    """
    Detect the plot area bounds by finding the axis frame border lines.

    Groups border lines to distinguish the plot frame from the colorbar frame
    and axis labels. The plot spines are the first two vertical border groups
    (left and right), and the first/last horizontal border groups (top/bottom).

    Returns (left, right, top, bottom) pixel coordinates.
    """
    h, w = image_array.shape[:2]

    # Near-black mask: all RGB channels below threshold
    near_black = np.all(image_array < 60, axis=2)

    row_counts = np.sum(near_black, axis=1)
    col_counts = np.sum(near_black, axis=0)

    # --- Horizontal borders (top/bottom spines) ---
    # Axis spines span the full plot width, so they have many near-black pixels.
    # Text rows (title, labels) have far fewer.
    h_threshold = w * 0.3
    border_rows = np.where(row_counts > h_threshold)[0]
    row_groups = _group_indices(border_rows)

    if len(row_groups) >= 2:
        # First group = top spine, last group = bottom spine
        top_bound = row_groups[0][0]
        bottom_bound = row_groups[-1][-1]
    elif len(row_groups) == 1:
        top_bound = row_groups[0][0]
        bottom_bound = row_groups[0][-1]
    else:
        if debug:
            print("Warning: Could not detect horizontal borders, using fallback")
        top_bound = int(h * 0.10)
        bottom_bound = int(h * 0.85)

    # --- Vertical borders (left/right spines, excluding colorbar) ---
    # The colorbar also has vertical border lines. By grouping border columns,
    # the first two groups correspond to the plot's left and right spines.
    # Additional groups are colorbar borders and are ignored.
    v_threshold = h * 0.2
    border_cols = np.where(col_counts > v_threshold)[0]
    col_groups = _group_indices(border_cols, min_gap=max(15, int(w * 0.02)))

    if len(col_groups) >= 2:
        # First group = left spine, second group = right spine
        left_bound = col_groups[0][0]
        right_bound = col_groups[1][-1]
        if debug and len(col_groups) > 2:
            print(f"  Found {len(col_groups)} vertical border groups "
                  f"(using first 2, ignoring {len(col_groups) - 2} colorbar groups)")
    elif len(col_groups) == 1:
        left_bound = col_groups[0][0]
        right_bound = col_groups[0][-1]
    else:
        if debug:
            print("Warning: Could not detect vertical borders, using fallback")
        left_bound = int(w * 0.08)
        right_bound = int(w * 0.92)

    # Validate bounds make sense
    if right_bound - left_bound < w * 0.3 or bottom_bound - top_bound < h * 0.3:
        if debug:
            print("Warning: Detected bounds seem too small, using fallback")
        left_bound = int(w * 0.08)
        right_bound = int(w * 0.92)
        top_bound = int(h * 0.10)
        bottom_bound = int(h * 0.85)

    if debug:
        print(f"Detected plot bounds: left={left_bound}, right={right_bound}, "
              f"top={top_bound}, bottom={bottom_bound}")
        print(f"  Plot size: {right_bound - left_bound} x {bottom_bound - top_bound} pixels")
        if row_groups:
            print(f"  Horizontal border groups: {len(row_groups)} "
                  f"(rows: {[g[0] for g in row_groups]})")
        if col_groups:
            print(f"  Vertical border groups: {len(col_groups)} "
                  f"(cols: {[g[0] for g in col_groups]})")

    return (left_bound, right_bound, top_bound, bottom_bound)


def plot_verification(location, output_dir, alongshore_m, elevation_m, n_meters):
    """Generate verification plot showing extracted line at all resolutions."""
    fig, axes = plt.subplots(len(RESOLUTIONS), 1, figsize=(14, 10))

    for i, resolution in enumerate(RESOLUTIONS):
        ax = axes[i]
        df = resample_line(alongshore_m, elevation_m, resolution, n_meters=n_meters)

        res_m = _resolution_to_meters(resolution)
        x_meters = df['Polygon_ID'].values * res_m

        ax.plot(x_meters, df['CliffTop_Z'], 'b-', linewidth=1.5)
        ax.scatter(x_meters, df['CliffTop_Z'], s=2, c='blue', alpha=0.5)

        ax.set_title(f"{resolution} Resolution ({len(df)} points)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Alongshore Position (m)")
        ax.set_ylabel("Elevation (m)")
        ax.set_ylim(0, max(elevation_m) * 1.1)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{location}: Digitized Cliff Top Line", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{location}_Visual_CliffTop_verification.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Digitize manually drawn cliff-top line from annotated image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect everything from grid data and shapefile:
    python3 digitize_cliff_top.py --location DelMar --line_color green --x_inverted

    # Specify image axis ranges manually (read from tick labels):
    python3 digitize_cliff_top.py --location DelMar \\
        --x_min 0 --x_max 2800 --y_max 48 --line_color green --x_inverted

    # Specify plot area bounds manually (pixel coordinates):
    python3 digitize_cliff_top.py --location DelMar --line_color green --x_inverted \\
        --plot_left 100 --plot_right 1800 --plot_top 50 --plot_bottom 400
        """
    )

    # Required arguments
    parser.add_argument('--location', required=True, help='Location name (e.g., DelMar, SanElijo)')

    # Image path (optional - auto-detected from location)
    parser.add_argument('--image', default=None,
                        help='Path to annotated PNG image. Auto-detected if not provided.')

    # Image axis extent (optional - auto-detected from grid data)
    # These define what data coordinates the image axes represent.
    parser.add_argument('--x_min', type=float, default=None,
                        help='Image x-axis minimum (alongshore meters). Read from image tick labels.')
    parser.add_argument('--x_max', type=float, default=None,
                        help='Image x-axis maximum (alongshore meters). Read from image tick labels.')
    parser.add_argument('--y_min', type=float, default=0.0,
                        help='Image y-axis minimum (elevation meters). Default: 0.')
    parser.add_argument('--y_max', type=float, default=None,
                        help='Image y-axis maximum (elevation meters). Read from image tick labels.')

    # Output polygon ID count (optional - auto-detected from shapefile)
    parser.add_argument('--n_meters', type=int, default=None,
                        help='Number of alongshore meters for output. Auto-detected from 1m shapefile.')

    # Line detection options
    parser.add_argument('--line_color', default='green',
                        choices=list(COLOR_RANGES.keys()),
                        help='Color of drawn line (default: green)')
    parser.add_argument('--x_inverted', action='store_true',
                        help='X-axis is inverted (decreasing left to right)')

    # Plot bounds (optional - for when auto-detection fails)
    parser.add_argument('--plot_left', type=int, help='Left pixel bound of plot area')
    parser.add_argument('--plot_right', type=int, help='Right pixel bound of plot area')
    parser.add_argument('--plot_top', type=int, help='Top pixel bound of plot area')
    parser.add_argument('--plot_bottom', type=int, help='Bottom pixel bound of plot area')
    parser.add_argument('--no_auto_bounds', action='store_true',
                        help='Disable auto-detection of plot bounds (use full image)')

    # Processing options
    parser.add_argument('--smooth', type=int, default=0,
                        help='Smoothing window size (0 = no smoothing)')
    parser.add_argument('--line_method', default='top', choices=['top', 'bottom', 'center'],
                        help='Method to extract line from thick strokes (default: top)')

    # Output options
    parser.add_argument('--output_dir', help='Output directory (default: utilities/cliff_top_cutoffs/<location>)')
    parser.add_argument('--dry_run', action='store_true', help='Show results without saving')
    parser.add_argument('--debug', action='store_true', help='Show debug visualizations')

    args = parser.parse_args()

    # --- 1. Find image ---
    if args.image is None:
        print(f"\nAuto-detecting image for location '{args.location}'...")
        args.image = find_image_for_location(args.location)
        print(f"  Found: {args.image}")

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    # --- 2. Determine image axis extents ---
    # Priority: user CLI args > grid data auto-detect > shapefile/HEIGHTS fallback
    x_min = args.x_min
    x_max = args.x_max
    y_min = args.y_min
    y_max = args.y_max
    n_meters = args.n_meters

    # Try auto-detecting from grid data (matches cliff_top_testing.py exactly)
    need_auto = (x_min is None or x_max is None or y_max is None)
    grid_extent = None
    if need_auto:
        print(f"\nAuto-detecting image axis extent from grid data...")
        try:
            grid_extent = get_image_extent_from_grids(args.location)
        except Exception as e:
            print(f"  Warning: Could not read grid data: {e}")

    if grid_extent is not None:
        gx_min, gx_max, gy_min, gy_max = grid_extent
        if x_min is None:
            x_min = gx_min
        if x_max is None:
            x_max = gx_max
        if y_max is None:
            y_max = gy_max
        print(f"  From grid data: X=[{gx_min:.1f}, {gx_max:.1f}], Y=[{gy_min:.1f}, {gy_max:.1f}]")
    else:
        # Fallback: shapefile for x, HEIGHTS for y
        if x_min is None or x_max is None:
            print(f"\nFalling back to shapefile for x extent...")
            try:
                sf_n_meters = get_n_meters_from_shapefile(args.location)
                if x_min is None:
                    x_min = 0.0
                if x_max is None:
                    x_max = float(sf_n_meters)
            except (FileNotFoundError, ImportError) as e:
                raise ValueError(
                    f"Could not determine x extent: {e}\n"
                    "Please provide --x_min and --x_max manually."
                )

        if y_max is None:
            if args.location in HEIGHTS:
                # Compute y_max the same way cliff_top_testing.py does:
                # n_bins * res_val, where n_bins = height / res + 1
                height = HEIGHTS[args.location]
                res_val = 0.25  # images are generated at 25cm
                n_bins = len(np.arange(0, height + res_val, res_val)) - 1
                y_max = n_bins * res_val
                print(f"  y_max = {y_max:.2f} m (from HEIGHTS[{args.location}]={height}, "
                      f"{n_bins} bins at 25cm)")
            else:
                raise ValueError(
                    f"Unknown location '{args.location}'. "
                    "Please provide --y_max manually."
                )

    if y_min is None:
        y_min = 0.0

    # Get n_meters for output polygon IDs (from shapefile)
    if n_meters is None:
        try:
            n_meters = get_n_meters_from_shapefile(args.location)
        except (FileNotFoundError, ImportError) as e:
            # Fall back to x_max if shapefile unavailable
            n_meters = int(x_max)
            print(f"  Using x_max={x_max:.0f} as n_meters (shapefile unavailable: {e})")

    print(f"\nImage axis extent: X=[{x_min:.1f}, {x_max:.1f}] m, Y=[{y_min:.1f}, {y_max:.2f}] m")
    print(f"Output polygon IDs: {n_meters} alongshore meters")

    # --- 3. Load image ---
    print(f"\nLoading image: {args.image}")
    img = Image.open(args.image).convert('RGB')
    image_array = np.array(img)
    h, w = image_array.shape[:2]
    print(f"Image size: {w} x {h} pixels")

    # --- 4. Detect line pixels ---
    print(f"Detecting {args.line_color} line pixels...")
    mask = detect_line_pixels(image_array, args.line_color)
    n_pixels = np.sum(mask)
    print(f"Found {n_pixels} line pixels")

    if n_pixels == 0:
        raise ValueError(f"No {args.line_color} pixels found. Try a different --line_color")

    # --- 5. Determine plot bounds ---
    plot_bounds = None
    if args.plot_left is not None:
        plot_bounds = (args.plot_left, args.plot_right, args.plot_top, args.plot_bottom)
        print(f"Using manual plot bounds: {plot_bounds}")
    elif args.no_auto_bounds:
        print("Auto-bounds disabled, using full image")
    else:
        plot_bounds = detect_plot_bounds(image_array, debug=args.debug)
        if plot_bounds:
            left, right, top, bottom = plot_bounds
            print(f"Auto-detected plot bounds: left={left}, right={right}, "
                  f"top={top}, bottom={bottom}")
            print(f"  Plot area: {right - left} x {bottom - top} pixels")
        else:
            print("Could not auto-detect bounds, using full image")

    # --- 6. Extract line coordinates ---
    print(f"Extracting line using '{args.line_method}' method...")
    x_pixels, y_pixels = extract_line_coordinates(mask, method=args.line_method,
                                                   plot_bounds=plot_bounds)
    print(f"Extracted {len(x_pixels)} points")

    # --- 7. Convert pixels to data coordinates ---
    x_data, y_data = pixels_to_data_coords(
        x_pixels, y_pixels,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        x_inverted=args.x_inverted,
        plot_bounds=plot_bounds,
        image_shape=(h, w)
    )

    print(f"Data range: X=[{x_data.min():.1f}, {x_data.max():.1f}] m, "
          f"Y=[{y_data.min():.1f}, {y_data.max():.1f}] m")

    # Show expected polygon IDs at each resolution
    print(f"\nExpected output (from {n_meters} alongshore meters):")
    for res in RESOLUTIONS:
        res_m = _resolution_to_meters(res)
        n_polygons = int(n_meters / res_m)
        print(f"  {res}: polygon IDs 0 to {n_polygons - 1} ({n_polygons} polygons)")

    # --- 8. Optional smoothing ---
    if args.smooth > 0:
        print(f"Applying smoothing (window={args.smooth})...")
        x_data, y_data = smooth_line(x_data, y_data, window=args.smooth)

    # --- 9. Debug visualization ---
    if args.debug:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image_array)
        if plot_bounds:
            left, right, top, bottom = plot_bounds
            from matplotlib.patches import Rectangle
            rect = Rectangle((left, top), right - left, bottom - top,
                              linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
        axes[0].set_title("Original Image + Detected Bounds")

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f"Detected {args.line_color} pixels")

        axes[2].plot(x_data, y_data, 'b-', linewidth=1)
        axes[2].set_xlabel("Alongshore (m)")
        axes[2].set_ylabel("Elevation (m)")
        axes[2].set_title("Extracted Line (data coords)")
        if args.x_inverted:
            axes[2].invert_xaxis()

        plt.tight_layout()
        plt.show()

    # --- 10. Save outputs ---
    output_dir = args.output_dir or get_output_dir(args.location)

    if args.dry_run:
        print("\n[DRY RUN] Would save to:")
        for resolution in RESOLUTIONS:
            df = resample_line(x_data, y_data, resolution, n_meters=n_meters)
            filename = f"{args.location}_Visual_CliffTop_{resolution}.csv"
            print(f"  {filename}: {len(df)} points, "
                  f"Z range [{df['CliffTop_Z'].min():.1f}, {df['CliffTop_Z'].max():.1f}] m")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    for resolution in RESOLUTIONS:
        df = resample_line(x_data, y_data, resolution, n_meters=n_meters)
        filename = f"{args.location}_Visual_CliffTop_{resolution}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {filename}: {len(df)} points, "
              f"Z range [{df['CliffTop_Z'].min():.1f}, {df['CliffTop_Z'].max():.1f}] m")

    # Generate verification plot
    verify_path = plot_verification(args.location, output_dir, x_data, y_data,
                                    n_meters=n_meters)
    print(f"Saved verification plot: {verify_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
