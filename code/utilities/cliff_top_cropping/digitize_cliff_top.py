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
    # Auto-detect x/y extents from location shapefile and elevation cutoffs:
    python3 digitize_cliff_top.py --image annotated.png --location DelMar \
        --line_color green --x_inverted

    # Override with manual extents if needed:
    python3 digitize_cliff_top.py --image annotated.png --location DelMar \
        --x_min 0 --x_max 1400 --y_min 0 --y_max 70 \
        --line_color green --x_inverted

    # Specify plot area bounds manually (pixels):
    python3 digitize_cliff_top.py --image annotated.png --location DelMar \
        --line_color green --x_inverted \
        --plot_left 100 --plot_right 1800 --plot_top 50 --plot_bottom 400
"""

import os
import argparse
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation
from collections import defaultdict

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


def get_base_path():
    """Return base path based on OS."""
    if platform.system() == "Darwin":
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"


def get_output_dir(location):
    """Return output directory for cliff top cutoff CSVs."""
    base = get_base_path()
    # Output to location-specific subdirectory
    return os.path.join(base, "utilities", "cliff_top_cutoffs", location)


def find_image_for_location(location):
    """
    Auto-detect the cliff top image for a given location.

    Expected naming pattern: {location}_clifftop.png (lowercase)
    Location: code/utilities/cliff_top_cropping/figures/manual_lines/

    Args:
        location: Location name (e.g., 'Torrey', 'DelMar')

    Returns:
        Path to the image file
    """
    # Get the script directory to find the figures folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "figures", "manual_lines")

    # Try lowercase location name
    image_name = f"{location.lower()}_clifftop.png"
    image_path = os.path.join(figures_dir, image_name)

    if os.path.exists(image_path):
        return image_path

    # Try other common patterns
    patterns = [
        f"{location}_clifftop.png",
        f"{location.lower()}_cliftop.png",  # Common typo
        f"{location}_CliffTop.png",
    ]

    for pattern in patterns:
        candidate = os.path.join(figures_dir, pattern)
        if os.path.exists(candidate):
            return candidate

    # List available images
    available = []
    if os.path.isdir(figures_dir):
        available = [f for f in os.listdir(figures_dir) if f.endswith('.png')]

    raise FileNotFoundError(
        f"Could not find cliff top image for '{location}'.\n"
        f"Expected: {image_path}\n"
        f"Available images in {figures_dir}:\n  " + "\n  ".join(available) if available else "(none)"
    )


def find_shapefile(location, resolution='1m'):
    """
    Locate the shapefile for a given location and resolution.

    Expected naming pattern: {Location}Polygon[e]s{MOP1}to{MOP2}at{resolution}
    Example: SanElijoPolygones683to708at25cm

    Args:
        location: Location name (e.g., 'SanElijo')
        resolution: Resolution string ('1m', '25cm', '10cm')

    Returns:
        Path to the .shp file
    """
    base = get_base_path()
    sf_root = os.path.join(base, 'utilities', 'shape_files')

    if not os.path.isdir(sf_root):
        raise FileNotFoundError(f"Shape files directory not found: {sf_root}")

    # Pattern: location name + "Polygon" (may have 'e' or 's') + MOP range + "at" + resolution
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
            f"No shapefile folder found for location='{location}' resolution='{resolution}'.\n"
            f"Searched in: {sf_root}\n"
            f"Available folders: {available}"
        )

    if len(candidates) > 1:
        print(f"Warning: Multiple shapefile folders found: {candidates}")
        print(f"Using first match: {candidates[0]}")

    fld = candidates[0]
    shp_path = os.path.join(sf_root, fld, fld + '.shp')

    if not os.path.isfile(shp_path):
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    return shp_path


def get_extent_from_shapefile(location, resolution='1m'):
    """
    Get the alongshore extent (x_min, x_max) from the location's shapefile.

    Reads the shapefile and converts polygon IDs back to alongshore meters.

    Args:
        location: Location name
        resolution: Resolution to use for shapefile lookup

    Returns:
        (x_min, x_max) in alongshore meters
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError("geopandas is required for auto-detecting extent from shapefile")

    shp_path = find_shapefile(location, resolution)
    print(f"Reading shapefile: {shp_path}")

    gdf = gpd.read_file(shp_path)

    # Get polygon IDs - they're the row indices
    polygon_ids = gdf.index.values

    # Convert resolution string to meters
    if resolution == '1m' or resolution == '100cm':
        res_m = 1.0
    elif resolution == '25cm':
        res_m = 0.25
    elif resolution == '10cm':
        res_m = 0.10
    else:
        raise ValueError(f"Unknown resolution: {resolution}")

    # Convert polygon IDs to alongshore meters
    # polygon_id = alongshore_m / resolution_m
    # so: alongshore_m = polygon_id * resolution_m
    alongshore_min = polygon_ids.min() * res_m
    alongshore_max = (polygon_ids.max() + 1) * res_m  # +1 to include the full extent

    return alongshore_min, alongshore_max


def get_elevation_extent(location):
    """
    Get the elevation extent (y_min, y_max) for a location.

    Args:
        location: Location name

    Returns:
        (y_min, y_max) in meters
    """
    if location not in HEIGHTS:
        available = list(HEIGHTS.keys())
        raise ValueError(f"Unknown location '{location}'. Available: {available}")

    return 0.0, float(HEIGHTS[location])


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

    # Create mask for pixels within color range
    mask = np.all((image_array >= min_rgb) & (image_array <= max_rgb), axis=2)

    return mask


def extract_line_coordinates(mask, method='top'):
    """
    Extract line coordinates from binary mask.

    For each column (x position), finds the line pixel using the specified method.

    Args:
        mask: Binary mask (H, W) where True = line pixel
        method: 'top' (highest y), 'bottom' (lowest y), 'center' (centroid)

    Returns:
        Arrays of (x_pixels, y_pixels) for the line
    """
    h, w = mask.shape
    x_pixels = []
    y_pixels = []

    for x in range(w):
        col = mask[:, x]
        indices = np.where(col)[0]

        if len(indices) == 0:
            continue

        if method == 'top':
            # In image coordinates, top = smaller y value
            y = indices.min()
        elif method == 'bottom':
            y = indices.max()
        else:  # center
            y = indices.mean()

        x_pixels.append(x)
        y_pixels.append(y)

    return np.array(x_pixels), np.array(y_pixels)


def pixels_to_data_coords(x_pixels, y_pixels, image_shape,
                          x_min, x_max, y_min, y_max,
                          x_inverted=False, plot_bounds=None):
    """
    Convert pixel coordinates to data coordinates.

    Args:
        x_pixels, y_pixels: Pixel coordinates
        image_shape: (height, width) of image
        x_min, x_max: Data extent in x (alongshore meters)
        y_min, y_max: Data extent in y (elevation meters)
        x_inverted: If True, x-axis runs right-to-left (decreasing)
        plot_bounds: Optional (left, right, top, bottom) pixel bounds of plot area
                     If None, assumes entire image is the plot

    Returns:
        x_data, y_data: Arrays of data coordinates
    """
    h, w = image_shape

    # Use plot bounds if provided, otherwise use full image
    if plot_bounds:
        px_left, px_right, px_top, px_bottom = plot_bounds
    else:
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


def alongshore_to_polygon_id(alongshore_m, resolution):
    """
    Convert alongshore meters to polygon ID for a given resolution.

    The polygon ID is essentially the bin index at that resolution.

    Args:
        alongshore_m: Alongshore distance in meters
        resolution: Resolution string ('1m', '25cm', '10cm')

    Returns:
        Polygon ID (integer index)
    """
    if resolution == '1m' or resolution == '100cm':
        res_m = 1.0
    elif resolution == '25cm':
        res_m = 0.25
    elif resolution == '10cm':
        res_m = 0.10
    else:
        raise ValueError(f"Unknown resolution: {resolution}")

    return (alongshore_m / res_m).astype(int)


def smooth_line(x_data, y_data, window=5):
    """Apply simple moving average smoothing to the line."""
    if len(y_data) < window:
        return x_data, y_data

    # Use convolution for smoothing
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y_data, kernel, mode='valid')

    # Trim x to match
    trim = (len(y_data) - len(y_smooth)) // 2
    x_smooth = x_data[trim:trim + len(y_smooth)]

    return x_smooth, y_smooth


def resample_line(x_data, y_data, resolution):
    """
    Resample line to have one point per polygon ID at the given resolution.

    Interpolates to fill every polygon ID in the range, ensuring complete
    alongshore coverage even when the source line has pixel-level gaps.

    Args:
        x_data: Alongshore meters
        y_data: Elevation meters
        resolution: Target resolution

    Returns:
        DataFrame with Polygon_ID and CliffTop_Z columns
    """
    polygon_ids = alongshore_to_polygon_id(x_data, resolution)

    # Group by polygon ID and take mean elevation
    df = pd.DataFrame({'Polygon_ID': polygon_ids, 'CliffTop_Z': y_data})
    df = df.groupby('Polygon_ID').agg({'CliffTop_Z': 'mean'}).reset_index()
    df = df.sort_values('Polygon_ID')

    # Interpolate to fill every polygon ID across the full range
    all_ids = np.arange(df['Polygon_ID'].min(), df['Polygon_ID'].max() + 1)
    interp_z = np.interp(all_ids, df['Polygon_ID'].values, df['CliffTop_Z'].values)
    df = pd.DataFrame({'Polygon_ID': all_ids, 'CliffTop_Z': interp_z})

    return df


def detect_plot_bounds(image_array, debug=False):
    """
    Detect the plot area bounds by finding where the colored line exists
    and the transition from white plot background to gray margins.

    Returns (left, right, top, bottom) pixel coordinates or None.
    """
    h, w = image_array.shape[:2]

    # Convert to grayscale
    gray = np.mean(image_array, axis=2)

    # Method 1: Find the bounding box of non-white pixels (the plot area)
    # Plot backgrounds are typically white (>250), margins have axis labels/ticks
    white_threshold = 245

    # Find rows and columns that are mostly white (plot area)
    # vs rows/columns with significant gray content (axis labels)

    # Scan from left to find where plot starts (first mostly-white column)
    left_bound = 0
    for x in range(w // 3):
        col = gray[:, x]
        white_fraction = np.mean(col > white_threshold)
        if white_fraction > 0.7:  # Mostly white = plot area
            left_bound = x
            break

    # Scan from right to find where plot ends
    right_bound = w - 1
    for x in range(w - 1, 2 * w // 3, -1):
        col = gray[:, x]
        white_fraction = np.mean(col > white_threshold)
        if white_fraction > 0.7:
            right_bound = x
            break

    # Scan from top to find where plot starts
    top_bound = 0
    for y in range(h // 3):
        row = gray[y, :]
        white_fraction = np.mean(row > white_threshold)
        if white_fraction > 0.7:
            top_bound = y
            break

    # Scan from bottom to find where plot ends
    bottom_bound = h - 1
    for y in range(h - 1, 2 * h // 3, -1):
        row = gray[y, :]
        white_fraction = np.mean(row > white_threshold)
        if white_fraction > 0.7:
            bottom_bound = y
            break

    # Validate bounds make sense
    if right_bound - left_bound < w * 0.5 or bottom_bound - top_bound < h * 0.5:
        if debug:
            print(f"Warning: Detected bounds seem too small, using fallback")
        # Fallback: assume standard matplotlib margins
        left_bound = int(w * 0.08)
        right_bound = int(w * 0.92)
        top_bound = int(h * 0.08)
        bottom_bound = int(h * 0.88)

    if debug:
        print(f"Detected plot bounds: left={left_bound}, right={right_bound}, "
              f"top={top_bound}, bottom={bottom_bound}")
        print(f"  Plot size: {right_bound - left_bound} x {bottom_bound - top_bound} pixels")

    return (left_bound, right_bound, top_bound, bottom_bound)


def plot_verification(location, output_dir, alongshore_m, elevation_m):
    """
    Generate verification plot showing extracted line at all resolutions.
    Uses alongshore meters on x-axis (same as testing visualization).
    """
    fig, axes = plt.subplots(len(RESOLUTIONS), 1, figsize=(14, 10))

    for i, resolution in enumerate(RESOLUTIONS):
        ax = axes[i]
        df = resample_line(alongshore_m, elevation_m, resolution)

        # Convert polygon IDs back to meters for plotting
        if resolution == '1m':
            res_m = 1.0
        elif resolution == '25cm':
            res_m = 0.25
        else:
            res_m = 0.10
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
    # Auto-detect image and extents from location:
    python3 digitize_cliff_top.py --location DelMar --line_color green --x_inverted

    # Override with manual image path:
    python3 digitize_cliff_top.py --image annotated.png --location DelMar \\
        --line_color green --x_inverted

    # Override with manual extents if needed:
    python3 digitize_cliff_top.py --location DelMar \\
        --x_min 0 --x_max 1400 --y_min 0 --y_max 70 --line_color green --x_inverted
        """
    )

    # Required arguments
    parser.add_argument('--location', required=True, help='Location name (e.g., DelMar, SanElijo)')

    # Image path (optional - auto-detected from location if not provided)
    parser.add_argument('--image', default=None,
                        help='Path to annotated PNG image. Auto-detected from location if not provided.')

    # Axis extent arguments (optional - auto-detected from shapefile/HEIGHTS if not provided)
    parser.add_argument('--x_min', type=float, default=None,
                        help='Minimum x value (alongshore meters). Auto-detected from shapefile if not provided.')
    parser.add_argument('--x_max', type=float, default=None,
                        help='Maximum x value (alongshore meters). Auto-detected from shapefile if not provided.')
    parser.add_argument('--y_min', type=float, default=None,
                        help='Minimum y value (elevation meters). Defaults to 0.')
    parser.add_argument('--y_max', type=float, default=None,
                        help='Maximum y value (elevation meters). Auto-detected from HEIGHTS if not provided.')
    parser.add_argument('--shp_resolution', default='1m', choices=['1m', '25cm', '10cm'],
                        help='Resolution of shapefile to use for auto-detecting x extent (default: 1m)')

    # Line detection options
    parser.add_argument('--line_color', default='green',
                        choices=list(COLOR_RANGES.keys()),
                        help='Color of drawn line (default: green)')
    parser.add_argument('--x_inverted', action='store_true',
                        help='X-axis is inverted (decreasing left to right)')

    # Plot bounds (optional - for when axis labels/margins need to be excluded)
    parser.add_argument('--plot_left', type=int, help='Left pixel bound of plot area')
    parser.add_argument('--plot_right', type=int, help='Right pixel bound of plot area')
    parser.add_argument('--plot_top', type=int, help='Top pixel bound of plot area')
    parser.add_argument('--plot_bottom', type=int, help='Bottom pixel bound of plot area')
    parser.add_argument('--auto_bounds', action='store_true', default=True,
                        help='Auto-detect plot bounds from image (default: True)')
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

    # Auto-detect image path if not provided
    if args.image is None:
        print(f"\nAuto-detecting image for location '{args.location}'...")
        args.image = find_image_for_location(args.location)
        print(f"  Found: {args.image}")

    # Validate image exists
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    # Auto-detect x extent from shapefile if not provided
    if args.x_min is None or args.x_max is None:
        print(f"\nAuto-detecting x extent from shapefile (resolution={args.shp_resolution})...")
        try:
            auto_x_min, auto_x_max = get_extent_from_shapefile(args.location, args.shp_resolution)
            if args.x_min is None:
                args.x_min = auto_x_min
                print(f"  x_min = {args.x_min:.1f} m (from shapefile)")
            if args.x_max is None:
                args.x_max = auto_x_max
                print(f"  x_max = {args.x_max:.1f} m (from shapefile)")
        except (FileNotFoundError, ImportError) as e:
            raise ValueError(
                f"Could not auto-detect x extent: {e}\n"
                "Please provide --x_min and --x_max manually."
            )

    # Auto-detect y extent from HEIGHTS if not provided
    if args.y_min is None:
        args.y_min = 0.0
        print(f"  y_min = {args.y_min:.1f} m (default)")
    if args.y_max is None:
        try:
            _, auto_y_max = get_elevation_extent(args.location)
            args.y_max = auto_y_max
            print(f"  y_max = {args.y_max:.1f} m (from HEIGHTS[{args.location}])")
        except ValueError as e:
            raise ValueError(
                f"Could not auto-detect y extent: {e}\n"
                "Please provide --y_max manually."
            )

    print(f"\nUsing extents: X=[{args.x_min:.1f}, {args.x_max:.1f}] m, Y=[{args.y_min:.1f}, {args.y_max:.1f}] m")

    # Load image
    print(f"\nLoading image: {args.image}")
    img = Image.open(args.image).convert('RGB')
    image_array = np.array(img)
    h, w = image_array.shape[:2]
    print(f"Image size: {w} x {h} pixels")

    # Detect line pixels
    print(f"Detecting {args.line_color} line pixels...")
    mask = detect_line_pixels(image_array, args.line_color)
    n_pixels = np.sum(mask)
    print(f"Found {n_pixels} line pixels")

    if n_pixels == 0:
        raise ValueError(f"No {args.line_color} pixels found. Try a different --line_color")

    # Determine plot bounds
    plot_bounds = None
    if args.plot_left is not None:
        plot_bounds = (args.plot_left, args.plot_right, args.plot_top, args.plot_bottom)
        print(f"Using manual plot bounds: {plot_bounds}")
    elif args.no_auto_bounds:
        print("Auto-bounds disabled, using full image")
    else:
        # Auto-detect by default
        plot_bounds = detect_plot_bounds(image_array, debug=args.debug)
        if plot_bounds:
            left, right, top, bottom = plot_bounds
            print(f"Auto-detected plot bounds: left={left}, right={right}, top={top}, bottom={bottom}")
            print(f"  Plot area: {right - left} x {bottom - top} pixels")
        else:
            print("Could not auto-detect bounds, using full image")

    # Extract line coordinates
    print(f"Extracting line using '{args.line_method}' method...")
    x_pixels, y_pixels = extract_line_coordinates(mask, method=args.line_method)
    print(f"Extracted {len(x_pixels)} points")

    # Convert to data coordinates
    x_data, y_data = pixels_to_data_coords(
        x_pixels, y_pixels,
        image_shape=(h, w),
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        x_inverted=args.x_inverted,
        plot_bounds=plot_bounds
    )

    print(f"Data range: X=[{x_data.min():.1f}, {x_data.max():.1f}] m, "
          f"Y=[{y_data.min():.1f}, {y_data.max():.1f}] m")

    # Show what polygon IDs this will produce at each resolution
    print("\nExpected polygon ID ranges:")
    for res in RESOLUTIONS:
        if res == '1m':
            res_m = 1.0
        elif res == '25cm':
            res_m = 0.25
        else:
            res_m = 0.10
        pid_min = int(x_data.min() / res_m)
        pid_max = int(x_data.max() / res_m)
        print(f"  {res}: polygon IDs {pid_min} to {pid_max}")

    # Optional smoothing
    if args.smooth > 0:
        print(f"Applying smoothing (window={args.smooth})...")
        x_data, y_data = smooth_line(x_data, y_data, window=args.smooth)

    # Debug visualization
    if args.debug:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image_array)
        axes[0].set_title("Original Image")

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

    # Set output directory
    output_dir = args.output_dir or get_output_dir(args.location)

    if args.dry_run:
        print("\n[DRY RUN] Would save to:")
        for resolution in RESOLUTIONS:
            df = resample_line(x_data, y_data, resolution)
            filename = f"{args.location}_Visual_CliffTop_{resolution}.csv"
            print(f"  {filename}: {len(df)} points")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Save CSVs for each resolution
    # Naming convention matches what step 8 expects: {location}_Visual_CliffTop_{resolution}.csv
    for resolution in RESOLUTIONS:
        df = resample_line(x_data, y_data, resolution)
        filename = f"{args.location}_Visual_CliffTop_{resolution}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {filename}: {len(df)} points")

    # Generate verification plot
    verify_path = plot_verification(args.location, output_dir, x_data, y_data)
    print(f"Saved verification plot: {verify_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
