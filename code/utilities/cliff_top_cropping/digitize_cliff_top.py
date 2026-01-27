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
    python3 digitize_cliff_top.py --image path/to/annotated.png \
        --location DelMar \
        --x_min 0 --x_max 1400 \
        --y_min 0 --y_max 70 \
        --line_color green \
        --x_inverted

    # Or use --detect_extent to attempt automatic axis detection (experimental)
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


# --- CONFIGURATION ---
RESOLUTIONS = ['1m', '25cm', '10cm']

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
    return os.path.join(base, "utilities", "cliff_top_cutoffs", location)


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

    return df


def detect_plot_bounds(image_array, debug=False):
    """
    Attempt to detect the plot area bounds by finding axis lines.

    This is experimental and may not work for all images.
    Returns (left, right, top, bottom) pixel coordinates or None.
    """
    # Convert to grayscale
    gray = np.mean(image_array, axis=2)

    h, w = gray.shape

    # Look for vertical black lines (y-axis) - scan from left
    left_bound = None
    for x in range(w // 4):  # Search left quarter
        col = gray[:, x]
        dark_pixels = np.sum(col < 50)
        if dark_pixels > h * 0.5:  # More than half the column is dark
            left_bound = x
            break

    # Look for horizontal black lines (x-axis) - scan from bottom
    bottom_bound = None
    for y in range(h - 1, h * 3 // 4, -1):  # Search bottom quarter
        row = gray[y, :]
        dark_pixels = np.sum(row < 50)
        if dark_pixels > w * 0.5:
            bottom_bound = y
            break

    if left_bound is None or bottom_bound is None:
        return None

    # Estimate right and top bounds (assume some margin)
    right_bound = w - 50  # Assume 50px margin on right
    top_bound = 50  # Assume 50px margin on top

    if debug:
        print(f"Detected plot bounds: left={left_bound}, right={right_bound}, "
              f"top={top_bound}, bottom={bottom_bound}")

    return (left_bound, right_bound, top_bound, bottom_bound)


def plot_verification(location, output_dir, alongshore_m, elevation_m):
    """
    Generate verification plot showing extracted line at all resolutions.
    """
    fig, axes = plt.subplots(len(RESOLUTIONS), 1, figsize=(14, 10))

    for i, resolution in enumerate(RESOLUTIONS):
        ax = axes[i]
        df = resample_line(alongshore_m, elevation_m, resolution)

        ax.plot(df['Polygon_ID'], df['CliffTop_Z'], 'b-', linewidth=1.5)
        ax.scatter(df['Polygon_ID'], df['CliffTop_Z'], s=2, c='blue', alpha=0.5)

        ax.set_title(f"{resolution} Resolution ({len(df)} points)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Polygon ID")
        ax.set_ylabel("Elevation (m)")
        ax.set_ylim(0, max(elevation_m) * 1.1)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{location}: Digitized Cliff Top Line", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{location}_digitized_verification.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Digitize manually drawn cliff-top line from annotated image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with known axis extents:
    python3 digitize_cliff_top.py --image annotated.png --location DelMar \\
        --x_min 0 --x_max 1400 --y_min 0 --y_max 70 --line_color green --x_inverted

    # Specify plot area bounds manually (pixels):
    python3 digitize_cliff_top.py --image annotated.png --location DelMar \\
        --x_min 0 --x_max 1400 --y_min 0 --y_max 70 --line_color green --x_inverted \\
        --plot_left 100 --plot_right 1800 --plot_top 50 --plot_bottom 400
        """
    )

    # Required arguments
    parser.add_argument('--image', required=True, help='Path to annotated PNG image')
    parser.add_argument('--location', required=True, help='Location name (e.g., DelMar, SanElijo)')

    # Axis extent arguments
    parser.add_argument('--x_min', type=float, required=True, help='Minimum x value (alongshore meters)')
    parser.add_argument('--x_max', type=float, required=True, help='Maximum x value (alongshore meters)')
    parser.add_argument('--y_min', type=float, required=True, help='Minimum y value (elevation meters)')
    parser.add_argument('--y_max', type=float, required=True, help='Maximum y value (elevation meters)')

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
    parser.add_argument('--auto_bounds', action='store_true',
                        help='Attempt to auto-detect plot bounds (experimental)')

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

    # Validate image exists
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    # Load image
    print(f"Loading image: {args.image}")
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
    elif args.auto_bounds:
        plot_bounds = detect_plot_bounds(image_array, debug=args.debug)
        if plot_bounds:
            print(f"Auto-detected plot bounds: {plot_bounds}")
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
            filename = f"{args.location}_CliffTop_{resolution}.csv"
            print(f"  {filename}: {len(df)} points")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Save CSVs for each resolution
    for resolution in RESOLUTIONS:
        df = resample_line(x_data, y_data, resolution)
        filename = f"{args.location}_CliffTop_{resolution}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {filename}: {len(df)} points")

    # Generate verification plot
    verify_path = plot_verification(args.location, output_dir, x_data, y_data)
    print(f"Saved verification plot: {verify_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
