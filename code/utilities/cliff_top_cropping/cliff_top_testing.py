#!/usr/bin/env python3
"""
cliff_top_testing.py

Overlays the digitized cliff top cutoff line on the source cliff top image
to verify the digitization was accurate. The digitized line (blue dashed)
should land directly on top of the drawn line (green) in the source image.

Usage:
    python3 cliff_top_testing.py --location DelMar
    python3 cliff_top_testing.py --location SanElijo
    python3 cliff_top_testing.py --all
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform
from PIL import Image

# Import shared functions from digitize_cliff_top (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from digitize_cliff_top import (
    detect_plot_bounds,
    find_image_for_location,
    get_extent_from_shapefile,
    get_elevation_extent,
    HEIGHTS,
)

# Plotting Style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


def get_base_path():
    if platform.system() == "Darwin":
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"


BASE_PATH = get_base_path()


def process_location(location):
    """
    Overlay digitized cliff top line on source image for visual verification.

    Loads the same source image the digitizer read, detects the same plot
    bounds, and maps the CSV data back onto the image. If digitization was
    accurate, the blue dashed line will sit directly on the drawn green line.
    """
    cutoff_dir = os.path.join(BASE_PATH, "utilities", "cliff_top_cutoffs", location)
    output_fig = os.path.join(cutoff_dir, f"{location}_Visual_CliffTop_test.png")
    os.makedirs(cutoff_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"--- {location}: Overlay Verification ---")
    print(f"Output: {output_fig}")

    # 1. Load source image (same one the digitizer read)
    try:
        source_path = find_image_for_location(location)
    except FileNotFoundError as e:
        print(f"    [ERROR] {e}")
        return None

    print(f"    Source: {source_path}")
    img = Image.open(source_path).convert('RGB')
    image_array = np.array(img)
    img_h, img_w = image_array.shape[:2]

    # 2. Detect plot bounds (same function the digitizer used)
    plot_bounds = detect_plot_bounds(image_array)
    px_left, px_right, px_top, px_bottom = plot_bounds
    print(f"    Plot bounds: left={px_left}, right={px_right}, top={px_top}, bottom={px_bottom}")

    # 3. Get axis limits (same sources the digitizer uses)
    try:
        x_min, x_max = get_extent_from_shapefile(location, '1m')
    except (FileNotFoundError, ImportError) as e:
        print(f"    [ERROR] Could not get x extent: {e}")
        return None
    y_min, y_max = get_elevation_extent(location)
    print(f"    Axis limits: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")

    # 4. Crop source image to plot area
    cropped = image_array[px_top:px_bottom + 1, px_left:px_right + 1]

    # 5. Load 25cm cutoff CSV
    cutoff_path = os.path.join(cutoff_dir, f"{location}_Visual_CliffTop_25cm.csv")
    cutoff_df = None
    if os.path.exists(cutoff_path):
        print(f"    Cutoff CSV: {cutoff_path}")
        cutoff_df = pd.read_csv(cutoff_path)
    else:
        print(f"    [WARNING] Cutoff file not found: {cutoff_path}")

    # 6. Create figure matching source image pixel dimensions
    dpi = 150
    fig_w = img_w / dpi
    fig_h = img_h / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # Display cropped source image with data-coordinate extent.
    # Source images have x inverted (decreasing left to right), so
    # the left edge of the crop = x_max, right edge = x_min.
    # origin='upper' because the first image row is the top (y_max).
    ax.imshow(cropped,
              extent=[x_max, x_min, y_min, y_max],
              aspect='auto',
              origin='upper')

    # 7. Overlay digitized line in data coordinates
    if cutoff_df is not None:
        cutoff_df = cutoff_df.sort_values('Polygon_ID')
        cutoff_meters = cutoff_df['Polygon_ID'].values * 0.25
        ax.plot(cutoff_meters, cutoff_df['CliffTop_Z'].values,
                color='blue', linewidth=2, linestyle='--',
                label='Digitized Cutoff')
        ax.legend(loc='upper right', fontsize='small')

    ax.set_xlim(x_max, x_min)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Alongshore Position (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"{location}: Digitized Line Verification (25cm)", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_fig, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Done. Saved to {output_fig}")

    return output_fig


def main():
    parser = argparse.ArgumentParser(
        description="Verify digitized cliff top by overlaying on source image."
    )
    parser.add_argument('--location', type=str, default=None,
                        help='Location name (e.g., DelMar, SanElijo)')
    parser.add_argument('--all', action='store_true',
                        help='Process all locations')

    args = parser.parse_args()

    if args.all:
        locations = list(HEIGHTS.keys())
        print(f"Processing all locations: {locations}")
        for loc in locations:
            try:
                process_location(loc)
            except Exception as e:
                print(f"[ERROR] {loc}: {e}")
    elif args.location:
        if args.location not in HEIGHTS:
            print(f"Unknown location '{args.location}'. Available: {list(HEIGHTS.keys())}")
            return
        process_location(args.location)
    else:
        parser.print_help()
        print(f"\nAvailable locations: {list(HEIGHTS.keys())}")


if __name__ == "__main__":
    main()
