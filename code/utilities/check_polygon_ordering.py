#!/usr/bin/env python3
"""
check_polygon_ordering.py

Diagnostic script to verify whether shapefile polygons are ordered
south-to-north (by increasing UTM Y / northing).

If Polygon_ID doesn't increase monotonically with northing, the
cliff-top cutoff files (which assume Polygon_ID * res = alongshore_m)
will be misaligned.

Usage:
    python3 check_polygon_ordering.py                    # All locations
    python3 check_polygon_ordering.py --location Torrey  # Single location
"""

import os
import argparse
import platform
import numpy as np
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except ImportError:
    print("ERROR: geopandas required. Install with: pip install geopandas")
    exit(1)


def get_base_path():
    """Return base path based on OS."""
    if platform.system() == "Darwin":
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"


LOCATIONS = ['DelMar', 'Solana', 'Encinitas', 'SanElijo', 'Torrey', 'Blacks']
RESOLUTIONS = ['1m', '25cm', '10cm']


def find_shapefile(location, resolution='1m'):
    """Locate the shapefile for a given location and resolution."""
    base = get_base_path()
    sf_root = os.path.join(base, 'utilities', 'shape_files')

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

    if not os.path.isfile(shp_path):
        return None

    return shp_path


def check_ordering(location, resolution='1m', verbose=True):
    """
    Check if polygons are ordered south-to-north.

    Returns:
        dict with keys: 'ordered', 'n_polygons', 'northing_range',
                       'polygon_ids', 'northings', 'out_of_order_count'
    """
    shp_path = find_shapefile(location, resolution)
    if shp_path is None:
        return {'error': f'Shapefile not found for {location} at {resolution}'}

    gdf = gpd.read_file(shp_path)
    gdf['Polygon_ID'] = gdf.index

    # Get centroid northing (UTM Y)
    centroids = gdf.geometry.centroid
    northings = np.array([c.y for c in centroids])
    polygon_ids = gdf['Polygon_ID'].values

    # Check if northings are monotonically increasing with Polygon_ID
    northing_diffs = np.diff(northings)
    out_of_order = np.sum(northing_diffs < 0)
    is_ordered = out_of_order == 0

    # Check correlation
    correlation = np.corrcoef(polygon_ids, northings)[0, 1]

    result = {
        'ordered': is_ordered,
        'n_polygons': len(gdf),
        'northing_min': northings.min(),
        'northing_max': northings.max(),
        'polygon_ids': polygon_ids,
        'northings': northings,
        'out_of_order_count': out_of_order,
        'correlation': correlation,
        'shp_path': shp_path
    }

    if verbose:
        status = "✓ ORDERED" if is_ordered else f"✗ NOT ORDERED ({out_of_order} reversals)"
        print(f"\n{location} ({resolution}):")
        print(f"  Shapefile: {os.path.basename(shp_path)}")
        print(f"  Polygons: {len(gdf)}")
        print(f"  Northing range: {northings.min():.1f} - {northings.max():.1f} m")
        print(f"  Correlation (ID vs Northing): {correlation:.4f}")
        print(f"  Status: {status}")

        if not is_ordered:
            # Find the worst violations
            worst_idx = np.argsort(northing_diffs)[:5]
            print(f"  Worst reversals (Polygon_ID -> next):")
            for idx in worst_idx:
                if northing_diffs[idx] < 0:
                    print(f"    ID {polygon_ids[idx]} -> {polygon_ids[idx+1]}: "
                          f"northing drops {northing_diffs[idx]:.1f} m")

    return result


def plot_ordering(results, output_path=None):
    """
    Create visualization of polygon ordering for all locations.
    """
    n_locations = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (location, data) in enumerate(results.items()):
        ax = axes[i]

        if 'error' in data:
            ax.text(0.5, 0.5, data['error'], ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(location)
            continue

        polygon_ids = data['polygon_ids']
        northings = data['northings']

        # Color by whether each segment is in order
        colors = ['green' if data['ordered'] else 'red']

        ax.scatter(polygon_ids, northings, s=1, alpha=0.5,
                  c='green' if data['ordered'] else 'red')

        # Add trend line
        z = np.polyfit(polygon_ids, northings, 1)
        p = np.poly1d(z)
        ax.plot(polygon_ids, p(polygon_ids), 'b--', alpha=0.5,
               label=f'Trend (r={data["correlation"]:.3f})')

        status = "✓ Ordered" if data['ordered'] else f"✗ {data['out_of_order_count']} reversals"
        ax.set_title(f"{location}\n{status}", fontsize=12,
                    color='green' if data['ordered'] else 'red')
        ax.set_xlabel('Polygon_ID (shapefile index)')
        ax.set_ylabel('UTM Northing (m)')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_locations, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Polygon Ordering Check: Polygon_ID vs UTM Northing\n'
                '(Should increase monotonically for cliff-top cutoffs to work correctly)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nSaved figure: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Check polygon ordering in shapefiles")
    parser.add_argument('--location', help='Single location to check (default: all)')
    parser.add_argument('--resolution', default='1m', choices=RESOLUTIONS,
                       help='Resolution to check (default: 1m)')
    parser.add_argument('--output', help='Output figure path (default: display)')
    parser.add_argument('--all-resolutions', action='store_true',
                       help='Check all resolutions')
    args = parser.parse_args()

    locations = [args.location] if args.location else LOCATIONS
    resolutions = RESOLUTIONS if args.all_resolutions else [args.resolution]

    print("=" * 60)
    print("POLYGON ORDERING CHECK")
    print("=" * 60)
    print(f"Checking if Polygon_ID increases with UTM Northing (south-to-north)")
    print(f"This is required for cliff-top cutoff files to align correctly.")

    all_results = {}

    for resolution in resolutions:
        if len(resolutions) > 1:
            print(f"\n{'=' * 60}")
            print(f"Resolution: {resolution}")
            print("=" * 60)

        results = {}
        for location in locations:
            results[location] = check_ordering(location, resolution)

        all_results[resolution] = results

        # Summary
        ordered_count = sum(1 for r in results.values() if r.get('ordered', False))
        print(f"\n--- Summary ({resolution}) ---")
        print(f"Ordered: {ordered_count}/{len(locations)}")

        if ordered_count < len(locations):
            print("\nLocations needing reindexing:")
            for loc, data in results.items():
                if not data.get('ordered', True):
                    print(f"  - {loc}")

    # Plot (use first resolution for simplicity)
    first_res = resolutions[0]
    output_path = args.output or None
    plot_ordering(all_results[first_res], output_path)


if __name__ == "__main__":
    main()
