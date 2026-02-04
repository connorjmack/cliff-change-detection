#!/usr/bin/env python3
"""
generate_polygons_from_transects.py

Generates polygon shapefiles from transect lines for use in step 7 gridding.
Transects are buffered to create 1m-wide strips, with Polygon_ID assigned
based on transect order (ensuring south-to-north ordering).

For finer resolutions (25cm, 10cm), transects are subdivided by interpolating
additional lines between existing transects.

Usage:
    python3 generate_polygons_from_transects.py --location Torrey
    python3 generate_polygons_from_transects.py --all
    python3 generate_polygons_from_transects.py --location Torrey --resolution 25cm
"""

import os
import argparse
import platform
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


def get_base_path():
    """Return base path based on OS."""
    if platform.system() == "Darwin":
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"


def get_repo_path():
    """Return the repository root path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(script_dir))


LOCATIONS = ['Blacks', 'Encinitas', 'SanElijo', 'Solana', 'Torrey', 'DelMar']
RESOLUTIONS = ['1m', '25cm', '10cm']

# MOP ranges for naming convention
MOP_RANGES = {
    'DelMar': (595, 620),
    'Solana': (637, 666),
    'Encinitas': (708, 764),
    'SanElijo': (683, 708),
    'Torrey': (567, 581),
    'Blacks': (520, 567),
}


def resolution_to_meters(resolution):
    """Convert resolution string to meters."""
    mapping = {'1m': 1.0, '25cm': 0.25, '10cm': 0.10}
    return mapping.get(resolution, 1.0)


def parse_mop_id(tr_id):
    """
    Parse tr_id string to numeric MOP value.

    Examples:
        "MOP 567" -> 567.000
        "MOP 567_001" -> 567.001
        "MOP 580_100" -> 580.100
    """
    import re
    # Remove "MOP " prefix
    cleaned = tr_id.replace("MOP ", "").strip()

    if "_" in cleaned:
        base, offset = cleaned.split("_")
        # Convert offset to decimal (e.g., "001" -> 0.001, "100" -> 0.100)
        return float(base) + float(offset) / 1000.0
    else:
        return float(cleaned)


def format_mop_id(mop_value, resolution):
    """
    Format MOP value with appropriate decimal places for resolution.

    Args:
        mop_value: Numeric MOP value (e.g., 567.001)
        resolution: Resolution string ('1m', '25cm', '10cm')

    Returns:
        Formatted string (e.g., "567.001" for 1m, "567.00025" for 25cm)
    """
    if resolution == '1m':
        # 3 decimal places for 1m (e.g., 567.001)
        return f"{mop_value:.3f}"
    elif resolution == '25cm':
        # 5 decimal places for 25cm (e.g., 567.00025)
        return f"{mop_value:.5f}"
    elif resolution == '10cm':
        # 4 decimal places for 10cm (e.g., 567.0001)
        return f"{mop_value:.4f}"
    else:
        return f"{mop_value:.3f}"


def find_transect_file(location):
    """Find the transect shapefile for a location."""
    repo_path = get_repo_path()
    transect_path = os.path.join(repo_path, 'utilities', 'transects',
                                  f'{location}_1m', 'transect_lines.shp')
    if os.path.exists(transect_path):
        return transect_path
    return None


def interpolate_line(line1, line2, fraction):
    """
    Interpolate a new line between two existing lines.

    Args:
        line1, line2: LineString geometries
        fraction: Position between lines (0 = at line1, 1 = at line2)

    Returns:
        Interpolated LineString
    """
    coords1 = np.array(line1.coords)
    coords2 = np.array(line2.coords)

    # Handle different numbers of vertices by resampling
    if len(coords1) != len(coords2):
        # Use the line with more points as reference
        n_points = max(len(coords1), len(coords2))
        coords1 = np.array([line1.interpolate(t, normalized=True).coords[0]
                           for t in np.linspace(0, 1, n_points)])
        coords2 = np.array([line2.interpolate(t, normalized=True).coords[0]
                           for t in np.linspace(0, 1, n_points)])

    # Linear interpolation
    new_coords = coords1 + fraction * (coords2 - coords1)
    return LineString(new_coords)


def subdivide_transects(transects_gdf, target_resolution):
    """
    Subdivide 1m transects to create finer resolution transects.

    Args:
        transects_gdf: GeoDataFrame with 1m transect lines
        target_resolution: Target resolution in meters (e.g., 0.25 for 25cm)

    Returns:
        GeoDataFrame with subdivided transects (includes mop_value column)
    """
    source_spacing = 1.0  # 1m transects
    subdivisions = int(source_spacing / target_resolution)

    if subdivisions <= 1:
        # Still need to add mop_value for 1m case
        result = transects_gdf.copy()
        if 'tr_id' in result.columns:
            result['mop_value'] = result['tr_id'].apply(parse_mop_id)
        return result

    new_lines = []
    new_ids = []
    new_mop_values = []

    # Sort by transect_i to ensure order
    transects_sorted = transects_gdf.sort_values('transect_i').reset_index(drop=True)

    # Parse MOP values from tr_id
    if 'tr_id' in transects_sorted.columns:
        mop_values = transects_sorted['tr_id'].apply(parse_mop_id).values
    else:
        # Fallback: use transect_i as MOP value
        mop_values = transects_sorted['transect_i'].values.astype(float)

    for i in range(len(transects_sorted) - 1):
        line1 = transects_sorted.geometry.iloc[i]
        line2 = transects_sorted.geometry.iloc[i + 1]
        mop1 = mop_values[i]
        mop2 = mop_values[i + 1]

        # Create subdivisions between this pair
        for j in range(subdivisions):
            fraction = j / subdivisions
            if j == 0:
                new_line = line1
                new_mop = mop1
            else:
                new_line = interpolate_line(line1, line2, fraction)
                # Interpolate MOP value
                new_mop = mop1 + fraction * (mop2 - mop1)

            new_lines.append(new_line)
            new_ids.append(i * subdivisions + j)
            new_mop_values.append(new_mop)

    # Add the last transect
    new_lines.append(transects_sorted.geometry.iloc[-1])
    new_ids.append(len(new_ids))
    new_mop_values.append(mop_values[-1])

    result = gpd.GeoDataFrame({
        'transect_i': new_ids,
        'mop_value': new_mop_values,
        'geometry': new_lines
    }, crs=transects_gdf.crs)

    return result


def transects_to_polygons(transects_gdf, buffer_distance, resolution='1m'):
    """
    Convert transect lines to polygons by buffering.

    Args:
        transects_gdf: GeoDataFrame with transect lines
        buffer_distance: Half-width of polygon (resolution / 2)
        resolution: Resolution string for MOP_ID formatting

    Returns:
        GeoDataFrame with polygons (includes Polygon_ID and MOP_ID)
    """
    # Buffer each transect line
    polygons = transects_gdf.geometry.buffer(buffer_distance, cap_style=2)  # flat caps

    data = {
        'Polygon_ID': transects_gdf['transect_i'].values,
        'geometry': polygons
    }

    # Add MOP_ID if mop_value is available
    if 'mop_value' in transects_gdf.columns:
        mop_ids = [format_mop_id(v, resolution) for v in transects_gdf['mop_value'].values]
        data['MOP_ID'] = mop_ids

    result = gpd.GeoDataFrame(data, crs=transects_gdf.crs)

    return result


def verify_ordering(gdf):
    """Verify polygons are ordered south-to-north."""
    northings = gdf.geometry.centroid.y.values
    diffs = np.diff(northings)
    reversals = np.sum(diffs < 0)
    correlation = np.corrcoef(gdf['Polygon_ID'].values, northings)[0, 1]
    return reversals == 0, correlation, reversals


def generate_polygons(location, resolution='1m', output_dir=None, dry_run=False):
    """
    Generate polygon shapefile from transects for a location.

    Args:
        location: Location name (e.g., 'Torrey')
        resolution: Target resolution ('1m', '25cm', '10cm')
        output_dir: Output directory (default: server utilities/shape_files)
        dry_run: If True, don't write files

    Returns:
        Path to generated shapefile, or None if failed
    """
    print(f"\n{'='*60}")
    print(f"Generating polygons: {location} at {resolution}")
    print('='*60)

    # Find transect file
    transect_path = find_transect_file(location)
    if transect_path is None:
        print(f"  [ERROR] No transect file found for {location}")
        return None

    print(f"  Input: {transect_path}")

    # Load transects
    transects = gpd.read_file(transect_path)
    print(f"  Loaded {len(transects)} transect lines")

    # Ensure sorted by transect_i
    transects = transects.sort_values('transect_i').reset_index(drop=True)

    # Parse MOP values from tr_id for 1m transects
    if 'tr_id' in transects.columns:
        transects['mop_value'] = transects['tr_id'].apply(parse_mop_id)
        print(f"  MOP range: {transects['mop_value'].min():.3f} - {transects['mop_value'].max():.3f}")

    # Subdivide if needed
    res_m = resolution_to_meters(resolution)
    if res_m < 1.0:
        print(f"  Subdividing for {resolution} resolution...")
        transects = subdivide_transects(transects, res_m)
        print(f"  Created {len(transects)} subdivided transects")

    # Convert to polygons
    buffer_dist = res_m / 2.0
    polygons = transects_to_polygons(transects, buffer_dist, resolution)
    print(f"  Generated {len(polygons)} polygons (buffer={buffer_dist}m)")

    # Verify ordering
    is_ordered, corr, reversals = verify_ordering(polygons)
    status = "✓ ORDERED" if is_ordered else f"✗ {reversals} reversals"
    print(f"  Ordering check: {status} (correlation={corr:.4f})")

    if not is_ordered:
        print("  [WARNING] Polygons are not properly ordered!")

    # Determine output path
    if output_dir is None:
        base = get_base_path()
        output_dir = os.path.join(base, 'utilities', 'shape_files')

    # Create folder name matching existing convention
    mop1, mop2 = MOP_RANGES.get(location, (0, 0))
    folder_name = f"{location}Polygons{mop1}to{mop2}at{resolution}"
    folder_path = os.path.join(output_dir, folder_name)
    shp_path = os.path.join(folder_path, f"{folder_name}.shp")

    print(f"  Output: {shp_path}")

    if dry_run:
        print("  [DRY RUN] Would write shapefile")
        return shp_path

    # Create output directory and save
    os.makedirs(folder_path, exist_ok=True)
    polygons.to_file(shp_path)
    print(f"  ✓ Saved {len(polygons)} polygons")

    return shp_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate polygon shapefiles from transect lines"
    )
    parser.add_argument('--location', help='Location name (e.g., Torrey)')
    parser.add_argument('--all', action='store_true', help='Process all locations')
    parser.add_argument('--resolution', default='1m', choices=RESOLUTIONS,
                       help='Target resolution (default: 1m)')
    parser.add_argument('--all-resolutions', action='store_true',
                       help='Generate all resolutions')
    parser.add_argument('--output-dir', help='Output directory (default: server shape_files)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--local', action='store_true',
                       help='Output to local repo instead of server')

    args = parser.parse_args()

    if not args.location and not args.all:
        parser.error("Must specify --location or --all")

    # Determine locations to process
    if args.all:
        # Only process locations that have transect files
        locations = []
        for loc in LOCATIONS:
            if find_transect_file(loc):
                locations.append(loc)
            else:
                print(f"[SKIP] No transect file for {loc}")
    else:
        locations = [args.location]

    # Determine resolutions to generate
    resolutions = RESOLUTIONS if args.all_resolutions else [args.resolution]

    # Determine output directory
    output_dir = args.output_dir
    if args.local:
        repo = get_repo_path()
        output_dir = os.path.join(repo, 'utilities', 'polygons')

    print("="*60)
    print("POLYGON GENERATION FROM TRANSECTS")
    print("="*60)
    print(f"Locations: {locations}")
    print(f"Resolutions: {resolutions}")
    print(f"Output: {output_dir or 'server (utilities/shape_files)'}")
    if args.dry_run:
        print("[DRY RUN MODE]")

    results = {}
    for location in locations:
        for resolution in resolutions:
            key = f"{location}_{resolution}"
            result = generate_polygons(
                location,
                resolution,
                output_dir=output_dir,
                dry_run=args.dry_run
            )
            results[key] = result

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    success = sum(1 for v in results.values() if v is not None)
    print(f"Generated: {success}/{len(results)}")

    for key, path in results.items():
        status = "✓" if path else "✗"
        print(f"  {status} {key}")


if __name__ == "__main__":
    main()
