#!/usr/bin/env python3
"""
make_risk_maps.py

Creates satellite-based risk maps by aggregating QCed event lists into 10m alongshore bins.
Generates heatmaps showing erosion and deposition activity patterns.

Output formats:
    - CSV: Aggregated metrics per bin with coordinates (for GIS import)
    - Heatmap plots: Visual representation of activity intensity
    - GeoJSON: Spatial data for satellite overlay (optional)

Usage:
    python3 make_risk_maps.py SanElijo
    python3 make_risk_maps.py --all
    python3 make_risk_maps.py SanElijo --bin-size 20 --qc-filter real
    python3 make_risk_maps.py SanElijo --export-geojson --shapefile-dir /path/to/shapefiles
"""
import os
import sys
import platform
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    print("[WARNING] geopandas not available. GeoJSON export disabled.")

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    print("[WARNING] contextily not available. Install with: pip install contextily")


# ============================================================================
# CONFIGURATION
# ============================================================================

ALL_LOCATIONS = ['DelMar', 'Solana', 'Encinitas', 'SanElijo', 'Torrey', 'Blacks']

# Default QC flags to include in risk analysis
DEFAULT_QC_FILTER = ['real']  # Options: 'real', 'needs_check', 'noise'


def get_base_dir():
    """Get base directory based on platform."""
    if platform.system() == "Darwin":
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"


# ============================================================================
# DATA LOADING
# ============================================================================

def find_latest_qc_file(qc_dir, location, event_type='erosion'):
    """
    Find the most recent QC'd event list file for a location.

    Args:
        qc_dir: Base QC directory (e.g., results/event_lists_qc/)
        location: Location name (e.g., 'SanElijo')
        event_type: 'erosion' or 'deposition'

    Returns:
        Path to latest QC file, or None if not found
    """
    type_dir = os.path.join(qc_dir, event_type)
    if not os.path.isdir(type_dir):
        return None

    # Find all QC files for this location
    qc_files = [
        f for f in os.listdir(type_dir)
        if f.startswith(f'{location}_events_qc_') and f.endswith('.csv')
    ]

    if not qc_files:
        return None

    # Sort by timestamp (YYYYMMDD_HHMMSS) in filename
    qc_files.sort(reverse=True)
    return os.path.join(type_dir, qc_files[0])


def load_qc_events(qc_file, qc_filter=None):
    """
    Load QC'd event list and filter by QC flags.

    Args:
        qc_file: Path to QC CSV file
        qc_filter: List of QC flags to include (e.g., ['real'])

    Returns:
        DataFrame of filtered events
    """
    if not os.path.exists(qc_file):
        return None

    df = pd.read_csv(qc_file, parse_dates=['mid_date', 'start_date', 'end_date'])

    # Filter by QC flag if column exists
    if 'qc_flag' in df.columns and qc_filter:
        df = df[df['qc_flag'].isin(qc_filter)]

    return df


# ============================================================================
# ALONGSHORE BINNING
# ============================================================================

def create_alongshore_bins(events_df, bin_size=10.0):
    """
    Create alongshore bins and aggregate event statistics.

    Args:
        events_df: DataFrame of events with alongshore positions
        bin_size: Bin size in meters (default: 10m)

    Returns:
        DataFrame with aggregated statistics per bin
    """
    if events_df is None or len(events_df) == 0:
        return None

    # Determine bin edges from min/max alongshore positions
    min_alongshore = events_df['alongshore_start_m'].min()
    max_alongshore = events_df['alongshore_end_m'].max()

    # Create bins aligned to nice numbers
    min_bin = np.floor(min_alongshore / bin_size) * bin_size
    max_bin = np.ceil(max_alongshore / bin_size) * bin_size
    bins = np.arange(min_bin, max_bin + bin_size, bin_size)

    # Assign each event to bins it overlaps
    bin_stats = []

    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        bin_center = (bin_start + bin_end) / 2

        # Find events that overlap this bin
        mask = (
            (events_df['alongshore_start_m'] < bin_end) &
            (events_df['alongshore_end_m'] > bin_start)
        )

        bin_events = events_df[mask]

        if len(bin_events) == 0:
            # Empty bin
            bin_stats.append({
                'bin_start_m': bin_start,
                'bin_end_m': bin_end,
                'bin_center_m': bin_center,
                'event_count': 0,
                'total_volume_m3': 0.0,
                'mean_volume_m3': np.nan,
                'max_volume_m3': np.nan,
                'mean_elevation_m': np.nan,
                'total_width_m': 0.0,
                'mean_height_m': np.nan,
            })
        else:
            # Aggregate statistics
            bin_stats.append({
                'bin_start_m': bin_start,
                'bin_end_m': bin_end,
                'bin_center_m': bin_center,
                'event_count': len(bin_events),
                'total_volume_m3': bin_events['volume'].sum(),
                'mean_volume_m3': bin_events['volume'].mean(),
                'max_volume_m3': bin_events['volume'].max(),
                'mean_elevation_m': bin_events['elevation'].mean(),
                'total_width_m': bin_events['width'].sum(),
                'mean_height_m': bin_events['height'].mean(),
            })

    return pd.DataFrame(bin_stats)


# ============================================================================
# COORDINATE MAPPING (OPTIONAL)
# ============================================================================

def map_alongshore_to_utm(bin_df, shapefile_path):
    """
    Map alongshore positions to UTM coordinates using shapefile.

    Args:
        bin_df: DataFrame with bin_center_m column
        shapefile_path: Path to polygon shapefile with MOP_ID

    Returns:
        bin_df with added 'utm_x' and 'utm_y' columns
    """
    if not HAS_GEO or not os.path.exists(shapefile_path):
        print("  [WARNING] Cannot map to UTM coordinates (shapefile not found)")
        return bin_df

    try:
        gdf = gpd.read_file(shapefile_path)
        centroids = gdf.geometry.centroid

        # Get UTM coordinates and alongshore distances
        coords = {int(pid): (c.x, c.y) for pid, c in zip(gdf.index, centroids)}
        all_y = [c[1] for c in coords.values()]
        min_y = min(all_y)
        alongshore_map = {pid: coord[1] - min_y for pid, coord in coords.items()}

        # Build interpolation mapping
        alongshore_vals = np.array(sorted(alongshore_map.values()))
        utm_x_vals = np.array([coords[pid][0] for pid in sorted(alongshore_map.keys())])
        utm_y_vals = np.array([coords[pid][1] for pid in sorted(alongshore_map.keys())])

        # Interpolate UTM coordinates for bin centers
        bin_df['utm_x'] = np.interp(bin_df['bin_center_m'], alongshore_vals, utm_x_vals)
        bin_df['utm_y'] = np.interp(bin_df['bin_center_m'], alongshore_vals, utm_y_vals)

        print(f"  Mapped {len(bin_df)} bins to UTM coordinates")
        return bin_df

    except Exception as e:
        print(f"  [ERROR] Failed to map coordinates: {e}")
        return bin_df


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_risk_heatmap(erosion_bins, deposition_bins, location, output_dir):
    """
    Create heatmap visualization of erosion and deposition risk.

    Args:
        erosion_bins: DataFrame of erosion bin statistics
        deposition_bins: DataFrame of deposition bin statistics
        location: Location name
        output_dir: Output directory for plots
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'{location} Coastal Risk Map - QCed Events', fontsize=16, fontweight='bold')

    metrics = [
        ('event_count', 'Event Count', 'Events'),
        ('total_volume_m3', 'Total Volume', 'm³'),
        ('mean_volume_m3', 'Mean Event Volume', 'm³')
    ]

    for idx, (metric, title, unit) in enumerate(metrics):
        # Erosion (left column)
        ax_ero = axes[idx, 0]
        if erosion_bins is not None and len(erosion_bins) > 0:
            plot_metric_heatmap(erosion_bins, metric, f'{title} - Erosion', unit,
                               ax_ero, cmap='Reds')
        else:
            ax_ero.text(0.5, 0.5, 'No Erosion Data', ha='center', va='center')
            ax_ero.set_title(f'{title} - Erosion')

        # Deposition (right column)
        ax_dep = axes[idx, 1]
        if deposition_bins is not None and len(deposition_bins) > 0:
            plot_metric_heatmap(deposition_bins, metric, f'{title} - Deposition', unit,
                               ax_dep, cmap='Blues')
        else:
            ax_dep.text(0.5, 0.5, 'No Deposition Data', ha='center', va='center')
            ax_dep.set_title(f'{title} - Deposition')

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{location}_risk_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved heatmap: {output_path}")


def plot_metric_heatmap(bin_df, metric, title, unit, ax, cmap='Reds'):
    """Plot a single metric as a heatmap."""
    x = bin_df['bin_center_m'].values
    y = bin_df[metric].values

    # Filter out NaN and zero values for better visualization
    valid_mask = ~np.isnan(y) & (y > 0)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if len(x_valid) == 0:
        ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center')
        ax.set_title(title)
        return

    # Create color-coded scatter plot
    scatter = ax.scatter(x_valid, np.zeros_like(x_valid), c=y_valid,
                        s=100, cmap=cmap, alpha=0.8, edgecolors='black', linewidths=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.15)
    cbar.set_label(unit, fontsize=10)

    ax.set_xlabel('Alongshore Distance (m)', fontsize=11)
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add statistics text
    stats_text = f'Total: {y_valid.sum():.1f} {unit}\nMax: {y_valid.max():.1f} {unit}\nBins: {len(y_valid)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# ============================================================================
# SATELLITE OVERLAY VISUALIZATION
# ============================================================================

def create_satellite_risk_map(location, shapefile_path, risk_data, output_dir, bin_size=10.0):
    """
    Create a satellite-based risk map with polygon overlays colored by risk score.

    Args:
        location: Location name
        shapefile_path: Path to polygon shapefile
        risk_data: DataFrame with bin_center_m and risk metrics
        output_dir: Output directory
        bin_size: Bin size used for aggregation
    """
    if not HAS_GEO:
        print("  [WARNING] Satellite overlay requires geopandas")
        return

    if not os.path.exists(shapefile_path):
        print(f"  [WARNING] Shapefile not found: {shapefile_path}")
        return

    print(f"  Creating satellite risk map...")

    # Load shapefile
    gdf = gpd.read_file(shapefile_path)

    # Use polygon index as alongshore position (south to north)
    # Polygon index directly corresponds to alongshore meter position
    gdf['polygon_idx'] = gdf.index
    gdf['alongshore_m'] = gdf['polygon_idx'].astype(float)

    # Map risk scores to polygons based on alongshore position
    # Each polygon gets the risk score from its corresponding bin
    def get_risk_score(alongshore_pos):
        # Find which bin this polygon belongs to
        bin_idx = int(np.floor(alongshore_pos / bin_size) * bin_size)

        # Find matching risk data
        matching = risk_data[
            (risk_data['bin_start_m'] <= alongshore_pos) &
            (risk_data['bin_end_m'] > alongshore_pos)
        ]

        if len(matching) > 0:
            return matching.iloc[0]
        return None

    # Assign risk scores to polygons
    risk_scores = []
    event_counts = []
    volumes = []

    for idx, row in gdf.iterrows():
        risk_info = get_risk_score(row['alongshore_m'])
        if risk_info is not None:
            # Use erosion data if available
            if 'event_count_erosion' in risk_info:
                event_counts.append(risk_info['event_count_erosion'])
                volumes.append(risk_info['total_volume_m3_erosion'])
            else:
                event_counts.append(risk_info.get('event_count', 0))
                volumes.append(risk_info.get('total_volume_m3', 0))
        else:
            event_counts.append(0)
            volumes.append(0)

    gdf['event_count'] = event_counts
    gdf['volume_m3'] = volumes

    # Calculate normalized risk score based on volume only
    # Normalize volume to 0-1 range across all polygons
    max_volume = gdf['volume_m3'].max()

    if max_volume > 0:
        gdf['risk_score'] = gdf['volume_m3'] / max_volume
    else:
        gdf['risk_score'] = 0

    # Reproject to Web Mercator for satellite basemap
    gdf_wm = gdf.to_crs(epsg=3857)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 16))

    # Add satellite basemap
    if HAS_CONTEXTILY:
        try:
            # Plot polygons first to set extent
            gdf_wm.plot(ax=ax, alpha=0)  # Invisible plot to set extent

            # Add satellite imagery
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=15)
            print("    Added satellite basemap")
        except Exception as e:
            print(f"    [WARNING] Could not add basemap: {e}")

    # Plot risk polygons with color scale (no edges to prevent black overlap)
    gdf_wm.plot(
        ax=ax,
        column='risk_score',
        cmap='RdYlGn_r',  # Red (high risk) to Green (low risk)
        alpha=0.8,
        edgecolor='none',  # No edges to prevent black appearance
        legend=False,
        vmin=0,
        vmax=1
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Risk Score (P(Failure))', fontsize=12, fontweight='bold')

    # Calculate statistics
    n_bins = len(gdf[gdf['event_count'] > 0])
    mean_risk = gdf[gdf['event_count'] > 0]['risk_score'].mean()
    max_risk = gdf['risk_score'].max()
    high_risk_pct = (gdf['risk_score'] > 0.7).sum() / len(gdf) * 100

    # Add statistics text box
    stats_text = f"Bins: {n_bins} ({bin_size:.0f}m)\n"
    stats_text += f"Mean Risk: {mean_risk:.3f}\n"
    stats_text += f"Max Risk: {max_risk:.3f}\n"
    stats_text += f">0.7: {high_risk_pct:.1f}%"

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
            family='monospace')

    # Add title
    ax.set_title(f'Rockfall Risk Map: {location}', fontsize=16, fontweight='bold', pad=20)

    # Remove axis labels (showing map coordinates)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Add scale bar (approximate)
    # Calculate 500m in Web Mercator at this location
    bounds = gdf_wm.total_bounds
    center_y = (bounds[1] + bounds[3]) / 2

    # Rough conversion: 1 degree ≈ 111km at equator, scaled by cos(latitude)
    # For Southern California (~33°N): cos(33°) ≈ 0.84
    lat_approx = 33  # degrees
    scale_length_m = 500
    scale_length_deg = scale_length_m / (111320 * np.cos(np.radians(lat_approx)))
    scale_length_wm = scale_length_deg * 111320  # Convert to Web Mercator units

    # Position scale bar at bottom center
    x_center = (bounds[0] + bounds[2]) / 2
    y_bottom = bounds[1] + (bounds[3] - bounds[1]) * 0.05

    ax.plot([x_center - scale_length_wm/2, x_center + scale_length_wm/2],
            [y_bottom, y_bottom],
            'k-', linewidth=3, solid_capstyle='butt')
    ax.plot([x_center - scale_length_wm/2, x_center - scale_length_wm/2],
            [y_bottom - scale_length_wm*0.05, y_bottom + scale_length_wm*0.05],
            'k-', linewidth=3)
    ax.plot([x_center + scale_length_wm/2, x_center + scale_length_wm/2],
            [y_bottom - scale_length_wm*0.05, y_bottom + scale_length_wm*0.05],
            'k-', linewidth=3)
    ax.text(x_center, y_bottom - scale_length_wm*0.15, f'{scale_length_m} m',
            ha='center', va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Save figure
    output_path = os.path.join(output_dir, f'{location}_satellite_risk_map.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved satellite risk map: {output_path}")


# ============================================================================
# EXPORT
# ============================================================================

def export_risk_map_csv(erosion_bins, deposition_bins, location, output_dir):
    """
    Export risk map data to CSV format.

    Args:
        erosion_bins: DataFrame of erosion statistics
        deposition_bins: DataFrame of deposition statistics
        location: Location name
        output_dir: Output directory
    """
    # Merge erosion and deposition data
    if erosion_bins is not None and deposition_bins is not None:
        merged = erosion_bins.merge(
            deposition_bins,
            on=['bin_start_m', 'bin_end_m', 'bin_center_m'],
            suffixes=('_erosion', '_deposition'),
            how='outer'
        )
    elif erosion_bins is not None:
        merged = erosion_bins.copy()
        merged.columns = [f'{col}_erosion' if col not in ['bin_start_m', 'bin_end_m', 'bin_center_m']
                         else col for col in merged.columns]
    elif deposition_bins is not None:
        merged = deposition_bins.copy()
        merged.columns = [f'{col}_deposition' if col not in ['bin_start_m', 'bin_end_m', 'bin_center_m']
                         else col for col in merged.columns]
    else:
        print(f"  [WARNING] No data to export for {location}")
        return

    # Calculate combined risk metrics
    if 'event_count_erosion' in merged.columns and 'event_count_deposition' in merged.columns:
        merged['total_event_count'] = merged['event_count_erosion'].fillna(0) + merged['event_count_deposition'].fillna(0)
        merged['total_volume_change_m3'] = merged['total_volume_m3_erosion'].fillna(0) + merged['total_volume_m3_deposition'].fillna(0)

        # Risk score: weighted combination of event count and volume
        merged['risk_score'] = (
            merged['total_event_count'] * 0.5 +  # Normalize by max
            (merged['total_volume_change_m3'] / merged['total_volume_change_m3'].max()) * 50
        )

    # Sort by alongshore position
    merged = merged.sort_values('bin_center_m')

    # Export to CSV
    csv_path = os.path.join(output_dir, f'{location}_risk_map.csv')
    merged.to_csv(csv_path, index=False)
    print(f"  Saved risk map CSV: {csv_path}")

    return merged


def export_geojson(bin_df, location, output_dir, event_type='combined'):
    """
    Export risk map as GeoJSON for satellite overlay.

    Args:
        bin_df: DataFrame with UTM coordinates
        location: Location name
        output_dir: Output directory
        event_type: 'erosion', 'deposition', or 'combined'
    """
    if not HAS_GEO:
        print("  [WARNING] GeoJSON export requires geopandas")
        return

    if 'utm_x' not in bin_df.columns or 'utm_y' not in bin_df.columns:
        print("  [WARNING] UTM coordinates required for GeoJSON export")
        return

    # Create point geometries
    geometry = [Point(x, y) for x, y in zip(bin_df['utm_x'], bin_df['utm_y'])]
    gdf = gpd.GeoDataFrame(bin_df, geometry=geometry, crs='EPSG:26911')  # UTM Zone 11N

    # Export to GeoJSON
    geojson_path = os.path.join(output_dir, f'{location}_risk_map_{event_type}.geojson')
    gdf.to_file(geojson_path, driver='GeoJSON')
    print(f"  Saved GeoJSON: {geojson_path}")


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_location(location, repo_root, args):
    """
    Process a single location and generate risk maps.

    Args:
        location: Location name
        repo_root: Repository root directory
        args: Command line arguments
    """
    print(f"\n{'='*60}")
    print(f"Processing: {location}")
    print(f"{'='*60}")

    qc_dir = os.path.join(repo_root, 'results', 'event_lists_qc')

    # Load QC'd event lists
    ero_file = find_latest_qc_file(qc_dir, location, 'erosion')
    dep_file = find_latest_qc_file(qc_dir, location, 'deposition')

    if not ero_file and not dep_file:
        print(f"  [WARNING] No QC'd event lists found for {location}")
        return

    # Load events
    erosion_events = load_qc_events(ero_file, args.qc_filter) if ero_file else None
    deposition_events = load_qc_events(dep_file, args.qc_filter) if dep_file else None

    if erosion_events is not None:
        print(f"  Loaded {len(erosion_events)} erosion events (QC filter: {args.qc_filter})")
    if deposition_events is not None:
        print(f"  Loaded {len(deposition_events)} deposition events (QC filter: {args.qc_filter})")

    # Create alongshore bins
    print(f"  Creating {args.bin_size}m alongshore bins...")
    erosion_bins = create_alongshore_bins(erosion_events, args.bin_size)
    deposition_bins = create_alongshore_bins(deposition_events, args.bin_size)

    if erosion_bins is not None:
        print(f"    Erosion: {len(erosion_bins)} bins, {erosion_bins['event_count'].sum():.0f} total events")
    if deposition_bins is not None:
        print(f"    Deposition: {len(deposition_bins)} bins, {deposition_bins['event_count'].sum():.0f} total events")

    # Map to UTM coordinates if shapefile provided
    if args.shapefile_dir:
        base_dir = get_base_dir()
        shapefile_path = os.path.join(base_dir, 'utilities', 'shape_files',
                                     f'{location}_polygons_at25cm', f'{location}_polygons_at25cm.shp')

        if os.path.exists(shapefile_path):
            if erosion_bins is not None:
                erosion_bins = map_alongshore_to_utm(erosion_bins, shapefile_path)
            if deposition_bins is not None:
                deposition_bins = map_alongshore_to_utm(deposition_bins, shapefile_path)
        else:
            print(f"  [WARNING] Shapefile not found: {shapefile_path}")

    # Create visualizations
    output_dir = os.path.join(repo_root, 'results', 'risk_maps')
    os.makedirs(output_dir, exist_ok=True)

    plot_risk_heatmap(erosion_bins, deposition_bins, location, output_dir)

    # Export CSV
    combined_df = export_risk_map_csv(erosion_bins, deposition_bins, location, output_dir)

    # Create satellite overlay map using local shapefiles
    # Determine MOP range for shapefile naming
    mop_ranges = {
        'DelMar': '595to620',
        'Solana': '637to666',
        'Encinitas': '708to764',
        'SanElijo': '683to708',
        'Torrey': '567to581',
        'Blacks': '520to567'
    }

    mop_range = mop_ranges.get(location)
    if mop_range:
        shapefile_dir = os.path.join(repo_root, 'utilities', 'polygons',
                                    f'{location}Polygons{mop_range}at1m')
        shapefile_path = os.path.join(shapefile_dir, f'{location}Polygons{mop_range}at1m.shp')

        if os.path.exists(shapefile_path):
            # Use erosion bins for satellite map (primary risk)
            risk_data = erosion_bins if erosion_bins is not None else deposition_bins
            if risk_data is not None:
                create_satellite_risk_map(location, shapefile_path, risk_data, output_dir, args.bin_size)
        else:
            print(f"  [WARNING] Cannot create satellite map - shapefile not found: {shapefile_path}")
    else:
        print(f"  [WARNING] Unknown MOP range for location: {location}")

    # Export GeoJSON if requested and coordinates available
    if args.export_geojson and combined_df is not None:
        export_geojson(combined_df, location, output_dir, 'combined')


def main():
    parser = argparse.ArgumentParser(
        description='Generate satellite-based risk maps from QCed event lists',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 make_risk_maps.py SanElijo
  python3 make_risk_maps.py --all
  python3 make_risk_maps.py SanElijo --bin-size 20
  python3 make_risk_maps.py SanElijo --qc-filter real needs_check
  python3 make_risk_maps.py SanElijo --export-geojson --shapefile-dir
        """
    )

    parser.add_argument('location', nargs='?',
                       help='Location name (e.g., SanElijo)')
    parser.add_argument('--all', action='store_true',
                       help='Process all locations')
    parser.add_argument('--bin-size', type=float, default=10.0,
                       help='Alongshore bin size in meters (default: 10m)')
    parser.add_argument('--qc-filter', nargs='+', default=DEFAULT_QC_FILTER,
                       help='QC flags to include (default: real)')
    parser.add_argument('--export-geojson', action='store_true',
                       help='Export GeoJSON for satellite overlay (requires UTM coordinates)')
    parser.add_argument('--shapefile-dir', action='store_true',
                       help='Enable shapefile lookup for UTM coordinate mapping')

    args = parser.parse_args()

    # Validate arguments
    if args.all and args.location:
        parser.error("Cannot specify both --all and a specific location")
    if not args.all and not args.location:
        parser.error("Must specify either a location or --all")

    # Get locations to process
    locations = ALL_LOCATIONS if args.all else [args.location]

    # Get repository root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))

    print(f"\n{'='*60}")
    print(f"RISK MAP GENERATION")
    print(f"Bin size: {args.bin_size}m")
    print(f"QC filter: {args.qc_filter}")
    print(f"Output: {os.path.join(repo_root, 'results', 'risk_maps')}")
    print(f"{'='*60}")

    # Process each location
    for location in locations:
        try:
            process_location(location, repo_root, args)
        except Exception as e:
            print(f"\n[ERROR] Failed to process {location}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
