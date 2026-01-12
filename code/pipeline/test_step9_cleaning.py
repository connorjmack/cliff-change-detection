#!/usr/bin/env python3
"""
test_step9_cleaning.py

Testing and visualization script for step 9 (grid cleaning and hole filling).
Generates side-by-side comparison figures showing:
  - Original grid vs Cleaned grid
  - Cleaned grid vs Filled grid (erosion only)
  - Cluster assignments before/after

Outputs saved to: figures/testing/

Usage:
    python3 test_step9_cleaning.py SanElijo --resolution 25cm
    python3 test_step9_cleaning.py SanElijo --resolution 25cm --n_samples 5
    python3 test_step9_cleaning.py SanElijo --resolution 25cm --survey 20231015_to_20231101
"""

import os
import re
import platform
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap
from matplotlib import cm
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

DPI = 150
FIGSIZE_COMPARISON = (16, 10)
FIGSIZE_FULL = (20, 12)

# Colormaps
CMAP_EROSION = 'magma_r'      # For M3C2 distance grids
CMAP_CLUSTERS = 'tab20'        # For cluster ID grids

# Font sizes
FS_TITLE = 14
FS_LABEL = 12
FS_TICK = 10
FS_SUPTITLE = 16

# ============================================================================
# PATH HELPERS
# ============================================================================

def get_base_dir():
    """Get base directory based on platform."""
    if platform.system() == 'Darwin':
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"


def get_output_dir():
    """Get output directory for test figures."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(repo_root, "figures", "testing")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def resolution_to_label(resolution):
    """Convert resolution string to file label."""
    res_map = {'10cm': '10cm', '25cm': '25cm', '1m': '100cm'}
    return res_map.get(resolution, resolution)


def parse_dates_from_folder(folder_name):
    """Extract date pair from folder name."""
    match = re.search(r'(\d{8})_to_(\d{8})', folder_name)
    if match:
        d1 = datetime.strptime(match.group(1), '%Y%m%d')
        d2 = datetime.strptime(match.group(2), '%Y%m%d')
        return d1, d2
    return None, None


# ============================================================================
# DATA LOADING
# ============================================================================

def load_grid_csv(filepath):
    """Load a grid CSV file and return data with metadata."""
    if not os.path.exists(filepath):
        return None, None, None

    try:
        df = pd.read_csv(filepath, index_col=0, header=0,
                         na_values=['', 'nan', 'NaN', 'NULL'])
        header_labels = df.columns.tolist()
        row_labels = df.index.astype(str).tolist()
        data = df.values.astype(float)
        return data, header_labels, row_labels
    except Exception as e:
        print(f"  [ERROR] Loading {filepath}: {e}")
        return None, None, None


def find_survey_folders(erosion_dir, n_samples=None, specific_survey=None):
    """Find survey folders with grid data."""
    if not os.path.isdir(erosion_dir):
        return []

    folders = []
    for name in sorted(os.listdir(erosion_dir)):
        path = os.path.join(erosion_dir, name)
        if os.path.isdir(path) and '_to_' in name:
            if specific_survey and specific_survey not in name:
                continue
            folders.append((name, path))

    if n_samples and len(folders) > n_samples:
        # Sample evenly across the dataset
        step = len(folders) // n_samples
        folders = folders[::step][:n_samples]

    return folders


def load_survey_grids(folder_path, resolution):
    """Load all grid versions for a survey."""
    label = resolution_to_label(resolution)

    grids = {}
    files_map = {
        'grid_original': f'grid_{label}.csv',
        'grid_cleaned': f'grid_{label}_cleaned.csv',
        'grid_filled': f'grid_{label}_filled.csv',
        'clusters_original': f'clusters_{label}.csv',
        'clusters_cleaned': f'clusters_{label}_cleaned.csv',
        'clusters_filled': f'clusters_{label}_filled.csv',
    }

    for key, filename in files_map.items():
        filepath = os.path.join(folder_path, filename)
        data, headers, rows = load_grid_csv(filepath)
        if data is not None:
            grids[key] = {
                'data': data,
                'headers': headers,
                'rows': rows,
                'filepath': filepath
            }

    return grids


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def get_erosion_cmap(vmax=2.0):
    """Create erosion colormap with white at 0."""
    base_cmap = cm.get_cmap(CMAP_EROSION, 256)
    newcolors = base_cmap(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 1])  # White for 0
    return LinearSegmentedColormap.from_list("White_magma_r", newcolors), Normalize(vmin=0, vmax=vmax)


def get_cluster_cmap(n_clusters):
    """Create cluster colormap with white for 0."""
    if n_clusters <= 0:
        n_clusters = 1
    base_cmap = cm.get_cmap(CMAP_CLUSTERS, max(20, n_clusters))
    colors = base_cmap(np.linspace(0, 1, max(20, n_clusters)))
    # Insert white at beginning for cluster 0 (no cluster)
    colors_with_white = np.vstack([[1, 1, 1, 1], colors])
    return ListedColormap(colors_with_white)


def plot_grid_heatmap(ax, data, title, cmap, norm=None, show_colorbar=True,
                      resolution=0.25, xlabel=True, ylabel=True):
    """Plot a single grid as heatmap."""
    # Transpose so rows=elevation (y-axis), cols=polygon (x-axis)
    plot_data = data.T

    # Handle NaN and zero
    plot_data = np.nan_to_num(plot_data, nan=0)

    # Take absolute value for erosion (stored as positive in filled grids)
    if norm is not None:
        plot_data = np.abs(plot_data)

    # Create extent: x = polygon index, y = elevation
    n_polys, n_elev = data.shape
    extent = [0, n_polys, 0, n_elev * resolution]

    im = ax.imshow(plot_data, aspect='auto', origin='lower',
                   cmap=cmap, norm=norm, extent=extent, interpolation='nearest')

    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold')
    if xlabel:
        ax.set_xlabel("Polygon Index", fontsize=FS_LABEL)
    if ylabel:
        ax.set_ylabel("Elevation (m)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=FS_TICK)
        return im, cbar

    return im, None


def compute_diff_stats(original, cleaned, filled=None):
    """Compute statistics about the cleaning/filling process."""
    stats = {}

    # Original stats
    orig_nonzero = np.sum((original != 0) & ~np.isnan(original))
    orig_volume = np.nansum(np.abs(original))
    stats['original_cells'] = orig_nonzero
    stats['original_volume'] = orig_volume

    # Cleaned stats
    clean_nonzero = np.sum((cleaned != 0) & ~np.isnan(cleaned))
    clean_volume = np.nansum(np.abs(cleaned))
    stats['cleaned_cells'] = clean_nonzero
    stats['cleaned_volume'] = clean_volume
    stats['cells_removed_cleaning'] = orig_nonzero - clean_nonzero
    stats['volume_removed_cleaning'] = orig_volume - clean_volume

    # Filled stats
    if filled is not None:
        fill_nonzero = np.sum((filled != 0) & ~np.isnan(filled))
        fill_volume = np.nansum(np.abs(filled))
        stats['filled_cells'] = fill_nonzero
        stats['filled_volume'] = fill_volume
        stats['cells_added_filling'] = fill_nonzero - clean_nonzero
        stats['volume_added_filling'] = fill_volume - clean_volume

    return stats


def create_comparison_figure(survey_name, grids, resolution, output_dir):
    """Create a comparison figure for a single survey."""

    res_val = {'10cm': 0.10, '25cm': 0.25, '1m': 1.0}.get(resolution, 0.25)

    # Check what data we have
    has_original = 'grid_original' in grids
    has_cleaned = 'grid_cleaned' in grids
    has_filled = 'grid_filled' in grids

    if not has_original and not has_cleaned:
        print(f"  [SKIP] No grid data for {survey_name}")
        return None

    # Determine grid layout
    n_cols = sum([has_original, has_cleaned, has_filled])
    if n_cols == 0:
        return None

    # Get data
    orig_data = grids.get('grid_original', {}).get('data')
    clean_data = grids.get('grid_cleaned', {}).get('data')
    fill_data = grids.get('grid_filled', {}).get('data')

    # Compute vmax from all data
    all_data = [d for d in [orig_data, clean_data, fill_data] if d is not None]
    vmax = max(np.nanmax(np.abs(d)) for d in all_data) if all_data else 2.0
    vmax = min(vmax, 3.0)  # Cap at 3m for visibility

    cmap, norm = get_erosion_cmap(vmax)

    # Create figure
    fig, axes = plt.subplots(2, n_cols, figsize=FIGSIZE_COMPARISON,
                             gridspec_kw={'height_ratios': [3, 1]})
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Parse dates for title
    d1, d2 = parse_dates_from_folder(survey_name)
    date_str = f"{d1.strftime('%Y-%m-%d')} to {d2.strftime('%Y-%m-%d')}" if d1 else survey_name

    fig.suptitle(f"Step 9 Cleaning Test: {date_str}\nResolution: {resolution}",
                 fontsize=FS_SUPTITLE, fontweight='bold')

    col_idx = 0
    stats_text = []

    # Plot Original
    if has_original and orig_data is not None:
        plot_grid_heatmap(axes[0, col_idx], orig_data, "Original Grid",
                          cmap, norm, resolution=res_val)

        # Stats panel
        axes[1, col_idx].axis('off')
        orig_cells = np.sum((orig_data != 0) & ~np.isnan(orig_data))
        orig_vol = np.nansum(np.abs(orig_data)) * (res_val ** 2)
        axes[1, col_idx].text(0.5, 0.5,
                              f"Cells: {orig_cells:,}\nVolume: {orig_vol:.2f} m³",
                              ha='center', va='center', fontsize=FS_LABEL,
                              transform=axes[1, col_idx].transAxes,
                              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        col_idx += 1

    # Plot Cleaned
    if has_cleaned and clean_data is not None:
        plot_grid_heatmap(axes[0, col_idx], clean_data, "Cleaned Grid",
                          cmap, norm, resolution=res_val)

        # Stats panel
        axes[1, col_idx].axis('off')
        clean_cells = np.sum((clean_data != 0) & ~np.isnan(clean_data))
        clean_vol = np.nansum(np.abs(clean_data)) * (res_val ** 2)

        if orig_data is not None:
            removed = np.sum((orig_data != 0) & ~np.isnan(orig_data)) - clean_cells
            axes[1, col_idx].text(0.5, 0.5,
                                  f"Cells: {clean_cells:,}\nVolume: {clean_vol:.2f} m³\n"
                                  f"Removed: {removed:,} cells",
                                  ha='center', va='center', fontsize=FS_LABEL,
                                  transform=axes[1, col_idx].transAxes,
                                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        else:
            axes[1, col_idx].text(0.5, 0.5,
                                  f"Cells: {clean_cells:,}\nVolume: {clean_vol:.2f} m³",
                                  ha='center', va='center', fontsize=FS_LABEL,
                                  transform=axes[1, col_idx].transAxes,
                                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        col_idx += 1

    # Plot Filled
    if has_filled and fill_data is not None:
        plot_grid_heatmap(axes[0, col_idx], fill_data, "Filled Grid (+ Cleanup)",
                          cmap, norm, resolution=res_val)

        # Stats panel
        axes[1, col_idx].axis('off')
        fill_cells = np.sum((fill_data != 0) & ~np.isnan(fill_data))
        fill_vol = np.nansum(np.abs(fill_data)) * (res_val ** 2)

        if clean_data is not None:
            added = fill_cells - np.sum((clean_data != 0) & ~np.isnan(clean_data))
            vol_added = fill_vol - np.nansum(np.abs(clean_data)) * (res_val ** 2)
            axes[1, col_idx].text(0.5, 0.5,
                                  f"Cells: {fill_cells:,}\nVolume: {fill_vol:.2f} m³\n"
                                  f"Net change: {added:+,} cells\n"
                                  f"Vol change: {vol_added:+.2f} m³",
                                  ha='center', va='center', fontsize=FS_LABEL,
                                  transform=axes[1, col_idx].transAxes,
                                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        else:
            axes[1, col_idx].text(0.5, 0.5,
                                  f"Cells: {fill_cells:,}\nVolume: {fill_vol:.2f} m³",
                                  ha='center', va='center', fontsize=FS_LABEL,
                                  transform=axes[1, col_idx].transAxes,
                                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, f"step9_test_{survey_name}_{resolution}.png")
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  [SAVED] {output_path}")
    return output_path


def create_cluster_comparison_figure(survey_name, grids, resolution, output_dir):
    """Create a cluster comparison figure for a single survey."""

    res_val = {'10cm': 0.10, '25cm': 0.25, '1m': 1.0}.get(resolution, 0.25)

    # Check what cluster data we have
    has_original = 'clusters_original' in grids
    has_cleaned = 'clusters_cleaned' in grids
    has_filled = 'clusters_filled' in grids

    if not has_original and not has_cleaned:
        return None

    # Get data
    orig_data = grids.get('clusters_original', {}).get('data')
    clean_data = grids.get('clusters_cleaned', {}).get('data')
    fill_data = grids.get('clusters_filled', {}).get('data')

    # Count unique clusters
    all_data = [d for d in [orig_data, clean_data, fill_data] if d is not None]
    if not all_data:
        return None

    max_cluster = max(int(np.nanmax(d)) for d in all_data if np.nanmax(d) > 0)
    cmap = get_cluster_cmap(max_cluster + 1)

    # Determine layout
    n_cols = sum([has_original, has_cleaned, has_filled])
    if n_cols == 0:
        return None

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 8))
    if n_cols == 1:
        axes = [axes]

    # Parse dates
    d1, d2 = parse_dates_from_folder(survey_name)
    date_str = f"{d1.strftime('%Y-%m-%d')} to {d2.strftime('%Y-%m-%d')}" if d1 else survey_name

    fig.suptitle(f"Cluster Assignments: {date_str}", fontsize=FS_SUPTITLE, fontweight='bold')

    col_idx = 0

    if has_original and orig_data is not None:
        plot_grid_heatmap(axes[col_idx], orig_data,
                          f"Original ({int(np.nanmax(orig_data))} clusters)",
                          cmap, show_colorbar=False, resolution=res_val)
        col_idx += 1

    if has_cleaned and clean_data is not None:
        plot_grid_heatmap(axes[col_idx], clean_data,
                          f"Cleaned ({int(np.nanmax(clean_data))} clusters)",
                          cmap, show_colorbar=False, resolution=res_val)
        col_idx += 1

    if has_filled and fill_data is not None:
        plot_grid_heatmap(axes[col_idx], fill_data,
                          f"Filled ({int(np.nanmax(fill_data))} clusters)",
                          cmap, show_colorbar=False, resolution=res_val)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"step9_clusters_{survey_name}_{resolution}.png")
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  [SAVED] {output_path}")
    return output_path


def create_difference_figure(survey_name, grids, resolution, output_dir):
    """Create a difference map showing what changed."""

    res_val = {'10cm': 0.10, '25cm': 0.25, '1m': 1.0}.get(resolution, 0.25)

    orig_data = grids.get('grid_original', {}).get('data')
    clean_data = grids.get('grid_cleaned', {}).get('data')
    fill_data = grids.get('grid_filled', {}).get('data')

    if orig_data is None or clean_data is None:
        return None

    # Compute differences
    # Cleaning removes cells (negative diff)
    clean_diff = np.where(
        (orig_data != 0) & ((clean_data == 0) | np.isnan(clean_data)),
        -np.abs(orig_data),  # Removed (show as negative)
        0
    )

    # Filling adds cells (positive diff)
    fill_diff = None
    if fill_data is not None:
        fill_diff = np.where(
            ((clean_data == 0) | np.isnan(clean_data)) & (fill_data != 0) & ~np.isnan(fill_data),
            np.abs(fill_data),  # Added (show as positive)
            0
        )

    # Create figure
    n_cols = 2 if fill_diff is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 8))
    if n_cols == 1:
        axes = [axes]

    # Parse dates
    d1, d2 = parse_dates_from_folder(survey_name)
    date_str = f"{d1.strftime('%Y-%m-%d')} to {d2.strftime('%Y-%m-%d')}" if d1 else survey_name

    fig.suptitle(f"Change Maps: {date_str}", fontsize=FS_SUPTITLE, fontweight='bold')

    # Removed cells (red colormap)
    vmax_remove = max(np.abs(np.nanmin(clean_diff)), 0.5)
    cmap_remove = plt.cm.Reds

    plot_data = clean_diff.T
    n_polys, n_elev = clean_diff.shape
    extent = [0, n_polys, 0, n_elev * res_val]

    im1 = axes[0].imshow(np.abs(plot_data), aspect='auto', origin='lower',
                         cmap=cmap_remove, vmin=0, vmax=vmax_remove,
                         extent=extent, interpolation='nearest')
    axes[0].set_title(f"Cells Removed by Cleaning\n({np.sum(clean_diff != 0):,} cells)",
                      fontsize=FS_TITLE, fontweight='bold')
    axes[0].set_xlabel("Polygon Index", fontsize=FS_LABEL)
    axes[0].set_ylabel("Elevation (m)", fontsize=FS_LABEL)
    plt.colorbar(im1, ax=axes[0], label="M3C2 Distance (m)")

    # Added cells (green colormap)
    if fill_diff is not None:
        vmax_add = max(np.nanmax(fill_diff), 0.5)
        cmap_add = plt.cm.Greens

        plot_data = fill_diff.T
        im2 = axes[1].imshow(plot_data, aspect='auto', origin='lower',
                             cmap=cmap_add, vmin=0, vmax=vmax_add,
                             extent=extent, interpolation='nearest')
        axes[1].set_title(f"Cells Added by Filling\n({np.sum(fill_diff != 0):,} cells)",
                          fontsize=FS_TITLE, fontweight='bold')
        axes[1].set_xlabel("Polygon Index", fontsize=FS_LABEL)
        axes[1].set_ylabel("Elevation (m)", fontsize=FS_LABEL)
        plt.colorbar(im2, ax=axes[1], label="M3C2 Distance (m)")

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"step9_diff_{survey_name}_{resolution}.png")
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  [SAVED] {output_path}")
    return output_path


def create_summary_figure(location, resolution, all_stats, output_dir):
    """Create a summary figure across all tested surveys."""

    if not all_stats:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    surveys = [s['survey'] for s in all_stats]
    x = np.arange(len(surveys))

    # Panel A: Cells before/after cleaning
    if all(s.get('original_cells') is not None for s in all_stats):
        orig_cells = [s['original_cells'] for s in all_stats]
        clean_cells = [s['cleaned_cells'] for s in all_stats]

        width = 0.35
        axes[0, 0].bar(x - width/2, orig_cells, width, label='Original', color='steelblue', alpha=0.8)
        axes[0, 0].bar(x + width/2, clean_cells, width, label='Cleaned', color='darkorange', alpha=0.8)
        axes[0, 0].set_ylabel("Cell Count", fontsize=FS_LABEL)
        axes[0, 0].set_title("A. Cell Counts: Original vs Cleaned", fontsize=FS_TITLE, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([s[:10] for s in surveys], rotation=45, ha='right', fontsize=8)

    # Panel B: Volume before/after
    if all(s.get('original_volume') is not None for s in all_stats):
        orig_vol = [s['original_volume'] for s in all_stats]
        clean_vol = [s['cleaned_volume'] for s in all_stats]

        axes[0, 1].bar(x - width/2, orig_vol, width, label='Original', color='steelblue', alpha=0.8)
        axes[0, 1].bar(x + width/2, clean_vol, width, label='Cleaned', color='darkorange', alpha=0.8)
        axes[0, 1].set_ylabel("Volume (sum of |M3C2|)", fontsize=FS_LABEL)
        axes[0, 1].set_title("B. Volume: Original vs Cleaned", fontsize=FS_TITLE, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([s[:10] for s in surveys], rotation=45, ha='right', fontsize=8)

    # Panel C: Cells removed by cleaning
    if all(s.get('cells_removed_cleaning') is not None for s in all_stats):
        removed = [s['cells_removed_cleaning'] for s in all_stats]
        colors = ['red' if r > 0 else 'green' for r in removed]
        axes[1, 0].bar(x, removed, color=colors, alpha=0.8)
        axes[1, 0].axhline(0, color='black', linewidth=0.5)
        axes[1, 0].set_ylabel("Cells Removed", fontsize=FS_LABEL)
        axes[1, 0].set_title("C. Cells Removed by Cleaning", fontsize=FS_TITLE, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([s[:10] for s in surveys], rotation=45, ha='right', fontsize=8)

    # Panel D: Cells added by filling
    if all(s.get('cells_added_filling') is not None for s in all_stats):
        added = [s.get('cells_added_filling', 0) for s in all_stats]
        colors = ['green' if a > 0 else 'red' for a in added]
        axes[1, 1].bar(x, added, color=colors, alpha=0.8)
        axes[1, 1].axhline(0, color='black', linewidth=0.5)
        axes[1, 1].set_ylabel("Net Cell Change (Fill - Cleanup)", fontsize=FS_LABEL)
        axes[1, 1].set_title("D. Net Cell Change from Filling + Cleanup", fontsize=FS_TITLE, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([s[:10] for s in surveys], rotation=45, ha='right', fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "No filled grids available",
                        ha='center', va='center', fontsize=FS_LABEL,
                        transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("D. Cells Added by Filling", fontsize=FS_TITLE, fontweight='bold')

    fig.suptitle(f"Step 9 Cleaning Summary: {location} ({resolution})",
                 fontsize=FS_SUPTITLE, fontweight='bold')

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"step9_summary_{location}_{resolution}_{timestamp}.png")
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\n[SAVED] Summary: {output_path}")
    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test and visualize Step 9 cleaning results')
    parser.add_argument('location', help='Location name (e.g., SanElijo)')
    parser.add_argument('--resolution', choices=['10cm', '25cm', '1m'], default='25cm',
                        help='Grid resolution (default: 25cm)')
    parser.add_argument('--n_samples', type=int, default=3,
                        help='Number of surveys to sample (default: 3)')
    parser.add_argument('--survey', type=str, default=None,
                        help='Specific survey folder to test (e.g., 20231015_to_20231101)')
    parser.add_argument('--skip_clusters', action='store_true',
                        help='Skip cluster comparison figures')
    parser.add_argument('--skip_diff', action='store_true',
                        help='Skip difference map figures')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"STEP 9 CLEANING TEST: {args.location}")
    print(f"Resolution: {args.resolution}")
    print(f"{'='*70}\n")

    # Setup paths
    base_dir = get_base_dir()
    erosion_dir = os.path.join(base_dir, 'results', args.location, 'erosion')
    output_dir = get_output_dir()

    print(f"Input directory: {erosion_dir}")
    print(f"Output directory: {output_dir}\n")

    if not os.path.isdir(erosion_dir):
        print(f"[ERROR] Erosion directory not found: {erosion_dir}")
        return

    # Find survey folders
    folders = find_survey_folders(erosion_dir,
                                  n_samples=args.n_samples if not args.survey else None,
                                  specific_survey=args.survey)

    if not folders:
        print("[ERROR] No survey folders found")
        return

    print(f"Found {len(folders)} survey(s) to process:\n")

    all_stats = []

    for survey_name, folder_path in folders:
        print(f"Processing: {survey_name}")

        # Load grids
        grids = load_survey_grids(folder_path, args.resolution)

        if not grids:
            print(f"  [SKIP] No grid files found")
            continue

        print(f"  Loaded: {', '.join(grids.keys())}")

        # Create comparison figure
        create_comparison_figure(survey_name, grids, args.resolution, output_dir)

        # Create cluster comparison
        if not args.skip_clusters:
            create_cluster_comparison_figure(survey_name, grids, args.resolution, output_dir)

        # Create difference map
        if not args.skip_diff:
            create_difference_figure(survey_name, grids, args.resolution, output_dir)

        # Collect stats
        orig_data = grids.get('grid_original', {}).get('data')
        clean_data = grids.get('grid_cleaned', {}).get('data')
        fill_data = grids.get('grid_filled', {}).get('data')

        if orig_data is not None and clean_data is not None:
            stats = compute_diff_stats(orig_data, clean_data, fill_data)
            stats['survey'] = survey_name
            all_stats.append(stats)

        print()

    # Create summary figure
    if len(all_stats) > 1:
        create_summary_figure(args.location, args.resolution, all_stats, output_dir)

    print(f"\n{'='*70}")
    print(f"Testing complete! Figures saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
