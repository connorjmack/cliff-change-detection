#!/usr/bin/env python3
"""
simple_gif.py

Generates a simplified animated dashboard with two panels:
  - Top (2/3): Cumulative erosion heatmap (cliff face view)
  - Bottom (1/3): Cumulative volume timeline with erosion rate

Professional-quality labels and axes for publication/presentation use.

Usage:
    python3 simple_gif.py --location SanElijo
    python3 simple_gif.py --location all --make-gif
"""

import os
import re
import platform
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, MaxNLocator
from datetime import datetime, timedelta

# --- Configuration ---
RESOLUTION = '25cm'
RES_VAL = 0.25
FILE_TAG = '25cm'

# --- COLORS ---
COLOR_VOLUME = '#2C3E50'      # Dark blue-grey for volume line
COLOR_VOLUME_FILL = '#85C1E9' # Light blue for uncertainty band
COLOR_RATE = '#E74C3C'        # Red for rate line
COLOR_SHADING = '#F8F9FA'     # Very light grey for winter bands

# --- COLORMAPS ---
CMAP_HEATMAP = 'magma_r'

DPI = 150
GIF_DPI = 100

# --- FONT SIZES (Publication Quality) ---
FS_TITLE = 22
FS_LABEL = 16
FS_TICK = 14
FS_ANNOTATION = 12

# Available locations
LOCATIONS_ALL = ['DelMar', 'Torrey', 'Solana', 'Encinitas', 'SanElijo']

# ==============================================================================
# 1. PATHS & HELPERS
# ==============================================================================

def get_base_dir():
    if platform.system() == 'Darwin':
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

def get_output_dir(base_dir):
    return os.path.join(base_dir, "figures", "gifs")

def parse_dates(folder_name):
    match = re.search(r'(\d{8})_to_(\d{8})', folder_name)
    if match:
        d1 = datetime.strptime(match.group(1), '%Y%m%d')
        d2 = datetime.strptime(match.group(2), '%Y%m%d')
        return d1, d2
    return None, None

def resolve_interval_paths(interval_dir, interval):
    """Locate grid/cluster/uncertainty files, preferring resolution subfolders."""
    candidate_dirs = [os.path.join(interval_dir, RESOLUTION), interval_dir]
    tags = [FILE_TAG]
    if RESOLUTION not in tags:
        tags.append(RESOLUTION)

    for base in candidate_dirs:
        if not os.path.isdir(base):
            continue
        for tag in tags:
            grid_file = os.path.join(base, f"{interval}_ero_grid_{tag}_filled.csv")
            unc_file = os.path.join(base, f"{interval}_ero_uncertainty_{tag}.csv")
            clus_file = os.path.join(base, f"{interval}_ero_clusters_{tag}_filled.csv")
            if os.path.exists(grid_file) and os.path.exists(clus_file):
                return grid_file, unc_file, clus_file
    return None, None, None

def add_winter_shading(ax, start_date, end_date):
    """Adds subtle shaded bands for Winter (Oct 1 - Mar 31)."""
    start_year = start_date.year - 1
    end_year = end_date.year + 1

    for year in range(start_year, end_year):
        winter_start = datetime(year, 10, 1)
        winter_end = datetime(year + 1, 3, 31)

        if winter_end < start_date or winter_start > end_date:
            continue

        ax.axvspan(winter_start, winter_end, color=COLOR_SHADING, alpha=0.6, zorder=0, linewidth=0)

def get_custom_cmap(name, vmax=6.0):
    """Creates a custom colormap with White at 0."""
    base_cmap = cm.get_cmap(name, 256)
    newcolors = base_cmap(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 1])
    return LinearSegmentedColormap.from_list(f"White_{name}", newcolors), Normalize(vmin=0, vmax=vmax)

# ==============================================================================
# 2. DATA LOADING LOGIC
# ==============================================================================

def load_uncertainty_stats(unc_path):
    if not unc_path or not os.path.exists(unc_path):
        return 0.0
    try:
        df = pd.read_csv(unc_path)
        unc_columns = [col for col in df.columns if col != 'Polygon_ID' and 'Uncertainty' in col]
        if not unc_columns:
            unc_values = df.iloc[:, 1:].values.flatten()
        else:
            unc_values = df[unc_columns].values.flatten()
        unc_values = unc_values[~np.isnan(unc_values)]
        unc_values = unc_values[unc_values > 0]
        return np.mean(unc_values) if len(unc_values) > 0 else 0.0
    except:
        return 0.0

def calculate_volume_bounds_properly(grid_path, unc_path, res_val):
    cell_area = res_val * res_val
    if not os.path.exists(grid_path):
        return None, None, None, None
    try:
        mean_uncertainty = load_uncertainty_stats(unc_path)
        df_grid = pd.read_csv(grid_path, index_col=0)
        df_grid = df_grid.apply(pd.to_numeric, errors='coerce').fillna(0)
        distances = df_grid.values
        erosion_mask = distances > 0.01
        vol_main = distances.sum() * cell_area
        distances_lower = distances.copy()
        distances_lower[erosion_mask] = np.maximum(distances[erosion_mask] - mean_uncertainty, 0)
        vol_lower = distances_lower.sum() * cell_area
        distances_upper = distances.copy()
        distances_upper[erosion_mask] = distances[erosion_mask] + mean_uncertainty
        vol_upper = distances_upper.sum() * cell_area
        return vol_main, vol_lower, vol_upper, df_grid
    except:
        return None, None, None, None

def clean_and_snap_grid(df, resolution_val):
    cleaned_cols = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)
    try:
        col_floats = cleaned_cols.astype(float)
        scale = 1.0 / resolution_val
        new_cols = (col_floats * scale).round().astype(int)
        df.columns = new_cols
        df.index = df.index.astype(int)
        return df
    except:
        return None

def collect_data(base_dir, location):
    print(f"\nCollecting data for {location} ({RESOLUTION})...")
    erosion_dir = os.path.join(base_dir, 'results', location, 'erosion')
    if not os.path.isdir(erosion_dir):
        return [], []

    intervals = sorted([d for d in os.listdir(erosion_dir) if os.path.isdir(os.path.join(erosion_dir, d))])
    stats_list = []
    grids_by_step = []

    for interval in intervals:
        d1, d2 = parse_dates(interval)
        if not d1:
            continue
        folder = os.path.join(erosion_dir, interval)
        grid_file, unc_file, clus_file = resolve_interval_paths(folder, interval)
        if not grid_file:
            continue
        vol_main, vol_low, vol_high, df_grid = calculate_volume_bounds_properly(grid_file, unc_file, RES_VAL)
        if vol_main is None:
            continue

        spatial_df = None
        if df_grid is not None:
            spatial_df = clean_and_snap_grid(df_grid.copy(), RES_VAL)
        grids_by_step.append(spatial_df)

        days = (d2 - d1).days
        if days < 1:
            days = 1
        stats_list.append({
            'start': d1, 'end': d2, 'mid': d1 + (d2 - d1) / 2, 'days': days,
            'volume': vol_main, 'vol_lower': vol_low, 'vol_upper': vol_high
        })
        print(f"  {interval}: Vol={vol_main:.1f} m3")

    return stats_list, grids_by_step

# ==============================================================================
# 3. FRAME PLOTTING
# ==============================================================================

def compute_cumulative_grid(grids_list, up_to_step):
    """Compute cumulative grid up to (and including) step index."""
    cumulative = None
    for i in range(up_to_step + 1):
        if i < len(grids_list) and grids_list[i] is not None:
            if cumulative is None:
                cumulative = grids_list[i].fillna(0.0).copy()
            else:
                cumulative = cumulative.add(grids_list[i].fillna(0.0), fill_value=0)
    return cumulative

def plot_simple_frame(stats, grids_list, step_idx, out_dir, location,
                      global_start, global_end, full_cum_grid, max_cum, max_rate):
    """
    Plot a simplified two-panel frame at a specific time step.
    """
    if not stats or step_idx < 0:
        return None

    # Slice data up to current step
    current_stats = stats[:step_idx + 1]

    # Compute metrics
    df_stats = pd.DataFrame(current_stats)
    df_stats['raw_rate'] = df_stats['volume'] / df_stats['days']
    df_stats['smooth_rate'] = df_stats['raw_rate'].rolling(window=3, center=True, min_periods=1).mean()

    ends = df_stats['end']
    vols = df_stats['volume']
    rates = df_stats['smooth_rate']

    cum_vol = np.cumsum(vols)
    cum_lower = np.cumsum([s['vol_lower'] for s in current_stats])
    cum_upper = np.cumsum([s['vol_upper'] for s in current_stats])

    # Current cumulative grid
    cum_grid = compute_cumulative_grid(grids_list, step_idx)

    # Colormap setup
    VMAX_GRID = 6.0
    cmap_grid, norm_grid = get_custom_cmap(CMAP_HEATMAP, vmax=VMAX_GRID)

    # --- FIGURE SETUP ---
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 0.08, 1], hspace=0.35)

    ax_heatmap = fig.add_subplot(gs[0])
    cbar_ax = fig.add_subplot(gs[1])
    ax_timeline = fig.add_subplot(gs[2])

    # Current date for title
    current_date = current_stats[-1]['end']

    # ==================== TOP PANEL: HEATMAP ====================
    if cum_grid is not None and full_cum_grid is not None:
        plot_df_full = full_cum_grid.T
        x_indices = plot_df_full.columns.astype(int)
        x_meters = x_indices * RES_VAL
        y_elev = plot_df_full.index.astype(int)
        max_elev_m = len(y_elev) * RES_VAL

        # Align current grid to full grid dimensions
        plot_df = cum_grid.T
        plot_df = plot_df.reindex(index=plot_df_full.index, columns=plot_df_full.columns, fill_value=0)
        matrix = plot_df.values
        extent = [x_meters.min(), x_meters.max(), 0, max_elev_m]

        im = ax_heatmap.imshow(matrix, origin='lower', extent=extent,
                               cmap=cmap_grid, norm=norm_grid, aspect='auto', interpolation='none')

        # Colorbar
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Cumulative Erosion Depth (m)', fontsize=FS_LABEL, fontweight='bold')
        cbar.ax.tick_params(labelsize=FS_TICK)

        # Axis formatting
        ax_heatmap.set_xlim(x_meters.max(), x_meters.min())  # Reversed for cliff-facing view
        ax_heatmap.set_ylim(0, max_elev_m)
        ax_heatmap.set_xlabel('Alongshore Position (m)', fontsize=FS_LABEL, fontweight='bold')
        ax_heatmap.set_ylabel('Elevation (m NAVD88)', fontsize=FS_LABEL, fontweight='bold')
        ax_heatmap.tick_params(axis='both', labelsize=FS_TICK)

        # Clean tick formatting
        ax_heatmap.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax_heatmap.yaxis.set_major_locator(MaxNLocator(nbins=8))

    else:
        ax_heatmap.text(0.5, 0.5, "Data Not Available", ha='center', va='center',
                        fontsize=FS_TITLE, transform=ax_heatmap.transAxes)
        ax_heatmap.set_xticks([])
        ax_heatmap.set_yticks([])
        cbar_ax.axis('off')

    ax_heatmap.set_title('Cumulative Cliff Erosion (Cliff-Facing View)',
                         fontsize=FS_TITLE, fontweight='bold', pad=15)

    # ==================== BOTTOM PANEL: TIMELINE ====================
    date_buffer = timedelta(days=30)
    ax_timeline.set_xlim(global_start - date_buffer, global_end + date_buffer)

    # Winter shading
    add_winter_shading(ax_timeline, global_start, global_end)

    # Uncertainty band
    ax_timeline.fill_between(ends, cum_lower, cum_upper, color=COLOR_VOLUME_FILL, alpha=0.5, label='Uncertainty')

    # Cumulative volume line
    ax_timeline.plot(ends, cum_vol, color=COLOR_VOLUME, linewidth=2.5, marker='o',
                     markersize=5, markerfacecolor='white', markeredgecolor=COLOR_VOLUME,
                     markeredgewidth=1.5, label='Cumulative Volume', zorder=5)

    # Rate on twin axis
    ax_rate = ax_timeline.twinx()
    ax_rate.plot(ends, rates, color=COLOR_RATE, linewidth=2, linestyle='--',
                 label='Erosion Rate', zorder=4)

    # Axis limits (consistent across frames)
    ax_timeline.set_ylim(0, max_cum)
    ax_rate.set_ylim(0, max_rate)

    # Labels and formatting
    ax_timeline.set_xlabel('Date', fontsize=FS_LABEL, fontweight='bold')
    ax_timeline.set_ylabel('Cumulative Volume (m$^3$)', fontsize=FS_LABEL, fontweight='bold', color=COLOR_VOLUME)
    ax_rate.set_ylabel('Erosion Rate (m$^3$/day)', fontsize=FS_LABEL, fontweight='bold', color=COLOR_RATE)

    ax_timeline.tick_params(axis='both', labelsize=FS_TICK)
    ax_timeline.tick_params(axis='y', colors=COLOR_VOLUME)
    ax_rate.tick_params(axis='y', labelsize=FS_TICK, colors=COLOR_RATE)

    # Date formatting
    ax_timeline.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_timeline.xaxis.set_major_locator(mdates.YearLocator())
    ax_timeline.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    plt.setp(ax_timeline.get_xticklabels(), rotation=0, ha='center')

    # Grid
    ax_timeline.grid(True, linestyle='-', alpha=0.3, which='major')
    ax_timeline.grid(True, linestyle=':', alpha=0.2, which='minor')

    # Combined legend
    lines1, labels1 = ax_timeline.get_legend_handles_labels()
    lines2, labels2 = ax_rate.get_legend_handles_labels()
    ax_timeline.legend(lines1 + lines2, labels1 + labels2,
                       loc='upper left', fontsize=FS_ANNOTATION, framealpha=0.9)

    ax_timeline.set_title('Cumulative Erosion Volume & Rate (3-Survey Rolling Average)',
                          fontsize=FS_TITLE, fontweight='bold', pad=15)

    # Vertical line at current date
    ax_timeline.axvline(current_date, color='#333333', linewidth=1.5, linestyle='-', alpha=0.7, zorder=10)

    # Main title with current date
    location_display = location.replace('DelMar', 'Del Mar').replace('SanElijo', 'San Elijo')
    fig.suptitle(f'{location_display} Coastal Cliff Erosion  |  {current_date.strftime("%B %d, %Y")}',
                 fontsize=FS_TITLE + 4, fontweight='bold', y=0.98)

    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save frame
    frame_num = str(step_idx).zfill(4)
    out_file = os.path.join(out_dir, f"{location}_simple_frame_{frame_num}.png")
    plt.savefig(out_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    return out_file

def generate_simple_gif_frames(stats, grids_list, out_dir, location):
    """Generate all frames for the simplified animated dashboard."""
    if not stats:
        print(f"  [Warning] No stats data for {location}")
        return []

    stats.sort(key=lambda x: x['end'])

    # Compute global bounds
    global_start = min(s['start'] for s in stats)
    global_end = max(s['end'] for s in stats)

    # Full cumulative grid
    full_cum_grid = compute_cumulative_grid(grids_list, len(grids_list) - 1)

    # Compute axis limits from full data
    full_df_stats = pd.DataFrame(stats)
    max_cum = full_df_stats['vol_upper'].cumsum().max() * 1.1
    max_rate = (full_df_stats['volume'] / full_df_stats['days']).rolling(window=3, min_periods=1).mean().max() * 1.15

    # Create output directory
    location_dir = os.path.join(out_dir, location)
    os.makedirs(location_dir, exist_ok=True)

    frame_files = []
    n_steps = len(stats)
    print(f"\n  Generating {n_steps} simplified frames for {location}...")

    for step_idx in range(n_steps):
        frame_file = plot_simple_frame(
            stats, grids_list, step_idx, location_dir, location,
            global_start, global_end, full_cum_grid, max_cum, max_rate
        )
        if frame_file:
            frame_files.append(frame_file)
            print(f"    Frame {step_idx + 1}/{n_steps}: {os.path.basename(frame_file)}")

    return frame_files

def create_gif(frame_files, out_path, duration=500):
    """Create animated GIF from frame files using PIL."""
    try:
        from PIL import Image
    except ImportError:
        print("  [Warning] PIL not available. Install with: pip install Pillow")
        print("  Frames saved but GIF not created.")
        return False

    if not frame_files:
        print("  [Warning] No frames to create GIF")
        return False

    print(f"  Creating GIF with {len(frame_files)} frames...")
    images = []
    for f in frame_files:
        img = Image.open(f)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)

    # Save as GIF
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"  GIF saved: {out_path}")
    return True

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Simplified Animated Dashboard")
    parser.add_argument('--location', type=str, default='all',
                        help=f"Available: {', '.join(LOCATIONS_ALL)}")
    parser.add_argument('--make-gif', action='store_true',
                        help="Create animated GIF from frames (requires Pillow)")
    parser.add_argument('--duration', type=int, default=150,
                        help="Duration per frame in milliseconds (default: 150)")
    args = parser.parse_args()

    base_dir = get_base_dir()
    out_dir = get_output_dir(base_dir)
    locations = LOCATIONS_ALL if args.location.lower() == 'all' else [args.location]

    print(f"--- Generating simplified animated dashboard ---")
    print(f"Output directory: {out_dir}")

    for loc in locations:
        print(f"\nProcessing: {loc}")
        stats, grids_list = collect_data(base_dir, loc)

        if not stats:
            print(f"  [Warning] No data found for {loc}")
            continue

        frame_files = generate_simple_gif_frames(stats, grids_list, out_dir, loc)

        if args.make_gif and frame_files:
            gif_path = os.path.join(out_dir, f"{loc}_SimpleTimelapse_{RESOLUTION}.gif")
            create_gif(frame_files, gif_path, duration=args.duration)

    print("\n--- Done ---")

if __name__ == "__main__":
    main()
