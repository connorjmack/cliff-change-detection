#!/usr/bin/env python3
"""
compare_filled_vs_cleaned.py

Creates a two-panel cumulative erosion comparison (filled vs. original grids)
using the same panel E styling from plot_dashboard.py. Designed for quick
visual QA of the fill step.

Usage:
    python3 compare_filled_vs_cleaned.py SanElijo --resolution 25cm
"""

import os
import re
import platform
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from datetime import datetime

DPI = 300
DEFAULT_RESOLUTION = '25cm'
DEFAULT_VMAX = 6.0


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def get_base_dir():
    if platform.system() == 'Darwin':
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"


def get_output_dir(base_dir):
    return os.path.join(base_dir, "figures", "fill_vs_original")


def normalize_resolution_for_files(resolution):
    if resolution == '1m':
        return '100cm'
    if resolution == '100cm':
        return '100cm'
    return resolution


def get_resolution_value(res_str):
    if 'cm' in res_str:
        return float(res_str.replace('cm', '')) / 100.0
    if 'm' in res_str:
        return float(res_str.replace('m', ''))
    return 0.25


def parse_interval_end(folder_name):
    match = re.search(r'(\d{8})_to_(\d{8})', folder_name)
    if match:
        try:
            return datetime.strptime(match.group(2), '%Y%m%d')
        except ValueError:
            return None
    return None


def clean_and_snap_grid(df, resolution_val):
    cleaned_cols = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)
    try:
        col_floats = cleaned_cols.astype(float)
    except ValueError:
        return None

    scale = 1.0 / resolution_val
    new_cols = (col_floats * scale).round().astype(int)
    df.columns = new_cols
    df.index = df.index.astype(int)
    return df


def load_grid_dataframe(filepath, res_val):
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, index_col=0)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = clean_and_snap_grid(df, res_val)
        if df is not None:
            return df.fillna(0.0)
    except Exception:
        return None
    return None


def grid_patterns(tag, use_filled):
    if use_filled:
        return [
            f"_ero_grid_{tag}_filled.csv",
            f"grid_{tag}_filled.csv",
            f"_ero_grid_{tag}_cleaned.csv",
            f"grid_{tag}_cleaned.csv",
        ]
    return [
        f"_ero_grid_{tag}_cleaned.csv",
        f"grid_{tag}_cleaned.csv",
        f"_ero_grid_{tag}.csv",
        f"grid_{tag}.csv",
    ]


def resolve_grid_file(interval_dir, interval, resolution, use_filled):
    tags = [resolution]
    normalized = normalize_resolution_for_files(resolution)
    if normalized not in tags:
        tags.append(normalized)
    if '1m' not in tags:
        tags.append('1m')

    candidate_dirs = [
        os.path.join(interval_dir, resolution),
        os.path.join(interval_dir, normalized),
        interval_dir,
    ]

    for base in candidate_dirs:
        if not os.path.isdir(base):
            continue
        files_here = os.listdir(base)
        for tag in tags:
            patterns = grid_patterns(tag, use_filled)
            matches = [f for f in files_here if f.endswith('.csv') and any(p in f for p in patterns)]
            if matches:
                return os.path.join(base, matches[0])
    return None


def build_cumulative_grid(base_dir, location, resolution, res_val, use_filled):
    erosion_dir = os.path.join(base_dir, 'results', location, 'erosion')
    if not os.path.isdir(erosion_dir):
        return None, []

    intervals = []
    for folder in os.listdir(erosion_dir):
        folder_path = os.path.join(erosion_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        end_date = parse_interval_end(folder)
        intervals.append((end_date, folder))
    intervals.sort(key=lambda x: (x[0] or datetime.max, x[1]))

    cumulative = None
    used_intervals = []

    for _, interval in intervals:
        interval_dir = os.path.join(erosion_dir, interval)
        grid_file = resolve_grid_file(interval_dir, interval, resolution, use_filled)
        if not grid_file:
            continue

        df_grid = load_grid_dataframe(grid_file, res_val)
        if df_grid is None:
            continue

        cumulative = df_grid if cumulative is None else cumulative.add(df_grid, fill_value=0)
        used_intervals.append(interval)

    return cumulative, used_intervals


def get_custom_cmap(name, vmax=6.0):
    base_cmap = cm.get_cmap(name, 256)
    newcolors = base_cmap(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 1])
    return LinearSegmentedColormap.from_list(f"White_{name}", newcolors), Normalize(vmin=0, vmax=vmax)


def plot_cumulative_panel(ax, grid_df, res_val, cmap, norm, title):
    plot_df = grid_df.T
    x_indices = plot_df.columns.astype(int)
    x_meters = x_indices * res_val
    max_elev_m = len(plot_df.index) * res_val

    extent = [x_meters.min(), x_meters.max(), 0, max_elev_m]
    im = ax.imshow(
        plot_df.values,
        origin='lower',
        extent=extent,
        cmap=cmap,
        norm=norm,
        aspect='auto',
        interpolation='none',
    )

    ax.set_xlim(x_meters.max(), x_meters.min())
    ax.set_ylim(0, max_elev_m)
    ax.set_xlabel("Alongshore Location (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(title, loc='left', fontsize=13, fontweight='bold')
    return im


# ------------------------------------------------------------------------------
# Main plotting routine
# ------------------------------------------------------------------------------

def plot_fill_vs_original(base_dir, location, resolution, vmax, output_path):
    res_val = get_resolution_value(resolution)
    cmap_grid, norm_grid = get_custom_cmap('magma_r', vmax=vmax)

    filled_grid, filled_used = build_cumulative_grid(base_dir, location, resolution, res_val, True)
    original_grid, original_used = build_cumulative_grid(base_dir, location, resolution, res_val, False)

    if filled_grid is None and original_grid is None:
        print(f"[Warning] No erosion grids found for {location} at {resolution}. Nothing to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    axes = axes.ravel()

    ims = []
    if filled_grid is not None:
        ims.append(
            plot_cumulative_panel(
                axes[0],
                filled_grid,
                res_val,
                cmap_grid,
                norm_grid,
                "Filled Grids",
            )
        )
        axes[0].text(
            0.01,
            0.96,
            f"{len(filled_used)} surveys",
            transform=axes[0].transAxes,
            ha='left',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
        )
    else:
        axes[0].text(0.5, 0.5, "No filled grids found", ha='center', va='center', fontsize=12)
        axes[0].set_axis_off()

    if original_grid is not None:
        ims.append(
            plot_cumulative_panel(
                axes[1],
                original_grid,
                res_val,
                cmap_grid,
                norm_grid,
                "Original / Not Filled",
            )
        )
        axes[1].text(
            0.01,
            0.96,
            f"{len(original_used)} surveys",
            transform=axes[1].transAxes,
            ha='left',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
        )
    else:
        axes[1].text(0.5, 0.5, "No original grids found", ha='center', va='center', fontsize=12)
        axes[1].set_axis_off()

    # Shared colorbar if at least one panel plotted
    if ims:
        cbar = fig.colorbar(
            ims[-1],
            ax=axes,
            orientation='horizontal',
            fraction=0.05,
            pad=0.12,
        )
        cbar.set_label("Cumulative Erosion Depth (m)", fontsize=12)

    fig.suptitle(f"{location} | Cumulative Erosion Comparison ({resolution})", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare filled vs. original cumulative erosion grids (panel E style).")
    parser.add_argument("location", type=str, help="Location name (e.g., SanElijo)")
    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION, help="Grid resolution tag (e.g., 25cm, 10cm, 1m)")
    parser.add_argument("--vmax", type=float, default=DEFAULT_VMAX, help="Upper bound for color scale (meters)")
    parser.add_argument("--output", type=str, help="Optional output filepath. Defaults to figures/fill_vs_original/<location>_fill_compare_<resolution>.png")
    parser.add_argument("--base-dir", type=str, help="Override default base directory detection")
    args = parser.parse_args()

    base_dir = args.base_dir if args.base_dir else get_base_dir()
    out_dir = get_output_dir(base_dir)
    default_name = f"{args.location}_fill_compare_{args.resolution}.png"
    output_path = args.output if args.output else os.path.join(out_dir, default_name)

    print(f"Base dir: {base_dir}")
    print(f"Location: {args.location}")
    print(f"Resolution: {args.resolution}")
    print(f"Output: {output_path}")

    plot_fill_vs_original(base_dir, args.location, args.resolution, args.vmax, output_path)


if __name__ == "__main__":
    main()
