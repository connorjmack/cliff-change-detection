#!/usr/bin/env python3
"""
10_plot_uncertainty.py

Generates a 2-panel diagnostic plot for uncertainty analysis:
1. Left Panel: Distribution of raw grid cell uncertainty values.
2. Right Panel: Distribution of aggregated Volume Uncertainty per Cluster.

Updates:
- SIGNIFICANTLY increased font sizes for all elements.
- Retains systematic error calculation (Linear Sum).

Usage:
    python3 10_plot_uncertainty.py --location DelMar
"""

import os
import glob
import argparse
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import sum as ndi_sum
from tqdm import tqdm

# === CONFIGURATION ===
RESOLUTION = "25cm"
CELL_SIZE = 0.25
CELL_AREA = CELL_SIZE * CELL_SIZE  # 0.0625 m^2

# Visual Settings
PLOT_COLOR = "#4c72b0"  # Seaborn Deep Blue
FONT_SCALE = 2.0        # Increased Base Scale

def get_base_path():
    """Determine system root path."""
    if platform.system() == "Darwin":
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

def load_csv_matrix(filepath):
    """Loads a CSV grid into a numpy array, ignoring headers/indices."""
    try:
        # Read data, skipping the index col (0)
        df = pd.read_csv(filepath, index_col=0)
        return df.values
    except Exception as e:
        print(f"[WARN] Failed to load {filepath}: {e}")
        return None

def collect_uncertainty_stats(location, base_dir):
    """
    Walks through all erosion folders for the location, collecting:
    1. All raw cell uncertainty values.
    2. All aggregated cluster volume uncertainties (Systematic Summation).
    """
    erosion_dir = os.path.join(base_dir, "results", location, "erosion")
    
    if not os.path.exists(erosion_dir):
        raise FileNotFoundError(f"Directory not found: {erosion_dir}")

    # Find all relevant subdirectories
    survey_dirs = sorted([
        d for d in glob.glob(os.path.join(erosion_dir, "*")) 
        if os.path.isdir(d) and "to" in os.path.basename(d)
    ])

    print(f"[INFO] Found {len(survey_dirs)} surveys for {location}...")

    all_cell_uncertainties = []
    all_cluster_uncertainties = []

    # Iterate through surveys
    for survey_path in tqdm(survey_dirs, desc="Processing Surveys"):
        
        # Define expected filenames
        unc_pattern = os.path.join(survey_path, f"*_uncertainty_{RESOLUTION}.csv")
        cls_pattern = os.path.join(survey_path, f"*_clusters_{RESOLUTION}.csv")

        unc_files = glob.glob(unc_pattern)
        cls_files = glob.glob(cls_pattern)

        if not unc_files or not cls_files:
            continue

        # Load Data
        unc_grid = load_csv_matrix(unc_files[0])
        cls_grid = load_csv_matrix(cls_files[0])

        if unc_grid is None or cls_grid is None:
            continue

        if unc_grid.shape != cls_grid.shape:
            continue

        # --- METRIC 1: Raw Cell Uncertainty ---
        # Flatten and keep only valid data (>0 and finite)
        valid_cells = unc_grid[np.isfinite(unc_grid) & (unc_grid > 1e-6)]
        if valid_cells.size > 0:
            all_cell_uncertainties.append(valid_cells)

        # --- METRIC 2: Per-Cluster Volume Uncertainty ---
        # Systematic Error (Linear Sum)
        
        flat_u = unc_grid.flatten()
        flat_c = cls_grid.flatten()

        mask = (flat_c > 0) & np.isfinite(flat_u)
        
        if np.any(mask):
            u_subset = flat_u[mask]
            c_subset = flat_c[mask].astype(int)

            unique_ids = np.unique(c_subset)
            
            # Sum of Uncertainty per cluster (Systematic Assumption)
            sum_unc = ndi_sum(u_subset, labels=c_subset, index=unique_ids)
            
            # Vol_Unc = Area * Sum(Cell_Uncertainties)
            vol_unc = CELL_AREA * sum_unc
            
            all_cluster_uncertainties.append(vol_unc)

    # Concatenate results
    total_cells = np.concatenate(all_cell_uncertainties) if all_cell_uncertainties else np.array([])
    total_clusters = np.concatenate(all_cluster_uncertainties) if all_cluster_uncertainties else np.array([])

    return total_cells, total_clusters

def plot_dual_distribution(data, ax, xlabel, title):
    """Helper to plot Histogram and CDF with 95th percentile cutoff."""
    if len(data) == 0:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=24)
        return

    # Calculate stats for limits and textbox
    data_min = np.min(data)
    cutoff_95 = np.percentile(data, 95)
    
    # Histogram (Density) - Left Axis
    sns.histplot(
        data, ax=ax, stat="density", kde=False, 
        color=PLOT_COLOR, alpha=0.6, linewidth=0, label="Histogram"
    )
    
    # Set X-axis limits from Min to 95th percentile
    ax.set_xlim(data_min, cutoff_95)
    
    # Styling Left Axis (LARGE FONTS)
    ax.set_ylabel("Probability Density", color=PLOT_COLOR, fontweight='bold', fontsize=20)
    ax.tick_params(axis='y', labelcolor=PLOT_COLOR, labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xlabel(xlabel, fontsize=20, fontweight='bold')
    ax.set_title(title, fontsize=24, fontweight='bold', pad=20)
    ax.grid(False)

    # CDF - Right Axis
    ax2 = ax.twinx()
    sns.ecdfplot(
        data, ax=ax2, color='black', linewidth=4, 
        linestyle='--', alpha=0.8, label="CDF"
    )
    
    # Styling Right Axis (LARGE FONTS)
    ax2.set_ylabel("Cumulative Probability", color='black', fontweight='bold', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=16)
    ax2.set_ylim(0, 1.05)
    ax2.grid(False)

    # Stats Box (LARGE FONTS)
    stats_text = (
        f"N: {len(data):,}\n"
        f"Mean: {np.mean(data):.3f}\n"
        f"Median: {np.median(data):.3f}\n"
        f"95th: {cutoff_95:.3f}"
    )
    ax.text(0.95, 0.5, stats_text, transform=ax.transAxes, 
            verticalalignment='center', horizontalalignment='right',
            fontsize=22, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

def main():
    parser = argparse.ArgumentParser(description="Plot Uncertainty Distributions")
    parser.add_argument("--location", required=True, help="Survey location (e.g., DelMar)")
    args = parser.parse_args()

    # 1. Setup Paths
    base_dir = get_base_path()
    output_dir = os.path.join(base_dir, "figures", "uncertainty")
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Uncertainty Analysis: {args.location} ---")
    
    # 2. Collect Data
    cells, clusters = collect_uncertainty_stats(args.location, base_dir)

    print(f"[STATS] Grid Cells: {len(cells):,}")
    print(f"[STATS] Clusters:   {len(clusters):,}")

    if len(cells) == 0 and len(clusters) == 0:
        print("[ERROR] No valid data found. Exiting.")
        return

    # 3. Setup Plotting Context (Seaborn)
    sns.set_context("talk", font_scale=FONT_SCALE)
    sns.set_style("white") 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 11))

    # 4. Left Panel: Grid Cell Uncertainty
    plot_dual_distribution(
        cells, ax1,
        xlabel="Uncertainty (m)",
        title=f"Grid Cell Uncertainty ({RESOLUTION})"
    )

    # 5. Right Panel: Cluster Volume Uncertainty
    plot_dual_distribution(
        clusters, ax2,
        xlabel="Volume Uncertainty (m³)",
        title=f"Aggregated Volume Uncertainty per Cluster"
    )

    # 6. Finalize (HUGE TITLE)
    plt.suptitle(f"Uncertainty Characterization: {args.location}", fontsize=30, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(output_dir, f"{args.location}_Uncertainty_Dist_{timestamp}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Plot saved to: {out_path}")

if __name__ == "__main__":
    main()