#!/usr/bin/env python3
"""
DBSCAN Clustering with Significance Filtering & Enhanced Visualization
----------------------------------------------------------------------
Processes M3C2 output, filters for significant changes, splits into
erosion/deposition, runs DBSCAN clustering, and generates detailed reports
with comprehensive visualizations.

Arguments:
  location      Survey location (e.g. Encinitas)
  --eps         DBSCAN epsilon parameter (default: 0.35)
  --min_samples DBSCAN min_samples parameter (default: 30)
  --n_jobs      Number of parallel workers (default: 5)
  --replace     Overwrite existing output files
  --min_change  Minimum absolute change threshold in meters (default: 0.0)

Usage Example:
  python3 run_dbscan_significance.py Encinitas --eps 0.35 --min_samples 30 --replace
"""

import os
import glob
import argparse
import time
import csv
import platform
import laspy
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import DBSCAN

# Matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    plt.switch_backend('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def find_latest_pipeline_run(base_m3c2_dir):
    """Finds the most recently created pipeline_run folder."""
    runs = glob.glob(os.path.join(base_m3c2_dir, "pipeline_run_*"))
    if not runs:
        return None
    runs.sort(key=os.path.getctime, reverse=True)
    return runs[0]


def run_dbscan_file(las_path, erosion_dir, deposition_dir, eps, min_samples, 
                    min_change_threshold=0.0, replace=False):
    """
    Runs DBSCAN on a single M3C2 LAS file:
    1. Filters for significant changes only
    2. Splits into erosion (negative) and deposition (positive)
    3. Clusters each separately
    4. Saves to separate output directories
    """
    start_time = time.time()
    base_name = os.path.basename(las_path)
    base_no_ext = os.path.splitext(base_name)[0]
    
    # Derive the subfolder name from the parent directory
    parent_subfolder = os.path.basename(os.path.dirname(las_path))
    
    # Create date-specific subdirectories
    final_ero_dir = os.path.join(erosion_dir, parent_subfolder)
    final_dep_dir = os.path.join(deposition_dir, parent_subfolder)
    
    os.makedirs(final_ero_dir, exist_ok=True)
    os.makedirs(final_dep_dir, exist_ok=True)
    
    ero_out = os.path.join(final_ero_dir, f"dbscan.las")
    dep_out = os.path.join(final_dep_dir, f"dbscan.las")

    stats = {
        "filename": base_name,
        "subfolder": parent_subfolder,
        "status": "Unknown",
        "total_points": 0,
        "significant_points": 0,
        "erosion_points": 0,
        "deposition_points": 0,
        "erosion_clusters": 0,
        "deposition_clusters": 0,
        "erosion_noise": 0,
        "deposition_noise": 0,
        "processing_time_sec": 0.0,
        "error_message": ""
    }

    try:
        # Check if both outputs exist
        both_exist = os.path.exists(ero_out) and os.path.exists(dep_out)
        if both_exist and not replace:
            print(f"[SKIP] {parent_subfolder}: Both outputs exist.")
            stats["status"] = "Skipped"
            return stats

        print(f"[LOADING] {base_name}...")
        
        # ================================================
        # 1. LOAD DATA & IDENTIFY FIELDS
        # ================================================
        las = laspy.read(las_path)
        stats["total_points"] = len(las.points)
        
        if stats["total_points"] == 0:
            stats["status"] = "Empty Input"
            return stats
        
        # Find M3C2 distance field
        m3c2_field = None
        for field in las.point_format.dimension_names:
            if 'm3c2' in field.lower() and 'distance' in field.lower():
                m3c2_field = field
                break
        
        if m3c2_field is None:
            m3c2_field = next((f for f in las.point_format.dimension_names 
                              if 'm3c2' == f.lower()), None)
        
        if m3c2_field is None:
            raise ValueError(f"No M3C2 distance field found. Available: {las.point_format.dimension_names}")
        
        # Find significant change field
        sig_field = None
        for field in las.point_format.dimension_names:
            field_lower = field.lower()
            if 'significant' in field_lower or 'sig_change' in field_lower:
                sig_field = field
                break
        
        if sig_field is None:
            print(f"[WARN] {base_name}: No 'significant_change' field found!")
            print(f"[WARN] Processing ALL points (no significance filtering)")
            sig_mask = np.ones(len(las.points), dtype=bool)
        else:
            sig_values = getattr(las, sig_field)
            sig_mask = sig_values == 1
        
        # Get M3C2 distances
        m3c2_dist = getattr(las, m3c2_field)
        
        # ================================================
        # 2. FILTER FOR SIGNIFICANT CHANGES
        # ================================================
        filtered_mask = sig_mask.copy()
        
        if min_change_threshold > 0:
            magnitude_mask = np.abs(m3c2_dist) >= min_change_threshold
            filtered_mask = filtered_mask & magnitude_mask
        
        stats["significant_points"] = np.sum(filtered_mask)
        
        if stats["significant_points"] == 0:
            print(f"[WARN] {base_name}: No significant changes found!")
            stats["status"] = "No Significant Changes"
            return stats
        
        # ================================================
        # 3. SPLIT INTO EROSION & DEPOSITION
        # ================================================
        erosion_mask = filtered_mask & (m3c2_dist < 0)
        deposition_mask = filtered_mask & (m3c2_dist > 0)
        
        stats["erosion_points"] = np.sum(erosion_mask)
        stats["deposition_points"] = np.sum(deposition_mask)
        
        # ================================================
        # 4. CLUSTER EROSION
        # ================================================
        if stats["erosion_points"] >= min_samples:
            xyz = np.vstack((las.x, las.y, las.z)).T
            erosion_pts = xyz[erosion_mask]
            
            ero_labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(erosion_pts)
            
            stats["erosion_clusters"] = len(set(ero_labels)) - (1 if -1 in ero_labels else 0)
            stats["erosion_noise"] = int(np.sum(ero_labels == -1))
            
            ero_las = las[erosion_mask]
            ero_las.add_extra_dim(laspy.ExtraBytesParams(name='ClusterID', type=np.int32))
            ero_las.ClusterID = ero_labels.astype(np.int32)
            
            ero_las.write(ero_out)
        else:
            stats["erosion_clusters"] = 0
        
        # ================================================
        # 5. CLUSTER DEPOSITION
        # ================================================
        if stats["deposition_points"] >= min_samples:
            xyz = np.vstack((las.x, las.y, las.z)).T
            deposition_pts = xyz[deposition_mask]
            
            dep_labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(deposition_pts)
            
            stats["deposition_clusters"] = len(set(dep_labels)) - (1 if -1 in dep_labels else 0)
            stats["deposition_noise"] = int(np.sum(dep_labels == -1))
            
            dep_las = las[deposition_mask]
            dep_las.add_extra_dim(laspy.ExtraBytesParams(name='ClusterID', type=np.int32))
            dep_las.ClusterID = dep_labels.astype(np.int32)
            
            dep_las.write(dep_out)
        else:
            stats["deposition_clusters"] = 0
        
        stats["status"] = "Success"
        print(f"[OK] {parent_subfolder}: E={stats['erosion_clusters']}, D={stats['deposition_clusters']}")

    except Exception as e:
        print(f"[ERROR] {base_name}: {e}")
        stats["status"] = "Error"
        stats["error_message"] = str(e)
    
    finally:
        stats["processing_time_sec"] = round(time.time() - start_time, 2)
        return stats


def generate_summary_figures(results, output_dir, location_name, timestamp):
    """Generates comprehensive publication-ready figures."""
    if not HAS_MATPLOTLIB:
        print("[WARN] Matplotlib not available. Skipping figures.")
        return

    print("\n[FIGURES] Generating comprehensive visualizations...")
    
    valid_results = [r for r in results if r["status"] == "Success"]
    if not valid_results:
        print("[WARN] No successful runs to plot.")
        return

    # ========================================
    # DATA EXTRACTION & PARSING
    # ========================================
    dates = []
    times = []
    ero_clusters = []
    dep_clusters = []
    sig_points = []
    total_points = []
    ero_points = []
    dep_points = []
    ero_noise = []
    dep_noise = []
    
    for r in valid_results:
        try:
            # Parse date from filename: 20171004_to_20180131_m3c2.las -> 20171004
            date_str = r['filename'].split('_')[0]
            if len(date_str) == 8 and date_str.isdigit():
                dt = datetime.strptime(date_str, "%Y%m%d")
                dates.append(dt)
                times.append(r['processing_time_sec'])
                ero_clusters.append(r['erosion_clusters'])
                dep_clusters.append(r['deposition_clusters'])
                sig_points.append(r['significant_points'])
                total_points.append(r['total_points'])
                ero_points.append(r['erosion_points'])
                dep_points.append(r['deposition_points'])
                ero_noise.append(r['erosion_noise'])
                dep_noise.append(r['deposition_noise'])
        except (ValueError, IndexError, KeyError):
            pass

    if not dates:
        print("[WARN] Could not parse dates from filenames. Skipping time series.")
        return

    # Sort all data by date
    sorted_data = sorted(zip(dates, times, ero_clusters, dep_clusters, sig_points, 
                             total_points, ero_points, dep_points, ero_noise, dep_noise))
    s_dates, s_times, s_ero_c, s_dep_c, s_sig, s_total, s_ero_p, s_dep_p, s_ero_n, s_dep_n = zip(*sorted_data)
    
    # Convert to numpy for calculations
    s_dates = np.array(s_dates)
    s_times = np.array(s_times)
    s_ero_c = np.array(s_ero_c)
    s_dep_c = np.array(s_dep_c)
    s_sig = np.array(s_sig)
    s_total = np.array(s_total)
    s_ero_p = np.array(s_ero_p)
    s_dep_p = np.array(s_dep_p)
    s_ero_n = np.array(s_ero_n)
    s_dep_n = np.array(s_dep_n)
    
    # Calculate derived metrics
    total_clusters = s_ero_c + s_dep_c
    sig_percentage = (s_sig / s_total) * 100
    ero_noise_pct = np.divide(s_ero_n, s_ero_p, out=np.zeros_like(s_ero_n, dtype=float), where=s_ero_p!=0) * 100
    dep_noise_pct = np.divide(s_dep_n, s_dep_p, out=np.zeros_like(s_dep_n, dtype=float), where=s_dep_p!=0) * 100
    processing_rate = s_total / (s_times + 0.001)  # Points per second

    # ========================================
    # FIGURE 1: COMPREHENSIVE OVERVIEW (2x2 grid)
    # ========================================
    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # --- Panel A: Cluster Counts Over Time ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(s_dates, s_ero_c, 'o-', color='#d95f02', linewidth=2.5, 
             markersize=7, label='Erosion', alpha=0.8)
    ax1.plot(s_dates, s_dep_c, 's-', color='#1b9e77', linewidth=2.5, 
             markersize=7, label='Deposition', alpha=0.8)
    ax1.plot(s_dates, total_clusters, '^--', color='#7570b3', linewidth=1.5, 
             markersize=6, label='Total', alpha=0.6)
    ax1.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax1.set_title('A. Detected Clusters Over Time', fontsize=13, fontweight='bold', loc='left')
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    
    # --- Panel B: Point Counts (Stacked Area) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(s_dates, 0, s_ero_p, color='#d95f02', alpha=0.6, label='Erosion Points')
    ax2.fill_between(s_dates, s_ero_p, s_ero_p + s_dep_p, color='#1b9e77', 
                     alpha=0.6, label='Deposition Points')
    ax2.set_ylabel('Number of Points', fontsize=12, fontweight='bold')
    ax2.set_title('B. Significant Change Point Distribution', fontsize=13, fontweight='bold', loc='left')
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # --- Panel C: Data Coverage (Significance Percentage) ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(s_dates, sig_percentage, 'D-', color='#e7298a', linewidth=2.5, 
             markersize=7, alpha=0.8)
    ax3.fill_between(s_dates, sig_percentage, alpha=0.2, color='#e7298a')
    ax3.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Survey Date', fontsize=12, fontweight='bold')
    ax3.set_title('C. Significant Change Detection Rate', fontsize=13, fontweight='bold', loc='left')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.axhline(y=np.mean(sig_percentage), color='red', linestyle='--', 
                linewidth=1.5, alpha=0.7, label=f'Mean: {np.mean(sig_percentage):.1f}%')
    ax3.legend(fontsize=10, framealpha=0.9)
    
    # --- Panel D: Processing Performance ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4_twin = ax4.twinx()
    
    ln1 = ax4.bar(s_dates, s_times, width=20, color='#2c7bb6', alpha=0.6, label='Processing Time (s)')
    ln2 = ax4_twin.plot(s_dates, processing_rate / 1000, 'o-', color='#e66101', 
                        linewidth=2, markersize=6, label='Throughput (K pts/sec)')
    
    ax4.set_ylabel('Processing Time (seconds)', fontsize=12, fontweight='bold', color='#2c7bb6')
    ax4_twin.set_ylabel('Processing Rate (1000 pts/sec)', fontsize=12, fontweight='bold', color='#e66101')
    ax4.set_xlabel('Survey Date', fontsize=12, fontweight='bold')
    ax4.set_title('D. Computational Performance', fontsize=13, fontweight='bold', loc='left')
    ax4.tick_params(axis='y', labelcolor='#2c7bb6')
    ax4_twin.tick_params(axis='y', labelcolor='#e66101')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Combined legend
    lns = [ln1] + ln2
    labs = [l.get_label() if hasattr(l, 'get_label') else 'Processing Time (s)' for l in lns]
    ax4.legend(lns, labs, fontsize=10, framealpha=0.9, loc='upper left')
    
    plt.suptitle(f'{location_name} - DBSCAN Analysis Overview', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    fig_path = os.path.join(output_dir, f"01_Overview_{location_name}_{timestamp}.png")
    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓ Saved: 01_Overview")

    # ========================================
    # FIGURE 2: CLUSTERING QUALITY METRICS
    # ========================================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    
    # --- Panel A: Noise Percentage ---
    ax1.plot(s_dates, ero_noise_pct, 'o-', color='#d95f02', linewidth=2, 
             markersize=7, label='Erosion Noise %', alpha=0.8)
    ax1.plot(s_dates, dep_noise_pct, 's-', color='#1b9e77', linewidth=2, 
             markersize=7, label='Deposition Noise %', alpha=0.8)
    ax1.axhline(y=20, color='red', linestyle='--', linewidth=1, alpha=0.5, label='20% Threshold')
    ax1.set_ylabel('Noise Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('A. DBSCAN Noise Ratio', fontsize=13, fontweight='bold', loc='left')
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    
    # --- Panel B: Erosion/Deposition Balance ---
    ratio = np.divide(s_ero_c, s_dep_c + 0.001, out=np.zeros_like(s_ero_c, dtype=float), where=s_dep_c!=0)
    ax2.plot(s_dates, ratio, 'D-', color='#e7298a', linewidth=2.5, markersize=7, alpha=0.8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='1:1 Balance')
    ax2.fill_between(s_dates, 1.0, ratio, where=(ratio >= 1.0), 
                     color='#d95f02', alpha=0.2, interpolate=True, label='Erosion Dominant')
    ax2.fill_between(s_dates, ratio, 1.0, where=(ratio < 1.0), 
                     color='#1b9e77', alpha=0.2, interpolate=True, label='Deposition Dominant')
    ax2.set_ylabel('Erosion / Deposition Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('B. Cluster Type Balance', fontsize=13, fontweight='bold', loc='left')
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # --- Panel C: Average Points per Cluster ---
    avg_pts_per_ero_cluster = np.divide(s_ero_p, s_ero_c, out=np.zeros_like(s_ero_p, dtype=float), where=s_ero_c!=0)
    avg_pts_per_dep_cluster = np.divide(s_dep_p, s_dep_c, out=np.zeros_like(s_dep_p, dtype=float), where=s_dep_c!=0)
    
    ax3.plot(s_dates, avg_pts_per_ero_cluster, 'o-', color='#d95f02', 
             linewidth=2, markersize=7, label='Erosion', alpha=0.8)
    ax3.plot(s_dates, avg_pts_per_dep_cluster, 's-', color='#1b9e77', 
             linewidth=2, markersize=7, label='Deposition', alpha=0.8)
    ax3.set_ylabel('Average Points per Cluster', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Survey Date', fontsize=12, fontweight='bold')
    ax3.set_title('C. Cluster Size Trends', fontsize=13, fontweight='bold', loc='left')
    ax3.legend(fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # --- Panel D: Processing Efficiency Scatter ---
    ax4.scatter(s_total / 1e6, s_times, c=total_clusters, cmap='viridis', 
                s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Fit linear trend
    if len(s_total) > 1:
        z = np.polyfit(s_total / 1e6, s_times, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(min(s_total / 1e6), max(s_total / 1e6), 100)
        ax4.plot(x_fit, p(x_fit), "r--", linewidth=2, alpha=0.7, 
                label=f'Trend: {z[0]:.1f}s per M pts')
    
    ax4.set_xlabel('Point Cloud Size (Million Points)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Processing Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Computational Scaling', fontsize=13, fontweight='bold', loc='left')
    ax4.legend(fontsize=10, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Total Clusters', fontsize=11, fontweight='bold')
    
    plt.suptitle(f'{location_name} - Clustering Quality & Performance Metrics', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, f"02_Quality_Metrics_{location_name}_{timestamp}.png")
    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓ Saved: 02_Quality_Metrics")

    # ========================================
    # FIGURE 3: CUMULATIVE STATISTICS
    # ========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    
    # --- Panel A: Cumulative Clusters ---
    cum_ero = np.cumsum(s_ero_c)
    cum_dep = np.cumsum(s_dep_c)
    cum_total = cum_ero + cum_dep
    
    ax1.fill_between(s_dates, 0, cum_ero, color='#d95f02', alpha=0.6, label='Erosion')
    ax1.fill_between(s_dates, cum_ero, cum_total, color='#1b9e77', alpha=0.6, label='Deposition')
    ax1.plot(s_dates, cum_total, 'k-', linewidth=3, alpha=0.8, label='Total')
    ax1.set_ylabel('Cumulative Cluster Count', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Survey Date', fontsize=13, fontweight='bold')
    ax1.set_title('A. Cumulative Clusters Detected', fontsize=14, fontweight='bold', loc='left')
    ax1.legend(fontsize=11, framealpha=0.9, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    
    # Add annotations for totals
    ax1.text(0.98, 0.02, f'Total: {int(cum_total[-1])} clusters\nErosion: {int(cum_ero[-1])}\nDeposition: {int(cum_dep[-1])}',
             transform=ax1.transAxes, fontsize=11, verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # --- Panel B: Cumulative Points Analyzed ---
    cum_sig_pts = np.cumsum(s_sig)
    cum_total_pts = np.cumsum(s_total)
    
    ax2.plot(s_dates, cum_total_pts / 1e6, 'o-', color='#7570b3', 
             linewidth=2.5, markersize=7, label='Total Points', alpha=0.8)
    ax2.plot(s_dates, cum_sig_pts / 1e6, 's-', color='#e7298a', 
             linewidth=2.5, markersize=7, label='Significant Points', alpha=0.8)
    ax2.fill_between(s_dates, cum_sig_pts / 1e6, alpha=0.2, color='#e7298a')
    ax2.set_ylabel('Cumulative Points (Millions)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Survey Date', fontsize=13, fontweight='bold')
    ax2.set_title('B. Cumulative Point Cloud Analysis', fontsize=14, fontweight='bold', loc='left')
    ax2.legend(fontsize=11, framealpha=0.9, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Add annotations
    total_sig_pct = (cum_sig_pts[-1] / cum_total_pts[-1]) * 100
    ax2.text(0.98, 0.02, f'Analyzed: {cum_total_pts[-1]/1e6:.1f}M pts\nSignificant: {cum_sig_pts[-1]/1e6:.1f}M ({total_sig_pct:.1f}%)',
             transform=ax2.transAxes, fontsize=11, verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'{location_name} - Cumulative Analysis Statistics', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, f"03_Cumulative_Stats_{location_name}_{timestamp}.png")
    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓ Saved: 03_Cumulative_Stats")

    # ========================================
    # SUMMARY STATISTICS TABLE (as text file)
    # ========================================
    summary_path = os.path.join(output_dir, f"Summary_Statistics_{location_name}_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"{'='*70}\n")
        f.write(f"DBSCAN CLUSTERING SUMMARY - {location_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*70}\n\n")
        
        f.write(f"SURVEY STATISTICS:\n")
        f.write(f"  Total Surveys Processed: {len(s_dates)}\n")
        f.write(f"  Date Range: {s_dates[0].strftime('%Y-%m-%d')} to {s_dates[-1].strftime('%Y-%m-%d')}\n\n")
        
        f.write(f"POINT CLOUD STATISTICS:\n")
        f.write(f"  Total Points Analyzed: {sum(s_total):,.0f}\n")
        f.write(f"  Significant Points: {sum(s_sig):,.0f} ({(sum(s_sig)/sum(s_total)*100):.2f}%)\n")
        f.write(f"  Erosion Points: {sum(s_ero_p):,.0f} ({(sum(s_ero_p)/sum(s_sig)*100):.2f}% of significant)\n")
        f.write(f"  Deposition Points: {sum(s_dep_p):,.0f} ({(sum(s_dep_p)/sum(s_sig)*100):.2f}% of significant)\n\n")
        
        f.write(f"CLUSTERING RESULTS:\n")
        f.write(f"  Total Clusters: {sum(total_clusters):,.0f}\n")
        f.write(f"  Erosion Clusters: {sum(s_ero_c):,.0f}\n")
        f.write(f"  Deposition Clusters: {sum(s_dep_c):,.0f}\n")
        f.write(f"  E/D Ratio: {sum(s_ero_c)/(sum(s_dep_c)+0.001):.2f}\n\n")
        
        f.write(f"QUALITY METRICS:\n")
        f.write(f"  Mean Erosion Noise %: {np.mean(ero_noise_pct):.2f}%\n")
        f.write(f"  Mean Deposition Noise %: {np.mean(dep_noise_pct):.2f}%\n")
        f.write(f"  Avg Points per Erosion Cluster: {sum(s_ero_p)/(sum(s_ero_c)+0.001):.1f}\n")
        f.write(f"  Avg Points per Deposition Cluster: {sum(s_dep_p)/(sum(s_dep_c)+0.001):.1f}\n\n")
        
        f.write(f"PERFORMANCE METRICS:\n")
        f.write(f"  Total Processing Time: {sum(s_times):.2f} seconds ({sum(s_times)/60:.2f} minutes)\n")
        f.write(f"  Mean Processing Time per Survey: {np.mean(s_times):.2f} ± {np.std(s_times):.2f} seconds\n")
        f.write(f"  Mean Processing Rate: {np.mean(processing_rate):,.0f} points/second\n")
        f.write(f"  Total Throughput: {sum(s_total)/sum(s_times):,.0f} points/second\n")
        
    print(f"  ✓ Saved: Summary_Statistics.txt")
    print(f"\n[FIGURES] Successfully generated 3 multi-panel figures + summary stats")


def main():
    script_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Run DBSCAN on M3C2 Results")
    parser.add_argument("location", help="Survey location (e.g. Encinitas)")
    parser.add_argument("--eps", type=float, default=0.35, help="DBSCAN eps (default: 0.35)")
    parser.add_argument("--min_samples", type=int, default=30, help="DBSCAN min_samples (default: 30)")
    parser.add_argument("--min_change", type=float, default=0.25, help="Min absolute change threshold")
    parser.add_argument("--n_jobs", type=int, default=5, help="Parallel workers")
    parser.add_argument("--replace", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    
    # [INSERT THIS LINE] Override threshold to 30cm if location is Torrey
    if args.location == "Torrey": args.min_change = 0.3
    else: args.min_change = 0.25

    # Base Paths
    system = platform.system()
    base_dir = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs" if system == "Darwin" else "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    
    loc = args.location
    results_dir = os.path.join(base_dir, "results", loc)
    m3c2_base_dir = os.path.join(results_dir, "m3c2")
    
    # Smart Input Finding
    print(f"[MAIN] Scanning for pipeline runs in: {m3c2_base_dir}")
    input_dir = find_latest_pipeline_run(m3c2_base_dir)
    
    if not input_dir:
        print(f"[ERROR] No 'pipeline_run_*' folders found in {m3c2_base_dir}")
        return
        
    print(f"[MAIN] Target Input Directory: {input_dir}")
    
    # Output Directories
    erosion_dir = os.path.join(results_dir, "erosion")
    deposition_dir = os.path.join(results_dir, "deposition")
    report_dir = os.path.join(base_dir, "utilities", "dbscan")
    
    os.makedirs(erosion_dir, exist_ok=True)
    os.makedirs(deposition_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Recursive File Search
    search_pattern = os.path.join(input_dir, "**", "*.las")
    las_files = sorted(glob.glob(search_pattern, recursive=True))
    las_files = [f for f in las_files if "m3c2" in os.path.basename(f).lower() and "_clustered" not in f]
    
    if not las_files:
        print(f"[ERROR] No suitable .las files found in {input_dir}")
        return

    print(f"[MAIN] Found {len(las_files)} files. Starting processing...")
    print(f"[CONFIG] EPS: {args.eps}, Min Samples: {args.min_samples}")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=args.n_jobs) as exe:
        futures = {
            exe.submit(
                run_dbscan_file, f, erosion_dir, deposition_dir, 
                args.eps, args.min_samples, args.min_change, args.replace
            ): f for f in las_files
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    # Reporting
    total_duration = time.time() - script_start_time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(report_dir, f"dbscan_report_{loc}_{timestamp}.csv")
    
    print(f"\n[MAIN] All tasks completed in {total_duration:.2f} seconds.")
    
    # Save CSV Report
    fieldnames = [
        "filename", "subfolder", "status", "total_points", "significant_points",
        "erosion_points", "deposition_points", 
        "erosion_clusters", "deposition_clusters",
        "erosion_noise", "deposition_noise",
        "processing_time_sec", "error_message"
    ]
    
    with open(report_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Generate Figures
    generate_summary_figures(results, report_dir, loc, timestamp)
    print(f"\n[COMPLETE] Report saved to: {report_file}")


if __name__ == "__main__":
    main()