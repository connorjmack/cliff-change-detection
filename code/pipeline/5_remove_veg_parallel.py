#!/usr/bin/env python3
"""
Vegetation Removal with Detailed Reporting
------------------------------------------
Wraps CloudCompare (CANUPO) to classify and remove vegetation, while generating
comprehensive CSV reports and performance visualizations.

Arguments:
  location      Survey location (e.g. SanElijo)
  --cc          (optional) Full path to CloudCompare executable
  --test        Number of surveys to process in test mode (default: all)
  --replace     If set, overwrite existing output files; otherwise skip them
  --single      If set, run one file at a time (no parallel processing)

Usage Example:
  python3 remove_veg_reporting.py DelMar --replace --n_jobs 4
"""

import os
import glob
import argparse
import subprocess
import multiprocessing
import random
import platform
import time
import csv
import laspy
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Try importing matplotlib for plotting; handle gracefully if missing
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    # 'Agg' backend allows plotting without a display (headless server safe)
    plt.switch_backend('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def get_point_count(las_path):
    """
    Efficiently reads the point count from the LAS header without loading data.
    """
    try:
        with laspy.open(las_path) as f:
            return f.header.point_count
    except Exception:
        return 0

def classify_file_with_stats(las_path, classifier_prm, output_dir, shift, cc_path, replace=False):
    """
    Runs CloudCompare to classify/remove vegetation and returns execution stats.
    """
    start_time = time.time()
    base_name = os.path.basename(las_path)
    base_no_ext = os.path.splitext(base_name)[0]
    out_name = f"{base_no_ext}_noveg.las"
    out_path = os.path.join(output_dir, out_name)

    stats = {
        "filename": base_name,
        "status": "Unknown",
        "input_points": 0,
        "output_points": 0,
        "removed_points": 0,
        "percent_removed": 0.0,
        "processing_time_sec": 0.0,
        "error_message": ""
    }

    try:
        # 1. Check if output exists
        if os.path.exists(out_path) and not replace:
            print(f"[SKIP] {base_name}: Output exists.")
            stats["status"] = "Skipped"
            return stats

        # 2. Get Input Stats
        input_count = get_point_count(las_path)
        stats["input_points"] = input_count

        print(f"[CLASSIFY] {base_name}: Running CloudCompare...")
        
        # 3. Construct CloudCompare Command
        # Note: -SILENT mode prevents GUI popups
        cmd = [
            cc_path,
            "-SILENT",
            "-AUTO_SAVE", "OFF",
            "-C_EXPORT_FMT", "LAS",
            "-O", "-GLOBAL_SHIFT", *shift, las_path,
            "-CANUPO_CLASSIFY", "-USE_CONFIDENCE", "0.8", classifier_prm,
            "-FILTER_SF", "MIN", "1.1",  # Keep points with class > 1.1 (Assuming Veg=1, Non-Veg=2)
            "-SAVE_CLOUDS", "FILE", out_path
        ]
        
        # 4. Execute
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        # 5. Verify Output and Get Result Stats
        if os.path.exists(out_path):
            output_count = get_point_count(out_path)
            removed_count = input_count - output_count
            
            stats["output_points"] = output_count
            stats["removed_points"] = removed_count
            stats["percent_removed"] = round((removed_count / input_count) * 100, 2) if input_count > 0 else 0.0
            stats["status"] = "Success"
            print(f"[OK] {base_name} -> Removed {removed_count} pts ({stats['percent_removed']}%)")
        else:
            raise FileNotFoundError(f"CloudCompare finished but {out_path} was not created.")

    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {base_name}: CloudCompare Error")
        stats["status"] = "CC Error"
        stats["error_message"] = f"Return Code {e.returncode}"
    except Exception as e:
        print(f"[ERROR] {base_name}: {e}")
        stats["status"] = "Error"
        stats["error_message"] = str(e)
    finally:
        stats["processing_time_sec"] = round(time.time() - start_time, 2)
        return stats

def generate_summary_figures(results, output_dir, location_name, timestamp):
    """Generates publication-ready PNG plots from the results."""
    if not HAS_MATPLOTLIB:
        print("[WARN] Matplotlib not found. Skipping figure generation.")
        return

    print("[MAIN] Generating summary figures...")
    
    # Filter for SUCCESS runs only
    valid_results = [r for r in results if r["status"] == "Success"]
    
    if not valid_results:
        print("[WARN] No successful processing runs to plot.")
        return

    # Extract Data
    dates = []
    percents = []
    times = []
    points = []
    
    for r in valid_results:
        try:
            # Assumes YYYYMMDD_... format
            date_str = r['filename'].split('_')[0]
            dt = datetime.strptime(date_str, "%Y%m%d")
            dates.append(dt)
            percents.append(r['percent_removed'])
            times.append(r['processing_time_sec'])
            points.append(r['input_points'])
        except ValueError:
            pass

    if not dates:
        print("[WARN] Could not parse dates from filenames. Skipping Time Series plots.")
        return

    # --- FIGURE 1: Time Series of Removal Percentage ---
    plt.figure(figsize=(12, 6), dpi=300)
    sorted_pairs = sorted(zip(dates, percents))
    s_dates, s_percents = zip(*sorted_pairs)
    
    plt.plot(s_dates, s_percents, marker='o', linestyle='-', color='#2c7bb6', alpha=0.7)
    plt.scatter(s_dates, s_percents, c=s_percents, cmap='viridis', zorder=10, s=50, edgecolors='k')
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    
    plt.title(f"Vegetation Removal % Over Time - {location_name}", fontsize=14, fontweight='bold')
    plt.ylabel("Points Removed (%)", fontsize=12)
    plt.xlabel("Survey Date", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_path = os.path.join(output_dir, f"Veg_Removal_TimeSeries_{location_name}_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    # --- FIGURE 2: Time Series of Processing Time ---
    plt.figure(figsize=(12, 6), dpi=300)
    sorted_pairs = sorted(zip(dates, times))
    s_dates, s_times = zip(*sorted_pairs)
    
    plt.plot(s_dates, s_times, marker='s', linestyle='-', color='#d95f02', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    
    plt.title(f"Computation Time per Survey - {location_name}", fontsize=14, fontweight='bold')
    plt.ylabel("Processing Time (seconds)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    out_path = os.path.join(output_dir, f"Veg_Computation_Time_{location_name}_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    # --- FIGURE 3: Efficiency Scatter (Points vs Time) ---
    plt.figure(figsize=(10, 6), dpi=300)
    points_m = [p / 1_000_000 for p in points]
    
    plt.scatter(points_m, times, alpha=0.7, c='#7570b3', edgecolors='k', s=60)
    plt.title(f"Processing Efficiency (Size vs Time) - {location_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Input Cloud Size (Millions of Points)", fontsize=12)
    plt.ylabel("Processing Time (seconds)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    if len(points) > 1:
        z = np.polyfit(points_m, times, 1)
        p = np.poly1d(z)
        plt.plot(points_m, p(points_m), "r--", alpha=0.4, label=f"Avg Speed: {1/z[0]:.1f} M pts/sec")
        plt.legend()

    out_path = os.path.join(output_dir, f"Veg_Efficiency_Scatter_{location_name}_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    
    print(f"[MAIN] Figures saved to {output_dir}")

def main():
    script_start_time = time.time()
    print("[START] CloudCompare vegetation-stripper with Reporting")
    
    parser = argparse.ArgumentParser(description="Strip vegetation using CANUPO via CloudCompare")
    parser.add_argument("location", help="Survey location (e.g. SanElijo)")
    parser.add_argument("--cc", default=None, help="Full path to CloudCompare executable")
    parser.add_argument("--test", type=int, default=None, help="Only process N files")
    parser.add_argument("--replace", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--single", action="store_true", help="Process one file at a time (no parallelism)")
    args = parser.parse_args()

    # Determine OS and base paths
    system = platform.system()
    if system == "Darwin":
        base_dir   = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
        default_cc = "/Applications/CloudCompare.app/Contents/MacOS/CloudCompare"
    else:
        base_dir   = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
        default_cc = "/usr/local/bin/CloudCompare"
    
    cc_path = args.cc or default_cc
    loc = args.location
    
    # Global Shift Configuration
    shift = {
        "SanElijo": ('-473000', '-3653000', '0'),
        "Encinitas": ('-472000', '-3655000', '0'),
        "Solana": ('-475000', '-3653000', '0'),
        "DelMar": ('-475000', '-3653000', '0'),
        "Torrey": ('-475000', '-3653000', '0')
    }.get(loc, ('-475000', '-3653000', '0'))

    # Directories
    las_dir        = os.path.join(base_dir, "results", loc, "nobeach")
    classifier_prm = os.path.join(base_dir, "utilities", "canupo", "veg_classifier_master.prm")
    output_dir     = os.path.join(base_dir, "results", loc, "noveg")
    report_dir     = os.path.join(base_dir, "utilities", "canupo", "reports")

    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Find Files
    all_las = sorted(glob.glob(os.path.join(las_dir, "*.las")))
    if not all_las:
        print(f"[ERROR] No .las files found in {las_dir}")
        return

    # Test sampling
    if args.test and args.test > 0:
        sample_n = min(args.test, len(all_las))
        all_las = random.sample(all_las, sample_n)
        print(f"[MAIN] TEST MODE: processing {len(all_las)} files")

    print(f"[MAIN] Processing {len(all_las)} files for {loc}...")
    
    results = []

    # Execution Mode
    if args.single:
        print(f"[MAIN] Single-threaded mode active")
        for las_path in all_las:
            res = classify_file_with_stats(las_path, classifier_prm, output_dir, shift, cc_path, args.replace)
            results.append(res)
    else:
        n_workers = max(1, multiprocessing.cpu_count() // 4)
        print(f"[MAIN] Parallel mode active ({n_workers} workers)")
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futures = {
                exe.submit(classify_file_with_stats, f, classifier_prm, output_dir, shift, cc_path, args.replace): f
                for f in all_las
            }
            for fut in as_completed(futures):
                results.append(fut.result())

    # =========================================================
    # Report Generation
    # =========================================================
    total_duration = time.time() - script_start_time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(report_dir, f"veg_report_{loc}_{timestamp}.csv")
    
    print(f"\n[MAIN] All tasks completed in {total_duration:.2f} seconds.")
    print(f"[MAIN] Saving report to: {report_file}")
    
    fieldnames = [
        "filename", "status", "input_points", "output_points", 
        "removed_points", "percent_removed", "processing_time_sec", "error_message"
    ]
    
    with open(report_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
        # Summary Row
        valid_results = [r for r in results if r["status"] == "Success"]
        total_in = sum(r['input_points'] for r in valid_results)
        total_rem = sum(r['removed_points'] for r in valid_results)
        avg_time = sum(r['processing_time_sec'] for r in valid_results) / len(valid_results) if valid_results else 0
        
        summary = {
            "filename": "SUMMARY_TOTALS",
            "status": "COMPLETE",
            "input_points": total_in,
            "output_points": sum(r['output_points'] for r in valid_results),
            "removed_points": total_rem,
            "percent_removed": round((total_rem/total_in)*100, 2) if total_in > 0 else 0,
            "processing_time_sec": round(total_duration, 2),
            "error_message": f"Avg time per file: {avg_time:.2f}s"
        }
        writer.writerow(summary)

    # =========================================================
    # Figure Generation
    # =========================================================
    generate_summary_figures(results, report_dir, loc, timestamp)

if __name__ == "__main__":
    main()