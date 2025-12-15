#!/usr/bin/env python3

"""
Command-line arguments:
  location       Survey location (e.g. SanElijo)

Optional flags:
  --n_jobs       Number of parallel workers (default: 5)
  --replace      If set, overwrite existing output files.
"""

import os
import glob
import argparse
import joblib
import numpy as np
import laspy
import platform
import time
import csv
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

# ==========================================
# HELPER: Histogram Matching
# ==========================================
def match_histograms(source, reference):
    """
    Warps 'source' values to match the distribution shape of 'reference'.
    Uses pure Numpy for interpolation.
    """
    src_sorted = np.sort(source)
    ref_sorted = np.sort(reference)
    
    src_indices = np.linspace(0, 1, len(source))
    ref_indices = np.linspace(0, 1, len(reference))
    
    ref_interp = np.interp(src_indices, ref_indices, ref_sorted)
    
    sorter = np.argsort(source)
    inverse_sorter = np.argsort(sorter)
    
    return ref_interp[inverse_sorter]

def read_las_to_array(las_path):
    """Read LAS with las.xyz and return (las, data[N×4])."""
    las = laspy.read(las_path)
    coords    = las.xyz                        # (N,3)
    intensity = las.intensity.astype(float)    # (N,)
    data = np.column_stack((coords, intensity))  # (N,4)
    return las, data

def classify_and_write(las_path, model_path, scaler_path, output_dir, ref_intensity=None, replace=False):
    start_time = time.time()
    base_name = os.path.basename(las_path)
    
    # Initialize stats dictionary
    stats = {
        "filename": base_name,
        "status": "Unknown",
        "total_points": 0,
        "beach_points_removed": 0,
        "cliff_points_kept": 0,
        "percent_removed": 0.0,
        "histogram_matched": False,
        "processing_time_sec": 0.0,
        "error_message": ""
    }

    try:
        out_name = f"{os.path.splitext(base_name)[0]}_nobeach.las"
        out_path = os.path.join(output_dir, out_name)

        if os.path.exists(out_path) and not replace:
            print(f"[SKIP] {out_name} already exists.")
            stats["status"] = "Skipped"
            return stats

        # Load model AND scaler
        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        las, data = read_las_to_array(las_path)
        
        # Extract features
        coords = data[:, :3]          # (N, 3)
        intensity = data[:, 3]        # (N,) - raw intensity
        
        # --- NEW: Histogram Matching ---
        if ref_intensity is not None:
            intensity = match_histograms(source=intensity, reference=ref_intensity)
            stats["histogram_matched"] = True
        # -------------------------------

        # Normalize intensity
        intensity_norm = scaler.transform(intensity.reshape(-1, 1)).flatten()
        
        # Assemble feature matrix
        X = np.column_stack((coords, intensity_norm))

        if X.shape[1] != clf.n_features_in_:
            raise ValueError(f"Model expects {clf.n_features_in_} features, got {X.shape[1]}")

        labels = clf.predict(X)
        cliff_points = np.sum(labels == 0)
        beach_points = np.sum(labels == 1)
        total_points = len(labels)

        # Update stats
        stats["total_points"] = total_points
        stats["beach_points_removed"] = beach_points
        stats["cliff_points_kept"] = cliff_points
        stats["percent_removed"] = round((beach_points / total_points) * 100, 2) if total_points > 0 else 0

        print(f"[INFO] {base_name}: Removed {beach_points} beach points ({stats['percent_removed']}%)")

        if cliff_points == 0:
            print(f"[WARNING] No cliff points in {base_name} — skipping write.")
            stats["status"] = "No Cliff Points"
            return stats

        mask = (labels == 0)
        las.points = las.points[mask]
        las.write(out_path)
        stats["status"] = "Success"
        print(f"    • Wrote {out_path}")

    except Exception as e:
        print(f"[ERROR] {base_name} failed: {e}")
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
    
    # Filter for SUCCESS runs only (exclude Skipped to avoid 0% bias)
    valid_results = [r for r in results if r["status"] == "Success"]
    
    if not valid_results:
        print("[WARN] No successful processing runs to plot (all files may have been skipped or failed).")
        return

    # Extract Data
    dates = []
    percents = []
    times = []
    points = []
    filenames = []
    
    for r in valid_results:
        # Try to extract date from filename (Assuming YYYYMMDD_... format)
        try:
            date_str = r['filename'].split('_')[0]
            dt = datetime.strptime(date_str, "%Y%m%d")
            dates.append(dt)
            percents.append(r['percent_removed'])
            times.append(r['processing_time_sec'])
            points.append(r['total_points'])
            filenames.append(r['filename'])
        except ValueError:
            pass

    # --- FIGURE 1: Time Series of Removal Percentage ---
    if dates:
        plt.figure(figsize=(12, 6), dpi=300)
        
        sorted_pairs = sorted(zip(dates, percents))
        s_dates, s_percents = zip(*sorted_pairs)
        
        plt.plot(s_dates, s_percents, marker='o', linestyle='-', color='#2c7bb6', alpha=0.7, linewidth=1)
        plt.scatter(s_dates, s_percents, c=s_percents, cmap='viridis', zorder=10, s=50, edgecolors='k')
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gcf().autofmt_xdate()
        
        plt.title(f"Beach Removal Percentage Over Time - {location_name}", fontsize=14, fontweight='bold')
        plt.ylabel("Points Removed (%)", fontsize=12)
        plt.xlabel("Survey Date", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if len(s_dates) > 5:
            x_nums = mdates.date2num(s_dates)
            z = np.polyfit(x_nums, s_percents, 1)
            p = np.poly1d(z)
            plt.plot(s_dates, p(x_nums), "r--", alpha=0.5, label=f"Trend (Slope: {z[0]:.4f})")
            plt.legend()

        out_path = os.path.join(output_dir, f"Figure_TimeSeries_Removal_{location_name}_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"    • Saved {os.path.basename(out_path)}")

    # --- FIGURE 2: Time Series of Processing Time (NEW) ---
    if dates:
        plt.figure(figsize=(12, 6), dpi=300)
        
        sorted_pairs = sorted(zip(dates, times))
        s_dates, s_times = zip(*sorted_pairs)
        
        plt.plot(s_dates, s_times, marker='s', linestyle='-', color='#d95f02', alpha=0.7, linewidth=1)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gcf().autofmt_xdate()
        
        plt.title(f"Processing Time per Survey - {location_name}", fontsize=14, fontweight='bold')
        plt.ylabel("Processing Time (seconds)", fontsize=12)
        plt.xlabel("Survey Date", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        out_path = os.path.join(output_dir, f"Figure_TimeSeries_Time_{location_name}_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"    • Saved {os.path.basename(out_path)}")

    # --- FIGURE 3: Efficiency Scatter Plot (Points vs Time) (NEW) ---
    if points:
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Convert points to millions for cleaner axis
        points_m = [p / 1_000_000 for p in points]
        
        plt.scatter(points_m, times, alpha=0.7, c='#7570b3', edgecolors='k', s=60)
        
        plt.title(f"Processing Efficiency (Size vs Time) - {location_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Total Points (Millions)", fontsize=12)
        plt.ylabel("Processing Time (seconds)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Optional Trendline for efficiency
        if len(points) > 1:
            z = np.polyfit(points_m, times, 1)
            p = np.poly1d(z)
            plt.plot(points_m, p(points_m), "r--", alpha=0.4, label=f"Avg Speed: {1/z[0]:.1f} M pts/sec")
            plt.legend()

        out_path = os.path.join(output_dir, f"Figure_Scatter_Efficiency_{location_name}_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"    • Saved {os.path.basename(out_path)}")

    # --- FIGURE 4: Histogram of Removal Percentages ---
    all_percents = [r['percent_removed'] for r in valid_results]
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.hist(all_percents, bins=20, color='#1a9850', edgecolor='black', alpha=0.7)
    
    mean_val = np.mean(all_percents)
    median_val = np.median(all_percents)
    
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.1f}%')
    plt.axvline(median_val, color='blue', linestyle='dotted', linewidth=1.5, label=f'Median: {median_val:.1f}%')
    
    plt.title(f"Distribution of Beach Removal Rates - {location_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Points Removed (%)", fontsize=12)
    plt.ylabel("Frequency (Processed Files Only)", fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    out_path = os.path.join(output_dir, f"Figure_Distribution_{location_name}_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"    • Saved {os.path.basename(out_path)}")

def main():
    script_start_time = time.time()
    
    p = argparse.ArgumentParser(description="Parallel beach removal with Histogram Matching")
    p.add_argument("location", help="e.g. SanElijo")
    p.add_argument("--n_jobs", type=int, default=5, help="parallel workers")
    p.add_argument("--replace", action="store_true", help="Overwrite existing output")
    args = p.parse_args()

    # Detect OS
    is_mac = platform.system() == "Darwin"
    if is_mac:
        root = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        root = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

    loc   = args.location
    base  = os.path.join(root, "results", loc)
    inp   = os.path.join(base, "cropped")
    
    joblib_dir = os.path.join(root, "utilities", "beach_removal", "joblib_files")
    model = os.path.join(joblib_dir, f"{loc}RF.joblib")
    scaler = os.path.join(joblib_dir, f"{loc}_scaler.joblib")
    
    outdir = os.path.join(base, "nobeach_new")
    report_dir = os.path.join(root, "utilities", "beach_removal", "classification_reports")

    # =========================================================
    # Reference Loading - Dynamic Selection
    # =========================================================
    
    training_root = os.path.join(root, "utilities", "beach_removal", "training_data")
    
    # -------------------------------------------------------------
    # DICTIONARY MAPPING LOCATIONS TO REFERENCE FILES
    # Uncomment lines below when you create training.las files for other locations
    # -------------------------------------------------------------
    ref_file_map = {
        "DelMar": os.path.join(training_root, "DelMar", "DelMar_training.las"),
        # "SanElijo": os.path.join(training_root, "SanElijo", "SanElijo_training.las"),
        # "TorreyPines": os.path.join(training_root, "TorreyPines", "TorreyPines_training.las"),
        # "SolanaBeach": os.path.join(training_root, "SolanaBeach", "SolanaBeach_training.las"),
        # "Carlsbad": os.path.join(training_root, "Carlsbad", "Carlsbad_training.las"),
    }
    
    # Select the file based on the location argument
    ref_path = ref_file_map.get(loc)
    ref_intensity_sample = None
    
    if ref_path and os.path.exists(ref_path):
        print(f"[MAIN] Loading baseline reference for {loc}: {os.path.basename(ref_path)}")
        try:
            ref_las = laspy.read(ref_path)
            full_ref_int = ref_las.intensity.astype(float)
            if len(full_ref_int) > 100000:
                ref_intensity_sample = np.random.choice(full_ref_int, 100000, replace=False)
            else:
                ref_intensity_sample = full_ref_int
            print(f"[MAIN] Baseline loaded. Histogram matching enabled.")
        except Exception as e:
            print(f"[WARNING] Could not read reference file: {e}")
    else:
        # Fallback/Warning Block
        print("\n" + "!"*80)
        if ref_path is None:
            print(f"WARNING: No reference file mapped for location: {loc}")
            print("       Please update 'ref_file_map' in the script if you have a training file.")
        else:
            print(f"WARNING: Reference file path defined but NOT FOUND: {ref_path}")
            
        print("       Processing will proceed WITHOUT Histogram Matching.")
        print("       (This may cause issues if sensors have different intensity scales)")
        print("!"*80 + "\n")

    # =========================================================

    if not os.path.exists(model): raise FileNotFoundError(f"Model not found: {model}")
    if not os.path.exists(scaler): raise FileNotFoundError(f"Scaler not found: {scaler}")

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    las_files = glob.glob(os.path.join(inp, "*.las"))
    if not las_files:
        print(f"[MAIN] No .las files found in {inp}")
        return

    print(f"[MAIN] Processing {len(las_files)} files...")
    results = []

    with ProcessPoolExecutor(max_workers=args.n_jobs) as exe:
        futures = [
            exe.submit(classify_and_write, f, model, scaler, outdir, ref_intensity_sample, args.replace)
            for f in las_files
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    # =========================================================
    # Report Generation
    # =========================================================
    total_duration = time.time() - script_start_time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(report_dir, f"classification_report_{loc}_{timestamp}.csv")
    
    print(f"\n[MAIN] All tasks completed in {total_duration:.2f} seconds.")
    print(f"[MAIN] Saving report to: {report_file}")
    
    fieldnames = [
        "filename", "status", "total_points", "beach_points_removed", 
        "cliff_points_kept", "percent_removed", "histogram_matched", 
        "processing_time_sec", "error_message"
    ]
    
    with open(report_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
        # Summary row
        total_pts = sum(r['total_points'] for r in results)
        total_removed = sum(r['beach_points_removed'] for r in results)
        avg_time = sum(r['processing_time_sec'] for r in results) / len(results) if results else 0
        
        summary = {
            "filename": "SUMMARY_TOTALS",
            "status": "COMPLETE",
            "total_points": total_pts,
            "beach_points_removed": total_removed,
            "cliff_points_kept": total_pts - total_removed,
            "percent_removed": round((total_removed/total_pts)*100, 2) if total_pts > 0 else 0,
            "histogram_matched": "N/A",
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