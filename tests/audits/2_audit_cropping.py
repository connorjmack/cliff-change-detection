#!/usr/bin/env python3
"""
2_audit_cropping.py

Usage:
    python3 tests/audits/2_audit_cropping.py
    python3 tests/audits/2_audit_cropping.py --location SanElijo
    python3 tests/audits/2_audit_cropping.py --delete_bad_files

Description:
    Scans the results/<location>/cropped/ directories for all locations.
    Extracts file size (MB) and LAS Point Counts.
    Generates distribution plots to identify outliers or failed crops.
    Compares cropped files against the survey_list CSV for each location scanned.
    
    Optional:
        --location         : Only scan a single location in results/<location>/cropped.
        --delete_bad_files : If set, permanently deletes files with points < threshold.
                             Plots will highlight deleted files in RED.

    Output:
        Creates a timestamped folder in .../code/pipeline/reports/
        Saves CSV inventories, QC figures, and a run summary there.
"""

import os
import glob
import platform
import laspy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime

# === CONFIGURATION ===
# Threshold for a "suspiciously small" file (e.g., less than 1000 points)
MIN_POINT_THRESHOLD = 1000 

def get_root_path():
    """Determine system root path based on OS."""
    system = platform.system()
    if system == "Darwin":
        # Mac Path
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        # Linux/Cluster Path
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

def get_lidar_base(project_root):
    """Return the base LiDAR directory (e.g., /Volumes/group/LiDAR)."""
    return os.path.dirname(os.path.dirname(project_root))

def analyze_file(filepath):
    """
    Worker function to get metrics from a single LAS file.
    Reads header only for speed.
    """
    try:
        # Get file size in MB
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)

        # Get point count from LAS header (fast, doesn't read points)
        with laspy.open(filepath) as f:
            point_count = f.header.point_count

        # Extract metadata from path
        parts = filepath.split(os.sep)
        try:
            res_idx = parts.index("results")
            location = parts[res_idx + 1]
        except ValueError:
            location = "Unknown"

        filename = os.path.basename(filepath)

        return {
            "Location": location,
            "Filename": filename,
            "Size_MB": size_mb,
            "Point_Count": point_count,
            "Path": filepath,
            "Status": "OK" if point_count > MIN_POINT_THRESHOLD else "SUSPECT"
        }
    except Exception as e:
        return {
            "Location": "Unknown",
            "Filename": os.path.basename(filepath),
            "Size_MB": 0,
            "Point_Count": 0,
            "Path": filepath,
            "Status": f"ERROR: {str(e)}"
        }

def compare_cropped_to_survey_list(location, project_root, cropped_files, output_dir):
    """Compare cropped files to the survey list and write mismatch reports."""
    summary = ["", f"Survey List Check: {location}"]
    survey_csv = os.path.join(project_root, "survey_lists", f"surveys_{location}.csv")

    if not os.path.exists(survey_csv):
        msg = f"Survey list not found: {survey_csv}"
        print(f"[WARNING] {msg}")
        summary.append(msg)
        return summary

    try:
        survey_df = pd.read_csv(survey_csv)
    except Exception as e:
        msg = f"Survey list read error: {e}"
        print(f"[ERROR] {msg}")
        summary.append(msg)
        return summary

    base_dir = get_lidar_base(project_root)
    expected_map = {}
    missing_source_rows = []

    for row in survey_df.to_dict(orient="records"):
        method_val = row.get("method", "")
        survey_val = row.get("path", "")
        if pd.isna(method_val) or pd.isna(survey_val):
            missing_source_rows.append({
                "Location": location,
                "Survey_Path": "",
                "Method": "",
                "Pattern": ""
            })
            continue
        method = str(method_val).strip()
        survey_raw = str(survey_val).strip()
        if not method or not survey_raw:
            missing_source_rows.append({
                "Location": location,
                "Survey_Path": survey_raw,
                "Method": method,
                "Pattern": ""
            })
            continue

        survey_folder = os.path.basename(os.path.normpath(survey_raw))
        base = os.path.join(base_dir, method, "LiDAR_Processed_Level2", survey_folder)
        pattern = os.path.join(base, "Beach_And_Backshore", "*beach_cliff_ground.las")
        matches = glob.glob(pattern)

        if not matches:
            missing_source_rows.append({
                "Location": location,
                "Survey_Path": survey_raw,
                "Method": method,
                "Pattern": pattern
            })
            continue

        las_in = matches[0]
        stem = os.path.splitext(os.path.basename(las_in))[0]
        expected_name = f"{stem}_cropped.las"
        expected_map[expected_name] = {
            "Location": location,
            "Survey_Path": survey_raw,
            "Method": method,
            "Source_LAS": las_in,
            "Expected_Cropped": expected_name
        }

    expected_names = set(expected_map.keys())
    missing_cropped = sorted(expected_names - cropped_files)
    extra_cropped = sorted(cropped_files - expected_names)

    summary.append(f"Survey list rows: {len(survey_df)}")
    summary.append(f"Expected cropped files: {len(expected_names)}")
    summary.append(f"Missing cropped files: {len(missing_cropped)}")
    summary.append(f"Extra cropped files: {len(extra_cropped)}")
    summary.append(f"Survey rows missing source LAS: {len(missing_source_rows)}")

    print(f"\n=== Survey List Check: {location} ===")
    print(f"Survey list rows: {len(survey_df)}")
    print(f"Expected cropped files: {len(expected_names)}")
    print(f"Missing cropped files: {len(missing_cropped)}")
    print(f"Extra cropped files: {len(extra_cropped)}")
    print(f"Survey rows missing source LAS: {len(missing_source_rows)}")

    if missing_cropped:
        missing_rows = [expected_map[name] for name in missing_cropped]
        missing_path = os.path.join(output_dir, f"survey_list_missing_cropped_{location}.csv")
        pd.DataFrame(missing_rows).to_csv(missing_path, index=False)
        print(f"[INFO] Missing cropped files saved to: {missing_path}")
        summary.append("Missing cropped files:")
        summary.extend([f"- {name}" for name in missing_cropped])
        summary.append(f"Missing cropped CSV: {missing_path}")

    if extra_cropped:
        extra_rows = [{"Location": location, "Filename": name} for name in extra_cropped]
        extra_path = os.path.join(output_dir, f"survey_list_extra_cropped_{location}.csv")
        pd.DataFrame(extra_rows).to_csv(extra_path, index=False)
        print(f"[INFO] Extra cropped files saved to: {extra_path}")
        summary.append("Extra cropped files:")
        summary.extend([f"- {name}" for name in extra_cropped])
        summary.append(f"Extra cropped CSV: {extra_path}")

    if missing_source_rows:
        missing_source_path = os.path.join(output_dir, f"survey_list_missing_source_{location}.csv")
        pd.DataFrame(missing_source_rows).to_csv(missing_source_path, index=False)
        print(f"[INFO] Survey list rows missing source LAS saved to: {missing_source_path}")
        summary.append("Survey rows missing source LAS:")
        for row in missing_source_rows:
            summary.append(f"- {row.get('Survey_Path', '')} ({row.get('Method', '')})")
        summary.append(f"Missing source CSV: {missing_source_path}")

    return summary

def plot_results(df, output_dir, delete_mode):
    """Generates QC charts and highlights deleted/suspect files."""
    sns.set_theme(style="whitegrid")
    
    # Split data for highlighting
    good_data = df[df["Status"] == "OK"]
    bad_data = df[df["Status"] != "OK"]
    
    bad_label = "DELETED Files" if delete_mode else "SUSPECT Files"
    bad_color = "#e74c3c" # Red
    good_palette = "viridis"

    # --- 1. Scatter: Points vs Size ---
    plt.figure(figsize=(12, 8))

    # Plot Good Data (Standard)
    if not good_data.empty:
        sns.scatterplot(
            x="Size_MB", y="Point_Count", hue="Location",
            data=good_data, palette=good_palette, alpha=0.6, s=50, legend='brief'
        )

    # Overlay Bad Data (Highlighted)
    if not bad_data.empty:
        plt.scatter(
            bad_data["Size_MB"], bad_data["Point_Count"],
            color=bad_color, marker='X', s=100, label=bad_label, zorder=10
        )

    plt.title(f"QC Sanity Check: Point Count vs File Size\n(Threshold: {MIN_POINT_THRESHOLD} points)")
    plt.xlabel("File Size (MB)")
    plt.ylabel("Point Count")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "QC_Points_vs_Size.png"), dpi=150)
    plt.close()

    # --- 2. Boxplot: File Sizes by Location ---
    plt.figure(figsize=(14, 7))
    
    # Draw boxplot of ALL data
    sns.boxplot(x="Location", y="Size_MB", data=df, color="lightgrey", boxprops=dict(alpha=0.3))
    
    # Strip plot for GOOD data
    if not good_data.empty:
        sns.stripplot(x="Location", y="Size_MB", data=good_data, palette=good_palette, alpha=0.5, size=3, legend=False)

    # Strip plot for BAD data (Highlighted)
    if not bad_data.empty:
        sns.stripplot(
            x="Location", y="Size_MB", data=bad_data,
            color=bad_color, marker="X", size=8, jitter=0.2, label=bad_label, zorder=10
        )
        plt.legend(loc='upper right')

    plt.title(f"Distribution of Cropped File Sizes\n(Red X = {bad_label})")
    plt.ylabel("File Size (MB)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "QC_FileSize_Distribution.png"), dpi=150)
    plt.close()

    print(f"[INFO] Plots saved to: {output_dir}")

def main():
    # --- Parse Arguments ---
    parser = argparse.ArgumentParser(description="QC Cropped LAS files.")
    parser.add_argument(
        "--location",
        type=str,
        help="Only scan results for a single location (e.g., SanElijo)."
    )
    parser.add_argument(
        "--delete_bad_files", 
        action="store_true", 
        help="If set, permanently delete files with points below threshold."
    )
    args = parser.parse_args()

    root = get_root_path()
    
    # Define Directories
    results_dir = os.path.join(root, "results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_base = os.path.join(root, "code", "pipeline", "reports")
    daily_reports_dir = os.path.join(root, "reports", "daily")
    run_output_dir = os.path.join(reports_base, f"QC_Run_{timestamp}")
    
    os.makedirs(run_output_dir, exist_ok=True)

    print(f"--- QC Cropped Files ---")
    print(f"Mode: {'DESTRUCTIVE (Deleting bad files)' if args.delete_bad_files else 'REPORT ONLY'}")
    print(f"Scanning: {results_dir}")
    print(f"Report:   {run_output_dir}")
    
    # Find files
    if args.location:
        search_pattern = os.path.join(results_dir, args.location, "cropped", "*.las")
    else:
        search_pattern = os.path.join(results_dir, "*", "cropped", "*.las")
    files = glob.glob(search_pattern)

    if not files:
        print("[ERROR] No files found. Check paths or run crop_files_parallel.py first.")
        return

    print(f"Found {len(files)} files. Analyzing headers...")

    # Parallel processing
    results = []
    max_workers = min(10, os.cpu_count() or 4) 
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(analyze_file, f) for f in files]
        for future in tqdm(as_completed(futures), total=len(files), unit="file"):
            results.append(future.result())

    df = pd.DataFrame(results)
    if df.empty:
        print("[ERROR] Dataframe is empty.")
        return

    # --- ANALYSIS & DELETION LOGIC ---
    
    summary_lines = []
    summary_lines.append(f"QC Run: {timestamp}")
    summary_lines.append(f"Delete Mode Enabled: {args.delete_bad_files}")
    summary_lines.append(f"Total Files Scanned: {len(df)}")
    summary_lines.append("-" * 30)

    # Calculate Stats
    stats = df.groupby("Location").agg({
        "Filename": "count",
        "Size_MB": "mean",
        "Point_Count": "mean"
    }).rename(columns={"Filename": "File Count", "Size_MB": "Avg MB", "Point_Count": "Avg Pts"})
    
    print("\n=== Summary Statistics ===")
    print(stats.round(2))

    # Identify Suspects
    suspects = df[df["Point_Count"] < MIN_POINT_THRESHOLD].sort_values("Point_Count")
    
    deleted_count = 0
    errors_count = 0

    if not suspects.empty:
        print(f"\n[WARNING] Found {len(suspects)} files with < {MIN_POINT_THRESHOLD} points.")
        summary_lines.append(f"Suspect Files Found: {len(suspects)}")

        # Handle Deletion
        if args.delete_bad_files:
            print("[ACTION] Deleting bad files...")
            deletion_log = []
            
            for idx, row in suspects.iterrows():
                fpath = row['Path']
                try:
                    os.remove(fpath)
                    print(f"  Deleted: {row['Filename']}")
                    deletion_log.append(f"DELETED: {fpath} ({row['Point_Count']} pts)")
                    deleted_count += 1
                except OSError as e:
                    print(f"  ERROR deleting {row['Filename']}: {e}")
                    deletion_log.append(f"ERROR: Could not delete {fpath}: {e}")
                    errors_count += 1
            
            # Update summary with deletion results
            summary_lines.append(f"Files Deleted: {deleted_count}")
            summary_lines.append(f"Deletion Errors: {errors_count}")
            summary_lines.append("\nDeletion Log:")
            summary_lines.extend(deletion_log)
            
        else:
            print("[INFO] Run with --delete_bad_files to remove these files.")
            summary_lines.append("Files Deleted: 0 (Dry Run)")
            summary_lines.append("\nSuspect Files (Not Deleted):")
            for _, row in suspects.iterrows():
                summary_lines.append(f"{row['Path']} ({row['Point_Count']} pts)")

        # Save bad file list CSV
        bad_file_report = os.path.join(run_output_dir, "suspect_files.csv")
        suspects.to_csv(bad_file_report, index=False)
        print(f"[INFO] List of suspect files saved to: {bad_file_report}")

    else:
        print("\n[SUCCESS] No files below point threshold found.")
        summary_lines.append("Suspect Files Found: 0")

    # --- SURVEY LIST COMPARISON ---
    if args.location:
        locations = [args.location]
    else:
        locations = sorted({loc for loc in df["Location"].dropna().unique() if loc != "Unknown"})

    for location in locations:
        cropped_files = set(df.loc[df["Location"] == location, "Filename"])
        summary_lines.extend(
            compare_cropped_to_survey_list(location, root, cropped_files, run_output_dir)
        )

    # --- SAVE OUTPUTS ---

    # 1. Plots (Pass delete_mode to highlight correctly)
    plot_results(df, run_output_dir, args.delete_bad_files)

    # 2. Full Inventory CSV
    full_report_path = os.path.join(run_output_dir, "full_cropped_file_inventory.csv")
    df.to_csv(full_report_path, index=False)
    
    # 3. Text Summary
    summary_path = os.path.join(run_output_dir, "run_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    
    print(f"[INFO] Run summary saved to: {summary_path}")

    # Append to daily report (if present)
    daily_report_name = f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
    daily_report_path = os.path.join(daily_reports_dir, daily_report_name)
    if not os.path.exists(daily_report_path):
        os.makedirs(daily_reports_dir, exist_ok=True)
        with open(daily_report_path, "w") as f:
            f.write(f"Daily Report {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write("=" * 40 + "\n")
        print(f"[INFO] Created daily report: {daily_report_path}")

    with open(daily_report_path, "a") as f:
        f.write("\n\n=== QC Cropped Files ===\n")
        f.write("\n".join(summary_lines))
    print(f"[INFO] Appended summary to daily report: {daily_report_path}")

if __name__ == "__main__":
    main()
