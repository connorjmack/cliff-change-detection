#!/usr/bin/env python3
"""
step0_update_survey_lists.py

Step 0 of the Daily Pipeline.
1. Scans for NEW surveys (newer than the latest in the CSV).
2. Validates they contain a '_beach_cliff_ground.las' file.
3. UPDATES the CSV list by appending new surveys AND re-sorting chronologically.
4. Ensures all paths in CSV use '/Volumes/' format.
5. LOGS the results to the daily report txt file.

Usage:
    python3 step0_update_survey_lists.py --all
"""

import os
import csv
import glob
import platform
import argparse
import pandas as pd
from datetime import datetime

# === CONFIGURATION ===
system = platform.system()
if system == "Darwin":
    ROOT_LIDAR = "/Volumes/group/LiDAR"
else:
    ROOT_LIDAR = "/project/group/LiDAR"

REPORT_DIR = os.path.join(ROOT_LIDAR, "LidarProcessing/LidarProcessingCliffs/code/pipeline/daily_reports")
CSV_DIR = os.path.join(ROOT_LIDAR, "LidarProcessing/LidarProcessingCliffs/survey_lists")

instrument_paths = {
    "MiniRanger_Truck": os.path.join(ROOT_LIDAR, "MiniRanger_Truck/LiDAR_Processed_Level2"),
    "MiniRanger_ATV"  : os.path.join(ROOT_LIDAR, "MiniRanger_ATV/LiDAR_Processed_Level2"),
    "VMQLZ_Truck"     : os.path.join(ROOT_LIDAR, "VMQLZ_Truck/LiDAR_Processed_Level2"),
    "VMZ2000_Truck"   : os.path.join(ROOT_LIDAR, "VMZ2000_Truck/LiDAR_Processed_Level2"),
}

mop_ranges = {
    "DelMar"      : [595, 620],
    "Solana"      : [637, 666],
    "Encinitas"   : [708, 764],
    "SanElijo"    : [683, 708],
    "Torrey"      : [567, 581],
    "Blacks"      : [520, 567]
}

# === REPORTING HELPERS ===

def get_report_path():
    today = datetime.now().strftime("%Y%m%d")
    return os.path.join(REPORT_DIR, f"daily_report_{today}.txt")

def init_report():
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = get_report_path()
    
    # Check if file exists and has content
    file_exists = os.path.exists(report_path)
    file_is_empty = file_exists and os.path.getsize(report_path) == 0
    
    # If file doesn't exist OR is empty, write the header
    if not file_exists or file_is_empty:
        with open(report_path, "w") as f:
            f.write(f"=== DAILY REPORT: {datetime.now().strftime('%Y-%m-%d')} ===\n\n")
            f.flush()
            os.fsync(f.fileno()) # Force write to disk
        print(f"[INFO] Created new daily report: {report_path}")
    else:
        print(f"[INFO] Appending to existing report: {report_path}")

def append_to_report(lines):
    if not lines: return
    report_path = get_report_path()
    try:
        with open(report_path, "a") as f:
            for line in lines:
                f.write(line + "\n")
                print(line) # Print to console
            f.flush()
            os.fsync(f.fileno()) # Force write to disk
    except Exception as e:
        print(f"[ERROR] Could not write to report file: {e}")

# === CSV HELPERS ===

def ensure_mac_path(path):
    """
    Forces the path to start with /Volumes/group/LiDAR...
    """
    if path.startswith("/project/group/LiDAR"):
        return path.replace("/project/group/LiDAR", "/Volumes/group/LiDAR")
    return path

def update_csv_sorted(csv_path, new_rows):
    """
    Reads existing CSV, appends new rows, sorts by date, rewrites file.
    """
    fieldnames = ["path", "date", "MOP1", "MOP2", "beach", "method"]
    
    # 1. Load existing data
    if os.path.exists(csv_path):
        # Read strictly as strings to avoid messing up formatting
        df = pd.read_csv(csv_path, dtype={'date': int})
    else:
        df = pd.DataFrame(columns=fieldnames)

    # 2. Append new rows
    new_df = pd.DataFrame(new_rows)
    # Ensure new_df has same columns
    if not new_df.empty:
        df = pd.concat([df, new_df], ignore_index=True)

    # 3. Sort by Date
    if 'date' in df.columns:
        df = df.sort_values(by="date")

    # 4. Save back to CSV
    df.to_csv(csv_path, index=False)
    print(f"[CSV UPDATE] Wrote {len(new_rows)} new rows to {os.path.basename(csv_path)}")

def get_max_existing_date(csv_path):
    """Reads existing CSV to find the cutoff date."""
    max_date = 0
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and 'date' in df.columns:
                max_date = df['date'].max()
        except Exception:
            pass
    return max_date

# === MAIN LOGIC ===

def find_target_las_files(directory):
    return sorted([
        f for f in glob.glob(os.path.join(directory, "**", "*.las"), recursive=True)
        if f.lower().endswith("_beach_cliff_ground.las")
    ])

def process_location(location):
    """
    Returns True if new surveys were found and processed, False otherwise.
    """
    if location not in mop_ranges: return False

    min_line, max_line = mop_ranges[location]
    csv_path = os.path.join(CSV_DIR, f"surveys_{location}.csv")
    max_existing_date = get_max_existing_date(csv_path)
    
    new_csv_rows = []
    report_lines = []

    for method, root in instrument_paths.items():
        if not os.path.isdir(root): continue

        for folder_name in os.listdir(root):
            survey_dir = os.path.join(root, folder_name)
            if not os.path.isdir(survey_dir): continue

            parts = folder_name.split("_")
            if len(parts) < 3: continue

            try:
                date_str = parts[0]
                date_int = int(date_str)
                mop1 = int(parts[1])
                mop2 = int(parts[2])
            except: continue

            # 1. NEWER DATE CHECK
            if date_int <= max_existing_date: continue

            # 2. OVERLAP CHECK
            overlap = min(mop2, max_line) - max(mop1, min_line)
            two_thirds = (max_line - min_line) * 2/3

            if overlap >= two_thirds:
                # 3. VALIDATE CONTENT
                las_files = find_target_las_files(survey_dir)
                
                if las_files:
                    # Prepare CSV Row
                    mac_path = ensure_mac_path(survey_dir)
                    new_row = {
                        "path": mac_path,
                        "date": date_int, 
                        "MOP1": mop1,
                        "MOP2": mop2,
                        "beach": location,
                        "method": method
                    }
                    new_csv_rows.append(new_row)

                    # Prepare Report Lines
                    for lp in las_files:
                        mac_las_path = ensure_mac_path(lp)
                        report_lines.append(f"NEW: {location} | {date_str} | {mac_las_path}")

    # Perform Updates if we found anything
    if new_csv_rows:
        update_csv_sorted(csv_path, new_csv_rows)
        append_to_report(report_lines)
        return True
    
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    # 1. Initialize report file
    init_report()
    
    found_something = False

    # 2. Run Checks
    if args.location:
        if process_location(args.location):
            found_something = True
    else:
        for loc in mop_ranges.keys():
            if process_location(loc):
                found_something = True
    
    # 3. If nothing found, explicitly log that
    if not found_something:
        timestamp = datetime.now().strftime('%H:%M:%S')
        msg = f"No new surveys found (Run: {timestamp}). CSVs are up to date."
        append_to_report([msg])

if __name__ == "__main__":
    main()