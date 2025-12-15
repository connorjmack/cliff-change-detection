#!/usr/bin/env python3
"""
15_extract_event_las_files.py

Forensic LAS Extractor (Final Robust Version):
1. Reads the survey list to find the folder name for a specific date.
2. Reconstructs the path using "Search Roots" (VMZ2000, VMQLZ, etc.) 
   to find the ORIGINAL raw .las file.
3. Locates the processed files (Cropped, NoBeach, NoVeg).
4. Crops all of them to the specific event bounding box.
5. Saves the subsets as new .las files.
"""

import os
import glob
import argparse
import platform
import pandas as pd
import laspy
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_LOC = "DelMar"
DEFAULT_D1  = "20250124"
DEFAULT_D2  = "20251117"

# Bounding Box (Point #0 to Point #3)
DEFAULT_BOUNDS = [475013.07, 475086.37, 3646171.70, 3646219.37]

# 1. OS Detection
if platform.system() == 'Darwin':
    LIDAR_ROOT = "/Volumes/group/LiDAR"
    PROJECT_ROOT = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
else:
    LIDAR_ROOT = "/project/group/LiDAR"
    PROJECT_ROOT = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

# 2. Search Roots (Where raw data lives)
POTENTIAL_ROOTS = [
    os.path.join(LIDAR_ROOT, "VMZ2000_Truck", "LiDAR_Processed_Level2"),
    os.path.join(LIDAR_ROOT, "MiniRanger_Truck", "LiDAR_Processed_Level2"),
    os.path.join(LIDAR_ROOT, "MiniRanger_ATV", "LiDAR_Processed_Level2"),
    os.path.join(LIDAR_ROOT, "VMQLZ_Truck", "LiDAR_Processed_Level2")  # <-- Catches your example
]

# ============================================================================
# PATH FINDING LOGIC
# ============================================================================

def find_raw_file_robust(path_entry):
    """
    Reconstructs the current path for a survey.
    Input: A potentially stale path from CSV (e.g. /old/path/20240905_..._NoWaves)
    Output: The correct, current path on disk.
    """
    if pd.isna(path_entry): return None
    
    raw_entry = str(path_entry).strip().rstrip('/')
    
    # SAFETY FIX: If CSV points to a file, get the parent folder
    if raw_entry.lower().endswith('.las'):
        # E.g. /path/to/SurveyName/file.las -> SurveyName
        # Or if the file IS the name we want, we split extension.
        # Usually standard is FolderName == FileName_prefix.
        # Safest bet: take the basename of the folder path.
        if os.path.isfile(raw_entry):
             raw_entry = os.path.dirname(raw_entry)
        else:
             # If path doesn't exist, just strip extension to guess folder name
             raw_entry = os.path.splitext(raw_entry)[0]

    folder_name = os.path.basename(raw_entry)
    
    # Construct the standard filename expected inside that folder
    # Matches your example: [Folder]_beach_cliff_ground.las
    target_filename = f"{folder_name}_beach_cliff_ground.las"
    
    # Strategy 1: Check if the CSV path happens to work (with OS fix)
    direct_path = os.path.join(raw_entry, "Beach_And_Backshore", target_filename)
    if platform.system() == 'Darwin' and '/project/' in direct_path:
        direct_path = direct_path.replace('/project/', '/Volumes/')
    elif platform.system() != 'Darwin' and '/Volumes/' in direct_path:
        direct_path = direct_path.replace('/Volumes/', '/project/')

    if os.path.exists(direct_path):
        return direct_path
    
    # Strategy 2: Search all known LiDAR roots
    for root in POTENTIAL_ROOTS:
        # Construct: ROOT / SurveyFolder / Beach_And_Backshore / SurveyFile.las
        candidate = os.path.join(root, folder_name, "Beach_And_Backshore", target_filename)
        if os.path.exists(candidate):
            return candidate
            
    return None

def get_raw_path_from_csv(survey_csv, date_str):
    if not os.path.exists(survey_csv):
        print(f"[ERROR] Survey list not found: {survey_csv}")
        return None

    try:
        df = pd.read_csv(survey_csv)
        mask = df.astype(str).apply(lambda x: x.str.contains(date_str)).any(axis=1)
        row = df[mask]
        
        if row.empty:
            print(f"  [WARN] Date {date_str} not found in survey CSV.")
            return None
        
        # Grab the path column (first column with a slash)
        path_entry = None
        for col in df.columns:
            val = str(row.iloc[0][col])
            if '/' in val or '\\' in val:
                path_entry = val
                break
        
        if not path_entry: path_entry = row.iloc[0][-1]

        return find_raw_file_robust(path_entry)

    except Exception as e:
        print(f"[ERROR] Reading CSV: {e}")
        return None

def find_pipeline_file(folder, date_str):
    if not os.path.exists(folder): return None
    candidates = [f for f in os.listdir(folder) if f.startswith(date_str) and f.endswith(".las")]
    if candidates:
        return os.path.join(folder, candidates[0])
    return None

def crop_and_save(in_path, out_path, bounds):
    if not in_path or not os.path.exists(in_path):
        print(f"  [SKIP] Input missing: {in_path}")
        return

    try:
        with laspy.open(in_path) as f:
            las = f.read()

        minx, maxx, miny, maxy = bounds
        mask = (las.x >= minx) & (las.x <= maxx) & \
               (las.y >= miny) & (las.y <= maxy)
        
        count = np.sum(mask)
        if count == 0:
            print(f"  [SKIP] No points in bounds ({os.path.basename(in_path)})")
            return

        out_las = laspy.LasData(las.header)
        out_las.points = las.points[mask]
        out_las.header.min = [out_las.x.min(), out_las.y.min(), out_las.z.min()]
        out_las.header.max = [out_las.x.max(), out_las.y.max(), out_las.z.max()]
        
        out_las.write(out_path)
        print(f"  [SAVED] {os.path.basename(out_path)} ({count} pts)")
        
    except Exception as e:
        print(f"  [ERROR] Processing {os.path.basename(in_path)}: {e}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("location", nargs='?', default=DEFAULT_LOC)
    parser.add_argument("--d1", default=DEFAULT_D1)
    parser.add_argument("--d2", default=DEFAULT_D2)
    args = parser.parse_args()
    
    loc, d1, d2 = args.location, args.d1, args.d2
    bounds = DEFAULT_BOUNDS
    
    # Output Paths
    survey_csv = os.path.join(PROJECT_ROOT, "survey_lists", f"surveys_{loc}.csv")
    results_dir = os.path.join(PROJECT_ROOT, "results", loc)
    out_dir = os.path.join(PROJECT_ROOT, "figures", "workflow", "las_files")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n--- LAS Forensic Extraction ---")
    print(f"Loc: {loc} | Dates: {d1}, {d2}")
    print(f"Out: {out_dir}\n")

    steps = [
        ("original", "raw", "raw"),
        ("cropped", "pipeline", "cropped"),
        ("noBeach", "pipeline", "noBeach"),
        ("noVeg", "pipeline", "noVeg")
    ]

    dates_map = {"c1": d1, "c2": d2}

    for prefix, date in dates_map.items():
        print(f"Processing {date} ({prefix})...")
        for step_name, source_type, suffix in steps:
            input_path = None
            
            if source_type == "raw":
                input_path = get_raw_path_from_csv(survey_csv, date)
                if not input_path:
                    print(f"  [WARN] Raw file not found for {date} (checked all roots)")
            
            elif source_type == "pipeline":
                search_dir = os.path.join(results_dir, step_name.lower())
                input_path = find_pipeline_file(search_dir, date)
            
            if input_path:
                out_name = f"{prefix}_{suffix}.las"
                print(f"  Step: {step_name}")
                print(f"    In: {input_path}")
                crop_and_save(input_path, os.path.join(out_dir, out_name), bounds)

    print("\n[COMPLETE]")

if __name__ == "__main__":
    main()