#!/usr/bin/env python3
"""
16_extract_rgb_subsets.py

Purpose:
    Extracts small, spatially-aligned subsets of the raw and processed LAS files
    for the MOST RECENT VMQLZ (RGB-colorized) survey found.
    
    Preserves ALL LAS dimensions (Red, Green, Blue, Intensity, etc.) by 
    copying the full header schema.

Usage:
    python3 16_extract_rgb_subsets.py --location DelMar
"""

import os
import glob
import argparse
import platform
import pandas as pd
import laspy
import numpy as np
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_LOC = "DelMar"

# UPDATED BOUNDING BOX (Derived from provided corner points)
# Min X: 474921.43 (Point #3)
# Max X: 475248.42 (Point #1)
# Min Y: 3645924.16 (Point #2)
# Max Y: 3646282.23 (Point #1)
BOUNDS = [474921.43, 475248.42, 3645924.16, 3646282.23] # [minx, maxx, miny, maxy]

# Paths
if platform.system() == 'Darwin':
    LIDAR_ROOT = "/Volumes/group/LiDAR"
    PROJECT_ROOT = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
else:
    LIDAR_ROOT = "/project/group/LiDAR"
    PROJECT_ROOT = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

# Output Directory
OUT_DIR = os.path.join(PROJECT_ROOT, "figures", "ml_classification", "input_clouds")

# Potential locations for Raw Data (VMQLZ usually has RGB)
SEARCH_ROOTS = [
    os.path.join(LIDAR_ROOT, "VMQLZ_Truck", "LiDAR_Processed_Level2"),
    os.path.join(LIDAR_ROOT, "VMZ2000_Truck", "LiDAR_Processed_Level2"),
    os.path.join(LIDAR_ROOT, "MiniRanger_Truck", "LiDAR_Processed_Level2")
]

# ============================================================================
# HELPERS
# ============================================================================

def get_latest_date_from_filesystem(results_dir):
    """
    Fallback: Scans the 'cropped' directory to find the most recent date
    actually present on the disk.
    """
    cropped_dir = os.path.join(results_dir, "cropped")
    if not os.path.exists(cropped_dir):
        return None
        
    files = glob.glob(os.path.join(cropped_dir, "*.las"))
    dates = []
    
    for f in files:
        # Extract first 8 digits (YYYYMMDD)
        basename = os.path.basename(f)
        match = re.match(r"(\d{8})", basename)
        if match:
            dates.append(match.group(1))
            
    if dates:
        dates.sort(reverse=True) # Sort Descending (Newest First)
        print(f"[INFO] Filesystem scan found latest date: {dates[0]}")
        return dates[0]
    return None

def get_latest_vmqlz_date(survey_csv):
    """Finds the MOST RECENT survey in the list that is marked as VMQLZ."""
    if not os.path.exists(survey_csv):
        print(f"[WARN] Survey list not found: {survey_csv}")
        return None

    try:
        df = pd.read_csv(survey_csv)
        
        # CLEANING: Drop rows where the first column is NaN
        df = df.dropna(subset=[df.columns[0]])
        
        # 1. Filter for rows that likely contain VMQLZ data
        # We look across all columns just in case, or specifically the Scanner column if known
        mask = df.astype(str).apply(lambda x: x.str.contains("VMQLZ", case=False)).any(axis=1)
        vmqlz_df = df[mask].copy()
        
        if vmqlz_df.empty:
            print("[WARN] No VMQLZ surveys found in CSV list.")
            return None
            
        # 2. Sort by date DESCENDING (Newest First)
        date_col = vmqlz_df.columns[0]
        
        # Ensure column is string and clean up potential suffixes like '.0'
        vmqlz_df[date_col] = vmqlz_df[date_col].astype(str).str.replace(r'\.0$', '', regex=True)
        
        # Sort descending to get the latest date
        vmqlz_df = vmqlz_df.sort_values(by=date_col, ascending=False)
        
        # Get top row (latest date)
        first_val = vmqlz_df.iloc[0][date_col]
        
        # Handle cases where CSV might have full paths instead of just dates
        latest_date = None
        match = re.search(r"(\d{8})", first_val)
        if match:
            latest_date = match.group(1)
        
        if latest_date and len(latest_date) == 8 and latest_date.isdigit():
            print(f"[INFO] CSV identified latest VMQLZ Survey: {latest_date}")
            return latest_date
            
        print(f"[WARN] Could not extract valid date from CSV value: {first_val}")
        return None

    except Exception as e:
        print(f"[ERROR] Reading CSV: {e}")
        return None

def find_raw_file(date_str, location):
    """
    Locates the raw file by scanning SEARCH_ROOTS for any folder starting with the date
    AND containing the location name (e.g. DelMar).
    """
    # Iterate through the main LiDAR roots
    for root in SEARCH_ROOTS:
        if not os.path.exists(root):
            continue
            
        # SMART SCAN: Look for ANY folder starting with the target date
        candidate_folders = glob.glob(os.path.join(root, f"{date_str}*"))
        
        for folder_path in candidate_folders:
            if not os.path.isdir(folder_path):
                continue
            
            folder_name = os.path.basename(folder_path)
            
            # --- CRITICAL FIX: Verify Location Name ---
            # If "DelMar" is not in the folder name, skip it.
            # This avoids picking up "Blacks" or "Torrey" folders surveyed on the same day.
            if location.lower() not in folder_name.lower():
                continue

            # Construct possible filenames based on the folder name
            possible_filenames = [
                f"{folder_name}_beach_cliff_ground.las",
                "beach_cliff_ground.las",
                f"{date_str}_{location}_VMQLZ_beach_cliff_ground.las"
            ]
            
            # Search location 1: Direct inside date folder
            # Search location 2: Inside Beach_And_Backshore subfolder
            search_subdirs = [folder_path, os.path.join(folder_path, "Beach_And_Backshore")]
            
            for subdir in search_subdirs:
                if not os.path.exists(subdir):
                    continue
                    
                # 1. Exact Name Match
                for fname in possible_filenames:
                    candidate = os.path.join(subdir, fname)
                    if os.path.exists(candidate):
                        print(f"[FOUND] Raw Source: {candidate}")
                        return candidate
                        
                # 2. Flexible Match (contains "beach_cliff_ground")
                all_las = glob.glob(os.path.join(subdir, "*beach_cliff_ground*.las"))
                if all_las:
                    # Pick the one that is most likely correct (max length usually implies full prefix)
                    best_match = max(all_las, key=len)
                    print(f"[FOUND] Raw Source (Glob match): {best_match}")
                    return best_match
            
    return None

def crop_and_save(in_path, out_path, bounds):
    """
    Crops LAS file to bounds and saves with ALL original fields (RGB).
    """
    if not in_path or not os.path.exists(in_path):
        print(f"  [MISSING] {in_path}")
        return

    try:
        # Read file
        with laspy.open(in_path) as f:
            las = f.read()

        # Check for RGB
        has_color = hasattr(las, 'red')
        print(f"  [READ] {os.path.basename(in_path)} | {len(las.points)} pts | RGB: {has_color}")

        # Crop
        minx, maxx, miny, maxy = bounds
        mask = (las.x >= minx) & (las.x <= maxx) & \
               (las.y >= miny) & (las.y <= maxy)
        
        points_kept = np.sum(mask)
        
        if points_kept == 0:
            print(f"  [SKIP] No points inside bounding box.")
            return

        # Create new LAS with SAME HEADER (preserves schema)
        out_las = laspy.LasData(las.header)
        out_las.points = las.points[mask]
        
        # Update header bounds
        out_las.header.min = [out_las.x.min(), out_las.y.min(), out_las.z.min()]
        out_las.header.max = [out_las.x.max(), out_las.y.max(), out_las.z.max()]
        
        out_las.write(out_path)
        print(f"  [SAVED] -> {os.path.basename(out_path)} ({points_kept} pts)")

    except Exception as e:
        print(f"  [ERROR] {e}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default=DEFAULT_LOC)
    parser.add_argument("--date", help="Override date (e.g. 20251117)", default=None)
    args = parser.parse_args()
    
    # 1. Setup
    os.makedirs(OUT_DIR, exist_ok=True)
    survey_csv = os.path.join(PROJECT_ROOT, "survey_lists", f"surveys_{args.location}.csv")
    results_base = os.path.join(PROJECT_ROOT, "results", args.location)
    
    # 2. Determine Date
    target_date = args.date
    if not target_date:
        # Try CSV first (prioritizes VMQLZ)
        target_date = get_latest_vmqlz_date(survey_csv)
    
    if not target_date:
        # Fallback to filesystem (just latest date found)
        target_date = get_latest_date_from_filesystem(results_base)
        
    if not target_date:
        print(f"[ERROR] Could not determine a valid survey date.")
        return

    print(f"\n--- Extracting RGB Subsets for {args.location} {target_date} ---")
    print(f"Output: {OUT_DIR}")
    print(f"Bounds: {BOUNDS}\n")

    # 3. Define Files to Extract
    tasks = [
        ("01_raw.las",      "raw",      None),
        ("02_cropped.las",  "pipeline", "cropped"),
        ("03_nobeach.las",  "pipeline", "noBeach"), # Case sensitive check needed
        ("03_nobeach.las",  "pipeline", "nobeach"), # Alternative casing
        ("04_noveg.las",    "pipeline", "noVeg"),
        ("04_noveg.las",    "pipeline", "noveg")
    ]
    
    processed_outputs = set()

    for out_name, source_type, subfolder in tasks:
        if out_name in processed_outputs:
            continue

        input_path = None
        
        if source_type == "raw":
            # Pass location to enforce check
            input_path = find_raw_file(target_date, args.location)
            if not input_path:
                print(f"  [WARN] Raw file search failed for date {target_date} location {args.location}")
        
        elif source_type == "pipeline":
            # Find pipeline file
            search_dir = os.path.join(results_base, subfolder)
            if os.path.exists(search_dir):
                candidates = glob.glob(os.path.join(search_dir, f"{target_date}*.las"))
                if candidates:
                    # If multiple match, pick shortest name (often cleanest) or largest size
                    input_path = candidates[0]
        
        if input_path and os.path.exists(input_path):
            full_out_path = os.path.join(OUT_DIR, out_name)
            crop_and_save(input_path, full_out_path, BOUNDS)
            processed_outputs.add(out_name)

    if not processed_outputs:
        print("\n[ERROR] No files were successfully processed.")
    else:
        print("\nExtraction Complete.")

if __name__ == "__main__":
    main()