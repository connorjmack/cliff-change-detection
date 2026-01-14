#!/usr/bin/env python3
"""
make_survey_lists.py

Usage:
    python3 make_survey_lists.py --location SanElijo
    python3 make_survey_lists.py --all
"""

import os
import csv
import sys
import glob
import numpy as np
import platform
import argparse

# === 0. Detect OS for Paths ===
system = platform.system()
if system == "Darwin":
    # MacOS
    ROOT_LIDAR = "/Volumes/group/LiDAR"
else:
    # Linux / Cluster
    ROOT_LIDAR = "/project/group/LiDAR"

print(f"[INFO] Detected OS: {system}. Using root: {ROOT_LIDAR}")

# === 1. Instrument root paths (Dynamic) ===
instrument_paths = {
    "MiniRanger_Truck": os.path.join(ROOT_LIDAR, "MiniRanger_Truck/LiDAR_Processed_Level2"),
    "MiniRanger_ATV"  : os.path.join(ROOT_LIDAR, "MiniRanger_ATV/LiDAR_Processed_Level2"),
    "VMQLZ_Truck"     : os.path.join(ROOT_LIDAR, "VMQLZ_Truck/LiDAR_Processed_Level2"),
    "VMZ2000_Truck"   : os.path.join(ROOT_LIDAR, "VMZ2000_Truck/LiDAR_Processed_Level2"),
}

# === 2. MOP-line ranges by location ===
mop_ranges = {
    "DelMar"      : [595, 620],
    "Solana"      : [637, 666],
    "Encinitas"   : [708, 764],
    "SanElijo"    : [683, 708],
    "Torrey"      : [567, 581],
    "Blacks"      : [520, 567]
}

def ensure_mac_path(path):
    """Normalize /project paths to /Volumes for consistency."""
    if path.startswith("/project/group/LiDAR"):
        return path.replace("/project/group/LiDAR", "/Volumes/group/LiDAR")
    return path

def find_target_las_files(directory):
    """
    Return list of *_beach_cliff_ground.las files under a survey directory.
    """
    return sorted([
        f for f in glob.glob(os.path.join(directory, "**", "*.las"), recursive=True)
        if f.lower().endswith("_beach_cliff_ground.las")
    ])

def export_surveys(location):
    """
    Walk each instrument folder, find subdirs with at least 2/3 of MOP lines overlapping the location range,
    confirm they contain a *_beach_cliff_ground.las, sort all hits by date, and write surveys_<location>.csv.
    """
    if location not in mop_ranges:
        print(f"❌ Error: Unknown location '{location}'. Skipping.")
        return

    # Get the target MOP range for the requested location
    min_line, max_line = mop_ranges[location]

    # Output directory setup
    out_dir = os.path.join(ROOT_LIDAR, "LidarProcessing/LidarProcessingCliffs/survey_lists")
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.join(out_dir, f"surveys_{location}.csv")

    print(f"\n{'='*70}")
    print(f"Processing {location} (MOP range: {min_line}-{max_line})")
    print(f"{'='*70}")

    # Counters for tracking
    total_scanned = 0
    total_matched = 0
    skipped_not_dir = 0
    skipped_naming = 0
    skipped_overlap = 0
    skipped_no_las = 0

    # 1) collect all matching rows with >= 2/3 overlap
    rows = []
    for method, root in instrument_paths.items():
        if not os.path.isdir(root):
            print(f"⚠️  Skipping {method} (path not found: {root})")
            continue

        print(f"\n📂 Scanning instrument: {method}")
        print(f"   Path: {root}")

        # Get list of subdirectories to scan
        try:
            subdirs = os.listdir(root)
        except PermissionError:
            print(f"   ❌ Permission denied - cannot access directory")
            continue

        instrument_matches = 0

        for name in subdirs:
            subdir = os.path.join(root, name)
            total_scanned += 1

            # Check 1: Must be a directory
            if not os.path.isdir(subdir):
                skipped_not_dir += 1
                continue

            # Check 2: Must follow naming convention YYYYMMDD_MOP1_MOP2_...
            parts = name.split("_")
            if len(parts) < 3:
                skipped_naming += 1
                continue

            try:
                date_str = parts[0]
                mop1 = int(parts[1])
                mop2 = int(parts[2])
            except ValueError:
                skipped_naming += 1
                continue

            # Check 3: Calculate Mathematical Overlap
            overlap = min(mop2, max_line) - max(mop1, min_line)

            # Calculate required overlap (2/3 of the TARGET location's length)
            two_thirds = (max_line - min_line) * 2/3

            if overlap < two_thirds:
                skipped_overlap += 1
                continue

            # Verify a beach_cliff_ground LAS exists; skip otherwise
            las_files = find_target_las_files(subdir)
            if not las_files:
                skipped_no_las += 1
                print(f"   ⚠️  {name}: No *_beach_cliff_ground.las file found")
                continue

            mac_path = ensure_mac_path(subdir)
            rows.append({
                "path":   mac_path,  # full survey directory path
                "date":   date_str,
                "MOP1":   mop1,
                "MOP2":   mop2,
                "beach":  location,
                "method": method
            })
            instrument_matches += 1
            total_matched += 1
            print(f"   ✓ {date_str} (MOP {mop1}-{mop2}) - MATCH")

        print(f"   → Found {instrument_matches} matching surveys in {method}")

    # 2) sort by date (YYYYMMDD strings sort lexically)
    rows.sort(key=lambda r: r["date"])

    # 3) write out in order
    with open(out_name, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile,
            fieldnames=["path","date","MOP1","MOP2","beach","method"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY for {location}")
    print(f"{'='*70}")
    print(f"Total items scanned:        {total_scanned}")
    print(f"Surveys matched:            {total_matched}")
    print(f"Skipped (not directory):    {skipped_not_dir}")
    print(f"Skipped (naming format):    {skipped_naming}")
    print(f"Skipped (MOP overlap):      {skipped_overlap}")
    print(f"Skipped (no LAS file):      {skipped_no_las}")
    print(f"\n✅ Wrote {out_name} with {len(rows)} surveys.")
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate survey lists based on MOP overlap.")
    parser.add_argument("--location", type=str, help="Specific location to process (e.g., Solana)")
    parser.add_argument("--all", action="store_true", help="Process all locations defined in mop_ranges")
    
    args = parser.parse_args()

    if args.location:
        export_surveys(args.location)
    elif args.all:
        for loc in mop_ranges.keys():
            export_surveys(loc)
    else:
        # Default behavior if nothing passed: Run all
        print("No arguments provided. Defaulting to processing ALL locations.")
        for loc in mop_ranges.keys():
            export_surveys(loc)

if __name__ == "__main__":
    main()
