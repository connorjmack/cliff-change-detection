#!/usr/bin/env python3
"""
Generate a CSV inventory of all noveg LAS files across all locations.

Usage:
    python3 list_noveg_files.py                    # Uses default Linux path
    python3 list_noveg_files.py --base-dir /path   # Custom base directory
    python3 list_noveg_files.py --output my.csv    # Custom output file
"""
import os
import glob
import argparse
import platform
import csv
from datetime import datetime


LOCATIONS = ["DelMar", "Solana", "Encinitas", "SanElijo", "Torrey", "Blacks"]


def get_default_base_dir():
    """Return OS-appropriate base directory."""
    system = platform.system()
    if system == "Darwin":
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    elif system == "Linux":
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    elif system == "Windows":
        return r"Z:\LiDAR\LidarProcessing\LidarProcessingCliffs"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV inventory of all noveg LAS files"
    )
    parser.add_argument(
        "--base-dir",
        help="Base directory for LidarProcessingCliffs (overrides OS default)"
    )
    parser.add_argument(
        "--output", "-o",
        default="noveg_inventory.csv",
        help="Output CSV filename (default: noveg_inventory.csv)"
    )
    parser.add_argument(
        "--linux-paths",
        action="store_true",
        help="Force output paths to use Linux format regardless of current OS"
    )
    args = parser.parse_args()

    base_dir = args.base_dir or get_default_base_dir()
    linux_base = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

    print(f"Scanning base directory: {base_dir}")

    all_files = []

    for location in LOCATIONS:
        noveg_dir = os.path.join(base_dir, "results", location, "noveg")

        if not os.path.isdir(noveg_dir):
            print(f"  [{location}] Directory not found: {noveg_dir}")
            continue

        las_files = sorted(glob.glob(os.path.join(noveg_dir, "*.las")))
        print(f"  [{location}] Found {len(las_files)} files")

        for filepath in las_files:
            filename = os.path.basename(filepath)

            # Convert to Linux path if requested
            if args.linux_paths:
                linux_path = filepath.replace(base_dir, linux_base)
                # Normalize separators for Linux
                linux_path = linux_path.replace("\\", "/")
            else:
                linux_path = filepath

            all_files.append({
                "location": location,
                "filename": filename,
                "path": linux_path
            })

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["location", "filename", "path"])
        writer.writeheader()
        writer.writerows(all_files)

    print(f"\nWrote {len(all_files)} files to {args.output}")


if __name__ == "__main__":
    main()
