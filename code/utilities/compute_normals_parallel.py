#!/usr/bin/env python3
"""
Compute normals for point clouds using CloudCompare CLI.

Arguments:
    input_csv       CSV file with 'path' column containing LAS file paths
    --cc            Full path to CloudCompare executable (overrides default)
    --output-dir    Directory to write output files (default: save in place)
    --suffix        Suffix for output files (default: _normals)
    --replace       Overwrite existing outputs; otherwise skip existing
    --single        Run single-threaded (no parallel processing)
    --test          Number of files to process (default: all)
    --n_jobs        Number of parallel workers (default: 4)

Normal Calculation Parameters:
    --radius        Radius for normal computation in meters (default: 0.5 for 1m diameter)
    --mst           Number of MST neighbors for orientation (default: 12)

Usage:
    # On Linux (requires Xvfb for headless)
    xvfb-run -a python3 compute_normals_parallel.py noveg_inventory.csv

    # Save to test directory
    python3 compute_normals_parallel.py noveg_inventory.csv --output-dir /tmp/normals_test

    # Save in place (alongside original files)
    python3 compute_normals_parallel.py noveg_inventory.csv

    # Custom parameters
    python3 compute_normals_parallel.py noveg_inventory.csv --radius 0.5 --mst 12
"""
import os
import csv
import argparse
import subprocess
import platform
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


# Global coordinate shifts for large UTM values (must match other pipeline steps)
GLOBAL_SHIFTS = {
    "SanElijo":   ("-473000", "-3653000", "0"),
    "Encinitas":  ("-472000", "-3655000", "0"),
    "Solana":     ("-475000", "-3650000", "0"),
    "Torrey":     ("-475000", "-3650000", "0"),
    "DelMar":     ("-473000", "-3650000", "0"),
    "Blacks":     ("-473000", "-3650000", "0"),
}
DEFAULT_SHIFT = ("-473000", "-3650000", "0")


def detect_location(filepath):
    """Attempt to detect location from filepath for global shift."""
    for location in GLOBAL_SHIFTS:
        if location.lower() in filepath.lower():
            return location
    return None


def get_global_shift(filepath):
    """Get appropriate global shift based on file path."""
    location = detect_location(filepath)
    if location:
        return GLOBAL_SHIFTS[location]
    return DEFAULT_SHIFT


def compute_normals(input_path, output_path, cc_path, radius, mst_neighbors, replace=False):
    """
    Compute normals for a single point cloud using CloudCompare.

    Returns a dictionary with processing metrics.
    """
    start_time = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    filename = os.path.basename(input_path)

    result = {
        "input_file": input_path,
        "output_file": output_path,
        "status": "UNKNOWN",
        "start_time": start_dt,
        "end_time": None,
        "duration_sec": 0.0,
        "error_message": ""
    }

    # Check if input exists
    if not os.path.exists(input_path):
        result["status"] = "ERROR"
        result["error_message"] = "Input file not found"
        result["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[NORMALS] {filename}... ERROR (file not found)")
        return result

    # Check if output exists and skip if not replacing
    if not replace and os.path.exists(output_path):
        print(f"[NORMALS] {filename}... SKIPPED (already exists)")
        result["status"] = "SKIPPED"
        result["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return result

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get global shift for this file
    shift = get_global_shift(input_path)

    print(f"[NORMALS] {filename}...", end="", flush=True)

    # Build CloudCompare command
    # -OCTREE_NORMALS <radius> computes normals using local surface model
    # -ORIENT_NORMS_MST <k> orients normals using Minimum Spanning Tree with k neighbors
    cmd = [
        cc_path,
        "-SILENT",
        "-AUTO_SAVE", "OFF",
        "-C_EXPORT_FMT", "LAS",
        "-O", "-GLOBAL_SHIFT", *shift, input_path,
        "-OCTREE_NORMALS", str(radius),
        "-ORIENT_NORMS_MST", str(mst_neighbors),
        "-SAVE_CLOUDS", "FILE", output_path
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        duration = time.time() - start_time
        print(f" OK ({duration:.2f}s)")

        result["status"] = "SUCCESS"
        result["duration_sec"] = round(duration, 2)

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f" FAILED (Return Code {e.returncode})")

        result["status"] = "FAILED"
        result["duration_sec"] = round(duration, 2)
        result["error_message"] = f"Return Code {e.returncode}"

    except Exception as e:
        duration = time.time() - start_time
        print(f" ERROR ({str(e)})")

        result["status"] = "ERROR"
        result["duration_sec"] = round(duration, 2)
        result["error_message"] = str(e)

    result["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return result


def safe_compute_normals(input_path, output_path, cc_path, radius, mst_neighbors, replace):
    """Wrapper for multiprocessing safety."""
    try:
        return compute_normals(input_path, output_path, cc_path, radius, mst_neighbors, replace)
    except Exception as e:
        return {
            "input_file": input_path,
            "output_file": output_path,
            "status": "CRASH",
            "error_message": str(e),
            "duration_sec": 0,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def read_input_csv(csv_path):
    """Read input CSV and return list of file paths."""
    paths = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "path" in row:
                paths.append(row["path"])
            elif "Path" in row:
                paths.append(row["Path"])
            else:
                # If no 'path' column, try first column
                first_key = list(row.keys())[0]
                paths.append(row[first_key])
    return paths


def write_report(results, report_path, args_dict, total_time):
    """Write processing report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV inventory
    csv_file = report_path.replace(".txt", f"_{timestamp}.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Summary stats
    total = len(results)
    success = [r for r in results if r["status"] == "SUCCESS"]
    skipped = [r for r in results if r["status"] == "SKIPPED"]
    failed = [r for r in results if r["status"] in ["FAILED", "ERROR", "CRASH"]]

    avg_duration = sum(r["duration_sec"] for r in success) / len(success) if success else 0

    # Text summary
    txt_file = report_path.replace(".csv", f"_{timestamp}.txt")
    with open(txt_file, "w") as f:
        f.write("NORMAL COMPUTATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Platform:     {platform.system()} ({platform.release()})\n")
        f.write(f"Parameters:   {args_dict}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Wall Time:  {total_time:.2f} seconds\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Files: {total}\n")
        f.write(f"  [+] Success: {len(success)}\n")
        f.write(f"  [-] Skipped: {len(skipped)}\n")
        f.write(f"  [!] Failed:  {len(failed)}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Average Processing Time (Successes): {avg_duration:.2f} sec/file\n")
        f.write("=" * 60 + "\n")

    print(f"\n[REPORT] Reports generated:")
    print(f"  -> CSV: {csv_file}")
    print(f"  -> TXT: {txt_file}")


def main():
    script_start = time.time()

    parser = argparse.ArgumentParser(
        description="Compute normals for point clouds using CloudCompare"
    )
    parser.add_argument("input_csv", help="CSV file with 'path' column")
    parser.add_argument("--cc", help="CloudCompare executable path")
    parser.add_argument("--output-dir", help="Output directory (default: save in place)")
    parser.add_argument("--suffix", default="_normals", help="Output filename suffix")
    parser.add_argument("--replace", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--single", action="store_true", help="Run single-threaded")
    parser.add_argument("--test", type=int, help="Only process N files")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel workers")

    # Normal computation parameters
    parser.add_argument("--radius", type=float, default=0.5,
                        help="Radius for normal computation (default: 0.5 for 1m diameter)")
    parser.add_argument("--mst", type=int, default=12,
                        help="MST neighbors for orientation (default: 12)")

    args = parser.parse_args()

    # Determine CloudCompare path
    system = platform.system()
    if args.cc:
        cc_path = args.cc
    elif system == "Darwin":
        cc_path = "/Applications/CloudCompare.app/Contents/MacOS/CloudCompare"
    elif system == "Linux":
        cc_path = "/usr/local/bin/CloudCompare"
    elif system == "Windows":
        cc_path = r"C:\Program Files\CloudCompare\CloudCompare.exe"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    # Read input paths
    input_paths = read_input_csv(args.input_csv)
    if args.test:
        input_paths = input_paths[:args.test]

    print(f"[CONFIG] CloudCompare: {cc_path}")
    print(f"[CONFIG] Radius: {args.radius}m (diameter: {args.radius * 2}m)")
    print(f"[CONFIG] MST neighbors: {args.mst}")
    print(f"[CONFIG] Output: {'in place' if not args.output_dir else args.output_dir}")
    print(f"[MAIN] Processing {len(input_paths)} files")

    # Build output paths
    tasks = []
    for input_path in input_paths:
        if args.output_dir:
            # Save to test directory with same filename + suffix
            basename = os.path.basename(input_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(args.output_dir, f"{name}{args.suffix}{ext}")
        else:
            # Save in place with suffix
            dirname = os.path.dirname(input_path)
            basename = os.path.basename(input_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(dirname, f"{name}{args.suffix}{ext}")

        tasks.append((input_path, output_path))

    # Process files
    results = []

    if args.single:
        print("[MODE] Single-threaded execution")
        for input_path, output_path in tasks:
            result = safe_compute_normals(
                input_path, output_path, cc_path,
                args.radius, args.mst, args.replace
            )
            results.append(result)
    else:
        n_workers = min(args.n_jobs, os.cpu_count() or 1)
        print(f"[MODE] Parallel execution ({n_workers} workers)")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    safe_compute_normals,
                    input_path, output_path, cc_path,
                    args.radius, args.mst, args.replace
                ): (input_path, output_path)
                for input_path, output_path in tasks
            }

            for future in as_completed(futures):
                results.append(future.result())

    # Generate report
    total_time = time.time() - script_start
    print(f"\n[DONE] Processed {len(results)} files in {total_time:.2f}s")

    report_dir = args.output_dir or os.path.dirname(args.input_csv) or "."
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "normals_report.csv")

    args_log = {
        "radius": args.radius,
        "mst": args.mst,
        "replace": args.replace,
        "output_dir": args.output_dir
    }
    write_report(results, report_path, args_log, total_time)


if __name__ == "__main__":
    main()
