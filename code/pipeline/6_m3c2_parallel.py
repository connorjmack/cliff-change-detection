#!/usr/bin/env python3
"""
Sequential M3C2 via CloudCompare (cross-platform) with Robust Reporting

Arguments:
  location      Survey location (e.g. SanElijo)
  --cc          Full path to CloudCompare executable (overrides default)
  --base-dir    Base directory for LidarProcessingCliffs (overrides defaults)
  --test        Number of survey pairs to process (default: all)
  --replace     Overwrite existing outputs; otherwise skip existing
  --single      Run single-threaded (no parallel processing)

Usage:
  # on Linux/macOS (requires Xvfb)
  xvfb-run --auto-servernum \
    --server-args="-screen 0 1024x768x24" \
    python3 m3c2_parallel_report.py Solana

  # on Windows
  python m3c2_parallel_report.py Solana --cc "C:\\Program Files\\CloudCompare\\CloudCompare.exe"
"""
import os
import glob
import argparse
import subprocess
import platform
import time
import csv
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_date(path):
    """
    Given a filename like "20171004_00590_00708_…", return "20171004".
    """
    return os.path.splitext(os.path.basename(path))[0].split("_", 1)[0]


def compute_m3c2_and_save_all(ref_path, cmp_path, params_file, base_output_dir, shift, cc_path, replace=False):
    """
    Runs CloudCompare CLI to compute M3C2 between ref_path and cmp_path.
    Returns a dictionary of detailed performance metrics.
    """
    start_time_epoch = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    date1 = extract_date(ref_path)
    date2 = extract_date(cmp_path)
    pair_name = f"{date1}_to_{date2}"
    pair_dir  = os.path.join(base_output_dir, pair_name)
    
    # Collect Input Metrics (cheap operation)
    try:
        ref_size = os.path.getsize(ref_path) / (1024 * 1024) # MB
        cmp_size = os.path.getsize(cmp_path) / (1024 * 1024) # MB
        total_input_mb = ref_size + cmp_size
    except:
        ref_size = cmp_size = total_input_mb = 0

    # Define expected outputs
    out_ref  = os.path.join(pair_dir, f"{date1}.las")
    out_cmp  = os.path.join(pair_dir, f"{date2}.las")
    out_m3c2 = os.path.join(pair_dir, f"{pair_name}_m3c2.las")
    save_list = f"{out_ref} {out_cmp} {out_m3c2}"

    # Initialize Result Dictionary
    result = {
        "pair": pair_name,
        "ref_date": date1,
        "cmp_date": date2,
        "status": "UNKNOWN",
        "start_time": start_dt,
        "end_time": None,
        "duration_sec": 0.0,
        "input_mb": round(total_input_mb, 2),
        "error_message": ""
    }

    # Check existence
    expected = [out_ref, out_cmp, out_m3c2]
    if not replace and all(os.path.exists(p) for p in expected):
        print(f"[SKIP] {pair_name} already exists")
        result["status"] = "SKIPPED"
        result["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return result

    os.makedirs(pair_dir, exist_ok=True)
    print(f"[M3C2] {pair_name}: running CloudCompare...")
    
    cmd = [
        cc_path,
        "-silent",
        "-auto_save", "off",
        "-c_export_fmt", "las",
        "-o", "-global_shift", *shift, ref_path,
        "-o", "-global_shift", *shift, cmp_path,
        "-M3C2", params_file,
        "-SAVE_CLOUDS", "FILE", save_list
    ]

    try:
        subprocess.run(cmd, check=True)
        duration = time.time() - start_time_epoch
        print(f"[OK] {pair_name} finished ({duration:.2f}s)")
        
        result["status"] = "SUCCESS"
        result["duration_sec"] = round(duration, 2)
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time_epoch
        print(f"[FAIL] {pair_name} CloudCompare error: {e}")
        
        result["status"] = "FAILED"
        result["duration_sec"] = round(duration, 2)
        result["error_message"] = f"Return Code {e.returncode}"
        
    except Exception as e:
        duration = time.time() - start_time_epoch
        print(f"[FAIL] {pair_name} Unexpected error: {e}")
        
        result["status"] = "ERROR"
        result["duration_sec"] = round(duration, 2)
        result["error_message"] = str(e)

    result["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return result


def write_robust_report(location, stats, report_dir, args_dict, total_script_time):
    """
    Generates:
      1. A machine-readable CSV inventory.
      2. A human-readable text summary with aggregates.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(report_dir, f"{location}_m3c2_inventory_{timestamp}.csv")
    txt_file = os.path.join(report_dir, f"{location}_m3c2_summary_{timestamp}.txt")
    
    # --- 1. CSV Output ---
    if stats:
        fieldnames = list(stats[0].keys())
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats)
    
    # --- 2. Calculate Aggregates ---
    total_pairs = len(stats)
    success = [s for s in stats if s['status'] == 'SUCCESS']
    skipped = [s for s in stats if s['status'] == 'SKIPPED']
    failed  = [s for s in stats if s['status'] in ['FAILED', 'ERROR']]
    
    avg_duration = sum(s['duration_sec'] for s in success) / len(success) if success else 0
    total_input_mb = sum(s['input_mb'] for s in stats)
    
    # --- 3. Text Report ---
    with open(txt_file, "w") as f:
        f.write(f"M3C2 PROCESSING REPORT: {location}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Platform:     {platform.system()} ({platform.release()})\n")
        f.write(f"Script Usage: {args_dict}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Wall Time:  {total_script_time:.2f} seconds\n")
        f.write(f"Total Data Size:  {total_input_mb/1024:.2f} GB (approx inputs)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Pairs: {total_pairs}\n")
        f.write(f"  [+] Success: {len(success)}\n")
        f.write(f"  [-] Skipped: {len(skipped)}\n")
        f.write(f"  [!] Failed:  {len(failed)}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Average Processing Time (Successes): {avg_duration:.2f} sec/pair\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DETAILED LOG:\n")
        f.write(f"{'PAIR NAME':<30} | {'STATUS':<10} | {'TIME(s)':<8} | {'INPUT(MB)':<9} | {'NOTE'}\n")
        f.write("-" * 90 + "\n")
        
        for s in sorted(stats, key=lambda x: x['pair']):
            note = s['error_message'] if s['error_message'] else ""
            f.write(f"{s['pair']:<30} | {s['status']:<10} | {s['duration_sec']:<8.1f} | {s['input_mb']:<9.1f} | {note}\n")
            
    print(f"\n[REPORT] Reports generated:")
    print(f"  -> CSV Inventory: {csv_file}")
    print(f"  -> Text Summary:  {txt_file}")


# --- GLOBAL WRAPPER FUNCTION FOR MULTIPROCESSING ---
def safe_execute(ref, cmp, params_file, output_dir, shift, cc_path, replace):
    """
    Wrapper for compute_m3c2_and_save_all to handle exceptions safely
    and ensure multiprocessing picklability.
    """
    try:
        return compute_m3c2_and_save_all(ref, cmp, params_file, output_dir, shift, cc_path, replace)
    except Exception as e:
        return {
            "pair": f"{extract_date(ref)}_to_{extract_date(cmp)}",
            "status": "CRASH",
            "error_message": str(e),
            "duration_sec": 0,
            "input_mb": 0,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def main():
    script_start = time.time()
    
    parser = argparse.ArgumentParser(description="Sequential M3C2 via CloudCompare with Robust Reporting")
    parser.add_argument("location", help="Survey location (e.g. SanElijo)")
    parser.add_argument("--cc",      help="CloudCompare executable path")
    parser.add_argument("--base-dir",help="Base directory for LidarProcessingCliffs")
    parser.add_argument("--test",    type=int, help="Only process this many pairs")
    parser.add_argument("--replace", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--single", action="store_true", help="Run single-threaded")
    args = parser.parse_args()

    # --- Path Configuration ---
    system = platform.system()
    if system == "Darwin":
        default_base = r"/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
        default_cc   = r"/Applications/CloudCompare.app/Contents/MacOS/CloudCompare"
    elif system == "Linux":
        default_base = r"/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
        default_cc   = r"/usr/local/bin/CloudCompare"
    elif system == "Windows":
        default_base = r"Z:\\LiDAR\\LidarProcessing\\LidarProcessingCliffs"
        default_cc   = r"C:\\Program Files\\CloudCompare\\CloudCompare.exe"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    base_dir = args.base_dir or default_base
    cc_path  = args.cc or default_cc

    # Global shift settings
    shift = {
        "SanElijo": ("-473000", "-3653000", "0"),
        "Encinitas": ("-472000", "-3655000", "0"),
        "Solana": ("-475000", "-3650000", "0"),
        "Torrey": ("-475000", "-3650000", "0")
    }.get(args.location, ("-473000", "-3650000", "0"))

    # I/O Paths
    las_dir = os.path.join(base_dir, "results", args.location, "noveg")
    
    if args.location == "Torrey":
        params_file = os.path.join(base_dir, "utilities", "m3c2_params", "m3c2_params_torrey.txt")
    else:
        params_file = os.path.join(base_dir, "utilities", "m3c2_params", "new_params.txt")

    # Output Folder with today's pipeline run ID
    pipeline_id = datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join(base_dir, "results", args.location, "m3c2", f"pipeline_run_{pipeline_id}")
    
    # Report Directory
    report_dir = os.path.join(base_dir, "validation", "m3c2")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # --- File Discovery ---
    all_las = sorted(glob.glob(os.path.join(las_dir, "*.las")))
    if len(all_las) < 2:
        print("[ERROR] Need at least 2 LAS files for sequential M3C2")
        return

    pairs = list(zip(all_las, all_las[1:]))
    if args.test:
        pairs = pairs[:args.test]

    print(f"[MAIN] Processing {len(pairs)} pairs for location: {args.location}")
    print(f"[CONFIG] Output Dir: {output_dir}")
    
    # --- Execution ---
    stats_results = []

    if args.single:
        # Single-threaded
        print("[MODE] Single-threaded execution")
        for ref, cmp in pairs:
            # We can still use the safe_execute wrapper for consistency
            stats_results.append(safe_execute(ref, cmp, params_file, output_dir, shift, cc_path, args.replace))
    else:
        # Parallel
        n_workers = min(4, os.cpu_count() or 1)
        print(f"[MODE] Parallel execution ({n_workers} workers)")
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            # Use safe_execute (defined at module level) so it can be pickled
            futures = {
                exe.submit(
                    safe_execute, 
                    ref, cmp, params_file, output_dir, shift, cc_path, args.replace
                ): (ref, cmp) for ref, cmp in pairs
            }
            
            for fut in as_completed(futures):
                stats_results.append(fut.result())

    # --- Reporting ---
    print("[DONE] All M3C2 runs complete. Generating robust reports...")
    
    # Clean args for logging
    args_log = {k: v for k, v in vars(args).items() if v is not None}
    
    write_robust_report(
        location=args.location, 
        stats=stats_results, 
        report_dir=report_dir, 
        args_dict=args_log, 
        total_script_time=time.time() - script_start
    )

if __name__ == "__main__":
    main()