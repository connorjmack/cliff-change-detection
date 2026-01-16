#!/usr/bin/env python3
"""
run_daily.py

The Master Pipeline Orchestrator.
UPDATED: Forces unbuffered output (-u) so logs appear immediately.
"""

import os
import sys
import subprocess
import argparse
import platform
from datetime import datetime

# === CONFIGURATION ===
SYSTEM = platform.system()
if SYSTEM == "Darwin":
    ROOT_DIR = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    IS_LINUX = False
else:
    ROOT_DIR = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    IS_LINUX = True

CODE_DIR = os.path.join(ROOT_DIR, "code", "pipeline")
REPORT_DIR = os.path.join(CODE_DIR, "daily_reports")

# Known locations to allow validation
ALL_LOCATIONS = ["DelMar", "Solana", "Encinitas", "SanElijo", "Torrey", "Blacks"]

PIPELINE_STEPS = [
    (2, "2_crop_files_parallel.py"),
    (3, os.path.join("tests", "audits", "2_audit_cropping.py")),
    (4, "4_remove_beach_parallel.py"),
    (5, "5_remove_veg_parallel.py"), # Requires CloudCompare (Headless)
    (6, "6_m3c2_parallel.py"),       # Requires CloudCompare (Headless)
    (7, "7_dbscan_parallel.py"),
    (8, "8_make_grids.py"),
    (9, "9_clean_fill_grids.py")
]

HEADLESS_STEPS = [5, 6]

def get_today_report_path():
    today = datetime.now().strftime("%Y%m%d")
    return os.path.join(REPORT_DIR, f"daily_report_{today}.txt")

def run_step_0_detection(report_file):
    script_name = "1_update_survey_lists.py"
    script_path = os.path.join(CODE_DIR, script_name)
    
    header = (f"\n{'-'*60}\n"
              f"Step 0: DETECTION ({script_name})\n"
              f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
              f"{'-'*60}\n")
    
    print(header.strip())
    report_file.write(header)
    report_file.flush()

    if not os.path.exists(script_path):
        msg = f"[CRITICAL] Step 0 script missing at {script_path}\n"
        print(msg)
        report_file.write(msg)
        return False

    try:
        # ADDED "-u" HERE
        process = subprocess.Popen(
            ["python3", "-u", script_path, "--console-only"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            print(line, end='') 
            report_file.write(line)
        
        process.wait()

        if process.returncode != 0:
            err = f"\n[FAILURE] Step 0 exited with code {process.returncode}\n"
            print(err)
            report_file.write(err)
            return False

    except Exception as e:
        err = f"\n[CRASH] Step 0 failed: {e}\n"
        print(err)
        report_file.write(err)
        return False
        
    return True

def get_locations_from_report(report_path):
    if not os.path.exists(report_path):
        return []

    active_locations = set()
    with open(report_path, 'r') as f:
        for line in f:
            if line.startswith("NEW:"):
                parts = line.split("|")
                if len(parts) > 1:
                    loc = parts[0].replace("NEW:", "").strip()
                    if loc in ALL_LOCATIONS:
                        active_locations.add(loc)
    return list(active_locations)

def run_processing_step(step_num, script_name, location, report_file):
    if os.path.isabs(script_name):
        script_path = script_name
        display_name = os.path.basename(script_name)
    elif os.sep in script_name:
        script_path = os.path.join(ROOT_DIR, script_name)
        display_name = script_name
    else:
        script_path = os.path.join(CODE_DIR, script_name)
        display_name = script_name
    
    header = (f"\n{'-'*60}\n"
              f"Step {step_num}: {display_name} [{location}]\n"
              f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
              f"{'-'*60}\n")
    print(header.strip())
    report_file.write(header)
    report_file.flush()

    if not os.path.exists(script_path):
        msg = f"[ERROR] Script missing: {script_path}\n"
        print(msg)
        report_file.write(msg)
        return False

    # --- Build Command ---
    cmd = []

    # 1. Headless Wrapper (xvfb) for CloudCompare steps on Linux
    if IS_LINUX and step_num in HEADLESS_STEPS:
        # Note: xvfb-run does not accept '-u' for python directly in the wrapper easily
        # but python typically detects the pipe. We will add -u to the python call inside.
        cmd = ["xvfb-run", "-a", "python3", "-u", script_path]
    else:
        # ADDED "-u" HERE
        cmd = ["python3", "-u", script_path]

    # 2. Add Location Argument
    if step_num == 2:
        cmd.extend(["--location", location])
    elif step_num == 3:
        pass 
    else:
        cmd.append(location)

    # 3. Step-Specific Flags
    if step_num == 9:
        cmd.extend(["--resolution", "25cm", "--erosion", "--deposition"])

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            print(line, end='') # Live print enabled
            report_file.write(line)
        
        process.wait()

        if process.returncode != 0:
            err = f"\n[FAILURE] Step {step_num} exited with code {process.returncode}\n"
            print(err)
            report_file.write(err)
            return False

    except Exception as e:
        err = f"\n[CRASH] Step {step_num} failed: {e}\n"
        print(err)
        report_file.write(err)
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description="Master Daily Pipeline")
    parser.add_argument("--force-all", action="store_true", help="Skip detection and force run ALL locations")
    args = parser.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = get_today_report_path()
    
    print(f"[MASTER] Pipeline Started. Logging to: {report_path}")

    with open(report_path, "a") as f:
        f.write(f"\n\n{'='*80}\n")
        f.write(f"DAILY PIPELINE START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")

        # --- PHASE 1: DETECTION ---
        if args.force_all:
            print("[MASTER] Skipping detection (Force All Mode)")
            f.write("[MASTER] Skipping Step 0 (Force All Mode Active)\n")
            target_locations = ALL_LOCATIONS
        else:
            print("[MASTER] Running Step 0 (Detection)...")
            success0 = run_step_0_detection(f)
            if not success0:
                print("[MASTER] Step 0 Failed. Aborting.")
                return

            target_locations = get_locations_from_report(report_path)
            
            if not target_locations:
                msg = "[MASTER] Step 0 finished. No new surveys found. Exiting.\n"
                print(msg.strip())
                f.write(msg)
                return

        # --- PHASE 2: PROCESSING ---
        msg = f"[MASTER] New data detected for: {', '.join(target_locations)}\n"
        print(msg.strip())
        f.write(msg)

        for loc in target_locations:
            f.write(f"\n>>> STARTING LOCATION: {loc} <<<\n")
            print(f"\n>>> PROCESSING: {loc} <<<")
            
            for step_num, script_name in PIPELINE_STEPS:
                success = run_processing_step(step_num, script_name, loc, f)
                if not success:
                    fail_msg = f"\n>>> ABORTING {loc} DUE TO ERROR IN STEP {step_num} <<<\n"
                    print(fail_msg)
                    f.write(fail_msg)
                    break 
            else:
                f.write(f"\n>>> COMPLETED LOCATION: {loc} <<<\n")

        f.write(f"\nPIPELINE FINISHED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"[MASTER] All tasks finished. Check report for details: {os.path.basename(report_path)}")

if __name__ == "__main__":
    main()
