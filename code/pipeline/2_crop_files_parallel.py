#!/usr/bin/env python3
"""
crop_files_parallel.py

Crops LAS files to MOP lines and generates a performance report.

Usage:
    python3 crop_files_parallel.py --location SanElijo [--replace]
"""

import os
import glob
import json
import csv
import argparse
import time
import platform
import math
import pdal
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from shapely.geometry import Polygon
from pyproj import Transformer
import xml.etree.ElementTree as ET

# === THREADING CONTROLS ===
os.environ["OMP_NUM_THREADS"]      = str(3)
os.environ["OPENBLAS_NUM_THREADS"] = str(3)
os.environ["MKL_NUM_THREADS"]      = str(3)

# === PATH SETUP (Cross-Platform) ===
SYSTEM_OS = platform.system()
if SYSTEM_OS == "Darwin":  # macOS
    BASE_DIR = "/Volumes/group/LiDAR"
else:  # Linux / Windows default
    BASE_DIR = "/project/group/LiDAR"

MOP_KML         = os.path.join(BASE_DIR, "MOPLines/MOPs_SD_County.kml")
PROJECT_ROOT    = os.path.join(BASE_DIR, "LidarProcessing/LidarProcessingCliffs")
SURVEY_LIST_DIR = os.path.join(PROJECT_ROOT, "survey_lists")
RESULTS_BASE    = os.path.join(PROJECT_ROOT, "results")
CROPBOX_DIR     = os.path.join(PROJECT_ROOT, "utilities")

# === MOP INDEX RANGES BY LOCATION ===
mop_ranges = {
    "DelMar":      (595, 620),
    "Solana":      (637, 666),
    "Encinitas":   (708, 764),
    "SanElijo":    (683, 708),
    "Torrey":      (567, 581),
    "Blacks":      (520, 567)
}

# === PDAL SETTINGS ===
SAMPLE_RADIUS = 0.05    # 5 cm
ALONG_BUFFER  = 500.0   # meters along-track extension


def extend_line(p1, p2, along):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = math.hypot(dx, dy)
    if L == 0:
        return [p1, p2]
    ux, uy = dx / L, dy / L
    return [
        (p1[0] - ux * along, p1[1] - uy * along),
        (p2[0] + ux * along, p2[1] + uy * along),
    ]


def load_or_create_crop_polygon(location: str, min_mop: int, max_mop: int) -> Polygon:
    cropbox_path = os.path.join(CROPBOX_DIR, f"{location}_cropbox.txt")

    if os.path.exists(cropbox_path):
        # print(f"📥 Loading existing crop polygon from: {cropbox_path}")
        with open(cropbox_path) as f:
            coords = [tuple(map(float, line.strip().split()[:2])) for line in f if line.strip()]
        return Polygon(coords)

    print(f"📐 No existing cropbox found. Building from MOP lines...")
    
    # Check if KML exists
    if not os.path.exists(MOP_KML):
        raise FileNotFoundError(f"MOP KML not found at: {MOP_KML}")

    # --- Parse KML and extract MOP lines ---
    ns = {"ns": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(MOP_KML)
    lines = {}
    for pm in tree.findall(".//ns:Placemark", ns):
        nm = pm.find("ns:name", ns)
        cr = pm.find(".//ns:coordinates", ns)
        if nm is None or cr is None or not nm.text or not cr.text:
            continue
        name = nm.text.strip()
        if not name.startswith("MOP_"):
            continue
        try:
            idx = int(name.split("_")[1])
        except ValueError:
            continue
        pts = cr.text.strip().split()
        if len(pts) < 2:
            continue
        try:
            lon1, lat1, _ = map(float, pts[0].split(","))
            lon2, lat2, _ = map(float, pts[1].split(","))
        except ValueError:
            continue
        lines[idx] = ((lon1, lat1), (lon2, lat2))

    if min_mop not in lines or max_mop not in lines:
        raise ValueError(f"MOP {min_mop} or {max_mop} not found in KML file.")

    # --- Build UTM polygon from MOP lines ---
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True)
    l1 = [tf.transform(*pt) for pt in lines[min_mop]]
    l2 = [tf.transform(*pt) for pt in lines[max_mop]]
    l1_ext = extend_line(*l1, ALONG_BUFFER)
    l2_ext = extend_line(*l2, ALONG_BUFFER)
    pts = [l1_ext[0], l1_ext[1], l2_ext[1], l2_ext[0]]
    polygon = Polygon(pts + [pts[0]])

    # --- Save polygon to file ---
    os.makedirs(CROPBOX_DIR, exist_ok=True)
    with open(cropbox_path, "w") as f:
        for x, y in polygon.exterior.coords[:-1]:
            f.write(f"{x:.6f} {y:.6f} 0.0\n")
    print(f"💾 Saved new crop polygon to: {cropbox_path}")

    return polygon


def _worker(task):
    """
    Worker function to process a single LAS file.
    Returns a dictionary of statistics for the report.
    """
    poly_wkt, las_in, las_out, replace = task
    start_time = time.time()
    
    if not replace and os.path.exists(las_out):
        return {
            "file": os.path.basename(las_in),
            "status": "skipped",
            "duration": 0.0,
            "points": 0
        }

    try:
        pipeline = {
            "pipeline": [
                {"type": "readers.las", "filename": las_in},
                {"type": "filters.crop", "polygon": poly_wkt},
                {"type": "filters.sample", "radius": SAMPLE_RADIUS},
                {"type": "writers.las", "filename": las_out},
            ]
        }
        p = pdal.Pipeline(json.dumps(pipeline))
        count = p.execute()
        duration = time.time() - start_time
        return {
            "file": os.path.basename(las_in),
            "status": "processed",
            "duration": duration,
            "points": count
        }
    except Exception as e:
        return {
            "file": os.path.basename(las_in),
            "status": "error",
            "error_msg": str(e),
            "duration": time.time() - start_time,
            "points": 0
        }


def write_report(location, stats, total_duration, out_dir):
    """Generates a summary text file in the pipeline_reports folder."""
    processed = [s for s in stats if s['status'] == 'processed']
    skipped = [s for s in stats if s['status'] == 'skipped']
    errors = [s for s in stats if s['status'] == 'error']
    
    report_dir = os.path.join(out_dir, "pipeline_reports")
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"cropping_report_{timestamp}.txt")
    
    avg_time = sum(p['duration'] for p in processed) / len(processed) if processed else 0
    
    with open(report_path, "w") as f:
        f.write(f"=== CROPPING PIPELINE REPORT: {location} ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"System: {platform.node()} ({platform.system()})\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Wall Time:   {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)\n")
        f.write(f"Files Processed:   {len(processed)}\n")
        f.write(f"Files Skipped:     {len(skipped)}\n")
        f.write(f"Errors:            {len(errors)}\n")
        f.write(f"Avg Time per File: {avg_time:.2f} seconds\n")
        f.write("-" * 40 + "\n")
        f.write("DETAILS:\n")
        for s in processed:
            f.write(f"[OK] {s['file']} | {s['duration']:.2f}s | {s['points']} pts\n")
        for s in errors:
            f.write(f"[FAIL] {s['file']} | {s['error_msg']}\n")
            
    print(f"\n📄 Report saved to: {report_path}")


def crop_location(location: str, replace: bool = False):
    total_start = time.time()
    
    if location not in mop_ranges:
        raise ValueError(f"Unknown location '{location}'")
    
    min_mop, max_mop = mop_ranges[location]
    poly = load_or_create_crop_polygon(location, min_mop, max_mop)
    poly_wkt = poly.wkt

    csv_path = os.path.join(SURVEY_LIST_DIR, f"surveys_{location}.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Error: Survey list not found at {csv_path}")
        return

    out_dir = os.path.join(RESULTS_BASE, location)
    crop_out_dir = os.path.join(out_dir, "cropped")
    os.makedirs(crop_out_dir, exist_ok=True)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    tasks = []
    print(f"🔎 Scanning survey list for {location}...")
    
    for row in rows:
        method = row["method"]
        survey_raw = row["path"].strip()
        survey_folder = os.path.basename(os.path.normpath(survey_raw))

        # Build path using dynamic BASE_DIR
        base = os.path.join(
            BASE_DIR,
            method,
            "LiDAR_Processed_Level2",
            survey_folder
        )

        pattern = os.path.join(
            base,
            "Beach_And_Backshore",
            "*beach_cliff_ground.las"
        )
        matches = glob.glob(pattern)
        if not matches:
            # print(f"⚠️ No LAS in: {pattern}")
            continue

        las_in  = matches[0]
        stem    = os.path.splitext(os.path.basename(las_in))[0]
        las_out = os.path.join(crop_out_dir, f"{stem}_cropped.las")
        tasks.append((poly_wkt, las_in, las_out, replace))

    if not tasks:
        print("No tasks found. Check paths and survey list.")
        return

    print(f"🚀 Starting processing for {len(tasks)} files...")
    
    # Run processing
    results = []
    nprocs = 3  # Matches OMP_NUM_THREADS settings
    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        # Use tqdm to show progress bar
        futures = tqdm(executor.map(_worker, tasks), total=len(tasks), desc=f"Cropping {location}")
        for res in futures:
            results.append(res)

    total_duration = time.time() - total_start
    write_report(location, results, total_duration, out_dir)
    print(f"\n🎉 All done: {location} → {crop_out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True, help="Location name (e.g., SanElijo)")
    parser.add_argument("--replace", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()
    
    crop_location(args.location, replace=args.replace)

if __name__ == "__main__":
    main()