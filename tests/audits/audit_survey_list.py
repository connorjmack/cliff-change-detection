#!/usr/bin/env python3
"""
audit_survey_list.py

Compare survey list CSVs to cropped outputs and write a report to results/.

Usage:
    python3 tests/audit_survey_list.py --all
    python3 tests/audit_survey_list.py --location SanElijo
    python3 tests/audit_survey_list.py --output-dir /path/to/results
"""

import argparse
import csv
import glob
import os
import platform
from collections import defaultdict
from datetime import datetime

MOP_LOCATIONS = [
    "Blacks",
    "DelMar",
    "Encinitas",
    "SanElijo",
    "Solana",
    "Torrey",
]


def detect_root_lidar():
    system = platform.system()
    if system == "Darwin":
        root_lidar = "/Volumes/group/LiDAR"
    else:
        root_lidar = "/project/group/LiDAR"
    return system, root_lidar


def read_survey_list(csv_path):
    names = set()
    if not os.path.exists(csv_path):
        return names
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = (row.get("path") or "").strip()
            if not path:
                continue
            names.add(os.path.basename(path.rstrip("/")))
    return names


def parse_cropped_name(filename):
    raw = os.path.splitext(filename)[0]
    if raw.endswith("_cropped"):
        raw = raw[: -len("_cropped")]

    flags = []
    if "_beach_cliff_ground_class_" in raw:
        flags.append("class")
        base = raw.split("_beach_cliff_ground_class_")[0]
    elif raw.endswith("_beach_cliff_ground"):
        base = raw[: -len("_beach_cliff_ground")]
    else:
        flags.append("nonstandard")
        base = raw

    return base, raw, flags


def read_cropped_dir(crop_dir):
    data = {
        "bases": set(),
        "by_base": defaultdict(list),
        "class_files": [],
        "nonstandard_files": [],
        "total_files": 0,
    }
    if not os.path.isdir(crop_dir):
        return data

    for path in glob.glob(os.path.join(crop_dir, "*.las")):
        filename = os.path.basename(path)
        base, raw, flags = parse_cropped_name(filename)
        data["bases"].add(base)
        data["by_base"][base].append(raw)
        data["total_files"] += 1
        if "class" in flags:
            data["class_files"].append(raw)
        if "nonstandard" in flags:
            data["nonstandard_files"].append(raw)
    return data


def audit_location(project_root, location):
    survey_dir = os.path.join(project_root, "survey_lists")
    results_dir = os.path.join(project_root, "results")

    csv_path = os.path.join(survey_dir, f"surveys_{location}.csv")
    crop_dir = os.path.join(results_dir, location, "cropped")

    survey_names = read_survey_list(csv_path)
    crop_data = read_cropped_dir(crop_dir)

    only_in_cropped = sorted(crop_data["bases"] - survey_names)
    only_in_list = sorted(survey_names - crop_data["bases"])
    duplicates = {
        base: raws
        for base, raws in crop_data["by_base"].items()
        if len(raws) > 1
    }

    return {
        "location": location,
        "survey_list_path": csv_path,
        "cropped_dir": crop_dir,
        "survey_list_count": len(survey_names),
        "cropped_file_count": crop_data["total_files"],
        "cropped_base_count": len(crop_data["bases"]),
        "class_file_count": len(crop_data["class_files"]),
        "nonstandard_file_count": len(crop_data["nonstandard_files"]),
        "only_in_cropped": only_in_cropped,
        "only_in_list": only_in_list,
        "duplicates": duplicates,
        "class_files": sorted(crop_data["class_files"]),
        "nonstandard_files": sorted(crop_data["nonstandard_files"]),
    }


def write_report(report_path, system, root_lidar, results):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("=== SURVEY LIST AUDIT ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"System: {system}\n")
        f.write(f"ROOT_LIDAR: {root_lidar}\n")
        f.write("-" * 60 + "\n\n")

        for res in results:
            f.write(f"[{res['location']}]\n")
            f.write(f"survey_list: {res['survey_list_path']}\n")
            f.write(f"cropped_dir: {res['cropped_dir']}\n")
            f.write(f"survey_list_count: {res['survey_list_count']}\n")
            f.write(f"cropped_file_count: {res['cropped_file_count']}\n")
            f.write(f"cropped_base_count: {res['cropped_base_count']}\n")
            f.write(f"class_file_count: {res['class_file_count']}\n")
            f.write(f"nonstandard_file_count: {res['nonstandard_file_count']}\n")
            f.write(f"only_in_cropped_count: {len(res['only_in_cropped'])}\n")
            f.write(f"only_in_list_count: {len(res['only_in_list'])}\n")
            f.write(f"duplicate_base_count: {len(res['duplicates'])}\n")

            if res["only_in_cropped"]:
                f.write("only_in_cropped:\n")
                for name in res["only_in_cropped"]:
                    f.write(f"  {name}\n")

            if res["only_in_list"]:
                f.write("only_in_list:\n")
                for name in res["only_in_list"]:
                    f.write(f"  {name}\n")

            if res["duplicates"]:
                f.write("duplicate_bases:\n")
                for base, raws in sorted(res["duplicates"].items()):
                    joined = ", ".join(sorted(raws))
                    f.write(f"  {base} -> {joined}\n")

            if res["class_files"]:
                f.write("class_files:\n")
                for name in res["class_files"]:
                    f.write(f"  {name}\n")

            if res["nonstandard_files"]:
                f.write("nonstandard_files:\n")
                for name in res["nonstandard_files"]:
                    f.write(f"  {name}\n")

            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Audit survey lists vs cropped outputs.")
    parser.add_argument("--location", type=str, help="Specific location to audit")
    parser.add_argument("--all", action="store_true", help="Audit all locations")
    parser.add_argument("--output-dir", type=str, help="Directory for report output")
    args = parser.parse_args()

    system, root_lidar = detect_root_lidar()
    project_root = os.path.join(root_lidar, "LidarProcessing", "LidarProcessingCliffs")
    results_dir = os.path.join(project_root, "results")
    output_dir = args.output_dir or results_dir

    if args.location:
        locations = [args.location]
    else:
        locations = MOP_LOCATIONS

    results = []
    for loc in locations:
        if loc not in MOP_LOCATIONS:
            print(f"Skipping unknown location: {loc}")
            continue
        results.append(audit_location(project_root, loc))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = args.location or "all"
    report_name = f"audit_survey_list_{suffix}_{timestamp}.txt"
    report_path = os.path.join(output_dir, report_name)
    write_report(report_path, system, root_lidar, results)

    print(f"Wrote audit report: {report_path}")


if __name__ == "__main__":
    main()
