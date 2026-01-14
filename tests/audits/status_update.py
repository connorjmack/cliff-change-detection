#!/usr/bin/env python3
"""
Status checker for the LiDAR processing pipeline.

The script scans all locations listed in survey CSVs and reports counts for
each pipeline stage (cropped → nobeach → noveg) based on files that actually
exist on disk. It also summarizes downstream change-detection artifacts
(m3c2, erosion, deposition). A text report is written to
<project_root>/results/reports.

Usage:
    python3 code/tests/status_update.py [--project-root PATH] [--data-root PATH]
"""

import argparse
import csv
import platform
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

DEFAULT_MAC_ROOT = Path("/Volumes/group/LiDAR")
DEFAULT_LINUX_ROOT = Path("/project/group/LiDAR")
DATE_PATTERN = re.compile(r"(\d{8})")


def resolve_roots(project_root_arg: Optional[str], data_root_arg: Optional[str]) -> Tuple[Path, Path]:
    system = platform.system()
    default_data_root = DEFAULT_MAC_ROOT if system == "Darwin" else DEFAULT_LINUX_ROOT

    project_root = (
        Path(project_root_arg).expanduser().resolve()
        if project_root_arg
        else (default_data_root / "LidarProcessing" / "LidarProcessingCliffs")
    )

    data_root = (
        Path(data_root_arg).expanduser().resolve()
        if data_root_arg
        else project_root.parent.parent
    )

    return project_root, data_root


def discover_locations(survey_dir: Path) -> List[str]:
    """
    Locations are derived strictly from survey list CSVs to avoid scanning
    unrelated folders that may live under results/ on shared storage.
    """
    locations: set[str] = set()

    if survey_dir.exists():
        for csv_path in survey_dir.glob("surveys_*.csv"):
            locations.add(csv_path.stem.replace("surveys_", "", 1))

    return sorted(locations)


def load_survey_rows(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        return []

    with csv_path.open() as f:
        return [row for row in csv.DictReader(f)]


def gather_las_names(directory: Path) -> set[str]:
    if not directory.exists():
        return set()

    names = {p.name for p in directory.glob("*.las")}
    names.update(p.name for p in directory.glob("*.laz"))
    return names


def extract_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    cleaned = value.strip()
    if len(cleaned) == 8 and cleaned.isdigit():
        return cleaned

    match = DATE_PATTERN.search(cleaned)
    if match:
        return match.group(1)

    return None


def collect_expected_dates(rows: List[dict]) -> set[str]:
    candidate_fields = (
        "date",
        "path",
        "file",
        "filename",
        "las",
        "las_file",
        "raw_file",
        "raw_filename",
        "raw_path",
    )
    dates: set[str] = set()
    for row in rows:
        for key in candidate_fields:
            date = extract_date(row.get(key))
            if date:
                dates.add(date)
                break
    return dates


def extract_dates_from_names(names: set[str]) -> set[str]:
    dates = set()
    for name in names:
        date = extract_date(name)
        if date:
            dates.add(date)
    return dates


def generate_expected_pairs(dates: List[str]) -> set[str]:
    """Generate expected sequential survey pairs from sorted dates."""
    if len(dates) < 2:
        return set()

    sorted_dates = sorted(dates)
    pairs = set()
    for i in range(len(sorted_dates) - 1):
        pairs.add(f"{sorted_dates[i]}_to_{sorted_dates[i+1]}")
    return pairs


def gather_m3c2_pairs(m3c2_root: Path) -> set[str]:
    """Get all m3c2 pairs that exist (from any pipeline_run_*)."""
    if not m3c2_root.exists():
        return set()

    pairs = set()
    run_dirs = [p for p in m3c2_root.glob("pipeline_run_*") if p.is_dir()]
    for run in run_dirs:
        for pair_dir in run.glob("*_to_*"):
            if pair_dir.is_dir():
                pairs.add(pair_dir.name)
    return pairs


def gather_change_pairs(root: Path) -> set[str]:
    """Get all erosion/deposition pairs that exist."""
    if not root.exists():
        return set()

    pairs = set()
    for pair_dir in root.glob("*_to_*"):
        if pair_dir.is_dir():
            pairs.add(pair_dir.name)
    return pairs


def analyze_location(location: str, project_root: Path) -> dict:
    survey_dir = project_root / "survey_lists"
    results_dir = project_root / "results"
    csv_path = survey_dir / f"surveys_{location}.csv"

    rows = load_survey_rows(csv_path)
    expected_dates = collect_expected_dates(rows)

    base_dir = results_dir / location
    cropped_found = gather_las_names(base_dir / "cropped")
    nobeach_found = gather_las_names(base_dir / "nobeach") | gather_las_names(base_dir / "nobeach_new")
    noveg_found = gather_las_names(base_dir / "noveg")

    cropped_dates = extract_dates_from_names(cropped_found)
    nobeach_dates = extract_dates_from_names(nobeach_found)
    noveg_dates = extract_dates_from_names(noveg_found)

    missing_dates = {
        "cropped": expected_dates - cropped_dates,
        "nobeach": expected_dates - nobeach_dates,
        "noveg": expected_dates - noveg_dates,
    }

    expected_pairs = generate_expected_pairs(sorted(expected_dates))

    # Gather actual pairs
    m3c2_pairs_found = gather_m3c2_pairs(base_dir / "m3c2")
    erosion_pairs_found = gather_change_pairs(base_dir / "erosion")
    deposition_pairs_found = gather_change_pairs(base_dir / "deposition")

    missing_pairs = {
        "m3c2": expected_pairs - m3c2_pairs_found,
        "erosion": expected_pairs - erosion_pairs_found,
        "deposition": expected_pairs - deposition_pairs_found,
    }

    return {
        "location": location,
        "survey_rows": len(rows),
        "cropped": len(cropped_found),
        "nobeach": len(nobeach_found),
        "noveg": len(noveg_found),
        "m3c2": len(m3c2_pairs_found),
        "erosion": len(erosion_pairs_found),
        "deposition": len(deposition_pairs_found),
        "missing_dates": missing_dates,
        "missing_pairs": missing_pairs,
    }


def format_name_list(names: set[str], limit: int) -> str:
    if not names:
        return "none"

    sorted_names = sorted(names)
    if len(sorted_names) > limit:
        remaining = len(sorted_names) - limit
        return ", ".join(sorted_names[:limit]) + f" ... (+{remaining} more)"
    return ", ".join(sorted_names)


def build_report(project_root: Path, data_root: Path, stats: List[dict], list_limit: int) -> List[str]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    locations = [s["location"] for s in stats]
    lines = [
        f"Pipeline status report generated {timestamp}",
        f"Project root: {project_root}",
        f"Data root: {data_root}",
        f"Locations scanned: {', '.join(locations) if locations else 'none'}",
        "",
    ]

    overall = defaultdict(int)
    for s in stats:
        overall["survey_rows"] += s["survey_rows"]

    lines.append("Survey counts by location:")
    for s in stats:
        lines.append(f"  {s['location']}: {s['survey_rows']}")
    lines.append(f"Total survey rows: {overall['survey_rows']}")
    lines.append("")

    for s in stats:
        lines.append(
            f"[{s['location']}] surveys: {s['survey_rows']} | "
            f"cropped: {s['cropped']} | nobeach: {s['nobeach']} | noveg: {s['noveg']} | "
            f"m3c2: {s['m3c2']} | erosion: {s['erosion']} | deposition: {s['deposition']}"
        )
        if s["survey_rows"] == 0:
            lines.append("  No survey list found.")
            lines.append("")
            continue

        lines.append(
            f"  Missing dates (cropped): {format_name_list(s['missing_dates']['cropped'], list_limit)}"
        )
        lines.append(
            f"  Missing dates (nobeach): {format_name_list(s['missing_dates']['nobeach'], list_limit)}"
        )
        lines.append(
            f"  Missing dates (noveg): {format_name_list(s['missing_dates']['noveg'], list_limit)}"
        )
        lines.append(f"  Missing pairs (m3c2): {format_name_list(s['missing_pairs']['m3c2'], list_limit)}")
        lines.append(
            f"  Missing pairs (erosion): {format_name_list(s['missing_pairs']['erosion'], list_limit)}"
        )
        lines.append(
            f"  Missing pairs (deposition): {format_name_list(s['missing_pairs']['deposition'], list_limit)}"
        )

        lines.append("")

    return lines


def write_report(report_dir: Path, lines: list[str]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    fname = f"status_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = report_dir / fname
    report_path.write_text("\n".join(lines))
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Summarize pipeline completeness across all locations.")
    parser.add_argument(
        "--project-root",
        help="Root directory containing code/, results/, survey_lists/, utilities/. Defaults to shared storage path.",
    )
    parser.add_argument(
        "--data-root",
        help="Root directory containing instrument folders (MiniRanger_Truck, etc.). Defaults to parent of project root.",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=15,
        help="Maximum number of missing dates or pairs to list per section.",
    )
    args = parser.parse_args()

    project_root, data_root = resolve_roots(args.project_root, args.data_root)
    print(f"Project root: {project_root}")
    print(f"Data root: {data_root}")

    survey_dir = project_root / "survey_lists"
    results_dir = project_root / "results"

    locations = discover_locations(survey_dir)
    print(f"Found {len(locations)} locations to scan.")

    stats = []
    for i, loc in enumerate(locations, 1):
        print(f"[{i}/{len(locations)}] Analyzing {loc}...")
        stats.append(analyze_location(loc, project_root))

    report_lines = build_report(project_root, data_root, stats, args.list_limit)
    report_path = write_report(project_root / "reports", report_lines)

    print(f"Wrote report to: {report_path}")


if __name__ == "__main__":
    main()
