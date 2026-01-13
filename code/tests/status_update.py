#!/usr/bin/env python3
"""
Status checker for the LiDAR processing pipeline.

The script scans all locations listed in survey CSVs and compares expected
outputs for each pipeline stage (cropped → nobeach → noveg) against what
actually exists on disk. It also summarizes downstream change-detection
artifacts (m3c2, erosion, deposition). A text report is written to
<project_root>/results/reports.

Usage:
    python3 code/tests/status_update.py [--project-root PATH] [--data-root PATH]
"""

import argparse
import csv
import platform
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_MAC_ROOT = Path("/Volumes/group/LiDAR")
DEFAULT_LINUX_ROOT = Path("/project/group/LiDAR")


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


def locate_raw_file(row: dict, data_root: Path) -> Optional[Path]:
    """
    Try multiple plausible locations for the raw LAS file based on the survey row.
    """
    path_field = (row.get("path") or "").strip()
    method = (row.get("method") or "").strip()

    folder_name = Path(path_field).name if path_field else ""
    candidates = []

    if method and folder_name:
        candidates.append(
            data_root
            / method
            / "LiDAR_Processed_Level2"
            / folder_name
            / "Beach_And_Backshore"
        )

    if path_field:
        given_path = Path(path_field)
        candidates.append(given_path / "Beach_And_Backshore")
        candidates.append(given_path)

        if path_field.startswith("/project/group/LiDAR"):
            mac_path = Path(path_field.replace("/project/group/LiDAR", "/Volumes/group/LiDAR", 1))
            candidates.append(mac_path / "Beach_And_Backshore")
        if path_field.startswith("/Volumes/group/LiDAR"):
            linux_path = Path(path_field.replace("/Volumes/group/LiDAR", "/project/group/LiDAR", 1))
            candidates.append(linux_path / "Beach_And_Backshore")

    if folder_name and not method:
        candidates.append(data_root / folder_name / "Beach_And_Backshore")

    seen = set()
    unique_candidates = []
    for cand in candidates:
        key = str(cand)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)

    for cand in unique_candidates:
        matches = []
        if cand.is_file() and cand.name.lower().endswith(("_beach_cliff_ground.las", "_beach_cliff_ground.laz")):
            matches = [cand]
        else:
            matches = list(cand.glob("*beach_cliff_ground.las")) + list(cand.glob("*beach_cliff_ground.laz"))

        if matches:
            return matches[0]

    return None


def gather_las_names(directory: Path) -> set[str]:
    if not directory.exists():
        return set()

    names = {p.name for p in directory.glob("*.las")}
    names.update(p.name for p in directory.glob("*.laz"))
    return names


def compare_stage(expected: set[str], found: set[str]) -> dict:
    return {
        "expected": len(expected),
        "found": len(found),
        "missing": expected - found,
        "extra": found - expected,
    }


def format_name_list(names: set[str], limit: int) -> str:
    if not names:
        return ""

    sorted_names = sorted(names)
    if len(sorted_names) > limit:
        remaining = len(sorted_names) - limit
        return ", ".join(sorted_names[:limit]) + f" ... (+{remaining} more)"
    return ", ".join(sorted_names)


def gather_m3c2_stats(m3c2_root: Path) -> Tuple[int, int, int]:
    if not m3c2_root.exists():
        return 0, 0, 0

    run_dirs = [p for p in m3c2_root.glob("pipeline_run_*") if p.is_dir()]
    pair_dirs = []
    for run in run_dirs:
        pair_dirs.extend([p for p in run.glob("*_to_*") if p.is_dir()])

    m3c2_files = len(list(m3c2_root.rglob("*_m3c2.las")))
    return len(run_dirs), len(pair_dirs), m3c2_files


def gather_change_stats(root: Path) -> Dict[str, int]:
    if not root.exists():
        return {"pairs": 0, "las": 0, "csv": 0}

    pairs = [p for p in root.iterdir() if p.is_dir()]
    return {
        "pairs": len(pairs),
        "las": len(list(root.rglob("*.las"))),
        "csv": len(list(root.rglob("*.csv"))),
    }


def analyze_location(location: str, project_root: Path, data_root: Path, list_limit: int) -> dict:
    survey_dir = project_root / "survey_lists"
    results_dir = project_root / "results"
    csv_path = survey_dir / f"surveys_{location}.csv"

    rows = load_survey_rows(csv_path)
    expected_cropped: set[str] = set()
    expected_nobeach: set[str] = set()
    expected_noveg: set[str] = set()
    raw_missing: List[str] = []

    for row in rows:
        raw_file = locate_raw_file(row, data_root)
        survey_label = Path(row.get("path") or "").name or row.get("path") or row.get("date") or "unknown"

        if not raw_file:
            raw_missing.append(survey_label)
            continue

        stem = raw_file.stem
        expected_cropped.add(f"{stem}_cropped.las")
        expected_nobeach.add(f"{stem}_nobeach.las")
        expected_noveg.add(f"{stem}_noveg.las")

    base_dir = results_dir / location
    cropped_found = gather_las_names(base_dir / "cropped")
    nobeach_found = gather_las_names(base_dir / "nobeach") | gather_las_names(base_dir / "nobeach_new")
    noveg_found = gather_las_names(base_dir / "noveg")

    cropped_stats = compare_stage(expected_cropped, cropped_found)
    nobeach_stats = compare_stage(expected_nobeach, nobeach_found)
    noveg_stats = compare_stage(expected_noveg, noveg_found)

    m3c2_runs, m3c2_pairs, m3c2_files = gather_m3c2_stats(base_dir / "m3c2")
    erosion_stats = gather_change_stats(base_dir / "erosion")
    deposition_stats = gather_change_stats(base_dir / "deposition")

    return {
        "location": location,
        "survey_rows": len(rows),
        "raw_found": len(expected_cropped),
        "raw_missing": raw_missing,
        "cropped": cropped_stats,
        "nobeach": nobeach_stats,
        "noveg": noveg_stats,
        "m3c2_runs": m3c2_runs,
        "m3c2_pairs": m3c2_pairs,
        "m3c2_files": m3c2_files,
        "erosion": erosion_stats,
        "deposition": deposition_stats,
        "list_limit": list_limit,
    }


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
        overall["raw_found"] += s["raw_found"]
        overall["raw_missing"] += len(s["raw_missing"])

        for stage in ("cropped", "nobeach", "noveg"):
            overall[f"{stage}_expected"] += s[stage]["expected"]
            overall[f"{stage}_found"] += s[stage]["found"]
            overall[f"{stage}_missing"] += len(s[stage]["missing"])
            overall[f"{stage}_extra"] += len(s[stage]["extra"])

    lines.extend(
        [
            "Overall:",
            f"  Survey rows: {overall['survey_rows']}",
            f"  Raw LAS located: {overall['raw_found']} (missing {overall['raw_missing']})",
            f"  Step 2 - cropped: expected {overall['cropped_expected']} | found {overall['cropped_found']} | missing {overall['cropped_missing']} | extra {overall['cropped_extra']}",
            f"  Step 4 - nobeach: expected {overall['nobeach_expected']} | found {overall['nobeach_found']} | missing {overall['nobeach_missing']} | extra {overall['nobeach_extra']}",
            f"  Step 5 - noveg: expected {overall['noveg_expected']} | found {overall['noveg_found']} | missing {overall['noveg_missing']} | extra {overall['noveg_extra']}",
            "",
        ]
    )

    for s in stats:
        lines.append(f"[{s['location']}]")
        if s["survey_rows"] == 0:
            lines.append("  No survey list found.")
            lines.append("")
            continue

        lines.append(
            f"  Survey rows: {s['survey_rows']} | raw located: {s['raw_found']} | raw missing: {len(s['raw_missing'])}"
        )
        if s["raw_missing"]:
            lines.append(f"    Missing raw folders/files: {format_name_list(set(s['raw_missing']), list_limit)}")

        for label, stage in (("Step 2 - cropped", "cropped"), ("Step 4 - nobeach", "nobeach"), ("Step 5 - noveg", "noveg")):
            data = s[stage]
            lines.append(
                f"  {label}: expected {data['expected']} | found {data['found']} | missing {len(data['missing'])} | extra {len(data['extra'])}"
            )
            if data["missing"]:
                lines.append(f"    Missing files: {format_name_list(data['missing'], list_limit)}")
            if data["extra"]:
                lines.append(f"    Extra files (not in survey list): {format_name_list(data['extra'], list_limit)}")

        lines.append(
            f"  Step 6 - m3c2: runs {s['m3c2_runs']} | pair folders {s['m3c2_pairs']} | result files {s['m3c2_files']}"
        )
        lines.append(
            "  Step 7+ - erosion: "
            f"{s['erosion']['pairs']} pairs | {s['erosion']['las']} las files | {s['erosion']['csv']} csv files"
        )
        lines.append(
            "  Step 7+ - deposition: "
            f"{s['deposition']['pairs']} pairs | {s['deposition']['las']} las files | {s['deposition']['csv']} csv files"
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
        help="Maximum number of missing/extra filenames to list per section.",
    )
    args = parser.parse_args()

    project_root, data_root = resolve_roots(args.project_root, args.data_root)
    survey_dir = project_root / "survey_lists"
    results_dir = project_root / "results"

    locations = discover_locations(survey_dir)
    stats = [analyze_location(loc, project_root, data_root, args.list_limit) for loc in locations]

    report_lines = build_report(project_root, data_root, stats, args.list_limit)
    report_path = write_report(results_dir / "reports", report_lines)

    print(f"Wrote report to: {report_path}")


if __name__ == "__main__":
    main()
