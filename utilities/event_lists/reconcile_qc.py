#!/usr/bin/env python3
"""
reconcile_qc.py

Transfers QC flags from a previously reviewed event list CSV to a freshly
generated one.

Matching logic:
  1. Exact match on (start_date, end_date)
  2. Rank candidates by closest volume
  3. Confirm the best volume match is at the same location
     (alongshore centroid within --along-tol, elevation within --elev-tol)

Events that pass date + volume but fail the location check are flagged
as suspicious rather than silently dropped.

Usage:
    python3 reconcile_qc.py <new_csv> <qc_csv>
    python3 reconcile_qc.py <new_csv> <qc_csv> --vol-tol 0.3
    python3 reconcile_qc.py <new_csv> <qc_csv> --dry-run
"""
import os
import argparse
import numpy as np
import pandas as pd


DEFAULT_VOL_TOL = 0.5       # max fractional volume difference (50%)
DEFAULT_ALONG_TOL = 3.0     # max alongshore distance (m) for location confirmation
DEFAULT_ELEV_TOL = 1.0      # max elevation distance (m) for location confirmation
QC_COL = "qc_flag"
DEFAULT_FLAG = "unreviewed"


def load_csv(path):
    """Load event CSV, parsing date columns."""
    return pd.read_csv(path, parse_dates=["start_date", "end_date", "mid_date"])


def reconcile(new_df, qc_df, vol_tol, along_tol, elev_tol):
    """Match reviewed QC rows to the new event list and transfer qc_flag.

    Strategy: exact dates → closest volume → confirm same location.

    Returns
    -------
    merged_df : DataFrame
        The new event list with qc_flag column added.
    unmatched : DataFrame
        Reviewed QC rows that found no match in the new list.
    match_stats : dict
        Summary statistics about the reconciliation.
    """
    reviewed = qc_df[qc_df[QC_COL] != DEFAULT_FLAG].copy()
    print(f"  Reviewed events to reconcile: {len(reviewed)}")
    print(f"  QC flag breakdown:")
    for flag, count in reviewed[QC_COL].value_counts().items():
        print(f"    {flag}: {count}")

    new_df[QC_COL] = DEFAULT_FLAG

    matched_new_idx = set()
    matched_qc_idx = set()
    match_details = []  # (vol_frac, along_dist, elev_dist) per match

    reviewed_groups = reviewed.groupby(["start_date", "end_date"])

    for (sd, ed), qc_group in reviewed_groups:
        date_mask = (new_df["start_date"] == sd) & (new_df["end_date"] == ed)
        candidates = new_df[date_mask]

        if candidates.empty:
            continue

        for qc_idx, qc_row in qc_group.iterrows():
            qc_vol = qc_row["volume"]
            qc_along = qc_row["alongshore_centroid_m"]
            qc_elev = qc_row["elevation"]

            # Score unclaimed candidates by volume similarity
            scored = []
            for new_idx, cand_row in candidates.iterrows():
                if new_idx in matched_new_idx:
                    continue

                max_vol = max(abs(qc_vol), abs(cand_row["volume"]), 1e-6)
                vol_frac = abs(qc_vol - cand_row["volume"]) / max_vol
                if vol_frac > vol_tol:
                    continue

                along_dist = abs(qc_along - cand_row["alongshore_centroid_m"])
                elev_dist = abs(qc_elev - cand_row["elevation"])
                scored.append((vol_frac, new_idx, along_dist, elev_dist))

            if not scored:
                continue

            # Sort by volume similarity (primary)
            scored.sort(key=lambda x: x[0])

            # Take the best volume match and confirm location
            for vol_frac, best_idx, along_dist, elev_dist in scored:
                if along_dist <= along_tol and elev_dist <= elev_tol:
                    new_df.at[best_idx, QC_COL] = qc_row[QC_COL]
                    matched_new_idx.add(best_idx)
                    matched_qc_idx.add(qc_idx)
                    match_details.append((vol_frac, along_dist, elev_dist))
                    break

    unmatched = reviewed[~reviewed.index.isin(matched_qc_idx)].copy()

    vol_fracs = [d[0] for d in match_details] if match_details else [0]
    along_dists = [d[1] for d in match_details] if match_details else [0]
    elev_dists = [d[2] for d in match_details] if match_details else [0]

    match_stats = {
        "reviewed_total": len(reviewed),
        "matched": len(matched_qc_idx),
        "unmatched": len(unmatched),
        "new_total": len(new_df),
        "mean_vol_frac": np.mean(vol_fracs),
        "max_vol_frac": np.max(vol_fracs),
        "mean_along_dist_m": np.mean(along_dists),
        "max_along_dist_m": np.max(along_dists),
        "mean_elev_dist_m": np.mean(elev_dists),
        "max_elev_dist_m": np.max(elev_dists),
    }

    return new_df, unmatched, match_stats


def main():
    parser = argparse.ArgumentParser(
        description="Transfer QC flags from a reviewed event list to a new one."
    )
    parser.add_argument("new_csv", help="Path to the freshly generated event list CSV")
    parser.add_argument("qc_csv", help="Path to the QC'd event list CSV (with qc_flag column)")
    parser.add_argument("--vol-tol", type=float, default=DEFAULT_VOL_TOL,
                        help=f"Max fractional volume difference (default: {DEFAULT_VOL_TOL})")
    parser.add_argument("--along-tol", type=float, default=DEFAULT_ALONG_TOL,
                        help=f"Max alongshore distance for location check (default: {DEFAULT_ALONG_TOL}m)")
    parser.add_argument("--elev-tol", type=float, default=DEFAULT_ELEV_TOL,
                        help=f"Max elevation distance for location check (default: {DEFAULT_ELEV_TOL}m)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print report only, don't write output file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: overwrites new_csv)")
    args = parser.parse_args()

    if not os.path.exists(args.new_csv):
        raise FileNotFoundError(f"New CSV not found: {args.new_csv}")
    if not os.path.exists(args.qc_csv):
        raise FileNotFoundError(f"QC CSV not found: {args.qc_csv}")

    print(f"New event list:  {args.new_csv}")
    print(f"QC'd event list: {args.qc_csv}")
    print(f"Tolerances:      volume={args.vol_tol:.0%}, alongshore={args.along_tol}m, elevation={args.elev_tol}m")
    print()

    new_df = load_csv(args.new_csv)
    qc_df = load_csv(args.qc_csv)

    if QC_COL not in qc_df.columns:
        raise ValueError(f"QC CSV is missing '{QC_COL}' column")

    print(f"New events: {len(new_df)}")
    print(f"QC events:  {len(qc_df)}")
    print()

    merged_df, unmatched, stats = reconcile(
        new_df, qc_df,
        vol_tol=args.vol_tol, along_tol=args.along_tol, elev_tol=args.elev_tol,
    )

    # --- Report ---
    print(f"\n{'='*50}")
    print(f"RECONCILIATION REPORT")
    print(f"{'='*50}")
    print(f"  Reviewed events:    {stats['reviewed_total']}")
    print(f"  Matched:            {stats['matched']}")
    print(f"  Unmatched:          {stats['unmatched']}")
    print(f"  Volume diff (frac): mean={stats['mean_vol_frac']:.3f}, max={stats['max_vol_frac']:.3f}")
    print(f"  Alongshore (m):     mean={stats['mean_along_dist_m']:.3f}, max={stats['max_along_dist_m']:.3f}")
    print(f"  Elevation (m):      mean={stats['mean_elev_dist_m']:.3f}, max={stats['max_elev_dist_m']:.3f}")
    print(f"  New events total:   {stats['new_total']}")
    print()

    print("New CSV flag breakdown after merge:")
    for flag, count in merged_df[QC_COL].value_counts().items():
        print(f"  {flag}: {count}")

    if len(unmatched) > 0:
        print(f"\nWARNING: {len(unmatched)} reviewed events could not be matched:")
        cols = ["start_date", "end_date", "alongshore_centroid_m", "volume", QC_COL]
        display_cols = [c for c in cols if c in unmatched.columns]
        print(unmatched[display_cols].to_string(index=False))

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        output_path = args.output or args.new_csv
        merged_df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
