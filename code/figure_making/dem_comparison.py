#!/usr/bin/env python3
"""
dem_comparison.py

Compares M3C2-based erosion volumes against traditional DEM-of-Difference
(DoD) volumes for the top-N largest QC'd erosion events at Del Mar.

For each date pair the script:
  1. Loads both full noveg LAS point clouds (no spatial clipping).
  2. Rasterises both to top-down DEMs (max Z per XY cell) on a common grid.
  3. Computes DoD = DEM_after - DEM_before, applies a LoD threshold.
  4. Reports per-survey total erosion volume from the DoD.
  5. Compares V_DoD to the sum of all M3C2 erosion events for that date pair.

Usage:
    python3 dem_comparison.py
    python3 dem_comparison.py --n_events 10 --dem_res 0.5
    python3 dem_comparison.py --no_figure   # skip figure, just print table
"""

import os
import glob
import platform
import argparse
import numpy as np
import pandas as pd
import laspy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage

# ── paths ──────────────────────────────────────────────────────────────────────
SYSTEM = platform.system()
if SYSTEM == "Darwin":
    BASE = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
else:
    BASE = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

LOCATION = "DelMar"
NOVEG_DIR = os.path.join(BASE, "results", LOCATION, "noveg")

# QC event list (repo-relative, auto-detect most recent)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
QC_DIR = os.path.join(REPO_ROOT, "results", "event_lists_qc", "erosion")

# Output
FIG_DIR = os.path.join(REPO_ROOT, "figures", "appendix")

# ── constants ──────────────────────────────────────────────────────────────────
DEM_RES = 0.25          # m – top-down DEM cell size (default)
LOD_THRESHOLD = 0.25    # m – Level of Detection for DoD


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def find_qc_csv():
    """Find the most recent DelMar QC event list CSV."""
    pattern = os.path.join(QC_DIR, "DelMar_events_qc_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No DelMar QC event list found in {QC_DIR}")
    return matches[-1]


def find_noveg_file(date_str):
    """Find the noveg LAS file whose name starts with *date_str* (YYYYMMDD)."""
    pattern = os.path.join(NOVEG_DIR, f"{date_str}*.las")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[0]


def load_full(las_path):
    """Load a LAS file and return (x, y, z) as float64 arrays."""
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    return x, y, z


def rasterise_to_common_grid(x, y, z, x_min, y_min, dem_res, nx, ny):
    """Rasterise points to a top-down DSM (max Z per cell) on a pre-defined grid.

    Returns
    -------
    dem : 2-D ndarray (ny, nx), NaN where no data
    """
    xi = np.clip(((np.asarray(x, dtype=np.float64) - x_min) / dem_res).astype(int), 0, nx - 1)
    yi = np.clip(((np.asarray(y, dtype=np.float64) - y_min) / dem_res).astype(int), 0, ny - 1)
    flat = yi * nx + xi
    df_pts = pd.DataFrame({"flat": flat, "z": np.asarray(z, dtype=np.float64)})
    maxz = df_pts.groupby("flat")["z"].max()
    dem = np.full(ny * nx, np.nan)
    dem[maxz.index.values] = maxz.values
    return dem.reshape(ny, nx)


def compute_dod(dem_before, dem_after, lod):
    """Compute DEM of Difference, zeroing cells below LoD.

    Returns
    -------
    dod : 2-D array (positive = deposition, negative = erosion), NaN = no data.
    """
    valid = ~np.isnan(dem_before) & ~np.isnan(dem_after)
    dod = np.full_like(dem_before, np.nan)
    dod[valid] = dem_after[valid] - dem_before[valid]
    below_lod = np.abs(dod) < lod
    dod[below_lod] = 0.0
    return dod


def dod_erosion_volume(dod, cell_area):
    """Total erosion volume from a DoD (sum of negative cells)."""
    ero_mask = dod < 0
    if not np.any(ero_mask):
        return 0.0
    return float(np.sum(np.abs(dod[ero_mask])) * cell_area)


def dod_deposition_volume(dod, cell_area):
    """Total deposition volume from a DoD (sum of positive cells)."""
    dep_mask = dod > 0
    if not np.any(dep_mask):
        return 0.0
    return float(np.sum(dod[dep_mask]) * cell_area)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def run_comparison(n_events=5, dem_res=DEM_RES, lod=LOD_THRESHOLD):
    """Run the DEM-vs-M3C2 comparison for the top-N event date pairs.

    Uses the full survey extent (no spatial clipping).
    Returns a DataFrame with one row per date pair.
    """
    qc_csv = find_qc_csv()
    print(f"Using QC file: {os.path.basename(qc_csv)}")

    df = pd.read_csv(qc_csv,
                     parse_dates=["mid_date", "start_date", "end_date"])
    real = df[df["qc_flag"] == "real"].copy()

    # Get top N events by volume to determine which date pairs to process
    top_events = (real.sort_values("volume", ascending=False)
                  .head(n_events)
                  .reset_index(drop=True))

    # For each date pair, sum ALL real M3C2 erosion events (not just the top one)
    date_pairs = []
    for _, row in top_events.iterrows():
        d1 = row["start_date"]
        d2 = row["end_date"]
        # Sum all real events for this date pair
        pair_mask = (real["start_date"] == d1) & (real["end_date"] == d2)
        total_m3c2 = real.loc[pair_mask, "volume"].sum()
        n_events_pair = pair_mask.sum()
        date_pairs.append({
            "start_date": d1,
            "end_date": d2,
            "V_M3C2_total": total_m3c2,
            "V_M3C2_top_event": row["volume"],
            "n_m3c2_events": n_events_pair,
        })

    # Deduplicate in case the same date pair appears for multiple top events
    seen = set()
    unique_pairs = []
    for dp in date_pairs:
        key = (dp["start_date"], dp["end_date"])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(dp)
    date_pairs = unique_pairs[:n_events]

    print(f"\n{'='*70}")
    print(f"DEM-of-Difference vs M3C2 Comparison  —  Del Mar (full survey)")
    print(f"DEM resolution: {dem_res} m  |  LoD threshold: {lod} m")
    print(f"Using {len(date_pairs)} unique date pairs from top {n_events} events")
    print(f"{'='*70}\n")

    cell_area = dem_res * dem_res
    results = []

    for i, dp in enumerate(date_pairs):
        d1_str = dp["start_date"].strftime("%Y%m%d")
        d2_str = dp["end_date"].strftime("%Y%m%d")
        v_m3c2 = dp["V_M3C2_total"]

        print(f"Survey pair {i+1}: {d1_str} → {d2_str}")
        print(f"  M3C2: {dp['n_m3c2_events']} real events, "
              f"total V = {v_m3c2:.1f} m³ "
              f"(top event: {dp['V_M3C2_top_event']:.1f} m³)")

        # --- find noveg files ---
        f1 = find_noveg_file(d1_str)
        f2 = find_noveg_file(d2_str)
        if f1 is None or f2 is None:
            print(f"  !! Missing noveg file(s): d1={f1}, d2={f2}")
            results.append({
                "pair": i + 1,
                "start_date": d1_str, "end_date": d2_str,
                "V_M3C2": v_m3c2, "V_DoD_ero": np.nan,
                "V_DoD_dep": np.nan, "ratio": np.nan,
                "n_m3c2_events": dp["n_m3c2_events"],
                "status": "missing_file"
            })
            continue

        # --- load full files ---
        print(f"  Loading {os.path.basename(f1)} ...")
        x1, y1, z1 = load_full(f1)
        print(f"    → {len(x1):,} points")

        print(f"  Loading {os.path.basename(f2)} ...")
        x2, y2, z2 = load_full(f2)
        print(f"    → {len(x2):,} points")

        # --- common-extent grid for both DEMs ---
        all_x = np.concatenate([x1, x2])
        all_y = np.concatenate([y1, y2])
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        del all_x, all_y  # free memory

        nx = int(np.ceil((x_max - x_min) / dem_res))
        ny = int(np.ceil((y_max - y_min) / dem_res))
        x_edges = np.linspace(x_min, x_min + nx * dem_res, nx + 1)
        y_edges = np.linspace(y_min, y_min + ny * dem_res, ny + 1)

        print(f"  Grid: {nx} × {ny} cells ({nx*ny:,} total)")
        print(f"  Extent: X [{x_min:.1f}, {x_max:.1f}], Y [{y_min:.1f}, {y_max:.1f}]")
        print(f"  Rasterising DEMs ...")

        dem1 = rasterise_to_common_grid(x1, y1, z1, x_min, y_min, dem_res, nx, ny)
        del x1, y1, z1
        dem2 = rasterise_to_common_grid(x2, y2, z2, x_min, y_min, dem_res, nx, ny)
        del x2, y2, z2

        # --- DoD ---
        dod = compute_dod(dem1, dem2, lod)
        v_dod_ero = dod_erosion_volume(dod, cell_area)
        v_dod_dep = dod_deposition_volume(dod, cell_area)
        ratio = v_m3c2 / v_dod_ero if v_dod_ero > 0 else np.inf

        # Raw DoD for diagnostics
        valid_both = ~np.isnan(dem1) & ~np.isnan(dem2)
        raw_diff = dem2[valid_both] - dem1[valid_both]

        n_dem1 = np.sum(~np.isnan(dem1))
        n_dem2 = np.sum(~np.isnan(dem2))
        n_overlap = np.sum(valid_both)

        print(f"  DEM1 data cells: {n_dem1:,} / {nx*ny:,} "
              f"({100*n_dem1/(nx*ny):.1f}%)")
        print(f"  DEM2 data cells: {n_dem2:,} / {nx*ny:,} "
              f"({100*n_dem2/(nx*ny):.1f}%)")
        print(f"  Overlap cells:   {n_overlap:,}")
        print(f"  Raw DoD range:   {raw_diff.min():.3f} to {raw_diff.max():.3f} m")
        print(f"  Raw DoD mean:    {raw_diff.mean():.4f} m")
        print(f"  Cells |DoD|>{lod}m:  {np.sum(np.abs(raw_diff) >= lod):,}")
        print(f"  Cells DoD<-{lod}m:   {np.sum(raw_diff <= -lod):,}  (erosion)")
        print(f"  Cells DoD>{lod}m:    {np.sum(raw_diff >= lod):,}  (deposition)")
        print(f"  V_DoD_erosion  = {v_dod_ero:.1f} m³")
        print(f"  V_DoD_deposition = {v_dod_dep:.1f} m³")
        print(f"  V_M3C2_total   = {v_m3c2:.1f} m³")
        print(f"  Ratio (M3C2/DoD) = {ratio:.1f}×")
        print()

        results.append({
            "pair": i + 1,
            "start_date": d1_str, "end_date": d2_str,
            "V_M3C2": v_m3c2, "V_DoD_ero": v_dod_ero,
            "V_DoD_dep": v_dod_dep, "ratio": ratio,
            "n_m3c2_events": dp["n_m3c2_events"],
            "status": "ok",
            # stash arrays for figures
            "_dem1": dem1, "_dem2": dem2,
            "_dod": dod,
            "_x_edges": x_edges, "_y_edges": y_edges,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def find_largest_dod_cluster(dod, dem_res, buffer_m=30.0):
    """Find the largest erosion cluster in the DoD.

    Uses connected-component labeling on erosion cells (DoD < 0,
    already LoD-thresholded). Returns the cluster with the largest
    erosion volume and its bounding box (in pixel coords).

    Returns
    -------
    row_slice, col_slice : slices for the bounding box (with buffer)
    cluster_vol : float, erosion volume (positive) for the cluster
    n_labels : int, total number of erosion clusters found
    """
    erosion = (dod < 0) & ~np.isnan(dod)
    if not np.any(erosion):
        return None, None, 0.0, 0

    labelled, n_labels = ndimage.label(erosion)

    # Find erosion cluster with largest volume (vectorised)
    cell_area = dem_res * dem_res
    labels_range = np.arange(1, n_labels + 1)
    cluster_vols = ndimage.sum(np.abs(dod), labelled, labels_range) * cell_area
    best_idx = np.argmax(cluster_vols)
    best_label = labels_range[best_idx]
    best_vol = float(cluster_vols[best_idx])

    # Bounding box of the best cluster
    rows, cols = np.where(labelled == best_label)
    buf_px = int(np.ceil(buffer_m / dem_res))
    ny, nx = dod.shape
    r_lo = max(0, rows.min() - buf_px)
    r_hi = min(ny, rows.max() + buf_px + 1)
    c_lo = max(0, cols.min() - buf_px)
    c_hi = min(nx, cols.max() + buf_px + 1)

    return slice(r_lo, r_hi), slice(c_lo, c_hi), best_vol, n_labels


def make_event_figures(results_df):
    """Save a zoomed DoD figure for each survey pair.

    Finds the largest cluster of significant change in the DoD and
    zooms to that area with a buffer.
    """
    dem_dir = os.path.join(FIG_DIR, "dems")
    os.makedirs(dem_dir, exist_ok=True)

    ok = results_df[results_df["status"] == "ok"]
    if ok.empty:
        print("No valid survey pairs to plot.")
        return

    for _, row in ok.iterrows():
        pair = int(row["pair"])
        dod = row["_dod"]
        x_edges = row["_x_edges"]
        y_edges = row["_y_edges"]
        dem_res = x_edges[1] - x_edges[0]

        # Find largest cluster
        r_slice, c_slice, cluster_vol, n_clusters = find_largest_dod_cluster(
            dod, dem_res)

        if r_slice is None:
            print(f"  Pair {pair}: no significant DoD cells, skipping figure")
            continue

        # Zoom to cluster region
        dod_zoom = dod[r_slice, c_slice]
        x_edges_zoom = x_edges[c_slice.start:c_slice.stop + 1]
        y_edges_zoom = y_edges[r_slice.start:r_slice.stop + 1]

        fig, ax = plt.subplots(figsize=(10, 8))

        dod_abs_max = max(abs(np.nanmin(dod_zoom)), abs(np.nanmax(dod_zoom)), 0.5)
        im = ax.pcolormesh(x_edges_zoom, y_edges_zoom, dod_zoom,
                           cmap="RdBu", vmin=-dod_abs_max, vmax=dod_abs_max,
                           shading="flat")
        cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("DoD elevation change (m)", fontsize=11)

        ax.set_xlabel("Easting (m)", fontsize=11)
        ax.set_ylabel("Northing (m)", fontsize=11)
        survey_ratio = row['V_M3C2'] / row['V_DoD_ero'] if row['V_DoD_ero'] > 0 else np.inf
        ax.set_title(
            f"Survey {row['start_date']} \u2192 {row['end_date']}  "
            f"(zoomed to largest erosion cluster)\n"
            f"Survey totals:  DoD erosion = {row['V_DoD_ero']:.1f} m\u00b3  |  "
            f"M3C2 erosion = {row['V_M3C2']:.1f} m\u00b3  "
            f"({int(row['n_m3c2_events'])} events)  |  "
            f"Ratio: {survey_ratio:.1f}\u00d7",
            fontweight="bold", fontsize=9)
        ax.set_aspect("equal")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=8, labelrotation=30)

        plt.tight_layout()

        out = os.path.join(dem_dir,
                           f"pair_{pair}_{row['start_date']}_to_{row['end_date']}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close()
        print(f"  Saved: {out}  "
              f"(V_DoD={row['V_DoD_ero']:.1f} m\u00b3 vs "
              f"V_M3C2={row['V_M3C2']:.1f} m\u00b3, "
              f"ratio={survey_ratio:.1f}\u00d7)")

    print(f"\nAll zoomed DoD figures saved to: {dem_dir}")


def make_multi_panel_figure(results_df):
    """Create 2×5 panel figure: M3C2 grid (top) vs DoD (bottom) for top 5 events.

    Top row shows cliff-face M3C2 grid zoomed to each event footprint.
    Bottom row shows top-down DoD zoomed to the largest erosion cluster.
    Both rows use the same color convention: red = erosion, blue = deposition.

    Outputs to figures/appendix/dems/multi-panel.png
    """
    dem_dir = os.path.join(FIG_DIR, "dems")
    os.makedirs(dem_dir, exist_ok=True)

    # --- Load QC CSV for individual event details ---
    qc_csv = find_qc_csv()
    df_qc = pd.read_csv(qc_csv, parse_dates=["mid_date", "start_date", "end_date"])
    real = df_qc[df_qc["qc_flag"] == "real"].copy()
    top5 = real.sort_values("volume", ascending=False).head(3).reset_index(drop=True)

    print(f"\nCreating multi-panel figure for top 3 real events ...")

    # --- Load NPZ cube for M3C2 grid panels ---
    npz_path = os.path.join(REPO_ROOT, "results", "data_cubes",
                             f"{LOCATION}_cube.npz")
    if not os.path.exists(npz_path):
        print(f"  NPZ cube not found: {npz_path}")
        return

    cube = np.load(npz_path, allow_pickle=True)
    along_m  = cube['alongshore_m']
    elev_m   = cube['elevation_m']
    ero_3d   = cube.get('erosion')       # (n_along, n_elev, n_time)
    dep_3d   = cube.get('deposition')
    dstrings = [str(s) for s in cube['date_strings']]

    # Sort alongshore for cliff-facing display
    asort   = np.argsort(along_m)
    along_s = along_m[asort]

    ok = results_df[results_df["status"] == "ok"]

    # --- Create figure ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for col, (_, ev) in enumerate(top5.iterrows()):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        d1 = ev["start_date"].strftime("%Y%m%d")
        d2 = ev["end_date"].strftime("%Y%m%d")
        dfolder = f"{d1}_to_{d2}"

        # ── Top row: M3C2 cliff-facing grid ─────────────────────────
        tidx = next((i for i, ds in enumerate(dstrings) if ds == dfolder), None)

        if tidx is not None and ero_3d is not None:
            # Extract and sort 2D slice
            ero2d = ero_3d[:, :, tidx][asort, :]
            combined = np.nan_to_num(ero2d, nan=0.0)
            if dep_3d is not None:
                dep2d = dep_3d[:, :, tidx][asort, :]
                dc = np.nan_to_num(dep2d, nan=0.0)
                dep_mask = (dc > 0) & (combined < 0.01)
                combined[dep_mask] = -dc[dep_mask]

            # Zoom to event footprint
            xpad, ypb, ypt = 10, 8, 3
            xlo = ev['alongshore_start_m'] - xpad
            xhi = ev['alongshore_end_m'] + xpad
            ylo = max(0, ev['elevation'] - ev['height'] / 2 - ypb)
            yhi = ev['elevation'] + ev['height'] / 2 + ypt

            xm = (along_s >= xlo) & (along_s <= xhi)
            ym = (elev_m >= ylo) & (elev_m <= yhi)

            if np.any(xm) and np.any(ym):
                xi = np.where(xm)[0]
                yi = np.where(ym)[0]
                mz = combined[xi[0]:xi[-1]+1, yi[0]:yi[-1]+1].T  # (elev, along)
                az = along_s[xi[0]:xi[-1]+1]
                ez = elev_m[yi[0]:yi[-1]+1]

                nz = mz[mz != 0]
                vm = max(np.percentile(np.abs(nz), 85), 0.1) if nz.size else 2.5

                im1 = ax_top.imshow(mz, origin='lower', aspect='auto',
                                     interpolation='nearest', cmap='RdBu_r',
                                     vmin=-vm, vmax=vm)

                # Tick labels
                nt = min(4, len(az))
                if len(az) > 1:
                    ti = np.linspace(0, len(az)-1, nt, dtype=int)
                    ax_top.set_xticks(ti)
                    ax_top.set_xticklabels([f'{az[j]:.0f}' for j in ti], fontsize=5)
                nt = min(4, len(ez))
                if len(ez) > 1:
                    ti = np.linspace(0, len(ez)-1, nt, dtype=int)
                    ax_top.set_yticks(ti)
                    ax_top.set_yticklabels([f'{ez[j]:.1f}' for j in ti], fontsize=5)

                ax_top.invert_xaxis()

                # Crosshairs at event centroid
                xc = np.argmin(np.abs(az - ev['alongshore_centroid_m']))
                yc = np.argmin(np.abs(ez - ev['elevation']))
                ax_top.axhline(yc, color='#666', ls='--', lw=0.6, alpha=0.4)
                ax_top.axvline(xc, color='#666', ls='--', lw=0.6, alpha=0.4)

                cb1 = plt.colorbar(im1, ax=ax_top, shrink=0.5, pad=0.02, aspect=12)
                cb1.ax.tick_params(labelsize=5)
                if col == 2:
                    cb1.set_label("M3C2 (m)", fontsize=7)
            else:
                ax_top.text(0.5, 0.5, "Outside grid", transform=ax_top.transAxes,
                            ha='center', va='center', fontsize=7)
        else:
            ax_top.text(0.5, 0.5, "No cube data", transform=ax_top.transAxes,
                        ha='center', va='center', fontsize=7)

        ax_top.set_title(f"{d1} \u2192 {d2}\nV = {ev['volume']:.1f} m\u00b3",
                          fontsize=7, fontweight='bold')

        # ── Bottom row: DoD (top-down) ──────────────────────────────
        match = ok[(ok["start_date"] == d1) & (ok["end_date"] == d2)]
        if not match.empty:
            r = match.iloc[0]
            dod    = r["_dod"]
            xedges = r["_x_edges"]
            yedges = r["_y_edges"]
            dres   = xedges[1] - xedges[0]

            rs, cs, cvol, _ = find_largest_dod_cluster(dod, dres)
            if rs is not None:
                dz = dod[rs, cs]
                xe = xedges[cs.start:cs.stop + 1]
                ye = yedges[rs.start:rs.stop + 1]
                vabs = max(abs(np.nanmin(dz)), abs(np.nanmax(dz)), 0.5)

                # Negate DoD so erosion is positive (red), matching M3C2 row
                im2 = ax_bot.pcolormesh(xe, ye, -dz, cmap="RdBu_r",
                                         vmin=-vabs, vmax=vabs, shading="flat")
                ax_bot.ticklabel_format(useOffset=False, style="plain")
                ax_bot.tick_params(labelsize=4, labelrotation=30)
                cb2 = plt.colorbar(im2, ax=ax_bot, shrink=0.5, pad=0.02, aspect=12)
                cb2.ax.tick_params(labelsize=5)
                if col == 2:
                    cb2.set_label("Elev. loss (m)", fontsize=7)
            else:
                ax_bot.text(0.5, 0.5, "No erosion\nin DoD", ha='center',
                            va='center', transform=ax_bot.transAxes, fontsize=7)

            ratio = r['V_M3C2'] / r['V_DoD_ero'] if r['V_DoD_ero'] > 0 else np.inf
            ax_bot.set_title(
                f"DoD ero = {r['V_DoD_ero']:.0f} m\u00b3 | ratio = {ratio:.1f}\u00d7",
                fontsize=6, fontweight='bold')
        else:
            ax_bot.text(0.5, 0.5, "No DoD data", ha='center',
                        va='center', transform=ax_bot.transAxes, fontsize=7)
            ax_bot.set_title(f"DoD: no data", fontsize=6)

    # Axis labels on leftmost panels
    axes[0, 0].set_ylabel("Elevation (m)", fontsize=8, fontweight='bold')
    axes[1, 0].set_ylabel("Northing (m)", fontsize=8, fontweight='bold')

    # Row labels on the left margin
    fig.text(0.008, 0.72, "M3C2 Grid\n(cliff-face)", fontsize=10,
             fontweight='bold', ha='left', va='center', rotation=90)
    fig.text(0.008, 0.28, "DoD\n(top-down)", fontsize=10,
             fontweight='bold', ha='left', va='center', rotation=90)

    fig.suptitle(
        "M3C2 vs DEM-of-Difference \u2014 Top 3 Erosion Events (Del Mar)\n"
        "Red = erosion, Blue = deposition in both rows",
        fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.93])

    out = os.path.join(dem_dir, "multi-panel.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: {out}")


def make_figure(results_df):
    """Summary comparison figure: bar chart + DoD map of largest pair + schematic."""
    os.makedirs(FIG_DIR, exist_ok=True)

    ok = results_df[results_df["status"] == "ok"].copy()
    if ok.empty:
        print("No valid survey pairs to plot.")
        return

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.2, 0.8], wspace=0.35)

    # ── Panel A: bar chart ─────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0])
    x_pos = np.arange(len(ok))
    w = 0.35
    ax_bar.bar(x_pos - w/2, ok["V_M3C2"], w, color="#2166ac", label="M3C2")
    ax_bar.bar(x_pos + w/2, ok["V_DoD_ero"], w, color="#b2182b", label="DoD")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([f"P{int(p)}" for p in ok["pair"]], fontsize=9)
    ax_bar.set_ylabel("Erosion Volume (m\u00b3)")
    ax_bar.set_xlabel("Survey Pair")
    ax_bar.legend(frameon=False)
    ax_bar.set_title("(a) Volume comparison", fontweight="bold", loc="left")

    # Annotate ratios
    for j, (_, r) in enumerate(ok.iterrows()):
        ratio_str = f"{r['ratio']:.1f}\u00d7" if np.isfinite(r["ratio"]) else "\u221e"
        y_top = max(r["V_M3C2"], r["V_DoD_ero"])
        ax_bar.text(j, y_top * 1.05, ratio_str, ha="center", va="bottom",
                    fontsize=8, color="#333")

    y_ceil = max(ok["V_M3C2"].max(), ok["V_DoD_ero"].max()) * 1.25
    ax_bar.set_ylim(0, y_ceil)

    # ── Panel B: Zoomed DoD map of largest survey pair ──────────────────────
    ax_map = fig.add_subplot(gs[1])
    biggest = ok.iloc[0]

    dod = biggest["_dod"]
    if dod is not None and not np.all(np.isnan(dod)):
        x_edges = biggest["_x_edges"]
        y_edges = biggest["_y_edges"]
        dem_res = x_edges[1] - x_edges[0]

        # Zoom to largest erosion cluster
        r_slice, c_slice, _, _ = find_largest_dod_cluster(dod, dem_res)
        if r_slice is not None:
            dod_zoom = dod[r_slice, c_slice]
            x_edges_zoom = x_edges[c_slice.start:c_slice.stop + 1]
            y_edges_zoom = y_edges[r_slice.start:r_slice.stop + 1]
        else:
            dod_zoom = dod
            x_edges_zoom = x_edges
            y_edges_zoom = y_edges

        vmax = max(abs(np.nanmin(dod_zoom)), abs(np.nanmax(dod_zoom)), 0.5)
        im = ax_map.pcolormesh(x_edges_zoom, y_edges_zoom, dod_zoom,
                               cmap="RdBu", vmin=-vmax, vmax=vmax,
                               shading="flat")
        cb = plt.colorbar(im, ax=ax_map, shrink=0.8, pad=0.02)
        cb.set_label("DoD elevation change (m)", fontsize=9)

        ax_map.set_xlabel("Easting (m)")
        ax_map.set_ylabel("Northing (m)")
        ax_map.set_title(
            f"(b) DoD — P{int(biggest['pair'])}:  "
            f"{biggest['start_date']} \u2192 {biggest['end_date']}",
            fontweight="bold", loc="left", fontsize=9)
        ax_map.set_aspect("equal")
        ax_map.ticklabel_format(useOffset=False, style="plain")
        ax_map.tick_params(labelsize=7, labelrotation=30)

    # ── Panel C: schematic cross-section ──────────────────────────────────
    ax_xs = fig.add_subplot(gs[2])
    ax_xs.set_xlim(-2, 12)
    ax_xs.set_ylim(-1, 14)
    ax_xs.set_aspect("equal")
    ax_xs.axis("off")
    ax_xs.set_title("(c) Why DEMs miss\n     cliff-face change",
                     fontweight="bold", loc="left", fontsize=9)

    # Cliff profile (before)
    cliff_x = [0, 0, 0.5, 0.5, 0, 0, 8, 8]
    cliff_z = [0, 4, 4, 9, 9, 12, 12, 0]
    ax_xs.fill(cliff_x, cliff_z, color="#d9c6a5", edgecolor="#333",
               linewidth=1.5, label="Cliff (before)")

    # Eroded notch
    notch_x = [0, 0.8, 0.8, 0]
    notch_z = [4, 4, 9, 9]
    ax_xs.fill(notch_x, notch_z, color="#ef8a62", edgecolor="#b2182b",
               linewidth=1.5, alpha=0.8, label="Eroded volume")

    # Beach
    ax_xs.fill([-2, 0, 0, -2], [0, 0, -0.5, -0.5],
               color="#f0e6c8", edgecolor="#aaa")
    ax_xs.text(-1, -0.3, "Beach", ha="center", fontsize=7, color="#888")

    # DEM view arrow (from above)
    ax_xs.annotate("", xy=(4, 12.8), xytext=(4, 13.8),
                   arrowprops=dict(arrowstyle="-|>", color="#2166ac", lw=2))
    ax_xs.text(4, 14.0, "DEM view\n(top-down)", ha="center", va="bottom",
               fontsize=8, color="#2166ac", fontweight="bold")

    # M3C2 view arrow (from side)
    ax_xs.annotate("", xy=(0.3, 6.5), xytext=(-1.8, 6.5),
                   arrowprops=dict(arrowstyle="-|>", color="#b2182b", lw=2))
    ax_xs.text(-1.8, 7.2, "M3C2 view\n(face-normal)",
               ha="center", fontsize=8, color="#b2182b", fontweight="bold")

    # Labels
    ax_xs.text(4, 11.5, "Cliff top \u2014 no change\nvisible from above",
               ha="center", fontsize=7, style="italic", color="#555")
    ax_xs.text(1.8, 6.5, "Face scar\n(M3C2 only)",
               ha="center", fontsize=7, fontweight="bold", color="#b2182b")

    # Save
    out_png = os.path.join(FIG_DIR, "dem_comparison.png")
    out_pdf = os.path.join(FIG_DIR, "dem_comparison.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.savefig(out_pdf, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare M3C2 vs DEM-of-Difference volumes (full survey).")
    parser.add_argument("--n_events", type=int, default=5,
                        help="Number of top events to compare (default: 5)")
    parser.add_argument("--dem_res", type=float, default=DEM_RES,
                        help=f"DEM cell size in metres (default: {DEM_RES})")
    parser.add_argument("--lod", type=float, default=LOD_THRESHOLD,
                        help=f"LoD threshold in metres (default: {LOD_THRESHOLD})")
    parser.add_argument("--no_figure", action="store_true",
                        help="Skip figure generation, print table only")
    args = parser.parse_args()

    results = run_comparison(n_events=args.n_events,
                             dem_res=args.dem_res,
                             lod=args.lod)

    # Print summary table
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    cols = ["pair", "start_date", "end_date", "V_M3C2", "V_DoD_ero",
            "V_DoD_dep", "ratio", "n_m3c2_events"]
    print(results[cols].to_string(index=False, float_format="%.1f"))
    print()

    if not args.no_figure:
        make_event_figures(results)
        make_multi_panel_figure(results)
        make_figure(results)


if __name__ == "__main__":
    main()
