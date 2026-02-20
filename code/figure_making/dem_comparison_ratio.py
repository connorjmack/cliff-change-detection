#!/usr/bin/env python3
"""
dem_comparison_ratio.py

Finds erosion events where the M3C2 cliff-facing method captures far more
volume than a traditional top-down DEM-of-Difference (DoD). This highlights
cases involving overhangs, wave-cut notches, and steep/vertical faces that
are invisible to nadir DEMs.

Strategy:
  1. Filter QC'd real erosion events to those above a volume threshold.
  2. Compute DoD erosion volume for each unique date pair.
  3. Discard pairs where DoD >= M3C2 (only keep ratio > 1).
  4. Rank remaining pairs by ratio (V_M3C2 / V_DoD) — highest = biggest
     DEM blind spot.
  5. Generate figures for the top-N date pairs by ratio.

Usage:
    python3 dem_comparison_ratio.py                          # top 15, 5 pages of 3
    python3 dem_comparison_ratio.py --min_volume 100 --n_top 10
    python3 dem_comparison_ratio.py --cols_per_page 5        # wider pages
    python3 dem_comparison_ratio.py --no_figure
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

# -- paths -----------------------------------------------------------------
SYSTEM = platform.system()
if SYSTEM == "Darwin":
    BASE = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
else:
    BASE = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

LOCATION = "DelMar"
NOVEG_DIR = os.path.join(BASE, "results", LOCATION, "noveg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
QC_DIR = os.path.join(REPO_ROOT, "results", "event_lists_qc", "erosion")

FIG_DIR = os.path.join(REPO_ROOT, "figures", "appendix")

# -- constants -------------------------------------------------------------
DEM_RES = 0.25          # m - top-down DEM cell size
LOD_THRESHOLD = 0.25    # m - Level of Detection for DoD
MIN_VOLUME = 50.0       # m3 - minimum M3C2 event volume to consider


# ==========================================================================
#  HELPERS  (shared with dem_comparison.py)
# ==========================================================================

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


def find_largest_dod_cluster(dod, dem_res, buffer_m=10.0):
    """Find the largest erosion cluster in the DoD.

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

    cell_area = dem_res * dem_res
    labels_range = np.arange(1, n_labels + 1)
    cluster_vols = ndimage.sum(np.abs(dod), labelled, labels_range) * cell_area
    best_idx = np.argmax(cluster_vols)
    best_label = labels_range[best_idx]
    best_vol = float(cluster_vols[best_idx])

    rows, cols = np.where(labelled == best_label)
    buf_px = int(np.ceil(buffer_m / dem_res))
    ny, nx = dod.shape
    r_lo = max(0, rows.min() - buf_px)
    r_hi = min(ny, rows.max() + buf_px + 1)
    c_lo = max(0, cols.min() - buf_px)
    c_hi = min(nx, cols.max() + buf_px + 1)

    return slice(r_lo, r_hi), slice(c_lo, c_hi), best_vol, n_labels


# ==========================================================================
#  MAIN COMPARISON — ratio-ranked
# ==========================================================================

def run_comparison(min_volume=MIN_VOLUME, n_top=10, dem_res=DEM_RES,
                   lod=LOD_THRESHOLD):
    """Run DEM-vs-M3C2 comparison for all events above a volume threshold.

    Processes every unique date pair that contains at least one real event
    with volume >= min_volume. After computing DoD for all pairs, ranks
    by ratio (V_M3C2_total / V_DoD_erosion) descending and returns the
    top n_top results.

    Returns
    -------
    all_results : DataFrame with every processed date pair
    top_results : DataFrame with the top n_top pairs by ratio
    """
    qc_csv = find_qc_csv()
    print(f"Using QC file: {os.path.basename(qc_csv)}")

    df = pd.read_csv(qc_csv,
                     parse_dates=["mid_date", "start_date", "end_date"])
    real = df[df["qc_flag"] == "real"].copy()

    # Filter to events above volume threshold
    big_events = real[real["volume"] >= min_volume].copy()
    print(f"Real events with volume >= {min_volume} m3: {len(big_events)}")

    if big_events.empty:
        raise ValueError(
            f"No real events with volume >= {min_volume} m3 found")

    # Build unique date pairs — sum all real M3C2 events per pair
    pair_groups = (real.groupby(["start_date", "end_date"])
                   .agg(V_M3C2_total=("volume", "sum"),
                        V_M3C2_max=("volume", "max"),
                        n_m3c2_events=("volume", "count"))
                   .reset_index())

    # Keep only pairs that contain at least one event above threshold
    big_pair_keys = set(
        zip(big_events["start_date"], big_events["end_date"]))
    pair_groups = pair_groups[
        pair_groups.apply(
            lambda r: (r["start_date"], r["end_date"]) in big_pair_keys,
            axis=1)
    ].reset_index(drop=True)

    print(f"Unique date pairs to process: {len(pair_groups)}")

    print(f"\n{'='*70}")
    print(f"DEM-of-Difference vs M3C2 Comparison  —  Del Mar (ratio-ranked)")
    print(f"DEM resolution: {dem_res} m  |  LoD threshold: {lod} m")
    print(f"Min M3C2 volume: {min_volume} m3  |  "
          f"Processing {len(pair_groups)} date pairs")
    print(f"{'='*70}\n")

    cell_area = dem_res * dem_res
    results = []

    for i, (_, pg) in enumerate(pair_groups.iterrows()):
        d1_str = pg["start_date"].strftime("%Y%m%d")
        d2_str = pg["end_date"].strftime("%Y%m%d")
        v_m3c2 = pg["V_M3C2_total"]

        print(f"[{i+1}/{len(pair_groups)}] {d1_str} -> {d2_str}")
        print(f"  M3C2: {int(pg['n_m3c2_events'])} real events, "
              f"total V = {v_m3c2:.1f} m3 "
              f"(largest: {pg['V_M3C2_max']:.1f} m3)")

        # --- find noveg files ---
        f1 = find_noveg_file(d1_str)
        f2 = find_noveg_file(d2_str)
        if f1 is None or f2 is None:
            print(f"  !! Missing noveg file(s): d1={f1}, d2={f2}")
            results.append({
                "start_date": d1_str, "end_date": d2_str,
                "V_M3C2": v_m3c2, "V_M3C2_max": pg["V_M3C2_max"],
                "V_DoD_ero": np.nan, "V_DoD_dep": np.nan,
                "ratio": np.nan,
                "n_m3c2_events": int(pg["n_m3c2_events"]),
                "status": "missing_file"
            })
            continue

        # --- load full files ---
        print(f"  Loading {os.path.basename(f1)} ...")
        x1, y1, z1 = load_full(f1)
        print(f"    -> {len(x1):,} points")

        print(f"  Loading {os.path.basename(f2)} ...")
        x2, y2, z2 = load_full(f2)
        print(f"    -> {len(x2):,} points")

        # --- common-extent grid ---
        all_x = np.concatenate([x1, x2])
        all_y = np.concatenate([y1, y2])
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        del all_x, all_y

        nx = int(np.ceil((x_max - x_min) / dem_res))
        ny = int(np.ceil((y_max - y_min) / dem_res))
        x_edges = np.linspace(x_min, x_min + nx * dem_res, nx + 1)
        y_edges = np.linspace(y_min, y_min + ny * dem_res, ny + 1)

        print(f"  Grid: {nx} x {ny} cells ({nx*ny:,} total)")
        print(f"  Rasterising DEMs ...")

        dem1 = rasterise_to_common_grid(x1, y1, z1, x_min, y_min,
                                        dem_res, nx, ny)
        del x1, y1, z1
        dem2 = rasterise_to_common_grid(x2, y2, z2, x_min, y_min,
                                        dem_res, nx, ny)
        del x2, y2, z2

        # --- DoD ---
        dod = compute_dod(dem1, dem2, lod)
        v_dod_ero = dod_erosion_volume(dod, cell_area)
        v_dod_dep = dod_deposition_volume(dod, cell_area)
        ratio = v_m3c2 / v_dod_ero if v_dod_ero > 0 else np.inf

        print(f"  V_DoD_erosion  = {v_dod_ero:.1f} m3")
        print(f"  V_M3C2_total   = {v_m3c2:.1f} m3")
        print(f"  Ratio (M3C2/DoD) = {ratio:.1f}x")
        print()

        results.append({
            "start_date": d1_str, "end_date": d2_str,
            "V_M3C2": v_m3c2, "V_M3C2_max": pg["V_M3C2_max"],
            "V_DoD_ero": v_dod_ero, "V_DoD_dep": v_dod_dep,
            "ratio": ratio,
            "n_m3c2_events": int(pg["n_m3c2_events"]),
            "status": "ok",
            "_dem1": dem1, "_dem2": dem2, "_dod": dod,
            "_x_edges": x_edges, "_y_edges": y_edges,
        })

    all_results = pd.DataFrame(results)

    # --- Keep only pairs where M3C2 > DoD, then rank by ratio ---
    ok = all_results[all_results["status"] == "ok"].copy()
    ok = ok[ok["ratio"] > 1.0]

    n_discarded = (all_results["status"] == "ok").sum() - len(ok)
    if n_discarded > 0:
        print(f"Discarded {n_discarded} pair(s) where DoD >= M3C2 "
              f"(ratio <= 1.0)")

    if ok.empty:
        print("No date pairs where M3C2 volume exceeds DoD volume.")
        return all_results, pd.DataFrame()

    # Sort: inf first (DoD found nothing), then by descending ratio
    ok["_sort_key"] = ok["ratio"].replace(np.inf, 1e12)
    ok = ok.sort_values("_sort_key", ascending=False).drop(columns="_sort_key")

    # Assign rank
    ok = ok.reset_index(drop=True)
    ok["rank"] = range(1, len(ok) + 1)

    top = ok.head(n_top).copy()

    print(f"\n{'='*70}")
    print(f"RATIO-RANKED RESULTS — M3C2 > DoD only "
          f"(top {min(n_top, len(ok))} of {len(ok)} qualifying pairs)")
    print(f"{'='*70}")
    display_cols = ["rank", "start_date", "end_date", "V_M3C2",
                    "V_DoD_ero", "ratio", "n_m3c2_events"]
    print(top[display_cols].to_string(index=False, float_format="%.1f"))
    print()

    return all_results, top


# ==========================================================================
#  FIGURES
# ==========================================================================

def make_event_figures(top_df):
    """Save a zoomed DoD figure for each top-ranked survey pair."""
    dem_dir = os.path.join(FIG_DIR, "dems_ratio")
    os.makedirs(dem_dir, exist_ok=True)

    for _, row in top_df.iterrows():
        rank = int(row["rank"])
        dod = row["_dod"]
        x_edges = row["_x_edges"]
        y_edges = row["_y_edges"]
        dem_res = x_edges[1] - x_edges[0]

        r_slice, c_slice, cluster_vol, n_clusters = find_largest_dod_cluster(
            dod, dem_res)

        if r_slice is None:
            print(f"  Rank {rank}: no significant DoD cells, skipping figure")
            continue

        dod_zoom = dod[r_slice, c_slice]
        x_edges_zoom = x_edges[c_slice.start:c_slice.stop + 1]
        y_edges_zoom = y_edges[r_slice.start:r_slice.stop + 1]

        fig, ax = plt.subplots(figsize=(10, 8))

        dod_masked = np.ma.masked_invalid(dod_zoom)
        dod_abs_max = max(abs(np.nanmin(dod_zoom)),
                         abs(np.nanmax(dod_zoom)), 0.5)
        im = ax.pcolormesh(x_edges_zoom, y_edges_zoom, dod_masked,
                           cmap="RdBu", vmin=-dod_abs_max, vmax=dod_abs_max,
                           shading="flat")
        ax.set_facecolor("white")
        cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("DoD elevation change (m)", fontsize=11)

        ax.set_xlabel("Easting (m)", fontsize=11)
        ax.set_ylabel("Northing (m)", fontsize=11)
        ratio_str = (f"{row['ratio']:.1f}" if np.isfinite(row["ratio"])
                     else "inf")
        ax.set_title(
            f"Rank #{rank}: {row['start_date']} -> {row['end_date']}  "
            f"(zoomed to largest erosion cluster)\n"
            f"DoD erosion = {row['V_DoD_ero']:.1f} m\u00b3  |  "
            f"M3C2 erosion = {row['V_M3C2']:.1f} m\u00b3  "
            f"({int(row['n_m3c2_events'])} events)  |  "
            f"Ratio: {ratio_str}\u00d7",
            fontweight="bold", fontsize=9)
        ax.set_aspect("equal")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=8, labelrotation=30)

        plt.tight_layout()

        out = os.path.join(
            dem_dir,
            f"rank{rank}_{row['start_date']}_to_{row['end_date']}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close()
        print(f"  Saved: {out}  (ratio={ratio_str}x)")

    print(f"\nAll zoomed DoD figures saved to: {dem_dir}")


def make_multi_panel_figures(top_df, cols_per_page=3):
    """Create 2xN panel figures: M3C2 grid (top) vs DoD (bottom).

    Chunks all rows in top_df into pages of `cols_per_page` columns each.
    E.g. 15 events with cols_per_page=3 produces 5 separate figures.
    """
    dem_dir = os.path.join(FIG_DIR, "dems_ratio")
    os.makedirs(dem_dir, exist_ok=True)

    n_total = len(top_df)
    if n_total == 0:
        print("No valid pairs for multi-panel figure.")
        return

    # --- Load QC CSV for individual event footprints ---
    qc_csv = find_qc_csv()
    df_qc = pd.read_csv(qc_csv,
                        parse_dates=["mid_date", "start_date", "end_date"])
    real = df_qc[df_qc["qc_flag"] == "real"].copy()

    # --- Load NPZ cube for M3C2 grid panels ---
    npz_path = os.path.join(REPO_ROOT, "results", "data_cubes",
                             f"{LOCATION}_cube.npz")
    if not os.path.exists(npz_path):
        print(f"  NPZ cube not found: {npz_path}")
        return

    cube = np.load(npz_path, allow_pickle=True)
    along_m = cube['alongshore_m']
    elev_m = cube['elevation_m']
    ero_3d = cube.get('erosion')
    dep_3d = cube.get('deposition')
    dstrings = [str(s) for s in cube['date_strings']]

    asort = np.argsort(along_m)
    along_s = along_m[asort]

    # --- Chunk into pages ---
    n_pages = int(np.ceil(n_total / cols_per_page))
    print(f"\nCreating {n_pages} multi-panel figure(s) "
          f"({cols_per_page} columns each, {n_total} total pairs) ...")

    rows_list = list(top_df.iterrows())

    for page in range(n_pages):
        start = page * cols_per_page
        end = min(start + cols_per_page, n_total)
        page_rows = rows_list[start:end]
        n_cols = len(page_rows)

        fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 9))
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        for col, (_, row) in enumerate(page_rows):
            ax_top = axes[0, col]
            ax_bot = axes[1, col]

            d1 = row["start_date"]
            d2 = row["end_date"]
            dfolder = f"{d1}_to_{d2}"
            rank = int(row["rank"])

            # Find the largest real event for this date pair (for footprint)
            d1_dt = pd.Timestamp(d1)
            d2_dt = pd.Timestamp(d2)
            pair_events = real[
                (real["start_date"] == d1_dt) & (real["end_date"] == d2_dt)
            ].sort_values("volume", ascending=False)

            # -- Top row: M3C2 cliff-facing grid --
            tidx = next((i for i, ds in enumerate(dstrings)
                         if ds == dfolder), None)

            if (tidx is not None and ero_3d is not None
                    and not pair_events.empty):
                ev = pair_events.iloc[0]

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
                    mz = combined[xi[0]:xi[-1]+1, yi[0]:yi[-1]+1].T
                    az = along_s[xi[0]:xi[-1]+1]
                    ez = elev_m[yi[0]:yi[-1]+1]

                    nz = mz[mz != 0]
                    vm = (max(np.percentile(np.abs(nz), 85), 0.1)
                          if nz.size else 2.5)

                    im1 = ax_top.imshow(mz, origin='lower', aspect='auto',
                                        interpolation='nearest',
                                        cmap='RdBu_r',
                                        vmin=-vm, vmax=vm)

                    nt = min(4, len(az))
                    if len(az) > 1:
                        ti = np.linspace(0, len(az)-1, nt, dtype=int)
                        ax_top.set_xticks(ti)
                        ax_top.set_xticklabels(
                            [f'{az[j]:.0f}' for j in ti], fontsize=5)
                    nt = min(4, len(ez))
                    if len(ez) > 1:
                        ti = np.linspace(0, len(ez)-1, nt, dtype=int)
                        ax_top.set_yticks(ti)
                        ax_top.set_yticklabels(
                            [f'{ez[j]:.1f}' for j in ti], fontsize=5)

                    ax_top.invert_xaxis()

                    cb1 = plt.colorbar(im1, ax=ax_top, shrink=0.5,
                                       pad=0.02, aspect=12)
                    cb1.ax.tick_params(labelsize=5)
                    if col == n_cols - 1:
                        cb1.set_label("M3C2 (m)", fontsize=7)
                else:
                    ax_top.text(0.5, 0.5, "Outside grid",
                                transform=ax_top.transAxes,
                                ha='center', va='center', fontsize=7)
            else:
                ax_top.text(0.5, 0.5, "No cube data",
                            transform=ax_top.transAxes,
                            ha='center', va='center', fontsize=7)

            ratio_str = (f"{row['ratio']:.1f}"
                         if np.isfinite(row["ratio"]) else "inf")
            ax_top.set_title(
                f"#{rank}: {d1} -> {d2}\n"
                f"M3C2: {row['V_M3C2']:.1f} m\u00b3  |  "
                f"Ratio: {ratio_str}\u00d7",
                fontsize=9, fontweight='bold')

            # -- Bottom row: DoD (top-down) --
            dod = row["_dod"]
            xedges = row["_x_edges"]
            yedges = row["_y_edges"]
            dres = xedges[1] - xedges[0]

            rs, cs, cvol, _ = find_largest_dod_cluster(dod, dres)
            if rs is not None:
                dz = dod[rs, cs]
                xe = xedges[cs.start:cs.stop + 1]
                ye = yedges[rs.start:rs.stop + 1]
                vabs = max(abs(np.nanmin(dz)), abs(np.nanmax(dz)), 0.5)

                dz_masked = np.ma.masked_invalid(-dz)
                im2 = ax_bot.pcolormesh(xe, ye, dz_masked, cmap="RdBu_r",
                                        vmin=-vabs, vmax=vabs,
                                        shading="flat")
                ax_bot.set_facecolor("white")
                ax_bot.ticklabel_format(useOffset=False, style="plain")
                ax_bot.tick_params(labelsize=4, labelrotation=30)
                cb2 = plt.colorbar(im2, ax=ax_bot, shrink=0.5,
                                   pad=0.02, aspect=12)
                cb2.ax.tick_params(labelsize=5)
                if col == n_cols - 1:
                    cb2.set_label("Elev. loss (m)", fontsize=7)
            else:
                ax_bot.text(0.5, 0.5, "No erosion\nin DoD", ha='center',
                            va='center', transform=ax_bot.transAxes,
                            fontsize=7)

            ax_bot.set_title(
                f"DoD: {row['V_DoD_ero']:.1f} m\u00b3",
                fontsize=9, fontweight='bold')

        axes[0, 0].set_ylabel("M3C2 Grid\n(cliff-facing)", fontsize=10,
                              fontweight='bold')
        axes[1, 0].set_ylabel("DoD\n(top-down)", fontsize=10,
                              fontweight='bold')

        rank_lo = int(top_df.iloc[start]["rank"])
        rank_hi = int(top_df.iloc[end - 1]["rank"])
        fig.suptitle(
            f"Largest M3C2/DoD Discrepancies — Ranks #{rank_lo}-{rank_hi} "
            f"(Del Mar)\n"
            "Red = erosion, Blue = deposition in both rows",
            fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        out = os.path.join(dem_dir,
                           f"multi_panel_ratio_{page+1}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close()
        print(f"  Saved: {out}  (ranks #{rank_lo}-{rank_hi})")


def make_comparison_table(top_df):
    """Create a publication-quality comparison table ranked by ratio."""
    dem_dir = os.path.join(FIG_DIR, "dems_ratio")
    os.makedirs(dem_dir, exist_ok=True)

    rows = []
    for _, r in top_df.iterrows():
        d1_fmt = pd.Timestamp(r["start_date"]).strftime("%Y-%m-%d")
        d2_fmt = pd.Timestamp(r["end_date"]).strftime("%Y-%m-%d")
        ratio_str = (f"{r['ratio']:.1f}\u00d7"
                     if np.isfinite(r["ratio"]) else "\u221e")

        if np.isfinite(r["V_DoD_ero"]) and r["V_DoD_ero"] > 0:
            diff_pct = (r["V_M3C2"] - r["V_DoD_ero"]) / r["V_DoD_ero"] * 100
            diff_str = f"+{diff_pct:.0f}%"
        else:
            diff_str = "\u2014"

        rows.append([
            str(int(r["rank"])),
            f"{d1_fmt}  \u2192  {d2_fmt}",
            f"{r['V_M3C2']:.1f}",
            f"{r['V_DoD_ero']:.1f}" if np.isfinite(r["V_DoD_ero"]) else "\u2014",
            ratio_str,
            diff_str,
        ])

    col_labels = [
        "Rank",
        "Survey Dates",
        "M3C2 Grid\nVolume (m\u00b3)",
        "DoD\nVolume (m\u00b3)",
        "Ratio\n(M3C2/DoD)",
        "Difference\n(%)",
    ]

    n_cols = len(col_labels)
    col_widths = [0.07, 0.32, 0.15, 0.15, 0.14, 0.14]

    n_rows = len(rows)
    fig_h = max(2.0, 0.6 + 0.4 * n_rows)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)

    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        cell.set_edgecolor("white")
        cell.set_linewidth(1.5)

    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_facecolor("#D9E2F3" if i % 2 == 0 else "#F2F2F2")
            cell.set_edgecolor("white")
            cell.set_linewidth(1.5)
            cell.set_text_props(fontsize=10)

    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor("#CCCCCC")

    out = os.path.join(dem_dir, "comparison_table_ratio.png")
    plt.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: {out}")


def make_summary_figure(top_df):
    """Summary figure: bar chart ranked by ratio + DoD map of #1 + schematic."""
    dem_dir = os.path.join(FIG_DIR, "dems_ratio")
    os.makedirs(dem_dir, exist_ok=True)

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.2, 0.8], wspace=0.35)

    # -- Panel A: bar chart (sorted by ratio) --
    ax_bar = fig.add_subplot(gs[0])
    x_pos = np.arange(len(top_df))
    w = 0.35
    ax_bar.bar(x_pos - w/2, top_df["V_M3C2"].values, w,
               color="#2166ac", label="M3C2 Grid")
    ax_bar.bar(x_pos + w/2, top_df["V_DoD_ero"].values, w,
               color="#b2182b", label="DoD")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(
        [f"#{int(r)}" for r in top_df["rank"]], fontsize=8)
    ax_bar.set_ylabel("Erosion Volume (m\u00b3)")
    ax_bar.set_xlabel("Rank (by M3C2/DoD ratio)")
    ax_bar.legend(frameon=False)
    ax_bar.set_title("(a) Volume comparison (sorted by ratio)",
                     fontweight="bold", loc="left", fontsize=9)

    for j, (_, r) in enumerate(top_df.iterrows()):
        ratio_str = (f"{r['ratio']:.1f}\u00d7"
                     if np.isfinite(r["ratio"]) else "\u221e")
        y_top = max(r["V_M3C2"], r["V_DoD_ero"])
        ax_bar.text(j, y_top * 1.05, ratio_str, ha="center", va="bottom",
                    fontsize=7, color="#333")

    y_ceil = max(top_df["V_M3C2"].max(),
                 top_df["V_DoD_ero"].max()) * 1.25
    ax_bar.set_ylim(0, y_ceil)

    # -- Panel B: zoomed DoD map of #1 ranked pair --
    ax_map = fig.add_subplot(gs[1])
    best = top_df.iloc[0]

    dod = best["_dod"]
    if dod is not None and not np.all(np.isnan(dod)):
        x_edges = best["_x_edges"]
        y_edges = best["_y_edges"]
        dem_res = x_edges[1] - x_edges[0]

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
        dod_masked = np.ma.masked_invalid(dod_zoom)
        im = ax_map.pcolormesh(x_edges_zoom, y_edges_zoom, dod_masked,
                               cmap="RdBu", vmin=-vmax, vmax=vmax,
                               shading="flat")
        ax_map.set_facecolor("white")
        cb = plt.colorbar(im, ax=ax_map, shrink=0.8, pad=0.02)
        cb.set_label("DoD elevation change (m)", fontsize=9)

        ax_map.set_xlabel("Easting (m)")
        ax_map.set_ylabel("Northing (m)")
        ratio_str = (f"{best['ratio']:.1f}"
                     if np.isfinite(best["ratio"]) else "inf")
        ax_map.set_title(
            f"(b) DoD - Rank #1: "
            f"{best['start_date']} -> {best['end_date']}\n"
            f"DoD: {best['V_DoD_ero']:.1f} m\u00b3 vs "
            f"M3C2: {best['V_M3C2']:.1f} m\u00b3 "
            f"({ratio_str}\u00d7)",
            fontweight="bold", loc="left", fontsize=8)
        ax_map.set_aspect("equal")
        ax_map.ticklabel_format(useOffset=False, style="plain")
        ax_map.tick_params(labelsize=7, labelrotation=30)

    # -- Panel C: schematic cross-section --
    ax_xs = fig.add_subplot(gs[2])
    ax_xs.set_xlim(-2, 12)
    ax_xs.set_ylim(-1, 14)
    ax_xs.set_aspect("equal")
    ax_xs.axis("off")
    ax_xs.set_title("(c) Why DEMs miss\n     cliff-face change",
                     fontweight="bold", loc="left", fontsize=9)

    cliff_x = [0, 0, 0.5, 0.5, 0, 0, 8, 8]
    cliff_z = [0, 4, 4, 9, 9, 12, 12, 0]
    ax_xs.fill(cliff_x, cliff_z, color="#d9c6a5", edgecolor="#333",
               linewidth=1.5, label="Cliff (before)")

    notch_x = [0, 0.8, 0.8, 0]
    notch_z = [4, 4, 9, 9]
    ax_xs.fill(notch_x, notch_z, color="#ef8a62", edgecolor="#b2182b",
               linewidth=1.5, alpha=0.8, label="Eroded volume")

    ax_xs.fill([-2, 0, 0, -2], [0, 0, -0.5, -0.5],
               color="#f0e6c8", edgecolor="#aaa")
    ax_xs.text(-1, -0.3, "Beach", ha="center", fontsize=7, color="#888")

    ax_xs.annotate("", xy=(4, 12.8), xytext=(4, 13.8),
                   arrowprops=dict(arrowstyle="-|>", color="#2166ac", lw=2))
    ax_xs.text(4, 14.0, "DEM view\n(top-down)", ha="center", va="bottom",
               fontsize=8, color="#2166ac", fontweight="bold")

    ax_xs.annotate("", xy=(0.3, 6.5), xytext=(-1.8, 6.5),
                   arrowprops=dict(arrowstyle="-|>", color="#b2182b", lw=2))
    ax_xs.text(-1.8, 7.2, "M3C2 view\n(face-normal)",
               ha="center", fontsize=8, color="#b2182b", fontweight="bold")

    ax_xs.text(4, 11.5, "Cliff top \u2014 no change\nvisible from above",
               ha="center", fontsize=7, style="italic", color="#555")
    ax_xs.text(1.8, 6.5, "Face scar\n(M3C2 only)",
               ha="center", fontsize=7, fontweight="bold", color="#b2182b")

    out_png = os.path.join(dem_dir, "summary_ratio.png")
    out_pdf = os.path.join(dem_dir, "summary_ratio.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.savefig(out_pdf, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")
    plt.close()


# ==========================================================================
#  CLI
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Find events with the largest M3C2/DoD discrepancy.")
    parser.add_argument("--min_volume", type=float, default=MIN_VOLUME,
                        help=f"Min M3C2 event volume in m3 (default: "
                             f"{MIN_VOLUME})")
    parser.add_argument("--n_top", type=int, default=15,
                        help="Number of top-ratio pairs to output "
                             "(default: 15)")
    parser.add_argument("--cols_per_page", type=int, default=3,
                        help="Columns per multi-panel figure page "
                             "(default: 3)")
    parser.add_argument("--dem_res", type=float, default=DEM_RES,
                        help=f"DEM cell size in metres (default: {DEM_RES})")
    parser.add_argument("--lod", type=float, default=LOD_THRESHOLD,
                        help=f"LoD threshold in metres (default: "
                             f"{LOD_THRESHOLD})")
    parser.add_argument("--no_figure", action="store_true",
                        help="Skip figure generation, print table only")
    args = parser.parse_args()

    all_results, top_results = run_comparison(
        min_volume=args.min_volume,
        n_top=args.n_top,
        dem_res=args.dem_res,
        lod=args.lod,
    )

    if not args.no_figure:
        make_event_figures(top_results)
        make_multi_panel_figures(top_results, cols_per_page=args.cols_per_page)
        make_comparison_table(top_results)
        make_summary_figure(top_results)


if __name__ == "__main__":
    main()
