#!/usr/bin/env python3
"""
dem_comparison_ratio.py

Per-event comparison of M3C2 cliff-facing volumes versus DEM-of-Difference
(DoD) volumes.

For each of the top-N events (ranked by M3C2 volume):
  1. Computes a top-down DoD from the noveg point clouds (cached per date pair).
  2. Finds the largest erosion cluster in the DoD via connected-component
     labeling and zooms to its bounding box.
  3. Computes DoD erosion volume within that bounding box only.
  4. Compares to the M3C2 event volume.

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
from scipy.interpolate import griddata

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
#  HELPERS
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


def fill_dem_nans(dem):
    """Fill NaN gaps in a DEM using linear interpolation.

    Uses scipy.interpolate.griddata (linear) to interpolate over interior
    holes.  Cells outside the convex hull of valid data remain NaN.

    Returns
    -------
    filled : 2-D ndarray, same shape as *dem*.
    n_filled : int, number of NaN cells that were filled.
    """
    ny, nx = dem.shape
    valid = ~np.isnan(dem)
    n_valid = int(valid.sum())

    if n_valid == 0 or n_valid == ny * nx:
        return dem.copy(), 0

    gy, gx = np.mgrid[0:ny, 0:nx]
    points = np.column_stack((gy[valid], gx[valid]))
    values = dem[valid]

    nan_mask = np.isnan(dem)
    interp_pts = np.column_stack((gy[nan_mask], gx[nan_mask]))

    filled_vals = griddata(points, values, interp_pts, method='linear')

    filled = dem.copy()
    filled[nan_mask] = filled_vals
    n_filled = int(np.isfinite(filled_vals).sum())

    return filled, n_filled


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


def find_largest_dod_cluster(dod, dem_res, buffer_m=10.0):
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
#  MAIN COMPARISON
# ==========================================================================

def run_comparison(min_volume=MIN_VOLUME, n_top=15, dem_res=DEM_RES,
                   lod=LOD_THRESHOLD):
    """Run per-event DEM-vs-M3C2 comparison.

    For each of the top-N events (by M3C2 volume):
      1. Computes DoD from noveg LAS files (cached per date pair).
      2. Finds the largest erosion cluster via connected-component labeling.
      3. Computes DoD erosion volume within that cluster's bounding box.
      4. Reports bounding-box DoD volume vs M3C2 volume.

    Returns
    -------
    DataFrame with one row per event (includes private _dod_* columns
    for figure generation).
    """
    qc_csv = find_qc_csv()
    print(f"Using QC file: {os.path.basename(qc_csv)}")

    df = pd.read_csv(qc_csv,
                     parse_dates=["mid_date", "start_date", "end_date"])
    real = df[df["qc_flag"] == "real"].copy()

    # Filter and rank by M3C2 volume
    big = real[real["volume"] >= min_volume].copy()
    n_eligible = len(big)
    big = (big.sort_values("volume", ascending=False)
           .head(n_top)
           .reset_index(drop=True))

    print(f"Real events >= {min_volume} m\u00b3: {n_eligible}")
    print(f"Processing top {len(big)} events")

    print(f"\n{'='*70}")
    print(f"Per-Event DEM-of-Difference vs M3C2 \u2014 Del Mar")
    print(f"DEM res: {dem_res} m  |  LoD: {lod} m")
    print(f"{'='*70}\n")

    cell_area = dem_res * dem_res
    # Cache per date pair: (dod, x_edges, y_edges, r_slice, c_slice, bbox_vol)
    dod_cache = {}

    results = []
    for i, (_, ev) in enumerate(big.iterrows()):
        d1_str = ev["start_date"].strftime("%Y%m%d")
        d2_str = ev["end_date"].strftime("%Y%m%d")
        pair_key = (d1_str, d2_str)
        rank = i + 1

        print(f"[{rank}/{len(big)}] {d1_str} -> {d2_str}  "
              f"(M3C2: {ev['volume']:.1f} m\u00b3)")
        print(f"  Alongshore: {ev['alongshore_start_m']:.0f} - "
              f"{ev['alongshore_end_m']:.0f} m  "
              f"(width: {ev['width']:.0f} m)")

        # --- Compute / cache DoD + cluster for this date pair ---
        if pair_key not in dod_cache:
            f1 = find_noveg_file(d1_str)
            f2 = find_noveg_file(d2_str)
            if f1 is None or f2 is None:
                print(f"  !! Missing noveg file(s): d1={f1}, d2={f2}")
                dod_cache[pair_key] = None
                results.append({
                    "rank": rank,
                    "start_date": d1_str, "end_date": d2_str,
                    "V_M3C2": ev["volume"], "V_DoD": np.nan,
                    "ratio": np.nan, "status": "missing_file",
                    "alongshore_start_m": ev["alongshore_start_m"],
                    "alongshore_end_m": ev["alongshore_end_m"],
                    "elevation": ev["elevation"],
                    "height": ev["height"],
                    "width": ev["width"],
                })
                continue

            print(f"  Loading {os.path.basename(f1)} ...")
            x1, y1, z1 = load_full(f1)
            print(f"    -> {len(x1):,} points")

            print(f"  Loading {os.path.basename(f2)} ...")
            x2, y2, z2 = load_full(f2)
            print(f"    -> {len(x2):,} points")

            all_x = np.concatenate([x1, x2])
            all_y = np.concatenate([y1, y2])
            x_min, x_max = all_x.min(), all_x.max()
            y_min, y_max = all_y.min(), all_y.max()
            del all_x, all_y

            nx = int(np.ceil((x_max - x_min) / dem_res))
            ny = int(np.ceil((y_max - y_min) / dem_res))
            x_edges = np.linspace(x_min, x_min + nx * dem_res, nx + 1)
            y_edges = np.linspace(y_min, y_min + ny * dem_res, ny + 1)

            print(f"  Grid: {nx} x {ny} cells ({nx * ny:,} total)")
            print(f"  Rasterising DEMs ...")

            dem1 = rasterise_to_common_grid(x1, y1, z1, x_min, y_min,
                                            dem_res, nx, ny)
            del x1, y1, z1
            dem2 = rasterise_to_common_grid(x2, y2, z2, x_min, y_min,
                                            dem_res, nx, ny)
            del x2, y2, z2

            # Fill NaN gaps via linear interpolation before DoD
            dem1, n1 = fill_dem_nans(dem1)
            dem2, n2 = fill_dem_nans(dem2)
            print(f"  Interpolated DEM gaps: {n1:,} + {n2:,} cells filled")

            dod = compute_dod(dem1, dem2, lod)
            del dem1, dem2

            # Find largest erosion cluster and its bounding box
            r_slice, c_slice, cluster_vol, n_clusters = \
                find_largest_dod_cluster(dod, dem_res)

            if r_slice is not None:
                dod_zoom = dod[r_slice, c_slice]
                x_edges_zoom = x_edges[c_slice.start:c_slice.stop + 1]
                y_edges_zoom = y_edges[r_slice.start:r_slice.stop + 1]

                # Volume = all erosion within the bounding box
                ero_mask = dod_zoom < 0
                bbox_vol = float(np.nansum(np.abs(dod_zoom[ero_mask]))
                                 * cell_area) if np.any(ero_mask) else 0.0

                print(f"  Largest DoD cluster: {cluster_vol:.1f} m\u00b3 "
                      f"(of {n_clusters} clusters)")
                print(f"  Bounding box erosion: {bbox_vol:.1f} m\u00b3")

                dod_cache[pair_key] = {
                    'dod_zoom': dod_zoom,
                    'x_edges': x_edges_zoom,
                    'y_edges': y_edges_zoom,
                    'bbox_vol': bbox_vol,
                    'cluster_vol': cluster_vol,
                    'n_clusters': n_clusters,
                }
            else:
                print(f"  No erosion clusters found in DoD")
                dod_cache[pair_key] = {
                    'dod_zoom': None,
                    'x_edges': None,
                    'y_edges': None,
                    'bbox_vol': 0.0,
                    'cluster_vol': 0.0,
                    'n_clusters': 0,
                }
        elif dod_cache[pair_key] is None:
            # Previously failed (missing file)
            results.append({
                "rank": rank,
                "start_date": d1_str, "end_date": d2_str,
                "V_M3C2": ev["volume"], "V_DoD": np.nan,
                "ratio": np.nan, "status": "missing_file",
                "alongshore_start_m": ev["alongshore_start_m"],
                "alongshore_end_m": ev["alongshore_end_m"],
                "elevation": ev["elevation"],
                "height": ev["height"],
                "width": ev["width"],
            })
            continue
        else:
            print(f"  Using cached DoD for {d1_str} -> {d2_str}")

        # --- Use the cluster bounding box volume ---
        cached = dod_cache[pair_key]
        v_dod = cached['bbox_vol']
        ratio = ev["volume"] / v_dod if v_dod > 0 else np.inf

        print(f"  V_DoD (bbox)  = {v_dod:.1f} m\u00b3")
        print(f"  V_M3C2        = {ev['volume']:.1f} m\u00b3")
        print(f"  Ratio (M3C2/DoD) = {ratio:.1f}x")
        print()

        results.append({
            "rank": rank,
            "start_date": d1_str, "end_date": d2_str,
            "V_M3C2": ev["volume"],
            "V_DoD": v_dod,
            "ratio": ratio,
            "status": "ok",
            "alongshore_start_m": ev["alongshore_start_m"],
            "alongshore_end_m": ev["alongshore_end_m"],
            "elevation": ev["elevation"],
            "height": ev["height"],
            "width": ev["width"],
            "_dod_zoom": cached['dod_zoom'],
            "_x_edges": cached['x_edges'],
            "_y_edges": cached['y_edges'],
        })

    all_results = pd.DataFrame(results)
    ok = all_results[all_results["status"] == "ok"].copy()

    if ok.empty:
        print("No valid events processed.")
        return all_results

    print(f"\n{'='*70}")
    print(f"TOP {len(ok)} EVENTS BY M3C2 VOLUME "
          f"(of {n_eligible} eligible)")
    print(f"{'='*70}")
    display_cols = ["rank", "start_date", "end_date",
                    "V_M3C2", "V_DoD", "ratio"]
    print(ok[display_cols].to_string(index=False, float_format="%.1f"))
    print()

    return all_results


# ==========================================================================
#  FIGURES
# ==========================================================================

def make_event_figures(results_df):
    """Save a zoomed DoD figure for each event."""
    dem_dir = os.path.join(FIG_DIR, "dems_ratio")
    os.makedirs(dem_dir, exist_ok=True)

    ok = results_df[results_df["status"] == "ok"]

    for _, row in ok.iterrows():
        rank = int(row["rank"])
        dod_zoom = row["_dod_zoom"]
        x_edges = row["_x_edges"]
        y_edges = row["_y_edges"]

        if dod_zoom is None:
            print(f"  Rank {rank}: no DoD erosion, skipping figure")
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        dod_masked = np.ma.masked_invalid(dod_zoom)
        dod_abs_max = max(abs(np.nanmin(dod_zoom)),
                         abs(np.nanmax(dod_zoom)), 0.5)
        im = ax.pcolormesh(x_edges, y_edges, dod_masked,
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
            f"DoD erosion = {row['V_DoD']:.1f} m\u00b3  |  "
            f"M3C2 erosion = {row['V_M3C2']:.1f} m\u00b3  |  "
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

    print(f"\nAll event DoD figures saved to: {dem_dir}")


def make_multi_panel_figures(results_df, cols_per_page=3):
    """Create 2xN panel figures: M3C2 grid (top) vs DoD (bottom) per event.

    Top row: M3C2 cliff-facing grid zoomed to the event footprint.
    Bottom row: DoD (top-down) zoomed to the largest erosion cluster.

    Chunks events into pages of `cols_per_page` columns each.
    """
    dem_dir = os.path.join(FIG_DIR, "dems_ratio")
    os.makedirs(dem_dir, exist_ok=True)

    ok = results_df[results_df["status"] == "ok"].copy()
    n_total = len(ok)
    if n_total == 0:
        print("No valid events for multi-panel figure.")
        return

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
          f"({cols_per_page} columns each, {n_total} total events) ...")

    rows_list = list(ok.iterrows())

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

            # -- Top row: M3C2 cliff-facing grid --
            tidx = next((i for i, ds in enumerate(dstrings)
                         if ds == dfolder), None)

            if tidx is not None and ero_3d is not None:
                ero2d = ero_3d[:, :, tidx][asort, :]
                combined = np.nan_to_num(ero2d, nan=0.0)
                if dep_3d is not None:
                    dep2d = dep_3d[:, :, tidx][asort, :]
                    dc = np.nan_to_num(dep2d, nan=0.0)
                    dep_mask = (dc > 0) & (combined < 0.01)
                    combined[dep_mask] = -dc[dep_mask]

                # Zoom to event footprint
                xpad, ypb, ypt = 10, 8, 3
                xlo = row['alongshore_start_m'] - xpad
                xhi = row['alongshore_end_m'] + xpad
                ylo = max(0, row['elevation'] - row['height'] / 2 - ypb)
                yhi = row['elevation'] + row['height'] / 2 + ypt

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

            # -- Bottom row: DoD zoomed to largest cluster --
            dod_zoom = row["_dod_zoom"]
            xedges = row["_x_edges"]
            yedges = row["_y_edges"]

            if dod_zoom is not None:
                vabs = max(abs(np.nanmin(dod_zoom)),
                           abs(np.nanmax(dod_zoom)), 0.5)

                # Negate so erosion is positive (red), matching M3C2 row
                dz_masked = np.ma.masked_invalid(-dod_zoom)
                im2 = ax_bot.pcolormesh(xedges, yedges, dz_masked,
                                        cmap="RdBu_r",
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
                f"DoD (largest cluster): {row['V_DoD']:.1f} m\u00b3",
                fontsize=9, fontweight='bold')

        axes[0, 0].set_ylabel("M3C2 Grid\n(cliff-facing)", fontsize=10,
                              fontweight='bold')
        axes[1, 0].set_ylabel("DoD\n(top-down)", fontsize=10,
                              fontweight='bold')

        rank_lo = int(ok.iloc[start]["rank"])
        rank_hi = int(ok.iloc[end - 1]["rank"])
        fig.suptitle(
            f"Per-Event M3C2 vs DoD \u2014 Ranks #{rank_lo}-{rank_hi} "
            f"(Del Mar)\n"
            "Red = erosion, Blue = deposition  |  "
            "DoD zoomed to largest erosion cluster",
            fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        out = os.path.join(dem_dir,
                           f"multi_panel_ratio_{page + 1}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close()
        print(f"  Saved: {out}  (ranks #{rank_lo}-{rank_hi})")


def make_comparison_table(results_df):
    """Create a publication-quality comparison table."""
    dem_dir = os.path.join(FIG_DIR, "dems_ratio")
    os.makedirs(dem_dir, exist_ok=True)

    ok = results_df[results_df["status"] == "ok"]

    rows = []
    for _, r in ok.iterrows():
        d1_fmt = pd.Timestamp(r["start_date"]).strftime("%Y-%m-%d")
        d2_fmt = pd.Timestamp(r["end_date"]).strftime("%Y-%m-%d")
        ratio_str = (f"{r['ratio']:.1f}\u00d7"
                     if np.isfinite(r["ratio"]) else "\u221e")

        if np.isfinite(r["V_DoD"]) and r["V_DoD"] > 0:
            diff_pct = (r["V_M3C2"] - r["V_DoD"]) / r["V_DoD"] * 100
            diff_str = f"+{diff_pct:.0f}%" if diff_pct >= 0 else f"{diff_pct:.0f}%"
        else:
            diff_str = "\u2014"

        rows.append([
            str(int(r["rank"])),
            f"{d1_fmt}  \u2192  {d2_fmt}",
            f"{r['V_M3C2']:.1f}",
            f"{r['V_DoD']:.1f}" if np.isfinite(r["V_DoD"]) else "\u2014",
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


def make_summary_figure(results_df):
    """Summary figure: bar chart + DoD map of #1 event + schematic."""
    dem_dir = os.path.join(FIG_DIR, "dems_ratio")
    os.makedirs(dem_dir, exist_ok=True)

    ok = results_df[results_df["status"] == "ok"].copy()
    if ok.empty:
        print("No valid events for summary figure.")
        return

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.2, 0.8], wspace=0.35)

    # -- Panel A: bar chart --
    ax_bar = fig.add_subplot(gs[0])
    x_pos = np.arange(len(ok))
    w = 0.35
    ax_bar.bar(x_pos - w/2, ok["V_M3C2"].values, w,
               color="#2166ac", label="M3C2 Grid")
    ax_bar.bar(x_pos + w/2, ok["V_DoD"].values, w,
               color="#b2182b", label="DoD (cluster)")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(
        [f"#{int(r)}" for r in ok["rank"]], fontsize=8)
    ax_bar.set_ylabel("Erosion Volume (m\u00b3)")
    ax_bar.set_xlabel("Rank (by M3C2 volume)")
    ax_bar.legend(frameon=False)
    ax_bar.set_title("(a) Per-event volume comparison",
                     fontweight="bold", loc="left", fontsize=9)

    for j, (_, r) in enumerate(ok.iterrows()):
        ratio_str = (f"{r['ratio']:.1f}\u00d7"
                     if np.isfinite(r["ratio"]) else "\u221e")
        y_top = max(r["V_M3C2"], r["V_DoD"])
        ax_bar.text(j, y_top * 1.05, ratio_str, ha="center", va="bottom",
                    fontsize=7, color="#333")

    y_ceil = max(ok["V_M3C2"].max(),
                 ok["V_DoD"].max()) * 1.25
    ax_bar.set_ylim(0, y_ceil)

    # -- Panel B: DoD map of #1 event --
    ax_map = fig.add_subplot(gs[1])
    best = ok.iloc[0]

    dod_zoom = best["_dod_zoom"]
    if dod_zoom is not None and not np.all(np.isnan(dod_zoom)):
        x_edges = best["_x_edges"]
        y_edges = best["_y_edges"]

        vmax = max(abs(np.nanmin(dod_zoom)), abs(np.nanmax(dod_zoom)), 0.5)
        dod_masked = np.ma.masked_invalid(dod_zoom)
        im = ax_map.pcolormesh(x_edges, y_edges, dod_masked,
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
            f"DoD: {best['V_DoD']:.1f} m\u00b3 vs "
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


def make_scatter_figure(results_df):
    """Scatter plot of M3C2 vs DoD volume, one point per date pair.

    Keeps only the largest M3C2 event per date pair to avoid duplicate
    DoD values. Points are coloured by elevation centroid and sized by
    M3C2 event volume. A 1:1 reference line shows where the two methods
    agree.
    """
    import matplotlib.colors as mcolors

    dem_dir = os.path.join(FIG_DIR, "dems_ratio")
    os.makedirs(dem_dir, exist_ok=True)

    ok = results_df[results_df["status"] == "ok"].copy()
    if ok.empty:
        print("No valid events for scatter figure.")
        return

    # Keep only the largest M3C2 event per date pair
    ok["pair_key"] = ok["start_date"] + "_" + ok["end_date"]
    deduped = (ok.sort_values("V_M3C2", ascending=False)
               .drop_duplicates(subset="pair_key", keep="first")
               .copy())

    # Drop any with zero or NaN DoD
    deduped = deduped[deduped["V_DoD"].notna() & (deduped["V_DoD"] > 0)]

    if deduped.empty:
        print("No valid pairs for scatter figure after deduplication.")
        return

    print(f"\nScatter plot: {len(deduped)} unique date pairs "
          f"(from {len(ok)} events)")

    v_m3c2 = deduped["V_M3C2"].values
    v_dod = deduped["V_DoD"].values
    elev = deduped["elevation"].values

    # Circle sizes scaled by M3C2 volume
    size_min, size_max = 30, 400
    vol_min, vol_max = v_m3c2.min(), v_m3c2.max()
    if vol_max > vol_min:
        sizes = size_min + (v_m3c2 - vol_min) / (vol_max - vol_min) * (size_max - size_min)
    else:
        sizes = np.full_like(v_m3c2, (size_min + size_max) / 2)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 7))

    # 1:1 reference line
    axis_max = max(v_m3c2.max(), v_dod.max()) * 1.15
    ax.plot([0, axis_max], [0, axis_max], '--', color='#888',
            linewidth=1.2, zorder=1, label='1:1 line')

    # Scatter
    sc = ax.scatter(v_m3c2, v_dod, c=elev, s=sizes,
                    cmap='plasma', edgecolors='#333', linewidths=0.4,
                    alpha=0.85, zorder=3)

    # Colorbar for elevation
    cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Event Elevation Centroid (m)", fontsize=11)

    # Size legend (3 representative volumes)
    vol_ticks = np.array([50, 200, 500])
    vol_ticks = vol_ticks[vol_ticks <= vol_max * 1.1]
    if len(vol_ticks) == 0:
        vol_ticks = np.array([vol_min, vol_max])
    for vt in vol_ticks:
        if vol_max > vol_min:
            sz = size_min + (vt - vol_min) / (vol_max - vol_min) * (size_max - size_min)
        else:
            sz = (size_min + size_max) / 2
        ax.scatter([], [], s=sz, c='#ccc', edgecolors='#333',
                   linewidths=0.4, label=f'{int(vt)} m\u00b3')

    # Region labels (close to 1:1 line)
    ax.text(axis_max * 0.75, axis_max * 0.55,
            "2.75D Grid > DoD",
            fontsize=10, color='#555', ha='center', style='italic',
            rotation=45, rotation_mode='anchor')
    ax.text(axis_max * 0.55, axis_max * 0.75,
            "DoD > 2.75D Grid",
            fontsize=10, color='#555', ha='center', style='italic',
            rotation=45, rotation_mode='anchor')

    ax.set_xlabel("2.75D Grid Volume, M3C2 (m\u00b3)", fontsize=12)
    ax.set_ylabel("DoD Volume (m\u00b3)", fontsize=12)

    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', frameon=True, framealpha=0.9,
              fontsize=9)

    plt.tight_layout()

    out_png = os.path.join(dem_dir, "scatter_m3c2_vs_dod.png")
    out_pdf = os.path.join(dem_dir, "scatter_m3c2_vs_dod.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.savefig(out_pdf, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")


# ==========================================================================
#  CLI
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-event M3C2 vs DoD comparison "
                    "(DoD volume from largest erosion cluster).")
    parser.add_argument("--min_volume", type=float, default=MIN_VOLUME,
                        help=f"Min M3C2 event volume in m3 (default: "
                             f"{MIN_VOLUME})")
    parser.add_argument("--n_top", type=int, default=15,
                        help="Number of top events to process "
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

    results = run_comparison(
        min_volume=args.min_volume,
        n_top=args.n_top,
        dem_res=args.dem_res,
        lod=args.lod,
    )

    if not args.no_figure:
        ok = results[results["status"] == "ok"]
        if not ok.empty:
            make_multi_panel_figures(results,
                                    cols_per_page=args.cols_per_page)
            make_comparison_table(results)
            make_summary_figure(results)
            make_scatter_figure(results)


if __name__ == "__main__":
    main()
