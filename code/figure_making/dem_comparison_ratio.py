#!/usr/bin/env python3
"""
dem_comparison_ratio.py

Per-event comparison of M3C2 cliff-facing volumes versus spatially-matched
DEM-of-Difference (DoD) volumes.

For each of the top-N events (ranked by M3C2 volume):
  1. Uses the polygon shapefile to convert the event's alongshore extent
     into a UTM bounding box.
  2. Computes a top-down DoD from the noveg point clouds (cached per date pair).
  3. Clips the DoD to the event's UTM footprint.
  4. Compares the spatially-matched DoD erosion volume to the M3C2 event volume.

This replaces the earlier whole-scene DoD approach, which inflated DoD volumes
by including erosion from the entire study area rather than just the event
footprint.

Usage:
    python3 dem_comparison_ratio.py                          # top 15, 5 pages of 3
    python3 dem_comparison_ratio.py --min_volume 100 --n_top 10
    python3 dem_comparison_ratio.py --cols_per_page 5        # wider pages
    python3 dem_comparison_ratio.py --buffer 30              # larger footprint buffer
    python3 dem_comparison_ratio.py --no_figure
"""

import os
import glob
import platform
import argparse
import numpy as np
import pandas as pd
import laspy
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
POLY_RES = "25cm"       # polygon shapefile resolution
FOOTPRINT_BUFFER = 20.0  # m - buffer around event footprint for DoD clipping


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


# ==========================================================================
#  POLYGON UTM MAPPING
# ==========================================================================

def load_polygon_mapping():
    """Load the polygon shapefile and build alongshore-to-UTM mapping.

    The polygon shapefile defines cross-shore strips along the cliff face.
    Each polygon's centroid gives a (easting, northing) position, and the
    alongshore coordinate is (northing - min_northing), matching the logic
    in make_event_lists.py.

    Returns
    -------
    dict with keys:
        sorted_along : 1D array of alongshore positions (m)
        sorted_easting : 1D array of easting values at each position
        sorted_northing : 1D array of northing values at each position
    """
    base_util = os.path.join(BASE, 'utilities')
    sf_root = os.path.join(base_util, 'shape_files')

    candidates = [
        d for d in os.listdir(sf_root)
        if d.lower().startswith(LOCATION.lower())
        and 'polygon' in d.lower()
        and f'at{POLY_RES}'.lower() in d.lower()
        and os.path.isdir(os.path.join(sf_root, d))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No polygon shapefile found for {LOCATION} at {POLY_RES} "
            f"in {sf_root}")

    fld = candidates[0]
    shp_path = os.path.join(sf_root, fld, fld + '.shp')
    print(f"  Shapefile: {shp_path}")

    gdf = gpd.read_file(shp_path)
    centroids = gdf.geometry.centroid

    eastings = np.array([c.x for c in centroids])
    northings = np.array([c.y for c in centroids])

    # Alongshore = northing - min_northing (same as make_event_lists.py)
    min_y = northings.min()
    alongshore = northings - min_y

    order = np.argsort(alongshore)

    return {
        'sorted_along': alongshore[order],
        'sorted_easting': eastings[order],
        'sorted_northing': northings[order],
    }


def get_event_utm_bbox(event, poly_map, buffer_m=FOOTPRINT_BUFFER):
    """Get UTM bounding box for an event from its alongshore extent.

    Finds polygon centroids within the event's alongshore range,
    then returns their easting/northing extent plus a buffer.

    Returns
    -------
    dict with x_min, x_max, y_min, y_max in UTM metres, or None.
    """
    along_start = event['alongshore_start_m']
    along_end = event['alongshore_end_m']

    sa = poly_map['sorted_along']
    se = poly_map['sorted_easting']
    sn = poly_map['sorted_northing']

    mask = (sa >= along_start - buffer_m) & (sa <= along_end + buffer_m)

    if not np.any(mask):
        return None

    return {
        'x_min': float(se[mask].min() - buffer_m),
        'x_max': float(se[mask].max() + buffer_m),
        'y_min': float(sn[mask].min() - buffer_m),
        'y_max': float(sn[mask].max() + buffer_m),
    }


def clip_dod_to_bbox(dod, x_edges, y_edges, bbox, cell_area):
    """Clip DoD to a UTM bounding box and compute erosion volume.

    Returns
    -------
    v_ero : float, erosion volume (m3) within the bounding box
    clip_info : dict with 'dod', 'x_edges', 'y_edges' for figures, or None
    """
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    col_mask = (x_centers >= bbox['x_min']) & (x_centers <= bbox['x_max'])
    row_mask = (y_centers >= bbox['y_min']) & (y_centers <= bbox['y_max'])

    if not np.any(col_mask) or not np.any(row_mask):
        return 0.0, None

    ci = np.where(col_mask)[0]
    ri = np.where(row_mask)[0]

    dod_clip = dod[ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1]

    ero_cells = dod_clip < 0
    if np.any(ero_cells):
        v_ero = float(np.nansum(np.abs(dod_clip[ero_cells])) * cell_area)
    else:
        v_ero = 0.0

    clip_info = {
        'dod': dod_clip,
        'x_edges': x_edges[ci[0]:ci[-1] + 2],
        'y_edges': y_edges[ri[0]:ri[-1] + 2],
    }

    return v_ero, clip_info


# ==========================================================================
#  MAIN COMPARISON
# ==========================================================================

def run_comparison(min_volume=MIN_VOLUME, n_top=15, dem_res=DEM_RES,
                   lod=LOD_THRESHOLD, buffer_m=FOOTPRINT_BUFFER):
    """Run per-event DEM-vs-M3C2 comparison with spatially-matched DoD.

    For each of the top-N events (by M3C2 volume):
      1. Maps event footprint to UTM using the polygon shapefile.
      2. Computes DoD from noveg LAS files (cached per date pair).
      3. Clips DoD to the event's UTM footprint.
      4. Reports spatially-matched DoD erosion vs M3C2 volume.

    Returns
    -------
    DataFrame with one row per event (includes private _dod_clip columns
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

    # Load polygon shapefile for UTM mapping
    print("\nLoading polygon shapefile for UTM mapping ...")
    poly_map = load_polygon_mapping()
    print(f"  {len(poly_map['sorted_along'])} polygon centroids loaded")

    print(f"\n{'='*70}")
    print(f"Per-Event DEM-of-Difference vs M3C2 \u2014 Del Mar")
    print(f"DEM res: {dem_res} m  |  LoD: {lod} m  |  "
          f"Footprint buffer: {buffer_m} m")
    print(f"{'='*70}\n")

    cell_area = dem_res * dem_res
    dod_cache = {}  # (d1, d2) -> (dod, x_edges, y_edges)

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

        # --- Event UTM bounding box ---
        bbox = get_event_utm_bbox(ev, poly_map, buffer_m=buffer_m)
        if bbox is None:
            print(f"  !! No polygons in alongshore range, skipping")
            results.append({
                "rank": rank,
                "start_date": d1_str, "end_date": d2_str,
                "V_M3C2": ev["volume"], "V_DoD": np.nan,
                "ratio": np.nan, "status": "no_polygons",
                "alongshore_start_m": ev["alongshore_start_m"],
                "alongshore_end_m": ev["alongshore_end_m"],
                "elevation": ev["elevation"],
                "height": ev["height"],
                "width": ev["width"],
            })
            continue

        bbox_w = bbox['x_max'] - bbox['x_min']
        bbox_h = bbox['y_max'] - bbox['y_min']
        print(f"  UTM bbox: E[{bbox['x_min']:.0f}, {bbox['x_max']:.0f}] "
              f"N[{bbox['y_min']:.0f}, {bbox['y_max']:.0f}]  "
              f"({bbox_w:.0f} x {bbox_h:.0f} m)")

        # --- Compute / cache DoD for this date pair ---
        if pair_key not in dod_cache:
            f1 = find_noveg_file(d1_str)
            f2 = find_noveg_file(d2_str)
            if f1 is None or f2 is None:
                print(f"  !! Missing noveg file(s): d1={f1}, d2={f2}")
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

            dod = compute_dod(dem1, dem2, lod)
            del dem1, dem2

            dod_cache[pair_key] = (dod, x_edges, y_edges)
            print(f"  DoD computed and cached")
        else:
            print(f"  Using cached DoD for {d1_str} -> {d2_str}")

        # --- Clip DoD to event footprint ---
        dod, x_edges, y_edges = dod_cache[pair_key]
        v_dod, clip_info = clip_dod_to_bbox(
            dod, x_edges, y_edges, bbox, cell_area)

        ratio = ev["volume"] / v_dod if v_dod > 0 else np.inf

        print(f"  V_DoD (footprint) = {v_dod:.1f} m\u00b3")
        print(f"  V_M3C2            = {ev['volume']:.1f} m\u00b3")
        print(f"  Ratio (M3C2/DoD)  = {ratio:.1f}x")
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
            "_dod_clip": clip_info["dod"] if clip_info else None,
            "_x_edges": clip_info["x_edges"] if clip_info else None,
            "_y_edges": clip_info["y_edges"] if clip_info else None,
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
    """Save a zoomed DoD figure for each event, clipped to its footprint."""
    dem_dir = os.path.join(FIG_DIR, "dems_ratio")
    os.makedirs(dem_dir, exist_ok=True)

    ok = results_df[results_df["status"] == "ok"]

    for _, row in ok.iterrows():
        rank = int(row["rank"])
        dod_clip = row["_dod_clip"]
        x_edges = row["_x_edges"]
        y_edges = row["_y_edges"]

        if dod_clip is None:
            print(f"  Rank {rank}: no clipped DoD, skipping figure")
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        dod_masked = np.ma.masked_invalid(dod_clip)
        dod_abs_max = max(abs(np.nanmin(dod_clip)),
                         abs(np.nanmax(dod_clip)), 0.5)
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
            f"(DoD clipped to event footprint)\n"
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
    Bottom row: DoD (top-down) clipped to the same event footprint.

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

            # -- Bottom row: DoD clipped to event footprint --
            dod_clip = row["_dod_clip"]
            xedges = row["_x_edges"]
            yedges = row["_y_edges"]

            if dod_clip is not None:
                vabs = max(abs(np.nanmin(dod_clip)),
                           abs(np.nanmax(dod_clip)), 0.5)

                # Negate so erosion is positive (red), matching M3C2 row
                dz_masked = np.ma.masked_invalid(-dod_clip)
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
                ax_bot.text(0.5, 0.5, "No DoD data", ha='center',
                            va='center', transform=ax_bot.transAxes,
                            fontsize=7)

            ax_bot.set_title(
                f"DoD (footprint): {row['V_DoD']:.1f} m\u00b3",
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
            "DoD clipped to event footprint",
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
               color="#b2182b", label="DoD (footprint)")
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

    # -- Panel B: DoD map of #1 event (clipped to footprint) --
    ax_map = fig.add_subplot(gs[1])
    best = ok.iloc[0]

    dod_clip = best["_dod_clip"]
    if dod_clip is not None and not np.all(np.isnan(dod_clip)):
        x_edges = best["_x_edges"]
        y_edges = best["_y_edges"]

        vmax = max(abs(np.nanmin(dod_clip)), abs(np.nanmax(dod_clip)), 0.5)
        dod_masked = np.ma.masked_invalid(dod_clip)
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


# ==========================================================================
#  CLI
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-event M3C2 vs spatially-matched DoD comparison.")
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
    parser.add_argument("--buffer", type=float, default=FOOTPRINT_BUFFER,
                        help=f"Buffer around event footprint in metres "
                             f"(default: {FOOTPRINT_BUFFER})")
    parser.add_argument("--no_figure", action="store_true",
                        help="Skip figure generation, print table only")
    args = parser.parse_args()

    results = run_comparison(
        min_volume=args.min_volume,
        n_top=args.n_top,
        dem_res=args.dem_res,
        lod=args.lod,
        buffer_m=args.buffer,
    )

    if not args.no_figure:
        ok = results[results["status"] == "ok"]
        if not ok.empty:
            make_event_figures(results)
            make_multi_panel_figures(results,
                                    cols_per_page=args.cols_per_page)
            make_comparison_table(results)
            make_summary_figure(results)


if __name__ == "__main__":
    main()
