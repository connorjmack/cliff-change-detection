#!/usr/bin/env python3
"""
dem_comparison.py

Compares M3C2-based volumes (this study) against traditional DEM-of-Difference
(DoD) volumes for the top-N largest QC'd erosion events at Del Mar.

For each event the script:
  1. Loads the two noveg LAS point clouds bracketing the event.
  2. Clips to the event's alongshore footprint (+ buffer).
  3. Rasterises both clouds to top-down DEMs (max Z per XY cell).
  4. Computes DoD = DEM_after - DEM_before, applies a LoD threshold.
  5. Computes V_DoD from the negative (erosion) cells.
  6. Compares V_DoD to V_M3C2 from the QC'd event list.

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
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

# ── paths ──────────────────────────────────────────────────────────────────────
SYSTEM = platform.system()
if SYSTEM == "Darwin":
    BASE = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
else:
    BASE = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

LOCATION = "DelMar"
NOVEG_DIR = os.path.join(BASE, "results", LOCATION, "noveg")
SHP_DIR = os.path.join(BASE, "utilities", "shape_files",
                       "DelMarPolygons595to620at25cm")
SHP_FILE = os.path.join(SHP_DIR, "DelMarPolygons595to620at25cm.shp")

# QC event list (repo-relative)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
QC_CSV = os.path.join(REPO_ROOT, "results", "event_lists_qc", "erosion",
                      "DelMar_events_qc_20260203_085647.csv")

# Filled-grid directory (for spatial map of M3C2 scar)
GRID_BASE = os.path.join(BASE, "results", LOCATION, "erosion")

# Output
FIG_DIR = os.path.join(REPO_ROOT, "figures", "appendix")

# ── constants ──────────────────────────────────────────────────────────────────
DEM_RES = 0.25          # m – top-down DEM cell size (default)
LOD_THRESHOLD = 0.25    # m – Level of Detection for DoD
ALONG_BUFFER = 20.0     # m – buffer around event footprint (UTM Y)
M3C2_CELL_AREA = 0.25 * 0.25   # m² – grid cell area used in M3C2 volumes


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_alongshore_to_utm(shp_path):
    """Return arrays (alongshore_m, centroid_y) sorted by alongshore_m.

    alongshore_m = centroid_Y - min(centroid_Y), matching the data-cube
    coordinate system used in the event lists.
    """
    gdf = gpd.read_file(shp_path)
    cy = gdf.geometry.centroid.y.values
    min_y = cy.min()
    along = cy - min_y
    order = np.argsort(along)
    return along[order], cy[order]


def alongshore_to_utm_y(along_m, along_arr, utm_y_arr):
    """Convert an alongshore_m value to UTM Y via linear interpolation."""
    return np.interp(along_m, along_arr, utm_y_arr)


def find_noveg_file(date_str):
    """Find the noveg LAS file whose name starts with *date_str* (YYYYMMDD).

    Parameters
    ----------
    date_str : str
        Eight-character date, e.g. '20181002'.

    Returns
    -------
    str or None
    """
    pattern = os.path.join(NOVEG_DIR, f"{date_str}*.las")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[0]


def load_and_clip(las_path, y_min, y_max):
    """Load a LAS file and return (x, y, z) clipped to y_min <= y <= y_max."""
    las = laspy.read(las_path)
    x, y, z = las.x, las.y, las.z
    mask = (y >= y_min) & (y <= y_max)
    return x[mask], y[mask], z[mask]


def rasterise_dem(x, y, z, res):
    """Create a top-down DSM (max Z per XY cell).

    Returns
    -------
    dem : 2-D ndarray (ny, nx) with NaN where no data
    x_edges, y_edges : 1-D arrays of bin edges
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Edges
    x_edges = np.arange(x_min, x_max + res, res)
    y_edges = np.arange(y_min, y_max + res, res)
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    # Bin indices
    xi = np.clip(((x - x_min) / res).astype(int), 0, nx - 1)
    yi = np.clip(((y - y_min) / res).astype(int), 0, ny - 1)

    # Max Z per cell
    dem = np.full((ny, nx), np.nan)
    flat_idx = yi * nx + xi
    for idx in range(len(z)):
        fi = flat_idx[idx]
        r, c = divmod(fi, nx)
        if np.isnan(dem[r, c]) or z[idx] > dem[r, c]:
            dem[r, c] = z[idx]

    return dem, x_edges, y_edges


def rasterise_dem_fast(x, y, z, res):
    """Vectorised DSM rasterisation using pandas groupby.

    Much faster than the per-point loop for large clouds.
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_edges = np.arange(x_min, x_max + res, res)
    y_edges = np.arange(y_min, y_max + res, res)
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    xi = np.clip(((x - x_min) / res).astype(int), 0, nx - 1)
    yi = np.clip(((y - y_min) / res).astype(int), 0, ny - 1)

    flat = yi * nx + xi
    df = pd.DataFrame({"flat": flat, "z": z})
    maxz = df.groupby("flat")["z"].max()

    dem = np.full(ny * nx, np.nan)
    dem[maxz.index.values] = maxz.values
    dem = dem.reshape(ny, nx)

    return dem, x_edges, y_edges


def compute_dod(dem_before, dem_after, lod):
    """Compute DEM of Difference, zeroing cells below LoD.

    Returns
    -------
    dod : 2-D array (positive = deposition, negative = erosion), NaN = no data
    """
    # Both DEMs must have data
    valid = ~np.isnan(dem_before) & ~np.isnan(dem_after)
    dod = np.full_like(dem_before, np.nan)
    dod[valid] = dem_after[valid] - dem_before[valid]
    # Apply LoD
    below_lod = np.abs(dod) < lod
    dod[below_lod] = 0.0
    return dod


def dod_erosion_volume(dod, cell_area):
    """Total erosion volume from a DoD (sum of negative cells)."""
    ero_mask = dod < 0
    if not np.any(ero_mask):
        return 0.0
    return float(np.sum(np.abs(dod[ero_mask])) * cell_area)


# ═══════════════════════════════════════════════════════════════════════════════
#  GRID LOADING (for spatial map panel)
# ═══════════════════════════════════════════════════════════════════════════════

def load_m3c2_grid(start_date, end_date):
    """Load the filled erosion grid CSV for a date pair.

    Returns
    -------
    grid : 2-D ndarray (polygons × elevation bins)
    poly_ids : 1-D array of polygon row indices
    elevations : 1-D array of elevation values (m)
    """
    d1 = start_date.strftime("%Y%m%d")
    d2 = end_date.strftime("%Y%m%d")
    folder = os.path.join(GRID_BASE, f"{d1}_to_{d2}", "25cm")
    grid_file = os.path.join(folder,
                             f"{d1}_to_{d2}_ero_grid_25cm_filled.csv")
    if not os.path.isfile(grid_file):
        return None, None, None

    df = pd.read_csv(grid_file, index_col=0, na_values=["", "nan", "NaN"])
    elevations = []
    for c in df.columns:
        try:
            elevations.append(float(c.replace("M3C2_", "").replace("m", "")))
        except ValueError:
            elevations.append(np.nan)
    elevations = np.array(elevations)
    grid = df.values.astype(float)
    poly_ids = df.index.values
    return grid, poly_ids, elevations


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def run_comparison(n_events=5, dem_res=DEM_RES, lod=LOD_THRESHOLD):
    """Run the DEM-vs-M3C2 comparison for the top-N events.

    Returns a DataFrame with one row per event.
    """
    # Load QC events, keep only 'real'
    df = pd.read_csv(QC_CSV, parse_dates=["mid_date", "start_date", "end_date"])
    real = (df[df["qc_flag"] == "real"]
            .sort_values("volume", ascending=False)
            .head(n_events)
            .reset_index(drop=True))

    print(f"\n{'='*70}")
    print(f"DEM-of-Difference vs M3C2 Comparison  —  Del Mar, top {n_events}")
    print(f"DEM resolution: {dem_res} m  |  LoD threshold: {lod} m")
    print(f"{'='*70}\n")

    # Build alongshore → UTM mapping from shapefile
    along_arr, utm_y_arr = build_alongshore_to_utm(SHP_FILE)
    cell_area = dem_res * dem_res

    results = []

    for i, row in real.iterrows():
        d1_str = row["start_date"].strftime("%Y%m%d")
        d2_str = row["end_date"].strftime("%Y%m%d")
        v_m3c2 = row["volume"]

        print(f"Event {i+1}: {d1_str} → {d2_str}  |  V_M3C2 = {v_m3c2:.1f} m³")

        # --- find noveg files ---
        f1 = find_noveg_file(d1_str)
        f2 = find_noveg_file(d2_str)
        if f1 is None or f2 is None:
            print(f"  !! Missing noveg file(s): d1={f1}, d2={f2}")
            results.append({
                "event": i + 1,
                "start_date": d1_str, "end_date": d2_str,
                "V_M3C2": v_m3c2, "V_DoD": np.nan,
                "ratio": np.nan, "status": "missing_file"
            })
            continue

        # --- spatial clip bounds (UTM Y) ---
        y_lo = alongshore_to_utm_y(row["alongshore_start_m"],
                                   along_arr, utm_y_arr) - ALONG_BUFFER
        y_hi = alongshore_to_utm_y(row["alongshore_end_m"],
                                   along_arr, utm_y_arr) + ALONG_BUFFER

        print(f"  Loading {os.path.basename(f1)} ...")
        x1, y1, z1 = load_and_clip(f1, y_lo, y_hi)
        print(f"    → {len(x1):,} points in clip window")

        print(f"  Loading {os.path.basename(f2)} ...")
        x2, y2, z2 = load_and_clip(f2, y_lo, y_hi)
        print(f"    → {len(x2):,} points in clip window")

        if len(x1) == 0 or len(x2) == 0:
            print("  !! No points in clip window")
            results.append({
                "event": i + 1,
                "start_date": d1_str, "end_date": d2_str,
                "V_M3C2": v_m3c2, "V_DoD": np.nan,
                "ratio": np.nan, "status": "no_points"
            })
            continue

        # --- rasterise to DEMs (use common extent) ---
        all_x = np.concatenate([x1, x2])
        all_y = np.concatenate([y1, y2])
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        x_edges = np.arange(x_min, x_max + dem_res, dem_res)
        y_edges = np.arange(y_min, y_max + dem_res, dem_res)
        nx = len(x_edges) - 1
        ny = len(y_edges) - 1

        def _raster(x, y, z):
            xi = np.clip(((x - x_min) / dem_res).astype(int), 0, nx - 1)
            yi = np.clip(((y - y_min) / dem_res).astype(int), 0, ny - 1)
            flat = yi * nx + xi
            df_pts = pd.DataFrame({"flat": flat, "z": z})
            maxz = df_pts.groupby("flat")["z"].max()
            dem = np.full(ny * nx, np.nan)
            dem[maxz.index.values] = maxz.values
            return dem.reshape(ny, nx)

        print(f"  Rasterising DEMs ({nx}×{ny} cells) ...")
        dem1 = _raster(x1, y1, z1)
        dem2 = _raster(x2, y2, z2)

        # --- DoD ---
        dod = compute_dod(dem1, dem2, lod)
        v_dod = dod_erosion_volume(dod, cell_area)
        ratio = v_m3c2 / v_dod if v_dod > 0 else np.inf

        print(f"  V_DoD = {v_dod:.1f} m³  |  ratio = {ratio:.1f}×\n")

        results.append({
            "event": i + 1,
            "start_date": d1_str, "end_date": d2_str,
            "V_M3C2": v_m3c2, "V_DoD": v_dod,
            "ratio": ratio, "status": "ok",
            # stash for figure
            "_dod": dod, "_dem1": dem1, "_dem2": dem2,
            "_x_edges": x_edges, "_y_edges": y_edges,
            "_along_start": row["alongshore_start_m"],
            "_along_end": row["alongshore_end_m"],
            "_elev": row["elevation"],
            "_height": row["height"],
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure(results_df):
    """Three-panel comparison figure."""

    os.makedirs(FIG_DIR, exist_ok=True)

    ok = results_df[results_df["status"] == "ok"].copy()
    if ok.empty:
        print("No valid events to plot.")
        return

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.2, 0.8],
                           wspace=0.35)

    # ── Panel A: bar chart ─────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0])
    x_pos = np.arange(len(ok))
    w = 0.35
    ax_bar.bar(x_pos - w/2, ok["V_M3C2"], w, color="#2166ac", label="M3C2")
    ax_bar.bar(x_pos + w/2, ok["V_DoD"],  w, color="#b2182b", label="DoD")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([f"E{int(e)}" for e in ok["event"]], fontsize=9)
    ax_bar.set_ylabel("Volume (m³)")
    ax_bar.set_xlabel("Event")
    ax_bar.legend(frameon=False)
    ax_bar.set_title("(a) Volume comparison", fontweight="bold", loc="left")

    # Annotate ratios
    for j, (_, r) in enumerate(ok.iterrows()):
        ratio_str = f"{r['ratio']:.0f}×" if np.isfinite(r["ratio"]) else "∞"
        y_top = max(r["V_M3C2"], r["V_DoD"])
        ax_bar.text(j, y_top * 1.05, ratio_str, ha="center", va="bottom",
                    fontsize=8, color="#333")

    ax_bar.set_ylim(0, ok["V_M3C2"].max() * 1.25)

    # ── Panel B: spatial map of largest event ──────────────────────────────
    ax_map = fig.add_subplot(gs[1])
    biggest = ok.iloc[0]

    if "_dod" in biggest and biggest["_dod"] is not None:
        dod = biggest["_dod"]
        x_edges = biggest["_x_edges"]
        y_edges = biggest["_y_edges"]

        # Plot DoD
        vmax = max(abs(np.nanmin(dod)), abs(np.nanmax(dod)), 0.5)
        im = ax_map.pcolormesh(x_edges, y_edges, dod,
                               cmap="RdBu", vmin=-vmax, vmax=vmax,
                               shading="flat")
        cb = plt.colorbar(im, ax=ax_map, shrink=0.8, pad=0.02)
        cb.set_label("DoD elevation change (m)", fontsize=9)

        ax_map.set_xlabel("Easting (m)")
        ax_map.set_ylabel("Northing (m)")
        ax_map.set_title(
            f"(b) DoD — Event E{int(biggest['event'])}:  "
            f"{biggest['start_date']} → {biggest['end_date']}",
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

    # Draw cliff profile (before)
    cliff_x = [0, 0, 0.5, 0.5, 0, 0, 8, 8]
    cliff_z = [0, 4, 4, 9, 9, 12, 12, 0]
    ax_xs.fill(cliff_x, cliff_z, color="#d9c6a5", edgecolor="#333",
               linewidth=1.5, label="Cliff (before)")

    # Draw eroded notch
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
    ax_xs.text(4, 11.5, "Cliff top — no change\nvisible from above",
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
        description="Compare M3C2 vs DEM-of-Difference volumes.")
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
    print("RESULTS")
    print(f"{'='*70}")
    cols = ["event", "start_date", "end_date", "V_M3C2", "V_DoD", "ratio"]
    print(results[cols].to_string(index=False, float_format="%.1f"))
    print()

    if not args.no_figure:
        make_figure(results)


if __name__ == "__main__":
    main()
