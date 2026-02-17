#!/usr/bin/env python3
"""
dem_comparison.py

Compares M3C2-based volumes (this study) against traditional DEM-of-Difference
(DoD) volumes for the top-N largest QC'd erosion events at Del Mar.

The top-down DEM (max Z per XY cell) is intentionally a naive representation
of the cliff — it captures the cliff top and beach surfaces but compresses the
near-vertical face into a 1-3 cell wide strip.  This is exactly the limitation
of classical DEM differencing for vertical cliffs, and the comparison quantifies
how much volume is missed.

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

# QC event list (repo-relative, auto-detect most recent)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
QC_DIR = os.path.join(REPO_ROOT, "results", "event_lists_qc", "erosion")

# Output
FIG_DIR = os.path.join(REPO_ROOT, "figures", "appendix")

# ── constants ──────────────────────────────────────────────────────────────────
DEM_RES = 0.25          # m – top-down DEM cell size (default)
LOD_THRESHOLD = 0.25    # m – Level of Detection for DoD
ALONG_BUFFER = 20.0     # m – buffer around event footprint (UTM Y)


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
    """Find the noveg LAS file whose name starts with *date_str* (YYYYMMDD)."""
    pattern = os.path.join(NOVEG_DIR, f"{date_str}*.las")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[0]


def load_and_clip(las_path, y_min, y_max):
    """Load a LAS file and return (x, y, z) clipped to y_min <= y <= y_max."""
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    mask = (y >= y_min) & (y <= y_max)
    return x[mask], y[mask], z[mask]


def rasterise_to_common_grid(x, y, z, x_min, y_min, dem_res, nx, ny):
    """Rasterise points to a top-down DSM (max Z per cell) on a pre-defined grid.

    Parameters
    ----------
    x, y, z : 1-D arrays of point coordinates
    x_min, y_min : float – origin of the grid
    dem_res : float – cell size (m)
    nx, ny : int – number of cells in X and Y

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
          NaN cells are preserved (np.abs(NaN) < lod is False, so they stay NaN).
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


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def run_comparison(n_events=5, dem_res=DEM_RES, lod=LOD_THRESHOLD):
    """Run the DEM-vs-M3C2 comparison for the top-N events.

    Returns a DataFrame with one row per event.
    """
    qc_csv = find_qc_csv()
    print(f"Using QC file: {os.path.basename(qc_csv)}")

    df = pd.read_csv(qc_csv,
                     parse_dates=["mid_date", "start_date", "end_date"])
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

        # --- common-extent grid for both DEMs ---
        all_x = np.concatenate([x1, x2])
        all_y = np.concatenate([y1, y2])
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        nx = int(np.ceil((x_max - x_min) / dem_res))
        ny = int(np.ceil((y_max - y_min) / dem_res))
        x_edges = np.linspace(x_min, x_min + nx * dem_res, nx + 1)
        y_edges = np.linspace(y_min, y_min + ny * dem_res, ny + 1)

        print(f"  Rasterising DEMs ({nx}×{ny} cells) ...")
        dem1 = rasterise_to_common_grid(x1, y1, z1,
                                        x_min, y_min, dem_res, nx, ny)
        dem2 = rasterise_to_common_grid(x2, y2, z2,
                                        x_min, y_min, dem_res, nx, ny)

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
            # stash arrays for figures
            "_dem1": dem1, "_dem2": dem2, "_dod": dod,
            "_x_edges": x_edges, "_y_edges": y_edges,
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

    y_ceil = max(ok["V_M3C2"].max(), ok["V_DoD"].max()) * 1.25
    ax_bar.set_ylim(0, y_ceil)

    # ── Panel B: spatial map of largest event ──────────────────────────────
    ax_map = fig.add_subplot(gs[1])
    biggest = ok.iloc[0]

    dod = biggest["_dod"]
    if dod is not None and not np.all(np.isnan(dod)):
        x_edges = biggest["_x_edges"]
        y_edges = biggest["_y_edges"]

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
#  PER-EVENT DIAGNOSTIC FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def make_event_figures(results_df):
    """Save a 3-panel diagnostic figure for each event: DEM1, DEM2, DoD."""

    dem_dir = os.path.join(FIG_DIR, "dems")
    os.makedirs(dem_dir, exist_ok=True)

    ok = results_df[results_df["status"] == "ok"]
    if ok.empty:
        print("No valid events to plot.")
        return

    for _, row in ok.iterrows():
        ev = int(row["event"])
        dem1 = row["_dem1"]
        dem2 = row["_dem2"]
        dod = row["_dod"]
        x_edges = row["_x_edges"]
        y_edges = row["_y_edges"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # -- shared elevation colour range for DEM panels --
        z_lo = np.nanmin([np.nanmin(dem1), np.nanmin(dem2)])
        z_hi = np.nanmax([np.nanmax(dem1), np.nanmax(dem2)])

        # Panel 1: DEM before
        ax = axes[0]
        im1 = ax.pcolormesh(x_edges, y_edges, dem1,
                            cmap="terrain", vmin=z_lo, vmax=z_hi,
                            shading="flat")
        plt.colorbar(im1, ax=ax, shrink=0.7, label="Elevation (m)")
        ax.set_title(f"DEM before  ({row['start_date']})",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=7, labelrotation=30)

        # Panel 2: DEM after
        ax = axes[1]
        im2 = ax.pcolormesh(x_edges, y_edges, dem2,
                            cmap="terrain", vmin=z_lo, vmax=z_hi,
                            shading="flat")
        plt.colorbar(im2, ax=ax, shrink=0.7, label="Elevation (m)")
        ax.set_title(f"DEM after  ({row['end_date']})",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=7, labelrotation=30)

        # Panel 3: DoD (negative=red=erosion, positive=blue=deposition)
        ax = axes[2]
        vmax_dod = max(abs(np.nanmin(dod)), abs(np.nanmax(dod)), 0.5)
        # RdBu_r: red for negative, blue for positive
        im3 = ax.pcolormesh(x_edges, y_edges, dod,
                            cmap="RdBu", vmin=-vmax_dod, vmax=vmax_dod,
                            shading="flat")
        cb = plt.colorbar(im3, ax=ax, shrink=0.7, label="DoD (m)")
        ax.set_title(
            f"DoD  (V_DoD={row['V_DoD']:.1f} m³,  "
            f"V_M3C2={row['V_M3C2']:.1f} m³,  "
            f"ratio={row['ratio']:.1f}×)",
            fontweight="bold", fontsize=9)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=7, labelrotation=30)

        fig.suptitle(
            f"Event E{ev}:  {row['start_date']} → {row['end_date']}",
            fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        out = os.path.join(dem_dir, f"event_{ev}_{row['start_date']}_to_{row['end_date']}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close()
        print(f"  Saved: {out}")

    print(f"\nAll event DEMs saved to: {dem_dir}")


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
        make_event_figures(results)
        make_figure(results)


if __name__ == "__main__":
    main()
