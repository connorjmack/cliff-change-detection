#!/usr/bin/env python3
"""
dem_comparison_combined.py

Publication figure combining:
  - Left:  scatter plot of M3C2 vs DoD erosion volume (one point per date
           pair, limited to n_scatter unique pairs).
  - Right: 2x2 panels showing M3C2 cliff-facing grid (top) and DoD
           (bottom) for two example ranked events.

Usage:
    python3 dem_comparison_combined.py
    python3 dem_comparison_combined.py --n_top 200 --min_volume 10
    python3 dem_comparison_combined.py --event_ranks 1 15
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import binary_fill_holes, label
from scipy.interpolate import griddata as scipy_griddata

# Allow import from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dem_comparison_ratio import (
    run_comparison, REPO_ROOT, LOCATION, FIG_DIR,
    MIN_VOLUME, DEM_RES, LOD_THRESHOLD,
)


def make_combined_figure(results_df, n_scatter=50, event_ranks=(1, 15)):
    """Create publication figure: scatter + two event comparison panels.

    Parameters
    ----------
    results_df : DataFrame from run_comparison (includes _dod_* columns).
    n_scatter  : Max unique date pairs to show in the scatter plot.
    event_ranks: Tuple of two event ranks to display in the right panels.
    """
    dem_dir = os.path.join(FIG_DIR, "dems_ratio")
    os.makedirs(dem_dir, exist_ok=True)

    ok = results_df[results_df["status"] == "ok"].copy()
    if ok.empty:
        print("No valid events for combined figure.")
        return

    # ---- Load NPZ cube for M3C2 panels ----
    npz_path = os.path.join(REPO_ROOT, "results", "data_cubes",
                            f"{LOCATION}_cube.npz")
    if not os.path.exists(npz_path):
        print(f"NPZ cube not found: {npz_path}")
        return

    cube = np.load(npz_path, allow_pickle=True)
    along_m = cube['alongshore_m']
    elev_m = cube['elevation_m']
    ero_3d = cube.get('erosion')
    dep_3d = cube.get('deposition')
    dstrings = [str(s) for s in cube['date_strings']]

    asort = np.argsort(along_m)
    along_s = along_m[asort]

    # ---- Prepare scatter data ----
    ok["pair_key"] = ok["start_date"] + "_" + ok["end_date"]
    deduped = (ok.sort_values("V_M3C2", ascending=False)
               .drop_duplicates(subset="pair_key", keep="first")
               .copy())
    deduped = deduped[deduped["V_DoD"].notna() & (deduped["V_DoD"] > 0)]
    deduped = deduped.head(n_scatter)

    print(f"\nScatter: {len(deduped)} unique date pairs "
          f"(from {len(ok)} events)")

    v_m3c2 = deduped["V_M3C2"].values
    v_dod = deduped["V_DoD"].values
    elev = deduped["elevation"].values

    # Circle sizes scaled by M3C2 volume
    size_min, size_max = 30, 400
    vol_min, vol_max = v_m3c2.min(), v_m3c2.max()
    if vol_max > vol_min:
        sizes = (size_min + (v_m3c2 - vol_min)
                 / (vol_max - vol_min) * (size_max - size_min))
    else:
        sizes = np.full_like(v_m3c2, (size_min + size_max) / 2)

    # ---- Figure layout ----
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.3, 1, 1],
                           wspace=0.2, hspace=0.25)

    # ==================================================================
    #  LEFT: Scatter plot
    # ==================================================================
    ax_sc = fig.add_subplot(gs[:, 0])

    # 1:1 reference line
    axis_max = max(v_m3c2.max(), v_dod.max()) * 1.15
    ax_sc.plot([0, axis_max], [0, axis_max], '--', color='#888',
               linewidth=1.2, zorder=1, label='1:1 line')

    # Scatter
    sc = ax_sc.scatter(v_m3c2, v_dod, c=elev, s=sizes,
                       cmap='plasma', edgecolors='#333', linewidths=0.4,
                       alpha=0.85, zorder=3)

    # Highlight the two event ranks shown in the right panels
    highlight_keys = set()
    for rank in event_ranks:
        ev_rows = ok[ok["rank"] == rank]
        if not ev_rows.empty:
            r = ev_rows.iloc[0]
            highlight_keys.add(r["start_date"] + "_" + r["end_date"])
    hl_mask = deduped["pair_key"].isin(highlight_keys).values
    if hl_mask.any():
        ax_sc.scatter(v_m3c2[hl_mask], v_dod[hl_mask],
                      c=elev[hl_mask], s=sizes[hl_mask],
                      cmap='plasma', vmin=elev.min(), vmax=elev.max(),
                      edgecolors='black', linewidths=2.0,
                      alpha=1.0, zorder=4)

    # Colorbar for elevation
    cb_sc = plt.colorbar(sc, ax=ax_sc, shrink=0.75, pad=0.02)
    cb_sc.set_label("Event Elevation Centroid (m)", fontsize=11)

    # Size legend (representative volumes)
    vol_ticks = np.array([50, 200, 500])
    vol_ticks = vol_ticks[(vol_ticks >= vol_min) & (vol_ticks <= vol_max * 1.1)]
    if len(vol_ticks) == 0:
        vol_ticks = np.array([int(vol_min), int(vol_max)])
    for vt in vol_ticks:
        if vol_max > vol_min:
            sz = (size_min + (vt - vol_min)
                  / (vol_max - vol_min) * (size_max - size_min))
        else:
            sz = (size_min + size_max) / 2
        ax_sc.scatter([], [], s=sz, c='#ccc', edgecolors='#333',
                      linewidths=0.4, label=f'{int(vt)} m\u00b3')

    # Region labels near 1:1 line
    ax_sc.text(axis_max * 0.75, axis_max * 0.55,
               "2.75D Grid > DoD",
               fontsize=10, color='#555', ha='center', style='italic',
               rotation=45, rotation_mode='anchor')
    ax_sc.text(axis_max * 0.55, axis_max * 0.75,
               "DoD > 2.75D Grid",
               fontsize=10, color='#555', ha='center', style='italic',
               rotation=45, rotation_mode='anchor')

    ax_sc.set_xlabel("2.75D Grid Volume, M3C2 (m\u00b3)", fontsize=12)
    ax_sc.set_ylabel("DoD Volume (m\u00b3)", fontsize=12)
    ax_sc.set_xlim(0, axis_max)
    ax_sc.set_ylim(0, axis_max)
    ax_sc.set_aspect('equal')
    ax_sc.legend(loc='upper left', frameon=True, framealpha=0.9,
                 fontsize=9)

    # ==================================================================
    #  RIGHT: Event comparison panels (2 rows x 2 cols)
    # ==================================================================
    # Track axes for row labels
    ax_m3c2_left = None
    ax_dod_left = None

    for col_idx, rank in enumerate(event_ranks):
        ev_rows = ok[ok["rank"] == rank]
        if ev_rows.empty:
            print(f"  Rank #{rank} not found in results, skipping.")
            continue
        row = ev_rows.iloc[0]

        d1 = row["start_date"]
        d2 = row["end_date"]
        dfolder = f"{d1}_to_{d2}"

        ax_top = fig.add_subplot(gs[0, col_idx + 1])
        ax_bot = fig.add_subplot(gs[1, col_idx + 1])

        if col_idx == 0:
            ax_m3c2_left = ax_top
            ax_dod_left = ax_bot

        # -- Top: M3C2 cliff-facing grid --
        tidx = next((i for i, ds in enumerate(dstrings)
                     if ds == dfolder), None)

        if tidx is not None and ero_3d is not None:
            ero2d = ero_3d[:, :, tidx][asort, :]
            combined = np.nan_to_num(ero2d, nan=0.0)
            if dep_3d is not None:
                dep2d = dep_3d[:, :, tidx][asort, :]
                # Fill internal NaN holes in the deposition footprint
                dep_valid = ~np.isnan(dep2d) & (dep2d > 0)
                dep_footprint = binary_fill_holes(dep_valid)
                holes = dep_footprint & ~dep_valid
                if holes.any() and dep_valid.any():
                    yx_valid = np.argwhere(dep_valid)
                    yx_holes = np.argwhere(holes)
                    filled_vals = scipy_griddata(
                        yx_valid, dep2d[dep_valid],
                        yx_holes, method='nearest')
                    dep2d_filled = dep2d.copy()
                    dep2d_filled[tuple(yx_holes.T)] = filled_vals
                    dc = np.nan_to_num(dep2d_filled, nan=0.0)
                else:
                    dc = np.nan_to_num(dep2d, nan=0.0)
                # Remove small deposition clusters (< 25 connected cells)
                dep_mask = (dc > 0) & (combined < 0.01)
                labeled, n_clusters = label(dep_mask)
                for cid in range(1, n_clusters + 1):
                    if np.sum(labeled == cid) < 25:
                        dep_mask[labeled == cid] = False
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
                        [f'{az[j]:.0f}' for j in ti], fontsize=9)
                nt = min(4, len(ez))
                if len(ez) > 1:
                    ti = np.linspace(0, len(ez)-1, nt, dtype=int)
                    ax_top.set_yticks(ti)
                    ax_top.set_yticklabels(
                        [f'{ez[j]:.1f}' for j in ti], fontsize=9)

                ax_top.invert_xaxis()
                ax_top.set_xlabel("Alongshore Location (m)", fontsize=10)
                ax_top.set_ylabel("Elevation (m)", fontsize=10)

                cb1 = plt.colorbar(im1, ax=ax_top, shrink=0.7,
                                   pad=0.02, aspect=15)
                cb1.ax.tick_params(labelsize=8)
                cb1.set_label("M3C2 Distance (m)", fontsize=10)
            else:
                ax_top.text(0.5, 0.5, "Outside grid",
                            transform=ax_top.transAxes,
                            ha='center', va='center', fontsize=9)
        else:
            ax_top.text(0.5, 0.5, "No cube data",
                        transform=ax_top.transAxes,
                        ha='center', va='center', fontsize=9)

        ratio_str = (f"{row['ratio']:.1f}"
                     if np.isfinite(row["ratio"]) else "inf")
        ax_top.set_title(
            f"Volume: {row['V_M3C2']:.1f} m\u00b3  |  "
            f"Ratio: {ratio_str}\u00d7",
            fontsize=11, fontweight='bold')

        # -- Bottom: DoD zoomed to largest cluster --
        # Use the wider crop for the first event rank to avoid the
        # diagonal rotation-edge artifact; keep the tight crop for others.
        if rank == event_ranks[0] and row.get("_dod_zoom_wide") is not None:
            dod_zoom = row["_dod_zoom_wide"]
            xedges = row["_x_edges_wide"]
            yedges = row["_y_edges_wide"]
        else:
            dod_zoom = row["_dod_zoom"]
            xedges = row["_x_edges"]
            yedges = row["_y_edges"]

        if dod_zoom is not None:
            vabs = max(abs(np.nanmin(dod_zoom)),
                       abs(np.nanmax(dod_zoom)), 0.5)
            vabs = min(vabs, 4.0)

            # Negate so erosion is positive (red), matching M3C2 row
            dz_masked = np.ma.masked_invalid(-dod_zoom)
            im2 = ax_bot.pcolormesh(xedges, yedges, dz_masked,
                                    cmap="RdBu_r",
                                    vmin=-vabs, vmax=vabs,
                                    shading="flat")
            ax_bot.set_facecolor("white")
            ax_bot.tick_params(labelsize=8)
            ax_bot.set_xlabel("Alongshore (m)", fontsize=10)
            ax_bot.set_ylabel("Cross-shore (m)", fontsize=10)

            # Per-panel display adjustments (crop/shift for visual match
            # with the M3C2 panel above; volumes are unaffected)
            if rank == event_ranks[0]:
                ax_bot.set_xlim(20, 60)
            if rank == event_ranks[1]:
                ax_bot.set_xlim(left=10)
                yl = ax_bot.get_ylim()
                ax_bot.set_ylim(yl[0] - 3, yl[1])
            cb2 = plt.colorbar(im2, ax=ax_bot, shrink=0.7,
                               pad=0.02, aspect=15)
            cb2.ax.tick_params(labelsize=8)
            cb2.set_label("Elev. change (m)", fontsize=10)
        else:
            ax_bot.text(0.5, 0.5, "No erosion\nin DoD", ha='center',
                        va='center', transform=ax_bot.transAxes,
                        fontsize=9)

        ax_bot.set_title(
            f"Volume: {row['V_DoD']:.1f} m\u00b3",
            fontsize=11, fontweight='bold')

    # Row labels on the leftmost event panels
    if ax_m3c2_left is not None:
        ax_m3c2_left.set_ylabel("2.75D Grid\n(cliff-facing)",
                                fontsize=12, fontweight='bold')
    if ax_dod_left is not None:
        ax_dod_left.set_ylabel("DoD\n(top-down)",
                               fontsize=12, fontweight='bold')

    plt.tight_layout()

    out_png = os.path.join(dem_dir, "combined_scatter_events.png")
    out_pdf = os.path.join(dem_dir, "combined_scatter_events.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.savefig(out_pdf, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description="Combined publication figure: scatter + event panels.")
    parser.add_argument("--n_top", type=int, default=200,
                        help="Number of top events to process (default: 200)")
    parser.add_argument("--min_volume", type=float, default=10.0,
                        help="Min M3C2 event volume in m3 (default: 10)")
    parser.add_argument("--n_scatter", type=int, default=50,
                        help="Max unique date pairs in scatter (default: 50)")
    parser.add_argument("--event_ranks", type=int, nargs=2, default=[1, 15],
                        help="Two event ranks to show as panels "
                             "(default: 1 15)")
    parser.add_argument("--dem_res", type=float, default=DEM_RES,
                        help=f"DEM cell size in metres (default: {DEM_RES})")
    parser.add_argument("--lod", type=float, default=LOD_THRESHOLD,
                        help=f"LoD threshold in metres "
                             f"(default: {LOD_THRESHOLD})")
    args = parser.parse_args()

    results = run_comparison(
        min_volume=args.min_volume,
        n_top=args.n_top,
        dem_res=args.dem_res,
        lod=args.lod,
    )

    make_combined_figure(results,
                         n_scatter=args.n_scatter,
                         event_ranks=tuple(args.event_ranks))


if __name__ == "__main__":
    main()
