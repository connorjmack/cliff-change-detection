#!/usr/bin/env python3
"""
geo_stats.py

Generates a 2x2 Geomorphology Statistics Dashboard.

Panel A now shows a spatiotemporally normalized cumulative magnitude-frequency
curve following Janeras et al. (2023).  Raw cumulative exceedance counts N(>=V)
are divided by the monitored cliff face area A_st (hm^2) and the observation
period T (yr) to yield F_st (hm^-2 yr^-1), enabling direct comparison across
sites and studies.  Per-location cliff face areas are pre-computed from the NPZ
data cubes (0.25 m polygon width, median cliff base subtracted per polygon).

Usage:
    python3 geo_stats.py --location DelMar
    python3 geo_stats.py --location all
"""

import os
import re
import platform
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

# --- CONFIGURATION ---
RESOLUTION = '25cm'
RES_VAL = 0.25
CELL_AREA = RES_VAL * RES_VAL
FILE_TAG = '25cm'
LOCATIONS_ALL = ['DelMar', 'Torrey', 'Solana', 'Encinitas', 'SanElijo']

# Per-location cliff face areas (hm²) derived from NPZ data cubes.
# Method: per-polygon cliff height = highest occupied 0.25-m bin minus median
# lowest-data-bin (z0).  Area = sum(heights) * 0.25 m polygon width / 10,000.
AREA_HM2 = {
    'DelMar':    2.6636,  # z0=2.50 m, 9,124 polygons
    'Solana':    3.3266,  # z0=3.25 m, 10,787 polygons
    'Encinitas': 7.3155,  # z0=2.75 m, 22,225 polygons
    'SanElijo':  2.3109,  # z0=2.25 m, 9,852 polygons
    'Torrey':    3.4274,  # z0=2.50 m, 5,328 polygons
    'Blacks':    7.4069,  # z0=3.25 m, 12,982 polygons
}
AREA_HM2['Regional'] = sum(AREA_HM2.values())  # combined regional area

# --- VISUAL SETTINGS ---
COLOR_MAIN   = '#08519c'    # Strong Blue (Winter)
COLOR_ACCENT = '#a50f15'    # Dark Red
COLOR_ROSE   = '#4292c6'    # Lighter Blue for Rose
COLOR_VIOLIN = '#6baed6'    # Violin body color

# --- GLOBAL FONT SIZES (INCREASED) ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 18          # Was 14
plt.rcParams['axes.titlesize'] = 24     # Was 20
plt.rcParams['axes.labelsize'] = 20     # Was 16
plt.rcParams['xtick.labelsize'] = 16    # Was 14
plt.rcParams['ytick.labelsize'] = 16    # Was 14
plt.rcParams['legend.fontsize'] = 16    # Was 14

# ==============================================================================
# 1. DATA PROCESSING
# ==============================================================================

def get_base_dir():
    if platform.system() == 'Darwin':
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

def parse_dates(folder_name):
    match = re.search(r'(\d{8})_to_(\d{8})', folder_name)
    if match:
        return datetime.strptime(match.group(1), '%Y%m%d'), datetime.strptime(match.group(2), '%Y%m%d')
    return None, None

def extract_detailed_events(cluster_path, grid_path, unc_path, res_val, date_mid, date_start, date_end):
    try:
        df_c = pd.read_csv(cluster_path, index_col=0).fillna(0)
        df_g = pd.read_csv(grid_path, index_col=0).fillna(0)
        
        if unc_path and os.path.exists(unc_path):
            df_u = pd.read_csv(unc_path, index_col=0).fillna(0)
        else:
            df_u = None

        df_c.columns = [c.split('_')[-1] for c in df_c.columns]
        df_g.columns = [c.split('_')[-1] for c in df_g.columns]
        if df_u is not None:
             df_u.columns = [c.split('_')[-1] for c in df_u.columns]
        
        common_index = df_c.index.intersection(df_g.index)
        common_cols = df_c.columns.intersection(df_g.columns)
        
        if len(common_cols) == 0: return []
            
        df_c = df_c.loc[common_index, common_cols]
        df_g = df_g.loc[common_index, common_cols]
        
        c_vals = df_c.values
        g_vals = df_g.values
        u_vals = df_u.loc[common_index, common_cols].values if df_u is not None else None
        
        try:
            z_map = [float(re.findall(r"[-+]?\d*\.\d+|\d+", c)[0]) for c in df_c.columns]
            z_map = np.array(z_map)
        except:
            z_map = np.arange(len(df_c.columns)) * res_val
            
        unique_ids = np.unique(c_vals)
        unique_ids = unique_ids[unique_ids != 0] 
        
        events = []
        for uid in unique_ids:
            mask = (c_vals == uid)
            rows, cols = np.where(mask)
            dists = g_vals[mask]
            
            vol = np.sum(np.abs(dists)) * CELL_AREA
            width = (np.max(rows) - np.min(rows) + 1) * res_val
            height = (np.max(cols) - np.min(cols) + 1) * res_val
            z_centroid = np.average(z_map[cols], weights=np.abs(dists))
            
            if u_vals is not None:
                u_subset = u_vals[mask]
                u_valid = u_subset[u_subset > 0]
                if u_valid.size > 0:
                    vol_unc = np.sum(u_valid) * CELL_AREA
                else:
                    vol_unc = 0.0
            else:
                vol_unc = 0.0

            events.append({
                'date':       date_mid,
                'date_start': date_start,
                'date_end':   date_end,
                'volume':     vol,
                'vol_unc':    vol_unc,
                'elevation':  z_centroid,
                'width':      width,
                'height':     height,
                'month':      date_mid.month,
            })
        return events
    except Exception as e:
        return []

def resolve_interval_paths(interval_dir, interval):
    """Return erosion grid/cluster/uncertainty paths, preferring resolution subfolders."""
    res_dir = os.path.join(interval_dir, RESOLUTION)
    candidates = [
        (
            os.path.join(res_dir, f"{interval}_ero_grid_{FILE_TAG}_filled.csv"),
            os.path.join(res_dir, f"{interval}_ero_clusters_{FILE_TAG}_filled.csv"),
            os.path.join(res_dir, f"{interval}_ero_uncertainty_{FILE_TAG}.csv"),
        ),
        (
            os.path.join(interval_dir, f"{interval}_ero_grid_{FILE_TAG}_filled.csv"),
            os.path.join(interval_dir, f"{interval}_ero_clusters_{FILE_TAG}_filled.csv"),
            os.path.join(interval_dir, f"{interval}_ero_uncertainty_{FILE_TAG}.csv"),
        ),
    ]
    for grid_path, clus_path, unc_path in candidates:
        if os.path.exists(grid_path) and os.path.exists(clus_path):
            return grid_path, clus_path, unc_path
    return None, None, None

def collect_all_data(base_dir, locations):
    all_events = []
    for loc in locations:
        print(f"Scanning {loc}...")
        erosion_dir = os.path.join(base_dir, 'results', loc, 'erosion')
        if not os.path.isdir(erosion_dir): continue
        
        intervals = sorted([d for d in os.listdir(erosion_dir) if os.path.isdir(os.path.join(erosion_dir, d))])
        
        for interval in intervals:
            d1, d2 = parse_dates(interval)
            if not d1: continue
            
            folder = os.path.join(erosion_dir, interval)
            grid_file, clus_file, unc_file = resolve_interval_paths(folder, interval)
            
            if grid_file and clus_file:
                events = extract_detailed_events(clus_file, grid_file, unc_file, RES_VAL,
                                                 d1 + (d2 - d1) / 2, d1, d2)
                all_events.extend(events)
                
    return pd.DataFrame(all_events)

# ==============================================================================
# 2. STATISTICAL HELPERS
# ==============================================================================

def calculate_gini(array):
    """Calculates the Gini Coefficient (0 to 1)."""
    if len(array) == 0: return 0.0
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((2 * index - n - 1) * array).sum() / (n * array.sum())

def fit_power_law_fixed_cutoff(volumes, cutoff=0.25, n_bins=30):
    """Histogram-based power law fit (original method)."""
    if len(volumes) == 0: return None, None, None, None, 0
    min_vol, max_vol = volumes.min(), volumes.max()
    bins = np.logspace(np.log10(min_vol), np.log10(max_vol), n_bins)
    hist, edges = np.histogram(volumes, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    valid_mask = (centers >= cutoff) & (hist > 0)
    if np.sum(valid_mask) < 3:
        return centers, hist, None, None, 0.0
    x_fit = centers[valid_mask]
    y_fit = hist[valid_mask]
    slope, intercept = np.polyfit(np.log10(x_fit), np.log10(y_fit), 1)
    y_fitted = 10**intercept * x_fit**slope
    return centers, hist, x_fit, y_fitted, -slope


def fit_power_law_cumulative(volumes, area_hm2, t_years, cutoff=1.0):
    """
    Computes cumulative exceedance frequency N(>=V), normalizes by A_st * T
    to yield F_st (hm^-2 yr^-1), then fits a power law above cutoff.

    Following Janeras et al. (2023):  F_st = alpha * V^(-B)

    Parameters
    ----------
    volumes  : pd.Series or np.ndarray of event volumes (m^3)
    area_hm2 : monitored cliff face area (hm^2)
    t_years  : observation period (yr)
    cutoff   : minimum volume for power law fit (m^3)

    Returns
    -------
    vols_sorted : all volumes sorted descending
    raw_n       : raw cumulative counts N(>=V)
    fst         : normalized F_st = N(>=V) / (area_hm2 * t_years)
    fit_v       : volumes used for fit (>= cutoff)
    fit_fst     : fitted F_st values on fit_v
    B           : scaling exponent (positive; slope = -B)
    alpha       : activity rate (events hm^-2 yr^-1 with V >= 1 m^3)
    """
    if len(volumes) == 0:
        return None, None, None, None, None, 0.0, 0.0

    vols_sorted = np.sort(np.asarray(volumes))[::-1]
    raw_n = np.arange(1, len(vols_sorted) + 1, dtype=float)
    norm = area_hm2 * t_years
    fst = raw_n / norm

    mask = vols_sorted >= cutoff
    if mask.sum() < 3:
        return vols_sorted, raw_n, fst, None, None, 0.0, 0.0

    fit_v   = vols_sorted[mask]
    fit_fst = fst[mask]

    slope, intercept = np.polyfit(np.log10(fit_v), np.log10(fit_fst), 1)
    fitted_fst = 10**intercept * fit_v**slope
    B     = -slope
    alpha = 10**intercept

    return vols_sorted, raw_n, fst, fit_v, fitted_fst, B, alpha


def _fit_log_range(vols_sorted, fst, v_lower, v_upper):
    """Fit power law to F_st vs V in [v_lower, v_upper]. Returns B, alpha, R², n."""
    mask = (vols_sorted >= v_lower) & (vols_sorted <= v_upper)
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, np.nan, n
    x = np.log10(vols_sorted[mask])
    y = np.log10(fst[mask])
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return -slope, 10**intercept, r2, n


def dual_regime_power_law_fit(volumes, area_hm2, t_years,
                               lower_bound=0.25,
                               bp_range=(1.5, 8.0), n_bp=30,
                               upper_range=(50.0, None), n_upper=20):
    """
    Fit two separate power law regimes on the normalized CCDF.

    Regime 1: [lower_bound, breakpoint]  — rollover / detection-limited regime
    Regime 2: [breakpoint, upper_cutoff] — true power law tail

    The breakpoint and upper cutoff are jointly optimized to maximize
    a point-count-weighted average R² across both regimes.

    Parameters
    ----------
    volumes     : event volumes (m³)
    area_hm2    : monitored cliff face area (hm²)
    t_years     : observation period (yr)
    lower_bound : fixed lower bound for Regime 1 (m³)
    bp_range    : (min, max) search range for breakpoint (m³)
    n_bp        : number of breakpoint candidates
    upper_range : (min, max) search range for upper cutoff; None = v_max
    n_upper     : number of upper cutoff candidates

    Returns
    -------
    dict with both regime fits, or None on failure.
    """
    if len(volumes) == 0:
        return None

    vols_sorted = np.sort(np.asarray(volumes))[::-1]
    raw_n = np.arange(1, len(vols_sorted) + 1, dtype=float)
    fst = raw_n / (area_hm2 * t_years)

    v_max = vols_sorted[0]

    # Candidate breakpoints (log-spaced)
    bp_lo = max(bp_range[0], lower_bound * 2)  # breakpoint must be > lower_bound
    bp_hi = min(bp_range[1], v_max * 0.5)
    bp_candidates = np.logspace(np.log10(bp_lo), np.log10(bp_hi), n_bp)

    # Candidate upper cutoffs (log-spaced)
    uc_lo = upper_range[0]
    uc_hi = v_max if upper_range[1] is None else min(upper_range[1], v_max)
    if uc_lo >= uc_hi:
        uc_lo = uc_hi * 0.5
    upper_candidates = np.logspace(np.log10(uc_lo), np.log10(uc_hi), n_upper)

    best_score = -np.inf
    best = None

    for bp in bp_candidates:
        for uc in upper_candidates:
            if bp >= uc:
                continue

            B1, a1, r2_1, n1 = _fit_log_range(vols_sorted, fst, lower_bound, bp)
            B2, a2, r2_2, n2 = _fit_log_range(vols_sorted, fst, bp, uc)

            if np.isnan(r2_1) or np.isnan(r2_2):
                continue

            score = (n1 * r2_1 + n2 * r2_2) / (n1 + n2)

            if score > best_score:
                best_score = score
                best = {
                    'breakpoint': bp, 'upper_cutoff': uc,
                    'B1': B1, 'alpha1': a1, 'r2_1': r2_1, 'n1': n1,
                    'B2': B2, 'alpha2': a2, 'r2_2': r2_2, 'n2': n2,
                }

    if best is None:
        return None

    # Generate fit lines for both regimes
    bp = best['breakpoint']
    uc = best['upper_cutoff']

    mask1 = (vols_sorted >= lower_bound) & (vols_sorted <= bp)
    mask2 = (vols_sorted >= bp) & (vols_sorted <= uc)

    fit_v1   = vols_sorted[mask1]
    fit_fst1 = best['alpha1'] * fit_v1 ** (-best['B1'])

    fit_v2   = vols_sorted[mask2]
    fit_fst2 = best['alpha2'] * fit_v2 ** (-best['B2'])

    best.update({
        'vols_sorted': vols_sorted, 'raw_n': raw_n, 'fst': fst,
        'fit_v1': fit_v1, 'fit_fst1': fit_fst1,
        'fit_v2': fit_v2, 'fit_fst2': fit_fst2,
        'lower_bound': lower_bound,
        'combined_r2': best_score,
    })

    return best


# ==============================================================================
# 3. PLOTTING
# ==============================================================================

def plot_geomorph_stats(df, out_dir, title_prefix, area_hm2=None, normalized=True):
    if df.empty: return

    # Filter minimal noise
    df_clean = df[df['volume'] >= 0.005].copy()

    # --- NORMALIZATION PARAMETERS ---
    # Observation period from actual survey dates in the data
    if 'date_start' in df_clean.columns and 'date_end' in df_clean.columns:
        t_start = df_clean['date_start'].min()
        t_end   = df_clean['date_end'].max()
        t_years = (t_end - t_start).days / 365.25
    else:
        t_years = 8.71  # fallback if date columns not present

    # Cliff face area: use passed value, look up by location, or sum all
    if area_hm2 is None:
        area_hm2 = AREA_HM2.get(title_prefix, AREA_HM2['Regional'])
    norm_denom = area_hm2 * t_years
    
    # --- CALCULATE SEASONAL STATISTICS ---
    monthly_vol = df_clean.groupby('month')['volume'].sum()
    full_months = pd.Series(0.0, index=np.arange(1, 13))
    monthly_vol = monthly_vol.combine(full_months, max).sort_index()
    
    total_vol = monthly_vol.sum()
    
    # Winter: Oct(10), Nov(11), Dec(12), Jan(1), Feb(2), Mar(3)
    winter_indices = [1, 2, 3, 10, 11, 12]
    winter_vol = monthly_vol.loc[winter_indices].sum()
    
    winter_pct = (winter_vol / total_vol * 100) if total_vol > 0 else 0
    summer_pct = 100.0 - winter_pct
    
    # Peak Month
    if total_vol > 0:
        max_month_idx = monthly_vol.idxmax()
        max_month_val = monthly_vol.max()
        max_month_pct = (max_month_val / total_vol) * 100
        month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                       7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
        peak_month_str = f"{month_names[max_month_idx]} ({max_month_pct:.1f}%)"
    else:
        peak_month_str = "N/A"

    # --- CALCULATE DESCRIPTIVE STATS (Median & IQR) ---
    vol_med = df_clean['volume'].median()
    vol_q1  = df_clean['volume'].quantile(0.25)
    vol_q3  = df_clean['volume'].quantile(0.75)
    
    elev_med = df_clean['elevation'].median()
    elev_q1  = df_clean['elevation'].quantile(0.25)
    elev_q3  = df_clean['elevation'].quantile(0.75)
    
    span_med = df_clean['width'].median()
    # span_q1  = df_clean['width'].quantile(0.25) # Optional if needed later
    # span_q3  = df_clean['width'].quantile(0.75) # Optional if needed later

    # --- PRINT TO TERMINAL ---
    print(f"\n[{title_prefix.upper()} STATS REPORT]")
    print(f"  Total Erosion Vol: {total_vol:.1f} m^3")
    print(f"  Winter (Oct-Mar):  {winter_pct:.1f}%")
    print(f"  Summer (Apr-Sep):  {summer_pct:.1f}%")
    print(f"  Peak Month:        {peak_month_str}")
    print("\n  Individual events typically had:")
    print(f"    - Volumes of {vol_med:.2f} m^3 (IQR: {vol_q1:.2f}-{vol_q3:.2f})")
    print(f"    - Occurred at elevations of {elev_med:.2f} m NAVD88 (IQR: {elev_q1:.2f}-{elev_q3:.2f})")
    print(f"    - Spanned {span_med:.2f} m alongshore")
    print(f"\n  Normalization (Janeras et al. 2023):")
    print(f"    - Cliff face area A_st: {area_hm2:.4f} hm^2")
    print(f"    - Observation period T: {t_years:.2f} yr")
    print(f"    - Denominator A_st x T: {norm_denom:.2f} hm^2 yr")
    print("-" * 60)
    # ----------------------------------------------------

    # Setup Figure
    fig = plt.figure(figsize=(24, 18)) # Slightly larger for big fonts
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], wspace=0.3, hspace=0.35)
    
    # ==================== PANEL A: MAGNITUDE-FREQUENCY ====================
    ax1 = fig.add_subplot(gs[0, 0])

    if normalized:
        # --- Dual-regime power law fit (Janeras et al. 2023) ---
        LOWER_BOUND = 0.25   # m³ — minimum detection volume
        BREAKPOINT  = 2.0    # m³ — regime boundary

        vols_sorted = np.sort(np.asarray(df_clean['volume']))[::-1]
        raw_n = np.arange(1, len(vols_sorted) + 1, dtype=float)
        fst = raw_n / norm_denom
        v_max = vols_sorted[0]

        B1, a1, r2_1, n1 = _fit_log_range(vols_sorted, fst, LOWER_BOUND, BREAKPOINT)
        B2, a2, r2_2, n2 = _fit_log_range(vols_sorted, fst, BREAKPOINT, v_max)

        # Plot all data as black dots
        ax1.loglog(vols_sorted, fst, 'o', color='black', markersize=7, alpha=0.6)

        # Regime 1 fit line
        mask1 = (vols_sorted >= LOWER_BOUND) & (vols_sorted <= BREAKPOINT)
        if not np.isnan(B1) and mask1.sum() > 0:
            fit_v1 = vols_sorted[mask1]
            ax1.loglog(fit_v1, a1 * fit_v1 ** (-B1), '--',
                       color=COLOR_MAIN, linewidth=3.5,
                       label=r'R1: $\beta_1$' + f' = {B1:.2f}')

        # Regime 2 fit line
        mask2 = vols_sorted >= BREAKPOINT
        if not np.isnan(B2) and mask2.sum() > 0:
            fit_v2 = vols_sorted[mask2]
            ax1.loglog(fit_v2, a2 * fit_v2 ** (-B2), '--',
                       color=COLOR_ACCENT, linewidth=3.5,
                       label=r'R2: $\beta_2$' + f' = {B2:.2f}')

        # Breakpoint line
        ax1.axvline(BREAKPOINT, color='black', linestyle=':', alpha=0.6,
                    linewidth=2.5)

        # Terminal report
        print(f"\n  Dual-Regime Power Law Fit:")
        print(f"    Regime 1 [{LOWER_BOUND} - {BREAKPOINT} m^3]:")
        print(f"      B1 = {B1:.3f}, alpha1 = {a1:.3f}, R^2 = {r2_1:.4f}, N = {n1}")
        print(f"    Regime 2 [{BREAKPOINT} - {v_max:.1f} m^3]:")
        print(f"      B2 = {B2:.3f}, alpha2 = {a2:.3f}, R^2 = {r2_2:.4f}, N = {n2}")

        ax1.text(0.03, 0.03,
                 f'$A_{{st}} \\times T = {norm_denom:.1f}$ hm$^2$ yr',
                 transform=ax1.transAxes, ha='left', va='bottom',
                 fontsize=13, color='#333333',
                 bbox=dict(facecolor='white', edgecolor='#aaaaaa',
                           boxstyle='round,pad=0.4', alpha=0.85))

        ax1.set_xlabel(r'Erosion Volume $V$ (m$^3$)', fontweight='bold')
        ax1.set_ylabel(r'$F_{st}$ (hm$^{-2}$ yr$^{-1}$)', fontweight='bold')
        ax1.set_title('A) Magnitude-Frequency (Normalized)', loc='left', fontweight='bold', pad=15)

    else:
        # --- Original histogram-based power law ---
        CUTOFF_VAL = 0.25  # fixed cutoff for non-normalized mode
        centers, hist, fit_x, fit_y, beta = fit_power_law_fixed_cutoff(
            df_clean['volume'], cutoff=CUTOFF_VAL
        )

        ax1.loglog(centers, hist, 'o', color='gray', alpha=0.4, markersize=10)

        # Uncertainty error bars per bin
        bin_edges = np.logspace(np.log10(df_clean['volume'].min()),
                                np.log10(df_clean['volume'].max()), 30)
        indices = np.digitize(df_clean['volume'], bin_edges)
        bin_uncs = []
        for i in range(1, len(bin_edges)):
            mask = indices == i
            bin_uncs.append(df_clean.loc[mask, 'vol_unc'].mean() if np.any(mask) else 0)

        valid_hist_mask = hist > 0
        if len(bin_uncs) >= len(centers):
            x_err = np.array(bin_uncs)[:len(centers)][valid_hist_mask]
            ax1.errorbar(centers[valid_hist_mask], hist[valid_hist_mask],
                         xerr=x_err, fmt='none', ecolor=COLOR_ACCENT,
                         alpha=0.3, capsize=3)

        mask_tail = centers >= CUTOFF_VAL
        ax1.loglog(centers[mask_tail], hist[mask_tail], 'o', color='black', markersize=10)
        if fit_x is not None:
            ax1.loglog(fit_x, fit_y, '--', color=COLOR_ACCENT, linewidth=4,
                       label=f'$\\beta = {beta:.2f}$')
            ax1.axvline(CUTOFF_VAL, color=COLOR_ACCENT, linestyle=':', alpha=0.6,
                        linewidth=3, label=f'Cutoff: {CUTOFF_VAL} m$^3$')

        ax1.set_xlabel(r'Erosion Object Volume (m$^3$)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('A) Magnitude-Frequency', loc='left', fontweight='bold', pad=15)

    ax1.legend(loc='lower left', fontsize=14)
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    
    # ==================== PANEL B: MORPHOLOGY (Original Style) ====================
    ax2 = fig.add_subplot(gs[0, 1])
    
    data = [df_clean['volume'], df_clean['elevation'], df_clean['width'], df_clean['height']]
    labels = ['Volume\n($m^3$)', 'Centroid Elev.\n(m)', 'Width\n(m)', 'Height\n(m)']
    
    # Violin Plot
    parts = ax2.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    
    for pc in parts['bodies']:
        pc.set_facecolor(COLOR_VIOLIN)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    parts['cmedians'].set_edgecolor(COLOR_ACCENT)
    parts['cmedians'].set_linewidth(3)
    
    # Overlay Boxplot (Thin)
    ax2.boxplot(data, sym='', widths=0.1, 
                boxprops=dict(color='black', linewidth=2), 
                whiskerprops=dict(color='black', linewidth=2),
                capprops=dict(color='black', linewidth=2),
                medianprops=dict(color=COLOR_ACCENT, linewidth=2))

    ax2.set_yscale('log')
    ax2.set_xticks(np.arange(1, len(labels) + 1))
    ax2.set_xticklabels(labels, fontweight='bold')
    ax2.set_ylabel('Magnitude (Log Scale)', fontweight='bold')
    ax2.set_title('B) Erosion Object Statistics', loc='left', fontweight='bold', pad=15)
    ax2.grid(True, axis='y', ls=":", alpha=0.5)

    # ==================== PANEL C: LORENZ & GINI ====================
    ax3 = fig.add_subplot(gs[1, 0])
    
    sorted_vols = np.sort(df_clean['volume'])[::-1]
    cum_vol = np.cumsum(sorted_vols)
    cum_vol_norm = cum_vol / cum_vol[-1] * 100
    events_norm = np.arange(1, len(sorted_vols) + 1) / len(sorted_vols) * 100
    
    gini = calculate_gini(df_clean['volume'].values)
    
    ax3.plot(events_norm, cum_vol_norm, linewidth=5, color=COLOR_MAIN)
    ax3.plot([0, 100], [0, 100], '--', color='gray', alpha=0.5, label='Uniform (G=0)', linewidth=2)
    
    idx_5 = int(len(sorted_vols) * 0.05)
    val_5 = cum_vol_norm[idx_5] if idx_5 < len(cum_vol_norm) else 100
    
    ax3.plot(5, val_5, 'o', color=COLOR_ACCENT, markersize=12)
    ax3.annotate(f"Top 5% events =\n{val_5:.1f}% of Volume", xy=(5, val_5), xytext=(15, val_5-15),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=16, bbox=dict(facecolor='white', alpha=0.9, edgecolor=COLOR_ACCENT))
    
    # Stats Box (Moved Further Left)
    stats_text = f"Gini Coeff (G) = {gini:.2f}"
    ax3.text(0.45, 0.1, stats_text, 
             transform=ax3.transAxes, fontsize=20, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)
    ax3.set_xlabel('% of Events', fontweight='bold')
    ax3.set_ylabel('% of Total Volume', fontweight='bold')
    ax3.set_title('C) Lorenz Curve', loc='left', fontweight='bold', pad=15)
    ax3.grid(True, linestyle=':', alpha=0.5)

    # ==================== PANEL D: SEASONALITY ROSE ====================
    ax4 = fig.add_subplot(gs[1, 1], projection='polar')
    
    total_vol = monthly_vol.sum()
    monthly_pct = (monthly_vol / total_vol) * 100
    
    theta = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
    width = (2 * np.pi) / 12
    ax4.set_theta_zero_location("N")
    ax4.set_theta_direction(-1)
    
    bars = ax4.bar(theta, monthly_pct.values, width=width, bottom=0.0, 
                   color=COLOR_ROSE, alpha=0.8, edgecolor='white')
    
    # Shading Logic: Indices 0-2 (Jan-Mar) and 9-11 (Oct-Dec) are Winter
    for i, bar in enumerate(bars):
        if i in [0, 1, 2, 9, 10, 11]: 
            bar.set_facecolor(COLOR_MAIN) # Dark Blue for Oct 1 - Mar 31
        else:
            bar.set_facecolor('#c6dbef') # Light Blue for Apr 1 - Sep 30

    ax4.set_xticks(theta)
    ax4.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 
                        fontweight='bold')
    ax4.set_rlabel_position(200) 
    ax4.tick_params(axis='y', labelsize=14, labelcolor='black')
    
    ax4.set_title('D) Seasonality (% of Annual Erosion)', loc='left', fontweight='bold', pad=25)

    # ==================== SAVE ====================
    plt.suptitle(f"{title_prefix} Geomorphology Dashboard", fontsize=32, fontweight='bold', y=0.96)
    
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{title_prefix}_Geomorph_Stats_{timestamp}.png"
    save_path = os.path.join(out_dir, filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[Success] Dashboard saved to: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default='all')
    parser.add_argument('--raw', action='store_true',
                        help='Use original histogram-based Panel A instead of '
                             'the normalized cumulative frequency curve.')
    args = parser.parse_args()

    normalized = not args.raw

    base_dir = get_base_dir()
    out_dir = os.path.join(base_dir, "figures", "stats_dashboard")

    if args.location.lower() == 'all':
        print("\n" + "="*60)
        print("GENERATING INDIVIDUAL LOCATION DASHBOARDS")
        print("="*60)
        for loc in LOCATIONS_ALL:
            print(f"\n--- Processing {loc} ---")
            df = collect_all_data(base_dir, [loc])
            if not df.empty:
                print(f"Generating plot for {len(df)} events at {loc}...")
                plot_geomorph_stats(df, out_dir, loc, normalized=normalized)
            else:
                print(f"No valid event data found for {loc}.")

        print("\n" + "="*60)
        print("GENERATING COMBINED REGIONAL DASHBOARD")
        print("="*60)
        df_all = collect_all_data(base_dir, LOCATIONS_ALL)
        if not df_all.empty:
            print(f"Generating combined plot for {len(df_all)} events across all locations...")
            plot_geomorph_stats(df_all, out_dir, "Regional", normalized=normalized)
        else:
            print("No valid event data found for combined regional plot.")
    else:
        df = collect_all_data(base_dir, [args.location])
        if not df.empty:
            print(f"Generating plot for {len(df)} events...")
            plot_geomorph_stats(df, out_dir, args.location, normalized=normalized)
        else:
            print("No valid event data found.")

if __name__ == "__main__":
    main()
