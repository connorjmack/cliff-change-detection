#!/usr/bin/env python3
"""
write_summary.py

Calculates specific geomorphological statistics for the paper text, filling in
placeholders for event counts, volumes, distributions, and seasonality.

Based on logic from: plot_geomorph_stats_v6.py

Usage:
    python3 write_summary.py --location DelMar
    python3 write_summary.py --location all
"""

import os
import re
import platform
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
RESOLUTION = '25cm'
RES_VAL = 0.25
CELL_AREA = RES_VAL * RES_VAL
FILE_TAG = '25cm'
LOCATIONS_ALL = ['DelMar', 'Torrey', 'Solana', 'Encinitas', 'SanElijo']

# ==============================================================================
# 1. DATA PROCESSING (Adapted from plot_geomorph_stats_v6.py)
# ==============================================================================

def get_base_dir():
    """Determines the base path based on the operating system."""
    if platform.system() == 'Darwin':
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

def parse_dates(folder_name):
    """Extracts start and end dates from folder naming convention."""
    match = re.search(r'(\d{8})_to_(\d{8})', folder_name)
    if match:
        return datetime.strptime(match.group(1), '%Y%m%d'), datetime.strptime(match.group(2), '%Y%m%d')
    return None, None

def extract_detailed_events(cluster_path, grid_path, res_val, date_mid):
    """Extracts stats for individual erosion events from cluster and grid CSVs."""
    try:
        df_c = pd.read_csv(cluster_path, index_col=0).fillna(0)
        df_g = pd.read_csv(grid_path, index_col=0).fillna(0)
        
        # Clean columns
        df_c.columns = [c.split('_')[-1] for c in df_c.columns]
        df_g.columns = [c.split('_')[-1] for c in df_g.columns]
        
        common_index = df_c.index.intersection(df_g.index)
        common_cols = df_c.columns.intersection(df_g.columns)
        
        if len(common_cols) == 0: return []
            
        df_c = df_c.loc[common_index, common_cols]
        df_g = df_g.loc[common_index, common_cols]
        
        c_vals = df_c.values
        g_vals = df_g.values
        
        # Determine Z (Elevation) map from headers or fallback to resolution steps
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
            
            # Key Physical Stats
            vol = np.sum(np.abs(dists)) * CELL_AREA
            width = (np.max(rows) - np.min(rows) + 1) * res_val # Alongshore span
            z_centroid = np.average(z_map[cols], weights=np.abs(dists))
            
            events.append({
                'volume': vol,
                'elevation': z_centroid,
                'width': width,
                'month': date_mid.month
            })
        return events
    except Exception as e:
        return []

def collect_all_data(base_dir, locations):
    """Iterates through folders to collect all event data."""
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
            grid_file = os.path.join(folder, f"{interval}_ero_grid_{FILE_TAG}_filled.csv")
            clus_file = os.path.join(folder, f"{interval}_ero_clusters_{FILE_TAG}_filled.csv")
            
            if os.path.exists(grid_file) and os.path.exists(clus_file):
                # Use midpoint date for seasonality
                date_mid = d1 + (d2 - d1)/2
                events = extract_detailed_events(clus_file, grid_file, RES_VAL, date_mid)
                all_events.extend(events)
                
    return pd.DataFrame(all_events)

def calculate_beta(volumes, cutoff=0.25):
    """Calculates power law beta for volumes above cutoff."""
    v_subset = volumes[volumes >= cutoff]
    if len(v_subset) < 10: return 0.0
    
    # Standard Maximum Likelihood Estimator for Power Law Beta
    # beta = 1 + n / sum(ln(x / x_min))
    n = len(v_subset)
    sum_log = np.sum(np.log(v_subset / cutoff))
    beta = 1 + n / sum_log
    return beta

# ==============================================================================
# 2. MAIN REPORT GENERATION
# ==============================================================================

def generate_report(df, location_name, output_path):
    if df.empty:
        print("No data found.")
        return

    # Filter very small noise for general stats (optional, consistent with plots)
    df_clean = df[df['volume'] >= 0.005].copy()

    # --- 1. General Counts ---
    X_events = len(df_clean)
    Y_volume = df_clean['volume'].sum()

    # --- 2. Magnitude-Frequency ---
    beta = calculate_beta(df_clean['volume'], cutoff=0.25)

    # --- 3. Individual Event Statistics (Median + IQR) ---
    # Volume
    vol_median = df_clean['volume'].median()
    vol_q1 = df_clean['volume'].quantile(0.25)
    vol_q3 = df_clean['volume'].quantile(0.75)

    # Elevation
    elev_median = df_clean['elevation'].median()
    elev_q1 = df_clean['elevation'].quantile(0.25)
    elev_q3 = df_clean['elevation'].quantile(0.75)

    # Width (Span) - using only median as per request
    width_median = df_clean['width'].median()

    # --- 4. Inequality (Top 5%) ---
    sorted_vols = df_clean['volume'].sort_values(ascending=False)
    top_5_count = int(np.ceil(len(sorted_vols) * 0.05))
    top_5_vol = sorted_vols.iloc[:top_5_count].sum()
    inequality_pct = (top_5_vol / Y_volume) * 100

    # --- 5. Seasonality ---
    # Winter = Oct (10), Nov (11), Dec (12), Jan (1), Feb (2), Mar (3)
    winter_months = [10, 11, 12, 1, 2, 3]
    winter_vol = df_clean[df_clean['month'].isin(winter_months)]['volume'].sum()
    winter_pct = (winter_vol / Y_volume) * 100

    # December contribution
    dec_vol = df_clean[df_clean['month'] == 12]['volume'].sum()
    dec_pct = (dec_vol / Y_volume) * 100

    # --- WRITE OUTPUT ---
    with open(output_path, "w") as f:
        f.write(f"GEOMORPHOLOGY STATS REPORT: {location_name}\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*60 + "\n\n")

        f.write("--- TEXT SNIPPET FILL-INS ---\n\n")
        
        f.write(f"1. [X] discrete erosion events: {X_events}\n")
        f.write(f"2. [Y] m^3 of material loss: {Y_volume:.2f}\n")
        f.write(f"3. Power-law scaling (beta): {beta:.2f}\n\n")
        
        f.write(f"4. Individual Events:\n")
        f.write(f"   - Volume: {vol_median:.3f} m^3 (IQR: {vol_q1:.3f}-{vol_q3:.3f})\n")
        f.write(f"   - Elevation: {elev_median:.2f} m NAVD88 (IQR: {elev_q1:.2f}-{elev_q3:.2f})\n")
        f.write(f"   - Span (Alongshore): {width_median:.2f} m\n\n")
        
        f.write(f"5. Inequality:\n")
        f.write(f"   - Largest 5% account for: {inequality_pct:.1f}% of total loss\n\n")
        
        f.write(f"6. Seasonality:\n")
        f.write(f"   - Winter (Oct-Mar) loss: {winter_pct:.1f}%\n")
        f.write(f"   - December loss: {dec_pct:.1f}%\n")

    print(f"Report written to: {output_path}")
    
    # Print preview to console
    print("\n--- PREVIEW ---")
    print(f"Detected {X_events} events totaling {Y_volume:.1f} m3.")
    print(f"Median Vol: {vol_median:.3f} m3. Top 5% = {inequality_pct:.1f}% of total.")
    print(f"Winter: {winter_pct:.1f}%. December: {dec_pct:.1f}%.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default='all', help="Location name or 'all'")
    args = parser.parse_args()
    
    base_dir = get_base_dir()
    output_dir = os.path.join(base_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    if args.location.lower() == 'all':
        # Regional Stats
        print("Calculating Combined Regional Statistics...")
        df_all = collect_all_data(base_dir, LOCATIONS_ALL)
        out_file = os.path.join(output_dir, "summary_stats_Regional.txt")
        generate_report(df_all, "Regional (All Locations)", out_file)
    else:
        # Single Location
        print(f"Calculating Statistics for {args.location}...")
        df = collect_all_data(base_dir, [args.location])
        out_file = os.path.join(output_dir, f"summary_stats_{args.location}.txt")
        generate_report(df, args.location, out_file)

if __name__ == "__main__":
    main()