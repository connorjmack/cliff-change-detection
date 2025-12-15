#!/usr/bin/env python3
"""
11_plot_classification_stats_v7.py

Analyzes point cloud reduction across pipeline stages:
  1. Cropped (Raw) -> NoBeach -> NoVeg

Updates:
  - STYLE CHANGE: Clear violins with thick, colored outlines (Sand/Green/Brown).
  - INNER LINES: Only shows the Median line (colored), IQR removed.
  - FONT SIZES: Significantly increased all plot labels and text.
  - Sensor Mapping: Continues to use dynamic CSV mapping.

Usage:
  python3 11_plot_classification_stats_v7.py                # Process ALL locations
  python3 11_plot_classification_stats_v7.py --location DelMar
"""

import os
import glob
import argparse
import platform
import laspy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# === CONFIGURATION ===
ALL_LOCATIONS = ['DelMar', 'Encinitas', 'SanElijo', 'Solana', 'Torrey']

def get_base_path():
    if platform.system() == "Darwin":
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

OUTPUT_DIR = os.path.join(get_base_path(), "figures", "ml_classification", "figs")

# --- COLORS ---
# Violin Colors (Material - for Outlines and Median)
HEX_BEACH = '#EDD9A3'  # Sand/Gold
HEX_VEG   = '#228B22'  # Forest Green
HEX_CLIFF = '#A0522D'  # Sienna/Brown

CLASS_PALETTE = {
    "Beach": HEX_BEACH,
    "Vegetation": HEX_VEG,
    "Cliff": HEX_CLIFF
}

# Sensor Colors (Points) - High Contrast Palette
SENSOR_PALETTE = {
    "MiniRanger": "#E69F00",  # Orange/Gold
    "VZ2000":     "#56B4E9",  # Sky Blue
    "VMQLZ":      "#009E73",  # Teal Green
    "Unknown":    "#999999"   # Grey
}

# ==============================================================================
# 1. SENSOR MAPPING LOGIC
# ==============================================================================

def load_sensor_map(location, base_dir):
    """
    Loads 'surveys_{location}.csv' and returns a dict: {date_str: sensor_label}
    """
    csv_path = os.path.join(base_dir, "survey_lists", f"surveys_{location}.csv")
    
    if not os.path.exists(csv_path):
        print(f"[WARN] Survey list not found for {location}: {csv_path}")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        df['date_str'] = df['date'].astype(str)
        
        date_to_sensor = {}
        for _, row in df.iterrows():
            method = str(row.get('method', '')).strip()
            date_val = row['date_str']
            
            if 'ATV' in method or 'MiniRanger' in method:
                sensor = "MiniRanger"
            elif 'VMZ2000' in method:
                sensor = "VZ2000"
            elif 'VMQLZ' in method:
                sensor = "VMQLZ"
            else:
                sensor = "Unknown"
            
            date_to_sensor[date_val] = sensor
            
        return date_to_sensor
    except Exception as e:
        print(f"[ERROR] Failed to read CSV for {location}: {e}")
        return {}

# ==============================================================================
# 2. DATA COLLECTION
# ==============================================================================

def get_point_count(path):
    try:
        with laspy.open(path) as f:
            return f.header.point_count
    except:
        return None

def find_matching_file(directory, date_str):
    if not os.path.exists(directory):
        return None
    pattern = os.path.join(directory, f"{date_str}*.las")
    matches = glob.glob(pattern)
    if matches:
        return sorted(matches, key=len)[0]
    return None

def process_survey_triplet(task):
    loc, date_str, crop_path, base_dir, sensor_label = task
    
    nobeach_dir = os.path.join(base_dir, "results", loc, "nobeach")
    noveg_dir   = os.path.join(base_dir, "results", loc, "noveg")
    
    nobeach_path = find_matching_file(nobeach_dir, date_str)
    noveg_path   = find_matching_file(noveg_dir, date_str)
    
    if not nobeach_path or not noveg_path:
        return None 

    c_crop = get_point_count(crop_path)
    c_nobeach = get_point_count(nobeach_path)
    c_noveg = get_point_count(noveg_path)
    
    if any(c is None for c in [c_crop, c_nobeach, c_noveg]) or c_crop == 0:
        return None

    if c_nobeach > c_crop or c_noveg > c_nobeach:
        return None 

    count_beach = max(0, c_crop - c_nobeach)
    count_veg   = max(0, c_nobeach - c_noveg)
    count_cliff = c_noveg
    
    pct_beach = (count_beach / c_crop) * 100
    pct_veg   = (count_veg / c_crop) * 100
    pct_cliff = (count_cliff / c_crop) * 100
    
    return [
        {"Location": loc, "Date": date_str, "Sensor": sensor_label, "Class": "Beach", "Percentage": pct_beach},
        {"Location": loc, "Date": date_str, "Sensor": sensor_label, "Class": "Vegetation", "Percentage": pct_veg},
        {"Location": loc, "Date": date_str, "Sensor": sensor_label, "Class": "Cliff", "Percentage": pct_cliff}
    ]

def collect_data(locations):
    base_dir = get_base_path()
    tasks = []
    print(f"Scanning locations: {locations}")
    
    for loc in locations:
        sensor_map = load_sensor_map(loc, base_dir)
        cropped_dir = os.path.join(base_dir, "results", loc, "cropped")
        if not os.path.exists(cropped_dir): continue
        crop_files = glob.glob(os.path.join(cropped_dir, "*.las"))
        
        for cf in crop_files:
            basename = os.path.basename(cf)
            date_str = basename.split('_')[0]
            if len(date_str) == 8 and date_str.isdigit():
                sensor = sensor_map.get(date_str, "Unknown")
                tasks.append((loc, date_str, cf, base_dir, sensor))

    results = []
    max_workers = min(10, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(process_survey_triplet, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            res = fut.result()
            if res: results.extend(res)
                
    return pd.DataFrame(results)

# ==============================================================================
# 3. PLOTTING
# ==============================================================================

def remove_outliers(df):
    clean_dfs = []
    for cls in df["Class"].unique():
        subset = df[df["Class"] == cls].copy()
        Q1 = subset["Percentage"].quantile(0.25)
        Q3 = subset["Percentage"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        clean_subset = subset[(subset["Percentage"] >= lower_bound) & (subset["Percentage"] <= upper_bound)]
        clean_dfs.append(clean_subset)
    return pd.concat(clean_dfs)

def plot_violin(df, title_label, filename_label):
    if df.empty: return

    # 1. Filter Outliers
    df_clean = remove_outliers(df)
    
    # Increase figure size for larger fonts
    plt.figure(figsize=(14, 10), dpi=300)
    sns.set_style("ticks")
    # Use 'poster' context for very large labels by default
    sns.set_context("poster", font_scale=1.1) 

    class_order = ["Beach", "Vegetation", "Cliff"]

    # 2. Violin Plot (Structure only)
    # inner=None removes the default boxplot entirely
    ax = sns.violinplot(
        x="Class", 
        y="Percentage", 
        data=df_clean, 
        order=class_order,
        color="white",        # Initialize as white fill
        linewidth=4,          # Thick outlines like the reference image
        inner=None,           # Remove default median/IQR stuff
        width=0.85,
        cut=0,
        density_norm="width",
        zorder=1
    )

    # Post-processing: Make fill transparent and color the outlines
    # The collections are drawn in the order of the x-axis categories
    for i, collection in enumerate(ax.collections):
        if i < len(class_order):
            cls_name = class_order[i]
            col = CLASS_PALETTE[cls_name]
            collection.set_edgecolor(col)
            collection.set_facecolor('none') # Transparent fill

    # 3. Manually Add Median Lines
    medians = df_clean.groupby("Class", observed=False)["Percentage"].median()
    for i, cls_name in enumerate(class_order):
        med_val = medians.get(cls_name)
        if med_val is not None:
            # Draw horizontal line centered on the category index (i)
            plt.hlines(y=med_val, xmin=i-0.35, xmax=i+0.35, 
                       color=CLASS_PALETTE[cls_name], linewidth=4, zorder=3)

    # 4. Strip Plot (Colored by Sensor)
    sns.stripplot(
        x="Class", 
        y="Percentage", 
        hue="Sensor",
        data=df_clean, 
        order=class_order,
        palette=SENSOR_PALETTE,
        size=7,               # Larger points
        alpha=0.8,
        jitter=0.3,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
        dodge=False
    )

    # Styling - Large Fonts
    title_text = "All Locations" if title_label == "All_Locations" else title_label
    plt.title(f"Point Cloud Composition: {title_text}", fontsize=26, fontweight='bold', pad=25)
    plt.ylabel("Percentage of Total Points (%)", fontsize=22, fontweight='bold', labelpad=15)
    plt.xlabel("", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.ylim(-5, 105)
    
    # Add annotations (Large Font)
    means = df_clean.groupby("Class", observed=False)["Percentage"].mean()
    for i, cls_name in enumerate(class_order):
        val = means.get(cls_name, 0)
        plt.text(i, 102, f"Avg: {val:.1f}%", ha='center', fontsize=18, fontweight='bold', color='black')

    # Legend - Large Font
    plt.legend(title="LiDAR Sensor", title_fontsize=18, fontsize=16, 
               loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=4, frameon=False)

    plt.grid(axis='y', linestyle='--', alpha=0.4)
    sns.despine(trim=True)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"Classification_Violin_v7_Styled_{filename_label}.png"
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [FIGURE] Saved: {out_path}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default=None)
    args = parser.parse_args()

    locations = [args.location] if args.location else ALL_LOCATIONS
    full_df = collect_data(locations)
    
    if full_df.empty:
        print("[ERROR] No valid data found.")
        return

    print(f"\nCollected {len(full_df)//3} valid surveys. Generating plots...")

    if args.location:
        plot_violin(full_df, args.location, args.location)
    else:
        plot_violin(full_df, "All_Locations", "All_Locations")
        for loc in sorted(full_df['Location'].unique()):
            df_loc = full_df[full_df['Location'] == loc]
            plot_violin(df_loc, loc, loc)

if __name__ == "__main__":
    main()