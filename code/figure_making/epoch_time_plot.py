#!/usr/bin/env python3
"""
plot_epoch_times.py

Visualizes the "Epoch Times" (duration between consecutive surveys).
Updates:
  - Legend vertical.
  - X-axis years only.
  - Violin points smaller, mean label moved to top-left.
  - Violin plot gets a black outline.
"""

import os
import glob
import argparse
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from datetime import datetime

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================

if platform.system() == 'Darwin':
    LIDAR_ROOT = "/Volumes/group/LiDAR"
    PROJECT_ROOT = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
else:
    LIDAR_ROOT = "/project/group/LiDAR"
    PROJECT_ROOT = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'figures', 'epoch_times')

# Map folder paths to a simplified Instrument Label
# Note: Both MiniRanger paths map to the same "MiniRanger" label
INSTRUMENT_PATHS = [
    ("VMZ2000", os.path.join(LIDAR_ROOT, "VMZ2000_Truck", "LiDAR_Processed_Level2")),
    ("MiniRanger", os.path.join(LIDAR_ROOT, "MiniRanger_Truck", "LiDAR_Processed_Level2")),
    ("MiniRanger", os.path.join(LIDAR_ROOT, "MiniRanger_ATV", "LiDAR_Processed_Level2")),
    ("VMQLZ", os.path.join(LIDAR_ROOT, "VMQLZ_Truck", "LiDAR_Processed_Level2"))
]

COLOR_MAP = {
    "VMZ2000": "#1f77b4",    # Blue
    "MiniRanger": "#ff7f0e", # Orange
    "VMQLZ": "#2ca02c",      # Green
    "Unknown": "#7f7f7f"     # Grey
}

plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")
sns.set_context("talk")

# ==============================================================================
# WORKER FUNCTIONS
# ==============================================================================

def get_survey_list_path(location):
    return os.path.join(PROJECT_ROOT, 'survey_lists', f'surveys_{location}.csv')

def identify_instrument(date_obj):
    """
    Checks paths to see which instrument captured the survey on this date.
    Returns the simplified category name (e.g., 'MiniRanger').
    """
    date_str = date_obj.strftime("%Y%m%d")
    
    for label, root_path in INSTRUMENT_PATHS:
        if not os.path.exists(root_path):
            continue
        
        # Look for folder starting with YYYYMMDD
        pattern = os.path.join(root_path, f"{date_str}*")
        if glob.glob(pattern):
            return label
            
    return "Unknown"

def process_location(location):
    print(f"--- Processing {location} ---")
    csv_path = get_survey_list_path(location)
    
    if not os.path.exists(csv_path):
        print(f"  [WARN] No survey list found: {csv_path}")
        return None

    try:
        # Robust CSV reading (flatten logic)
        df_raw = pd.read_csv(csv_path, header=None)
        potential_dates = []
        for item in df_raw.values.flatten():
            item = str(item).strip()
            if len(item) >= 8 and item[:8].isdigit():
                potential_dates.append(item[:8])

        if not potential_dates:
            print("  [WARN] No dates found.")
            return None

        dates = sorted([datetime.strptime(d, "%Y%m%d") for d in potential_dates])
        
        data = []
        for i, date in enumerate(dates):
            inst = identify_instrument(date)
            
            # Calculate epoch (days since previous)
            if i == 0:
                epoch_days = np.nan
            else:
                delta = date - dates[i-1]
                epoch_days = delta.days
            
            data.append({
                "Date": date,
                "Instrument": inst,
                "Epoch_Days": epoch_days
            })
            
        df = pd.DataFrame(data)
        return df.dropna(subset=['Epoch_Days'])

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

def plot_dashboard(df, location):
    # Landscape aspect ratio (18, 5)
    fig = plt.figure(figsize=(18, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.1)
    
    ax_timeline = fig.add_subplot(gs[0])
    ax_dist = fig.add_subplot(gs[1], sharey=ax_timeline)

    # === PANEL 1: TIMELINE ===
    sns.scatterplot(
        data=df, x='Date', y='Epoch_Days', 
        hue='Instrument', palette=COLOR_MAP, 
        s=30, edgecolor='k', alpha=0.9, ax=ax_timeline, zorder=10
    )
    
    # Line connecting points
    ax_timeline.plot(df['Date'], df['Epoch_Days'], color='gray', linestyle='-', alpha=0.3, zorder=1)

    ax_timeline.set_title(f"Survey Intervals: {location}", fontweight='bold', loc='left')
    ax_timeline.set_ylabel("Days Between Surveys")
    ax_timeline.set_xlabel("")
    
    # Date Formatting: YEARS ONLY
    ax_timeline.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_timeline.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax_timeline.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    # Legend: Vertical (ncol=1)
    ax_timeline.legend(title=None, loc='upper left', frameon=True, ncol=1)

    # === PANEL 2: SINGLE VIOLIN DISTRIBUTION ===
    
    # Draw Violin with Outline
    # NOTE: seaborn violinplot doesn't natively support a simple 'edgecolor' kwarg easily for the outline
    # without complex dicts, but we can set linewidth and color via kwargs or iterating collections.
    # The clean way is usually linewidth/edgecolor directly if supported or setting inner to None.
    parts = sns.violinplot(
        y=df['Epoch_Days'], 
        ax=ax_dist, 
        color='#f0f0f0',  # Neutral grey background
        linewidth=1.5,    # Thicker outline
        inner=None,       # No inner box
        saturation=0.5
    )
    
    # Force black outline on the violin body
    for collection in parts.collections:
        collection.set_edgecolor('black')

    # Overlay colored points (SMALLER)
    sns.stripplot(
        y=df['Epoch_Days'], 
        hue=df['Instrument'], 
        palette=COLOR_MAP,
        ax=ax_dist,
        size=3,           # <--- Reduced Size
        edgecolor='k', 
        linewidth=0.5,
        jitter=0.30,
        alpha=0.9,
        legend=False
    )
    
    # Add Mean Line
    mean_val = df['Epoch_Days'].mean()
    ax_dist.axhline(mean_val, color='k', linestyle='--', alpha=0.6)
    
    # Move Mean Label to Upper Left (using axes coordinates)
    # x=0.05, y=0.95 places it in top-left corner
    ax_dist.text(
        0.05, 0.95,
        f"Mean: {mean_val:.2f} days",
        color='k',
        ha='left', va='top', fontweight='bold',
        transform=ax_dist.transAxes,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
    )

    ax_dist.set_title("Distribution", fontweight='bold')
    ax_dist.set_xlabel("")
    ax_dist.set_ylabel("")
    ax_dist.tick_params(left=False, labelleft=False) # Hide Y axis on right panel
    ax_dist.grid(True, axis='y', linestyle=':', alpha=0.5)

    # Final Polish
    sns.despine(ax=ax_timeline)
    sns.despine(ax=ax_dist, left=True)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"EpochTimes_{location}.png")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  ✓ Saved: {out_path}")
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default=None)
    args = parser.parse_args()

    # Determine locations
    if args.location:
        locations = [args.location]
    else:
        sl_dir = os.path.join(PROJECT_ROOT, 'survey_lists')
        files = glob.glob(os.path.join(sl_dir, 'surveys_*.csv'))
        locations = sorted([os.path.basename(f).replace('surveys_', '').replace('.csv', '') for f in files])
        print(f"Found {len(locations)} locations.")

    for loc in locations:
        df = process_location(loc)
        if df is not None and not df.empty:
            plot_dashboard(df, loc)
        else:
            print(f"  [SKIP] {loc} (Insufficient data)")

if __name__ == "__main__":
    main()