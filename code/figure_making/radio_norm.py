#!/usr/bin/env python3
"""
Radiometric Normalization & Visualization Script
------------------------------------------------
1. Locates one representative LAS file for each instrument (VMZ2000, MiniRanger, VMQLZ).
2. Uses VMZ2000 as the radiometric baseline.
3. Normalizes MiniRanger and VMQLZ intensities to match the VMZ2000 distribution.
4. Generates a publication-quality comparison figure.
"""

import os
import platform
import glob
import pandas as pd
import numpy as np
import laspy
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================

# Detect OS and set roots
if platform.system() == 'Darwin':
    LIDAR_ROOT = "/Volumes/group/LiDAR"
    PROJECT_ROOT = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
else:
    LIDAR_ROOT = "/project/group/LiDAR"
    PROJECT_ROOT = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

# Input/Output paths
CSV_PATH = os.path.join(PROJECT_ROOT, 'survey_lists', 'surveys_DelMar.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'figures', 'random_forest')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Search roots for LAS files
POTENTIAL_ROOTS = [
    os.path.join(LIDAR_ROOT, "VMZ2000_Truck", "LiDAR_Processed_Level2"),
    os.path.join(LIDAR_ROOT, "MiniRanger_Truck", "LiDAR_Processed_Level2"),
    os.path.join(LIDAR_ROOT, "MiniRanger_ATV", "LiDAR_Processed_Level2"),
    os.path.join(LIDAR_ROOT, "VMQLZ_Truck", "LiDAR_Processed_Level2")
]

# Visualization Settings
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
COLORS = {
    "VMZ2000": "#1f77b4",    # Blue (Baseline)
    "MiniRanger": "#ff7f0e", # Orange
    "VMQLZ": "#2ca02c"       # Green
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_instrument_type(path):
    """Determines instrument type from file path string."""
    path_str = str(path)
    if "VMZ2000" in path_str:
        return "VMZ2000"
    elif "MiniRanger" in path_str:
        return "MiniRanger"
    elif "VMQLZ" in path_str:
        return "VMQLZ"
    return None

def find_file(survey_folder_name):
    """Locates the beach_cliff_ground.las file for a given survey folder."""
    target_filename = f"{survey_folder_name}_beach_cliff_ground.las"
    
    # Iterate through known processed data roots
    for root in POTENTIAL_ROOTS:
        candidate = os.path.join(root, survey_folder_name, "Beach_And_Backshore", target_filename)
        if os.path.exists(candidate):
            return candidate
    return None

def find_representative_files(csv_path):
    """Scans the CSV to find one valid file per instrument type."""
    print(f"[SEARCH] Scanning {os.path.basename(csv_path)} for instruments...")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Survey CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # We look for 'path' or 'folder' column usually found in these CSVs
    # Assuming the user's CSV structure implies the column has the folder name
    # If the CSV has a header like 'path', we use that.
    col_name = df.columns[0] # Fallback to first column
    for potential in ['path', 'survey', 'folder', 'name']:
        if potential in df.columns:
            col_name = potential
            break
            
    found_files = {}
    required = {"VMZ2000", "MiniRanger", "VMQLZ"}

    for _, row in df.iterrows():
        # Clean the entry to get just the folder name
        raw_entry = str(row[col_name]).strip().rstrip('/')
        folder_name = os.path.basename(raw_entry)
        
        full_path = find_file(folder_name)
        
        if full_path:
            inst = get_instrument_type(full_path)
            if inst and inst in required and inst not in found_files:
                found_files[inst] = full_path
                print(f"  ✓ Found {inst}: {os.path.basename(full_path)}")
        
        if len(found_files) == 3:
            break
            
    return found_files

def match_histograms(source, reference):
    """
    Warps 'source' values to match the distribution shape of 'reference' via Quantile Mapping.
    """
    # 1. Sort both arrays
    src_sorted = np.sort(source)
    ref_sorted = np.sort(reference)
    
    # 2. Create quantiles
    src_indices = np.linspace(0, 1, len(source))
    ref_indices = np.linspace(0, 1, len(reference))
    
    # 3. Interpolate the reference values at the source quantiles
    ref_interp = np.interp(src_indices, ref_indices, ref_sorted)
    
    # 4. Map back to original order
    sorter = np.argsort(source)
    inverse_sorter = np.argsort(sorter)
    
    return ref_interp[inverse_sorter]

def load_intensity(path, sample_size=500_000):
    """Reads LAS intensity and downsamples for efficient plotting."""
    with laspy.open(path) as f:
        # Check if file is huge, if so, sparse read
        if f.header.point_count > sample_size:
            # Random sampling via skipping
            step = max(1, f.header.point_count // sample_size)
            las = f.read()
            intensity = las.intensity[::step].astype(float)
        else:
            las = f.read()
            intensity = las.intensity.astype(float)
    return intensity

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    # A. Locate Files
    files_map = find_representative_files(CSV_PATH)
    
    if "VMZ2000" not in files_map:
        raise RuntimeError("Could not find a VMZ2000 file to serve as the baseline!")

    # B. Load Data
    print("\n[LOAD] Reading LAS files...")
    data_raw = {}
    
    # Load Baseline
    print(f"  Loading Baseline (VMZ2000)...")
    data_raw["VMZ2000"] = load_intensity(files_map["VMZ2000"])
    
    # Load Targets
    for inst in ["MiniRanger", "VMQLZ"]:
        if inst in files_map:
            print(f"  Loading Target ({inst})...")
            data_raw[inst] = load_intensity(files_map[inst])

    # C. Normalize
    print("\n[PROCESS] Normalizing distributions...")
    data_norm = {}
    
    # VMZ2000 stays as is
    data_norm["VMZ2000"] = data_raw["VMZ2000"]
    
    ref_dist = data_raw["VMZ2000"]
    
    for inst in ["MiniRanger", "VMQLZ"]:
        if inst in data_raw:
            print(f"  Matching {inst} -> VMZ2000...")
            data_norm[inst] = match_histograms(data_raw[inst], ref_dist)

    # D. Visualize
    print("\n[PLOT] Generating figure...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Define common plot kwargs
    kde_kws = {"fill": True, "alpha": 0.3, "linewidth": 2}

    # --- Panel 1: Raw Distributions ---
    ax_raw = axes[0]
    for inst, intensities in data_raw.items():
        sns.kdeplot(intensities, ax=ax_raw, color=COLORS[inst], label=inst, **kde_kws)
    
    ax_raw.set_title("Before Normalization (Raw Intensity)", fontweight='bold')
    ax_raw.set_xlabel("Intensity Value")
    ax_raw.set_ylabel("Density")
    ax_raw.legend(title="Instrument")
    ax_raw.set_xlim(0, 65535) # Assuming 16-bit intensity, adjust if needed (e.g. 0-4000)

    # --- Panel 2: Normalized Distributions ---
    ax_norm = axes[1]
    
    # Plot Baseline again
    sns.kdeplot(data_norm["VMZ2000"], ax=ax_norm, color=COLORS["VMZ2000"], 
                label="VMZ2000 (Baseline)", **kde_kws)
    
    # Plot Normalized Targets
    for inst in ["MiniRanger", "VMQLZ"]:
        if inst in data_norm:
            sns.kdeplot(data_norm[inst], ax=ax_norm, color=COLORS[inst], 
                        label=f"{inst} (Matched)", **kde_kws)

    ax_norm.set_title("After Normalization (Matched to VMZ2000)", fontweight='bold')
    ax_norm.set_xlabel("Intensity Value")
    ax_norm.legend(title="Instrument")
    ax_norm.set_xlim(0, 65535)

    # # Add text annotation
    # ax_norm.text(0.5, 0.5, "Distributions Aligned", 
    #              transform=ax_norm.transAxes, ha='center', va='center',
    #              bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

    plt.tight_layout()
    
    # E. Save
    out_file = os.path.join(OUTPUT_DIR, "Figure_Radiometric_Normalization_Seaborn.png")
    plt.savefig(out_file, dpi=300)
    print(f"\n[DONE] Figure saved to: {out_file}")

if __name__ == "__main__":
    main()