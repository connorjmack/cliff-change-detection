import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import os
import numpy as np
import laspy
import platform
import multiprocessing
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# 1. OS Detection & Paths
if platform.system() == 'Darwin':
    LIDAR_ROOT = "/Volumes/group/LiDAR"
    PROJECT_ROOT = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
else:
    LIDAR_ROOT = "/project/group/LiDAR"
    PROJECT_ROOT = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

CSV_PATH = os.path.join(PROJECT_ROOT, 'survey_lists', 'surveys_DelMar.csv')

# Search roots
POTENTIAL_ROOTS = [
    os.path.join(LIDAR_ROOT, "VMZ2000_Truck", "LiDAR_Processed_Level2"),
    os.path.join(LIDAR_ROOT, "MiniRanger_Truck", "LiDAR_Processed_Level2"),
    os.path.join(LIDAR_ROOT, "MiniRanger_ATV", "LiDAR_Processed_Level2"),
    os.path.join(LIDAR_ROOT, "VMQLZ_Truck", "LiDAR_Processed_Level2")
]

# 2. Optimization Settings
GRID_CELL_SIZE = 1.0     # Meter
CHUNK_SIZE = 1_000_000   # Points per chunk read
MAX_SAMPLE_POINTS = 20_000_000  # <--- INCREASED to 10 million

# 3. Instrument Colors
COLOR_MAP = {
    "VMZ2000": "#1f77b4",    # Blue
    "MiniRanger": "#ff7f0e", # Orange
    "VMQLZ": "#2ca02c",      # Green
    "Other": "#7f7f7f"       # Grey
}

# ============================================================================
# WORKER FUNCTIONS
# ============================================================================

def get_instrument_type(path):
    """Determines instrument type from file path string."""
    path_str = str(path)
    if "VMZ2000" in path_str:
        return "VMZ2000"
    elif "MiniRanger" in path_str:
        return "MiniRanger"
    elif "VMQLZ" in path_str:
        return "VMQLZ"
    else:
        return "Other"

def find_file(path_entry):
    """Replicates the 'Old Version' search logic."""
    raw_entry = str(path_entry).strip().rstrip('/')
    folder_name = os.path.basename(raw_entry)
    target_filename = f"{folder_name}_beach_cliff_ground.las"
    
    # Strategy 1: Direct path from CSV
    direct_path = os.path.join(raw_entry, "Beach_And_Backshore", target_filename)
    if os.path.exists(direct_path):
        return direct_path
    
    # Strategy 2: Search Roots (if CSV path is stale/moved)
    for root in POTENTIAL_ROOTS:
        candidate = os.path.join(root, folder_name, "Beach_And_Backshore", target_filename)
        if os.path.exists(candidate):
            return candidate
    return None

def calculate_estimated_density(las_path, cell_size=1.0, max_sample=MAX_SAMPLE_POINTS):
    occupied_cells = set()
    sample_points = 0
    
    try:
        with laspy.open(las_path) as f:
            header = f.header
            off_x, off_y = header.x_offset, header.y_offset
            
            for chunk in f.chunk_iterator(CHUNK_SIZE):
                points_in_chunk = len(chunk)
                if points_in_chunk == 0: continue
                
                sample_points += points_in_chunk
                gx = np.floor((chunk.x - off_x) / cell_size).astype(np.int64)
                gy = np.floor((chunk.y - off_y) / cell_size).astype(np.int64)
                occupied_cells.update(zip(gx, gy))
                
                if sample_points >= max_sample:
                    break

        occupied_area = len(occupied_cells) * (cell_size ** 2)
        if occupied_area == 0: return 0.0
        return sample_points / occupied_area

    except Exception:
        return None

def process_survey(path_entry):
    found_path = find_file(path_entry)
    
    if found_path:
        try:
            # 1. Identify Instrument
            instrument = get_instrument_type(found_path)

            # 2. File size
            size_gb = os.path.getsize(found_path) / (1000**3)
            
            # 3. Density Estimate
            density = calculate_estimated_density(found_path, GRID_CELL_SIZE)
            
            if density is not None:
                return {
                    'file_size_gb': size_gb,
                    'density': density,
                    'path': found_path,
                    'instrument': instrument,
                    'status': 'found'
                }
        except Exception:
            pass
            
    return {'status': 'missing', 'path': path_entry}

# ============================================================================
# MAIN
# ============================================================================

def main():
    # ---------------------------------------------------------
    # VISUALIZATION SETTINGS (LARGE FONTS)
    # ---------------------------------------------------------
    sns.set_style("whitegrid")
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 22
    })

    # ---------------------------------------------------------
    # 1. Setup Parallelism
    # ---------------------------------------------------------
    total_cores = multiprocessing.cpu_count()
    n_workers = max(1, total_cores // 2) 
    
    print(f"Reading survey list: {CSV_PATH}")
    print(f"Parallel Workers: {n_workers}")
    
    if not os.path.exists(CSV_PATH):
        print(f"CRITICAL ERROR: CSV not found at {CSV_PATH}")
        return

    df_input = pd.read_csv(CSV_PATH)
    survey_paths = df_input['path'].tolist()
    
    results_list = []
    missing_count = 0

    # ---------------------------------------------------------
    # 2. Process Files
    # ---------------------------------------------------------
    print(f"Scanning {len(survey_paths)} files...")
    with multiprocessing.Pool(processes=n_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_survey, survey_paths), total=len(survey_paths)):
            if result['status'] == 'found':
                results_list.append(result)
            else:
                missing_count += 1

    # ---------------------------------------------------------
    # 3. Calculate Stats & Plot
    # ---------------------------------------------------------
    res_df = pd.DataFrame(results_list)

    if not res_df.empty:
        n_files = len(res_df)
        
        # --- STATISTICS ---
        avg_size = res_df['file_size_gb'].mean()
        med_size = res_df['file_size_gb'].median()
        total_size_gb = res_df['file_size_gb'].sum()
        total_size_tb = total_size_gb / 1024.0  # <--- NEW: Convert GB to TB
        
        avg_density = res_df['density'].mean()
        med_density = res_df['density'].median()
        
        inst_counts = res_df['instrument'].value_counts()
        
        print(f"\nStats (n={n_files}):")
        print(f"  - Size:    Mean={avg_size:.2f} GB, Total={total_size_tb:.2f} TB")
        print(f"  - Density: Mean={avg_density:.1f}, Median={med_density:.1f} pts/m²")
        print(f"  - Missing: {missing_count}")
        print(f"  - Instruments: {inst_counts.to_dict()}")

        output_dir = os.path.join(PROJECT_ROOT, 'figures', 'quality_metrics')
        os.makedirs(output_dir, exist_ok=True)

        # ====================================================================
        # FIGURE 1: Original Summary (Strip Plot)
        # ====================================================================
        fig1, axes1 = plt.subplots(1, 2, figsize=(16, 9))
        
        # --- LEFT PLOT (File Size) ---
        sns.boxplot(data=res_df, y='file_size_gb', ax=axes1[0], color='lightgrey', 
                   width=0.4, showfliers=False, boxprops={'alpha': 0.5}) 
        
        sns.stripplot(data=res_df, y='file_size_gb', hue='instrument', ax=axes1[0],
                     palette=COLOR_MAP, size=8, alpha=0.8, jitter=True, legend=False)
        
        # Updated Title with Total in TB
        axes1[0].set_title(f'File Size Distribution\n(Mean: {avg_size:.2f} | Median: {med_size:.2f} | Total: {total_size_tb:.2f} TB)', 
                           fontweight='bold')
        axes1[0].set_ylabel('Size (GB)')

        # --- RIGHT PLOT (Density) ---
        sns.boxplot(data=res_df, y='density', ax=axes1[1], color='lightgrey',
                   width=0.4, showfliers=False, boxprops={'alpha': 0.5}) 
        
        sns.stripplot(data=res_df[res_df['density'] < res_df['density'].quantile(0.99)], 
                      y='density', hue='instrument', ax=axes1[1],
                      palette=COLOR_MAP, size=8, alpha=0.8, jitter=True, legend=False)
        
        # Removed Red Line Logic Here
        
        axes1[1].set_title(f'Point Density Estimate\n(Mean: {avg_density:.1f} | Median: {med_density:.1f} pts/m²)', 
                           fontweight='bold')
        axes1[1].set_ylabel('Density (pts/m²)')
        
        # --- CUSTOM LEGEND (Left Plot Only) ---
        legend_handles = []
        for inst_name, count in inst_counts.items():
            if inst_name in COLOR_MAP:
                label_text = f"{inst_name} (n={count})"
                patch = mpatches.Patch(color=COLOR_MAP[inst_name], label=label_text)
                legend_handles.append(patch)
        
        axes1[0].legend(handles=legend_handles, title='Survey Counts', loc='upper right', frameon=True)

        plt.suptitle(f'LiDAR Survey Quality Summary (n={n_files})', fontweight='bold')
        plt.tight_layout()
        out_file1 = os.path.join(output_dir, 'survey_quality_summary.png')
        fig1.savefig(out_file1, dpi=300, bbox_inches='tight')
        print(f"\nFigure 1 saved: {out_file1}")


        # ====================================================================
        # FIGURE 2: Instrument Comparison
        # ====================================================================
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 9))
        
        order = [i for i in ["VMZ2000", "MiniRanger", "VMQLZ"] if i in res_df['instrument'].unique()]

        # Plot 2A: File Size
        sns.boxplot(data=res_df, x='instrument', y='file_size_gb', order=order, ax=axes2[0],
                   palette=COLOR_MAP, showfliers=False)
        
        axes2[0].set_title('File Size by Instrument', fontweight='bold')
        axes2[0].set_ylabel('Size (GB)')
        axes2[0].set_xlabel('')

        # Plot 2B: Density
        sns.boxplot(data=res_df, x='instrument', y='density', order=order, ax=axes2[1],
                   palette=COLOR_MAP, showfliers=False)
        
        axes2[1].set_title('Point Density by Instrument', fontweight='bold')
        axes2[1].set_ylabel('Density (pts/m²)')
        axes2[1].set_xlabel('')

        plt.suptitle(f'Instrument Performance Comparison', fontweight='bold')
        plt.tight_layout()
        
        out_file2 = os.path.join(output_dir, 'survey_quality_by_instrument.png')
        fig2.savefig(out_file2, dpi=300, bbox_inches='tight')
        print(f"Figure 2 saved: {out_file2}")
        
    else:
        print("\nNo valid files found.")

if __name__ == "__main__":
    main()