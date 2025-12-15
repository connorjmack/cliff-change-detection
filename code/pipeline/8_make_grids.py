#!/usr/bin/env python3
import os
import platform
import argparse
import laspy
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import multiprocessing as mp
from functools import partial

# Set publication-quality plotting defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
sns.set_style("whitegrid")

def makeGrid(pathin, pathout_m3c2, pathout_cluster, pathout_uncertainty,
             polys, res, height, overwrite=False):
    """
    Reads in a LAS file and a shapefile of polygons, calculates:
      - median absolute M3C2 distance
      - mode of ClusterID
      - RMS of pointwise distance uncertainty
    for points within each polygon for vertical bins.
    
    Optimized to use Geopandas Spatial Joins (sjoin) and Pandas Aggregation.
    STRICT SHAPE ENFORCEMENT: Ensures all output CSVs have identical Row and Column dimensions.
    """
    start_time = time.time()
    stats = {}
    
    print(f"\n--- Processing LAS: {pathin} ---")
    
    # 1. READ LAS DATA
    with laspy.open(pathin) as lasf:
        las = lasf.read()
        
    dims = las.point_format.dimension_names
    
    # Identify fields
    m3c2_field = None
    unc_field = None
    for name in dims:
        lname = name.lower()
        if m3c2_field is None and "m3c2" in lname and "dist" in lname:
            m3c2_field = name
        if unc_field is None and "uncert" in lname:
            unc_field = name
            
    if m3c2_field is None or unc_field is None:
        raise KeyError(f"Required fields not found in {pathin}")

    stats['total_points'] = len(las.x)

    # 2. PREPARE ARRAYS (Numpy extraction)
    # Filter unclustered points (-1) early to save memory
    valid_mask = np.array(las['ClusterID']) != -1
    
    x = np.array(las.x)[valid_mask]
    y = np.array(las.y)[valid_mask]
    z = np.array(las.z)[valid_mask]
    # Take absolute value of M3C2 immediately
    m3c2 = np.abs(np.array(las[m3c2_field])[valid_mask])
    clusters = np.array(las['ClusterID'])[valid_mask]
    uncert = np.array(las[unc_field])[valid_mask]
    
    stats['clustered_points'] = len(x)
    
    # Basic Stats
    stats['n_clusters'] = len(np.unique(clusters))
    stats['mean_abs_m3c2'] = np.mean(m3c2)
    stats['mean_uncertainty'] = np.mean(uncert)

    # 3. LOAD POLYGONS
    polys_gdf = gpd.read_file(polys)
    polys_gdf["Polygon_ID"] = polys_gdf.index
    print(f"Loaded {len(polys_gdf)} polygons")
    stats['n_polygons'] = len(polys_gdf)

    # 4. OPTIMIZED SPATIAL JOIN
    print("Performing Spatial Join...")
    points_gdf = gpd.GeoDataFrame(
        {
            'Z': z, 
            'M3C2': m3c2, 
            'ClusterID': clusters, 
            'Uncertainty': uncert
        },
        geometry=gpd.points_from_xy(x, y),
        crs=polys_gdf.crs
    )

    # Use sjoin (Spatial Indexing) instead of loops
    joined = gpd.sjoin(points_gdf, polys_gdf[['Polygon_ID', 'geometry']],
                       how='inner', predicate='within')
    
    stats['points_in_polygons'] = len(joined)
    stats['polygons_with_data'] = joined['Polygon_ID'].nunique()

    # Handle empty case (create empty DF to ensure logic proceeds to grid generation)
    if len(joined) == 0:
        print("No points fell inside polygons. Generating empty grids.")
        joined = pd.DataFrame(columns=['Polygon_ID', 'Z', 'M3C2', 'ClusterID', 'Uncertainty', 'z_bin'])

    # 5. VERTICAL BINNING
    z_bins = np.arange(0, height + res, res)
    z_labels = [f"{b:.2f}" for b in z_bins[:-1]]
    joined['z_bin'] = pd.cut(joined['Z'], bins=z_bins, right=False, labels=z_labels)

    # 6. AGGREGATION
    print("Aggregating statistics...")
    
    def mode_func(x):
        # Safer mode calculation
        m = x.mode()
        return m.iloc[0] if not m.empty else np.nan
        
    def rms_func(x):
        return np.sqrt(np.mean(x**2))

    # observed=True is faster for Categorical data (z_bin)
    grp = joined.groupby(['Polygon_ID','z_bin'], observed=False).agg({
        'M3C2': 'median',
        'ClusterID': mode_func,
        'Uncertainty': rms_func
    })

    # 7. PIVOT & SAVE (STRICT SHAPE ENFORCEMENT)
    all_ids = polys_gdf['Polygon_ID']

    def save_pivot(col_name, output_path, prefix):
        # Pivot table
        if len(joined) > 0:
            df_pivot = grp[col_name].unstack()
        else:
            df_pivot = pd.DataFrame()

        # --- SHAPE ENFORCEMENT LOGIC ---
        
        # 1. Define the EXACT expected columns based on height and resolution
        expected_cols = [f"{prefix}_{lbl}m" for lbl in z_labels]
        
        # 2. Rename existing columns to match format (e.g. 0.10 -> M3C2_0.10m)
        # Note: df_pivot columns are currently just the z_labels (strings)
        if not df_pivot.empty:
            df_pivot.columns = [f"{prefix}_{c}m" for c in df_pivot.columns]

        # 3. Reindex ROWS (Polygons) AND COLUMNS (Height Bins)
        # This forces the DataFrame to have exactly 'all_ids' rows and 'expected_cols' columns.
        # Missing data is filled with NaN.
        df_pivot = df_pivot.reindex(index=all_ids, columns=expected_cols)
        
        # 4. Reset index for saving
        df_pivot = df_pivot.reset_index()
        
        if os.path.exists(output_path) and not overwrite:
            print(f"Skipping existing: {output_path}")
        else:
            df_pivot.to_csv(output_path, index=False)
            print(f"Wrote: {output_path} (Shape: {df_pivot.shape})")

    save_pivot('M3C2', pathout_m3c2, 'M3C2')
    save_pivot('ClusterID', pathout_cluster, 'ClusterID')
    save_pivot('Uncertainty', pathout_uncertainty, 'Uncertainty')
    
    stats['processing_time_sec'] = time.time() - start_time
    return stats


def find_shapefile(base_util, location, poly_res):
    """
    Locate the shapefile for a given location and polygon resolution.
    """
    sf_root = os.path.join(base_util, 'shape_files')
    candidates = [
        d for d in os.listdir(sf_root)
        if d.lower().startswith(location.lower())
           and poly_res.lower() in d.lower()
           and os.path.isdir(os.path.join(sf_root, d))
    ]
    if not candidates:
        raise FileNotFoundError(f"No folder for '{location}' with '{poly_res}' under {sf_root}")
    fld = candidates[0]
    shp_path = os.path.join(sf_root, fld, fld + '.shp')
    if not os.path.isfile(shp_path):
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")
    return shp_path


def process_single_survey(task_dict):
    """
    Worker function to process a single survey.
    """
    try:
        mode = task_dict['mode']
        dt = task_dict['date']
        overwrite = task_dict['overwrite']
        
        # Check outputs first
        outputs = (task_dict['grid_csv'], task_dict['cluster_csv'], task_dict['uncert_csv'])
        if not overwrite and all(os.path.exists(p) for p in outputs):
            return None
        
        # Check input
        if not os.path.isfile(task_dict['las_in']):
            return None
        
        print(f"[Worker PID {os.getpid()}] Processing {mode}/{dt}")
        stats = makeGrid(
            task_dict['las_in'],
            task_dict['grid_csv'],
            task_dict['cluster_csv'],
            task_dict['uncert_csv'],
            task_dict['shp'],
            task_dict['res'],
            task_dict['height'],
            overwrite=overwrite
        )
        
        # Add metadata
        stats['mode'] = mode
        stats['date'] = dt
        stats['location'] = task_dict['location']
        return stats
        
    except Exception as e:
        print(f"ERROR in {task_dict.get('mode','?')}/{task_dict.get('date','?')}: {e}")
        return None


def plot_performance_report(df_stats, output_dir, location):
    """
    Create publication-quality figures from performance statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert date strings to datetime
    df_stats['date'] = pd.to_datetime(df_stats['date'], format='%Y%m%d', errors='coerce')
    df_stats = df_stats.sort_values('date')
    
    # Figure 1: Processing time histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for i, mode in enumerate(['erosion', 'deposition']):
        data = df_stats[df_stats['mode'] == mode]['processing_time_sec']
        if len(data) > 0:
            axes[i].hist(data, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            axes[i].axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {data.mean():.1f}s')
            axes[i].set_xlabel('Processing Time (seconds)')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{mode.capitalize()} Processing Time')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{location}_processing_times.png'))
    plt.close()
    
    # Figure 2: Cluster counts over time
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for mode in ['erosion', 'deposition']:
        data = df_stats[df_stats['mode'] == mode]
        if len(data) > 0:
            ax.plot(data['date'], data['n_clusters'], marker='o', label=mode.capitalize())
    
    ax.set_ylabel('Number of Clusters')
    ax.set_title(f'{location}: Detected Change Clusters')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{location}_clusters_timeseries.png'))
    plt.close()
    
    # Figure 3: Stats Grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Points processed
    for mode in ['erosion', 'deposition']:
        data = df_stats[df_stats['mode'] == mode]
        if len(data) > 0:
            axes[0,0].plot(data['date'], data['total_points']/1000, marker='o', label=mode)
    axes[0,0].set_ylabel('Total Points (k)')
    axes[0,0].set_title('Point Cloud Size')
    
    # Clustering efficiency
    for mode in ['erosion', 'deposition']:
        data = df_stats[df_stats['mode'] == mode].copy()
        if len(data) > 0:
            data['cluster_pct'] = 100 * data['clustered_points'] / data['total_points']
            axes[0,1].plot(data['date'], data['cluster_pct'], marker='o', label=mode)
    axes[0,1].set_ylabel('Clustered Points (%)')
    axes[0,1].set_title('Clustering Efficiency')
    
    # Mean |M3C2|
    for mode in ['erosion', 'deposition']:
        data = df_stats[df_stats['mode'] == mode]
        if len(data) > 0:
            axes[1,0].plot(data['date'], data['mean_abs_m3c2']*100, marker='o', label=mode)
    axes[1,0].set_ylabel('Mean |M3C2| (cm)')
    axes[1,0].set_title('Change Magnitude')
    
    # Mean uncertainty
    for mode in ['erosion', 'deposition']:
        data = df_stats[df_stats['mode'] == mode]
        if len(data) > 0:
            axes[1,1].plot(data['date'], data['mean_uncertainty']*100, marker='o', label=mode)
    axes[1,1].set_ylabel('Mean Uncertainty (cm)')
    axes[1,1].set_title('Uncertainty')
    
    for ax in axes.flat:
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{location}_comprehensive_stats.png'))
    plt.close()


def generate_performance_report(df_stats, output_dir, location):
    """
    Generate a detailed text report of processing statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f'{location}_performance_report.txt')
    
    with open(report_path, 'w') as f:
        f.write(f"GRID MAKING REPORT: {location}\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total surveys: {len(df_stats)}\n")
        f.write(f"Total time: {df_stats['processing_time_sec'].sum()/60:.1f} min\n\n")
        
        for mode in ['erosion', 'deposition']:
            data = df_stats[df_stats['mode'] == mode]
            if len(data) > 0:
                f.write(f"{mode.upper()}:\n")
                f.write(f"  Count: {len(data)}\n")
                f.write(f"  Avg Time: {data['processing_time_sec'].mean():.2f}s\n")
                f.write(f"  Avg Clusters: {data['n_clusters'].mean():.1f}\n")
                f.write(f"  Avg |M3C2|: {data['mean_abs_m3c2'].mean()*100:.2f} cm\n\n")


def main():
    # Force 'spawn' to prevent Geopandas/NumPy deadlocks in multiprocessing
    mp.set_start_method('spawn', force=True)

    p = argparse.ArgumentParser(
        description="Grid dbscan.las by polygon (OPTIMIZED PARALLEL VERSION)"
    )
    p.add_argument("location", help="e.g., SanElijo or Solana")
    p.add_argument("--res", type=float, default=None,
                   help="vertical bin size (defaults to match resolution)")
    p.add_argument("--replace", action="store_true",
                   help="replace existing CSVs")
    p.add_argument("-t","--test", action="store_true",
                   help="process ONLY the first date")
    p.add_argument("--resolution", choices=["10cm","25cm","1m"], default="10cm",
                   help="polygon resolution to use")
    p.add_argument("--n_workers", type=int, default=None,
                   help="number of parallel workers (default: n_cores//4)")
    args = p.parse_args()

    # Determine number of workers
    n_cores = mp.cpu_count()
    n_workers = max(1, n_cores // 4) if args.n_workers is None else max(1, args.n_workers)
    
    print(f"\n{'='*60}")
    print(f"PARALLEL CONFIG: {n_cores} cores detected, using {n_workers} workers")
    print(f"{'='*60}\n")

    # Set default vertical resolution based on polygon resolution
    if args.res is None:
        if args.resolution == "10cm":
            args.res = 0.1
        elif args.resolution == "25cm":
            args.res = 0.25
        else:  # 1m
            args.res = 1.0

    heights = {'DelMar':30,'SanElijo':40,'Solana':50,
               'Encinitas':50,'Torrey':75,'Blacks':100}

    base = ("/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
            if platform.system()=="Darwin"
            else "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs")
    results_root = os.path.join(base,"results",args.location)
    util_root    = os.path.join(base,"utilities")
    
    # Output directory for reports
    report_dir = os.path.join(base, "utilities", "grid_making")

    shp = find_shapefile(util_root,args.location,args.resolution)
    print(f"Using shapefile: {shp}")

    res_cm = int(round(args.res*100))
    label = f"{res_cm}cm"
    
    # Build tasks
    tasks = []
    for mode in ("deposition","erosion"):
        mode_dir = os.path.join(results_root,mode)
        if not os.path.isdir(mode_dir):
            continue

        dates = sorted(d for d in os.listdir(mode_dir)
                       if os.path.isdir(os.path.join(mode_dir,d)))
        if args.test:
            dates = dates[:1]

        prefix = "dep" if mode=="deposition" else "ero"
        for dt in dates:
            out_base = os.path.join(mode_dir,dt)
            las_in   = os.path.join(out_base,"dbscan.las")
            
            task = {
                'mode': mode,
                'date': dt,
                'las_in': las_in,
                'grid_csv': os.path.join(out_base, f"{dt}_{prefix}_grid_{label}.csv"),
                'cluster_csv': os.path.join(out_base, f"{dt}_{prefix}_clusters_{label}.csv"),
                'uncert_csv': os.path.join(out_base, f"{dt}_{prefix}_uncertainty_{label}.csv"),
                'shp': shp,
                'res': args.res,
                'height': heights[args.location],
                'overwrite': args.replace,
                'location': args.location
            }
            tasks.append(task)
    
    print(f"Queued {len(tasks)} tasks.")
    if not tasks:
        return

    # Process
    start_time = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(process_single_survey, tasks)
    
    # Collect results
    all_stats = [r for r in results if r is not None]

    if all_stats:
        df_stats = pd.DataFrame(all_stats)
        stats_csv = os.path.join(report_dir, f'{args.location}_processing_stats.csv')
        df_stats.to_csv(stats_csv, index=False)
        
        generate_performance_report(df_stats, report_dir, args.location)
        plot_performance_report(df_stats, report_dir, args.location)
        
        print(f"\nProcessing Complete. Time: {(time.time()-start_time)/60:.1f} min")
        print(f"Reports saved to: {report_dir}")
    else:
        print("\nNo surveys processed (all skipped or failed).")

if __name__=="__main__":
    main()