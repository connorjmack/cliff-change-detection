# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a coastal cliff LiDAR processing pipeline that ingests raw terrestrial LiDAR surveys, preprocesses them, removes non-relevant features (beach, vegetation), performs change detection using M3C2, clusters erosion/deposition events with DBSCAN, and aggregates data into spatiotemporal grids for analysis.

## Environment Setup

### Installation Options

**Option 1: Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate cliff-change-detection
```

**Option 2: pip with venv**
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### External Dependencies

**PDAL**: Required for cropping operations. Must be installed at system level.
- macOS: `brew install pdal`
- Linux: `apt-get install pdal` or `conda install -c conda-forge pdal`

**CloudCompare**: Required for M3C2 and CANUPO (vegetation removal) operations.
- Must be accessible via CLI
- Pass path using `--cc /path/to/CloudCompare` flag

## Directory Structure Requirements

The pipeline expects a specific directory structure on shared storage:

```
/LidarProcessing/LidarProcessingCliffs/  (this repo)
├── code/pipeline/           # Main processing scripts (0-8 sequential)
├── results/<Location>/      # Output directories (e.g., DelMar, SanElijo)
│   ├── cropped/
│   ├── nobeach/
│   ├── noveg/
│   ├── m3c2/
│   ├── erosion/
│   │   └── DATE1_to_DATE2/
│   │       ├── ero_clusters.las
│   │       ├── ero_outliers.las
│   │       ├── 10cm/                 # Resolution-specific subdirectory
│   │       │   ├── DATE1_to_DATE2_ero_grid_10cm.csv
│   │       │   ├── DATE1_to_DATE2_ero_clusters_10cm.csv
│   │       │   └── DATE1_to_DATE2_ero_stats_10cm.npz
│   │       ├── 25cm/                 # Alternative resolution
│   │       └── 1m/                   # Alternative resolution
│   └── deposition/
│       └── DATE1_to_DATE2/
│           ├── dep_clusters.las
│           ├── dep_outliers.las
│           ├── 10cm/                 # Resolution-specific subdirectory
│           ├── 25cm/
│           └── 1m/
├── survey_lists/            # CSV inventories of surveys
└── utilities/
    ├── shape_files/         # Polygons for gridding
    ├── beach_removal/       # RF Models (.joblib) and Scalers
    ├── canupo/              # Vegetation classifiers (.prm)
    └── m3c2_params/         # CloudCompare parameter files

/LiDAR/MOPLines/
└── MOPs_SD_County.kml       # MOP line definitions
```

**Path Handling**: Scripts auto-detect OS and use:
- macOS: `/Volumes/group/LiDAR`
- Linux/HPC: `/project/group/LiDAR`

## Pipeline Execution Sequence

The pipeline consists of numbered scripts (0-8) that must be run sequentially. Each step transforms data for the next.

### Step 0: Create Survey Lists
```bash
# Initial creation of survey list
python3 code/pipeline/0_make_survey_lists.py --location SanElijo
python3 code/pipeline/0_make_survey_lists.py --all
```

### Step 1: Update Survey Lists
```bash
# Update with new surveys
python3 code/pipeline/1_update_survey_lists.py --location SanElijo
# or update all locations
python3 code/pipeline/1_update_survey_lists.py --all
```

### Step 2: Crop to Study Area
```bash
python3 code/pipeline/2_crop_files_parallel.py --location SanElijo --replace
```
Uses PDAL to crop raw LAS files to MOP line ranges defined in KML.

### Audit: Cropped Files QC (Optional)
```bash
# Generate QC report
python3 tests/audits/2_audit_cropping.py

# Remove bad files below threshold
python3 tests/audits/2_audit_cropping.py --delete_bad_files
```
Generates point-count vs file-size plots to identify corrupt scans.

### Step 3: Beach Removal
```bash
python3 code/pipeline/3_remove_beach_parallel.py SanElijo --n_jobs 5
```
Random Forest classifier using intensity/geometry features. Includes histogram matching to normalize intensity across sensors.

### Step 4: Vegetation Removal
```bash
python3 code/pipeline/4_remove_veg_parallel.py SanElijo --cc "/path/to/CloudCompare"
```
Uses CloudCompare's CANUPO plugin for geometric scale-based classification.

**Important**: Requires display environment. On headless servers:
```bash
xvfb-run --auto-servernum python3 code/pipeline/4_remove_veg_parallel.py SanElijo
```

### Step 5: M3C2 Change Detection
```bash
# Basic usage
python3 code/pipeline/5_m3c2_parallel.py SanElijo --cc "/path/to/CloudCompare"

# With replace flag to overwrite existing outputs
python3 code/pipeline/5_m3c2_parallel.py SanElijo --replace

# On headless servers (required)
xvfb-run -a python3 code/pipeline/5_m3c2_parallel.py SanElijo --replace

# Process all locations
for loc in DelMar Solana Encinitas SanElijo Torrey Blacks; do
  xvfb-run -a python3 code/pipeline/5_m3c2_parallel.py $loc --replace
done
```
Calculates normal surface change between sequential surveys. Computationally intensive.

**Output Structure:** Creates timestamped `pipeline_run_YYYYMMDD/` subdirectories within `results/<Location>/m3c2/`, each containing date-pair folders with M3C2 results.

**Verbosity:** CloudCompare output is suppressed. Displays clean single-line progress per survey pair:
```
[M3C2] 20170301_to_20170323... OK (123.45s)
[M3C2] 20170323_to_20170411... OK (98.32s)
```

**Important**: Requires display environment (use `xvfb-run -a` on headless systems).

### Step 6: DBSCAN Clustering
```bash
python3 code/pipeline/6_dbscan_parallel.py SanElijo --eps 0.35 --min_samples 30 --min_change 0.25
```
Filters M3C2 results for significant change, splits into Erosion/Deposition, clusters with DBSCAN, generates visualization reports.

**Automatic Input Detection:** Automatically finds and processes the most recent `pipeline_run_*` folder from Step 5.

**Critical Significance Filtering:**
- **Only processes points with significant M3C2 changes** (based on Level of Detection)
- Searches for significance field with multiple name variations: 'significant change', 'Significant change', 'significant_change', 'SignificantChange', etc.
- **Fails with clear error** if significance field is not found (prevents accidental processing of all points)
- Displays which significance field is being used: `[INFO] Using significance field: 'significant change'`

**Important:** Ensure M3C2 parameter files (`utilities/m3c2_params/*.txt`) are configured to output the significance field.

### Step 7: Spatial Gridding
```bash
python3 code/pipeline/7_make_grids.py SanElijo --resolution 10cm --replace
# Available: 10cm, 25cm, 1m
```
Aggregates clustered points into vertical bins within geospatial polygons using Geopandas spatial joins.

**Output Structure:** Creates resolution-specific subdirectories within each date folder:
- `results/<Location>/erosion/DATE1_to_DATE2/<resolution>/`
- `results/<Location>/deposition/DATE1_to_DATE2/<resolution>/`

For example: `results/DelMar/erosion/20250813_to_20250821/1m/20250813_to_20250821_ero_grid_1m.csv`

### Step 8: Grid Cleaning & Hole Filling
```bash
python3 code/pipeline/8_clean_fill_grids.py SanElijo --resolution 10cm --erosion --min_volume 2.0
```
Applies cliff-top cutoffs and fills occlusion holes using Alpha Shapes and interpolation to correct volume estimates.

## Location-Specific Configuration

### MOP Ranges
Defined in `code/pipeline/2_crop_files_parallel.py`:
- DelMar: 595-620
- Solana: 637-666
- Encinitas: 708-764
- SanElijo: 683-708
- Torrey: 567-581
- Blacks: 520-567

### Global Coordinate Shifts
CloudCompare requires coordinate shifts for some locations to handle large coordinate values. These are defined in scripts 5 (vegetation) and 6 (M3C2) in the `shift` dictionary. Check these when adding new locations.

## Utilities & Supporting Tools

### Streamlit Survey Browser
Interactive web app for filtering and browsing surveys:
```bash
cd code/streamlit
streamlit run survey_browser.py
```
Features: MOP range filtering, date range selection, instrument filtering, CSV export.

### Training Random Forest Models
```bash
python3 code/training/0_train_rf_model.py
```
Used to create beach removal classifiers stored in `utilities/beach_removal/`.

### Figure Generation
Scripts in `code/figure_making/` create publication-quality visualizations and dashboards. These follow manuscript aesthetics and use consistent color schemes.

Example:
```bash
python3 code/figure_making/sensitivity_grids.py --location DelMar
```

## Architecture Notes

### Parallel Processing
Most scripts use `multiprocessing` or `concurrent.futures.ProcessPoolExecutor` for parallel execution. Control with `--n_jobs` flag (default: 5).

Thread limits are set via environment variables in scripts:
```python
os.environ["OMP_NUM_THREADS"] = str(3)
os.environ["OPENBLAS_NUM_THREADS"] = str(3)
os.environ["MKL_NUM_THREADS"] = str(3)
```

### Point Cloud Handling
- LAS/LAZ reading: `laspy` library
- Spatial operations: `geopandas`, `shapely`
- PDAL for cropping operations
- CloudCompare CLI for M3C2 and CANUPO

### Reporting
All pipeline steps generate detailed CSV reports and PNG visualizations:
- `code/pipeline/reports/` - QC reports
- `utilities/beach_removal/classification_reports/` - Beach removal stats
- `utilities/dbscan/` - Clustering reports
- `validation/m3c2/` - Change detection validation

### Data Flow
Raw Survey → Cropped → (Audit QC, optional) → Beach Removed → Vegetation Removed → M3C2 (`pipeline_run_YYYYMMDD/`) → DBSCAN Clustered → Gridded → Cleaned/Filled

Each stage reads from previous stage output directory and writes to next stage directory within `results/<Location>/`.

**Pipeline Run Tracking:** Step 5 (M3C2) creates timestamped `pipeline_run_YYYYMMDD/` subdirectories. Step 6 (DBSCAN) automatically processes the most recent pipeline run, allowing for reprocessing with different parameters while preserving historical runs.

## Important Implementation Details

### Histogram Matching
Beach removal (step 3) includes histogram matching to normalize intensity values across different LiDAR instruments, enabling consistent Random Forest classification.

### Spatial Aggregation
Gridding (step 7) uses optimized Geopandas spatial joins (`sjoin`) with pandas aggregation for efficiency. Calculates:
- Median absolute M3C2 distance
- Mode of ClusterID
- RMS of pointwise distance uncertainty

### Grid Resolution Sensitivity
The pipeline supports multiple grid resolutions (10cm, 25cm, 1m) for sensitivity analysis. Resolution choice affects volume estimates and cluster detection.

## Testing & Validation

Run partial tests using `--test N` flag on most scripts to process only N files:
```bash
python3 code/pipeline/3_remove_beach_parallel.py SanElijo --test 10
```

Always check generated reports and visualizations after each pipeline stage to validate results before proceeding.
