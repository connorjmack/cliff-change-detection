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
├── .github/workflows/       # CI/CD (claude.yml, claude-code-review.yml)
├── code/
│   ├── pipeline/            # Main processing scripts (0-8 sequential)
│   │   └── run_daily.py     # Daily orchestrator script
│   ├── figure_making/       # Publication visualization scripts
│   ├── streamlit/           # Survey browser web app
│   └── training/            # ML model training scripts
├── figures/                 # Generated output figures
├── reports/
│   └── daily/               # Daily update logs from step 1
├── results/<Location>/      # Output directories (e.g., DelMar, SanElijo)
│   ├── cropped/
│   ├── nobeach/
│   ├── noveg/
│   ├── m3c2/
│   │   └── pipeline_run_YYYYMMDD/  # Timestamped run folders
│   ├── erosion/
│   │   └── DATE1_to_DATE2/
│   │       ├── ero_clusters.las
│   │       ├── ero_outliers.las
│   │       ├── 10cm/                 # Resolution-specific subdirectory
│   │       │   ├── DATE1_to_DATE2_ero_grid_10cm.csv
│   │       │   ├── DATE1_to_DATE2_ero_clusters_10cm.csv
│   │       │   ├── DATE1_to_DATE2_ero_stats_10cm.npz
│   │       │   └── DATE1_to_DATE2_ero_grid_10cm_filled.csv  # Step 8 output
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
├── tests/
│   ├── pytest/              # Unit tests for all pipeline steps
│   └── audits/              # QC scripts (cropping audit, survey audit)
├── utilities/
│   ├── shape_files/         # Polygons for gridding
│   ├── beach_removal/       # RF Models (.joblib) and Scalers
│   │   └── classification_reports/  # Beach removal stats
│   ├── canupo/              # Vegetation classifiers (.prm)
│   ├── m3c2_params/         # CloudCompare parameter files
│   ├── cropping_boxes/      # Auto-generated crop polygons (UTM)
│   ├── cliff_top_cutoffs/   # Visual cutoff CSVs per location
│   └── dbscan/              # DBSCAN clustering reports
└── validation/
    ├── m3c2/                # M3C2 reports (step 5 output)
    └── hole_filling/        # Step 8 validation reports

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
# or update all locations (default if no --location specified)
python3 code/pipeline/1_update_survey_lists.py --all
```
Generates daily report at `reports/daily/daily_report_YYYYMMDD.txt`.

### Step 2: Crop to Study Area
```bash
python3 code/pipeline/2_crop_files_parallel.py --location SanElijo --replace
# or process all locations
python3 code/pipeline/2_crop_files_parallel.py --all
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
# Process all locations
python3 code/pipeline/8_clean_fill_grids.py --all --resolution 25cm

# Process single location (erosion only)
python3 code/pipeline/8_clean_fill_grids.py SanElijo --resolution 10cm --erosion --min_volume 2.0

# Deposition only (cleaning only, no filling)
python3 code/pipeline/8_clean_fill_grids.py SanElijo --resolution 25cm --deposition

# Skip hole filling (cleaning only)
python3 code/pipeline/8_clean_fill_grids.py SanElijo --resolution 10cm --erosion --skip_filling

# Testing mode (dry-run, no file writes)
python3 code/pipeline/8_clean_fill_grids.py SanElijo --resolution 10cm --testing
```
Applies cliff-top cutoffs and fills occlusion holes using Alpha Shapes and interpolation to correct volume estimates.

**Parameters:**
- `--resolution`: Grid resolution (10cm, 25cm, 1m; default: 10cm)
- `--erosion`/`--deposition`: Process specific type (default: both)
- `--min_volume`: Minimum cluster volume for hole filling (default: 2.0 m³)
- `--threshold`: Minimum cells for cluster retention (resolution-dependent defaults)
- `--cleanup_size`: Morphological cleanup kernel size (default: 3)
- `--skip_filling`: Skip hole filling (cleaning only)
- `--testing`: Dry-run mode without file writes
- `--replace`: Overwrite existing outputs

**Output:** Saves only final `_filled.csv` files (no intermediate `_cleaned.csv`). For erosion: includes cleaning + hole filling + morphological cleanup. For deposition: includes cleaning only.

### Automated Daily Pipeline
```bash
python3 code/pipeline/run_daily.py
python3 code/pipeline/run_daily.py --force-all  # Force reprocessing
```
Orchestrates the full pipeline (steps 2-8) for all locations. On Linux, automatically wraps CloudCompare steps with `xvfb-run`. Runs step 8 with `--resolution 25cm --erosion --deposition`.

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
CloudCompare requires coordinate shifts for some locations to handle large coordinate values. These are defined in scripts 4 (vegetation) and 5 (M3C2) in the `shift` dictionary:
```python
shift = {
    "SanElijo":   ("-473000", "-3653000", "0"),
    "Encinitas":  ("-472000", "-3655000", "0"),
    "Solana":     ("-475000", "-3650000", "0"),
    "Torrey":     ("-475000", "-3650000", "0"),
    # DelMar and Blacks may not need shifts (verify in scripts)
}
```
**Important:** These shifts must be consistent across steps 4 and 5. Check when adding new locations.

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
- `reports/daily/` - Daily update logs (step 1)
- `results/<Location>/pipeline_reports/` - Cropping reports (step 2)
- `code/pipeline/reports/QC_Run_*/` - Cropping audit QC reports
- `utilities/beach_removal/classification_reports/` - Beach removal stats (step 3)
- `validation/m3c2/` - M3C2 inventory reports (step 5)
- `utilities/dbscan/` - DBSCAN clustering reports (step 6)
- `validation/hole_filling/reports/<Location>/` - Hole filling reports (step 8)

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

### Unit Tests (pytest)
Comprehensive test suite in `tests/pytest/`:
```bash
# Run all tests
pytest tests/pytest/

# Run tests for a specific step
pytest tests/pytest/test_7_dbscan_parallel.py -v
```

### Partial Pipeline Runs
Run partial tests using `--test N` flag on most scripts to process only N files:
```bash
python3 code/pipeline/3_remove_beach_parallel.py SanElijo --test 10
python3 code/pipeline/6_dbscan_parallel.py SanElijo --test 3
```

### Audit Scripts
Additional QC scripts in `tests/audits/`:
- `2_audit_cropping.py` - QC cropped files (point count vs size analysis)
- `audit_survey_list.py` - Validate survey inventory CSVs
- `status_update.py` - Pipeline progress tracking

Always check generated reports and visualizations after each pipeline stage to validate results before proceeding.
