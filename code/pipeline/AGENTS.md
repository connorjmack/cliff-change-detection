# Pipeline Step Guide

## Server Directory Structure
The pipeline expects and creates the following directory structure on the server:

```
/Volumes/group/LiDAR/  (macOS) or /project/group/LiDAR/  (Linux)
│
├── MiniRanger_Truck/LiDAR_Processed_Level2/
├── MiniRanger_ATV/LiDAR_Processed_Level2/
├── VMQLZ_Truck/LiDAR_Processed_Level2/
├── VMZ2000_Truck/LiDAR_Processed_Level2/
│   └── YYYYMMDD_MOP1_MOP2_*/
│       └── Beach_And_Backshore/
│           └── *_beach_cliff_ground.las     # Raw input files
│
├── MOPLines/
│   └── MOPs_SD_County.kml                   # MOP line definitions (EPSG:4326)
│
└── LidarProcessing/LidarProcessingCliffs/
    ├── survey_lists/
    │   └── surveys_<Location>.csv           # Survey inventories (created by step 0/1)
    │
    ├── utilities/
    │   ├── cropping_boxes/
    │   │   └── <Location>_cropbox.txt       # UTM polygon coords (auto-generated)
    │   ├── beach_removal/
    │   │   ├── <Location>_rf_model.joblib   # Random Forest models
    │   │   ├── <Location>_scaler.joblib     # StandardScaler for features
    │   │   └── classification_reports/      # Beach removal stats (step 4 output)
    │   ├── canupo/
    │   │   └── *.prm                        # CANUPO vegetation classifiers
    │   ├── m3c2_params/
    │   │   ├── new_params.txt               # Default M3C2 parameters
    │   │   └── m3c2_params_torrey.txt       # Torrey-specific parameters
    │   ├── shape_files/
    │   │   └── <Location>_<resolution>/     # Shapefiles for gridding (step 8)
    │   ├── cliff_top_cutoffs/
    │   │   └── <Location>_Visual_CliffTop_<resolution>.csv  # Step 9 cutoff data
    │   └── dbscan/                          # DBSCAN reports (step 7 output)
    │
    ├── validation/
    │   └── m3c2/                            # M3C2 reports (step 6 output)
    │
    ├── code/pipeline/
    │   └── reports/
    │       └── QC_Run_YYYYMMDD_HHMMSS/      # QC reports (step 3 output)
    ├── reports/
    │   └── daily/
    │       └── daily_report_YYYYMMDD.txt    # Daily update logs (step 1 output)
    │
    └── results/<Location>/                   # ALL processed data outputs
        ├── cropped/                         # Step 2 output
        │   └── *_cropped.las
        ├── nobeach/                         # Step 4 output
        │   └── *_nobeach.las
        ├── noveg/                           # Step 5 output
        │   └── *_noveg.las
        ├── m3c2/                            # Step 6 output
        │   └── pipeline_run_YYYYMMDD/
        │       └── DATE1_to_DATE2/
        │           ├── DATE1.las            # Reference cloud
        │           ├── DATE2.las            # Comparison cloud
        │           └── DATE1_to_DATE2_m3c2.las  # M3C2 distances + uncertainty
        ├── erosion/                         # Step 7 output
        │   └── DATE1_to_DATE2/
        │       ├── ero_clusters.las         # Clustered erosion points
        │       ├── ero_outliers.las         # Noise points (DBSCAN=-1)
        │       ├── clusters_<resolution>.csv     # Step 8 output: ClusterIDs grid
        │       ├── grid_<resolution>.csv         # Step 8 output: M3C2 distance grid
        │       ├── clusters_<resolution>_cleaned.csv  # Step 9 output
        │       ├── grid_<resolution>_cleaned.csv      # Step 9 output
        │       ├── clusters_<resolution>_filled.csv   # Step 9 output (erosion only)
        │       └── grid_<resolution>_filled.csv       # Step 9 output (erosion only)
        └── deposition/                      # Step 7 output
            └── DATE1_to_DATE2/
                ├── dep_clusters.las
                ├── dep_outliers.las
                ├── dep_clusters_<resolution>.csv
                ├── dep_grid_<resolution>.csv
                ├── dep_clusters_<resolution>_cleaned.csv
                └── dep_grid_<resolution>_cleaned.csv
```

This document describes each numbered pipeline step (0-9), its purpose, inputs, outputs, and example usage. Steps must be run in order because each stage reads from the previous stage's output.

## Prerequisites
- Python 3.8+ with dependencies installed (see repo root `environment.yml` or `requirements.txt`).
- PDAL installed system-wide for cropping (step 2).
- CloudCompare CLI available for CANUPO and M3C2 (steps 5-6).
- Expected shared storage layout (macOS `/Volumes/group/LiDAR`, Linux `/project/group/LiDAR`).

## Step 0: Create Survey Lists
**Script:** `0_make_survey_lists.py`

**Purpose:** Scan instrument directories and build initial CSV inventories per location.

**Inputs:** Raw survey folders under `*/LiDAR_Processed_Level2/` and MOP ranges defined in the script.

**Behavior:** Requires >= 2/3 MOP overlap with the target location, confirms `*_beach_cliff_ground.las` exists, normalizes paths to `/Volumes/group/LiDAR`, and sorts by date.

**Outputs:** `survey_lists/surveys_<Location>.csv`

**Example:**
```bash
python3 0_make_survey_lists.py --location SanElijo
```

## Step 1: Update Survey Lists
**Script:** `1_update_survey_lists.py`

**Purpose:** Append new surveys since the last CSV update; writes daily logs for scheduled runs.

**Inputs:** Existing `survey_lists/surveys_<Location>.csv` and raw survey folders.

**Behavior:** Accepts newer dates and same-date surveys with new folder paths, requires `*_beach_cliff_ground.las`, normalizes paths to `/Volumes/group/LiDAR`, and re-sorts by date.

**Outputs:** Updated CSV and `reports/daily/daily_report_YYYYMMDD.txt`.

**Example:**
```bash
python3 1_update_survey_lists.py --all
```

## Step 2: Crop Files to Study Area
**Script:** `2_crop_files_parallel.py`

**Purpose:** Crop raw LAS files to the study area polygon (derived from MOP lines) with PDAL.

**Inputs:** `survey_lists/surveys_<Location>.csv`, raw LAS `*_beach_cliff_ground.las`, MOP KML.

**Outputs:** `results/<Location>/cropped/*_cropped.las` and `utilities/cropping_boxes/<Location>_cropbox.txt`.

**Example:**
```bash
python3 2_crop_files_parallel.py --location SanElijo --replace
```

## Step 3: Quality Control
**Script:** `tests/audits/2_audit_cropping.py`

**Purpose:** Identify suspect cropped files based on point count vs file size; optional deletion.

**Inputs:** `results/<Location>/cropped/*.las`

**Outputs:** QC reports in `code/pipeline/reports/QC_Run_*/`.

**Example:**
```bash
python3 tests/audits/2_audit_cropping.py --delete_bad_files
```

## Step 4: Remove Beach Points
**Script:** `4_remove_beach_parallel.py`

**Purpose:** Apply Random Forest classifier to remove beach points (with histogram matching).

**Inputs:** Cropped LAS files and models in `utilities/beach_removal/`.

**Outputs:** `results/<Location>/nobeach/*_nobeach.las` and classification reports.

**Example:**
```bash
python3 4_remove_beach_parallel.py SanElijo --n_jobs 5
```

## Step 5: Remove Vegetation
**Script:** `5_remove_veg_parallel.py`

**Purpose:** Use CloudCompare CANUPO classifier to remove vegetation.

**Inputs:** `results/<Location>/nobeach/*.las` and `utilities/canupo/*.prm`.

**Outputs:** `results/<Location>/noveg/*_noveg.las`.

**Example:**
```bash
python3 5_remove_veg_parallel.py SanElijo --cc "/path/to/CloudCompare"
```

## Step 6: M3C2 Change Detection
**Script:** `6_m3c2_parallel.py`

**Purpose:** Compute M3C2 distances between sequential surveys with CloudCompare.

**Inputs:** `results/<Location>/noveg/*.las` and `utilities/m3c2_params/*.txt`.

**Outputs:** `results/<Location>/m3c2/pipeline_run_*/DATE1_to_DATE2/*_m3c2.las` and validation logs.

**Example (headless):**
```bash
xvfb-run --auto-servernum python3 6_m3c2_parallel.py SanElijo
```

## Step 7: DBSCAN Clustering
**Script:** `7_dbscan_parallel.py`

**Purpose:** Filter significant change, split erosion/deposition, cluster with DBSCAN, and report stats.

**Inputs:** Latest M3C2 run under `results/<Location>/m3c2/pipeline_run_*/`.

**Outputs:** `results/<Location>/erosion/DATE1_to_DATE2/` and `results/<Location>/deposition/DATE1_to_DATE2/`.

**Example:**
```bash
python3 7_dbscan_parallel.py SanElijo --eps 0.35 --min_samples 30 --min_change 0.25
```

## Step 8: Spatial Gridding
**Script:** `8_make_grids.py`

**Purpose:** Aggregate clustered points into polygon/elevation grids for time series analysis.

**Inputs:** Clustered LAS files and `utilities/shape_files/<Location>_<resolution>/` polygons.

**Outputs:** `grid_<resolution>.csv`, `clusters_<resolution>.csv`, and uncertainty grids per pair.

**Example:**
```bash
python3 8_make_grids.py SanElijo --resolution 10cm --replace
```

## Step 9: Grid Cleaning & Hole Filling
**Script:** `9_clean_fill_grids.py`

**Purpose:** Apply cliff-top cutoffs, filter small clusters, and fill erosion holes.

**Inputs:** Grids from step 8 and cutoff CSVs in `utilities/cliff_top_cutoffs/`.

**Outputs:** `*_cleaned.csv` and (erosion) `*_filled.csv` per pair; validation reports.

**Example:**
```bash
python3 9_clean_fill_grids.py SanElijo --resolution 10cm --erosion --min_volume 2.0
```

## Common Options
- `--test N`: Process only N items for quick validation (many steps).
- `--replace`: Overwrite existing outputs when re-running.
- `--n_jobs`: Parallel workers (steps 4, 7); check script defaults.
