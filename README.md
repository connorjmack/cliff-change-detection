````markdown
# Coastal Cliff LiDAR Processing Pipeline

## Overview
This repository contains a modular, parallelized Python pipeline designed to process terrestrial LiDAR surveys of coastal cliffs. The system ingests raw LAS point clouds, preprocesses them (cropping), optionally audits cropped outputs, removes non-relevant features (beach, vegetation), calculates change detection (M3C2), clusters significant erosion/deposition events, and aggregates data into spatiotemporal grids for analysis.

## Pipeline Architecture
The workflow operates sequentially, with each step transforming the data for the next. Most scripts support parallel processing (`multiprocessing` or `concurrent.futures`) to handle large datasets efficiently.

```mermaid
graph TD
    A[Raw Survey Data] -->|Step 0| B[Survey List CSV (Initial)]
    B -->|Step 1| C[Survey List CSV (Updated)]
    C -->|Step 2| D[Cropped LAS]
    D -->|Step 3| E[Beach Removal - RF Model]
    E -->|Step 4| F[Vegetation Removal - CANUPO]
    F -->|Step 5| G[M3C2 Change Detection]
    G -->|Step 6| H[DBSCAN Clustering]
    H -->|Step 7| I[Spatial Gridding]
    I -->|Step 8| J[Cleaning & Hole Filling]
    J --> K[Event Lists & 3D Cubes]
    K --> L[Filtered Significant Events]
````

## System Requirements

  * **OS:** Linux or macOS recommended (Path handling is optimized for `/project` or `/Volumes`). Windows is supported but may require path configuration adjustments.
  * **Python:** 3.8+
  * **External Software:**
      * **CloudCompare:** Required for M3C2 and CANUPO operations. Must be accessible via CLI.
      * **PDAL:** Required for cropping operations.

### Python Dependencies

Install the required libraries using pip:

```bash
pip install numpy pandas laspy[lazrs] pdal scikit-learn scipy shapely geopandas matplotlib seaborn tqdm alphashape joblib pyproj
```

-----

## Directory Structure

The pipeline operates on data stored on a shared server. Base paths are automatically detected by platform:
- **macOS:** `/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs/`
- **Linux/HPC:** `/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs/`

### Full Directory Layout

```text
/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs/
│
├── code/
│   └── pipeline/                    # Pipeline scripts (this repo)
│       └── daily_reports/           # Master daily run logs
│
├── survey_lists/
│   └── surveys_<Location>.csv       # Survey inventories (Step 0/1 output)
│
├── utilities/
│   ├── shape_files/
│   │   └── <Location>Polygons<MOP1>to<MOP2>at<resolution>/
│   │       └── *.shp                # Polygons for spatial gridding
│   ├── beach_removal/
│   │   ├── <Location>_rf_model.joblib
│   │   ├── <Location>_scaler.joblib
│   │   └── classification_reports/  # Step 3 output
│   ├── canupo/
│   │   └── *.prm                    # Vegetation classifiers
│   ├── m3c2_params/
│   │   ├── new_params.txt           # Default M3C2 parameters
│   │   └── m3c2_params_torrey.txt   # Location-specific params
│   ├── cliff_top_cutoffs/
│   │   └── <Location>_Visual_CliffTop_<resolution>.csv
│   ├── dbscan/                      # Step 6 reports
│   └── event_lists/                 # Event list generation scripts
│
├── validation/
│   ├── m3c2/                        # Step 5 reports
│   └── hole_filling/reports/        # Step 8 reports
│
└── results/
    ├── event_lists/                 # Generated event CSVs
    │   ├── erosion/
    │   │   ├── <Location>_events.csv
    │   │   └── <Location>_vol_<V>_elv_<E>.csv    # Filtered significant events
    │   ├── deposition/
    │   │   └── <Location>_events.csv
    │   └── combined/
    │       ├── <Location>_events.csv
    │       └── <Location>_vol_<V>_elv_<E>.csv    # Filtered significant events
    │
    ├── data_cubes/                  # 3D NPZ data cubes
    │   ├── <Location>_cube.npz                   # Full 3D data cube
    │   └── <Location>_vol_<V>_elv_<E>_cube.npz   # Filtered 3D data cube
    │
    └── <Location>/                  # Per-location results (e.g., DelMar, SanElijo)
        ├── cropped/                 # Step 2 output
        │   └── *_cropped.las
        │
        ├── nobeach/                 # Step 3 output
        │   └── *_nobeach.las
        │
        ├── noveg/                   # Step 4 output
        │   └── *_noveg.las
        │
        ├── m3c2/                    # Step 5 output
        │   └── pipeline_run_YYYYMMDD/
        │       └── DATE1_to_DATE2/
        │           ├── DATE1.las              # Reference cloud
        │           ├── DATE2.las              # Comparison cloud
        │           └── DATE1_to_DATE2_m3c2.las
        │
        ├── erosion/                 # Steps 6-8 output
        │   └── DATE1_to_DATE2/
        │       ├── ero_clusters.las           # Step 6: clustered points
        │       ├── ero_outliers.las           # Step 6: noise points
        │       ├── 10cm/                      # Step 7-8: resolution subdirs
        │       │   ├── DATE1_to_DATE2_ero_grid_10cm.csv
        │       │   ├── DATE1_to_DATE2_ero_clusters_10cm.csv
        │       │   ├── DATE1_to_DATE2_ero_stats_10cm.npz
        │       │   ├── DATE1_to_DATE2_ero_grid_10cm_filled.csv
        │       │   └── DATE1_to_DATE2_ero_clusters_10cm_filled.csv
        │       ├── 25cm/
        │       │   ├── DATE1_to_DATE2_ero_grid_25cm.csv
        │       │   ├── DATE1_to_DATE2_ero_clusters_25cm.csv
        │       │   ├── DATE1_to_DATE2_ero_stats_25cm.npz
        │       │   ├── DATE1_to_DATE2_ero_grid_25cm_filled.csv
        │       │   └── DATE1_to_DATE2_ero_clusters_25cm_filled.csv
        │       └── 1m/
        │           └── ... (same pattern)
        │
        └── deposition/              # Steps 6-8 output (same structure as erosion)
            └── DATE1_to_DATE2/
                ├── dep_clusters.las
                ├── dep_outliers.las
                ├── 10cm/
                ├── 25cm/
                └── 1m/
```

### Locations

The pipeline supports the following study sites with their MOP (Monitoring and Prediction) line ranges:

| Location  | MOP Range |
|-----------|-----------|
| DelMar    | 595-620   |
| Solana    | 637-666   |
| SanElijo  | 683-708   |
| Encinitas | 708-764   |
| Torrey    | 567-581   |
| Blacks    | 520-567   |

-----

## Workflow & Usage

### 0\. Create Survey Lists (Initial)

**Script:** `0_make_survey_lists.py`  
Scans storage volumes for surveys based on date and MOP ranges, then writes the initial CSV inventory.

```bash
# Create survey list for a specific location
python3 0_make_survey_lists.py --location SanElijo

# Create survey lists for all locations
python3 0_make_survey_lists.py --all
```

### 1\. Update Survey Lists (Daily)

**Script:** `1_update_survey_lists.py`  
Checks for new surveys and updates the master CSV inventory. This drives the rest of the pipeline and writes a daily log to `reports/daily/`.

```bash
# Update specific location
python3 1_update_survey_lists.py --location SanElijo

# Update all locations
python3 1_update_survey_lists.py --all
```

### Daily Orchestrator (Optional)

**Script:** `run_daily.py`  
Runs the detection step (Step 1) and then steps 2-8 for each location with a master log.

```bash
# Run detection then process locations with new data
python3 run_daily.py

# Skip detection and force all locations
python3 run_daily.py --force-all
```

On Linux, CloudCompare steps are automatically wrapped with `xvfb-run`.

### 2\. Cropping (Preprocessing)

**Script:** `2_crop_files_parallel.py`  
Crops raw LAS files to the specific study area (defined by MOP lines in KML format) using PDAL.

```bash
python3 2_crop_files_parallel.py --location SanElijo --replace
```

### Audit: Cropped Files QC (Optional)

**Script:** `tests/audits/2_audit_cropping.py`  
Generates point-count vs. file-size distribution plots to identify corrupt scans or failed crops. Can destructively remove bad files.

```bash
# Generate report only
python3 tests/audits/2_audit_cropping.py 

# Delete files below point threshold
python3 tests/audits/2_audit_cropping.py --delete_bad_files
```

### 3\. Beach Removal

**Script:** `3_remove_beach_parallel.py`  
Uses a Random Forest classifier (intensity/geometry) to remove beach points. Features histogram matching to normalize intensity across sensors.

```bash
python3 3_remove_beach_parallel.py SanElijo --n_jobs 5
```

### 4\. Vegetation Removal

**Script:** `4_remove_veg_parallel.py`  
Wraps CloudCompare's CANUPO plugin to classify and remove vegetation based on geometric scale.

```bash
python3 4_remove_veg_parallel.py SanElijo --cc "/path/to/CloudCompare"
```

### 5\. Change Detection (M3C2)

**Script:** `5_m3c2_parallel.py`
Calculates normal surface change between sequential surveys. *Note: This step is computationally intensive and relies on CloudCompare*.

**Output Format:** Creates timestamped `pipeline_run_YYYYMMDD/` subdirectories containing date-pair folders with M3C2 results. CloudCompare output is suppressed for clean single-line progress tracking.

```bash
# Basic usage
python3 5_m3c2_parallel.py SanElijo --cc "/path/to/CloudCompare"

# Replace existing outputs
python3 5_m3c2_parallel.py SanElijo --replace

# On headless servers (required for CloudCompare)
xvfb-run -a python3 5_m3c2_parallel.py SanElijo --replace

# Process all locations with xvfb
for loc in DelMar Solana Encinitas SanElijo Torrey Blacks; do
  xvfb-run -a python3 5_m3c2_parallel.py $loc --replace
done
```

### 6\. Clustering (DBSCAN)

**Script:** `6_dbscan_parallel.py`
Filters M3C2 results for significant change, splits data into Erosion/Deposition, and clusters points using DBSCAN. Generates detailed visualization reports.

**Critical Feature:** Automatically finds the most recent `pipeline_run_*` folder from Step 5. **Only processes points with significant M3C2 changes** - the script will fail with a clear error if the M3C2 significance field is not found, preventing accidental processing of all points.

**Important:** M3C2 parameter files must be configured to output the 'significant change' field.

```bash
python3 6_dbscan_parallel.py SanElijo --eps 0.35 --min_samples 30 --min_change 0.25
```

### 7\. Gridding & Time Series Generation

**Script:** `7_make_grids.py`  
Aggregates clustered points into vertical bins within geospatial polygons. Uses Geopandas for optimized spatial joins.

```bash
# Available resolutions: 10cm, 25cm, 1m
python3 7_make_grids.py SanElijo --resolution 10cm --replace
```

### 8\. Post-Processing (Cleaning & Filling)

**Script:** `8_clean_fill_grids.py`  
Applies visual cliff-top cutoffs and fills occlusion holes in erosion clusters using Alpha Shapes and interpolation to correct volume estimates.

```bash
python3 8_clean_fill_grids.py SanElijo --resolution 10cm --erosion --min_volume 2.0
```

### Event List Generation

**Scripts:** `utilities/event_lists/make_event_lists.py`, `utilities/event_lists/make_sig_event_lists.py`

These utilities generate event-level summaries from the 25cm filled grid outputs.

#### Generate Event CSVs

Extracts individual cluster events with volume, elevation, and spatial extent metrics:

```bash
# Generate event lists for a single location
python3 utilities/event_lists/make_event_lists.py SanElijo

# Generate for all locations
python3 utilities/event_lists/make_event_lists.py --all

# Erosion only
python3 utilities/event_lists/make_event_lists.py SanElijo --erosion
```

**Output columns:** `mid_date`, `start_date`, `end_date`, `volume`, `elevation`, `alongshore_centroid_m`, `alongshore_start_m`, `alongshore_end_m`, `width`, `height`, `vol_unc`, `month`

#### Generate 3D Data Cubes (NPZ)

Creates 3D numpy arrays (alongshore × elevation × time) from the filled grids:

```bash
# Generate NPZ cube for a location
python3 utilities/event_lists/make_event_lists.py SanElijo --make-npz

# Generate for all locations
python3 utilities/event_lists/make_event_lists.py --all --make-npz
```

**Output:** `results/data_cubes/<Location>_cube.npz` containing:
- `erosion`: 3D array of M3C2 values (alongshore, elevation, time)
- `deposition`: 3D array of M3C2 values
- `alongshore_m`: 1D array of alongshore positions (meters)
- `elevation_m`: 1D array of elevation bin centers (meters)
- `dates`: 1D array of mid-dates (ordinal integers, use `datetime.fromordinal()`)
- `date_strings`: 1D array of date folder names (YYYYMMDD_to_YYYYMMDD)

#### Filter Significant Events

Filters event lists to keep only significant erosion events meeting volume and elevation thresholds:

```bash
# Filter with default thresholds (volume > 5 m³, elevation > 5 m)
python3 utilities/event_lists/make_sig_event_lists.py

# Custom thresholds
python3 utilities/event_lists/make_sig_event_lists.py --min_volume 10 --min_elevation 3
```

**Output:** `<Location>_vol_<V>_elv_<E>.csv` (e.g., `SanElijo_vol_5_elv_5.csv`)

#### Generate Filtered 3D Data Cubes

Creates filtered NPZ cubes containing only cells belonging to clusters that meet the significance criteria:

```bash
# Generate filtered cube with default thresholds
python3 utilities/event_lists/make_sig_event_lists.py --make-npz

# Custom thresholds
python3 utilities/event_lists/make_sig_event_lists.py --make-npz --min_volume 10 --min_elevation 3
```

**Output:** `results/data_cubes/<Location>_vol_<V>_elv_<E>_cube.npz` containing the same structure as the full cube, plus:
- `min_volume`: The volume threshold used for filtering
- `min_elevation`: The elevation threshold used for filtering

-----

## Configuration & Strategic Notes

### Hardcoded Parameters

Some location-specific parameters are defined in dictionaries within the scripts. Ensure these match your specific geography:

  * **MOP Ranges:** See `mop_ranges` dict in `2_crop_files_parallel.py`.
  * **Global Shifts:** See `shift` dict in `4_remove_veg_parallel.py` and `5_m3c2_parallel.py` to ensure proper coordinate handling in CloudCompare.

### Headless Environments

Scripts utilizing CloudCompare (`4_remove_veg` and `5_m3c2`) require a display environment. On headless Linux servers (e.g., HPC clusters), wrap execution with `xvfb` (X Virtual Framebuffer):

```bash
# Use -a (auto-servernum) for automatic display assignment
xvfb-run -a python3 5_m3c2_parallel.py SanElijo --replace

# Or use the explicit form with server arguments
xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" \
  python3 5_m3c2_parallel.py SanElijo --replace
```

**Note:** Step 5 (M3C2) now suppresses verbose CloudCompare output, showing only clean single-line progress per survey pair.

### Reporting

The pipeline automatically generates detailed logs, CSV inventories, and PNG visualizations in the following directories:

  * `reports/daily/`
  * `code/pipeline/daily_reports/`
  * `code/pipeline/reports/` (QC)
  * `utilities/beach_removal/classification_reports/`
  * `utilities/dbscan/`
  * `validation/hole_filling/reports/`
  * `validation/m3c2/`

## Testing

Tests live in `tests/pytest/` with audit scripts in `tests/audits/`.

```bash
pytest tests/pytest
pytest tests/pytest/test_2_crop_files_parallel.py
```

## Authors

  * **LiDAR Processing Group**
  * *Last Updated:* December 2025

<!-- end list -->

```
```
