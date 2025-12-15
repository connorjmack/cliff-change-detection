Here is the complete, professional `README.md` file in Markdown format, ready for copy-pasting.

````markdown
# Coastal Cliff LiDAR Processing Pipeline

## Overview
This repository contains a modular, parallelized Python pipeline designed to process terrestrial LiDAR surveys of coastal cliffs. The system ingests raw LAS point clouds, preprocesses them (cropping, quality control), removes non-relevant features (beach, vegetation), calculates change detection (M3C2), clusters significant erosion/deposition events, and aggregates data into spatiotemporal grids for analysis.

## Pipeline Architecture
The workflow operates sequentially, with each step transforming the data for the next. Most scripts support parallel processing (`multiprocessing` or `concurrent.futures`) to handle large datasets efficiently.

```mermaid
graph TD
    A[Raw Survey Data] -->|Step 0| B[Survey List CSV]
    B -->|Step 2| C[Cropped LAS]
    C -->|Step 3| D[QC & Outlier Removal]
    D -->|Step 4| E[Beach Removal - RF Model]
    E -->|Step 5| F[Vegetation Removal - CANUPO]
    F -->|Step 6| G[M3C2 Change Detection]
    G -->|Step 7| H[DBSCAN Clustering]
    H -->|Step 8| I[Spatial Gridding]
    I -->|Step 9| J[Cleaning & Hole Filling]
    J --> K[Final Analytical Data]
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

The pipeline relies on a specific directory structure to locate assets, models, and save results. Ensure your environment matches the following layout:

```text
/LidarProcessingCliffs/
├── code/
│   └── pipeline/           # Source code (this repo)
├── results/
│   └── <Location>/         # e.g., DelMar, SanElijo
│       ├── cropped/
│       ├── nobeach/
│       ├── noveg/
│       ├── m3c2/
│       ├── erosion/
│       └── deposition/
├── survey_lists/           # CSV inventories of surveys
└── utilities/
    ├── shape_files/        # Polygons for gridding
    ├── beach_removal/      # RF Models (.joblib) and Scalers
    ├── canupo/             # Vegetation classifiers (.prm)
    └── m3c2_params/        # Parameter files for CloudCompare
```

-----

## Workflow & Usage

### 0\. Ingest & Inventory

**Script:** `0_update_survey_lists.py`  
Scans storage volumes for new surveys based on date and MOP ranges, then updates the master CSV inventory. This drives the rest of the pipeline.

```bash
# Update specific location
python3 0_update_survey_lists.py --location SanElijo

# Update all locations
python3 0_update_survey_lists.py --all
```

### 1\. Cropping (Preprocessing)

**Script:** `2_crop_files_parallel.py`  
Crops raw LAS files to the specific study area (defined by MOP lines in KML format) using PDAL.

```bash
python3 2_crop_files_parallel.py --location SanElijo --replace
```

### 2\. Quality Control

**Script:** `3_qc_cropped_files.py`  
Generates point-count vs. file-size distribution plots to identify corrupt scans or failed crops. Can destructively remove bad files.

```bash
# Generate report only
python3 3_qc_cropped_files.py 

# Delete files below point threshold
python3 3_qc_cropped_files.py --delete_bad_files
```

### 3\. Classification (Beach & Vegetation)

**Script:** `4_remove_beach_parallel.py`  
Uses a Random Forest classifier (intensity/geometry) to remove beach points. Features histogram matching to normalize intensity across sensors.

```bash
python3 4_remove_beach_parallel.py SanElijo --n_jobs 5
```

**Script:** `5_remove_veg_parallel.py`  
Wraps CloudCompare's CANUPO plugin to classify and remove vegetation based on geometric scale.

```bash
python3 5_remove_veg_parallel.py SanElijo --cc "/path/to/CloudCompare"
```

### 4\. Change Detection (M3C2)

**Script:** `6_m3c2_parallel.py`  
Calculates normal surface change between sequential surveys. *Note: This step is computationally intensive and relies on CloudCompare*.

```bash
python3 6_m3c2_parallel.py SanElijo --cc "/path/to/CloudCompare"
```

### 5\. Clustering (DBSCAN)

**Script:** `7_dbscan_parallel.py`  
Filters M3C2 results for significant change, splits data into Erosion/Deposition, and clusters points using DBSCAN. Generates detailed visualization reports.

```bash
python3 7_dbscan_parallel.py SanElijo --eps 0.35 --min_samples 30 --min_change 0.25
```

### 6\. Gridding & Time Series Generation

**Script:** `8_make_grids.py`  
Aggregates clustered points into vertical bins within geospatial polygons. Uses Geopandas for optimized spatial joins.

```bash
# Available resolutions: 10cm, 25cm, 1m
python3 8_make_grids.py SanElijo --resolution 10cm --replace
```

### 7\. Post-Processing (Cleaning & Filling)

**Script:** `9_clean_fill_grids.py`  
Applies visual cliff-top cutoffs and fills occlusion holes in erosion clusters using Alpha Shapes and interpolation to correct volume estimates.

```bash
python3 9_clean_fill_grids.py SanElijo --resolution 10cm --erosion --min_volume 2.0
```

-----

## Configuration & Strategic Notes

### Hardcoded Parameters

Some location-specific parameters are defined in dictionaries within the scripts. Ensure these match your specific geography:

  * **MOP Ranges:** See `mop_ranges` dict in `2_crop_files_parallel.py`.
  * **Global Shifts:** See `shift` dict in `5_remove_veg_parallel.py` and `6_m3c2_parallel.py` to ensure proper coordinate handling in CloudCompare.

### Headless Environments

Scripts utilizing CloudCompare (`5_remove_veg` and `6_m3c2`) require a display environment. On headless Linux servers (e.g., HPC clusters), wrap execution with `xvfb` (X Virtual Framebuffer):

```bash
xvfb-run --auto-servernum python3 6_m3c2_parallel.py SanElijo
```

### Reporting

The pipeline automatically generates detailed logs, CSV inventories, and PNG visualizations in the following directories:

  * `code/pipeline/reports/` (QC)
  * `utilities/beach_removal/classification_reports/`
  * `utilities/dbscan/`
  * `validation/m3c2/`

## Authors

  * **LiDAR Processing Group**
  * *Last Updated:* December 2025

<!-- end list -->

```
```