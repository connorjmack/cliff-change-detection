# Pipeline Documentation

This document provides detailed technical documentation for each pipeline step, including input/output file structures, data formats, and server directory organization.

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
    │   │   └── classification_reports/      # Beach removal stats (step 3 output)
    │   ├── canupo/
    │   │   └── *.prm                        # CANUPO vegetation classifiers
    │   ├── m3c2_params/
    │   │   ├── new_params.txt               # Default M3C2 parameters
    │   │   └── m3c2_params_torrey.txt       # Torrey-specific parameters
    │   ├── shape_files/
    │   │   └── <Location>_<resolution>/     # Shapefiles for gridding (step 7)
    │   ├── cliff_top_cutoffs/
    │   │   └── <Location>_Visual_CliffTop_<resolution>.csv  # Step 8 cutoff data
    │   └── dbscan/                          # DBSCAN reports (step 6 output)
    │
    ├── validation/
    │   └── m3c2/                            # M3C2 reports (step 5 output)
    │
    ├── code/pipeline/
    │   ├── reports/
    │   │   └── QC_Run_YYYYMMDD_HHMMSS/      # Cropping QC reports (audit output)
    │   └── daily_reports/
    │       └── daily_report_YYYYMMDD.txt    # Daily update logs (step 1 output)
    │
    └── results/<Location>/                   # ALL processed data outputs
        ├── cropped/                         # Step 2 output
        │   └── *_cropped.las
        ├── nobeach/                         # Step 3 output
        │   └── *_nobeach.las
        ├── noveg/                           # Step 4 output
        │   └── *_noveg.las
        ├── m3c2/                            # Step 5 output
        │   └── pipeline_run_YYYYMMDD/
        │       └── DATE1_to_DATE2/
        │           ├── DATE1.las            # Reference cloud
        │           ├── DATE2.las            # Comparison cloud
        │           └── DATE1_to_DATE2_m3c2.las  # M3C2 distances + uncertainty
        ├── erosion/                         # Step 6 output
        │   └── DATE1_to_DATE2/
        │       ├── ero_clusters.las         # Clustered erosion points
        │       ├── ero_outliers.las         # Noise points (DBSCAN=-1)
        │       ├── 10cm/                        # Step 7 resolution subdirectory
        │       │   ├── DATE1_to_DATE2_ero_grid_10cm.csv              # Step 7 output
        │       │   ├── DATE1_to_DATE2_ero_clusters_10cm.csv          # Step 7 output
        │       │   ├── DATE1_to_DATE2_ero_stats_10cm.npz             # Step 7 output
        │       │   ├── DATE1_to_DATE2_ero_grid_10cm_filled.csv       # Step 8 output (final)
        │       │   └── DATE1_to_DATE2_ero_clusters_10cm_filled.csv   # Step 8 output (final)
        │       ├── 25cm/                        # Alternative resolution
        │       │   ├── DATE1_to_DATE2_ero_grid_25cm.csv
        │       │   ├── DATE1_to_DATE2_ero_clusters_25cm.csv
        │       │   ├── DATE1_to_DATE2_ero_stats_25cm.npz
        │       │   ├── DATE1_to_DATE2_ero_grid_25cm_filled.csv
        │       │   └── DATE1_to_DATE2_ero_clusters_25cm_filled.csv
        │       └── 1m/                          # Alternative resolution
        │           ├── DATE1_to_DATE2_ero_grid_1m.csv
        │           ├── DATE1_to_DATE2_ero_clusters_1m.csv
        │           ├── DATE1_to_DATE2_ero_stats_1m.npz
        │           ├── DATE1_to_DATE2_ero_grid_1m_filled.csv
        │           └── DATE1_to_DATE2_ero_clusters_1m_filled.csv
        └── deposition/                      # Step 6 output
            └── DATE1_to_DATE2/
                ├── dep_clusters.las
                ├── dep_outliers.las
                ├── 10cm/                        # Step 7 resolution subdirectory
                │   ├── DATE1_to_DATE2_dep_grid_10cm.csv              # Step 7 output
                │   ├── DATE1_to_DATE2_dep_clusters_10cm.csv          # Step 7 output
                │   ├── DATE1_to_DATE2_dep_stats_10cm.npz             # Step 7 output
                │   ├── DATE1_to_DATE2_dep_grid_10cm_filled.csv       # Step 8 output (final)
                │   └── DATE1_to_DATE2_dep_clusters_10cm_filled.csv   # Step 8 output (final)
                ├── 25cm/                        # Alternative resolution
                │   ├── DATE1_to_DATE2_dep_grid_25cm.csv
                │   ├── DATE1_to_DATE2_dep_clusters_25cm.csv
                │   ├── DATE1_to_DATE2_dep_stats_25cm.npz
                │   ├── DATE1_to_DATE2_dep_grid_25cm_filled.csv
                │   └── DATE1_to_DATE2_dep_clusters_25cm_filled.csv
                └── 1m/                          # Alternative resolution
                    ├── DATE1_to_DATE2_dep_grid_1m.csv
                    ├── DATE1_to_DATE2_dep_clusters_1m.csv
                    ├── DATE1_to_DATE2_dep_stats_1m.npz
                    ├── DATE1_to_DATE2_dep_grid_1m_filled.csv
                    └── DATE1_to_DATE2_dep_clusters_1m_filled.csv
```

---

## Step 0: Create Survey Lists

**Script:** `0_make_survey_lists.py`

### Purpose
Creates initial CSV inventories by scanning instrument directories for surveys that overlap with location MOP ranges.

### Usage
```bash
python3 0_make_survey_lists.py --location SanElijo
python3 0_make_survey_lists.py --all
```

### Output Files
**Location:** `survey_lists/surveys_<Location>.csv`

**Format:**
```csv
path,date,MOP1,MOP2,beach,method
20171004_00590_00708_...,20171004,590,708,SanElijo,MiniRanger_Truck
```

**Columns:**
- `path`: Survey folder name (not full path)
- `date`: Integer YYYYMMDD format
- `MOP1, MOP2`: MOP line range covered by survey
- `beach`: Location name
- `method`: Instrument type (MiniRanger_Truck, MiniRanger_ATV, VMQLZ_Truck, VMZ2000_Truck)

### Overlap Logic
Surveys are included if they have ≥2/3 overlap with the location's MOP range:
```python
overlap = min(survey_MOP2, location_max) - max(survey_MOP1, location_min)
required_overlap = floor((location_max - location_min) * 2/3)
```

### Location MOP Ranges
```python
mop_ranges = {
    "DelMar":    (595, 620),
    "Solana":    (637, 666),
    "Encinitas": (708, 764),
    "SanElijo":  (683, 708),
    "Torrey":    (567, 581),
    "Blacks":    (520, 567)
}
```

---

## Step 1: Update Survey Lists

**Script:** `1_update_survey_lists.py`

### Purpose
Incrementally updates survey lists by scanning for NEW surveys (date > max existing date) and validating they contain `*_beach_cliff_ground.las` files.

### Usage
```bash
python3 1_update_survey_lists.py --location SanElijo
python3 1_update_survey_lists.py --all
```

### Output Files
1. **Updates:** `survey_lists/surveys_<Location>.csv` (appends and re-sorts)
2. **Daily Log:** `code/pipeline/daily_reports/daily_report_YYYYMMDD.txt`

**Daily Report Format:**
```
=== DAILY REPORT: 2024-01-15 ===

NEW: SanElijo | 20240115 | /Volumes/group/LiDAR/.../file.las
No new surveys found (Run: 14:32:10). CSVs are up to date.
```

### Key Differences from Step 0
- Only scans surveys with `date > max(existing_csv_dates)`
- Validates presence of `*_beach_cliff_ground.las` before adding
- Forces all paths to use `/Volumes/` format (macOS convention)
- Appends to existing CSV rather than recreating

---

## Step 2: Crop Files to Study Area

**Script:** `2_crop_files_parallel.py`

### Purpose
Crops raw LAS files to MOP line boundaries using PDAL. Creates UTM polygon cropboxes from KML on first run.

### Usage
```bash
python3 2_crop_files_parallel.py --location SanElijo
python3 2_crop_files_parallel.py --location SanElijo --replace
```

### Input
- **Survey list:** `survey_lists/surveys_<Location>.csv`
- **MOP KML:** `/Volumes/group/LiDAR/MOPLines/MOPs_SD_County.kml` (EPSG:4326)
- **Raw LAS files:** `<instrument>/LiDAR_Processed_Level2/<survey>/Beach_And_Backshore/*_beach_cliff_ground.las`

### Output Files
1. **Cropped LAS:** `results/<Location>/cropped/*_beach_cliff_ground_cropped.las`
2. **Cropbox polygon:** `utilities/cropping_boxes/<Location>_cropbox.txt` (auto-generated once)
3. **Report:** `results/<Location>/pipeline_reports/cropping_report_YYYYMMDD_HHMMSS.txt`

### Cropbox File Format
```
473526.123456 3650234.567890 0.0
473628.234567 3650345.678901 0.0
...
```
- Space-separated X Y Z coordinates in UTM Zone 11N (EPSG:32611)
- Auto-generated by transforming KML MOP lines from WGS84 to UTM
- Extended 500m along-track (buffer) beyond min/max MOP lines

### PDAL Pipeline
```json
{
  "pipeline": [
    {"type": "readers.las", "filename": "input.las"},
    {"type": "filters.crop", "polygon": "POLYGON((x1 y1, x2 y2, ...))"},
    {"type": "filters.sample", "radius": 0.05},  // 5cm sampling
    {"type": "writers.las", "filename": "output_cropped.las"}
  ]
}
```

### Report Format
```
=== CROPPING PIPELINE REPORT: SanElijo ===
Date: 2024-01-15 14:30:00
System: hostname (Darwin)
----------------------------------------
Total Wall Time:   245.32 seconds (4.09 minutes)
Files Processed:   42
Files Skipped:     0
Errors:            0
Avg Time per File: 5.84 seconds
----------------------------------------
DETAILS:
[OK] file1.las | 5.23s | 1234567 pts
[OK] file2.las | 6.01s | 1456789 pts
```

---

## Audit: Cropped Files QC (Optional)

**Script:** `tests/audits/2_audit_cropping.py`

### Purpose
Analyzes cropped files to identify corrupt scans or failed crops based on point count vs file size distributions. Optionally deletes bad files.

### Usage
```bash
python3 tests/audits/2_audit_cropping.py                    # Report only
python3 tests/audits/2_audit_cropping.py --delete_bad_files # Delete files < threshold
```

### Input
- **Cropped files:** `results/*/cropped/*.las`

### Output Files
**Location:** `code/pipeline/reports/QC_Run_YYYYMMDD_HHMMSS/`

1. **full_cropped_file_inventory.csv**
   ```csv
   Location,Filename,Size_MB,Point_Count,Path,Status
   SanElijo,file1.las,45.2,1234567,/full/path.las,OK
   SanElijo,file2.las,0.3,234,/full/path.las,SUSPECT
   ```

2. **suspect_files.csv** (if any found)
   - Subset of files with `Point_Count < 1000`

3. **run_summary.txt**
   ```
   QC Run: 20240115_143000
   Delete Mode Enabled: True
   Total Files Scanned: 42
   ------------------------------
   Suspect Files Found: 2
   Files Deleted: 2
   Deletion Errors: 0
   ```

4. **QC_Points_vs_Size.png** (scatter plot)
5. **QC_FileSize_Distribution.png** (boxplot)

### Quality Threshold
```python
MIN_POINT_THRESHOLD = 1000  # Files below this are flagged as SUSPECT
```

---

## Step 3: Remove Beach Points

**Script:** `3_remove_beach_parallel.py`

### Purpose
Uses pre-trained Random Forest classifier to remove beach points based on intensity and geometric features. Includes histogram matching to normalize intensity across sensors.

### Usage
```bash
python3 3_remove_beach_parallel.py SanElijo
python3 3_remove_beach_parallel.py SanElijo --n_jobs 5 --replace
```

### Input
- **Cropped files:** `results/<Location>/cropped/*_cropped.las`
- **RF Model:** `utilities/beach_removal/<Location>_rf_model.joblib`
- **Scaler:** `utilities/beach_removal/<Location>_scaler.joblib`

### Output Files
1. **Classified LAS:** `results/<Location>/nobeach/*_nobeach.las`
   - Same structure as input but with beach points removed
   - Classification labels NOT stored in output (points are simply excluded)

2. **Classification Report:** `utilities/beach_removal/classification_reports/<Location>_classification_report_YYYYMMDD_HHMMSS.csv`
   ```csv
   filename,status,input_points,output_points,removed_points,percent_removed,processing_time_sec,error_message
   file1.las,Success,1234567,856234,378333,30.65,12.3,
   ```

### Feature Engineering
The RF model uses these features (computed from point cloud):
- Intensity (after histogram matching)
- Z elevation
- Local geometric features (computed per-point)

### Histogram Matching
Normalizes intensity distributions across different instruments:
```python
def match_histograms(source, reference):
    # Warps source intensity to match reference distribution
    # Uses CDF interpolation
```

---

## Step 4: Remove Vegetation

**Script:** `4_remove_veg_parallel.py`

### Purpose
Wraps CloudCompare's CANUPO plugin to classify and remove vegetation based on multi-scale geometric analysis.

### Usage
```bash
# Requires display environment
xvfb-run --auto-servernum python3 4_remove_veg_parallel.py SanElijo

# With explicit CloudCompare path
python3 4_remove_veg_parallel.py SanElijo --cc "/path/to/CloudCompare"
```

### Input
- **Classified files:** `results/<Location>/nobeach/*_nobeach.las`
- **CANUPO classifier:** `utilities/canupo/*.prm`

### Output Files
1. **Vegetation-free LAS:** `results/<Location>/noveg/*_noveg.las`
   - Points classified as vegetation are removed
   - Remaining points represent cliff face

2. **Classification Report:** (written to stdout during execution)

### CloudCompare Command Structure
```bash
CloudCompare -silent -auto_save off \
  -o -global_shift X Y Z input.las \
  -canupo_classify classifier.prm \
  -c_export_fmt las \
  -save_clouds FILE output_noveg.las
```

### Global Coordinate Shifts
Required for CloudCompare to handle large UTM coordinates:
```python
shift = {
    "SanElijo":   ("-473000", "-3653000", "0"),
    "Encinitas":  ("-472000", "-3655000", "0"),
    "Solana":     ("-475000", "-3650000", "0"),
    "Torrey":     ("-475000", "-3650000", "0")
}
```

**IMPORTANT:** These shifts must be consistent across steps 4 and 5 for coordinate alignment.

---

## Step 5: M3C2 Change Detection

**Script:** `5_m3c2_parallel.py`

### Purpose
Computes Multi-scale Model-to-Model Cloud Comparison (M3C2) normal distances between sequential surveys using CloudCompare.

### Usage
```bash
# Basic usage (requires display environment)
xvfb-run -a python3 5_m3c2_parallel.py SanElijo

# With replace flag to overwrite existing outputs
xvfb-run -a python3 5_m3c2_parallel.py SanElijo --replace

# Single-threaded mode
python3 5_m3c2_parallel.py SanElijo --single

# Process all locations
for loc in DelMar Solana Encinitas SanElijo Torrey Blacks; do
  xvfb-run -a python3 5_m3c2_parallel.py $loc --replace
done
```

### Verbosity Control
CloudCompare output is suppressed (`stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL`). Clean single-line progress output:
```
[M3C2] 20170301_to_20170323... OK (123.45s)
[M3C2] 20170323_to_20170411... OK (98.32s)
[M3C2] 20170411_to_20170501... SKIPPED (already exists)
```

### Input
- **Vegetation-free files:** `results/<Location>/noveg/*.las` (sorted chronologically)
- **M3C2 parameters:** `utilities/m3c2_params/new_params.txt` (or `m3c2_params_torrey.txt` for Torrey)

### Output Directory Structure
```
results/<Location>/m3c2/pipeline_run_YYYYMMDD/
└── DATE1_to_DATE2/
    ├── DATE1.las                    # Reference cloud (earlier date)
    ├── DATE2.las                    # Comparison cloud (later date)
    └── DATE1_to_DATE2_m3c2.las      # M3C2 results
```

### M3C2 Output LAS Fields
The `*_m3c2.las` file contains these scalar fields:
- **M3C2 distance (m):** Normal distance (positive = deposition, negative = erosion)
- **Distance uncertainty (m):** Pointwise uncertainty estimate
- **Significant change:** Binary flag (1 = significant, 0 = not significant based on LoD)
- **Original coordinates:** From comparison cloud (DATE2)

### CloudCompare M3C2 Command
```bash
CloudCompare -silent -auto_save off -c_export_fmt las \
  -o -global_shift X Y Z reference.las \
  -o -global_shift X Y Z comparison.las \
  -M3C2 params.txt \
  -SAVE_CLOUDS FILE "ref.las comp.las m3c2.las"
```

### Report Files
**Location:** `validation/m3c2/`

1. **<Location>_m3c2_inventory_YYYYMMDD_HHMMSS.csv**
   ```csv
   pair,ref_date,cmp_date,status,start_time,end_time,duration_sec,input_mb,error_message
   20240101_to_20240115,20240101,20240115,SUCCESS,2024-01-15 14:00:00,2024-01-15 14:05:23,323.45,245.2,
   ```

2. **<Location>_m3c2_summary_YYYYMMDD_HHMMSS.txt**
   ```
   M3C2 PROCESSING REPORT: SanElijo
   ============================================================
   Date:         2024-01-15 14:30:00
   Platform:     Linux (5.15.0)
   ------------------------------------------------------------
   Total Pairs: 41
     [+] Success: 41
     [-] Skipped: 0
     [!] Failed:  0
   Average Processing Time (Successes): 298.23 sec/pair
   ```

---

## Step 6: DBSCAN Clustering

**Script:** `6_dbscan_parallel.py`

### Purpose
Filters M3C2 results for significant changes, splits into erosion/deposition, applies DBSCAN clustering to identify discrete geomorphic events, and generates visualization reports.

### Usage
```bash
python3 6_dbscan_parallel.py SanElijo
python3 6_dbscan_parallel.py SanElijo --eps 0.35 --min_samples 30 --min_change 0.25 --replace
```

### Input
- **M3C2 files:** `results/<Location>/m3c2/pipeline_run_*/*/DATE1_to_DATE2_m3c2.las`
  - **Automatically detects and uses most recent `pipeline_run_*` folder** via `find_latest_pipeline_run()` function

### Critical Significance Filtering
**IMPORTANT:** The script enforces strict significance filtering to ensure only meaningful changes are processed:

1. **Searches for significance field** with multiple name variations:
   - 'significant change', 'Significant change', 'significant_change', 'SignificantChange', 'sig change', 'sig_change', 'sigchange'

2. **Fails with clear error if significance field not found:**
   ```
   ValueError: No significance field found in <filename>!
   Available fields: [...]
   M3C2 must be configured to output 'significant change' field.
   Cannot proceed without significance filtering.
   ```

3. **Displays which field is being used:**
   ```
   [INFO] Using significance field: 'significant change'
   ```

4. **Previous dangerous fallback removed:** Earlier versions would process ALL points if the field was missing. This has been replaced with a hard error to prevent accidental processing of non-significant data.

**Configuration Requirement:** Ensure M3C2 parameter files (`utilities/m3c2_params/*.txt`) are configured to output the significance field

### Output Directory Structure
```
results/<Location>/erosion/DATE1_to_DATE2/
├── ero_clusters.las         # Points with ClusterID ≥ 0
└── ero_outliers.las         # Points with ClusterID = -1 (noise)

results/<Location>/deposition/DATE1_to_DATE2/
├── dep_clusters.las
└── dep_outliers.las
```

### LAS File Fields
**ero_clusters.las / dep_clusters.las:**
- X, Y, Z: Original point coordinates
- **M3C2_distance:** Signed change magnitude (m)
- **Distance_uncertainty:** From M3C2
- **ClusterID:** DBSCAN cluster assignment (0, 1, 2, ...)

**ero_outliers.las / dep_outliers.las:**
- Same fields but ClusterID = -1 (noise points rejected by DBSCAN)

### Significance Filtering Logic
Points must meet BOTH criteria to be clustered:
```python
1. Significance field == 1  (from M3C2 Level of Detection test)
2. abs(M3C2_distance) >= min_change  (default: 0.25m, 0.30m for Torrey)
```
The significance field check is performed first, then optional magnitude threshold is applied.

### DBSCAN Parameters
- `--eps`: Neighborhood radius in meters (default: 0.35m)
- `--min_samples`: Min points to form cluster (default: 30)
- `--min_change`: Absolute change threshold in meters (default: 0.0)

### Report Files
**Location:** `utilities/dbscan/`

**<Location>_dbscan_inventory_YYYYMMDD_HHMMSS.csv:**
```csv
pair,status,duration_sec,erosion_input,erosion_clusters,erosion_outliers,erosion_volume,deposition_input,deposition_clusters,deposition_outliers,deposition_volume
20240101_to_20240115,SUCCESS,12.3,45623,12,34,123.45,23456,8,15,67.89
```

### Cluster Statistics
For each pair, the script calculates:
- Total points meeting significance threshold
- Number of clusters found
- Number of outlier (noise) points
- Total volume change (sum of abs(M3C2_distance) * point_spacing²)

---

## Step 7: Spatial Gridding

**Script:** `7_make_grids.py`

### Purpose
Aggregates clustered points into vertical bins within geospatial polygons. Creates spatiotemporal grids for time-series analysis.

### Usage
```bash
python3 7_make_grids.py SanElijo --resolution 10cm
python3 7_make_grids.py SanElijo --resolution 25cm --replace
python3 7_make_grids.py SanElijo --resolution 1m
```

### Input
1. **Clustered LAS files:**
   - `results/<Location>/erosion/*/ero_clusters.las`
   - `results/<Location>/deposition/*/dep_clusters.las`

2. **Polygon shapefiles:** `utilities/shape_files/<Location>_<resolution>/`
   - Polygons define horizontal spatial bins
   - Must contain `Polygon_ID` field

### Output Files
**Location:** `results/<Location>/erosion/DATE1_to_DATE2/<resolution>/` and `.../deposition/DATE1_to_DATE2/<resolution>/`

Files are organized into resolution-specific subdirectories within each date folder. For each survey pair and resolution, generates:

**Erosion:**
1. `results/<Location>/erosion/DATE1_to_DATE2/<resolution>/DATE1_to_DATE2_ero_grid_<resolution>.csv` (M3C2 median values)
2. `results/<Location>/erosion/DATE1_to_DATE2/<resolution>/DATE1_to_DATE2_ero_clusters_<resolution>.csv` (ClusterID mode)
3. `results/<Location>/erosion/DATE1_to_DATE2/<resolution>/DATE1_to_DATE2_ero_stats_<resolution>.npz` (Uncertainty RMS, point counts, variance)

**Deposition:**
Same structure with `dep_` prefix instead of `ero_`.

**Example:**
`/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs/results/DelMar/erosion/20250813_to_20250821/1m/20250813_to_20250821_ero_grid_1m.csv`

### CSV Grid Format

**Structure:** Rows = Polygon_IDs, Columns = Elevation bins

**Example: grid_10cm.csv (erosion)**
```csv
Polygon_ID,M3C2_0.10m,M3C2_0.20m,M3C2_0.30m,...,M3C2_50.00m
1001,,-0.234,-0.189,,-0.456,...,
1002,-0.123,,,,-0.678,...,
```

**Example: clusters_10cm.csv**
```csv
Polygon_ID,ClusterID_0.10m,ClusterID_0.20m,...
1001,,5,5,,7,...
1002,3,,,12,...
```

**Example: uncertainty_10cm.csv**
```csv
Polygon_ID,Uncertainty_0.10m,Uncertainty_0.20m,...
1001,,0.012,0.015,,0.018,...
1002,0.011,,,0.014,...
```

### Grid Cell Aggregation
For each (Polygon, Elevation) cell:
- **M3C2 distance:** Median of absolute values within cell
- **ClusterID:** Mode (most common) cluster ID
- **Uncertainty:** RMS of point uncertainties

### Resolution-to-Label Mapping
```python
'10cm'  -> '10cm'   # 0.10m vertical bins
'25cm'  -> '25cm'   # 0.25m vertical bins
'1m'    -> '1m'     # 1.00m vertical bins
```

**Note:** The resolution string is used directly as the folder name and file suffix. For example, `--resolution 1m` creates:
- Folder: `results/<Location>/erosion/DATE1_to_DATE2/1m/`
- Files: `DATE1_to_DATE2_ero_grid_1m.csv`, `DATE1_to_DATE2_ero_clusters_1m.csv`, etc.

### Legacy File Cleanup
When running with `--replace` and `--resolution 1m`, the script automatically removes old `100cm` files if they exist:
- Checks for `/100cm/` subdirectory in the same date folder
- Deletes `*_grid_100cm.csv`, `*_clusters_100cm.csv`, `*_stats_100cm.npz`
- Prints cleanup messages: `[Cleanup] Removed old file: <filename>`

This ensures a clean transition from the previous `100cm` naming convention to the standardized `1m` format.

### Vertical Binning
- Elevation range: 0.0m to 50.0m (hardcoded)
- Bin size depends on resolution
- Each column header format: `M3C2_<elevation>m`, `ClusterID_<elevation>m`, `Uncertainty_<elevation>m`

### Spatial Join Method
Uses optimized Geopandas spatial join:
```python
gdf_points = gpd.GeoDataFrame(points_df, geometry=gpd.points_from_xy(x, y))
joined = gpd.sjoin(gdf_points, polygon_gdf, how='inner', predicate='within')
```

---

## Step 8: Grid Cleaning & Hole Filling

**Script:** `8_clean_fill_grids.py`

### Purpose
Post-processes grids by:
1. Applying visual cliff-top cutoffs (removes data above cliff edge)
2. Filtering clusters by size threshold
3. Footprint checking (deposition validation against erosion)
4. Filling holes within erosion clusters using diffusion with smoothed boundaries
5. Morphological cleanup (removes isolated cells)

**Note:** Only saves final `_filled.csv` outputs. For erosion, this includes cleaning + hole filling + morphological cleanup. For deposition, this includes cleaning only (no filling).

**Hole Filling Method:** Uses index-based iterative diffusion with circular convolution boundary smoothing (radius=4). Empty cells inside cluster boundaries are filled by averaging valid neighbors, with nearest-neighbor fallback for disconnected cells.

### Usage
```bash
# Process all locations
python3 8_clean_fill_grids.py --all --resolution 25cm

# Erosion only (with filling)
python3 8_clean_fill_grids.py SanElijo --resolution 10cm --erosion --min_volume 2.0

# Deposition only (cleaning only, no filling)
python3 8_clean_fill_grids.py SanElijo --resolution 25cm --deposition

# Both types
python3 8_clean_fill_grids.py SanElijo --resolution 1m

# Skip filling (cleaning only, for erosion)
python3 8_clean_fill_grids.py SanElijo --resolution 10cm --erosion --skip_filling

# Testing mode (no file writes)
python3 8_clean_fill_grids.py SanElijo --resolution 10cm --testing
```

### Input Files
1. **Grid CSVs from step 7:**
   - `results/<Location>/erosion/DATE1_to_DATE2/<resolution>/DATE1_to_DATE2_ero_grid_<resolution>.csv`
   - `results/<Location>/erosion/DATE1_to_DATE2/<resolution>/DATE1_to_DATE2_ero_clusters_<resolution>.csv`
   - Same for deposition with `dep_` prefix

2. **Visual cutoff file:** `utilities/cliff_top_cutoffs/<Location>_Visual_CliffTop_<resolution>.csv`
   ```csv
   Polygon_ID,CliffTop_Z
   1001,12.5
   1002,15.3
   ```

### Output Files
**Location:** Same resolution-specific subdirectories as input files (`results/<Location>/erosion/DATE1_to_DATE2/<resolution>/`)

**Erosion (with hole filling):**
1. `DATE1_to_DATE2_ero_grid_<resolution>_filled.csv` - M3C2 distances after cleaning + hole filling + morphological cleanup
2. `DATE1_to_DATE2_ero_clusters_<resolution>_filled.csv` - Cluster IDs after cleaning + hole filling + morphological cleanup

**Deposition (cleaning only, no filling):**
1. `DATE1_to_DATE2_dep_grid_<resolution>_filled.csv` - M3C2 distances after cleaning
2. `DATE1_to_DATE2_dep_clusters_<resolution>_filled.csv` - Cluster IDs after cleaning

**Note:** Intermediate `_cleaned.csv` files are not saved. Only final `_filled.csv` outputs are written.

### Processing Steps

#### 1. Visual Cliff-Top Cutoff
Removes any grid cell where `elevation > CliffTop_Z` for that polygon:
```python
# Zeros out cells above visual cutoff line
mask_above = col_elevations[None, :] > cutoff_z[:, None]
grid[mask_above] = 0
clusters[mask_above] = 0
```

#### 2. Threshold Filtering
Removes clusters with fewer than N non-zero cells:
```python
# Resolution-dependent thresholds
'10cm': default_threshold = 25 cells
'25cm': default_threshold = 4 cells
'1m':   default_threshold = 1 cell
```

#### 3. Deposition Footprint Check
For deposition clusters, validates against erosion:
- Checks if erosion exists within vertical buffer (±2m for 10cm resolution)
- Removes deposition clusters with no erosion foundation

#### 4. Hole Filling (Erosion Only)
Uses index-based diffusion filling with smoothed boundaries:

**Parameters:**
- `dilation=1`: Initial morphological dilation iterations
- `conv_radius=4`: Circular convolution radius for boundary smoothing
- `--min_volume 2.0`: Skip clusters below this volume (m³)

**Algorithm:**
1. Calculate cluster volumes
2. Skip clusters below `--min_volume` threshold (default: 2.0 m³)
3. For each qualifying cluster:
   - Create morphological fill mask (`binary_fill_holes` + dilation)
   - Smooth boundary using circular convolution (radius=4)
   - Fill holes using iterative diffusion:
     - For each empty cell inside boundary, compute mean of valid 4-connected neighbors
     - Only use neighbors that are also inside the boundary
     - Iterate until all cells are filled
   - Fallback to nearest-neighbor for any disconnected cells
   - Assign filled cells to cluster ID

**Boundary Smoothing:**
```python
# Circular convolution smooths jagged cluster boundaries
y_k, x_k = np.ogrid[-radius:radius+1, -radius:radius+1]
circular_kernel = (x_k**2 + y_k**2 <= radius**2).astype(float)
circular_kernel /= circular_kernel.sum()
smoothed_mask = convolve(base_mask.astype(float), circular_kernel) > 0.5
```

**Diffusion Fill:**
```python
# Iteratively fill empty cells with mean of valid neighbors
for each empty_cell inside boundary:
    neighbors = [n for n in 4-connected if inside_boundary and has_data]
    if neighbors:
        new_value = mean(neighbors)
# Fallback: nearest-neighbor for disconnected cells
```

### Report Files
**Location:** `validation/hole_filling/reports/<Location>/`

**combined_report_<resolution>_YYYYMMDD_HHMMSS.txt:**
```
CLEANING + HOLE FILLING (Index-Based Diffusion)
================================================================================
Location: SanElijo
Resolution: 10cm
Method: Diffusion fill with smoothed boundaries (conv_radius=4)

CLEANING SUMMARY (Visual Cutoff Applied)
------------------------------
Total Cells Removed by Visual Cutoff: 1234

HOLE FILLING SUMMARY
------------------------------
Total Vol Change: 12.3456 m³
Total Holes Filled: 456

DETAILED LOG
------------------------------------------------------------------------------------------
Survey                    Type     CutoffCells  Holes   Vol∆       Fill%
20240101_to_20240115      erosion  89           23      1.234      45.6
```

---

## Data Flow Summary

```
Raw LAS
  └─> [Step 2: Crop] ──> cropped/
       ├─> [Audit: QC] ──> reports/
       └─> [Step 3: Beach] ──> nobeach/
            └─> [Step 4: Veg] ──> noveg/
                 └─> [Step 5: M3C2] ──> m3c2/pipeline_run_*/DATE1_to_DATE2/
                      └─> [Step 6: DBSCAN] ──> erosion/ & deposition/
                           └─> [Step 7: Grid] ──> grid_*.csv & clusters_*.csv
                                └─> [Step 8: Clean/Fill] ──> *_filled.csv (final output only)
```

**Note:** Step 8 only saves final `*_filled.csv` outputs (no intermediate `*_cleaned.csv` files).

---

## Critical File Format Notes

### LAS Point Cloud Format
All `.las` files use LAS 1.2 or later with these characteristics:
- **Coordinate System:** UTM Zone 11N (EPSG:32611)
- **Units:** Meters
- **Point Format:** Varies (0-3), typically format 3 (XYZ + Intensity + RGB)
- **Scalar Fields:** Added by M3C2 and DBSCAN steps

### CSV Grid Format Details
- **Missing data:** Empty string `""` (NOT "NaN", "nan", or "0")
- **Index column:** First column is `Polygon_ID` (row labels)
- **Header row:** First row contains elevation/cluster labels
- **Numeric precision:** 6 significant figures (`%.6g`)
- **Zero vs Empty:** `0` = explicit zero value, `""` = no data/occluded

### Reading Grid CSVs in Python
```python
import pandas as pd

# Correct way to read
df = pd.read_csv(filepath, index_col=0, header=0,
                 na_values=['', 'nan', 'NaN', 'NULL'])

# Access data
elevations = df.columns.tolist()  # ['M3C2_0.10m', 'M3C2_0.20m', ...]
polygon_ids = df.index.tolist()    # [1001, 1002, ...]
data = df.values                   # numpy array (rows=polygons, cols=elevations)
```

### Writing Grid CSVs in Python
```python
import pandas as pd

df = pd.DataFrame(data, columns=header_labels, index=row_labels)
df.to_csv(filepath, na_rep='', float_format='%.6g')
```

---

## Location-Specific Configurations

### Adding a New Location

1. **Define MOP Range** in all relevant scripts:
   ```python
   mop_ranges["NewLocation"] = (MIN_MOP, MAX_MOP)
   ```

2. **Create/Train Beach Removal Model:**
   ```bash
   python3 code/training/0_train_rf_model.py
   # Outputs: utilities/beach_removal/NewLocation_rf_model.joblib
   #          utilities/beach_removal/NewLocation_scaler.joblib
   ```

3. **Set Global Shift** (if needed) in steps 5 & 6:
   ```python
   shift["NewLocation"] = ("-X_offset", "-Y_offset", "0")
   # Typically -1 * first 3 digits of UTM coordinates
   ```

4. **Create Polygon Shapefile** for gridding:
   - Place in `utilities/shape_files/NewLocation_<resolution>/`
   - Must contain `Polygon_ID` field

5. **Create Visual Cutoff File** (optional):
   - `utilities/cliff_top_cutoffs/NewLocation_Visual_CliffTop_<resolution>.csv`
   - Format: `Polygon_ID,CliffTop_Z`

---

## Performance Optimization

### Parallel Processing Settings
Most scripts use parallel processing with configurable workers:
```python
# Adjust based on your system
--n_jobs 5        # Steps 3, 6
max_workers=3     # Step 2 (also set in OMP_NUM_THREADS)
max_workers=4     # Step 5
workers = cpu_count() // 4  # Step 8
```

### Thread Control
Set at script start to avoid oversubscription:
```python
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"
```

### Memory Considerations
- **Step 5 (M3C2):** Most memory-intensive (CloudCompare loads full clouds)
- **Step 7 (Gridding):** Uses Geopandas spatial joins (optimized)
- **Step 8 (Filling):** Loads full grid into memory per survey pair

---

## Troubleshooting Common Issues

### CloudCompare Failures (Steps 4 & 5)
**Symptom:** "Display required" error on headless servers

**Solution:** Use Xvfb (X Virtual Framebuffer)
```bash
xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" python3 script.py
```

### Missing Survey Files
**Symptom:** "No LAS in: ..." warnings in step 2

**Solution:** Check that survey folders contain:
```
<survey>/Beach_And_Backshore/*_beach_cliff_ground.las
```

### Grid Shape Mismatches
**Symptom:** "Shape enforcement error" in step 7

**Cause:** Inconsistent polygon shapefiles or elevation ranges

**Solution:** Ensure all surveys for a location use identical:
- Polygon shapefile (same Polygon_IDs)
- Elevation range (0-50m)
- Resolution settings

### Coordinate System Issues
**Symptom:** Points outside expected bounds after M3C2

**Cause:** Inconsistent `global_shift` between steps 4 and 5

**Solution:** Verify shift values match exactly in both scripts for the location

---

## Testing & Validation

### Partial Runs
Most scripts support `--test N` flag to process only N items:
```bash
python3 3_remove_beach_parallel.py SanElijo --test 5
python3 6_dbscan_parallel.py SanElijo --test 3
```

### Dry Run Mode
Step 8 supports testing without file writes:
```bash
python3 8_clean_fill_grids.py SanElijo --resolution 10cm --testing
```

### Validation Checks
After each step, verify:
1. **File counts:** Match expected survey pairs
2. **Reports:** Check for errors/failures
3. **Visualizations:** Review generated plots (audit and step 6)
4. **Sample data:** Spot-check LAS files in CloudCompare

---

## Known Issues / TODO

### Step 8: Vertical Stripes in Filled Grids

**Status:** RESOLVED (2026-02-02)

**Original symptom:** Thin vertical stripes (unfilled holes at specific alongshore positions) visible in the filled grid data.

**Root cause:** The original alphashape-based filling approach used UTM physical coordinates for boundary detection and interpolation. This caused issues with:
1. `binary_fill_holes` only filling completely surrounded holes (stripes connecting to edges weren't filled)
2. `scipy.interpolate.griddata` sometimes failing at boundary edges
3. Complex boundary shapes from alphashape not matching grid structure

**Solution implemented:**
Replaced alphashape/physical coordinate approach with index-based diffusion filling:

1. **Boundary smoothing:** Use circular convolution (radius=4) on morphological fill mask to create smooth, well-defined cluster boundaries

2. **Iterative diffusion fill:** For each empty cell inside the boundary:
   - Compute mean of valid 4-connected neighbors (only neighbors inside boundary)
   - Iterate until all cells are filled
   - Guarantees all cells inside boundary get filled

3. **Nearest-neighbor fallback:** For any disconnected cells that diffusion can't reach, copy value from nearest cell with data

**Testing script:** `code/pipeline/grid_fill_index_based.py`
```bash
# Test boundary smoothing options
python3 grid_fill_index_based.py --test-radii --top 10

# Test diffusion fill
python3 grid_fill_index_based.py --test-fill --top 10

# Test specific event range
python3 grid_fill_index_based.py --test-fill --start 19 --top 11
```

**Test figures location:** `figures/testing/boundary/` and `figures/testing/diffusion_fill/`
