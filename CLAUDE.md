# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a coastal cliff LiDAR processing pipeline that ingests raw terrestrial LiDAR surveys, preprocesses them, removes non-relevant features (beach, vegetation), performs change detection using M3C2, clusters erosion/deposition events with DBSCAN, and aggregates data into spatiotemporal grids for analysis.

## External Dependencies

- **PDAL**: Required for cropping (step 2). Install via `brew install pdal` (macOS) or `apt-get install pdal` (Linux).
- **CloudCompare**: Required for M3C2 and CANUPO. Pass path using `--cc /path/to/CloudCompare` flag.
- **xvfb-run**: Required on headless servers for CloudCompare steps (4, 5). Use `xvfb-run -a python3 ...`

## Directory Structure

```
code/
├── pipeline/            # Main processing scripts (0-8 sequential)
├── figure_making/       # Publication visualization scripts
├── streamlit/           # Survey browser web app
└── training/            # ML model training scripts
results/<Location>/      # Output directories (e.g., DelMar, SanElijo)
├── cropped/ → nobeach/ → noveg/ → m3c2/pipeline_run_YYYYMMDD/ → erosion/ & deposition/
survey_lists/            # CSV inventories of surveys
tests/pytest/            # Unit tests
utilities/
├── beach_removal/       # RF Models (.joblib) and Scalers
├── canupo/              # Vegetation classifiers (.prm)
├── m3c2_params/         # CloudCompare parameter files
├── cliff_top_cutoffs/   # Visual cutoff CSVs per location
├── dbscan/              # DBSCAN clustering reports
└── event_lists/         # Event list generation scripts
results/event_lists/     # Generated event CSVs
results/data_cubes/      # 3D NPZ data cubes
```

**Path Handling**: Scripts auto-detect OS: macOS uses `/Volumes/group/LiDAR`, Linux/HPC uses `/project/group/LiDAR`.

## Pipeline Execution Sequence

Scripts 0-8 run sequentially. Each step transforms data for the next.

### Step 0-1: Survey Lists
```bash
python3 code/pipeline/0_make_survey_lists.py --location SanElijo  # Initial creation
python3 code/pipeline/1_update_survey_lists.py --all              # Update with new surveys
```

### Step 2: Crop to Study Area
```bash
python3 code/pipeline/2_crop_files_parallel.py --location SanElijo --replace
```

### Step 3: Beach Removal
```bash
python3 code/pipeline/3_remove_beach_parallel.py SanElijo --n_jobs 5
```
Random Forest classifier with histogram matching to normalize intensity across sensors.

### Step 4: Vegetation Removal
```bash
python3 code/pipeline/4_remove_veg_parallel.py SanElijo --cc "/path/to/CloudCompare"
```

### Step 5: M3C2 Change Detection
```bash
python3 code/pipeline/5_m3c2_parallel.py SanElijo --cc "/path/to/CloudCompare" --replace
```
Creates timestamped `pipeline_run_YYYYMMDD/` subdirectories in `results/<Location>/m3c2/`.

### Step 6: DBSCAN Clustering
```bash
python3 code/pipeline/6_dbscan_parallel.py SanElijo --eps 0.35 --min_samples 30 --min_change 0.25
```
Automatically processes most recent `pipeline_run_*` folder. **Critical:** Only processes points with significant M3C2 changes. Fails if significance field not found in M3C2 output.

### Step 7: Spatial Gridding
```bash
python3 code/pipeline/7_make_grids.py SanElijo --resolution 10cm --replace
```
Available resolutions: 10cm, 25cm, 1m. Creates resolution-specific subdirectories.

### Step 8: Grid Cleaning & Hole Filling
```bash
python3 code/pipeline/8_clean_fill_grids.py --all --resolution 25cm
```
Applies cliff-top cutoffs and fills occlusion holes. Use `--erosion`/`--deposition` for specific types, `--testing` for dry-run.

### Automated Daily Pipeline
```bash
python3 code/pipeline/run_daily.py
python3 code/pipeline/run_daily.py --force-all  # Force reprocessing
```

### Event List Generation
```bash
# Generate event CSVs only
python3 utilities/event_lists/make_event_lists.py SanElijo
python3 utilities/event_lists/make_event_lists.py --all

# Generate both CSVs and 3D data cubes
python3 utilities/event_lists/make_event_lists.py --all --make-npz

# Optional: Filter for significant events (volume > 5 m³, elevation > 5 m)
python3 utilities/event_lists/make_sig_event_lists.py
python3 utilities/event_lists/make_sig_event_lists.py --min_volume 10 --min_elevation 3

# Optional: Generate filtered 3D data cubes
python3 utilities/event_lists/make_sig_event_lists.py --make-npz
```

Output: CSVs in `results/event_lists/`, NPZ cubes in `results/data_cubes/`

## Location-Specific Configuration

### MOP Ranges (defined in step 2 script)
- DelMar: 595-620, Solana: 637-666, Encinitas: 708-764
- SanElijo: 683-708, Torrey: 567-581, Blacks: 520-567

### Global Coordinate Shifts
CloudCompare requires coordinate shifts for large UTM values. Defined in steps 4 and 5:
```python
shift = {
    "SanElijo":   ("-473000", "-3653000", "0"),
    "Encinitas":  ("-472000", "-3655000", "0"),
    "Solana":     ("-475000", "-3650000", "0"),
    "Torrey":     ("-475000", "-3650000", "0"),
}
```
**Important:** Shifts must be consistent across steps 4 and 5.

## Architecture Notes

- **Parallel Processing**: Most scripts use `ProcessPoolExecutor`. Control with `--n_jobs` flag (default: 5).
- **Point Cloud Handling**: `laspy` for LAS/LAZ, `geopandas`/`shapely` for spatial ops, PDAL for cropping, CloudCompare CLI for M3C2/CANUPO.
- **Data Flow**: Raw → Cropped → Beach Removed → Veg Removed → M3C2 → DBSCAN Clustered → Gridded → Cleaned/Filled → Event Lists/Cubes

## Testing

```bash
pytest tests/pytest/                                    # Run all tests
python3 code/pipeline/3_remove_beach_parallel.py SanElijo --test 10  # Partial run (N files)
```
