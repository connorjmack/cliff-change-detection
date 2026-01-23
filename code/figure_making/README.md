# Figure Making Scripts

This directory contains visualization scripts for generating publication-quality figures, dashboards, and animations for the coastal cliff erosion analysis project.

## Output Directories

All figures are saved to subdirectories within the main `figures/` folder:

| Output Directory | Description |
|------------------|-------------|
| `figures/dashboards/` | Main erosion dashboards (5-panel) |
| `figures/gifs/` | Animated GIF frames and final animations |
| `figures/cumulative_erosion/` | Cumulative cliff face heatmaps |
| `figures/sensitivity/` | Grid resolution sensitivity comparisons |
| `figures/geomorph_stats/` | Geomorphology statistics plots |
| `figures/uncertainty/` | Uncertainty distribution diagnostics |

---

## Dashboard Scripts

### `plot_dashboard.py`
Generates the main 5-panel "Master Dashboard" for each location showing:
- Panel A: Total erosion volume per interval (bar chart)
- Panel B: Cumulative erosion volume (line with uncertainty)
- Panel C: Spatio-temporal distribution of large events (bubble plot)
- Panel D: Spatial distribution by elevation (bubble plot)
- Panel E: Cumulative cliff activity heatmap (cliff-facing view)

```bash
python3 plot_dashboard.py --location SanElijo
python3 plot_dashboard.py --location all
```

### `dashboard_gif.py`
Generates animated dashboard frames (one per survey interval) showing progressive buildup of erosion data. Creates the same 5-panel layout as `plot_dashboard.py` but animated over time.

```bash
# Generate frames only
python3 dashboard_gif.py --location SanElijo

# Generate frames and create GIF
python3 dashboard_gif.py --location SanElijo --make-gif

# Custom frame duration (milliseconds)
python3 dashboard_gif.py --location all --make-gif --duration 100
```

**Output:** `figures/gifs/<Location>/<Location>_frame_XXXX.png` and `figures/gifs/<Location>_Dashboard_25cm_animated.gif`

### `simple_gif.py`
Generates a simplified 2-panel animated dashboard:
- Top (2/3): Cumulative erosion heatmap (cliff-facing view)
- Bottom (1/3): Cumulative volume timeline with uncertainty bounds

Professional-quality labels optimized for presentations.

```bash
python3 simple_gif.py --location SanElijo
python3 simple_gif.py --location SanElijo --make-gif --duration 150
```

**Output:** `figures/gifs/<Location>/<Location>_simple_frame_XXXX.png` and `figures/gifs/<Location>_SimpleTimelapse_25cm.gif`

### `sensitvity_dashboard.py`
4-panel dashboard comparing erosion results across grid resolutions (10cm, 25cm, 1m).

```bash
python3 sensitvity_dashboard.py --location DelMar
```

### `del_mar_timeseries.py`
Specialized 4-panel dashboard for Del Mar location with custom styling.

```bash
python3 del_mar_timeseries.py
```

---

## Time Series & Cumulative Plots

### `erosion_time_series.py`
Generates cumulative erosion plots with 6 curves per location:
- 3 resolutions (10cm, 25cm, 1m)
- 2 types (filled, cleaned)

Includes uncertainty bounds for all curves.

```bash
python3 erosion_time_series.py --location SanElijo
```

### `cum_erosion.py`
Creates 3x1 figures showing cumulative erosion/deposition across resolutions.

```bash
python3 cum_erosion.py --erosion
python3 cum_erosion.py --deposition
```

### `cum_raw_vis.py`
Visualizes cumulative erosion/deposition grids as single heatmaps.

```bash
python3 cum_raw_vis.py --location SanElijo --resolution 25cm --type erosion
python3 cum_raw_vis.py --all
```

### `make_cumulative_gif.py`
Creates cumulative erosion/deposition GIF animations from grid data. Generates both high-res and low-res versions.

```bash
python3 make_cumulative_gif.py SanElijo --resolution 10cm --erosion
```

---

## Statistics & Analysis Plots

### `geo_stats.py`
Generates a 2x2 geomorphology statistics dashboard including event size distributions, temporal patterns, and seasonal analysis.

```bash
python3 geo_stats.py --location DelMar
python3 geo_stats.py --location all
```

### `write_report.py`
Calculates and prints geomorphological statistics for paper text (event counts, volumes, distributions, seasonality).

```bash
python3 write_report.py --location DelMar
python3 write_report.py --location all
```

### `uncertainty_dist_plot.py`
2-panel diagnostic plot for uncertainty analysis:
- Left: Distribution of raw grid cell uncertainty
- Right: Aggregated volume uncertainty per cluster

```bash
python3 uncertainty_dist_plot.py --location DelMar
```

### `epoch_time_plot.py`
Visualizes "Epoch Times" (duration between consecutive surveys) with violin plots.

```bash
python3 epoch_time_plot.py
```

---

## Point Cloud & Classification Visualization

### `ml_class_distribution.py`
Analyzes point cloud reduction across pipeline stages (Cropped → NoBeach → NoVeg). Shows violin plots of point counts by classification.

```bash
python3 ml_class_distribution.py --location DelMar
python3 ml_class_distribution.py  # All locations
```

### `vis_ml_classes.py`
Generates 3D renderings comparing RGB point clouds to classified outputs (beach, cliff, vegetation).

```bash
python3 vis_ml_classes.py
```

### `radio_norm.py`
Radiometric normalization visualization comparing intensity distributions across instruments (VMZ2000, MiniRanger, VMQLZ).

```bash
python3 radio_norm.py
```

### `file_size_density.py`
Analyzes and visualizes point cloud file sizes and point densities.

```bash
python3 file_size_density.py
```

---

## Sensitivity & Comparison Plots

### `sensitivity_grids.py`
Visualizes "Cliff Activity Index" across different grid resolutions with proper spatial alignment.

```bash
python3 sensitivity_grids.py --location DelMar
python3 sensitivity_grids.py --location DelMar --zoom 1450 1550
```

### `compare_filled_vs_cleaned.py`
Two-panel cumulative erosion comparison (filled vs. original grids) for QA of the fill step.

```bash
python3 compare_filled_vs_cleaned.py SanElijo --resolution 25cm
```

---

## Utility Scripts

### `workflow_fig.py`
Extracts event-specific LAS file subsets for forensic analysis and figure generation.

```bash
python3 workflow_fig.py --location DelMar --d1 20250124 --d2 20251117
```

### `get_cloud_bbox.py`
Extracts RGB-colorized point cloud subsets from VMQLZ surveys, preserving all LAS dimensions.

```bash
python3 get_cloud_bbox.py --location DelMar
```

---

## Common Arguments

Most scripts support these arguments:

| Argument | Description |
|----------|-------------|
| `--location` | Location name (DelMar, Torrey, Solana, Encinitas, SanElijo) or "all" |
| `--resolution` | Grid resolution (10cm, 25cm, 1m) |
| `--make-gif` | Create animated GIF from frames |
| `--duration` | Frame duration in milliseconds (default: 150) |

## Requirements

- Python 3.8+
- matplotlib, numpy, pandas, seaborn
- laspy (for point cloud scripts)
- pyvista (for 3D rendering scripts)
- Pillow (for GIF creation)
