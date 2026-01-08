# LiDAR Survey Browser

A Streamlit web application for filtering and browsing LiDAR surveys by MOP range, date range, and instrument type.

## Features

- **MOP Range Selection**: Set custom min/max MOP line values
- **Date Range Selection**: Filter surveys by start and end dates
- **Instrument Filtering**: Choose which instruments to include in results
- **Interactive Table**: View all matching surveys with full paths
- **CSV Export**: Download filtered results as CSV file
- **Summary Statistics**: Quick overview of results (count, date range, unique methods)

## Installation

First, ensure you have Streamlit installed:

```bash
pip install streamlit pandas numpy
```

Or if using conda:

```bash
conda install -c conda-forge streamlit pandas numpy
```

## Usage

Run the application from the command line:

```bash
cd code/streamlit
streamlit run survey_browser.py
```

This will open a new browser window with the interactive GUI.

## How It Works

1. **Configure Filters** (left sidebar):
   - Set MOP range (min and max values)
   - Set date range (start and end dates)
   - Select which instruments to include

2. **Search**:
   - Click "Search Surveys" button
   - App scans all instrument directories for matching surveys

3. **View Results**:
   - Table shows: date, MOP1, MOP2, method, folder_name, full_path
   - Summary statistics displayed above table

4. **Export**:
   - Click "Download Results as CSV" to save filtered results

## Filter Logic

- **MOP Overlap**: Surveys must have ≥2/3 overlap with the selected MOP range (same logic as `0_make_survey_lists.py`)
- **Date Range**: Only includes surveys within the specified date range
- **Instruments**: Only scans selected instrument directories

## Predefined Locations

The home screen shows MOP ranges for common beach locations:
- Del Mar: MOP 595-620
- Solana: MOP 637-666
- Encinitas: MOP 708-764
- San Elijo: MOP 683-708
- Torrey: MOP 567-581
- Blacks: MOP 520-567

## Output Format

CSV files contain the following columns:
- `date`: Survey date (YYYYMMDD format)
- `MOP1`: Starting MOP line number
- `MOP2`: Ending MOP line number
- `method`: Instrument type (e.g., MiniRanger_Truck, VMQLZ_Truck)
- `folder_name`: Survey folder name
- `full_path`: Complete path to survey directory
