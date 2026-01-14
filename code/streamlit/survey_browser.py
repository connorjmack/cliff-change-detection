#!/usr/bin/env python3
"""
Survey Browser - Streamlit GUI for filtering LiDAR surveys by MOP range and date

Usage:
    streamlit run survey_browser.py
"""

import os
import platform
import calendar
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# === OS Detection and Path Setup ===
system = platform.system()
if system == "Darwin":
    ROOT_LIDAR = "/Volumes/group/LiDAR"
else:
    ROOT_LIDAR = "/project/group/LiDAR"

# === Instrument Paths ===
INSTRUMENT_PATHS = {
    "MiniRanger_Truck": os.path.join(ROOT_LIDAR, "MiniRanger_Truck/LiDAR_Processed_Level2"),
    "MiniRanger_ATV": os.path.join(ROOT_LIDAR, "MiniRanger_ATV/LiDAR_Processed_Level2"),
    "VMQLZ_Truck": os.path.join(ROOT_LIDAR, "VMQLZ_Truck/LiDAR_Processed_Level2"),
    "VMZ2000_Truck": os.path.join(ROOT_LIDAR, "VMZ2000_Truck/LiDAR_Processed_Level2"),
}

# === Constants ===
MOP_RANGES = {
    "DelMar"      : [595, 620],
    "Solana"      : [637, 666],
    "Encinitas"   : [708, 764],
    "SanElijo"    : [683, 708],
    "Torrey"      : [567, 581],
    "Blacks"      : [520, 567]
}

def scan_surveys(target_ranges, start_date, end_date, selected_instruments):
    """
    Scan all instrument folders and return surveys that match the criteria.

    Parameters:
    - target_ranges: list of [min_mop, max_mop] lists or tuples
    - start_date, end_date: datetime objects for date range
    - selected_instruments: list of instrument names to include

    Returns:
    - DataFrame with columns: date, MOP1, MOP2, method, folder_name, full_path
    """
    rows = []

    # Convert dates to YYYYMMDD format for comparison
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    for method, root in INSTRUMENT_PATHS.items():
        # Skip if instrument not selected
        if method not in selected_instruments:
            continue

        # Skip if path doesn't exist
        if not os.path.isdir(root):
            continue

        for name in os.listdir(root):
            subdir = os.path.join(root, name)

            # Must be a directory
            if not os.path.isdir(subdir):
                continue

            # Must follow naming convention YYYYMMDD_MOP1_MOP2_...
            parts = name.split("_")
            if len(parts) < 3:
                continue

            try:
                date_str = parts[0]
                mop1 = int(parts[1])
                mop2 = int(parts[2])
            except ValueError:
                continue

            # Check date range
            if date_str < start_str or date_str > end_str:
                continue

            # Check MOP overlap with ANY of the target ranges
            match_found = False
            for t_min, t_max in target_ranges:
                # Calculate 2/3 overlap threshold for this target range
                threshold = np.floor((t_max - t_min) * 2/3)
                
                # Calculate overlap
                overlap = min(mop2, t_max) - max(mop1, t_min)
                
                if overlap >= threshold:
                    match_found = True
                    break
            
            if not match_found:
                continue

            # Construct full path to the beach_cliff_ground.las file
            las_path = os.path.join(subdir, "Beach_And_Backshore", f"{name}_beach_cliff_ground.las")
            dem_path = f"{os.path.splitext(las_path)[0]}.tif"

            # Add to results
            rows.append({
                "date": date_str,
                "MOP1": mop1,
                "MOP2": mop2,
                "method": method,
                "folder_name": name,
                "full_path": las_path,
                "dem_path": dem_path
            })

    # Convert to DataFrame and sort by date
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values("date").reset_index(drop=True)
        return df
    else:
        return pd.DataFrame(columns=["date", "MOP1", "MOP2", "method", "folder_name", "full_path", "dem_path"])

# === Streamlit App ===
st.set_page_config(page_title="LiDAR Survey Browser", layout="wide")

st.title("🌊 LiDAR Survey Browser")
st.markdown("Filter surveys by MOP range, date range, and instrument type")

# Sidebar controls
st.sidebar.header("Filter Options")

# Location Selection
st.sidebar.subheader("Location")
selected_beaches = st.sidebar.multiselect(
    "Select Beaches",
    options=list(MOP_RANGES.keys()),
    help="Select one or more beaches to filter surveys."
)

target_ranges = []

if selected_beaches:
    # Use selected beaches
    for beach in selected_beaches:
        target_ranges.append(MOP_RANGES[beach])
    
    st.sidebar.markdown("**Selected Regions:**")
    for beach in selected_beaches:
        r = MOP_RANGES[beach]
        st.sidebar.markdown(f"- {beach}: MOP {r[0]} - {r[1]}")
else:
    # Manual MOP Range
    st.sidebar.markdown("Or define a custom MOP range:")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_mop = st.number_input("Min MOP", min_value=0, max_value=1000, value=520, step=1)
    with col2:
        max_mop = st.number_input("Max MOP", min_value=0, max_value=1000, value=764, step=1)

    # Validate MOP range
    if min_mop >= max_mop:
        st.sidebar.error("Min MOP must be less than Max MOP")
        target_ranges = [] # Prevent search if invalid
    else:
        target_ranges = [[min_mop, max_mop]]
        st.sidebar.markdown(f"**Custom Range:** MOP {min_mop} - {max_mop}")
        st.sidebar.markdown(f"*Surveys need ≥{int(np.floor((max_mop - min_mop) * 2/3))} MOP lines overlap*")

# Date Range
st.sidebar.subheader("Date Range")

# Start Date
st.sidebar.markdown("**Start Date**")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_month = st.selectbox("Month", range(1, 13), index=0, key="start_month",
                               format_func=lambda x: datetime(2000, x, 1).strftime("%B"))
with col2:
    current_year = datetime.now().year
    start_year = st.selectbox("Year", range(2017, current_year + 1), index=0, key="start_year")

# End Date
st.sidebar.markdown("**End Date**")
col1, col2 = st.sidebar.columns(2)
with col1:
    end_month = st.selectbox("Month", range(1, 13), index=datetime.now().month - 1, key="end_month",
                            format_func=lambda x: datetime(2000, x, 1).strftime("%B"))
with col2:
    end_year = st.selectbox("Year", range(2017, current_year + 1),
                           index=current_year - 2017, key="end_year")

# Convert to datetime objects (first day of start month, last day of end month)
start_date = datetime(start_year, start_month, 1)
# For end date, get the last day of the month
last_day = calendar.monthrange(end_year, end_month)[1]
end_date = datetime(end_year, end_month, last_day)

# Validate date range
if start_date > end_date:
    st.sidebar.error("Start date must be before end date")
    st.stop()

# Instrument Selection
st.sidebar.subheader("Instruments")
instruments = list(INSTRUMENT_PATHS.keys())
selected_instruments = st.sidebar.multiselect(
    "Select instruments to include:",
    options=instruments,
    default=instruments
)

if not selected_instruments:
    st.sidebar.warning("Please select at least one instrument")
    st.stop()

# Run Search Button
st.sidebar.markdown("---")
search_button = st.sidebar.button("🔍 Search Surveys", type="primary", use_container_width=True)

# Main content area
if search_button:
    if not target_ranges:
        st.error("Invalid configuration. Please select a beach or define a valid MOP range.")
    else:
        with st.spinner("Scanning survey directories..."):
            results_df = scan_surveys(target_ranges, start_date, end_date, selected_instruments)

        if len(results_df) > 0:
            st.success(f"✅ Found {len(results_df)} surveys matching your criteria")

            # Display results
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date": st.column_config.TextColumn("Date", width="small"),
                    "MOP1": st.column_config.NumberColumn("MOP1", width="small"),
                    "MOP2": st.column_config.NumberColumn("MOP2", width="small"),
                    "method": st.column_config.TextColumn("Method", width="medium"),
                    "folder_name": st.column_config.TextColumn("Folder Name", width="large"),
                    "full_path": st.column_config.TextColumn("Full Path", width="large"),
                    "dem_path": st.column_config.TextColumn("DEM Path", width="large")
                }
            )

            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Surveys", len(results_df))
            with col2:
                date_range_days = (datetime.strptime(results_df['date'].max(), "%Y%m%d") -
                                  datetime.strptime(results_df['date'].min(), "%Y%m%d")).days
                st.metric("Date Range (days)", date_range_days)
            with col3:
                st.metric("Unique Methods", results_df['method'].nunique())

            # Download button
            st.markdown("---")
            csv = results_df.to_csv(index=False)
            
            # Generate filename based on selection
            if selected_beaches:
                loc_str = "-".join(selected_beaches)
                if len(loc_str) > 30:
                    loc_str = "Multiple_Beaches"
            else:
                # Manual range (target_ranges[0] is [min, max])
                loc_str = f"MOP{target_ranges[0][0]}-{target_ranges[0][1]}"

            filename = f"survey_results_{loc_str}_{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.csv"
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("⚠️ No surveys found matching your criteria. Try adjusting the filters.")
            st.info("**Tips:**\n- Expand the MOP range\n- Expand the date range\n- Check that selected instruments have data in the specified location")
else:
    st.info("👈 Configure your filters in the sidebar and click **Search Surveys** to begin")
