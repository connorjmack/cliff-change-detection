#!/usr/bin/env python3
"""
Survey Browser - Streamlit GUI for filtering LiDAR surveys by MOP range and date

Usage:
    streamlit run survey_browser.py
"""

import os
import platform
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

def scan_surveys(min_mop, max_mop, start_date, end_date, selected_instruments):
    """
    Scan all instrument folders and return surveys that match the criteria.

    Parameters:
    - min_mop, max_mop: MOP line range
    - start_date, end_date: datetime objects for date range
    - selected_instruments: list of instrument names to include

    Returns:
    - DataFrame with columns: date, MOP1, MOP2, method, folder_name, full_path
    """
    rows = []

    # Convert dates to YYYYMMDD format for comparison
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    # Calculate 2/3 overlap threshold
    two_thirds = np.floor((max_mop - min_mop) * 2/3)

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

            # Check MOP overlap (2/3 threshold)
            overlap = min(mop2, max_mop) - max(mop1, min_mop)
            if overlap < two_thirds:
                continue

            # Add to results
            rows.append({
                "date": date_str,
                "MOP1": mop1,
                "MOP2": mop2,
                "method": method,
                "folder_name": name,
                "full_path": subdir
            })

    # Convert to DataFrame and sort by date
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values("date").reset_index(drop=True)
        return df
    else:
        return pd.DataFrame(columns=["date", "MOP1", "MOP2", "method", "folder_name", "full_path"])

# === Streamlit App ===
st.set_page_config(page_title="LiDAR Survey Browser", layout="wide")

st.title("🌊 LiDAR Survey Browser")
st.markdown("Filter surveys by MOP range, date range, and instrument type")

# Sidebar controls
st.sidebar.header("Filter Options")

# MOP Range
st.sidebar.subheader("MOP Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    min_mop = st.number_input("Min MOP", min_value=0, max_value=1000, value=520, step=1)
with col2:
    max_mop = st.number_input("Max MOP", min_value=0, max_value=1000, value=764, step=1)

# Validate MOP range
if min_mop >= max_mop:
    st.sidebar.error("Min MOP must be less than Max MOP")
    st.stop()

st.sidebar.markdown(f"**Selected Range:** MOP {min_mop} - {max_mop} (length: {max_mop - min_mop})")
st.sidebar.markdown(f"*Surveys need ≥{int(np.floor((max_mop - min_mop) * 2/3))} MOP lines overlap (2/3 threshold)*")

# Date Range
st.sidebar.subheader("Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime(2010, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime.now())

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
    with st.spinner("Scanning survey directories..."):
        results_df = scan_surveys(min_mop, max_mop, start_date, end_date, selected_instruments)

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
                "full_path": st.column_config.TextColumn("Full Path", width="large")
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
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"survey_results_MOP{min_mop}-{max_mop}_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("⚠️ No surveys found matching your criteria. Try adjusting the filters.")
        st.info("**Tips:**\n- Expand the MOP range\n- Expand the date range\n- Check that selected instruments have data in the specified location")
else:
    st.info("👈 Configure your filters in the sidebar and click **Search Surveys** to begin")

    # Show predefined location examples
    st.markdown("---")
    st.subheader("📍 Quick Start: Predefined Locations")

    locations = {
        "Del Mar": [595, 620],
        "Solana": [637, 666],
        "Encinitas": [708, 764],
        "San Elijo": [683, 708],
        "Torrey": [567, 581],
        "Blacks": [520, 567]
    }

    cols = st.columns(3)
    for idx, (location, mop_range) in enumerate(locations.items()):
        with cols[idx % 3]:
            st.markdown(f"**{location}**")
            st.markdown(f"MOP {mop_range[0]} - {mop_range[1]}")
