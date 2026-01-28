#!/usr/bin/env python3
"""
Event QC Tool - Streamlit GUI for manual quality control of erosion/deposition events

Usage:
    streamlit run event_qc.py
"""

import os
import platform
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from datetime import datetime
from collections import Counter

# Import local utilities
from utils.grid_loader import (
    DEFAULT_RESULTS_DIR,
    get_resolution_value,
    event_dates_to_folder,
    infer_location_from_filename,
    infer_event_type_from_filename,
    find_grid_file,
    load_and_prepare_grid,
    get_zoom_extent,
    load_event_csv,
    scan_event_csvs,
)

# === Constants ===
QC_FLAGS = ['unreviewed', 'real', 'construction', 'noise']
RESOLUTIONS = ['25cm', '10cm', '1m']


def get_custom_cmap(cmap_name: str = 'magma_r', vmax: float = 1.0):
    """
    Create colormap with white at zero.
    Matches existing dashboard pattern from cum_raw_vis.py.
    """
    base_cmap = cm.get_cmap(cmap_name, 256)
    newcolors = base_cmap(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 1])  # Pure white at 0
    cmap = LinearSegmentedColormap.from_list(f"White_{cmap_name}", newcolors)
    norm = Normalize(vmin=0, vmax=vmax)
    return cmap, norm


def plot_event_heatmap(grid_df: pd.DataFrame, event: pd.Series,
                       resolution_m: float, event_type: str = 'erosion') -> plt.Figure:
    """
    Create heatmap zoomed to event extent with centroid overlay.

    Args:
        grid_df: Prepared grid DataFrame (cleaned and snapped)
        event: Event row from events DataFrame
        resolution_m: Resolution in meters
        event_type: 'erosion' or 'deposition'

    Returns:
        Matplotlib Figure object
    """
    # Get zoom extent from event coordinates (in meters)
    extent = get_zoom_extent(event, resolution_m)

    # Transpose for display (elevation on Y-axis, alongshore on X-axis)
    # Grid structure: rows=alongshore position (m), cols=elevation (m)
    plot_df = grid_df.T

    # Get coordinate arrays (both already in meters after clean_and_snap_grid)
    x_values = plot_df.columns.values.astype(float)  # Alongshore (m)
    y_values = plot_df.index.values.astype(float)    # Elevation (m)

    # Get matrix values
    matrix = plot_df.values

    # Calculate full grid extent for imshow
    # Note: imshow extent is [left, right, bottom, top]
    x_min_grid, x_max_grid = x_values.min(), x_values.max()
    y_min_grid, y_max_grid = y_values.min(), y_values.max()
    img_extent = [x_min_grid, x_max_grid, y_min_grid, y_max_grid]

    # Choose colormap based on event type
    if event_type == 'erosion':
        cmap_name = 'magma_r'
        vmax = max(1.0, np.nanmax(matrix) * 1.1) if matrix.size > 0 else 1.0
    else:
        cmap_name = 'viridis_r'
        vmax = max(0.5, np.nanmax(matrix) * 1.1) if matrix.size > 0 else 0.5

    cmap, norm = get_custom_cmap(cmap_name, vmax=vmax)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot heatmap with full data
    im = ax.imshow(matrix, origin='lower', extent=img_extent,
                   cmap=cmap, norm=norm, aspect='auto',
                   interpolation='nearest')

    # Add event centroid crosshairs
    ax.axhline(event['elevation'], color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axvline(event['alongshore_centroid_m'], color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    # Add event bounding box
    ax.axvline(event['alongshore_start_m'], color='orange', linestyle=':', linewidth=1, alpha=0.6)
    ax.axvline(event['alongshore_end_m'], color='orange', linestyle=':', linewidth=1, alpha=0.6)

    # ZOOM: Set axis limits to event extent
    # Reverse x-axis for cliff-facing view (higher alongshore values on left)
    ax.set_xlim(extent['x_max'], extent['x_min'])
    ax.set_ylim(extent['y_min'], extent['y_max'])

    # Labels
    ax.set_xlabel("Alongshore Position (m)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Elevation (m)", fontsize=12, fontweight='bold')

    # Title with event info
    title = f"Event: Vol={event['volume']:.2f} m³, Elev={event['elevation']:.1f} m"
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    label = "Erosion Depth (m)" if event_type == 'erosion' else "Deposition Depth (m)"
    cbar.set_label(label, fontsize=11)

    plt.tight_layout()
    return fig


def get_progress_stats(events_df: pd.DataFrame, qc_flags: dict) -> dict:
    """Calculate QC progress statistics."""
    total = len(events_df)
    reviewed = sum(1 for v in qc_flags.values() if v != 'unreviewed')
    by_status = Counter(qc_flags.values())

    return {
        'total': total,
        'reviewed': reviewed,
        'unreviewed': total - reviewed,
        'real': by_status.get('real', 0),
        'construction': by_status.get('construction', 0),
        'noise': by_status.get('noise', 0),
    }


def export_qc_csv(events_df: pd.DataFrame, qc_flags: dict, original_path: str) -> tuple:
    """
    Prepare CSV export with QC flags column.

    Args:
        events_df: Original events DataFrame
        qc_flags: Dict mapping row index to QC flag
        original_path: Original CSV file path

    Returns:
        Tuple of (csv_string, suggested_filename)
    """
    # Create copy with QC column
    export_df = events_df.copy()
    export_df['qc_flag'] = export_df.index.map(lambda i: qc_flags.get(i, 'unreviewed'))

    # Generate filename
    base = os.path.basename(original_path)
    name, ext = os.path.splitext(base)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{name}_qc_{timestamp}{ext}"

    return export_df.to_csv(index=False), filename


# === Initialize Session State ===
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'events_df': None,
        'current_index': 0,
        'qc_flags': {},
        'event_type': 'erosion',
        'resolution': '25cm',
        'csv_path': None,
        'location': None,
        'results_dir': DEFAULT_RESULTS_DIR,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# === Streamlit App ===
st.set_page_config(page_title="Event QC Tool", layout="wide")
init_session_state()

st.title("Event QC Tool")
st.markdown("Manual quality control for erosion/deposition events")

# === Sidebar ===
st.sidebar.header("Configuration")

# Event CSV Selection
st.sidebar.subheader("1. Load Events")

csv_dir = st.sidebar.text_input(
    "Event CSV Directory",
    value=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                       "results", "event_lists_erosion"),
    help="Directory containing event CSV files"
)

if os.path.isdir(csv_dir):
    csv_files = scan_event_csvs(csv_dir)
    if csv_files:
        # Show relative paths from base directory for readability
        csv_display = {os.path.relpath(f, csv_dir): f for f in csv_files}
        selected_display = st.sidebar.selectbox("Select Event File", list(csv_display.keys()))

        if st.sidebar.button("Load Events", type="primary"):
            csv_path = csv_display[selected_display]
            events_df = load_event_csv(csv_path)

            if events_df is not None:
                st.session_state.events_df = events_df
                st.session_state.csv_path = csv_path
                st.session_state.current_index = 0
                st.session_state.location = infer_location_from_filename(csv_path)
                st.session_state.event_type = infer_event_type_from_filename(csv_path)

                # Initialize QC flags for all events as unreviewed
                st.session_state.qc_flags = {i: 'unreviewed' for i in range(len(events_df))}

                st.sidebar.success(f"Loaded {len(events_df)} events")
            else:
                st.sidebar.error("Failed to load CSV file")
    else:
        st.sidebar.warning("No CSV files found in directory")
else:
    st.sidebar.warning("Directory does not exist")

# Settings
if st.session_state.events_df is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Settings")

    st.session_state.resolution = st.sidebar.selectbox(
        "Resolution",
        options=RESOLUTIONS,
        index=RESOLUTIONS.index(st.session_state.resolution)
    )

    st.session_state.event_type = st.sidebar.selectbox(
        "Event Type",
        options=['erosion', 'deposition'],
        index=0 if st.session_state.event_type == 'erosion' else 1
    )

    st.session_state.results_dir = st.sidebar.text_input(
        "Results Directory",
        value=st.session_state.results_dir,
        help="Directory containing grid files"
    )

    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Navigation")

    total_events = len(st.session_state.events_df)

    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    with col1:
        if st.button("< Prev"):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
    with col2:
        new_index = st.number_input(
            "Event #",
            min_value=0,
            max_value=total_events - 1,
            value=st.session_state.current_index,
            label_visibility="collapsed"
        )
        if new_index != st.session_state.current_index:
            st.session_state.current_index = new_index
    with col3:
        if st.button("Next >"):
            st.session_state.current_index = min(total_events - 1, st.session_state.current_index + 1)

    st.sidebar.markdown(f"**Event {st.session_state.current_index + 1} / {total_events}**")

    # Jump to unreviewed
    if st.sidebar.button("Jump to Next Unreviewed"):
        for i in range(st.session_state.current_index + 1, total_events):
            if st.session_state.qc_flags.get(i, 'unreviewed') == 'unreviewed':
                st.session_state.current_index = i
                break
        else:
            # Wrap around to start
            for i in range(0, st.session_state.current_index):
                if st.session_state.qc_flags.get(i, 'unreviewed') == 'unreviewed':
                    st.session_state.current_index = i
                    break
            else:
                st.sidebar.info("All events reviewed!")

    # Progress
    st.sidebar.markdown("---")
    st.sidebar.subheader("4. Progress")

    stats = get_progress_stats(st.session_state.events_df, st.session_state.qc_flags)
    st.sidebar.progress(stats['reviewed'] / stats['total'] if stats['total'] > 0 else 0)
    st.sidebar.markdown(f"**Reviewed:** {stats['reviewed']} / {stats['total']}")
    st.sidebar.markdown(f"- Real: {stats['real']}")
    st.sidebar.markdown(f"- Construction: {stats['construction']}")
    st.sidebar.markdown(f"- Noise: {stats['noise']}")

    # Export
    st.sidebar.markdown("---")
    st.sidebar.subheader("5. Export")

    csv_data, filename = export_qc_csv(
        st.session_state.events_df,
        st.session_state.qc_flags,
        st.session_state.csv_path
    )

    st.sidebar.download_button(
        label="Download QC Results",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

# === Main Content ===
if st.session_state.events_df is not None:
    current_idx = st.session_state.current_index
    event = st.session_state.events_df.iloc[current_idx]

    # Event header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Event #{current_idx + 1}")
    with col2:
        current_flag = st.session_state.qc_flags.get(current_idx, 'unreviewed')
        flag_colors = {
            'unreviewed': 'gray',
            'real': 'green',
            'construction': 'orange',
            'noise': 'red'
        }
        st.markdown(f"**Status:** :{flag_colors[current_flag]}[{current_flag.upper()}]")

    # Heatmap
    st.markdown("---")

    # Try to load grid
    date_folder = event_dates_to_folder(event['start_date'], event['end_date'])
    grid_path = find_grid_file(
        st.session_state.results_dir,
        st.session_state.location,
        st.session_state.event_type,
        date_folder,
        st.session_state.resolution
    )

    if grid_path:
        resolution_m = get_resolution_value(st.session_state.resolution)
        grid_df = load_and_prepare_grid(grid_path, resolution_m)

        if grid_df is not None:
            fig = plot_event_heatmap(
                grid_df, event, resolution_m, st.session_state.event_type
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning(f"Could not load grid from: {grid_path}")
    else:
        st.warning(f"Grid file not found for: {st.session_state.location}/{st.session_state.event_type}/{date_folder}/{st.session_state.resolution}")
        st.info("Event details are shown below. You can still assign QC flags.")

    # Event Details
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Temporal**")
        st.markdown(f"- Start: {event['start_date']}")
        st.markdown(f"- End: {event['end_date']}")
        st.markdown(f"- Month: {int(event['month'])}")

    with col2:
        st.markdown("**Spatial**")
        st.markdown(f"- Alongshore: {event['alongshore_centroid_m']:.1f} m")
        st.markdown(f"- Elevation: {event['elevation']:.2f} m")
        st.markdown(f"- Width: {event['width']:.1f} m")
        st.markdown(f"- Height: {event['height']:.1f} m")

    with col3:
        st.markdown("**Volume**")
        st.markdown(f"- Volume: {event['volume']:.3f} m³")
        st.markdown(f"- Uncertainty: {event['vol_unc']:.4f} m³")
        if event['vol_unc'] > 0:
            snr = event['volume'] / event['vol_unc']
            st.markdown(f"- S/N Ratio: {snr:.1f}")

    # QC Buttons
    st.markdown("---")
    st.markdown("**Assign QC Flag:**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Real", type="primary" if current_flag == 'real' else "secondary",
                     use_container_width=True):
            st.session_state.qc_flags[current_idx] = 'real'
            # Auto-advance to next
            if current_idx < len(st.session_state.events_df) - 1:
                st.session_state.current_index += 1
            st.rerun()

    with col2:
        if st.button("Construction", type="primary" if current_flag == 'construction' else "secondary",
                     use_container_width=True):
            st.session_state.qc_flags[current_idx] = 'construction'
            if current_idx < len(st.session_state.events_df) - 1:
                st.session_state.current_index += 1
            st.rerun()

    with col3:
        if st.button("Noise", type="primary" if current_flag == 'noise' else "secondary",
                     use_container_width=True):
            st.session_state.qc_flags[current_idx] = 'noise'
            if current_idx < len(st.session_state.events_df) - 1:
                st.session_state.current_index += 1
            st.rerun()

    with col4:
        if st.button("Clear Flag", use_container_width=True):
            st.session_state.qc_flags[current_idx] = 'unreviewed'
            st.rerun()

else:
    st.info("Load an event CSV file from the sidebar to begin QC review.")
    st.markdown("""
    **Instructions:**
    1. Enter the path to your event CSV directory in the sidebar
    2. Select an event file and click "Load Events"
    3. Configure the resolution and results directory if needed
    4. Navigate through events using Prev/Next buttons
    5. Assign QC flags: Real, Construction, or Noise
    6. Export results when finished
    """)
