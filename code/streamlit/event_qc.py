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
    # NPZ cube utilities
    find_npz_for_csv,
    load_npz_cube,
    extract_grid_slice_from_cube,
    extract_both_slices_from_cube,
)

# === Constants ===
QC_FLAGS = ['unreviewed', 'real', 'construction', 'noise', 'needs_check']
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


def get_diverging_cmap(vmax: float = 1.0):
    """
    Create a diverging colormap: blue for deposition (negative), white at zero,
    orange/red for erosion (positive).
    """
    # Use RdBu_r: red for positive (erosion), blue for negative (deposition)
    cmap = cm.get_cmap('RdBu_r', 256)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    return cmap, norm


def plot_event_heatmap(erosion_df: pd.DataFrame, deposition_df: pd.DataFrame,
                       event: pd.Series, resolution_m: float) -> plt.Figure:
    """
    Create heatmap showing both erosion and deposition data with diverging colormap.

    Args:
        erosion_df: Erosion grid DataFrame (rows=alongshore, cols=elevation)
        deposition_df: Deposition grid DataFrame (same structure)
        event: Event row from events DataFrame
        resolution_m: Resolution in meters

    Returns:
        Matplotlib Figure object
    """
    # Get zoom extent from event coordinates (in meters)
    # Reduced horizontal padding, more padding below (to see beach), less above
    extent_m = get_zoom_extent(event, resolution_m, x_pad_m=10, y_pad_bottom_m=8, y_pad_top_m=3)

    # Use erosion_df for coordinates (both should have same structure)
    grid_df = erosion_df if erosion_df is not None else deposition_df
    if grid_df is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14)
        return fig

    alongshore_m = grid_df.index.values.astype(float)
    elevation_m = grid_df.columns.values.astype(float)

    # Get erosion matrix (positive values = warm colors)
    if erosion_df is not None:
        erosion_matrix = erosion_df.values.T  # Transpose: (n_elev, n_along)
        erosion_matrix = np.nan_to_num(erosion_matrix, nan=0.0)
    else:
        erosion_matrix = np.zeros((len(elevation_m), len(alongshore_m)))

    # Get deposition matrix (will be shown as negative = cool colors)
    if deposition_df is not None:
        deposition_matrix = deposition_df.values.T  # Transpose: (n_elev, n_along)
        deposition_matrix = np.nan_to_num(deposition_matrix, nan=0.0)
    else:
        deposition_matrix = np.zeros((len(elevation_m), len(alongshore_m)))

    # Combine: erosion positive, deposition negative
    # Where both exist, erosion takes precedence (shows the dominant signal)
    combined_matrix = erosion_matrix.copy()
    # Only add deposition (as negative) where erosion is zero/small
    dep_mask = (deposition_matrix > 0) & (erosion_matrix < 0.01)
    combined_matrix[dep_mask] = -deposition_matrix[dep_mask]

    n_elev, n_along = combined_matrix.shape

    # Find indices for the zoom extent
    x_mask = (alongshore_m >= extent_m['x_min']) & (alongshore_m <= extent_m['x_max'])
    y_mask = (elevation_m >= extent_m['y_min']) & (elevation_m <= extent_m['y_max'])

    if not np.any(x_mask) or not np.any(y_mask):
        x_idx_min, x_idx_max = 0, n_along
        y_idx_min, y_idx_max = 0, n_elev
    else:
        x_indices = np.where(x_mask)[0]
        y_indices = np.where(y_mask)[0]
        x_idx_min, x_idx_max = x_indices.min(), x_indices.max() + 1
        y_idx_min, y_idx_max = y_indices.min(), y_indices.max() + 1

    # Extract the zoomed region
    matrix_zoom = combined_matrix[y_idx_min:y_idx_max, x_idx_min:x_idx_max]
    alongshore_zoom = alongshore_m[x_idx_min:x_idx_max]
    elevation_zoom = elevation_m[y_idx_min:y_idx_max]

    # Determine color scale (symmetric around 0, capped at 5m)
    cmap, norm = get_diverging_cmap(vmax=5.0)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot using array indices
    n_y, n_x = matrix_zoom.shape
    im = ax.imshow(matrix_zoom, origin='lower', aspect='auto', interpolation='nearest',
                   cmap=cmap, norm=norm)

    # Set up axis labels in meters
    n_xticks = min(6, n_x) if n_x > 0 else 1
    if n_x > 1:
        xtick_idx = np.linspace(0, n_x - 1, n_xticks, dtype=int)
        ax.set_xticks(xtick_idx)
        ax.set_xticklabels([f'{alongshore_zoom[i]:.0f}' for i in xtick_idx])

    n_yticks = min(6, n_y) if n_y > 0 else 1
    if n_y > 1:
        ytick_idx = np.linspace(0, n_y - 1, n_yticks, dtype=int)
        ax.set_yticks(ytick_idx)
        ax.set_yticklabels([f'{elevation_zoom[i]:.1f}' for i in ytick_idx])

    # Reverse x-axis for cliff-facing view
    ax.invert_xaxis()

    # Add event centroid crosshairs (lighter style)
    if len(alongshore_zoom) > 0 and len(elevation_zoom) > 0:
        x_crosshair = np.argmin(np.abs(alongshore_zoom - event['alongshore_centroid_m']))
        y_crosshair = np.argmin(np.abs(elevation_zoom - event['elevation']))
        ax.axhline(y_crosshair, color='#666666', linestyle='--', linewidth=1.0, alpha=0.5)
        ax.axvline(x_crosshair, color='#666666', linestyle='--', linewidth=1.0, alpha=0.5)

        # Add bounding box lines
        x_start_idx = np.argmin(np.abs(alongshore_zoom - event['alongshore_start_m']))
        x_end_idx = np.argmin(np.abs(alongshore_zoom - event['alongshore_end_m']))
        ax.axvline(x_start_idx, color='#999999', linestyle=':', linewidth=1, alpha=0.4)
        ax.axvline(x_end_idx, color='#999999', linestyle=':', linewidth=1, alpha=0.4)

    # Labels
    ax.set_xlabel("Alongshore Position (m)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Elevation (m)", fontsize=12, fontweight='bold')

    # Title with event info
    title = f"Event: {event['start_date']} to {event['end_date']} | Vol={event['volume']:.2f} m³, Elev={event['elevation']:.1f} m"
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Colorbar with diverging labels
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("← Deposition | Erosion →  (m)", fontsize=11)

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
        'needs_check': by_status.get('needs_check', 0),
    }


def get_qc_output_path(original_path: str) -> str:
    """
    Determine the QC output path based on the original CSV path.

    Maps: results/event_lists/<subdir>/file.csv -> results/event_lists_qc/<subdir>/file_qc_<timestamp>.csv

    Args:
        original_path: Original CSV file path

    Returns:
        Full output path for QC results
    """
    # Get the subdirectory (erosion, deposition, or combined)
    parent_dir = os.path.basename(os.path.dirname(original_path))
    if parent_dir in ('erosion', 'deposition', 'combined'):
        subdir = parent_dir
    else:
        # Fallback: try to infer from filename or default to combined
        basename_lower = os.path.basename(original_path).lower()
        if 'dep' in basename_lower:
            subdir = 'deposition'
        elif 'ero' in basename_lower:
            subdir = 'erosion'
        else:
            subdir = 'combined'

    # Find results directory by walking up from original path
    current = os.path.dirname(original_path)
    while current:
        if os.path.basename(current) == 'event_lists':
            results_dir = os.path.dirname(current)
            break
        parent = os.path.dirname(current)
        if parent == current:  # Reached root
            # Fallback: use grandparent of CSV directory
            results_dir = os.path.dirname(os.path.dirname(os.path.dirname(original_path)))
            break
        current = parent

    # Build output directory
    output_dir = os.path.join(results_dir, 'event_lists_qc', subdir)

    # Generate filename with timestamp
    base = os.path.basename(original_path)
    name, ext = os.path.splitext(base)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{name}_qc_{timestamp}{ext}"

    return os.path.join(output_dir, filename)


def export_qc_csv(events_df: pd.DataFrame, qc_flags: dict, original_path: str) -> str:
    """
    Save CSV with QC flags column to results/event_lists_qc/<subdir>/.

    Args:
        events_df: Original events DataFrame
        qc_flags: Dict mapping row index to QC flag
        original_path: Original CSV file path

    Returns:
        Path where file was saved
    """
    # Create copy with QC column
    export_df = events_df.copy()
    export_df['qc_flag'] = export_df.index.map(lambda i: qc_flags.get(i, 'unreviewed'))

    # Get output path
    output_path = get_qc_output_path(original_path)

    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to disk
    export_df.to_csv(output_path, index=False)

    return output_path


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
        'npz_data': None,  # Loaded NPZ cube data
        'npz_path': None,  # Path to NPZ file (for display)
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
                       "results", "event_lists", "erosion"),
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
                # Sort by volume (largest first) if volume column exists
                if 'volume' in events_df.columns:
                    events_df = events_df.sort_values('volume', ascending=False).reset_index(drop=True)

                st.session_state.events_df = events_df
                st.session_state.csv_path = csv_path
                st.session_state.location = infer_location_from_filename(csv_path)
                st.session_state.event_type = infer_event_type_from_filename(csv_path)

                # Check if CSV has existing QC flags (resuming previous work)
                resumed = False
                if 'qc_flag' in events_df.columns:
                    # Load existing flags
                    st.session_state.qc_flags = {
                        i: row['qc_flag'] if pd.notna(row['qc_flag']) else 'unreviewed'
                        for i, row in events_df.iterrows()
                    }
                    # Find first unreviewed event
                    first_unreviewed = 0
                    for i in range(len(events_df)):
                        if st.session_state.qc_flags.get(i, 'unreviewed') == 'unreviewed':
                            first_unreviewed = i
                            break
                    st.session_state.current_index = first_unreviewed
                    resumed = True
                else:
                    # Initialize all as unreviewed, start at beginning
                    st.session_state.qc_flags = {i: 'unreviewed' for i in range(len(events_df))}
                    st.session_state.current_index = 0

                # Try to load corresponding NPZ cube
                npz_path = find_npz_for_csv(csv_path)
                if npz_path:
                    npz_data = load_npz_cube(npz_path)
                    st.session_state.npz_data = npz_data
                    st.session_state.npz_path = npz_path
                    msg = f"Loaded {len(events_df)} events + NPZ cube"
                else:
                    st.session_state.npz_data = None
                    st.session_state.npz_path = None
                    msg = f"Loaded {len(events_df)} events (no NPZ found)"

                if resumed:
                    reviewed = sum(1 for f in st.session_state.qc_flags.values() if f != 'unreviewed')
                    msg += f" | Resumed: {reviewed} already reviewed"
                st.sidebar.success(msg)
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

    # Show data source info
    if st.session_state.npz_data is not None:
        st.sidebar.info(f"Using NPZ cube")
        npz_name = os.path.basename(st.session_state.npz_path) if st.session_state.npz_path else "unknown"
        st.sidebar.caption(f"File: {npz_name}")
    else:
        st.sidebar.warning("No NPZ cube - using grid CSVs")

    st.session_state.event_type = st.sidebar.selectbox(
        "Event Type",
        options=['erosion', 'deposition'],
        index=0 if st.session_state.event_type == 'erosion' else 1
    )

    # Only show resolution and results_dir if not using NPZ (they're not needed with NPZ)
    if st.session_state.npz_data is None:
        st.session_state.resolution = st.sidebar.selectbox(
            "Resolution",
            options=RESOLUTIONS,
            index=RESOLUTIONS.index(st.session_state.resolution)
        )

        st.session_state.results_dir = st.sidebar.text_input(
            "Results Directory",
            value=st.session_state.results_dir,
            help="Directory containing grid CSV files (only used when no NPZ cube)"
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

    # Jump to needs check
    if st.sidebar.button("Jump to Next Needs Check"):
        for i in range(st.session_state.current_index + 1, total_events):
            if st.session_state.qc_flags.get(i, 'unreviewed') == 'needs_check':
                st.session_state.current_index = i
                break
        else:
            # Wrap around to start
            for i in range(0, st.session_state.current_index):
                if st.session_state.qc_flags.get(i, 'unreviewed') == 'needs_check':
                    st.session_state.current_index = i
                    break
            else:
                st.sidebar.info("No events flagged for manual check")

    # Progress
    st.sidebar.markdown("---")
    st.sidebar.subheader("4. Progress")

    stats = get_progress_stats(st.session_state.events_df, st.session_state.qc_flags)
    st.sidebar.progress(stats['reviewed'] / stats['total'] if stats['total'] > 0 else 0)
    st.sidebar.markdown(f"**Reviewed:** {stats['reviewed']} / {stats['total']}")
    st.sidebar.markdown(f"- Real: {stats['real']}")
    st.sidebar.markdown(f"- Construction: {stats['construction']}")
    st.sidebar.markdown(f"- Noise: {stats['noise']}")
    st.sidebar.markdown(f"- Needs Check: {stats['needs_check']}")

    # Export
    st.sidebar.markdown("---")
    st.sidebar.subheader("5. Export")

    # Show where file will be saved
    output_path = get_qc_output_path(st.session_state.csv_path)
    st.sidebar.caption(f"Saves to: `{os.path.relpath(output_path, os.path.dirname(st.session_state.csv_path))}`")

    if st.sidebar.button("Save QC Results", use_container_width=True):
        saved_path = export_qc_csv(
            st.session_state.events_df,
            st.session_state.qc_flags,
            st.session_state.csv_path
        )
        st.sidebar.success(f"Saved to {os.path.basename(saved_path)}")

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
            'noise': 'red',
            'needs_check': 'violet'
        }
        st.markdown(f"**Status:** :{flag_colors[current_flag]}[{current_flag.upper()}]")

    # Heatmap
    st.markdown("---")

    # Try to get grid data - prefer NPZ cube, fallback to individual CSV files
    erosion_df = None
    deposition_df = None
    data_source = None
    resolution_m = get_resolution_value(st.session_state.resolution)

    # Method 1: Try NPZ cube first (faster, pre-loaded) - get BOTH erosion and deposition
    if st.session_state.npz_data is not None:
        erosion_df, deposition_df, _, _ = extract_both_slices_from_cube(
            st.session_state.npz_data,
            event
        )
        if erosion_df is not None or deposition_df is not None:
            data_source = "NPZ cube"

    # Method 2: Fallback to individual grid CSV files (erosion only for now)
    if erosion_df is None and deposition_df is None:
        date_folder = event_dates_to_folder(event['start_date'], event['end_date'])
        grid_path = find_grid_file(
            st.session_state.results_dir,
            st.session_state.location,
            'erosion',
            date_folder,
            st.session_state.resolution
        )

        if grid_path:
            erosion_df = load_and_prepare_grid(grid_path, resolution_m)
            if erosion_df is not None:
                data_source = "grid CSV"

        # Try deposition CSV too
        dep_grid_path = find_grid_file(
            st.session_state.results_dir,
            st.session_state.location,
            'deposition',
            date_folder,
            st.session_state.resolution
        )
        if dep_grid_path:
            deposition_df = load_and_prepare_grid(dep_grid_path, resolution_m)

    # Display heatmap or warning
    if erosion_df is not None or deposition_df is not None:
        fig = plot_event_heatmap(
            erosion_df, deposition_df, event, resolution_m
        )
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Data source: {data_source} | Red/Orange = Erosion, Blue = Deposition")
    else:
        date_folder = event_dates_to_folder(event['start_date'], event['end_date'])
        st.warning(f"Grid data not found for: {st.session_state.location}/{date_folder}")
        if st.session_state.npz_path:
            st.info(f"NPZ cube loaded from: {st.session_state.npz_path}")
            st.info("Date range may not match cube contents.")
        else:
            st.info("No NPZ cube found. Configure Results Directory to load individual grid CSVs.")

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

    col1, col2, col3, col4, col5 = st.columns(5)

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
        if st.button("Needs Check", type="primary" if current_flag == 'needs_check' else "secondary",
                     use_container_width=True):
            st.session_state.qc_flags[current_idx] = 'needs_check'
            if current_idx < len(st.session_state.events_df) - 1:
                st.session_state.current_index += 1
            st.rerun()

    with col5:
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
    5. Assign QC flags: Real, Construction, Noise, or Needs Check
    6. Use "Jump to Next Needs Check" to revisit flagged events
    7. Export results when finished
    """)
