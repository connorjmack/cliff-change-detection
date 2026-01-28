#!/usr/bin/env python3
"""
make_sig_event_lists.py

Filters event list CSVs to keep only significant erosion events.

Significance criteria:
  - Volume > 5 m³
  - Elevation > 5 m (originated above 5m elevation)

Filters erosion/ and combined/ subdirectories only (not deposition/).
Outputs files with thresholds encoded in filename: <location>_vol_<V>_elv_<E>.csv

Usage:
    python3 make_sig_event_lists.py
    python3 make_sig_event_lists.py --min_volume 10 --min_elevation 3
    python3 make_sig_event_lists.py --input_dir /path/to/event_lists
"""
import os
import platform
import argparse
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_MIN_VOLUME = 5.0      # m³
DEFAULT_MIN_ELEVATION = 5.0   # m

ALL_LOCATIONS = ['DelMar', 'Solana', 'Encinitas', 'SanElijo', 'Torrey']


def get_base_dir():
    """Get base directory based on platform."""
    if platform.system() == "Darwin":
        return "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
    else:
        return "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"


# ============================================================================
# FILTERING
# ============================================================================

def filter_significant_events(df, min_volume, min_elevation):
    """
    Filter events to keep only significant ones.

    Args:
        df: DataFrame with event data
        min_volume: Minimum volume threshold (m³)
        min_elevation: Minimum elevation threshold (m)

    Returns:
        Filtered DataFrame
    """
    mask = (df['volume'] > min_volume) & (df['elevation'] > min_elevation)
    return df[mask].copy()


def make_output_filename(location, min_volume, min_elevation):
    """
    Generate output filename with threshold values encoded.

    Args:
        location: Location name (e.g., 'SanElijo')
        min_volume: Minimum volume threshold
        min_elevation: Minimum elevation threshold

    Returns:
        Filename like 'SanElijo_vol_5_elv_5.csv'
    """
    # Format as integers if whole numbers, otherwise keep decimals
    vol_str = str(int(min_volume)) if min_volume == int(min_volume) else str(min_volume)
    elv_str = str(int(min_elevation)) if min_elevation == int(min_elevation) else str(min_elevation)
    return f"{location}_vol_{vol_str}_elv_{elv_str}.csv"


def process_directory(subdir_path, min_volume, min_elevation):
    """
    Process all *_events.csv files in a directory.

    Args:
        subdir_path: Path to erosion/ or combined/ subdirectory
        min_volume: Minimum volume threshold
        min_elevation: Minimum elevation threshold

    Returns:
        Tuple of (total_events, significant_events) counts
    """
    total = 0
    significant = 0

    if not os.path.isdir(subdir_path):
        return total, significant

    for filename in os.listdir(subdir_path):
        if filename.endswith('_events.csv') and '_vol_' not in filename:
            input_path = os.path.join(subdir_path, filename)
            location = filename.replace('_events.csv', '')
            output_filename = make_output_filename(location, min_volume, min_elevation)
            output_path = os.path.join(subdir_path, output_filename)

            try:
                df = pd.read_csv(input_path)
                total += len(df)

                df_sig = filter_significant_events(df, min_volume, min_elevation)
                significant += len(df_sig)

                df_sig.to_csv(output_path, index=False)

                print(f"    {location}: {len(df)} -> {len(df_sig)} events ({output_filename})")

            except Exception as e:
                print(f"    [ERROR] {filename}: {e}")

    return total, significant


def main():
    parser = argparse.ArgumentParser(
        description='Filter event lists to keep only significant erosion events'
    )
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory (default: results/event_lists/)')
    parser.add_argument('--min_volume', type=float, default=DEFAULT_MIN_VOLUME,
                        help=f'Minimum volume threshold in m³ (default: {DEFAULT_MIN_VOLUME})')
    parser.add_argument('--min_elevation', type=float, default=DEFAULT_MIN_ELEVATION,
                        help=f'Minimum elevation threshold in m (default: {DEFAULT_MIN_ELEVATION})')
    args = parser.parse_args()

    base_dir = get_base_dir()

    if args.input_dir:
        input_dir = args.input_dir
    else:
        input_dir = os.path.join(base_dir, 'results', 'event_lists')

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory not found: {input_dir}")
        return 1

    print(f"\n{'='*60}")
    print(f"SIGNIFICANT EVENT FILTERING")
    print(f"Input: {input_dir}")
    print(f"Criteria:")
    print(f"  - Volume > {args.min_volume} m³")
    print(f"  - Elevation > {args.min_elevation} m")
    print(f"{'='*60}\n")

    total_all = 0
    sig_all = 0

    # Process erosion/
    erosion_dir = os.path.join(input_dir, 'erosion')
    if os.path.isdir(erosion_dir):
        print("Processing erosion/...")
        total, sig = process_directory(erosion_dir, args.min_volume, args.min_elevation)
        total_all += total
        sig_all += sig
    else:
        print(f"[WARNING] erosion/ directory not found: {erosion_dir}")

    # Process combined/
    combined_dir = os.path.join(input_dir, 'combined')
    if os.path.isdir(combined_dir):
        print("\nProcessing combined/...")
        total, sig = process_directory(combined_dir, args.min_volume, args.min_elevation)
        total_all += total
        sig_all += sig
    else:
        print(f"[WARNING] combined/ directory not found: {combined_dir}")

    print(f"\n{'='*60}")
    print(f"Summary: {sig_all} significant events from {total_all} total")
    print(f"{'='*60}\n")

    return 0


if __name__ == '__main__':
    exit(main())
