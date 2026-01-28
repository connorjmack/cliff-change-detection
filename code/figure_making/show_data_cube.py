#!/usr/bin/env python3
"""
show_data_cube.py

Visualizes the structure and metadata of an NPZ data cube file.
Draws a 3D cube diagram with labeled dimensions and lists all arrays/metadata.

Usage:
    python3 show_data_cube.py path/to/cube.npz
    python3 show_data_cube.py path/to/cube.npz --save output.png
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_npz_info(npz_path):
    """
    Load NPZ file and extract metadata about all arrays.

    Returns:
        dict with keys:
            - '3d_arrays': dict of {name: (shape, dtype, non_nan_count)}
            - '1d_arrays': dict of {name: (shape, dtype, sample_values)}
            - 'scalars': dict of {name: value}
            - 'filename': basename of the file
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    info = {
        '3d_arrays': {},
        '2d_arrays': {},
        '1d_arrays': {},
        'scalars': {},
        'filename': os.path.basename(npz_path)
    }

    for key in data.files:
        arr = data[key]

        # Handle scalar values (0-d arrays or single values)
        if arr.ndim == 0:
            info['scalars'][key] = arr.item()
        elif arr.ndim == 1:
            # Sample first/last values for display
            if len(arr) > 0:
                if np.issubdtype(arr.dtype, np.number):
                    sample = f"[{arr[0]:.2f} ... {arr[-1]:.2f}]"
                else:
                    sample = f"[{arr[0]} ... {arr[-1]}]"
            else:
                sample = "[]"
            info['1d_arrays'][key] = {
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'sample': sample,
                'length': len(arr)
            }
        elif arr.ndim == 2:
            non_nan = np.sum(~np.isnan(arr)) if np.issubdtype(arr.dtype, np.floating) else arr.size
            info['2d_arrays'][key] = {
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'non_nan': non_nan
            }
        elif arr.ndim == 3:
            non_nan = np.sum(~np.isnan(arr)) if np.issubdtype(arr.dtype, np.floating) else arr.size
            info['3d_arrays'][key] = {
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'non_nan': non_nan
            }

    data.close()
    return info


def draw_cube(ax, origin, size, color='steelblue', alpha=0.3, edge_color='black'):
    """Draw a 3D rectangular prism (cube) on the given axes."""
    x, y, z = origin
    dx, dy, dz = size

    # Define the 8 vertices
    vertices = [
        [x, y, z],
        [x + dx, y, z],
        [x + dx, y + dy, z],
        [x, y + dy, z],
        [x, y, z + dz],
        [x + dx, y, z + dz],
        [x + dx, y + dy, z + dz],
        [x, y + dy, z + dz]
    ]

    # Define the 6 faces
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]]   # top
    ]

    # Create collection and add to axes
    collection = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                                   edgecolor=edge_color, linewidth=1)
    ax.add_collection3d(collection)

    return vertices


def add_dimension_labels(ax, vertices, shape, dim_names):
    """Add dimension labels with arrows to the cube."""
    # Dimension 0: along x-axis (alongshore)
    mid_x = (vertices[0][0] + vertices[1][0]) / 2
    ax.text(mid_x, vertices[0][1] - 0.5, vertices[0][2] - 0.3,
            f"{dim_names[0]}\n({shape[0]})",
            fontsize=11, ha='center', va='top', fontweight='bold', color='#2E86AB')

    # Dimension 1: along y-axis (elevation)
    mid_y = (vertices[0][1] + vertices[3][1]) / 2
    ax.text(vertices[0][0] - 0.5, mid_y, vertices[0][2] - 0.3,
            f"{dim_names[1]}\n({shape[1]})",
            fontsize=11, ha='right', va='center', fontweight='bold', color='#06D6A0')

    # Dimension 2: along z-axis (time)
    mid_z = (vertices[0][2] + vertices[4][2]) / 2
    ax.text(vertices[0][0] - 0.5, vertices[0][1] - 0.5, mid_z,
            f"{dim_names[2]}\n({shape[2]})",
            fontsize=11, ha='right', va='center', fontweight='bold', color='#8338EC')


def create_visualization(info, output_path=None):
    """Create the full visualization figure."""
    fig = plt.figure(figsize=(14, 10))

    # Create grid: 3D cube on left, info panels on right
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1],
                          wspace=0.3, hspace=0.3)

    # 3D Cube subplot
    ax_cube = fig.add_subplot(gs[:, 0], projection='3d')

    # Info text subplots
    ax_arrays = fig.add_subplot(gs[0, 1])
    ax_meta = fig.add_subplot(gs[1, 1])

    # Turn off axes for text panels
    ax_arrays.axis('off')
    ax_meta.axis('off')

    # --- Draw 3D Cube ---
    # Find the primary 3D array to visualize
    if info['3d_arrays']:
        # Prefer 'erosion' if available, otherwise use first 3D array
        primary_key = 'erosion' if 'erosion' in info['3d_arrays'] else list(info['3d_arrays'].keys())[0]
        shape = info['3d_arrays'][primary_key]['shape']

        # Normalize dimensions for visualization (scale to reasonable size)
        max_dim = max(shape)
        normalized = [s / max_dim * 3 for s in shape]

        # Draw the cube
        vertices = draw_cube(ax_cube, (0, 0, 0), normalized,
                            color='steelblue', alpha=0.25)

        # Add dimension labels
        # Infer dimension names from 1D arrays if available
        dim_names = ['alongshore', 'elevation', 'time']  # defaults
        if 'alongshore_m' in info['1d_arrays']:
            dim_names[0] = 'alongshore_m'
        if 'elevation_m' in info['1d_arrays']:
            dim_names[1] = 'elevation_m'
        if 'dates' in info['1d_arrays'] or 'date_strings' in info['1d_arrays']:
            dim_names[2] = 'time'

        add_dimension_labels(ax_cube, vertices, shape, dim_names)

        # Set axis properties
        ax_cube.set_xlim(-1, normalized[0] + 1)
        ax_cube.set_ylim(-1, normalized[1] + 1)
        ax_cube.set_zlim(-1, normalized[2] + 1)
        ax_cube.set_xlabel('')
        ax_cube.set_ylabel('')
        ax_cube.set_zlabel('')
        ax_cube.set_xticks([])
        ax_cube.set_yticks([])
        ax_cube.set_zticks([])

        # Nice viewing angle
        ax_cube.view_init(elev=20, azim=45)

        # Add shape annotation
        shape_str = f"Shape: {shape[0]} × {shape[1]} × {shape[2]}"
        ax_cube.text2D(0.5, 0.95, shape_str, transform=ax_cube.transAxes,
                      fontsize=14, ha='center', fontweight='bold')
    else:
        ax_cube.text(0.5, 0.5, 0.5, "No 3D arrays found", ha='center', va='center')

    # --- Arrays Panel ---
    array_text = "ARRAYS IN FILE\n" + "=" * 30 + "\n\n"

    if info['3d_arrays']:
        array_text += "3D Arrays (Cubes):\n"
        for name, data in info['3d_arrays'].items():
            non_nan_pct = data['non_nan'] / np.prod(data['shape']) * 100
            array_text += f"  • {name}\n"
            array_text += f"      shape: {data['shape']}\n"
            array_text += f"      dtype: {data['dtype']}\n"
            array_text += f"      valid: {data['non_nan']:,} ({non_nan_pct:.1f}%)\n"
        array_text += "\n"

    if info['2d_arrays']:
        array_text += "2D Arrays:\n"
        for name, data in info['2d_arrays'].items():
            array_text += f"  • {name}: {data['shape']} ({data['dtype']})\n"
        array_text += "\n"

    if info['1d_arrays']:
        array_text += "1D Arrays (Axes):\n"
        for name, data in info['1d_arrays'].items():
            array_text += f"  • {name}\n"
            array_text += f"      length: {data['length']}\n"
            array_text += f"      dtype: {data['dtype']}\n"
            array_text += f"      range: {data['sample']}\n"

    ax_arrays.text(0.05, 0.95, array_text, transform=ax_arrays.transAxes,
                   fontsize=10, family='monospace', va='top',
                   bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    # --- Metadata Panel ---
    meta_text = "METADATA\n" + "=" * 30 + "\n\n"
    meta_text += f"File: {info['filename']}\n\n"

    if info['scalars']:
        meta_text += "Scalar Values:\n"
        for name, value in info['scalars'].items():
            if isinstance(value, float):
                meta_text += f"  • {name}: {value:.4f}\n"
            else:
                meta_text += f"  • {name}: {value}\n"
        meta_text += "\n"

    # Add computed stats
    meta_text += "Computed Statistics:\n"
    total_elements = 0
    total_valid = 0
    for data in info['3d_arrays'].values():
        total_elements += np.prod(data['shape'])
        total_valid += data['non_nan']

    if total_elements > 0:
        meta_text += f"  • Total 3D elements: {total_elements:,}\n"
        meta_text += f"  • Valid (non-NaN): {total_valid:,}\n"
        meta_text += f"  • Sparsity: {(1 - total_valid/total_elements)*100:.1f}%\n"

    # If date arrays exist, show date range
    if 'date_strings' in info['1d_arrays']:
        meta_text += f"\nTemporal Range:\n"
        meta_text += f"  {info['1d_arrays']['date_strings']['sample']}\n"

    ax_meta.text(0.05, 0.95, meta_text, transform=ax_meta.transAxes,
                 fontsize=10, family='monospace', va='top',
                 bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8))

    # Title
    fig.suptitle(f"NPZ Data Cube Structure: {info['filename']}",
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()


def print_summary(info):
    """Print a text summary of the NPZ contents."""
    print(f"\n{'='*60}")
    print(f"NPZ FILE: {info['filename']}")
    print('='*60)

    if info['3d_arrays']:
        print("\n3D Arrays:")
        for name, data in info['3d_arrays'].items():
            print(f"  {name}: {data['shape']} ({data['dtype']})")
            print(f"    Non-NaN values: {data['non_nan']:,}")

    if info['2d_arrays']:
        print("\n2D Arrays:")
        for name, data in info['2d_arrays'].items():
            print(f"  {name}: {data['shape']} ({data['dtype']})")

    if info['1d_arrays']:
        print("\n1D Arrays:")
        for name, data in info['1d_arrays'].items():
            print(f"  {name}: length={data['length']} ({data['dtype']})")
            print(f"    Range: {data['sample']}")

    if info['scalars']:
        print("\nScalar Values:")
        for name, value in info['scalars'].items():
            print(f"  {name}: {value}")

    print('='*60 + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the structure and metadata of an NPZ data cube file."
    )
    parser.add_argument('npz_path', type=str,
                        help="Path to the NPZ file to visualize")
    parser.add_argument('--save', '-s', type=str, default=None,
                        help="Save visualization to file (e.g., output.png)")
    parser.add_argument('--no-plot', action='store_true',
                        help="Only print text summary, skip visualization")

    args = parser.parse_args()

    try:
        info = load_npz_info(args.npz_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Always print text summary
    print_summary(info)

    # Create visualization unless --no-plot
    if not args.no_plot:
        create_visualization(info, args.save)


if __name__ == "__main__":
    main()
