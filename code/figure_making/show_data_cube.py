#!/usr/bin/env python3
"""
show_data_cube.py

Visualizes the structure of an NPZ data cube file.

Usage:
    python3 show_data_cube.py path/to/cube.npz
    python3 show_data_cube.py path/to/cube.npz --save output.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_npz_info(npz_path):
    """Load NPZ file and extract metadata about all arrays."""
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

        if arr.ndim == 0:
            info['scalars'][key] = arr.item()
        elif arr.ndim == 1:
            if len(arr) > 0:
                if np.issubdtype(arr.dtype, np.number):
                    min_val, max_val = float(arr[0]), float(arr[-1])
                    sample = f"{min_val:.2f} to {max_val:.2f}"
                else:
                    min_val, max_val = str(arr[0]), str(arr[-1])
                    sample = f"{min_val} to {max_val}"
            else:
                sample = "empty"
                min_val, max_val = None, None
            info['1d_arrays'][key] = {
                'length': len(arr),
                'sample': sample,
                'min': min_val,
                'max': max_val
            }
        elif arr.ndim == 2:
            info['2d_arrays'][key] = {'shape': arr.shape}
        elif arr.ndim == 3:
            info['3d_arrays'][key] = {'shape': arr.shape}

    data.close()
    return info


def draw_cube(ax, origin, size, color='steelblue', alpha=0.4, edge_color='#333333'):
    """Draw a 3D rectangular prism on the given axes."""
    x, y, z = origin
    dx, dy, dz = size

    vertices = [
        [x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z],
        [x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]
    ]

    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]]
    ]

    collection = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                                   edgecolor=edge_color, linewidth=1.5)
    ax.add_collection3d(collection)
    return vertices


def create_visualization(info, output_path=None):
    """Create the visualization figure."""
    fig = plt.figure(figsize=(16, 9))

    # Large cube on left, compact info on right
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.05)

    ax_cube = fig.add_subplot(gs[0], projection='3d')
    ax_info = fig.add_subplot(gs[1])
    ax_info.axis('off')

    # --- Draw 3D Cube ---
    if info['3d_arrays']:
        primary_key = 'erosion' if 'erosion' in info['3d_arrays'] else list(info['3d_arrays'].keys())[0]
        shape = info['3d_arrays'][primary_key]['shape']

        # Scale cube to fill the space well, with proportional dimensions
        base_size = 8
        max_dim = max(shape)
        cube_dims = [s / max_dim * base_size for s in shape]

        # Ensure minimum visible size for small dimensions
        cube_dims = [max(d, base_size * 0.15) for d in cube_dims]

        vertices = draw_cube(ax_cube, (0, 0, 0), cube_dims, color='#4A90D9', alpha=0.35)

        # Dimension labels - positioned at midpoints of edges
        label_fontsize = 13
        colors = {'x': '#1a5f8a', 'y': '#0a8a6a', 'z': '#7a3ab0'}

        # Build labels with physical ranges from 1D arrays
        def get_range_str(arr_name):
            """Get physical range string for a 1D array."""
            if arr_name in info['1d_arrays']:
                d = info['1d_arrays'][arr_name]
                if d['min'] is not None and isinstance(d['min'], float):
                    return f"({d['min']:.0f}-{d['max']:.0f} m)"
            return ""

        # X-axis label (alongshore)
        mid_x = cube_dims[0] / 2
        along_range = get_range_str('alongshore_m')
        ax_cube.text(mid_x, -1.5, -0.5, f"alongshore_m\n{shape[0]} bins {along_range}",
                    fontsize=label_fontsize, ha='center', va='top',
                    fontweight='bold', color=colors['x'])

        # Y-axis label (elevation)
        mid_y = cube_dims[1] / 2
        elev_range = get_range_str('elevation_m')
        ax_cube.text(-1.5, mid_y, -0.5, f"elevation_m\n{shape[1]} bins {elev_range}",
                    fontsize=label_fontsize, ha='center', va='top',
                    fontweight='bold', color=colors['y'])

        # Z-axis label (time) - show count of intervals
        mid_z = cube_dims[2] / 2
        ax_cube.text(-1.5, -1.0, mid_z, f"time\n{shape[2]} intervals",
                    fontsize=label_fontsize, ha='center', va='center',
                    fontweight='bold', color=colors['z'])

        # Set axis limits with padding
        pad = 2
        ax_cube.set_xlim(-pad, cube_dims[0] + pad)
        ax_cube.set_ylim(-pad, cube_dims[1] + pad)
        ax_cube.set_zlim(-pad, cube_dims[2] + pad)

        ax_cube.set_xticks([])
        ax_cube.set_yticks([])
        ax_cube.set_zticks([])
        ax_cube.set_xlabel('')
        ax_cube.set_ylabel('')
        ax_cube.set_zlabel('')

        # Remove panes and grid
        ax_cube.xaxis.pane.fill = False
        ax_cube.yaxis.pane.fill = False
        ax_cube.zaxis.pane.fill = False
        ax_cube.xaxis.pane.set_edgecolor('none')
        ax_cube.yaxis.pane.set_edgecolor('none')
        ax_cube.zaxis.pane.set_edgecolor('none')
        ax_cube.grid(False)

        ax_cube.view_init(elev=20, azim=45)

    # --- Info Panel ---
    lines = []
    lines.append(f"File: {info['filename']}")
    lines.append("")

    # 3D arrays
    if info['3d_arrays']:
        lines.append("3D Arrays:")
        for name, data in info['3d_arrays'].items():
            s = data['shape']
            lines.append(f"  {name}  [{s[0]} x {s[1]} x {s[2]}]")
        lines.append("")

    # 1D arrays (axes)
    if info['1d_arrays']:
        lines.append("1D Arrays (Axes):")
        for name, data in info['1d_arrays'].items():
            lines.append(f"  {name}  [{data['length']}]")
            lines.append(f"    {data['sample']}")
        lines.append("")

    # Scalars
    if info['scalars']:
        lines.append("Scalars:")
        for name, value in info['scalars'].items():
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.2f}")
            else:
                lines.append(f"  {name}: {value}")

    info_text = "\n".join(lines)
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=12, family='monospace', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5',
                          edgecolor='#cccccc', alpha=0.9))

    # Title
    fig.suptitle(f"Data Cube Structure", fontsize=18, fontweight='bold', y=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def print_summary(info):
    """Print a text summary of the NPZ contents."""
    print(f"\n{info['filename']}")
    print("-" * 40)

    if info['3d_arrays']:
        for name, data in info['3d_arrays'].items():
            print(f"  {name}: {data['shape']}")

    if info['1d_arrays']:
        for name, data in info['1d_arrays'].items():
            print(f"  {name}: [{data['length']}] {data['sample']}")

    if info['scalars']:
        for name, value in info['scalars'].items():
            print(f"  {name}: {value}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Visualize NPZ data cube structure")
    parser.add_argument('npz_path', help="Path to the NPZ file")
    parser.add_argument('--save', '-s', default=None, help="Save to file")
    parser.add_argument('--no-plot', action='store_true', help="Text summary only")

    args = parser.parse_args()

    try:
        info = load_npz_info(args.npz_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print_summary(info)

    if not args.no_plot:
        create_visualization(info, args.save)


if __name__ == "__main__":
    main()
