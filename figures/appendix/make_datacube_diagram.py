"""
Create a diagram showing the structure and dimensions of the NPZ data cube.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Load cube to get actual dimensions
cube = np.load('results/data_cubes/DelMar_cube.npz', allow_pickle=True)

# Get dimensions
n_along = len(cube['alongshore_m'])  # 9140
n_elev = len(cube['elevation_m'])    # 120
n_time = len(cube['dates'])          # 405

# Coordinate ranges in meters
along_min, along_max = cube['alongshore_m'].min(), cube['alongshore_m'].max()
elev_min, elev_max = cube['elevation_m'].min(), cube['elevation_m'].max()

# Create figure - wider to give more space
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')

# Draw 3D cube using 2D projection (isometric-like)
cube_w = 3.5  # width (alongshore - X)
cube_h = 3.5  # height (elevation - Y)

# Offset for 3D effect
dx = 1.0
dy = 0.7

# Base position - shifted right to make room for left panel
bx, by = 4.5, 2.8

# Front face (alongshore x elevation)
front = patches.FancyBboxPatch((bx, by), cube_w, cube_h,
                                boxstyle="round,pad=0.02",
                                facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
ax.add_patch(front)

# Top face (alongshore x time)
top_verts = [(bx, by + cube_h), (bx + dx, by + cube_h + dy),
             (bx + cube_w + dx, by + cube_h + dy), (bx + cube_w, by + cube_h)]
top = patches.Polygon(top_verts, facecolor='#BBDEFB', edgecolor='#1565C0', linewidth=2)
ax.add_patch(top)

# Right face (elevation x time)
right_verts = [(bx + cube_w, by), (bx + cube_w + dx, by + dy),
               (bx + cube_w + dx, by + cube_h + dy), (bx + cube_w, by + cube_h)]
right = patches.Polygon(right_verts, facecolor='#90CAF9', edgecolor='#1565C0', linewidth=2)
ax.add_patch(right)

# === DIMENSION LABELS WITH ARROWS ===

# Alongshore (bottom arrow - X axis)
ax.annotate('', xy=(bx + cube_w, by - 0.4), xytext=(bx, by - 0.4),
            arrowprops=dict(arrowstyle='<->', color='#2E86AB', lw=2.5))
ax.text(bx + cube_w/2, by - 0.75,
        f'alongshore_m\n{n_along} cells = 2285 m',
        ha='center', va='top', fontsize=11, fontweight='bold', color='#2E86AB')

# Elevation (left arrow - Y axis) - moved further left
ax.annotate('', xy=(bx - 0.5, by + cube_h), xytext=(bx - 0.5, by),
            arrowprops=dict(arrowstyle='<->', color='#A23B72', lw=2.5))
ax.text(bx - 0.8, by + cube_h/2,
        f'elevation_m\n{n_elev} cells\n= 30 m',
        ha='right', va='center', fontsize=11, fontweight='bold', color='#A23B72')

# Time (diagonal arrow - Z axis) - repositioned to avoid overlap
ax.annotate('', xy=(bx + cube_w + dx + 0.15, by + cube_h + dy + 0.1),
            xytext=(bx + cube_w + 0.15, by + cube_h + 0.1),
            arrowprops=dict(arrowstyle='<->', color='#F18F01', lw=2.5))
ax.text(bx + cube_w + dx + 0.4, by + cube_h + dy + 0.3,
        f'dates / date_strings\n{n_time} time steps',
        ha='left', va='bottom', fontsize=11, fontweight='bold', color='#F18F01')

# === DATA ARRAYS LABEL (in center of front face) ===
ax.text(bx + cube_w/2, by + cube_h/2,
        'erosion\ndeposition\n\nshape:\n(9140, 120, 405)',
        ha='center', va='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#666', alpha=0.9))

# === TITLE ===
ax.text(6, 9.5, 'Del Mar Data Cube Structure',
        ha='center', va='center', fontsize=18, fontweight='bold')
ax.text(6, 9.0, 'DelMar_cube.npz',
        ha='center', va='center', fontsize=13, fontstyle='italic', color='#666')

# === NPZ ACCESS KEYS TABLE (left side, no overlap) ===
table_box = FancyBboxPatch((0.2, 5.5), 3.5, 3.2, boxstyle="round,pad=0.1",
                            facecolor='#FAFAFA', edgecolor='#999', linewidth=1.5)
ax.add_patch(table_box)

ax.text(2.0, 8.5, 'NPZ Access Keys', fontsize=13, fontweight='bold', ha='center', va='top')

table_content = [
    ('erosion', '(9140, 120, 405)'),
    ('deposition', '(9140, 120, 405)'),
    ('alongshore_m', '(9140,)'),
    ('elevation_m', '(120,)'),
    ('dates', '(405,)'),
    ('date_strings', '(405,)'),
]

for i, (key, shape) in enumerate(table_content):
    y_pos = 8.0 - i * 0.4
    ax.text(0.4, y_pos, f'{key}', fontsize=10, fontfamily='monospace', va='top', fontweight='bold')
    ax.text(2.2, y_pos, f'{shape}', fontsize=10, fontfamily='monospace', va='top', color='#555')

# === ARRAY INDEXING BOX (bottom left) ===
axis_box = FancyBboxPatch((0.2, 0.3), 3.5, 1.8, boxstyle="round,pad=0.1",
                          facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=1.5)
ax.add_patch(axis_box)

ax.text(2.0, 1.95, 'Array Indexing', fontsize=12, fontweight='bold',
        ha='center', va='top', color='#6A1B9A')
ax.text(2.0, 1.55,
        'cube[key][i, j, k]\n\n'
        'i = alongshore (axis 0)\n'
        'j = elevation (axis 1)\n'
        'k = time (axis 2)',
        fontsize=10, fontfamily='monospace', ha='center', va='top')

plt.tight_layout()

# Save figure
plt.savefig('figures/appendix/datacube_structure.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figures/appendix/datacube_structure.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("Saved: figures/appendix/datacube_structure.png")
print("Saved: figures/appendix/datacube_structure.pdf")
