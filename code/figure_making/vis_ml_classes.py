#!/usr/bin/env python3
"""
vis_split_render_v6_fixed.py

Purpose:
    Generates two separate images with IDENTICAL camera angles and explicit files:
    1. A_rgb_oblique.png: RGB view (01_raw.las)
    2. B_class_oblique.png: Classification view (beach.las, cliff.las, veg.las)

    Fixes:
    - RELATIVE CAMERA: Calculates camera position based on data centroid (fixes blank images).
    - SYNCED VIEWS: Uses the exact same calculated coordinates for both panels.
    - ZOOM: Physically moves camera closer for a tighter shot.
"""

import os
import platform
import numpy as np
import laspy
import pyvista as pv

# ============================================================================
# CONFIGURATION
# ============================================================================

if platform.system() == 'Darwin':
    PROJECT_ROOT = "/Volumes/group/LiDAR/LidarProcessing/LidarProcessingCliffs"
else:
    PROJECT_ROOT = "/project/group/LiDAR/LidarProcessing/LidarProcessingCliffs"

# Directories
INPUT_DIR = os.path.join(PROJECT_ROOT, "figures", "ml_classification", "input_clouds")
OUT_DIR   = os.path.join(PROJECT_ROOT, "figures", "ml_classification", "figs")

os.makedirs(OUT_DIR, exist_ok=True)

# Input Files
FILE_RGB   = os.path.join(INPUT_DIR, "01_raw.las")   # Panel A
FILE_BEACH = os.path.join(INPUT_DIR, "beach.las")    # Panel B
FILE_CLIFF = os.path.join(INPUT_DIR, "cliff.las")    # Panel B
FILE_VEG   = os.path.join(INPUT_DIR, "veg.las")      # Panel B

# Output Files
OUT_FIG_A = os.path.join(OUT_DIR, "A_rgb_oblique.png")
OUT_FIG_B = os.path.join(OUT_DIR, "B_class_oblique.png")

# --- Visual Settings ---
MAX_Z_ELEVATION = 25.0
POINT_SIZE      = 4.0
CLIFF_SIZE_ADD  = 3.0  # Extra size for cliff points

# --- Camera Offsets (Relative to Data Center) ---
# Moves the camera closer (Zoomed in) and keeps the oblique angle
# Previous was approx (-180, -450, 80). We scale this down to zoom in physically.
REL_CAMERA_POS   = (-100, -250, 50) 
REL_FOCAL_SHIFT  = (20, -50, 5)     
CAMERA_VIEW_UP   = (0, 0, 1)

# Colors (Hex)
HEX_BEACH = '#EDD9A3'
HEX_VEG   = '#228B22' # Green
HEX_CLIFF = '#A0522D' # Brown

# ============================================================================
# HELPERS
# ============================================================================

def read_las_xyz(filepath):
    """Reads only XYZ."""
    if not os.path.exists(filepath):
        print(f"[ERROR] Missing file: {filepath}")
        return None
    try:
        with laspy.open(filepath) as f:
            las = f.read()
            return np.vstack((las.x, las.y, las.z)).T
    except Exception as e:
        print(f"[ERROR] Could not read {filepath}: {e}")
        return None

def read_las_rgb_robust(filepath):
    """Reads XYZ and RGB with Percentile Normalization."""
    if not os.path.exists(filepath):
        print(f"[ERROR] Missing file: {filepath}")
        return None, None

    print(f"Reading RGB from: {os.path.basename(filepath)}")
    try:
        with laspy.open(filepath) as f:
            las = f.read()
            points = np.vstack((las.x, las.y, las.z)).T
            
            rgb = None
            if hasattr(las, 'red'):
                def robust_scale(channel):
                    p2, p98 = np.percentile(channel, (2, 98))
                    if p98 - p2 == 0: return np.zeros_like(channel, dtype=np.uint8)
                    scaled = (channel - p2) / (p98 - p2)
                    return np.clip(scaled * 255, 0, 255).astype(np.uint8)

                r = robust_scale(las.red)
                g = robust_scale(las.green)
                b = robust_scale(las.blue)
                rgb = np.vstack((r, g, b)).T
            
        return points, rgb
    except Exception as e:
        print(f"[ERROR] Could not read {filepath}: {e}")
        return None, None

def calculate_camera_params(points):
    """Calculates absolute camera coordinates based on the point cloud centroid."""
    centroid = np.mean(points, axis=0)
    
    # Calculate absolute positions
    abs_pos   = centroid + np.array(REL_CAMERA_POS)
    abs_focal = centroid + np.array(REL_FOCAL_SHIFT)
    
    print(f"Camera Calculated:")
    print(f"  Centroid: {centroid}")
    print(f"  Pos:      {abs_pos}")
    print(f"  Focal:    {abs_focal}")
    
    return abs_pos, abs_focal

def apply_camera(plotter, pos, focal):
    """Applies the pre-calculated camera parameters."""
    plotter.camera.position = pos
    plotter.camera.focal_point = focal
    plotter.camera.up = CAMERA_VIEW_UP
    # We don't use zoom() here because we moved the physical position closer
    
# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    # 1. READ RGB DATA (To establish coordinate system)
    print("\n--- Step 1: Loading RGB Cloud ---")
    pts_rgb, colors_rgb = read_las_rgb_robust(FILE_RGB)
    
    if pts_rgb is None or len(pts_rgb) == 0:
        print("[CRITICAL ERROR] Could not load RGB points. Exiting.")
        return

    # Filter Elevation (RGB)
    mask = pts_rgb[:, 2] < MAX_Z_ELEVATION
    pts_rgb = pts_rgb[mask]
    if colors_rgb is not None: colors_rgb = colors_rgb[mask]
    
    # Decimate (RGB)
    pts_rgb = pts_rgb[::2]
    if colors_rgb is not None: colors_rgb = colors_rgb[::2]

    # 2. CALCULATE CAMERA (Once, based on RGB cloud)
    # This ensures both images look at the exact same spot
    cam_pos, cam_focal = calculate_camera_params(pts_rgb)

    # 3. RENDER FIGURE A (RGB)
    print("\n--- Step 2: Rendering Figure A (RGB) ---")
    cloud_rgb = pv.PolyData(pts_rgb)
    if colors_rgb is not None: cloud_rgb.point_data['RGB'] = colors_rgb

    pl_a = pv.Plotter(off_screen=True, window_size=(1000, 900))
    pl_a.set_background('white')
    pl_a.add_text("A) Original RGB Point Cloud", position='upper_left', color='black', font_size=14)
    
    if colors_rgb is not None:
        pl_a.add_points(cloud_rgb, scalars='RGB', rgba=True, point_size=POINT_SIZE)
    else:
        pl_a.add_points(cloud_rgb, color='black', point_size=POINT_SIZE)

    apply_camera(pl_a, cam_pos, cam_focal)
    
    print(f"Saving {OUT_FIG_A}...")
    pl_a.screenshot(OUT_FIG_A)

    # 4. LOAD CLASSIFICATION DATA
    print("\n--- Step 3: Loading Classification Data ---")
    pts_beach = read_las_xyz(FILE_BEACH)
    pts_cliff = read_las_xyz(FILE_CLIFF)
    pts_veg   = read_las_xyz(FILE_VEG)

    # Filter & Decimate
    def process_pts(pts):
        if pts is None: return np.array([])
        pts = pts[pts[:, 2] < MAX_Z_ELEVATION]
        return pts[::2]

    pts_beach = process_pts(pts_beach)
    pts_cliff = process_pts(pts_cliff)
    pts_veg   = process_pts(pts_veg)

    # 5. RENDER FIGURE B (Classification)
    print("\n--- Step 4: Rendering Figure B (Classification) ---")
    pl_b = pv.Plotter(off_screen=True, window_size=(1000, 900))
    pl_b.set_background('white')
    pl_b.add_text("B) ML Classification Results", position='upper_left', color='black', font_size=14)

    has_points = False
    
    if len(pts_beach) > 0:
        pl_b.add_points(pv.PolyData(pts_beach), color=HEX_BEACH, point_size=POINT_SIZE+1, label="Beach")
        has_points = True

    if len(pts_veg) > 0:
        pl_b.add_points(pv.PolyData(pts_veg), color=HEX_VEG, point_size=POINT_SIZE, label="Vegetation")
        has_points = True

    if len(pts_cliff) > 0:
        # Cliff is rendered larger
        pl_b.add_points(pv.PolyData(pts_cliff), color=HEX_CLIFF, point_size=POINT_SIZE, label="Cliff")
        has_points = True

    if not has_points:
        print("[ERROR] No classification points found to render!")
        return

    # Add Legend
    legend_entries = [['Beach', HEX_BEACH], ['Vegetation', HEX_VEG], ['Cliff', HEX_CLIFF]]
    pl_b.add_legend(legend_entries, bcolor='white', border=True, size=(0.2, 0.2), loc='lower right')

    # Apply SAME Camera
    apply_camera(pl_b, cam_pos, cam_focal)

    print(f"Saving {OUT_FIG_B}...")
    pl_b.screenshot(OUT_FIG_B)

if __name__ == "__main__":
    main()
    print("\nDone.")