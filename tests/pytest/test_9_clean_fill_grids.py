"""
Tests for 9_clean_fill_grids.py

Tests grid cleaning and hole filling including:
- Resolution-dependent parameters
- Cluster filtering by threshold
- Alpha shape hole detection
- Spatial interpolation
- Connected component cleanup
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))


class TestResolutionParameters:
    """Test resolution-dependent parameter settings."""

    def test_get_resolution_params_10cm(self):
        """Test parameters for 10cm resolution."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "clean_fill_grids",
            Path(__file__).parent.parent / "pipeline" / "9_clean_fill_grids.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        params = module.get_resolution_params('10cm')

        assert params['cell_size'] == 0.10
        assert params['cell_area'] == 0.01
        assert 'default_threshold' in params
        assert 'alpha' in params

    def test_get_resolution_params_25cm(self):
        """Test parameters for 25cm resolution."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "clean_fill_grids",
            Path(__file__).parent.parent / "pipeline" / "9_clean_fill_grids.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        params = module.get_resolution_params('25cm')

        assert params['cell_size'] == 0.25
        assert params['cell_area'] == 0.0625

    def test_get_resolution_params_1m(self):
        """Test parameters for 1m resolution."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "clean_fill_grids",
            Path(__file__).parent.parent / "pipeline" / "9_clean_fill_grids.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        params = module.get_resolution_params('1m')

        assert params['cell_size'] == 1.0
        assert params['cell_area'] == 1.0


class TestCSVDataHandling:
    """Test CSV loading and saving."""

    def test_load_csv_data(self, temp_dir):
        """Test loading CSV data."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "clean_fill_grids",
            Path(__file__).parent.parent / "pipeline" / "9_clean_fill_grids.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Create test CSV
        test_csv = Path(temp_dir) / "test.csv"
        df = pd.DataFrame({
            'M3C2_0.00m': [1.0, 2.0, 3.0],
            'M3C2_0.10m': [1.5, 2.5, 3.5]
        }, index=[0, 1, 2])
        df.to_csv(test_csv)

        header, rows, data = module.load_csv_data(str(test_csv))

        assert len(header) == 2
        assert len(rows) == 3
        assert data.shape == (3, 2)

    def test_save_csv_data(self, temp_dir):
        """Test saving CSV data."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "clean_fill_grids",
            Path(__file__).parent.parent / "pipeline" / "9_clean_fill_grids.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        headers = ['col1', 'col2']
        rows = ['0', '1']

        output_path = Path(temp_dir) / "output.csv"

        module.save_csv_data(str(output_path), data, headers, rows, testing=False, replace=True)

        assert output_path.exists()

        # Read back and verify
        df = pd.read_csv(output_path, index_col=0)
        assert df.shape == (2, 2)


class TestElevationParsing:
    """Test elevation parsing from headers."""

    def test_parse_elevation_from_header(self):
        """Test parsing elevation value from column header."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "clean_fill_grids",
            Path(__file__).parent.parent / "pipeline" / "9_clean_fill_grids.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        header = "M3C2_0.10m"
        elevation = module.parse_elevation_from_header(header)

        assert elevation == 0.10

    def test_parse_elevation_various_formats(self):
        """Test parsing different header formats."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "clean_fill_grids",
            Path(__file__).parent.parent / "pipeline" / "9_clean_fill_grids.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        test_cases = [
            ("M3C2_0.00m", 0.0),
            ("M3C2_1.50m", 1.5),
            ("ClusterID_2.00m", 2.0)
        ]

        for header, expected in test_cases:
            result = module.parse_elevation_from_header(header)
            assert result == expected


class TestAlphaShapes:
    """Test alpha shape functionality for hole detection."""

    def test_convex_hull_creation(self):
        """Test convex hull creation for points."""
        points = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0.5, 0.5]  # Interior point
        ])

        hull = ConvexHull(points)

        # Hull should have 4 vertices (the corners)
        assert len(hull.vertices) == 4


class TestClusterFiltering:
    """Test cluster filtering by threshold."""

    def test_resolution_to_label(self):
        """Test resolution string conversion to label."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "clean_fill_grids",
            Path(__file__).parent.parent / "pipeline" / "9_clean_fill_grids.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.resolution_to_label('10cm') == '10cm'
        assert module.resolution_to_label('25cm') == '25cm'
        assert module.resolution_to_label('1m') == '100cm'

    def test_cluster_threshold_filtering(self):
        """Test filtering clusters by non-zero cell count."""
        # Create test cluster data
        cluster_grid = np.array([
            [1, 1, 1, 0, 0],  # Cluster 1: 3 non-zero cells
            [0, 0, 2, 2, 0],  # Cluster 2: 2 non-zero cells
            [0, 3, 3, 3, 3],  # Cluster 3: 4 non-zero cells
        ])

        threshold = 3

        # Count non-zero cells per cluster
        unique_clusters = np.unique(cluster_grid)
        unique_clusters = unique_clusters[unique_clusters != 0]

        keep_clusters = []
        for cluster_id in unique_clusters:
            mask = cluster_grid == cluster_id
            non_zero_count = np.sum(mask)
            if non_zero_count >= threshold:
                keep_clusters.append(cluster_id)

        # Clusters 1 and 3 should pass (3 and 4 cells)
        assert 1 in keep_clusters
        assert 3 in keep_clusters
        assert 2 not in keep_clusters


class TestShapefileIntegration:
    """Test shapefile loading for polygon centroids."""

    def test_find_shapefile(self):
        """Test shapefile finding functionality."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "clean_fill_grids",
            Path(__file__).parent.parent / "pipeline" / "9_clean_fill_grids.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Verify function exists
        assert hasattr(module, 'find_shapefile')
        assert hasattr(module, 'load_polygon_centroids')


class TestInterpolation:
    """Test spatial interpolation for hole filling."""

    def test_griddata_interpolation(self):
        """Test scipy griddata interpolation."""
        from scipy.interpolate import griddata

        # Known points
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        values = np.array([0, 1, 1, 2])

        # Interpolation point
        grid_x, grid_y = np.meshgrid([0.5], [0.5])

        # Interpolate
        result = griddata(points, values, (grid_x, grid_y), method='linear')

        # At (0.5, 0.5), value should be 1.0 (average of corners)
        assert np.isclose(result[0, 0], 1.0, rtol=0.1)
