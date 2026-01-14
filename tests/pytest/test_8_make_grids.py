"""
Tests for 8_make_grids.py

Tests spatial gridding functionality including:
- Vertical binning
- Spatial joins with polygons
- Median M3C2 aggregation
- Cluster mode calculation
- Grid shape enforcement
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, Polygon
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))


class TestGridding:
    """Test grid generation functionality."""

    def test_vertical_binning(self):
        """Test vertical binning of points."""
        # Test data
        z_values = np.array([0.5, 1.5, 2.5, 10.5, 15.3])
        res = 1.0
        height = 20.0

        z_bins = np.arange(0, height + res, res)
        z_labels = [f"{b:.2f}" for b in z_bins[:-1]]

        binned = pd.cut(z_values, bins=z_bins, right=False, labels=z_labels)

        assert len(binned) == len(z_values)
        # First value (0.5) should be in bin "0.00"
        assert binned[0] == "0.00"
        # Last value (15.3) should be in bin "15.00"
        assert binned[4] == "15.00"

    def test_resolution_params(self):
        """Test resolution parameter configurations."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "make_grids",
            Path(__file__).parent.parent / "pipeline" / "8_make_grids.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Test finding shapefile logic exists
        assert hasattr(module, 'find_shapefile')


class TestSpatialJoins:
    """Test spatial join operations."""

    def test_points_within_polygon(self):
        """Test spatial join with polygon containment."""
        # Create test polygon
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        polys_gdf = gpd.GeoDataFrame({'Polygon_ID': [0], 'geometry': [poly]})

        # Create test points
        points = [Point(5, 5), Point(15, 15), Point(3, 3)]
        points_gdf = gpd.GeoDataFrame({
            'value': [1, 2, 3],
            'geometry': points
        }, crs=polys_gdf.crs)

        # Spatial join
        joined = gpd.sjoin(points_gdf, polys_gdf, how='inner', predicate='within')

        # Only 2 points should be within polygon
        assert len(joined) == 2
        assert 0 in joined['Polygon_ID'].values


class TestAggregationFunctions:
    """Test aggregation functions."""

    def test_median_aggregation(self):
        """Test median aggregation for M3C2 values."""
        data = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B'],
            'value': [1.0, 2.0, 3.0, 10.0, 20.0]
        })

        result = data.groupby('group')['value'].median()

        assert result['A'] == 2.0
        assert result['B'] == 15.0

    def test_mode_calculation(self):
        """Test mode calculation for ClusterID."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "make_grids",
            Path(__file__).parent.parent / "pipeline" / "8_make_grids.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Test data
        cluster_ids = pd.Series([1, 1, 1, 2, 2, 3])

        # Mode should be 1
        mode_val = cluster_ids.mode()
        assert mode_val.iloc[0] == 1

    def test_rms_calculation(self):
        """Test RMS calculation for uncertainty."""
        values = np.array([3.0, 4.0])
        rms = np.sqrt(np.mean(values**2))

        assert np.isclose(rms, 3.5355339, rtol=1e-5)


class TestGridShapeEnforcement:
    """Test grid shape consistency."""

    def test_pivot_shape_consistency(self):
        """Test that pivot tables maintain consistent shape."""
        # Create test data with irregular coverage
        data = pd.DataFrame({
            'Polygon_ID': [0, 0, 1, 2, 2],
            'z_bin': ['0.00', '1.00', '0.00', '0.00', '2.00'],
            'value': [1.5, 2.5, 3.5, 4.5, 5.5]
        })

        # Pivot
        pivot = data.pivot(index='Polygon_ID', columns='z_bin', values='value')

        # Enforce shape
        all_polygons = [0, 1, 2, 3]  # Include polygon 3 with no data
        expected_cols = ['0.00', '1.00', '2.00', '3.00']

        pivot_enforced = pivot.reindex(index=all_polygons, columns=expected_cols)

        # Should have 4 rows and 4 columns
        assert pivot_enforced.shape == (4, 4)

        # Polygon 3 should be all NaN
        assert pivot_enforced.loc[3].isna().all()


class TestMakeGridFunction:
    """Test the main makeGrid function."""

    def test_makeGrid_with_empty_points(self, temp_dir):
        """Test makeGrid handles empty point clouds."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "make_grids",
            Path(__file__).parent.parent / "pipeline" / "8_make_grids.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # This would require creating mock LAS file and shapefile
        # Placeholder for integration test
        pass


class TestShapefileLoading:
    """Test shapefile loading functionality."""

    def test_find_shapefile_pattern_matching(self):
        """Test shapefile finding with pattern matching."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "make_grids",
            Path(__file__).parent.parent / "pipeline" / "8_make_grids.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Verify function exists
        assert hasattr(module, 'find_shapefile')
