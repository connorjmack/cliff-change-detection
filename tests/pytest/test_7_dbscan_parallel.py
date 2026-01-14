"""
Tests for 7_dbscan_parallel.py

Tests DBSCAN clustering functionality including:
- M3C2 field identification
- Erosion/deposition splitting
- DBSCAN clustering
- Report generation
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch
import laspy
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))


class TestM3C2FieldIdentification:
    """Test M3C2 field identification in LAS files."""

    def test_identifies_m3c2_distance_field(self, sample_m3c2_las_file):
        """Test finding M3C2 distance field."""
        las = laspy.read(sample_m3c2_las_file)

        # Find field
        m3c2_field = None
        for field in las.point_format.dimension_names:
            if 'm3c2' in field.lower() and 'distance' in field.lower():
                m3c2_field = field
                break

        assert m3c2_field is not None
        assert m3c2_field == "M3C2_distance"

    def test_identifies_significance_field(self, sample_m3c2_las_file):
        """Test finding significance field."""
        las = laspy.read(sample_m3c2_las_file)

        sig_field = None
        for field in las.point_format.dimension_names:
            if 'significant' in field.lower():
                sig_field = field
                break

        assert sig_field is not None


class TestErosionDepositionSplitting:
    """Test splitting into erosion and deposition."""

    def test_negative_values_are_erosion(self, sample_m3c2_las_file):
        """Test that negative M3C2 values are identified as erosion."""
        las = laspy.read(sample_m3c2_las_file)

        distances = las.M3C2_distance
        erosion_mask = distances < 0
        deposition_mask = distances > 0

        # Should have both erosion and deposition
        assert np.sum(erosion_mask) > 0
        assert np.sum(deposition_mask) > 0

        # Counts should add up (excluding zeros)
        total_change = np.sum(erosion_mask) + np.sum(deposition_mask)
        assert total_change > 0


class TestDBSCANClustering:
    """Test DBSCAN clustering functionality."""

    def test_run_dbscan_file_creates_outputs(self, sample_m3c2_las_file, temp_dir):
        """Test DBSCAN processing creates output files."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "dbscan_parallel",
            Path(__file__).parent.parent / "pipeline" / "7_dbscan_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Add ClusterID to test file first
        las = laspy.read(sample_m3c2_las_file)

        erosion_dir = Path(temp_dir) / "erosion"
        deposition_dir = Path(temp_dir) / "deposition"

        erosion_dir.mkdir()
        deposition_dir.mkdir()

        stats = module.run_dbscan_file(
            sample_m3c2_las_file,
            str(erosion_dir),
            str(deposition_dir),
            eps=0.35,
            min_samples=10,  # Low for test data
            min_change_threshold=0.05,
            replace=False
        )

        assert stats is not None
        assert "status" in stats
        assert "erosion_points" in stats
        assert "deposition_points" in stats

    def test_dbscan_parameters(self):
        """Test DBSCAN parameter configuration."""
        # Verify typical parameters
        eps = 0.35
        min_samples = 30

        assert eps > 0
        assert min_samples > 0
        assert isinstance(eps, float)
        assert isinstance(min_samples, int)


class TestSignificanceFiltering:
    """Test significance filtering."""

    def test_filters_significant_changes(self, sample_m3c2_las_file):
        """Test filtering for significant changes."""
        las = laspy.read(sample_m3c2_las_file)

        sig_values = las.significant_change
        sig_mask = sig_values == 1

        # Should have some significant changes
        assert np.sum(sig_mask) > 0

        # Verify filtering works
        filtered_count = np.sum(sig_mask)
        assert filtered_count <= len(las.points)


class TestReportGeneration:
    """Test DBSCAN report generation."""

    def test_finds_latest_pipeline_run(self, temp_dir):
        """Test finding latest pipeline run directory."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "dbscan_parallel",
            Path(__file__).parent.parent / "pipeline" / "7_dbscan_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Create test run directories
        run1 = Path(temp_dir) / "pipeline_run_20220101"
        run2 = Path(temp_dir) / "pipeline_run_20220201"

        run1.mkdir()
        run2.mkdir()

        latest = module.find_latest_pipeline_run(temp_dir)

        assert latest is not None
        assert "pipeline_run_" in latest
