"""
Tests for 4_remove_beach_parallel.py

Tests beach removal functionality including:
- Random Forest classification
- Histogram matching for intensity normalization
- Parallel processing
- Report and figure generation
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))


class TestHistogramMatching:
    """Test histogram matching functionality."""

    def test_match_histograms_shape_preservation(self):
        """Test that histogram matching preserves array shape."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "remove_beach_parallel",
            Path(__file__).parent.parent / "pipeline" / "4_remove_beach_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        source = np.random.rand(100) * 1000
        reference = np.random.rand(200) * 1000

        result = module.match_histograms(source, reference)

        assert result.shape == source.shape
        assert len(result) == len(source)

    def test_match_histograms_distribution_change(self):
        """Test that histogram matching changes distribution."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "remove_beach_parallel",
            Path(__file__).parent.parent / "pipeline" / "4_remove_beach_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Source with uniform distribution
        source = np.linspace(0, 100, 100)
        # Reference with different distribution
        reference = np.random.normal(50, 10, 200)

        result = module.match_histograms(source, reference)

        # Result should have different statistics than source
        assert not np.allclose(np.mean(result), np.mean(source), rtol=0.1)


class TestLASReading:
    """Test LAS file reading."""

    def test_read_las_to_array(self, sample_las_file):
        """Test reading LAS file to numpy array."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "remove_beach_parallel",
            Path(__file__).parent.parent / "pipeline" / "4_remove_beach_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        las, data = module.read_las_to_array(sample_las_file)

        assert las is not None
        assert data.shape[1] == 4  # x, y, z, intensity
        assert len(data) == 100  # Sample file has 100 points


class TestClassification:
    """Test beach classification."""

    def test_classify_and_write_with_model(self, sample_las_file, mock_rf_model, temp_dir):
        """Test classification with RF model."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "remove_beach_parallel",
            Path(__file__).parent.parent / "pipeline" / "4_remove_beach_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model_path, scaler_path = mock_rf_model

        stats = module.classify_and_write(
            sample_las_file,
            model_path,
            scaler_path,
            temp_dir,
            ref_intensity=None,
            replace=False
        )

        assert stats is not None
        assert "status" in stats
        assert "total_points" in stats

    def test_classify_skips_existing(self, sample_las_file, mock_rf_model, temp_dir):
        """Test that classification skips existing files when replace=False."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "remove_beach_parallel",
            Path(__file__).parent.parent / "pipeline" / "4_remove_beach_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model_path, scaler_path = mock_rf_model

        # Create output file
        output_name = Path(sample_las_file).stem + "_nobeach.las"
        output_path = Path(temp_dir) / output_name
        output_path.touch()

        stats = module.classify_and_write(
            sample_las_file,
            model_path,
            scaler_path,
            temp_dir,
            ref_intensity=None,
            replace=False
        )

        assert stats["status"] == "Skipped"


class TestFigureGeneration:
    """Test figure generation."""

    def test_generate_summary_figures_creates_plots(self, temp_dir):
        """Test that summary figures are created."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "remove_beach_parallel",
            Path(__file__).parent.parent / "pipeline" / "4_remove_beach_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Mock results data
        results = [
            {
                "filename": "20220101_test.las",
                "status": "Success",
                "total_points": 1000,
                "beach_points_removed": 300,
                "percent_removed": 30.0,
                "processing_time_sec": 1.5
            },
            {
                "filename": "20220201_test.las",
                "status": "Success",
                "total_points": 1200,
                "beach_points_removed": 400,
                "percent_removed": 33.3,
                "processing_time_sec": 1.8
            }
        ]

        module.generate_summary_figures(results, temp_dir, "TestLocation", "20220101_120000")

        # Check that figures were created
        fig_files = list(Path(temp_dir).glob("Figure_*.png"))
        assert len(fig_files) >= 2  # At least time series and distribution

    def test_figure_generation_handles_no_success(self, temp_dir):
        """Test figure generation with no successful runs."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "remove_beach_parallel",
            Path(__file__).parent.parent / "pipeline" / "4_remove_beach_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        results = [
            {"filename": "test.las", "status": "Error", "percent_removed": 0}
        ]

        # Should not crash
        try:
            module.generate_summary_figures(results, temp_dir, "Test", "20220101")
        except Exception as e:
            pytest.fail(f"Should handle no success cases: {e}")
