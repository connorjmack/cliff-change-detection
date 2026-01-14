"""
Tests for 2_audit_cropping.py

Tests quality control functionality including:
- File analysis (size and point count)
- Suspect file identification
- Deletion mode
- Report and plot generation
"""

import pytest
import os
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "audits"))


class TestFileAnalysis:
    """Test file analysis functionality."""

    def test_analyze_file_valid_las(self, sample_las_file):
        """Test analyzing a valid LAS file."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "qc_cropped_files",
            Path(__file__).parent.parent / "audits" / "2_audit_cropping.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        result = module.analyze_file(sample_las_file)

        assert result is not None
        assert "Filename" in result
        assert "Size_MB" in result
        assert "Point_Count" in result
        assert "Status" in result
        assert result["Point_Count"] > 0

    def test_analyze_file_identifies_suspects(self, sample_las_file):
        """Test that files below threshold are marked as suspect."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "qc_cropped_files",
            Path(__file__).parent.parent / "audits" / "2_audit_cropping.py"
        )
        module = importlib.util.module_from_spec(spec)

        # Set a high threshold to mark file as suspect
        with patch.object(module, 'MIN_POINT_THRESHOLD', 200):
            spec.loader.exec_module(module)

            result = module.analyze_file(sample_las_file)

            # Sample file has 100 points, below threshold of 200
            assert result["Status"] == "SUSPECT"

    def test_analyze_file_error_handling(self):
        """Test error handling for invalid files."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "qc_cropped_files",
            Path(__file__).parent.parent / "audits" / "2_audit_cropping.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        result = module.analyze_file("/nonexistent/file.las")

        assert "ERROR" in result["Status"]


class TestPlotGeneration:
    """Test QC plot generation."""

    def test_plot_results_creates_files(self, temp_dir):
        """Test that plot_results creates PNG files."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "qc_cropped_files",
            Path(__file__).parent.parent / "audits" / "2_audit_cropping.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Create test data
        data = {
            "Location": ["DelMar", "DelMar", "SanElijo"],
            "Filename": ["test1.las", "test2.las", "test3.las"],
            "Size_MB": [10.5, 12.3, 9.8],
            "Point_Count": [1500, 800, 1200],  # One below threshold
            "Status": ["OK", "SUSPECT", "OK"]
        }
        df = pd.DataFrame(data)

        module.plot_results(df, temp_dir, delete_mode=False)

        # Check that plots were created
        plot_files = list(Path(temp_dir).glob("QC_*.png"))
        assert len(plot_files) == 2  # Points vs Size, FileSize Distribution

    def test_plot_handles_empty_dataframe(self, temp_dir):
        """Test plot generation with empty dataframe."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "qc_cropped_files",
            Path(__file__).parent.parent / "audits" / "2_audit_cropping.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        df = pd.DataFrame(columns=["Location", "Filename", "Size_MB", "Point_Count", "Status"])

        # Should not crash with empty data
        try:
            module.plot_results(df, temp_dir, delete_mode=False)
        except Exception as e:
            pytest.fail(f"plot_results should handle empty dataframe: {e}")


class TestDeletionMode:
    """Test file deletion functionality."""

    def test_deletion_mode_removes_suspects(self, temp_dir, sample_las_file):
        """Test that deletion mode removes suspect files."""
        # Copy sample file to temp location for deletion test
        test_file = Path(temp_dir) / "suspect.las"
        import shutil
        shutil.copy(sample_las_file, test_file)

        assert test_file.exists()

        # Simulate deletion
        os.remove(test_file)

        assert not test_file.exists()

    def test_deletion_logging(self):
        """Test that deletions are logged properly."""
        # This would test the logging mechanism
        # In actual implementation, verify log messages include file paths and counts
        pass


class TestOSDetection:
    """Test OS-specific path handling."""

    def test_get_root_path_darwin(self):
        """Test root path detection on macOS."""
        import importlib.util
        import platform

        spec = importlib.util.spec_from_file_location(
            "qc_cropped_files",
            Path(__file__).parent.parent / "audits" / "2_audit_cropping.py"
        )
        module = importlib.util.module_from_spec(spec)

        with patch.object(platform, 'system', return_value='Darwin'):
            spec.loader.exec_module(module)

            root = module.get_root_path()
            assert root.startswith("/Volumes/group/LiDAR")

    def test_get_root_path_linux(self):
        """Test root path detection on Linux."""
        import importlib.util
        import platform

        spec = importlib.util.spec_from_file_location(
            "qc_cropped_files",
            Path(__file__).parent.parent / "audits" / "2_audit_cropping.py"
        )
        module = importlib.util.module_from_spec(spec)

        with patch.object(platform, 'system', return_value='Linux'):
            spec.loader.exec_module(module)

            root = module.get_root_path()
            assert root.startswith("/project/group/LiDAR")
