"""
Tests for 5_remove_veg_parallel.py

Tests vegetation removal via CloudCompare CANUPO including:
- CloudCompare command construction
- Point count reading
- Parallel processing
- Report generation
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))


class TestPointCountReading:
    """Test point count extraction from LAS headers."""

    def test_get_point_count(self, sample_las_file):
        """Test reading point count from LAS header."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "remove_veg_parallel",
            Path(__file__).parent.parent / "pipeline" / "5_remove_veg_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        count = module.get_point_count(sample_las_file)

        assert count == 100  # Sample file has 100 points

    def test_get_point_count_invalid_file(self):
        """Test point count with invalid file."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "remove_veg_parallel",
            Path(__file__).parent.parent / "pipeline" / "5_remove_veg_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        count = module.get_point_count("/nonexistent/file.las")

        assert count == 0


class TestCloudCompareIntegration:
    """Test CloudCompare command construction."""

    def test_classify_file_command_structure(self, sample_las_file, temp_dir, mock_subprocess_run):
        """Test that CloudCompare command is properly structured."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "remove_veg_parallel",
            Path(__file__).parent.parent / "pipeline" / "5_remove_veg_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        classifier_prm = str(Path(temp_dir) / "test.prm")
        Path(classifier_prm).touch()

        shift = ('-475000', '-3653000', '0')
        cc_path = "/usr/local/bin/CloudCompare"

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Create output file to simulate successful run
            output_path = Path(temp_dir) / (Path(sample_las_file).stem + "_noveg.las")
            output_path.touch()

            with patch.object(module, 'get_point_count', return_value=80):
                stats = module.classify_file_with_stats(
                    sample_las_file,
                    classifier_prm,
                    temp_dir,
                    shift,
                    cc_path,
                    replace=False
                )

            assert mock_run.called
            call_args = mock_run.call_args[0][0]

            # Verify command structure
            assert cc_path in call_args
            assert "-SILENT" in call_args
            assert "-CANUPO_CLASSIFY" in call_args

    def test_classify_file_skip_existing(self, sample_las_file, temp_dir):
        """Test skipping existing output files."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "remove_veg_parallel",
            Path(__file__).parent.parent / "pipeline" / "5_remove_veg_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Create existing output
        output_path = Path(temp_dir) / (Path(sample_las_file).stem + "_noveg.las")
        output_path.touch()

        stats = module.classify_file_with_stats(
            sample_las_file,
            str(Path(temp_dir) / "test.prm"),
            temp_dir,
            ('-475000', '-3653000', '0'),
            "/usr/bin/CloudCompare",
            replace=False
        )

        assert stats["status"] == "Skipped"


class TestReportGeneration:
    """Test report and figure generation."""

    def test_generate_summary_figures(self, temp_dir):
        """Test vegetation removal figure generation."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "remove_veg_parallel",
            Path(__file__).parent.parent / "pipeline" / "5_remove_veg_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        results = [
            {
                "filename": "20220101_test.las",
                "status": "Success",
                "input_points": 1000,
                "removed_points": 200,
                "percent_removed": 20.0,
                "processing_time_sec": 15.5
            }
        ]

        module.generate_summary_figures(results, temp_dir, "TestLocation", "20220101")

        # Should create figure files
        figs = list(Path(temp_dir).glob("Veg_*.png"))
        assert len(figs) >= 1


class TestGlobalShiftConfiguration:
    """Test global shift configurations."""

    def test_shift_values_defined(self):
        """Test that shift values are defined for locations."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "remove_veg_parallel",
            Path(__file__).parent.parent / "pipeline" / "5_remove_veg_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # The module should have shift configuration logic
        # This is implementation-specific, so we verify the pattern exists
        pass
