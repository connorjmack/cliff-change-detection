"""
Tests for 6_m3c2_parallel.py

Tests M3C2 change detection via CloudCompare including:
- Date extraction from filenames
- CloudCompare command construction
- Sequential pair processing
- Report generation
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))


class TestDateExtraction:
    """Test date extraction from filenames."""

    def test_extract_date_from_filename(self):
        """Test extracting date from standard filename format."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "m3c2_parallel",
            Path(__file__).parent.parent / "pipeline" / "6_m3c2_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        filename = "20220101_595_620_beach_cliff_ground_cropped_nobeach_noveg.las"
        date = module.extract_date(filename)

        assert date == "20220101"

    def test_extract_date_from_path(self):
        """Test extracting date from full path."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "m3c2_parallel",
            Path(__file__).parent.parent / "pipeline" / "6_m3c2_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        path = "/path/to/20220215_683_708_noveg.las"
        date = module.extract_date(path)

        assert date == "20220215"


class TestM3C2Computation:
    """Test M3C2 computation functionality."""

    def test_compute_m3c2_creates_output_structure(self, sample_las_file, temp_dir, mock_subprocess_run):
        """Test that M3C2 computation creates proper output structure."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "m3c2_parallel",
            Path(__file__).parent.parent / "pipeline" / "6_m3c2_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ref_path = sample_las_file
        cmp_path = sample_las_file  # Using same file for simplicity
        params_file = str(Path(temp_dir) / "params.txt")
        Path(params_file).touch()

        shift = ('-475000', '-3653000', '0')
        cc_path = "/usr/bin/CloudCompare"

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = module.compute_m3c2_and_save_all(
                ref_path,
                cmp_path,
                params_file,
                temp_dir,
                shift,
                cc_path,
                replace=False
            )

            assert "pair" in result
            assert "status" in result
            assert "duration_sec" in result
            assert mock_run.called

    def test_compute_m3c2_skip_existing(self, sample_las_file, temp_dir):
        """Test skipping existing M3C2 outputs."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "m3c2_parallel",
            Path(__file__).parent.parent / "pipeline" / "6_m3c2_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Create mock output directory with required files
        pair_dir = Path(temp_dir) / "20220101_to_20220101"
        pair_dir.mkdir()
        (pair_dir / "20220101.las").touch()
        (pair_dir / "20220101_to_20220101_m3c2.las").touch()

        result = module.compute_m3c2_and_save_all(
            sample_las_file,
            sample_las_file,
            str(Path(temp_dir) / "params.txt"),
            temp_dir,
            ('-475000', '-3653000', '0'),
            "/usr/bin/CloudCompare",
            replace=False
        )

        assert result["status"] == "SKIPPED"


class TestReportGeneration:
    """Test M3C2 report generation."""

    def test_write_robust_report_creates_files(self, temp_dir):
        """Test that report generation creates CSV and TXT files."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "m3c2_parallel",
            Path(__file__).parent.parent / "pipeline" / "6_m3c2_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        stats = [
            {
                "pair": "20220101_to_20220201",
                "ref_date": "20220101",
                "cmp_date": "20220201",
                "status": "SUCCESS",
                "duration_sec": 120.5,
                "input_mb": 25.3,
                "error_message": ""
            }
        ]

        args_dict = {"location": "TestLocation", "test": None}

        module.write_robust_report("TestLocation", stats, temp_dir, args_dict, 150.0)

        # Check CSV was created
        csv_files = list(Path(temp_dir).glob("*_m3c2_inventory_*.csv"))
        assert len(csv_files) == 1

        # Check TXT was created
        txt_files = list(Path(temp_dir).glob("*_m3c2_summary_*.txt"))
        assert len(txt_files) == 1

        # Check content
        txt_content = txt_files[0].read_text()
        assert "TestLocation" in txt_content
        assert "SUCCESS" in txt_content


class TestErrorHandling:
    """Test error handling in M3C2 processing."""

    def test_safe_execute_wrapper(self):
        """Test safe_execute wrapper for multiprocessing."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "m3c2_parallel",
            Path(__file__).parent.parent / "pipeline" / "6_m3c2_parallel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # The safe_execute function should catch exceptions
        # This is a wrapper for multiprocessing safety
        assert hasattr(module, 'safe_execute')
