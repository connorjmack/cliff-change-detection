"""
Tests for 2_crop_files_parallel.py

Tests the PDAL-based cropping functionality including:
- Polygon generation from KML MOP lines
- PDAL pipeline execution
- Parallel processing
- Report generation
"""

import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from shapely.geometry import Polygon
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = REPO_ROOT / "code" / "pipeline"
MODULE_PATH = PIPELINE_DIR / "2_crop_files_parallel.py"

sys.path.insert(0, str(PIPELINE_DIR))


def load_module():
    """Load step 2 module fresh for each test."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "crop_files_parallel",
        MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCropPolygonGeneration:
    """Test crop polygon generation from MOP lines."""

    def test_extend_line_function(self):
        """Test line extension calculation."""
        module = load_module()

        p1 = (0, 0)
        p2 = (10, 0)
        along = 5

        extended = module.extend_line(p1, p2, along)

        # Should extend 5 units in both directions
        assert extended[0][0] == -5  # Extended backwards
        assert extended[1][0] == 15  # Extended forwards

    def test_load_existing_crop_polygon(self, temp_dir):
        """Test loading an existing crop polygon from file."""
        module = load_module()

        cropbox_dir = Path(temp_dir) / "cropboxes"
        cropbox_dir.mkdir()

        # Create a test cropbox file
        cropbox_file = cropbox_dir / "TestLocation_cropbox.txt"
        cropbox_data = (
            "480000.0 3650000.0 0.0\n"
            "481000.0 3650000.0 0.0\n"
            "481000.0 3651000.0 0.0\n"
            "480000.0 3651000.0 0.0\n"
        )
        cropbox_file.write_text(cropbox_data)

        with patch.object(module, 'CROPBOX_DIR', str(cropbox_dir)):
            # Load the polygon
            poly = module.load_or_create_crop_polygon("TestLocation", 595, 620)

            assert isinstance(poly, Polygon)
            assert not poly.is_empty
            assert poly.area > 0
            # Should have 4 corners (plus closing point)
            assert len(poly.exterior.coords) == 5

    def test_create_crop_polygon_from_kml(self, mock_kml_file, temp_dir):
        """Test creating a new crop polygon from KML."""
        module = load_module()

        cropbox_dir = Path(temp_dir) / "cropboxes"
        cropbox_dir.mkdir()

        with patch.object(module, 'MOP_KML', mock_kml_file), \
            patch.object(module, 'CROPBOX_DIR', str(cropbox_dir)):
            # Create polygon for DelMar (MOP 595-620)
            poly = module.load_or_create_crop_polygon("DelMar", 595, 620)

            assert isinstance(poly, Polygon)
            assert not poly.is_empty
            assert poly.area > 0

            # Check cropbox file was saved
            cropbox_file = cropbox_dir / "DelMar_cropbox.txt"
            assert cropbox_file.exists()


class TestPDALProcessing:
    """Test PDAL pipeline construction and execution."""

    def test_worker_creates_valid_pdal_pipeline(self, sample_las_file, temp_dir):
        """Test that worker function creates a valid PDAL pipeline."""
        module = load_module()

        # Create a simple polygon WKT
        poly_wkt = "POLYGON ((480000 3650000, 481000 3650000, 481000 3651000, 480000 3651000, 480000 3650000))"

        las_out = str(Path(temp_dir) / "output.las")

        # Mock pdal.Pipeline to capture the pipeline configuration
        with patch('pdal.Pipeline') as mock_pipeline:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_instance.execute.return_value = 100  # 100 points
            mock_pipeline.return_value = mock_pipeline_instance

            task = (poly_wkt, sample_las_file, las_out, False, True)
            result = module._worker(task)

            # Check that Pipeline was called
            assert mock_pipeline.called

            # Get the pipeline JSON that was passed
            pipeline_json = mock_pipeline.call_args[0][0]
            pipeline_config = json.loads(pipeline_json)

            # Verify pipeline structure
            assert "pipeline" in pipeline_config
            assert len(pipeline_config["pipeline"]) == 4  # reader, crop, sample, writer

            # Check filters
            filter_types = [stage["type"] for stage in pipeline_config["pipeline"]]
            assert "readers.las" in filter_types
            assert "filters.crop" in filter_types
            assert "filters.sample" in filter_types
            assert "writers.las" in filter_types

    def test_worker_no_sample_pipeline(self, sample_las_file, temp_dir):
        """Test that worker omits sampling when disabled."""
        module = load_module()

        poly_wkt = "POLYGON ((480000 3650000, 481000 3650000, 481000 3651000, 480000 3651000, 480000 3650000))"
        las_out = str(Path(temp_dir) / "output.las")

        with patch('pdal.Pipeline') as mock_pipeline:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_instance.execute.return_value = 100
            mock_pipeline.return_value = mock_pipeline_instance

            task = (poly_wkt, sample_las_file, las_out, False, False)
            module._worker(task)

            pipeline_json = mock_pipeline.call_args[0][0]
            pipeline_config = json.loads(pipeline_json)

            filter_types = [stage["type"] for stage in pipeline_config["pipeline"]]
            assert "filters.sample" not in filter_types
            assert len(pipeline_config["pipeline"]) == 3  # reader, crop, writer

    def test_worker_skip_existing_file(self, sample_las_file, temp_dir):
        """Test that worker skips existing output when replace=False."""
        module = load_module()

        poly_wkt = "POLYGON ((480000 3650000, 481000 3650000, 481000 3651000, 480000 3651000, 480000 3650000))"
        las_out = str(Path(temp_dir) / "existing.las")

        # Create the output file
        Path(las_out).touch()

        task = (poly_wkt, sample_las_file, las_out, False, True)  # replace=False
        result = module._worker(task)

        assert result["status"] == "skipped"
        assert result["points"] == 0

    def test_worker_error_handling(self, temp_dir):
        """Test worker error handling for invalid input."""
        module = load_module()

        poly_wkt = "POLYGON ((480000 3650000, 481000 3650000, 481000 3651000, 480000 3651000, 480000 3650000))"
        invalid_las = "/nonexistent/file.las"
        las_out = str(Path(temp_dir) / "output.las")

        task = (poly_wkt, invalid_las, las_out, False, True)
        result = module._worker(task)

        assert result["status"] == "error"
        assert "error_msg" in result
        assert len(result["error_msg"]) > 0


class TestReportGeneration:
    """Test cropping report generation."""

    def test_write_report_creates_file(self, temp_dir):
        """Test that write_report creates a report file."""
        module = load_module()

        stats = [
            {"file": "test1.las", "status": "processed", "duration": 1.5, "points": 1000},
            {"file": "test2.las", "status": "processed", "duration": 2.0, "points": 1500},
            {"file": "test3.las", "status": "skipped", "duration": 0.0, "points": 0},
        ]

        module.write_report("TestLocation", stats, 10.5, temp_dir)

        # Check report was created
        report_dir = Path(temp_dir) / "pipeline_reports"
        report_files = list(report_dir.glob("cropping_report_*.txt"))

        assert len(report_files) == 1

        # Check report content
        content = report_files[0].read_text()
        assert "TestLocation" in content
        assert "Files Processed:   2" in content
        assert "Files Skipped:     1" in content
        assert "10.5" in content  # Total duration

    def test_report_includes_error_details(self, temp_dir):
        """Test that report includes error messages."""
        module = load_module()

        stats = [
            {"file": "error.las", "status": "error", "error_msg": "File not found", "duration": 0.5, "points": 0}
        ]

        module.write_report("TestLocation", stats, 1.0, temp_dir)

        report_dir = Path(temp_dir) / "pipeline_reports"
        report_files = list(report_dir.glob("cropping_report_*.txt"))

        content = report_files[0].read_text()
        assert "FAIL" in content
        assert "File not found" in content


class TestMOPRangeConfiguration:
    """Test MOP range configurations."""

    def test_all_locations_have_mop_ranges(self):
        """Test that all locations have valid MOP range definitions."""
        module = load_module()

        expected_locations = ["DelMar", "Solana", "Encinitas", "SanElijo", "Torrey", "Blacks"]

        for location in expected_locations:
            assert location in module.mop_ranges
            min_mop, max_mop = module.mop_ranges[location]
            assert min_mop < max_mop
            assert isinstance(min_mop, int)
            assert isinstance(max_mop, int)
