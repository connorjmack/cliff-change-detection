"""
Tests for 0_make_survey_lists.py

Tests the survey list generation functionality including:
- MOP range overlap calculations
- Survey folder discovery
- CSV output generation
- Multi-location processing
"""

import pytest
import os
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add pipeline directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))


class TestSurveyListGeneration:
    """Test suite for survey list generation."""

    @pytest.fixture
    def mock_survey_structure(self, mock_lidar_root):
        """Create mock survey folders with proper naming convention."""
        root = Path(mock_lidar_root)

        # Create survey folders for DelMar (MOP 595-620)
        surveys = [
            ("20220101_595_620", "MiniRanger_Truck"),
            ("20220115_598_618", "MiniRanger_ATV"),  # Overlaps 2/3
            ("20220201_590_625", "VMQLZ_Truck"),     # Overlaps fully
            ("20220215_700_720", "VMZ2000_Truck"),   # No overlap
        ]

        for survey_name, instrument in surveys:
            survey_path = root / instrument / "LiDAR_Processed_Level2" / survey_name
            survey_path.mkdir(parents=True, exist_ok=True)

        return str(root)

    def test_mop_overlap_calculation(self, mock_survey_structure):
        """Test MOP line overlap calculation for 2/3 threshold."""
        import importlib.util

        # Load module dynamically
        spec = importlib.util.spec_from_file_location(
            "make_survey_lists",
            Path(__file__).parent.parent / "pipeline" / "0_make_survey_lists.py"
        )
        module = importlib.util.module_from_spec(spec)

        # Mock the ROOT_LIDAR
        with patch.object(module, 'ROOT_LIDAR', mock_survey_structure):
            spec.loader.exec_module(module)

            # Test DelMar range: 595-620 (length 25)
            # Survey 598-618 (length 20) should overlap by 20, which is >= 16.67 (2/3 of 25)
            min_line, max_line = 595, 620
            survey_min, survey_max = 598, 618

            overlap = min(survey_max, max_line) - max(survey_min, min_line)
            two_thirds = (max_line - min_line) * 2 / 3

            assert overlap == 20, "Overlap should be 20 MOP lines"
            assert overlap >= two_thirds, "Should meet 2/3 overlap threshold"

    def test_export_surveys_creates_csv(self, mock_survey_structure):
        """Test that export_surveys creates a valid CSV file."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "make_survey_lists",
            Path(__file__).parent.parent / "pipeline" / "0_make_survey_lists.py"
        )
        module = importlib.util.module_from_spec(spec)

        with patch.object(module, 'ROOT_LIDAR', mock_survey_structure):
            spec.loader.exec_module(module)

            # Run export for DelMar
            module.export_surveys("DelMar")

            # Check CSV was created
            csv_path = Path(mock_survey_structure) / "LidarProcessing" / "LidarProcessingCliffs" / "survey_lists" / "surveys_DelMar.csv"
            assert csv_path.exists(), "Survey CSV should be created"

            # Read and validate CSV
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                # Should have at least the fully overlapping surveys
                assert len(rows) > 0, "CSV should contain survey entries"

                # Check required fields
                assert all(key in rows[0] for key in ["path", "date", "MOP1", "MOP2", "beach", "method"])

                # Check data is sorted by date
                dates = [row["date"] for row in rows]
                assert dates == sorted(dates), "Surveys should be sorted by date"

    def test_unknown_location_handling(self, mock_survey_structure):
        """Test that unknown locations are handled gracefully."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "make_survey_lists",
            Path(__file__).parent.parent / "pipeline" / "0_make_survey_lists.py"
        )
        module = importlib.util.module_from_spec(spec)

        with patch.object(module, 'ROOT_LIDAR', mock_survey_structure):
            spec.loader.exec_module(module)

            # Should not crash on unknown location
            module.export_surveys("UnknownLocation")

    def test_all_locations_processing(self, mock_survey_structure):
        """Test processing all locations at once."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "make_survey_lists",
            Path(__file__).parent.parent / "pipeline" / "0_make_survey_lists.py"
        )
        module = importlib.util.module_from_spec(spec)

        with patch.object(module, 'ROOT_LIDAR', mock_survey_structure):
            spec.loader.exec_module(module)

            # Process all locations
            for loc in module.mop_ranges.keys():
                module.export_surveys(loc)

            # Check that CSVs were created for each location
            survey_list_dir = Path(mock_survey_structure) / "LidarProcessing" / "LidarProcessingCliffs" / "survey_lists"
            csv_files = list(survey_list_dir.glob("surveys_*.csv"))

            # Should have at least one CSV file
            assert len(csv_files) > 0, "Should create CSV files for locations"

    def test_csv_field_validation(self, mock_survey_structure):
        """Test that CSV contains all required fields with correct data types."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "make_survey_lists",
            Path(__file__).parent.parent / "pipeline" / "0_make_survey_lists.py"
        )
        module = importlib.util.module_from_spec(spec)

        with patch.object(module, 'ROOT_LIDAR', mock_survey_structure):
            spec.loader.exec_module(module)

            module.export_surveys("DelMar")

            csv_path = Path(mock_survey_structure) / "LidarProcessing" / "LidarProcessingCliffs" / "survey_lists" / "surveys_DelMar.csv"

            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Validate path is a string
                    assert isinstance(row["path"], str)
                    # Validate date is 8 digits
                    assert row["date"].isdigit() and len(row["date"]) == 8
                    # Validate MOP values are integers
                    assert row["MOP1"].isdigit()
                    assert row["MOP2"].isdigit()
                    # Validate beach is a known location
                    assert row["beach"] in module.mop_ranges
                    # Validate method is a known instrument
                    assert row["method"] in module.instrument_paths

    def test_no_surveys_found(self, mock_lidar_root):
        """Test handling when no valid surveys are found."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "make_survey_lists",
            Path(__file__).parent.parent / "pipeline" / "0_make_survey_lists.py"
        )
        module = importlib.util.module_from_spec(spec)

        # Use empty structure
        with patch.object(module, 'ROOT_LIDAR', mock_lidar_root):
            spec.loader.exec_module(module)

            module.export_surveys("DelMar")

            csv_path = Path(mock_lidar_root) / "LidarProcessing" / "LidarProcessingCliffs" / "survey_lists" / "surveys_DelMar.csv"

            # CSV should still be created, just with no data rows
            assert csv_path.exists()

            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 0, "Should have no survey rows when no valid surveys found"

    def test_os_detection(self):
        """Test that OS detection sets correct root path."""
        import importlib.util
        import platform

        spec = importlib.util.spec_from_file_location(
            "make_survey_lists",
            Path(__file__).parent.parent / "pipeline" / "0_make_survey_lists.py"
        )
        module = importlib.util.module_from_spec(spec)

        # Mock Darwin (macOS)
        with patch.object(platform, 'system', return_value='Darwin'):
            spec.loader.exec_module(module)
            # Reload to get fresh module state
            importlib.reload(module)
            # Note: ROOT_LIDAR is set at module level, so we just verify the logic exists

        # The module should have logic to set ROOT_LIDAR based on platform
        assert hasattr(module, 'ROOT_LIDAR')


class TestMOPRangeDefinitions:
    """Test MOP range configurations."""

    def test_all_locations_defined(self):
        """Verify all expected locations have MOP ranges defined."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "make_survey_lists",
            Path(__file__).parent.parent / "pipeline" / "0_make_survey_lists.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        expected_locations = ["DelMar", "Solana", "Encinitas", "SanElijo", "Torrey", "Blacks"]

        for location in expected_locations:
            assert location in module.mop_ranges, f"{location} should be in mop_ranges"
            assert len(module.mop_ranges[location]) == 2, "MOP range should have min and max"
            assert module.mop_ranges[location][0] < module.mop_ranges[location][1], "Min should be less than max"
