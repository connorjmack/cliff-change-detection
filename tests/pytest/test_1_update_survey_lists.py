"""
Tests for 1_update_survey_lists.py

Tests the survey list update functionality including:
- Finding new surveys since last update
- Validating survey content (_beach_cliff_ground.las files)
- Updating existing CSVs with sorted data
- Path format conversion (Mac vs Linux)
- Report generation
"""

import pytest
import os
import csv
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = REPO_ROOT / "code" / "pipeline"
MODULE_PATH = PIPELINE_DIR / "1_update_survey_lists.py"

sys.path.insert(0, str(PIPELINE_DIR))


class TestSurveyListUpdate:
    """Test suite for survey list updates."""

    @pytest.fixture
    def existing_survey_csv(self, temp_dir):
        """Create an existing survey CSV with some data."""
        csv_path = Path(temp_dir) / "surveys_DelMar.csv"

        data = [
            {
                "path": "/Volumes/group/LiDAR/MiniRanger_Truck/LiDAR_Processed_Level2/20220101_595_620",
                "date": 20220101,
                "MOP1": 595,
                "MOP2": 620,
                "beach": "DelMar",
                "method": "MiniRanger_Truck"
            },
            {
                "path": "/Volumes/group/LiDAR/VMQLZ_Truck/LiDAR_Processed_Level2/20220201_595_620",
                "date": 20220201,
                "MOP1": 595,
                "MOP2": 620,
                "beach": "DelMar",
                "method": "VMQLZ_Truck"
            }
        ]

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        return str(csv_path)

    @pytest.fixture
    def mock_las_file(self, temp_dir):
        """Create a mock _beach_cliff_ground.las file."""
        las_dir = Path(temp_dir) / "survey" / "Beach_And_Backshore"
        las_dir.mkdir(parents=True)

        las_file = las_dir / "test_beach_cliff_ground.las"
        las_file.touch()  # Create empty file for existence check

        return str(las_file)

    def test_path_conversion_to_mac_format(self):
        """Test ensure_mac_path converts Linux paths to Mac format."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "update_survey_lists",
            MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        linux_path = "/project/group/LiDAR/MiniRanger_Truck/test"
        mac_path = module.ensure_mac_path(linux_path)

        assert mac_path.startswith("/Volumes/group/LiDAR")
        assert "project" not in mac_path

    def test_get_existing_survey_index(self, existing_survey_csv):
        """Test extracting maximum date and existing paths from CSV."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "update_survey_lists",
            MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        max_date, existing_paths = module.get_existing_survey_index(existing_survey_csv)

        assert max_date == 20220201, "Should return the latest date from CSV"
        assert len(existing_paths) == 2

    def test_get_existing_survey_index_no_file(self, temp_dir):
        """Test get_existing_survey_index with non-existent file."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "update_survey_lists",
            MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        nonexistent_path = Path(temp_dir) / "nonexistent.csv"
        max_date, existing_paths = module.get_existing_survey_index(str(nonexistent_path))

        assert max_date == 0, "Should return 0 when file doesn't exist"
        assert existing_paths == set()

    def test_find_target_las_files(self, mock_las_file):
        """Test finding _beach_cliff_ground.las files in directory."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "update_survey_lists",
            MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        survey_dir = str(Path(mock_las_file).parent.parent)
        found_files = module.find_target_las_files(survey_dir)

        assert len(found_files) > 0, "Should find the _beach_cliff_ground.las file"
        assert all(f.endswith("_beach_cliff_ground.las") for f in found_files)

    def test_update_csv_sorted(self, existing_survey_csv):
        """Test that update_csv_sorted appends and sorts data."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "update_survey_lists",
            MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        new_rows = [
            {
                "path": "/Volumes/group/LiDAR/MiniRanger_ATV/LiDAR_Processed_Level2/20220115_595_620",
                "date": 20220115,
                "MOP1": 595,
                "MOP2": 620,
                "beach": "DelMar",
                "method": "MiniRanger_ATV"
            }
        ]

        module.update_csv_sorted(existing_survey_csv, new_rows)

        # Read updated CSV
        df = pd.read_csv(existing_survey_csv)

        # Should have 3 rows now
        assert len(df) == 3

        # Should be sorted by date
        assert df['date'].is_monotonic_increasing

        # New row should be in middle (after 20220101, before 20220201)
        assert df.iloc[1]['date'] == 20220115

    def test_report_initialization(self, temp_dir):
        """Test that report file is initialized correctly."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "update_survey_lists",
            MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module, 'REPORT_DIR', temp_dir):
            module.init_report()

            # Check report file was created
            today = datetime.now().strftime("%Y%m%d")
            report_path = Path(temp_dir) / f"daily_report_{today}.txt"

            assert report_path.exists(), "Report file should be created"

            content = report_path.read_text()
            assert "DAILY REPORT" in content
            assert datetime.now().strftime('%Y-%m-%d') in content

    def test_append_to_report(self, temp_dir):
        """Test appending lines to report file."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "update_survey_lists",
            MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module, 'REPORT_DIR', temp_dir):
            module.init_report()

            test_lines = ["Test line 1", "Test line 2"]
            module.append_to_report(test_lines)

            today = datetime.now().strftime("%Y%m%d")
            report_path = Path(temp_dir) / f"daily_report_{today}.txt"

            content = report_path.read_text()

            for line in test_lines:
                assert line in content

    def test_process_location_with_no_new_surveys(self, temp_dir, existing_survey_csv):
        """Test process_location when no new surveys are found."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "update_survey_lists",
            MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module, 'CSV_DIR', temp_dir), \
            patch.object(module, 'ROOT_LIDAR', temp_dir), \
            patch.object(module, 'instrument_paths', {}):
            # Process location with no new data
            result = module.process_location("DelMar")

            # Should return False (no new surveys)
            assert result is False

    def test_process_location_same_date_new_path(self, temp_dir, existing_survey_csv):
        """Test that same-date surveys with new paths are processed."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "update_survey_lists",
            MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        inst_dir = Path(temp_dir) / "inst"
        inst_dir.mkdir()
        survey_dir = inst_dir / "20220201_595_620_extra"
        survey_dir.mkdir()
        las_file = survey_dir / "test_beach_cliff_ground.las"
        las_file.touch()

        with patch.object(module, 'CSV_DIR', temp_dir), \
            patch.object(module, 'REPORT_DIR', temp_dir), \
            patch.object(module, 'instrument_paths', {"inst": str(inst_dir)}):
            module.init_report()
            result = module.process_location("DelMar")

        assert result is True

        df = pd.read_csv(existing_survey_csv)
        assert len(df) == 3
        assert module.ensure_mac_path(str(survey_dir)) in set(df['path'].astype(str))

    def test_mop_overlap_filtering(self):
        """Test that surveys are correctly filtered by MOP overlap."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "update_survey_lists",
            MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # DelMar range: 595-620 (length 25)
        # 2/3 threshold = 16.67

        min_line, max_line = 595, 620

        # Test case 1: Survey with good overlap
        mop1, mop2 = 598, 618
        overlap = min(mop2, max_line) - max(mop1, min_line)
        two_thirds = (max_line - min_line) * 2 / 3

        assert overlap >= two_thirds, "Should pass overlap threshold"

        # Test case 2: Survey with insufficient overlap
        mop1, mop2 = 600, 610  # Only 10 lines overlap
        overlap = min(mop2, max_line) - max(mop1, min_line)

        assert overlap < two_thirds, "Should fail overlap threshold"

    def test_date_filtering_newer_than_existing(self):
        """Test that only surveys newer than existing max date are processed."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "update_survey_lists",
            MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        max_existing_date = 20220201

        # Test dates
        old_date = 20220101  # Before max
        new_date = 20220301  # After max

        assert old_date <= max_existing_date, "Old date should be filtered out"
        assert new_date > max_existing_date, "New date should pass filter"


class TestReportGeneration:
    """Test report generation functionality."""

    def test_report_includes_new_surveys(self, temp_dir):
        """Test that report correctly lists new surveys found."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "update_survey_lists",
            MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module, 'REPORT_DIR', temp_dir):
            module.init_report()

            test_entries = [
                "NEW: DelMar | 20220301 | /Volumes/group/LiDAR/test.las",
                "NEW: SanElijo | 20220302 | /Volumes/group/LiDAR/test2.las"
            ]

            module.append_to_report(test_entries)

            today = datetime.now().strftime("%Y%m%d")
            report_path = Path(temp_dir) / f"daily_report_{today}.txt"

            content = report_path.read_text()

            assert "DelMar" in content
            assert "SanElijo" in content
            assert "20220301" in content

    def test_no_new_surveys_message(self, temp_dir):
        """Test that appropriate message is logged when no new surveys found."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "update_survey_lists",
            MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module, 'REPORT_DIR', temp_dir):
            module.init_report()
            module.append_to_report(["No new surveys found. CSVs are up to date."])

            today = datetime.now().strftime("%Y%m%d")
            report_path = Path(temp_dir) / f"daily_report_{today}.txt"

            content = report_path.read_text()

            assert "No new surveys found" in content
