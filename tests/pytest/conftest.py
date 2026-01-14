"""
Shared pytest fixtures for cliff-change-detection pipeline tests.

This module provides common test fixtures for mocking external dependencies,
creating temporary test data, and setting up test environments.
"""

import pytest
import tempfile
import os
import numpy as np
import laspy
from pathlib import Path
from unittest.mock import Mock, MagicMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_lidar_root(temp_dir):
    """Create a mock LiDAR root directory structure."""
    root = Path(temp_dir) / "group" / "LiDAR"
    root.mkdir(parents=True)

    # Create instrument directories
    instruments = ["MiniRanger_Truck", "MiniRanger_ATV", "VMQLZ_Truck", "VMZ2000_Truck"]
    for instrument in instruments:
        instr_path = root / instrument / "LiDAR_Processed_Level2"
        instr_path.mkdir(parents=True)

    # Create project structure
    project = root / "LidarProcessing" / "LidarProcessingCliffs"
    (project / "survey_lists").mkdir(parents=True)
    (project / "results").mkdir(parents=True)
    (project / "utilities" / "shape_files").mkdir(parents=True)
    (project / "utilities" / "beach_removal" / "joblib_files").mkdir(parents=True)
    (project / "utilities" / "canupo").mkdir(parents=True)
    (project / "utilities" / "m3c2_params").mkdir(parents=True)

    return str(root)


@pytest.fixture
def mock_survey_folder(mock_lidar_root):
    """Create a mock survey folder with sample LAS file."""
    survey_name = "20220101_595_620"
    survey_path = Path(mock_lidar_root) / "MiniRanger_Truck" / "LiDAR_Processed_Level2" / survey_name
    survey_path.mkdir(parents=True)

    # Create Beach_And_Backshore subdirectory
    beach_dir = survey_path / "Beach_And_Backshore"
    beach_dir.mkdir(parents=True)

    return str(survey_path)


@pytest.fixture
def sample_las_file(mock_survey_folder):
    """Create a minimal valid LAS file for testing."""
    beach_dir = Path(mock_survey_folder) / "Beach_And_Backshore"
    las_path = beach_dir / "20220101_beach_cliff_ground.las"

    # Create a minimal LAS file with sample data
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [0, 0, 0]
    header.scales = [0.01, 0.01, 0.01]

    las = laspy.LasData(header)

    # Add 100 sample points
    n_points = 100
    las.x = np.random.uniform(480000, 481000, n_points)
    las.y = np.random.uniform(3650000, 3651000, n_points)
    las.z = np.random.uniform(0, 50, n_points)
    las.intensity = np.random.randint(0, 65535, n_points, dtype=np.uint16)

    las.write(str(las_path))
    return str(las_path)


@pytest.fixture
def sample_m3c2_las_file(temp_dir):
    """Create a sample M3C2 output LAS file with distance and significance fields."""
    las_path = Path(temp_dir) / "test_m3c2.las"

    # Create LAS with extra dimensions for M3C2
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [0, 0, 0]
    header.scales = [0.01, 0.01, 0.01]

    las = laspy.LasData(header)

    # Add sample points
    n_points = 200
    las.x = np.random.uniform(480000, 481000, n_points)
    las.y = np.random.uniform(3650000, 3651000, n_points)
    las.z = np.random.uniform(0, 50, n_points)

    # Add M3C2 distance field (half erosion, half deposition)
    las.add_extra_dim(laspy.ExtraBytesParams(name="M3C2_distance", type=np.float32))
    distances = np.concatenate([
        np.random.uniform(-1.0, -0.1, n_points // 2),  # Erosion
        np.random.uniform(0.1, 1.0, n_points // 2)     # Deposition
    ])
    las.M3C2_distance = distances

    # Add significance field
    las.add_extra_dim(laspy.ExtraBytesParams(name="significant_change", type=np.uint8))
    las.significant_change = np.random.choice([0, 1], n_points, p=[0.3, 0.7])

    # Add uncertainty field
    las.add_extra_dim(laspy.ExtraBytesParams(name="distance_uncertainty", type=np.float32))
    las.distance_uncertainty = np.random.uniform(0.01, 0.1, n_points)

    las.write(str(las_path))
    return str(las_path)


@pytest.fixture
def sample_clustered_las_file(sample_m3c2_las_file):
    """Create a LAS file with cluster IDs (for testing grid making)."""
    # Read the M3C2 file and add ClusterID
    las = laspy.read(sample_m3c2_las_file)

    # Add ClusterID field
    n_points = len(las.points)
    las.add_extra_dim(laspy.ExtraBytesParams(name="ClusterID", type=np.int32))

    # Assign some points to clusters, some as noise (-1)
    cluster_ids = np.random.choice([-1, 0, 1, 2, 3], n_points, p=[0.2, 0.2, 0.2, 0.2, 0.2])
    las.ClusterID = cluster_ids

    output_path = str(Path(sample_m3c2_las_file).parent / "test_clustered.las")
    las.write(output_path)
    return output_path


@pytest.fixture
def mock_kml_file(temp_dir):
    """Create a mock KML file with MOP lines."""
    kml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>MOP_595</name>
      <LineString>
        <coordinates>
          -117.25,32.92,0 -117.24,32.93,0
        </coordinates>
      </LineString>
    </Placemark>
    <Placemark>
      <name>MOP_620</name>
      <LineString>
        <coordinates>
          -117.23,32.94,0 -117.22,32.95,0
        </coordinates>
      </LineString>
    </Placemark>
    <Placemark>
      <name>MOP_683</name>
      <LineString>
        <coordinates>
          -117.28,33.01,0 -117.27,33.02,0
        </coordinates>
      </LineString>
    </Placemark>
    <Placemark>
      <name>MOP_708</name>
      <LineString>
        <coordinates>
          -117.26,33.03,0 -117.25,33.04,0
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>'''

    kml_path = Path(temp_dir) / "MOPs_SD_County.kml"
    kml_path.write_text(kml_content)
    return str(kml_path)


@pytest.fixture
def mock_rf_model(temp_dir):
    """Create a mock Random Forest model and scaler for beach removal testing."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # Create a simple RF model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    # Train on dummy data (4 features: x, y, z, intensity)
    X_dummy = np.random.rand(100, 4)
    y_dummy = np.random.choice([0, 1], 100)  # 0=cliff, 1=beach
    model.fit(X_dummy, y_dummy)

    # Create scaler for intensity normalization
    scaler = StandardScaler()
    scaler.fit(np.random.rand(100, 1))

    model_path = Path(temp_dir) / "test_model.joblib"
    scaler_path = Path(temp_dir) / "test_scaler.joblib"

    joblib.dump(model, str(model_path))
    joblib.dump(scaler, str(scaler_path))

    return str(model_path), str(scaler_path)


@pytest.fixture
def mock_subprocess_run(monkeypatch):
    """Mock subprocess.run for CloudCompare testing."""
    mock_result = MagicMock()
    mock_result.returncode = 0

    def mock_run(*args, **kwargs):
        return mock_result

    monkeypatch.setattr("subprocess.run", mock_run)
    return mock_run


@pytest.fixture
def location_configs():
    """Return standard location configurations."""
    return {
        "DelMar": {
            "mop_range": (595, 620),
            "shift": ('-475000', '-3653000', '0')
        },
        "SanElijo": {
            "mop_range": (683, 708),
            "shift": ('-473000', '-3653000', '0')
        },
        "Solana": {
            "mop_range": (637, 666),
            "shift": ('-475000', '-3653000', '0')
        }
    }
