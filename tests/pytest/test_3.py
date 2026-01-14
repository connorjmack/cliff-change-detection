import numpy as np
import laspy
from importlib import util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "audits" / "2_audit_cropping.py"
    spec = util.spec_from_file_location("qc_cropped_files", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_las(path, n_points):
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(hdr)
    las.x = np.arange(n_points, dtype=float)
    las.y = np.arange(n_points, dtype=float)
    las.z = np.zeros(n_points, dtype=float)
    las.intensity = np.ones(n_points, dtype=np.uint16)
    las.write(path)


def test_analyze_file_reports_point_count(tmp_path, monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "MIN_POINT_THRESHOLD", 1)
    las_path = tmp_path / "sample.las"
    make_las(las_path, 5)

    result = mod.analyze_file(str(las_path))
    assert result["Point_Count"] == 5
    assert result["Status"] == "OK"


def test_analyze_file_flags_small_cloud(tmp_path, monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "MIN_POINT_THRESHOLD", 10)
    las_path = tmp_path / "tiny.las"
    make_las(las_path, 2)

    result = mod.analyze_file(str(las_path))
    assert result["Status"] == "SUSPECT"
