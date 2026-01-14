import numpy as np
import laspy
from importlib import util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "pipeline" / "4_remove_beach_parallel.py"
    spec = util.spec_from_file_location("remove_beach_parallel", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyModel:
    n_features_in_ = 4

    def predict(self, X):
        labels = np.ones(len(X), dtype=int)
        labels[0] = 0
        return labels


class DummyScaler:
    def transform(self, arr):
        return arr * 0.01


def make_las(path):
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(hdr)
    las.x = np.array([0.0, 1.0])
    las.y = np.array([0.0, 1.0])
    las.z = np.array([0.0, 1.0])
    las.intensity = np.array([10, 20], dtype=np.uint16)
    las.write(path)


def test_classify_and_write_filters_points(tmp_path, monkeypatch):
    mod = load_module()
    inp = tmp_path / "input.las"
    make_las(inp)

    model = DummyModel()
    scaler = DummyScaler()

    def fake_load(path):
        return scaler if "scaler" in str(path) else model

    monkeypatch.setattr(mod.joblib, "load", fake_load)
    stats = mod.classify_and_write(str(inp), "model.joblib", "scaler.joblib", str(tmp_path), replace=True)

    out_file = tmp_path / "input_nobeach.las"
    assert out_file.exists()
    assert stats["status"] == "Success"
    assert stats["cliff_points_kept"] == 1
    assert stats["beach_points_removed"] == 1
