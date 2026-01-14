import numpy as np
from importlib import util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "pipeline" / "9_clean_fill_grids.py"
    spec = util.spec_from_file_location("clean_fill_grids", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_get_resolution_params_defaults():
    mod = load_module()
    params = mod.get_resolution_params("10cm")
    assert params["cell_size"] == 0.10
    assert params["default_threshold"] == 25


def test_save_and_load_csv_data_roundtrip(tmp_path):
    mod = load_module()
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    headers = ["A", "B"]
    rows = ["r1", "r2"]
    csv_path = tmp_path / "grid.csv"

    mod.save_csv_data(str(csv_path), data, headers, rows, testing=False, replace=True)
    header_labels, row_labels, loaded = mod.load_csv_data(str(csv_path))

    assert header_labels == headers
    assert row_labels == rows
    assert loaded.shape == data.shape
