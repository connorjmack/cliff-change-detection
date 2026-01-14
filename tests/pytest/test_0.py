import csv
from importlib import util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "pipeline" / "0_make_survey_lists.py"
    spec = util.spec_from_file_location("make_survey_lists", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_surveys_writes_sorted_csv(tmp_path, monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "ROOT_LIDAR", str(tmp_path))
    location = "TestSite"
    monkeypatch.setattr(mod, "mop_ranges", {location: [0, 10]})

    inst_dir = tmp_path / "inst"
    inst_dir.mkdir()
    monkeypatch.setattr(mod, "instrument_paths", {"inst": str(inst_dir)})

    (inst_dir / "20240101_0_10_extra").mkdir()
    (inst_dir / "20230101_0_10").mkdir()

    mod.export_surveys(location)

    out_csv = tmp_path / "LidarProcessing" / "LidarProcessingCliffs" / "survey_lists" / f"surveys_{location}.csv"
    assert out_csv.exists()

    with out_csv.open() as f:
        rows = list(csv.DictReader(f))

    assert [r["date"] for r in rows] == ["20230101", "20240101"]
    assert rows[0]["beach"] == location
