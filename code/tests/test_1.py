import csv
from importlib import util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "pipeline" / "1_update_survey_lists.py"
    spec = util.spec_from_file_location("update_survey_lists", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_location_appends_new_survey(tmp_path, monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "ROOT_LIDAR", str(tmp_path))
    location = "TestSite"
    monkeypatch.setattr(mod, "mop_ranges", {location: (0, 10)})

    csv_dir = tmp_path / "survey_lists"
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(mod, "CSV_DIR", str(csv_dir))
    monkeypatch.setattr(mod, "REPORT_DIR", str(report_dir))

    inst_dir = tmp_path / "inst"
    inst_dir.mkdir()
    monkeypatch.setattr(mod, "instrument_paths", {"inst": str(inst_dir)})

    survey_dir = inst_dir / "20240202_0_10"
    survey_dir.mkdir()
    target_las = survey_dir / "test_beach_cliff_ground.las"
    target_las.touch()

    mod.init_report()
    assert mod.process_location(location) is True

    out_csv = csv_dir / f"surveys_{location}.csv"
    assert out_csv.exists()
    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["path"].endswith("20240202_0_10")
    assert (report_dir / mod.get_report_path().name).exists()
