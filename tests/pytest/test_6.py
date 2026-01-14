import shutil
from importlib import util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "pipeline" / "6_m3c2_parallel.py"
    spec = util.spec_from_file_location("m3c2_parallel", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_m3c2_and_save_all_creates_outputs(tmp_path, monkeypatch):
    mod = load_module()

    ref = tmp_path / "ref.las"
    cmp_file = tmp_path / "cmp.las"
    ref.write_text("ref")
    cmp_file.write_text("cmp")

    def fake_run(cmd, check):
        out_paths = cmd[-1].split()
        for p in out_paths:
            shutil.copy(ref, p)
        return None

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    result = mod.compute_m3c2_and_save_all(
        str(ref),
        str(cmp_file),
        params_file="params.txt",
        base_output_dir=str(tmp_path / "out"),
        shift=("-1", "-1", "0"),
        cc_path="cloudcompare",
        replace=True,
    )

    pair_dir = tmp_path / "out" / f"{ref.stem}_to_{cmp_file.stem}"
    assert (pair_dir / f"{ref.stem}.las").exists()
    assert result["status"] == "SUCCESS"
