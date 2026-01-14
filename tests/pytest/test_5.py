import shutil
import laspy
import numpy as np
from importlib import util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "pipeline" / "5_remove_veg_parallel.py"
    spec = util.spec_from_file_location("remove_veg_parallel", path)
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


def test_classify_file_with_stats_invokes_cloudcompare(tmp_path, monkeypatch):
    mod = load_module()
    inp = tmp_path / "sample.las"
    make_las(inp, 4)

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def fake_run(cmd, check, stdout, stderr):
        out_path = Path(cmd[-1])
        shutil.copy(inp, out_path)
        return None

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    stats = mod.classify_file_with_stats(
        str(inp),
        classifier_prm="classifier.prm",
        output_dir=str(output_dir),
        shift=("-1", "-1", "0"),
        cc_path="cloudcompare",
        replace=True,
    )

    out_file = output_dir / "sample_noveg.las"
    assert out_file.exists()
    assert stats["status"] == "Success"
    assert stats["removed_points"] == 0
