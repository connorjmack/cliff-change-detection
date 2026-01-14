import numpy as np
import laspy
from importlib import util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "pipeline" / "7_dbscan_parallel.py"
    spec = util.spec_from_file_location("dbscan_parallel", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_m3c2_las(path):
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.add_extra_dim(laspy.ExtraBytesParams(name="M3C2_distance", type=np.float32))
    hdr.add_extra_dim(laspy.ExtraBytesParams(name="significant_change", type=np.uint8))
    las = laspy.LasData(hdr)
    las.x = np.array([0.0, 0.1, 1.0, 1.1])
    las.y = np.array([0.0, 0.1, 1.0, 1.1])
    las.z = np.zeros(4, dtype=float)
    las.M3C2_distance = np.array([-0.5, -0.4, 0.6, 0.7], dtype=np.float32)
    las.significant_change = np.ones(4, dtype=np.uint8)
    las.write(path)


def test_run_dbscan_file_splits_outputs(tmp_path):
    mod = load_module()
    las_path = tmp_path / "m3c2.las"
    make_m3c2_las(las_path)

    erosion_dir = tmp_path / "erosion"
    deposition_dir = tmp_path / "deposition"

    stats = mod.run_dbscan_file(
        str(las_path),
        str(erosion_dir),
        str(deposition_dir),
        eps=0.5,
        min_samples=1,
        min_change_threshold=0.1,
        replace=True,
    )

    assert stats["status"] == "Success"
    assert stats["erosion_points"] > 0
    assert stats["deposition_points"] > 0
    assert (erosion_dir / las_path.stem / "dbscan.las").exists()
    assert (deposition_dir / las_path.stem / "dbscan.las").exists()
