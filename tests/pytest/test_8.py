import numpy as np
import geopandas as gpd
import laspy
from shapely.geometry import Polygon
from importlib import util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "pipeline" / "8_make_grids.py"
    spec = util.spec_from_file_location("make_grids", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_las(path):
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.add_extra_dim(laspy.ExtraBytesParams(name="ClusterID", type=np.int32))
    hdr.add_extra_dim(laspy.ExtraBytesParams(name="M3C2_distance", type=np.float32))
    hdr.add_extra_dim(laspy.ExtraBytesParams(name="Uncertainty", type=np.float32))
    las = laspy.LasData(hdr)
    las.x = np.array([0.1, 0.2, 0.8])
    las.y = np.array([0.1, 0.2, 0.8])
    las.z = np.array([0.5, 0.7, 1.2])
    las.ClusterID = np.array([1, 1, 2], dtype=np.int32)
    las.M3C2_distance = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    las.Uncertainty = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    las.write(path)


def test_makeGrid_generates_pivot_csvs(tmp_path):
    mod = load_module()

    las_path = tmp_path / "clustered.las"
    make_las(las_path)

    shp_path = tmp_path / "polys.shp"
    gdf = gpd.GeoDataFrame({"name": ["poly"]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:32611")
    gdf.to_file(shp_path)

    out_grid = tmp_path / "grid.csv"
    out_cluster = tmp_path / "cluster.csv"
    out_uncert = tmp_path / "uncert.csv"

    stats = mod.makeGrid(
        pathin=str(las_path),
        pathout_m3c2=str(out_grid),
        pathout_cluster=str(out_cluster),
        pathout_uncertainty=str(out_uncert),
        polys=str(shp_path),
        res=0.5,
        height=2.0,
        overwrite=True,
    )

    assert out_grid.exists() and out_cluster.exists() and out_uncert.exists()
    assert stats["n_polygons"] == 1
    assert stats["clustered_points"] == 3
