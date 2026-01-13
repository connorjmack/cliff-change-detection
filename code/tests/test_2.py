from importlib import util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "pipeline" / "2_crop_files_parallel.py"
    spec = util.spec_from_file_location("crop_files_parallel", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extend_line_expands_length():
    mod = load_module()
    p1 = (0.0, 0.0)
    p2 = (1.0, 0.0)
    extended = mod.extend_line(p1, p2, 2.0)
    assert extended[0][0] < p1[0] and extended[1][0] > p2[0]


def test_load_or_create_crop_polygon_creates_file(tmp_path, monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "CROPBOX_DIR", str(tmp_path / "crop"))
    kml_path = tmp_path / "mop.kml"
    monkeypatch.setattr(mod, "MOP_KML", str(kml_path))

    kml_text = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark><name>MOP_0</name><LineString><coordinates>0,0,0 0,1,0</coordinates></LineString></Placemark>
    <Placemark><name>MOP_10</name><LineString><coordinates>1,0,0 1,1,0</coordinates></LineString></Placemark>
  </Document>
</kml>"""
    kml_path.write_text(kml_text)

    poly = mod.load_or_create_crop_polygon("TestSite", 0, 10)
    crop_file = Path(mod.CROPBOX_DIR) / "TestSite_cropbox.txt"
    assert crop_file.exists()
    assert not poly.is_empty
