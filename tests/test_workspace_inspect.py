"""Phase 5 API (inspection, archaeology, twin) end to end on the synthetic two-epoch site."""
import base64
import io
import json
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_ops_scene  # noqa: E402
import workspace_api as api  # noqa: E402
import workspace_inspect_api as ins  # noqa: E402


class Server:
    process_lock = threading.Lock()
    active_job_info = {}

    def job_busy_locked(self):
        return False


class InspectApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PIL import Image
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        make_ops_scene.write_epoch(cls.root / "work" / "opsbefore", after=False)
        make_ops_scene.write_epoch(cls.root / "work" / "opsafter", after=True)
        cams = json.loads((cls.root / "work/opsafter/viewer_assets/cameras.json").read_text())
        for cam in cams:
            path = cls.root / "work/opsafter/frames_full" / cam["name"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (cam["width"] * 2, cam["height"] * 2), (110, 120, 100)).save(path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def post(self, route, **data):
        return ins.post(api, self.root, route, dict(scene="opsafter", **data), Server())

    def get(self, route, **query):
        return ins.get(api, self.root, route, {k: [str(v)] for k, v in dict(scene="opsafter", **query).items()})

    def test_frames_and_register(self):
        x, z = make_ops_scene.PEOPLE[0]
        y = float(make_ops_scene.ground_y(np.array(x), np.array(z)))
        out = self.get("inspect/frames", x=x, y=y, z=z)
        self.assertGreaterEqual(out["seen_by"], 1)
        self.assertTrue(out["frames"][0]["url"].startswith("/runtime/work/opsafter/frames_full/"))
        self.assertIn("mgrs", out["where"])
        reg = self.post("inspect/annotation", item={"position": [x, y, z], "type": "spalling", "severity": 3})
        created = reg["created"]
        listing = self.get("inspect/annotations")
        item = next(a for a in listing["items"] if a["id"] == created)
        self.assertTrue(item["photo"]["url"].endswith(f"{created}.jpg"))
        self.assertTrue(api.file_allowed(f"work/opsafter/inspection/photos/{created}.jpg"))
        crack = self.get("inspect/crack", id=created)
        self.assertEqual(crack["candidates"], [])                      # a flat grey photo has no cracks
        pdf = self.get("inspect/report")
        self.assertTrue(base64.b64decode(pdf["content"]).startswith(b"%PDF"))
        with self.assertRaises(api.Error):
            self.post("inspect/annotation", item={"position": [0, 0], "type": "crack"})

    def test_section_through_a_building(self):
        cx, cz, lx, dz, h = make_ops_scene.BUILDINGS["B1"]
        out = self.post("inspect/section", a=[cx - 15, 0, cz], b=[cx + 15, 0, cz], half_width_m=0.4)
        rep = out["report"]
        self.assertAlmostEqual(rep["length_m"], 30.0, delta=0.01)
        base = float(make_ops_scene.ground_y(np.array(cx), np.array(cz)))
        self.assertAlmostEqual(max(rep["outline_m"]), base + h, delta=0.4)
        self.assertTrue(base64.b64decode(out["svg"]["content"]).startswith(b"<svg"))
        self.assertIn(b"OUTLINE", base64.b64decode(out["dxf"]["content"]))

    def test_terrain_layers(self):
        for layer in ("ortho", "hillshade_dtm", "lrm", "slope_deg"):
            out = self.get("inspect/terrain", layer=layer)
            self.assertTrue(base64.b64decode(out["png_base64"]).startswith(b"\x89PNG"))
        # The 2.5 m river-channel step stands out in the local relief model.
        self.assertTrue(self.get("inspect/terrain", layer="lrm")["overlay"])

    def test_m3c2_sees_the_stockpile(self):
        sx, sz, r, h = make_ops_scene.STOCKPILE
        ring = [[sx - 12, sz - 12], [sx + 12, sz - 12], [sx + 12, sz + 12], [sx - 12, sz + 12]]
        out = self.post("inspect/m3c2", before="opsbefore", region=ring, core_spacing_m=1.0)
        rep = out["report"]
        self.assertGreater(rep["significant"], 20)
        self.assertGreater(rep["moved_towards_viewer"], rep["moved_away"])     # the pile rose towards the cameras
        self.assertTrue(out["overlay"])

    def test_tilt_refuses_empty_ground(self):
        with self.assertRaises(api.Error):
            self.post("inspect/tilt", base=[-70.0, 0.0, 60.0], radius_m=0.3)

    def test_provenance_and_fixity(self):
        rec = self.get("inspect/provenance")
        self.assertEqual(rec["dublin_core"]["title"], "Ops site after the event (synthetic)")
        self.assertEqual(rec["spatial_reference"]["status"], "georeferenced")
        self.assertIn("UTM", rec["spatial_reference"]["crs"]["name"])
        files = {f["file"] for f in rec["fixity"]}
        self.assertIn("viewer_assets/scene.ply", files)
        self.assertEqual(self.post("inspect/provenance/verify", record=rec)["changed"], [])
        tampered = json.loads(json.dumps(rec))
        tampered["fixity"][0]["sha256"] = "0" * 64
        self.assertEqual(len(self.post("inspect/provenance/verify", record=tampered)["changed"]), 1)

    def test_hypothesis_scheme_exports_as_inferred(self):
        import workspace_plan_api
        import workspace_proposals as wp
        work = self.root / "work" / "opsafter"
        p = wp.create_proposal(work, "Restored tower")
        p = wp.upsert_feature(work, p["id"], {"type": "building", "name": "Tower", "params": {
            "footprint": [[-60, 60], [-50, 60], [-50, 70], [-60, 70]], "floors": 3}}, p["revision"])
        state = workspace_plan_api.post(api, self.root, "plan/proposals/inferred",
                                        {"scene": "opsafter", "id": p["id"], "inferred": True, "basis": "1910 photograph"}, Server())
        self.assertTrue(state["proposal"]["inferred"])
        self.assertEqual(state["proposal"]["inferred_basis"], "1910 photograph")
        ev, ground, registry, existing = workspace_plan_api._evaluate(work, state["proposal"])
        geo = wp.export_geojson(work, state["proposal"], ev, registry)
        self.assertEqual(geo["features"][0]["properties"]["status"], "inferred hypothesis, not measured")
        import plan_exports
        city = plan_exports.cityjson(state["proposal"], ev, registry, ground, existing)
        self.assertIn("inferred hypothesis", json.dumps(city))

    def test_crane_swing_finds_the_building(self):
        import workspace_plan_api
        import workspace_proposals as wp
        work = self.root / "work" / "opsafter"
        p = wp.create_proposal(work, "Site logistics")
        # A short crane (13.5 m mast, jib 12 m above its base) beside the 9 m building B1: 5 m clearance fails.
        p = wp.upsert_feature(work, p["id"], {"type": "object", "name": "TC1", "params": {
            "item": "tower_crane", "position": [-10.0, -35.0], "scale": 0.3, "jib_radius_m": 25.0, "clearance_m": 5.0}}, p["revision"])
        ev, *_ = workspace_plan_api._evaluate(work, p)
        crane = p["features"][0]["id"]
        m = ev["features"][crane]["metrics"]
        self.assertGreater(m["swing_conflict_m2"], 20)
        self.assertEqual(ev["violations"][-1]["rule"], "crane_swing_clearance")
        # The same crane at full height clears everything.
        p = wp.upsert_feature(work, p["id"], {**p["features"][0], "params": {**p["features"][0]["params"], "scale": 1.0}}, p["revision"])
        ev, *_ = workspace_plan_api._evaluate(work, p)
        self.assertEqual(ev["features"][crane]["metrics"]["swing_conflict_m2"], 0.0)
        self.assertFalse(any(v["rule"] == "crane_swing_clearance" for v in ev["violations"]))

    def _shapefile_zip(self, ring_utm, props):
        import struct
        pts = np.vstack([ring_utm, ring_utm[:1]])
        if wp_area(pts) > 0:                                   # shapefile outer rings are clockwise
            pts = pts[::-1]
        content = struct.pack("<i4d", 5, *pts.min(0), *pts.max(0)) + struct.pack("<ii", 1, len(pts)) + struct.pack("<i", 0) + pts.astype("<f8").tobytes()
        rec = struct.pack(">ii", 1, len(content) // 2) + content
        header = struct.pack(">i", 9994) + b"\0" * 20 + struct.pack(">i", (100 + len(rec)) // 2) + struct.pack("<ii", 1000, 5) + struct.pack("<4d", *pts.min(0), *pts.max(0)) + b"\0" * 32
        fields = [("NAME", "C", 20), ("FSI", "N", 6)]
        dbf_header = struct.pack("<BBBBIHH", 3, 126, 1, 1, 1, 32 + 32 * len(fields) + 1, 1 + sum(w for *_, w in fields)) + b"\0" * 20
        for name, typ, width in fields:
            dbf_header += name.encode().ljust(11, b"\0") + typ.encode() + b"\0" * 4 + bytes([width, 1 if typ == "N" else 0]) + b"\0" * 14
        dbf_header += b"\r"
        row = b" " + str(props["NAME"]).encode().ljust(20) + f"{props['FSI']:6.1f}".encode()
        import pyproj
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("plots.shp", header + rec)
            z.writestr("plots.dbf", dbf_header + row + b"\x1a")
            z.writestr("plots.prj", pyproj.CRS.from_epsg(32643).to_wkt(version="WKT1_ESRI"))
        return base64.b64encode(buf.getvalue()).decode()

    def test_shapefile_and_other_crs_parcels(self):
        import pyproj
        import workspace_plan_api
        import workspace_proposals as wp
        global wp_area
        wp_area = wp.polygon_area
        work = self.root / "work" / "opsafter"
        origin = make_ops_scene.ORIGIN
        to_utm = pyproj.Transformer.from_crs(4326, 32643, always_xy=True)
        e0, n0 = to_utm.transform(origin["longitude_deg"], origin["latitude_deg"])
        # A 20 x 20 m plot 30 m east and 20 m north of the origin (viewer x 30..50, z -20..-40).
        ring = np.array([[e0 + 30, n0 + 20], [e0 + 50, n0 + 20], [e0 + 50, n0 + 40], [e0 + 30, n0 + 40]])
        p = wp.create_proposal(work, "Cadastre")
        state = workspace_plan_api.post(api, self.root, "plan/import", {"scene": "opsafter", "id": p["id"], "revision": p["revision"],
                                        "filename": "plots.zip", "content": self._shapefile_zip(ring, {"NAME": "Khasra 12", "FSI": 1.5})}, Server())
        zone = state["proposal"]["features"][-1]
        # Expected: each UTM corner through WGS84 to ENU at the origin. Not the raw UTM
        # offsets: UTM grid north is ~0.9 deg off true north here (grid convergence).
        import survey_measure
        frame = survey_measure.frame_for_origin(origin["latitude_deg"], origin["longitude_deg"], origin["altitude_m"])
        lon, lat = pyproj.Transformer.from_crs(32643, 4326, always_xy=True).transform(ring[:, 0], ring[:, 1])
        enu = survey_measure.geodetic_to_enu(np.column_stack([lat, lon, np.full(len(lat), origin["altitude_m"])]), frame)
        got = np.array(zone["params"]["polygon"])
        expected = np.column_stack([enu[:, 0], -enu[:, 1]])
        self.assertLess(np.abs(np.sort(got, axis=0) - np.sort(expected, axis=0)).max(), 0.05)
        self.assertEqual(zone["name"], "Khasra 12")                           # from the .dbf
        self.assertEqual(zone["params"]["rules"].get("max_fsi"), 1.5)
        self.assertIn("UTM zone 43N", state["import"]["basis"])
        # GeoJSON in Web Mercator (EPSG:3857) used to be refused; it now goes through pyproj.
        to_merc = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
        lonlat = pyproj.Transformer.from_crs(32643, 4326, always_xy=True).transform(ring[:, 0], ring[:, 1])
        mx, my = to_merc.transform(*lonlat)
        doc = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
               "features": [{"type": "Feature", "properties": {"name": "Merc plot"},
                             "geometry": {"type": "Polygon", "coordinates": [[[x, y] for x, y in zip(mx, my)] + [[mx[0], my[0]]]]}}]}
        p = state["proposal"]
        state = workspace_plan_api.post(api, self.root, "plan/import", {"scene": "opsafter", "id": p["id"], "revision": p["revision"],
                                        "filename": "plots.geojson", "content": json.dumps(doc)}, Server())
        got = np.array(state["proposal"]["features"][-1]["params"]["polygon"])
        self.assertLess(np.abs(np.sort(got, axis=0) - np.sort(expected, axis=0)).max(), 0.05)
        self.assertIn("3857", state["import"]["basis"])

    def test_twin_inventory_attributes_epochs_package(self):
        inv = self.get("twin/inventory")
        self.assertGreaterEqual(inv["counts"]["building"], 2)
        asset = next(a for a in inv["assets"] if a["kind"] == "building")
        self.post("twin/attributes", id=asset["id"], values={"asset_tag": "BLD-001", "owner": "PWD", "bogus": "x"})
        inv = self.get("twin/inventory")
        again = next(a for a in inv["assets"] if a["id"] == asset["id"])
        self.assertEqual(again["attributes"]["asset_tag"], "BLD-001")
        self.assertNotIn("bogus", again["attributes"])
        ep = self.get("twin/epochs")
        self.assertEqual([e["scene"] for e in ep["epochs"]], ["opsbefore", "opsafter"])   # by capture date
        out = self.post("twin/package", splats=True)
        path = self.root / out["url"].split("?")[0].replace("/runtime/", "")
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            ts = json.loads(z.read("tiles/tileset.json"))
        self.assertTrue({"surface.glb", "surface.fbx", "tiles/tileset.json", "splats/index.json", "README.txt"} <= names)
        self.assertEqual(len(ts["root"]["children"]), 4)
        self.assertIn("transform", ts["root"])                               # georeferenced: ECEF placed
        import plan_glb
        doc, _ = plan_glb.read_glb(open(path, "rb").read() if False else zipfile.ZipFile(path).read("surface.glb"))
        self.assertIn("COLOR_0", doc["meshes"][0]["primitives"][0]["attributes"])


if __name__ == "__main__":
    unittest.main()
