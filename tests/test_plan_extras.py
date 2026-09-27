"""CPU tests for the Phase 2 remainder: object models, shadow study, CityJSON / DXF /
3D Tiles export and cadastral import. Uses the same synthetic town as
test_workspace_proposals (gentle slope, a 2 m hill, one 9 m building, one tree).
"""
import importlib
import io
import json
import math
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_workspace_proposals import make_scene, square  # noqa: E402


class Fixture(unittest.TestCase):
    georeferenced = True

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = make_scene(self.tmp.name, georeferenced=self.georeferenced)
        self.wp = importlib.import_module("workspace_proposals")
        self.sf = importlib.import_module("scene_frames")
        self.ground = self.wp.Ground(self.work)
        self.existing = self.wp.existing_inventory(self.work, self.ground)
        self.registry = self.sf.load(self.work)
        self.p = self.wp.create_proposal(self.work, "Scheme A")

    def add(self, feature):
        self.p = self.wp.upsert_feature(self.work, self.p["id"], feature, self.p["revision"])
        return self.p["features"][-1]["id"]

    def evaluate(self):
        return self.wp.evaluate(self.p, self.ground, self.existing)


class ObjectModelTests(Fixture):
    def extent(self, item, yaw=0.0, scale=1.0):
        fid = self.add({"type": "object", "params": {"item": item, "position": [-30, -30], "yaw_deg": yaw, "scale": scale}})
        meshes = self.evaluate()["features"][fid]["meshes"]
        pts = np.vstack([np.asarray(m["positions"]).reshape(-1, 3) for m in meshes])
        return meshes, pts

    def test_detailed_models_have_parts_and_fit_their_catalogue_box(self):
        for item in ("tree_small", "tree_large", "street_light", "car"):
            meshes, pts = self.extent(item)
            size = self.wp.URBAN_OBJECTS[item][0]
            base = float(self.ground.sample(np.array([[-30.0, -30.0]]))[0][0])
            self.assertGreaterEqual(len(meshes), 2, item)
            self.assertAlmostEqual(pts[:, 1].max() - pts[:, 1].min(), size[1], delta=0.05 * size[1] + 0.05, msg=item)
            self.assertLessEqual(pts[:, 1].min(), base + 0.3, item)
            # Nothing pokes far outside the nominal footprint the metrics use.
            self.assertLessEqual(np.abs(pts[:, 0] + 30).max(), max(size[0], size[2]) / 2 + 1.7, item)

    def test_yaw_turns_the_model_and_other_items_stay_boxes(self):
        _, straight = self.extent("car")
        _, turned = self.extent("car", yaw=90)
        self.assertGreater(np.ptp(straight[:, 0]), np.ptp(straight[:, 2]))
        self.assertGreater(np.ptp(turned[:, 2]), np.ptp(turned[:, 0]))
        meshes, _ = self.extent("bench")
        self.assertEqual(len(meshes), 1)
        self.assertEqual(len(meshes[0]["indices"]) // 3, 12)


class SunTests(unittest.TestCase):
    def setUp(self):
        self.ps = importlib.import_module("plan_shadow")

    def noon(self, lat, lon, day):
        t0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return max(self.ps.sun_position(lat, lon, t0 + timedelta(minutes=m))[::-1] + (m,) for m in range(1440))

    def test_solar_noon_elevation_matches_declination(self):
        el, az, minute = self.noon(30.7333, 76.7794, "2026-12-21")
        self.assertAlmostEqual(el, 90 - 30.7333 - 23.44, delta=0.1)
        self.assertAlmostEqual(az, 180, delta=1.0)
        self.assertAlmostEqual(minute, 12 * 60 - 76.7794 * 4, delta=20)     # plus equation of time
        el, _, _ = self.noon(30.7333, 76.7794, "2026-06-21")
        self.assertAlmostEqual(el, 90 - 30.7333 + 23.44, delta=0.1)
        el, az, _ = self.noon(-33.87, 151.21, "2026-12-21")                     # Sydney: sun to the north
        self.assertAlmostEqual(el, 90 - 33.87 + 23.44, delta=0.1)
        self.assertLess(abs(az % 360), 1.5 if az < 180 else 361)

    def test_morning_sun_is_east_and_evening_west(self):
        base = datetime(2026, 3, 20, tzinfo=timezone(timedelta(hours=5.5)))
        az_am, el_am = self.ps.sun_position(28.6, 77.2, base.replace(hour=8))
        az_pm, el_pm = self.ps.sun_position(28.6, 77.2, base.replace(hour=16))
        self.assertTrue(60 < az_am < 130 and 230 < az_pm < 300, (az_am, az_pm))
        self.assertGreater(el_am, 0)
        self.assertGreater(el_pm, 0)


class ShadowTests(Fixture):
    def setUp(self):
        super().setUp()
        self.ps = importlib.import_module("plan_shadow")
        # The shared town only labels its existing building; give the surface model its
        # 9 m roof too, as a real DSM would have it, so demolishing it can free sunlight.
        va = self.work / "viewer_assets"
        top = np.fromfile(va / "heights.f32", dtype="<f4").reshape(200, 200)
        centres = -50.0 + (np.arange(200) + 0.5) * 0.5
        xx, zz = np.meshgrid(centres, centres)
        top[(xx > 10) & (xx < 20) & (np.abs(zz) < 5)] += 9.0
        top.tofile(va / "heights.f32")
        self.ground = self.wp.Ground(self.work)

    def test_box_shadow_has_the_right_length_and_direction(self):
        n = 120
        g = {"floor": np.zeros((n, n)), "top": np.zeros((n, n)), "supported": np.ones((n, n), bool),
             "cell": 0.5, "ox": -30.0, "oz": -30.0}
        work = self.ps._work_grid(g)
        s = work["top"].copy()
        xx, zz = self.ps._centres(work)
        s[(np.abs(xx) <= 5) & (np.abs(zz) <= 5)] = 12.0
        el = math.radians(30)
        mask = self.ps.shade(s, work, [0.0, math.sin(el), -math.cos(el)])     # sun to the north (-z)
        column = mask[:, n // 2] & ~(np.abs(zz[:, n // 2]) <= 5)
        zs = zz[:, n // 2][column]
        self.assertTrue(np.all(zs > 5))                                        # shadow falls south (+z)
        self.assertAlmostEqual(zs.max() - 5, 12 / math.tan(el), delta=1.0)

    def test_crowns_are_porous_buildings_are_not(self):
        n = 120
        g = {"floor": np.zeros((n, n)), "top": np.zeros((n, n)), "supported": np.ones((n, n), bool),
             "cell": 0.5, "ox": -30.0, "oz": -30.0}
        work = self.ps._work_grid(g)
        s = work["top"].copy()
        xx, zz = self.ps._centres(work)
        crown = (np.abs(xx) <= 5) & (np.abs(zz) <= 5)
        s[crown] = 12.0
        el = math.radians(60)
        sun = [0.0, math.sin(el), -math.cos(el)]
        porous = self.ps.porous_mask(work, [[[-5, -5], [5, -5], [5, 5], [-5, 5]]], s)
        lit_tree = self.ps.light(s, work, sun, porous)
        lit_block = self.ps.light(s, work, sun, np.zeros_like(porous))
        j = int(np.argmin(np.abs(zz[:, 0] - 6.0)))                           # just south of the crown
        i = n // 2
        self.assertEqual(lit_block[j, i], 0.0)
        # The ray climbs through ~10 m of crown horizontally at 60 deg: exp(-0.24 * ~11.5 m) ~ 0.06..0.1
        self.assertTrue(0.02 < lit_tree[j, i] < 0.35, lit_tree[j, i])
        self.assertEqual(lit_tree[0, 0], 1.0)                                 # open ground far away

    def test_instant_study_shades_new_ground_and_demolition_frees_it(self):
        self.add({"type": "building", "params": {"footprint": square(-20, -20, 6), "floors": 8}})
        self.add({"type": "clip", "params": {"polygon": square(15, 0, 6)}})
        out = self.ps.study(self.ground.grid, self.evaluate(), self.registry, date="2026-12-21",
                            time="10:00", utc_offset=5.5)
        rows = {r["metric"]: r for r in out["metrics"]}
        self.assertGreater(out["sun"]["elevation_deg"], 10)
        self.assertGreater(rows["Newly shaded by the scheme"]["proposal"], 100)
        self.assertGreater(rows["Sunlit again (demolitions)"]["proposal"], 10)
        parts = {m["part"] for m in out["overlay"]}
        self.assertTrue({"new_shadow", "newly_lit"} <= parts, parts)
        # Winter morning in the northern hemisphere: shadows fall to the north-west, i.e.
        # toward -z (north) and -x (west) from the building.
        new = next(m for m in out["overlay"] if m["part"] == "new_shadow")
        pts = np.asarray(new["positions"]).reshape(-1, 3)
        self.assertLess(np.median(pts[:, 2]), -20)
        self.assertLess(np.median(pts[:, 0]), -20)

    def test_day_study_counts_lost_hours(self):
        self.add({"type": "building", "params": {"footprint": square(-20, -20, 6), "floors": 10}})
        out = self.ps.study(self.ground.grid, self.evaluate(), self.registry, date="2026-03-20",
                            mode="day", start_h=9, end_h=15, step_min=30, utc_offset=5.5)
        rows = {r["metric"]: r for r in out["metrics"]}
        self.assertGreater(rows["Losing ≥ 1 h of sun"]["proposal"], 50)
        self.assertLessEqual(rows["Open ground with ≥ 2 h sun"]["proposal"], rows["Open ground with ≥ 2 h sun"]["existing"])
        self.assertEqual(len(out["sun"]["samples"]), 12)

    def test_night_and_local_scene_are_handled_honestly(self):
        out = self.ps.study(self.ground.grid, self.evaluate(), self.registry, date="2026-12-21",
                            time="23:00", utc_offset=5.5)
        self.assertEqual(out["overlay"], [])
        self.assertIn("below the horizon", out["notes"][0])
        with self.assertRaises(self.ps.ShadowError) as caught:
            self.ps.study(self.ground.grid, self.evaluate(), None, date="2026-12-21")
        self.assertEqual(caught.exception.status, 409)
        out = self.ps.study(self.ground.grid, self.evaluate(), None, date="2026-12-21", lat=28.6, lon=77.2,
                            utc_offset=5.5, north_deg=0)
        self.assertTrue(any("north is assumed" in n for n in out["notes"]))


class ExportTests(Fixture):
    def setUp(self):
        super().setUp()
        self.pe = importlib.import_module("plan_exports")
        self.b = self.add({"type": "building", "params": {"footprint": square(-20, -20, 5), "floors": 4, "roof": "gable"}})
        self.add({"type": "road", "params": {"centerline": [[-40, 30], [40, 30]], "width_m": 7}})
        self.add({"type": "zone", "params": {"polygon": square(-20, -20, 12), "rules": {"max_height_m": 30}}})
        self.add({"type": "object", "params": {"item": "tree_large", "position": [0, -35]}})
        self.add({"type": "clip", "params": {"polygon": square(15, 0, 6)}})

    def test_cityjson_building_is_a_closed_solid_in_utm(self):
        doc = self.pe.cityjson(self.p, self.evaluate(), self.registry, self.ground, self.existing)
        self.assertEqual((doc["type"], doc["version"]), ("CityJSON", "2.0"))
        self.assertTrue(doc["metadata"]["referenceSystem"].endswith("/32643"))
        kinds = sorted(o["type"] for o in doc["CityObjects"].values())
        self.assertEqual(kinds, ["Building", "GenericCityObject", "LandUse", "Road", "SolitaryVegetationObject"])
        n = len(doc["vertices"])
        building = doc["CityObjects"][self.b]
        shell = building["geometry"][0]["boundaries"][0]
        edges = {}
        for surface in shell:
            ring = surface[0]
            self.assertTrue(all(0 <= i < n for i in ring))
            for a, b in zip(ring, ring[1:] + ring[:1]):
                edges[(a, b)] = edges.get((a, b), 0) + 1
        # Closed and consistently oriented: every directed edge has exactly one reverse twin.
        self.assertTrue(all(edges.get((b, a)) == 1 and c == 1 for (a, b), c in edges.items()))
        self.assertEqual(set(building["geometry"][0]["semantics"]["values"][0]), {0, 1, 2})
        t = doc["transform"]
        east = np.asarray(doc["vertices"])[:, 0] * t["scale"][0] + t["translate"][0]
        self.assertTrue(np.all((east > 1e5) & (east < 9e5)))                  # UTM eastings
        self.assertEqual(doc["CityObjects"][self.p["features"][-1]["id"]]["attributes"]["demolishes_existing"],
                         self.evaluate()["demolished"])

    def test_dxf_round_trips_the_footprints(self):
        text = self.pe.dxf(self.p, self.evaluate(), self.registry, self.ground)
        self.assertTrue(text.rstrip().endswith("EOF"))
        self.assertIn("AC1009", text)
        polys = self.pe.read_dxf_polygons(text)
        layers = [layer for layer, _ in polys]
        self.assertIn("PLAN-BLDG", layers)
        self.assertIn("PLAN-PLOT", layers)
        footprint = next(p for layer, p in polys if layer == "PLAN-BLDG")
        back = self.pe.OutFrame(self.registry).from_utm(footprint)
        np.testing.assert_allclose(np.sort(back, axis=0), np.sort(np.asarray(square(-20, -20, 5)), axis=0), atol=0.01)

    def test_3dtiles_zip_places_the_scheme_on_the_globe(self):
        pg = importlib.import_module("plan_glb")
        crs = importlib.import_module("survey_crs")
        data, tileset = self.pe.tiles3d(self.p, self.evaluate(), self.registry)
        archive = zipfile.ZipFile(io.BytesIO(data))
        self.assertEqual(sorted(archive.namelist()), ["proposal.glb", "tileset.json"])
        doc, binary = pg.read_glb(archive.read("proposal.glb"))
        self.assertEqual(json.loads(archive.read("tileset.json"))["asset"]["version"], "1.1")
        m = np.asarray(tileset["root"]["transform"]).reshape(4, 4).T
        ecef0 = crs.ecef_from_geodetic([28.6], [77.2], [210.0])[0]
        np.testing.assert_allclose(m[:3, 3], ecef0, atol=1e-3)
        box = np.asarray(tileset["root"]["boundingVolume"]["box"])
        centre, half = box[:3], np.array([box[3], box[7], box[11]])
        for k, mesh in enumerate(doc["meshes"]):
            pos = pg.accessor_array(doc, binary, mesh["primitives"][0]["attributes"]["POSITION"])
            enu = np.column_stack([pos[:, 0], -pos[:, 2], pos[:, 1]])          # glTF Y-up -> ENU
            self.assertTrue(np.all(np.abs(enu - centre) <= half + 1e-3), k)
        # The building's ENU corner lands where scene_frames puts it on the globe.
        geo = self.sf.viewer_to_geodetic([[-25, 0, -25]], self.registry)[0]
        self.assertAlmostEqual(geo[0], 28.6 + 25 / 110_850, delta=2e-5)


class LocalExportTests(Fixture):
    georeferenced = False

    def test_local_exports_claim_no_crs(self):
        pe = importlib.import_module("plan_exports")
        self.add({"type": "building", "params": {"footprint": square(-20, -20, 5)}})
        doc = pe.cityjson(self.p, self.evaluate(), self.registry, self.ground)
        self.assertNotIn("referenceSystem", doc["metadata"])
        self.assertIn("LOCAL", doc["+proposal"]["note"])
        self.assertIn("LOCAL", pe.dxf(self.p, self.evaluate(), self.registry, self.ground).splitlines()[1])
        _, tileset = pe.tiles3d(self.p, self.evaluate(), self.registry)
        self.assertNotIn("transform", tileset["root"])


class CadastralTests(Fixture):
    bounds = ((-50.0, -50.0), (50.0, 50.0))

    def setUp(self):
        super().setUp()
        self.pe = importlib.import_module("plan_exports")

    def test_wgs84_geojson_round_trips_through_our_own_export(self):
        self.add({"type": "zone", "name": "Plot 7", "params": {"polygon": square(10, -30, 8)}})
        geo = self.wp.export_geojson(self.work, self.p, self.evaluate(), self.registry)
        geo["features"][0]["properties"].update(FSI="2.5", max_height="24")
        features, skipped, basis = self.pe.parse_parcels("parcels.geojson", json.dumps(geo), self.registry, self.bounds)
        self.assertEqual((len(features), skipped), (1, []))
        self.assertIn("WGS84", basis)
        np.testing.assert_allclose(features[0]["params"]["polygon"], square(10, -30, 8), atol=0.01)
        self.assertEqual(features[0]["params"]["rules"], {"max_height_m": 24.0, "max_fsi": 2.5})
        self.assertEqual(features[0]["name"], "Plot 7")

    def test_kml_dxf_and_utm_geojson_are_placed(self):
        frame = self.pe.OutFrame(self.registry)
        ring = np.asarray(square(0, 20, 5), dtype=float)
        geo = self.sf.viewer_to_geodetic(np.column_stack([ring[:, 0], np.zeros(4), ring[:, 1]]), self.registry)
        coords = " ".join(f"{lon},{lat},0" for lat, lon, _ in geo)
        kml = (f'<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><name>Khasra 12</name>'
               f'<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing>'
               f'</outerBoundaryIs></Polygon></Placemark></Document></kml>')
        features, _, _ = self.pe.parse_parcels("site.kml", kml, self.registry, self.bounds)
        np.testing.assert_allclose(features[0]["params"]["polygon"], ring, atol=0.01)
        self.assertEqual(features[0]["name"], "Khasra 12")
        utm = frame.out(np.column_stack([ring[:, 0], np.zeros(4), ring[:, 1]]))[:, :2]
        doc = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32643"}},
               "features": [{"type": "Feature", "properties": {"plot_no": "A-4"},
                             "geometry": {"type": "Polygon", "coordinates": [utm.tolist()]}}]}
        features, _, basis = self.pe.parse_parcels("utm.json", json.dumps(doc), self.registry, self.bounds)
        np.testing.assert_allclose(features[0]["params"]["polygon"], ring, atol=0.01)
        self.assertIn("32643", basis)
        body = self.pe._Dxf()
        body.g(0, "LWPOLYLINE"); body.g(8, "PARCELS"); body.g(90, 4); body.g(70, 1)
        for x, y in utm:
            body.g(10, float(x)); body.g(20, float(y))
        text = "\n".join(["0", "SECTION", "2", "ENTITIES"] + body.lines + ["0", "ENDSEC", "0", "EOF"])
        features, _, _ = self.pe.parse_parcels("cadastre.dxf", text, self.registry, self.bounds)
        np.testing.assert_allclose(features[0]["params"]["polygon"], ring, atol=0.01)

    def test_far_parcels_are_skipped_and_wgs84_needs_a_gps_fit(self):
        far = {"type": "Feature", "properties": {"name": "Far"},
               "geometry": {"type": "Polygon", "coordinates": [[[77.3, 28.7], [77.31, 28.7], [77.31, 28.71], [77.3, 28.71]]]}}
        with_near = {"type": "FeatureCollection", "features": [far]}
        features, skipped, _ = self.pe.parse_parcels("x.geojson", json.dumps(with_near), self.registry, self.bounds)
        self.assertEqual((features, skipped[0]["reason"]), ([], "outside the scanned area"))
        with self.assertRaises(self.pe.ExportError) as caught:
            self.pe.parse_parcels("x.geojson", json.dumps(with_near), None, self.bounds)
        self.assertEqual(caught.exception.status, 409)


class PlanExtrasHttpTests(unittest.TestCase):
    def setUp(self):
        import functools
        import threading
        from http.server import ThreadingHTTPServer
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import _serve
        from test_workspace_proposals import PlanHttpTests
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        make_scene(self.tmp.name, georeferenced=True)
        with _serve.process_lock:
            _serve.active_process = None
            _serve.active_job_info = {"status": "idle", "scene": "", "step": "", "logs": []}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_serve.H, directory=self.tmp.name))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.request = lambda *a, **k: PlanHttpTests.request(self, *a, **k)

    def test_shadow_export_formats_and_import(self):
        status, created = self.request("POST", "/api/workspace/plan/proposals", {"scene": "town", "name": "S"})
        pid = created["proposal"]["id"]
        status, state = self.request("POST", "/api/workspace/plan/feature", {
            "scene": "town", "id": pid, "revision": created["proposal"]["revision"],
            "feature": {"type": "building", "params": {"footprint": square(-20, -20, 5), "floors": 6}}})
        self.assertEqual(status, 200, state)
        status, shadow = self.request("GET", f"/api/workspace/plan/shadow?scene=town&id={pid}&date=2026-12-21&time=11:00&utc_offset=5.5")
        self.assertEqual(status, 200, shadow)
        self.assertTrue(shadow["overlay"])
        status, bad = self.request("GET", f"/api/workspace/plan/shadow?scene=town&id={pid}&date=21-12-2026")
        self.assertEqual(status, 400, bad)
        for fmt, name in (("cityjson", ".city.json"), ("dxf", ".dxf"), ("3dtiles", "-3dtiles.zip"), ("geojson", ".geojson")):
            status, out = self.request("GET", f"/api/workspace/plan/export?scene=town&id={pid}&format={fmt}")
            self.assertEqual(status, 200, out)
            self.assertTrue(out["filename"].endswith(name), out["filename"])
        geo = {"type": "FeatureCollection", "crs_note": "LOCAL test", "features": [
            {"type": "Feature", "properties": {"name": "P1", "setback": 3},
             "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}}]}
        status, imported = self.request("POST", "/api/workspace/plan/import", {
            "scene": "town", "id": pid, "revision": state["proposal"]["revision"],
            "filename": "p.geojson", "content": json.dumps(geo)})
        self.assertEqual(status, 200, imported)
        self.assertEqual(imported["import"]["imported"], 1)
        zone = imported["proposal"]["features"][-1]
        self.assertEqual((zone["type"], zone["name"], zone["params"]["rules"]), ("zone", "P1", {"setback_m": 3.0}))
        np.testing.assert_allclose(zone["params"]["polygon"], [[0, 0], [10, 0], [10, -10], [0, -10]])


if __name__ == "__main__":
    unittest.main()
