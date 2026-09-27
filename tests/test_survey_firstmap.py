"""CPU-only tests for the per-window first-map preview (DIS-01)."""
import copy
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

FRAME = {"type": "ENU", "units": "m", "origin": {"latitude_deg": 30.7333, "longitude_deg": 76.7794,
                                                  "altitude_m": 300.0},
         "geodetic_crs": "EPSG:4979", "altitude_datum": "ellipsoidal"}
SCALE, YAW = 4.0, np.radians(30)
ROT = np.array([[np.cos(YAW), -np.sin(YAW), 0], [np.sin(YAW), np.cos(YAW), 0], [0, 0, 1]])
SHIFT = np.array([100.0, -50.0, 20.0])


def r_to_q(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)


def model(folder, n_cams=12):
    rng = np.random.default_rng(0)
    folder.mkdir(parents=True, exist_ok=True)
    centres = np.column_stack([np.linspace(0, 20, n_cams), 3 * np.sin(np.linspace(0, 3, n_cams)),
                               np.full(n_cams, 15.0) + rng.normal(0, 0.3, n_cams)])
    lines, times = [], {}
    R = np.diag([1.0, -1.0, -1.0])                     # looking straight down
    for i, c in enumerate(centres):
        t = -R @ c
        q = r_to_q(R)
        lines += [f"{i + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 f_{i:03d}.jpg", ""]
        times[f"f_{i:03d}.jpg"] = float(i)
    (folder / "images.txt").write_text("# header\n" + "\n".join(lines) + "\n")
    pts = np.column_stack([rng.uniform(-5, 25, 3000), rng.uniform(-8, 8, 3000), rng.normal(0, 0.05, 3000)])
    pts[(pts[:, 0] > 8) & (pts[:, 0] < 12) & (np.abs(pts[:, 1]) < 2), 2] += 2.0   # a 2-unit block
    rows = [f"{k + 1} {x} {y} {z} 120 {100 + k % 50} 90 0.5" for k, (x, y, z) in enumerate(pts)]
    (folder / "points3D.txt").write_text("\n".join(rows) + "\n")
    enu = (SCALE * (ROT @ centres.T)).T + SHIFT
    telemetry = dict(schema_version=1, coordinate_frame=copy.deepcopy(FRAME), samples=[
        dict(t_sec=float(i), position=p.tolist(), horizontal_std_m=0.3, vertical_std_m=0.8)
        for i, p in enumerate(enu)])
    return times, telemetry


class FirstMapTests(unittest.TestCase):
    def setUp(self):
        self.f = importlib.import_module("survey_firstmap")
        import survey_georef
        self.georef = survey_georef

    def test_georeferenced_preview_in_utm(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            times, telemetry = model(run / "txt")
            rec = self.f.publish(run, 0, run / "txt", times=times,
                                 georeference=lambda rows: self.georef.align_camera_trajectory(rows, telemetry))
            self.assertEqual(rec["status"], "done", rec)
            self.assertTrue(rec["georeferenced"], rec)
            self.assertIn("UTM zone 43N", rec["crs"])
            self.assertIn("dsm.tif", rec["files"])
            self.assertEqual(rec["registered_images"], 12)            # empty 2D-point lines kept
            import survey_formats
            tif = survey_formats.read_geotiff(run / rec["path"] / "dsm.tif")
            self.assertGreater(tif["transform"][0], 100000)          # a UTM easting
            # The block is 2 model units = 8 m after the GPS scale.
            self.assertAlmostEqual(rec["height_range"][1] - rec["height_range"][0], 8.0, delta=1.0)
            png = (run / rec["path"] / "preview.png").read_bytes()
            self.assertTrue(png.startswith(b"\x89PNG"))
            latest = json.loads((run / "progressive/preview/latest.json").read_text())
            self.assertEqual(latest["window"], 0)

    def test_local_preview_says_unscaled(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            model(run / "txt")
            rec = self.f.publish(run, 1, run / "txt")
            self.assertEqual(rec["status"], "done")
            self.assertFalse(rec["georeferenced"])
            self.assertEqual(rec["files"], ["preview.png"])
            self.assertIn("unscaled", rec["units"])

    def test_failure_is_recorded_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.f.publish(Path(tmp), 2, Path(tmp) / "missing")
            self.assertEqual(rec["status"], "failed")
            self.assertFalse((Path(tmp) / "progressive/preview/latest.json").exists())

    def test_latest_follows_the_scene_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            run = work / "survey" / "runs" / "R1"
            model(run / "txt")
            self.f.publish(run, 0, run / "txt")
            (work / "survey" / "latest_run.json").write_text(json.dumps({"id": "R1"}))
            self.assertEqual(self.f.latest(work)["run"], "R1")


if __name__ == "__main__":
    unittest.main()
