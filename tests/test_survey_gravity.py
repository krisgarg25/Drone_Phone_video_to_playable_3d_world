"""CPU-only tests for straight-track georeferencing (M11) on synthetic corridors.

Every scene is built from a known similarity, so each test asks whether that exact
transform comes back, not whether a number merely looks plausible.
"""
import copy
import importlib
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

FRAME = {"type": "ENU", "units": "m", "origin": {
    "latitude_deg": 28.6, "longitude_deg": 77.2, "altitude_m": 210.0},
    "geodetic_crs": "EPSG:4979", "altitude_datum": "ellipsoidal"}


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


LEVEL_NORTH = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # ENU -> OpenCV camera


def enu_to_camera(yaw, pitch_deg, roll_deg):
    return rz(-math.radians(roll_deg)) @ rx(-math.radians(pitch_deg)) @ LEVEL_NORTH @ rz(-yaw)


def corridor(n=40, length=1200.0, height=80.0, *, true_rotation, scale=4.0,
             translation=(500.0, -300.0, 12.0), pitch=-60.0, roll=0.0, gps_noise=0.0,
             lateral_noise=0.02, outliers=(), seed=1):
    """Cameras on a straight ENU line, their local-frame poses, GPS and attitude."""
    rng = np.random.default_rng(seed)
    translation = np.asarray(translation, float)
    heading = math.radians(35.0)
    direction = np.array([math.sin(heading), math.cos(heading), 0.0])
    along = np.linspace(0, length, n)
    enu = along[:, None] * direction + np.array([0, 0, height])
    enu += rng.normal(0, lateral_noise, enu.shape)
    local = ((enu - translation) / scale) @ true_rotation   # inverse similarity
    rows, samples, attitude = [], [], []
    for i, (c_local, c_enu) in enumerate(zip(local, enu)):
        r_enu_cam = enu_to_camera(heading, pitch, roll)
        r_local = r_enu_cam @ true_rotation
        rows.append(dict(file=f"f{i:04d}.jpg", t_sec=float(i),
                         camera={"R_rowmajor": r_local.tolist(), "t": (-r_local @ c_local).tolist()}))
        gps = c_enu + rng.normal(0, gps_noise, 3)
        if i in outliers:
            gps = gps + np.array([40.0, -25.0, 15.0])
        samples.append(dict(t_sec=float(i), position=gps.tolist(),
                            horizontal_std_m=1.0, vertical_std_m=2.0))
        attitude.append(dict(t_sec=float(i), pitch_deg=pitch, roll_deg=roll))
    telemetry = dict(schema_version=1, coordinate_frame=copy.deepcopy(FRAME), samples=samples)
    return rows, telemetry, attitude, local, enu


TRUE_R = rz(0.7) @ rx(0.4) @ ry(-0.9)


def angle_between_rotations(a, b):
    return math.degrees(math.acos(np.clip((np.trace(np.asarray(a).T @ np.asarray(b)) - 1) / 2, -1, 1)))


class GravityTests(unittest.TestCase):
    def setUp(self):
        self.g = importlib.import_module("survey_gravity")
        self.geo = importlib.import_module("survey_georef")

    def test_camera_convention_matches_helper(self):
        for pitch, roll in ((-90, 0), (-45, 3), (0, -5), (20, 10)):
            up = enu_to_camera(0.4, pitch, roll) @ np.array([0, 0, 1.0])
            np.testing.assert_allclose(up, self.g.up_in_camera(pitch, roll), atol=1e-12)

    def test_similarity_fit_refuses_the_corridor(self):
        rows, telemetry, *_ = corridor(true_rotation=TRUE_R)
        with self.assertRaises(ValueError) as ctx:
            self.geo.align_camera_trajectory(rows, telemetry)
        self.assertTrue(self.g.is_degenerate_track(ctx.exception))

    def test_attitude_gravity_recovers_known_transform(self):
        rows, telemetry, attitude, *_ = corridor(true_rotation=TRUE_R, roll=0.0)
        gravity = self.g.gravity_from_attitude(rows, attitude)
        self.assertLess(self.g._angle_deg(gravity["up_local"], TRUE_R.T @ [0, 0, 1]), 1e-6)
        result = self.g.align_line_with_gravity(rows, telemetry, gravity)
        self.assertLess(angle_between_rotations(result["rotation"], TRUE_R), 0.05)
        self.assertAlmostEqual(result["scale"], 4.0, delta=0.01)
        self.assertEqual(result["inlier_count"], 40)
        # A point 60 m off the track (where a wrong roll would show) lands where it should.
        probe_enu = np.array([[560.0, 100.0, 5.0], [300.0, 900.0, -3.0]])
        probe_local = ((probe_enu - [500.0, -300.0, 12.0]) / 4.0) @ TRUE_R
        mapped = self.geo.transform_points(probe_local, result)
        np.testing.assert_allclose(mapped, probe_enu, atol=0.2)

    def test_noisy_gps_and_outliers(self):
        rows, telemetry, attitude, *_ = corridor(true_rotation=TRUE_R, gps_noise=1.0,
                                                  outliers=(3, 17, 31))
        result = self.g.align_line_with_gravity(rows, telemetry,
                                                self.g.gravity_from_attitude(rows, attitude),
                                                inlier_threshold_m=5.0)
        self.assertEqual(result["inlier_count"], 37)
        self.assertNotIn("f0003.jpg", result["inlier_files"])
        self.assertLess(angle_between_rotations(result["rotation"], TRUE_R), 0.5)
        self.assertAlmostEqual(result["scale"], 4.0, delta=0.05)

    def test_rolled_gimbal_is_handled(self):
        rows, telemetry, attitude, *_ = corridor(true_rotation=TRUE_R, roll=7.0, pitch=-30.0)
        result = self.g.align_line_with_gravity(rows, telemetry,
                                                self.g.gravity_from_attitude(rows, attitude))
        self.assertLess(angle_between_rotations(result["rotation"], TRUE_R), 0.05)

    def test_ground_plane_gravity_on_level_terrain(self):
        rows, telemetry, _, local, enu = corridor(true_rotation=TRUE_R)
        rng = np.random.default_rng(3)
        ground_enu = np.column_stack([rng.uniform(-100, 1100, 4000), rng.uniform(-100, 1100, 4000),
                                      rng.normal(0, 0.05, 4000)])
        clutter = np.column_stack([rng.uniform(0, 1000, 400), rng.uniform(0, 1000, 400),
                                   rng.uniform(1, 25, 400)])
        points = ((np.vstack([ground_enu, clutter]) - [500.0, -300.0, 12.0]) / 4.0) @ TRUE_R
        gravity = self.g.gravity_from_ground(points, local)
        self.assertLess(self.g._angle_deg(gravity["up_local"], TRUE_R.T @ [0, 0, 1]), 0.2)
        result = self.g.align_line_with_gravity(rows, telemetry, gravity)
        self.assertLess(angle_between_rotations(result["rotation"], TRUE_R), 0.3)
        self.assertIn("level across the track", " ".join(result["warnings"]))

    def test_cross_slope_becomes_tilt_as_documented(self):
        rows, telemetry, _, local, _ = corridor(true_rotation=TRUE_R)
        rng = np.random.default_rng(4)
        xy = np.column_stack([rng.uniform(-100, 1100, 4000), rng.uniform(-100, 1100, 4000)])
        heading = math.radians(35.0)
        cross = xy @ np.array([math.cos(heading), -math.sin(heading)])  # across the track
        z = cross * math.tan(math.radians(3.0))
        points = ((np.column_stack([xy, z]) - [500.0, -300.0, 12.0]) / 4.0) @ TRUE_R
        result = self.g.align_line_with_gravity(rows, telemetry, self.g.gravity_from_ground(points, local))
        # The documented failure mode: the slope angle reappears as a rotation error.
        self.assertAlmostEqual(angle_between_rotations(result["rotation"], TRUE_R), 3.0, delta=0.2)

    def test_near_vertical_track_is_refused(self):
        n = 12
        enu = np.column_stack([np.zeros(n), np.zeros(n), np.linspace(10, 120, n)])
        local = (enu - [1.0, 2.0, 3.0]) @ TRUE_R
        rows = [dict(file=f"v{i}.jpg", t_sec=float(i),
                     camera={"R_rowmajor": TRUE_R.tolist(), "t": (-TRUE_R @ c).tolist()})
                for i, c in enumerate(local)]
        telemetry = dict(schema_version=1, coordinate_frame=copy.deepcopy(FRAME), samples=[
            dict(t_sec=float(i), position=p.tolist(), horizontal_std_m=1.0, vertical_std_m=2.0)
            for i, p in enumerate(enu)])
        gravity = dict(source="attitude", up_local=(TRUE_R.T @ [0, 0, 1]).tolist(), tilt_std_deg=0.1)
        with self.assertRaises(ValueError):
            self.g.align_line_with_gravity(rows, telemetry, gravity)

    def test_attitude_needs_time_overlap(self):
        rows, _, attitude, *_ = corridor(true_rotation=TRUE_R)
        shifted = [dict(a, t_sec=a["t_sec"] + 1000) for a in attitude]
        with self.assertRaises(ValueError):
            self.g.gravity_from_attitude(rows, shifted)

    def test_tilt_check_on_ordinary_fit(self):
        rows, telemetry, attitude, local, enu = corridor(true_rotation=TRUE_R)
        # Bend the track into an arc so the ordinary similarity fit is well posed.
        bend = np.linspace(0, 1, len(enu)) ** 2 * 150.0
        heading = math.radians(35.0)
        arc = enu + bend[:, None] * np.array([math.cos(heading), -math.sin(heading), 0.0])
        arc_local = ((arc - [500.0, -300.0, 12.0]) / 4.0) @ TRUE_R
        for row, c in zip(rows, arc_local):
            r = np.asarray(row["camera"]["R_rowmajor"])
            row["camera"]["t"] = (-r @ c).tolist()
        for sample, p in zip(telemetry["samples"], arc):
            sample["position"] = p.tolist()
        alignment = self.geo.align_camera_trajectory(rows, telemetry)
        good = self.g.gravity_from_attitude(rows, attitude)
        self.assertLess(self.g.tilt_check(alignment, good)["angle_deg"], 0.05)
        biased = dict(good, up_local=(TRUE_R.T @ rx(math.radians(5)) @ [0, 0, 1]).tolist())
        self.assertAlmostEqual(self.g.tilt_check(alignment, biased)["angle_deg"], 5.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()


class WorkflowIntegrationTests(unittest.TestCase):
    """save_inputs -> prepare -> _georeference with a DJI subtitle carrying side channels."""

    def setUp(self):
        import json as _json
        import tempfile
        self.json = _json
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "videos" / "corridor"
        self.source.mkdir(parents=True)
        (self.source / "corridor.mp4").write_bytes(b"not-decoded")
        self.wf = importlib.import_module("survey_workflow")
        self.g = importlib.import_module("survey_gravity")
        self.metadata = {"schema_version": 1, "time_reference": "video", "time_offset_s": 0,
                         "altitude_datum": "ellipsoidal", "position_reference": "camera_center",
                         "single_pass": True, "video_duration_s": 120,
                         "altitude_source": "absolute", "horizontal_std_m": 1.5,
                         "vertical_std_m": 3.0}

    def srt(self, n=100, *, pitch=-60.0, roll=0.0, seed=2):
        rng = np.random.default_rng(seed)
        blocks = []
        for i in range(n):
            lat = 28.6 + i * 1e-4
            rel = 60.0 + 3 * math.sin(i / 10)
            gnss = 212.0 + rel + rng.normal(0, 2.0)
            blocks.append(f"{i + 1}\n00:{i // 60:02d}:{i % 60:02d},000 --> "
                          f"00:{(i + 1) // 60:02d}:{(i + 1) % 60:02d},000\n"
                          f"Time:{i * 1000} Latitude:{lat:.7f} Longitude:77.2000000 "
                          f"AbsoluteAltitude:{gnss:.2f} RelativeAltitude:{rel:.2f} "
                          f"GimbalPitch:{pitch:.1f} GimbalRoll:{roll:.1f} GimbalYaw:0.0\n")
        return "\n".join(blocks)

    def test_srt_side_channels_reach_the_sidecar_and_fuse_heights(self):
        self.wf.save_inputs(self.root, "corridor", self.srt(), self.metadata)
        aux = self.json.loads((self.source / "telemetry_aux.json").read_text(encoding="utf-8"))
        self.assertEqual(len(aux["attitude"]), 100)
        self.assertEqual(len(aux["relative_altitude"]), 100)
        self.assertAlmostEqual(aux["attitude"][0]["pitch_deg"], -60.0)
        self.wf.prepare_scene(self.root, "corridor")
        prep = self.json.loads((self.root / "work/corridor/survey/preparation.json")
                               .read_text(encoding="utf-8"))
        self.assertEqual(prep["vertical_fusion"]["status"], "fused")
        self.assertEqual(prep["vertical_fusion"]["samples_fused"], 100)
        self.assertEqual(len(prep["auxiliary_inputs"]), 1)
        ups = np.array([s["position"][2] for s in prep["telemetry"]["samples"]])
        rel = np.array([r["relative_altitude_m"] for r in aux["relative_altitude"]])
        # Fused heights follow the barometric shape exactly (up to the constant level).
        self.assertLess(np.std((ups - ups[0]) - (rel - rel[0])), 0.05)
        # Editing the sidecar makes the preparation stale.
        aux["attitude"][0]["pitch_deg"] = -59.0
        (self.source / "telemetry_aux.json").write_text(self.json.dumps(aux), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.wf._prepared(self.root, "corridor")

    def test_georeference_falls_back_to_gravity_only_on_a_straight_track(self):
        rows, telemetry, attitude, *_ = corridor(true_rotation=TRUE_R)
        aux = {"attitude": attitude}
        line = self.wf._georeference(rows, telemetry, {"time_offset_s": 0}, aux, None)
        self.assertEqual(line["method"], "deterministic_ransac_weighted_line_similarity_gravity_roll")
        self.assertTrue(self.g.is_degenerate_track(line["fallback_from"]))
        self.assertLess(angle_between_rotations(line["rotation"], TRUE_R), 0.05)
        with self.assertRaises(ValueError) as ctx:
            self.wf._georeference(rows, telemetry, {"time_offset_s": 0}, None, None)
        self.assertIn("needs gravity", str(ctx.exception))

    def test_ordinary_fit_is_kept_and_gets_a_tilt_check(self):
        rows, telemetry, attitude, _, enu = corridor(true_rotation=TRUE_R)
        heading = math.radians(35.0)
        arc = enu + (np.linspace(0, 1, len(enu)) ** 2 * 150.0)[:, None] * np.array(
            [math.cos(heading), -math.sin(heading), 0.0])
        for row, sample, c in zip(rows, telemetry["samples"],
                                  ((arc - [500.0, -300.0, 12.0]) / 4.0) @ TRUE_R):
            row["camera"]["t"] = (-np.asarray(row["camera"]["R_rowmajor"]) @ c).tolist()
        for sample, p in zip(telemetry["samples"], arc):
            sample["position"] = p.tolist()
        result = self.wf._georeference(rows, telemetry, {"time_offset_s": 0},
                                       {"attitude": attitude}, None)
        self.assertEqual(result["method"], "deterministic_ransac_weighted_sim3")
        self.assertLess(result["tilt_check"]["angle_deg"], 0.05)
