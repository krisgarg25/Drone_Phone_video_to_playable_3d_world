"""E: the scenario engine. Every test here pins a claim that is otherwise only visible
in a console line or a file nobody opens.

Run:  .venv/Scripts/python.exe tests/test_scenario.py
"""
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pipeline  # noqa: E402
import scenario_audit  # noqa: E402


def args(cmd="run", name="scene", preset="room", quality="high", **over):
    base = dict(cmd=cmd, name=name, preset=preset, quality=quality, variant="default",
                photometric="auto", cull="auto", dynamics="auto", mapper=None,
                target=None, width=None, steps=None, cap=None, voxel=None,
                grow_grad=None, init_min_tri_angle=None, overlap=None, prior_std=None,
                cross_clip=None, vocab_tree=None, speed_anchor=None, height_anchor=None,
                timeout_scale=1.0)
    base.update(over)
    return types.SimpleNamespace(**base)


def sources(videos=(), poses=()):
    return {"videos": [Path(v) for v in videos],
            "poses": {c: Path(p) for c, p in poses}, "frames_dirs": {}}


def cfg(tmp, preset="room", **over):
    """A resolved config with its work dir pointed at a temp folder, never a real scene."""
    c = pipeline.build_config(args(preset=preset, **over), sources(),
                              allow_auto_diag=False)
    c["work"] = Path(tmp)
    return c


class PresetReachesTheMapper(unittest.TestCase):
    """E2a: two preset keys were read by COLMAP's extractor and never reached the plan,
    so every capture style ran the room detector. That is the whole bug class."""

    def test_sift_thresholds_land_in_the_plan_colmap_reads(self):
        with tempfile.TemporaryDirectory() as td:
            c = cfg(Path(td), preset="drone")
            steps = {s["name"]: s for s in pipeline.build_steps(c)}
            steps["colmap"]["pre"]()                       # write_plan
            plan = json.loads((Path(td) / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["sift_peak_threshold"],
                             pipeline.PRESETS["drone"]["sift_peak_threshold"])
            self.assertEqual(plan["sift_edge_threshold"],
                             pipeline.PRESETS["drone"]["sift_edge_threshold"])

    def test_the_room_detector_is_no_longer_everywhere(self):
        with tempfile.TemporaryDirectory() as td:
            plans = {}
            for preset in ("room", "drone", "object"):
                c = cfg(Path(td) / preset, preset=preset)
                steps = {s["name"]: s for s in pipeline.build_steps(c)}
                steps["colmap"]["pre"]()
                plans[preset] = json.loads((Path(td) / preset / "plan.json")
                                           .read_text(encoding="utf-8"))
            self.assertNotEqual(plans["room"]["sift_peak_threshold"],
                                plans["drone"]["sift_peak_threshold"],
                                "distinct presets must produce distinct detector settings")
            # and the plan hash moves with them, or a re-run would look up to date
            digests = {p: json.dumps(plans[p], sort_keys=True) for p in plans}
            self.assertEqual(len(set(digests.values())), 3)

    def test_default_plan_carries_the_same_keys_so_the_hash_is_honest(self):
        from run_colmap import DEFAULT_PLAN
        self.assertIn("sift_peak_threshold", DEFAULT_PLAN)
        self.assertIn("sift_edge_threshold", DEFAULT_PLAN)


class SceneScaleBeatsQualityTier(unittest.TestCase):
    """E3: `high` quality set a 0.25 m collider voxel over a turntable preset's own
    value, so no close-up scene could ever ask for a finer one."""

    def test_object_keeps_its_collider_resolution_at_high_quality(self):
        with tempfile.TemporaryDirectory() as td:
            c = cfg(Path(td), preset="object", quality="high")
            self.assertEqual(c["voxel"], pipeline.PRESETS["object"]["voxel"])
            self.assertEqual(c["_scenario"]["origin"]["voxel"], "preset:object")
            self.assertTrue(any(e.get("param") == "voxel"
                                and "quality tier" in e.get("because", "")
                                for e in c["_scenario"]["evidence"]),
                            "the deferred override has to be recorded, not silent")

    def test_a_command_line_voxel_still_beats_both_layers(self):
        with tempfile.TemporaryDirectory() as td:
            c = cfg(Path(td), preset="object", quality="high", voxel="0.05")
            self.assertEqual(c["voxel"], "0.05")
            self.assertEqual(c["_scenario"]["origin"]["voxel"], "cli")


class HumanBodyNotDiorama(unittest.TestCase):
    """E1: the room preset shipped a 0.15 m character because the router's lateral
    clearance was 2.65 capsule radii, which no scanned room survives."""

    def test_every_walkable_preset_ships_a_person(self):
        for preset, body in pipeline.PRESETS.items():
            if preset == "auto" or preset == "object":
                continue
            self.assertEqual(body.get("character_height"), 1.75,
                             f"{preset} must declare the body it walks with")

    def test_clearance_is_a_multiple_of_the_body_not_an_absolute_corridor(self):
        import walk_path_from_glb as wp
        self.assertAlmostEqual(wp.CLEARANCE_M, wp.CLEARANCE_BODY_MULT * wp.CAPSULE_R)
        self.assertLess(wp.CLEARANCE_BODY_MULT, 1.6,
                        "a clearance demand above ~1.6 radii is what made a real flat "
                        "unwalkable at human scale")
        self.assertEqual(wp.CLEARANCE_M * wp.char_scale({"character_height": 1.75}),
                         wp.CLEARANCE_M)

    def test_the_bake_agent_is_the_preset_body_not_a_constant(self):
        with tempfile.TemporaryDirectory() as td:
            c = cfg(Path(td), preset="room")
            argv = pipeline.nav_params(c)
            self.assertIn("1.75", argv)                       # --height
            self.assertIn("0.340", argv)                      # --radius, the capsule
            self.assertIn("0.3", argv)                        # --climb, the max step
            drone = cfg(Path(td) / "d", preset="drone")
            self.assertIn("--cell", pipeline.nav_params(drone),
                          "aerial keeps its coarser grid, and only its grid")


class ScenarioRecord(unittest.TestCase):
    """E0: the resolved preset used to exist only in a console line."""

    def test_the_record_names_who_set_every_parameter(self):
        with tempfile.TemporaryDirectory() as td:
            c = cfg(Path(td), preset="room", quality="high", overlap=27)
            origin = c["_scenario"]["origin"]
            self.assertEqual(origin["overlap"], "cli")
            self.assertEqual(origin["prior_std"], "preset:room")
            self.assertEqual(origin["steps"], f"quality:high")
            self.assertTrue(any(e["param"] == "overlap" and e["set_by"] == "operator"
                                and e["from"] == 20 for e in c["_scenario"]["evidence"]))

    def test_the_record_is_written_and_holds_no_machine_path(self):
        with tempfile.TemporaryDirectory() as td:
            c = cfg(Path(td), preset="room")
            c["_scenario"]["applied"]["vocab_tree"] = str(ROOT / "tools" / "vocab_tree.bin")
            out = pipeline.write_scenario(c, steps=[{"name": "gate"}])
            text = out.read_text(encoding="utf-8")
            self.assertNotIn(str(ROOT), text)
            self.assertIn("tools", text)                      # kept, but relative
            self.assertIn("gate", json.loads(text)["steps"])

    def test_no_preset_is_left_undeclaring_a_knob_the_gate_judges_on(self):
        # `room` alone used to declare the collider, gate and body knobs; every other
        # preset inherited a script default nobody chose.
        shared = set(pipeline.PRESETS["room"]) & {
            "character_height", "max_step", "cull", "overlap", "prior_std",
            "sift_peak_threshold", "sift_edge_threshold", "cross_clip", "loop_detection",
            "target"}
        for preset, body in pipeline.PRESETS.items():
            if preset == "auto":
                continue
            self.assertFalse(shared - set(body),
                             f"{preset} does not declare {shared - set(body)}")


class GateMetricsReachTheApi(unittest.TestCase):
    """E-UI: the gate printed numbers to a console and wrote prose nobody opened."""

    GATE = {"status": "warnings",
            "checks": [
                {"name": "heightfield coverage", "status": "pass", "severity": "soft",
                 "detail": "19% measured", "metric": {"value": 19.0, "unit": "% of grid cells",
                                                      "threshold": ">= 5.0",
                                                      "better": "higher", "basis": "coverage.u8 == 1"}},
                {"name": "headroom above the walk surface", "status": "na",
                 "severity": "soft", "detail": "no ceiling observed"}],
            "hard_failures": [], "warnings": ["heightfield coverage"],
            "thresholds": {"character_height_m": 1.75, "cell_m": 0.035}}

    def test_rows_keep_their_numbers_and_the_na_state(self):
        import workspace_api as W
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "work" / "s" / "viewer_assets").mkdir(parents=True)
            (root / "work" / "s" / "viewer_assets" / "world_check.json").write_text(
                json.dumps(self.GATE), encoding="utf-8")
            q = W._quality_check(root, root / "work" / "s")
            by = {c["name"]: c for c in q["checks"]}
            self.assertEqual(by["heightfield coverage"]["value"], 19.0)
            self.assertEqual(by["heightfield coverage"]["threshold"], ">= 5.0")
            self.assertIsNone(by["headroom above the walk surface"]["value"],
                              "an unmeasurable check must carry null, never a 0.0 that "
                              "reads as a measurement of nothing")
            self.assertEqual(by["headroom above the walk surface"]["status"], "na")
            self.assertEqual(q["status"], "warnings")

    def test_no_verdict_is_reported_as_absent_not_as_passed(self):
        import workspace_api as W
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(W._quality_check(Path(td), Path(td) / "work" / "s"))


class HeadroomIsThreeState(unittest.TestCase):
    """A room scan that never got the roof in frame has no headroom measurement, and
    'unknown' has to be a different answer from 'passes'."""

    def _evidence(self, rooms):
        import check_world as cw
        with tempfile.TemporaryDirectory() as td:
            asset = Path(td)
            (asset / "rooms.json").write_text(json.dumps(rooms), encoding="utf-8")
            return cw.rooms_evidence(None, asset)

    def test_measured_ceiling_reports_the_lowest_one(self):
        e = self._evidence({"rooms": [
            {"ceiling": {"clear_height_above_floor_m": 2.71}},
            {"ceiling": {"clear_height_above_floor_m": 2.15}}]})
        self.assertEqual(e["status"], "measured")
        self.assertEqual(e["value"], 2.15, "the LOWEST ceiling governs whether you fit")

    def test_walls_only_bound_and_say_so(self):
        e = self._evidence({"rooms": [{"ceiling": {"clear_height_above_floor_m": None},
                                       "walls": [{"height_above_floor_m": 2.69}]}]})
        self.assertEqual(e["status"], "bounded")
        self.assertIsNone(e["value"])
        self.assertIn("not a headroom measurement", e["detail"])

    def test_nothing_overhead_is_not_captured(self):
        self.assertEqual(self._evidence({"rooms": []})["status"], "not_captured")


class AuditComparesPromiseWithWorld(unittest.TestCase):
    """E2b: nothing used to compare a preset's claim with what the scene became."""

    def test_a_room_that_measured_like_an_airfield_is_flagged(self):
        facts = {"has_assets": True, "recorded": True, "declared_preset": "room",
                 "camera_agl_m": 21.5, "footprint_m": 111.2, "cell_m": 3.49,
                 "character_height_m": 1.75, "nav_cell_m": 0.25,
                 "gate_status": "warnings", "gate_hard": [], "gate_warnings": []}
        kinds = {f["kind"] for f in scenario_audit.findings(facts)}
        self.assertIn("regime", kinds)
        self.assertIn("resolution", kinds)
        self.assertIn("inconsistent-resolution", kinds)
        fix = next(f for f in scenario_audit.findings(facts) if f["kind"] == "regime")["fix"]
        self.assertIn("`drone`", fix, "the audit must name the preset that fits")

    def test_an_unverified_speed_anchor_is_a_finding_not_a_footnote(self):
        facts = {"has_assets": True, "recorded": True, "declared_preset": "drone",
                 "camera_agl_m": 12.0, "footprint_m": 72.0, "cell_m": 2.18,
                 "character_height_m": 1.75, "scale_source": "flight speed x clip duration",
                 "gate_status": "pass", "gate_hard": [], "gate_warnings": []}
        kinds = {f["kind"] for f in scenario_audit.findings(facts)}
        self.assertIn("scale", kinds)

    def test_a_disagreement_between_the_two_path_planners_is_caught(self):
        facts = {"has_assets": True, "recorded": True, "declared_preset": "room",
                 "camera_agl_m": 1.0, "footprint_m": 6.8, "cell_m": 0.035,
                 "character_height_m": 0.15, "nav_agent_height_m": 1.7,
                 "gate_status": "pass", "gate_hard": [], "gate_warnings": []}
        msgs = [f["message"] for f in scenario_audit.findings(facts)]
        self.assertTrue(any("navigation mesh was baked for a 1.7 m agent" in m for m in msgs),
                        "physics and nav walking different bodies is a defect, not a detail")

    def test_a_hard_gate_failure_is_reported_as_a_failure(self):
        facts = {"has_assets": True, "recorded": True, "declared_preset": "drone",
                 "camera_agl_m": 12.0, "footprint_m": 72.0, "cell_m": 1.0,
                 "character_height_m": 1.75, "gate_status": "failed",
                 "gate_hard": ["spawn on supported ground"], "gate_warnings": []}
        self.assertIn("fail", [f["severity"] for f in scenario_audit.findings(facts)])

    def test_registration_is_counted_from_the_manifests_not_a_space_split(self):
        # COLMAP's images.txt puts the filename in field 10 and the point3D id LAST, so a
        # whitespace split reading parts[-1] parses the tail of a name. The manifests are
        # the repo's own definition of "registered", and what the project API reports.
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            (work / "keyframes.jsonl").write_text(
                "".join(json.dumps({"clip": "a", "file": f"f{i}.jpg"}) + "\n"
                        for i in range(10)), encoding="utf-8")
            (work / "keyframes_poses.jsonl").write_text(
                "".join(json.dumps({"file": f"f{i}.jpg"}) + "\n" for i in range(7)),
                encoding="utf-8")
            self.assertEqual(scenario_audit.registration(work), (7, 10))

class GpsRulerMeasuresScale(unittest.TestCase):
    """E: the drone scale finding. With the brief's mandatory GPS log beside the video,
    scale is read off a camera-to-GPS fit instead of an assumed flight speed."""

    HEADER = "t_sec,latitude_deg,longitude_deg,altitude_m,horizontal_std_m,vertical_std_m\n"
    # A climbing arc: curved, so rotation and scale are both observable.
    TELEMETRY = HEADER + "".join(
        f"{t},{28 + 0.0002 * math.sin(t / 3)},{77 + 0.0002 * math.cos(t / 3)},"
        f"{120 + 0.4 * t},0.5,1\n" for t in range(20))
    META = {"schema_version": 1, "time_reference": "video", "time_offset_s": 0,
            "altitude_datum": "ellipsoidal", "position_reference": "camera_center",
            "single_pass": True, "video_duration_s": 19}

    def flight(self, tmp, telemetry, scale=4.0):
        import numpy as np
        import survey_georef
        work = Path(tmp)
        (work / "telemetry.csv").write_text(telemetry)
        (work / "meta.json").write_text(json.dumps(self.META))
        track = survey_georef.normalize_telemetry(work / "telemetry.csv", self.META)
        rows = []
        for sample in track["samples"]:
            centre = np.asarray(sample["position"]) / scale    # COLMAP units
            rows.append({"file": f"{sample['t_sec']}.jpg", "t_sec": sample["t_sec"],
                         "camera": {"R_rowmajor": np.eye(3).tolist(), "t": (-centre).tolist()}})
        (work / "keyframes_poses.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        return work

    def test_the_fit_recovers_the_scale_the_track_was_built_at(self):
        import solve_frame
        with tempfile.TemporaryDirectory() as tmp:
            work = self.flight(tmp, self.TELEMETRY, scale=4.0)
            result = solve_frame.gps_scale(work, work / "telemetry.csv", work / "meta.json")
        self.assertNotIn("refused", result)
        self.assertAlmostEqual(result["scale"], 4.0, places=3)
        self.assertEqual(result["inliers"], 20)

    def test_a_straight_line_is_refused_and_falls_back(self):
        import solve_frame
        straight = self.HEADER + "".join(f"{t},{28 + 0.0001 * t},77,120,0.5,1\n"
                                         for t in range(20))
        with tempfile.TemporaryDirectory() as tmp:
            work = self.flight(tmp, straight)
            result = solve_frame.gps_scale(work, work / "telemetry.csv", work / "meta.json")
        self.assertIn("refused", result)

    def test_a_gps_scale_is_labelled_metric_and_clears_the_audit_finding(self):
        import workspace_api
        source = "GPS telemetry similarity fit (RMSE 0.42 m, 20 cameras)"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "work/flight").mkdir(parents=True)
            (root / "work/flight/frame.json").write_text(json.dumps(
                {"scale_m_per_unit": 4.0, "scale_source": source}))
            server = types.SimpleNamespace(_log_files=lambda _d: [], job_snapshot=None)
            original = workspace_api.job_snapshot
            workspace_api.job_snapshot = lambda _s, _r: {}
            try:
                project = workspace_api.summary(root, "flight", server)
            finally:
                workspace_api.job_snapshot = original
        self.assertEqual(project["scale"]["status"], "metric")

    def test_the_flight_log_names_the_pattern_the_frames_cannot(self):
        import numpy as np
        rng = np.random.default_rng(0)
        t = np.linspace(0, 2 * np.pi * 0.95, 120)
        orbit = np.c_[40 * np.cos(t) + rng.normal(0, .5, 120), 40 * np.sin(t), np.zeros(120)]
        legs = [np.c_[np.linspace(0, 100, 30)[::(1 if i % 2 == 0 else -1)],
                      np.full(30, i * 20.0), np.zeros(30)] for i in range(5)]
        line = np.c_[np.linspace(0, 300, 80), rng.normal(0, 2, 80), np.zeros(80)]
        self.assertEqual(pipeline.flight_pattern(orbit)["pattern"], "orbit")
        self.assertEqual(pipeline.flight_pattern(np.vstack(legs))["pattern"], "grid")
        self.assertEqual(pipeline.flight_pattern(line)["pattern"], "corridor")
        # Wandering is not a pattern, however many times it turns around.
        for seed in range(2, 8):
            wander = np.random.default_rng(seed).normal(0, 30, (60, 3))
            self.assertIsNone(pipeline.flight_pattern(wander)["pattern"], seed)

    def test_a_steady_mount_with_a_grid_log_picks_mapping_and_says_why(self):
        steady = {"mount_residual_px": 0.009}
        grid = {"pattern": "grid", "because": "4 near-reversals"}
        preset, evidence = pipeline.pick_preset({**steady, "gps_pattern": grid}, False)
        self.assertEqual(preset, "drone_mapping")
        self.assertIn("gps_flight_pattern", [e["signal"] for e in evidence])
        self.assertEqual(pipeline.pick_preset(steady, False)[0], "drone")
        undecided = {"pattern": None, "reason": "neither an orbit, a grid nor a line"}
        preset, evidence = pipeline.pick_preset({**steady, "gps_pattern": undecided}, False)
        self.assertEqual(preset, "drone")
        self.assertIn("neither", " ".join(e["because"] for e in evidence))
        # A shaky hand is still a hand, whatever a GPS log claims.
        self.assertEqual(pipeline.pick_preset({"mount_residual_px": 5.0, "gps_pattern": grid},
                                              False)[0], "room")

    def test_the_frame_step_takes_the_gps_log_when_the_scene_has_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = cfg(tmp, preset="drone")
            videos = pipeline.ROOT / "videos" / config["name"]
            steps = {s["name"]: s for s in pipeline.build_steps(config)}
            argv = [str(a) for a in steps["frame"]["argv"]]
            self.assertEqual("--telemetry" in argv, (videos / "telemetry.csv").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
