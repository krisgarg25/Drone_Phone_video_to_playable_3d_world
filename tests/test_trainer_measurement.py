"""The trainer's measurement-scale contracts, on the CPU lane.

Three things this pins, because all three have already produced a wrong number:

1.  A reported metric or a shipped cloud may never come from an opacity-reset
    step. gsplat clamps EVERY opacity to `prune_opa * 2` on that interval, so a
    run that ends there measures the reset: work/rocks at --steps 6000 reported
    16.14 dB held-out PSNR where the same model one step earlier reports 34.16.
2.  The room-anchor seeds are placed by this scene's own measured geometry, so
    they scale when the scene scales. A bare "1.5 metres" does not: on
    work/rocks (scale_m_per_unit 4.734) the same three literals landed 26-36 m
    in front of the cloud the camera could actually see.
3.  A room prior is only applied to a scene the room detector measured as a
    room, and it says which way it decided.

Everything here is the pure half of scripts/train_splat.py: the same AST
extraction tests/test_camera_intrinsics.py uses, extended with the helpers this
file owns, so none of it needs torch and none of it needs a GPU.
"""
import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

# The CPU-extractable half of the trainer. Adding to this set is a statement that
# the function must stay torch-free; nn_spacing/DepthTerm/evaluate are not in it
# because they allocate CUDA buffers.
CPU_FUNCTIONS = {
    "load_colmap", "load_points3d", "prepare_dataset", "sparse_depth_targets",
    "reset_lands_on", "artefact_reset_state",
    "room_prior_decision", "anchor_seed_distances",
}
CPU_CONSTANTS = {
    "SH_DEG", "REFINE_START", "RESET_EVERY", "RESET_RECOVERY_TAG_STEPS",
    "POINT_VIEWS_HALF", "MIN_POINTS_PER_VIEW", "MIN_FRAME_COVERAGE",
    "NN_SPACING_FLOOR", "ALPHA_MIN_FOR_DEPTH", "ROOM_PRIOR_MIN_WALLS",
    "ANCHOR_DEPTH_QUANTILES", "ANCHOR_JITTER_FRACTION", "MIN_POINTS_PER_CAMERA",
}


def load_trainer_cpu_half():
    """Exec the torch-free functions and constants out of scripts/train_splat.py."""
    tree = ast.parse((SCRIPTS / "train_splat.py").read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in CPU_FUNCTIONS:
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & CPU_CONSTANTS:
                nodes.append(node)
        elif isinstance(node, ast.Import) and all(
                alias.name.split(".")[0] in {"cv2", "numpy", "hashlib", "json"}
                for alias in node.names):
            nodes.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module in {
                "pathlib", "PIL", "camera_intrinsics"}:
            nodes.append(node)
    parser = ast.parse((SCRIPTS / "parse_colmap.py").read_text(encoding="utf-8"))
    nodes.extend(n for n in parser.body
                 if isinstance(n, ast.FunctionDef) and n.name == "qvec2rot")
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 str(SCRIPTS / "train_splat.py"), "exec"), ns)
    return ns


NS = load_trainer_cpu_half()


class ResetStepScheduleTests(unittest.TestCase):
    """The 15 dB artefact: a final metric taken on an opacity-reset step."""

    def setUp(self):
        self.reset_lands_on = NS["reset_lands_on"]
        self.state = NS["artefact_reset_state"]
        self.every = NS["RESET_EVERY"]                 # steps between resets
        self.tag_window = NS["RESET_RECOVERY_TAG_STEPS"]   # steps, log annotation

    def test_it_mirrors_the_installed_gsplat_strategy(self):
        """Read the reset out of gsplat's own source, not out of a memory of it."""
        spec = importlib.util.find_spec("gsplat")
        if spec is None:      # the CPU lane has no gsplat: this half still runs
            self.skipTest("gsplat is only installed in the GPU lane (.venv310)")
        from gsplat import DefaultStrategy
        text = (Path(spec.submodule_search_locations[0]) / "strategy"
                / "default.py").read_text(encoding="utf-8")
        self.assertIn("if step % self.reset_every == 0 and step > 0:", text)
        self.assertIn("if step >= self.refine_stop_iter:", text)
        fields = {f.name: f.default for f in
                  __import__("dataclasses").fields(DefaultStrategy)}
        self.assertEqual(fields["reset_every"], self.every,
                         "gsplat's reset interval moved; the trainer's RESET_EVERY "
                         "has to move with it or the reporting is off by a reset")

    def test_a_reset_step_is_recognised_exactly_where_gsplat_fires(self):
        lands = self.reset_lands_on
        self.assertTrue(lands(3000, self.every, 9000))
        self.assertTrue(lands(6000, self.every, 9000))
        # default.py returns before the reset from refine_stop_iter onward, so a
        # step at or past the refine stop is a normal step and needs no settle.
        self.assertFalse(lands(9000, self.every, 9000))
        self.assertFalse(lands(12000, self.every, 9000))
        self.assertFalse(lands(0, self.every, 9000))
        self.assertFalse(lands(200, self.every, 9000))
        self.assertFalse(lands(3000, self.every, 3000))
        self.assertFalse(lands(3000, 0, 9000), "reset_every 0 means no resets")

    def test_a_reset_step_is_always_answered_by_one_of_three_branches(self):
        """Report the trained cloud, or refuse. Never report the demoted one."""
        state = self.state
        # A step that is not a reset step: nothing special, and nothing refused.
        self.assertEqual(state(5999, self.every, 9000), (False, False, False))
        # A reset step whose pre-reset opacities were captured: restore and report.
        self.assertEqual(state(6000, self.every, 9000, snapshot_ready=True),
                         (True, True, False))
        # A reset step where the capture did not happen: refuse. This is the branch
        # that keeps a gsplat which moved its internals from becoming a 16 dB
        # "model" - 0.04 is prune_opa * 2, what every opacity is clamped to.
        self.assertEqual(state(6000, self.every, 9000, snapshot_ready=False),
                         (True, False, True))
        for steps in range(1, 12001):
            for refine in (500, 3000, 9000, 18000):
                for snap in (True, False):
                    demoted, restore, refuse = state(steps, self.every, refine, snap)
                    self.assertEqual(demoted,
                                     self.reset_lands_on(steps, self.every, refine))
                    self.assertFalse(demoted and not (restore or refuse),
                                     f"--steps {steps} (refine {refine}, snap {snap}) "
                                     f"would report a demoted cloud as the model")

    def test_the_reset_dip_is_a_transient_so_the_log_tags_it_for_that_long(self):
        """The tag window is in steps and shorter than the interval that causes it."""
        self.assertGreater(self.tag_window, 0)
        self.assertLess(self.tag_window, self.every,
                        "annotating past the next reset would tag the wrong dip")



class RoomAnchorDistanceTests(unittest.TestCase):
    """Anchor distances come from this scene's cloud, not from a metre."""

    def setUp(self):
        self.dist = NS["anchor_seed_distances"]

    def _scene(self, n_cams=8, scale=1.0, wall_z=10.0, jitter=0.0):
        """Cameras at z=0.6*scale looking along +z at a plane of points at z=wall_z.

        The points span +/- 35% of the wall distance laterally, which is inside a
        400 px frame with a 400 px focal length (half-angle ~26.5 deg), so they are
        points the camera actually sees.
        """
        rng = np.random.default_rng(7)
        data, pts = [], []
        for i in range(n_cams):
            # viewmat is world->cam: identity rotation puts the camera at the
            # origin looking down +z, so t = -centre.
            vm = np.eye(4, dtype=np.float32)
            vm[:3, 3] = [0.0, 0.0, -0.6 * scale]
            data.append(dict(viewmat=vm, K=np.array([[400.0, 0, 200], [0, 400, 200],
                                                     [0, 0, 1]]), width=400,
                             height=400))
        u = rng.uniform(-0.35, 0.35, 400) * wall_z * scale
        v = rng.uniform(-0.35, 0.35, 400) * wall_z * scale
        pts = np.stack([u, v, np.full_like(u, wall_z * scale + jitter)], 1)
        return data, pts

    def test_distances_are_the_measured_cloud_depth_in_scene_units(self):
        data, pts = self._scene()
        got, ev = self.dist(data, pts)
        centres, fwds, D = got
        self.assertEqual(D.shape, (len(data), len(NS["ANCHOR_DEPTH_QUANTILES"])))
        # Every camera is 0.6 units behind the plane, so the depth of the points
        # it sees is 9.4 units: that is the number, in scene units, not a metre.
        np.testing.assert_allclose(D, np.full(D.shape, 9.4), rtol=0.02)
        self.assertEqual(ev["units"], "scene units (COLMAP), NOT metres")
        self.assertEqual(ev["source"], "cloud depth inside the frustum, per camera")
        self.assertEqual(ev["cameras_with_own_distribution"], len(data))

    def test_distances_scale_with_the_scene_a_bare_metre_never_would(self):
        """The whole point of the fix: double the scene, double the seeds."""
        small, s_pts = self._scene(scale=1.0)
        big, b_pts = self._scene(scale=10.0)
        Ds = self.dist(small, s_pts)[0][2]
        Db = self.dist(big, b_pts)[0][2]
        np.testing.assert_allclose(Db, Ds * 10.0, rtol=0.02)
        self.assertNotAlmostEqual(float(Ds.mean()), 1.5,
                                  msg="a hard-coded 1.5 has crept back in")

    def test_a_camera_that_sees_nothing_takes_the_pooled_measurement(self):
        data, pts = self._scene(n_cams=4)
        # Park one camera beyond the plane, still looking along +z, so the cloud is
        # behind it: that camera has no depth distribution of its own.
        vm = np.eye(4, dtype=np.float32)
        vm[:3, 3] = [0, 0, -500.0]
        data[2]["viewmat"] = vm
        got, ev = self.dist(data, pts)
        D = got[2]
        self.assertEqual(ev["cameras_with_own_distribution"], 3)
        self.assertIn("pooled quantile", ev["source"])
        np.testing.assert_allclose(D[2], D[0], rtol=1e-6)

    def test_no_usable_cloud_falls_back_to_the_camera_hull_not_to_zero(self):
        data, pts = self._scene(n_cams=6)
        for i, d in enumerate(data):
            # Park every camera BEYOND the point cloud, looking the other way, so
            # nothing is inside any frustum - but spread along x, so the cameras
            # themselves still measure a length.
            vm = np.eye(4, dtype=np.float32)
            vm[:3, :3] = np.diag([1.0, 1.0, -1.0])
            vm[:3, 3] = [-2.0 * i, 0.0, -50.0]
            d["viewmat"] = vm
        got, ev = self.dist(data, pts)
        self.assertIsNotNone(got)
        self.assertIn("camera hull", ev["source"])
        self.assertGreater(ev["camera_hull_diag_units"], 0.0)
        self.assertTrue((got[2] > 0).all(), "a fallback distance must never be 0")

    def test_a_scene_with_no_geometry_and_no_camera_spread_gets_no_distance(self):
        data, _ = self._scene(n_cams=3)
        for d in data:
            d["viewmat"] = np.eye(4, dtype=np.float32)   # all one point: no hull
        got, ev = self.dist(data, np.zeros((0, 3)))
        self.assertIsNone(got, "no cloud and no camera spread must not become a "
                               "distance of 0: it means no anchors at all")
        self.assertIn("no cloud in any frustum", ev["source"])
        self.assertEqual(ev["span_units"], 0.0, "the measured hull, printed as what "
                                                "it is: zero, i.e. nothing to work on")

    def test_the_jitter_is_a_fraction_of_a_measured_distance(self):
        data, pts = self._scene(scale=10.0)
        got, _ = self.dist(data, pts)
        # The constant that multiplies it must stay dimensionless, or the seeding
        # inherits a length again.
        self.assertLess(NS["ANCHOR_JITTER_FRACTION"], 1.0)
        self.assertGreater(NS["ANCHOR_JITTER_FRACTION"], 0.0)
        self.assertEqual(len(got[2][0]), len(NS["ANCHOR_DEPTH_QUANTILES"]))


class RoomPriorGateTests(unittest.TestCase):
    """detect_rooms.py decides whether a room prior applies, in its own words."""

    def setUp(self):
        self.gate = NS["room_prior_decision"]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def write(self, payload):
        out = self.work / "viewer_assets"
        out.mkdir(parents=True, exist_ok=True)
        (out / "rooms.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_no_rooms_json_is_an_absent_measurement_not_a_refusal(self):
        apply, why, ev = self.gate(self.work)
        self.assertFalse(apply)
        self.assertEqual(ev, {})
        self.assertIn("has not run", why)
        self.assertIn("not a room", why)
        self.assertNotIn("outdoor", why.lower(),
                         "an unmeasured scene must not be called outdoors")

    def test_a_drone_scene_declines_with_the_detectors_own_reason(self):
        self.write({"units": {"metric_confidence": "provisional"}, "rooms": [],
                    "refusal": {"rooms_found": 0, "walls_found": 0,
                                "floor": "floor is 416 disconnected patches and the "
                                         "largest holds 3% of the floor cells"}})
        apply, why, ev = self.gate(self.work)
        self.assertFalse(apply)
        self.assertIn("416 disconnected patches", why)
        self.assertEqual(ev["n_rooms"], 0)

    def test_one_wall_is_not_yet_a_room_prior(self):
        self.write({"units": {"metric_confidence": "anchored"}, "rooms": [
            {"floor": {"coherent_share_of_floor": 0.96}, "walls": [{"azimuth_deg": 74}],
             "enclosure": {"wall_count": 1, "note": "no opposing wall pair"}}]})
        apply, why, ev = self.gate(self.work)
        self.assertFalse(apply)
        self.assertIn("only 1 fitted wall plane", why)
        self.assertEqual(ev["walls"], 1)

    def test_a_measured_interior_applies_it_and_quotes_the_evidence(self):
        self.write({"units": {"metric_confidence": "anchored",
                              "scale_source": "AR pose-prior metric path"},
                    "rooms": [{"floor": {"coherent_share_of_floor": 0.786,
                                         "observed_area_m2": 5.79},
                               "walls": [{"azimuth_deg": 166.2, "length_m": 4.78},
                                         {"azimuth_deg": 74.0, "length_m": 5.4}],
                               "enclosure": {"wall_count": 2}}]})
        apply, why, ev = self.gate(self.work)
        self.assertTrue(apply)
        self.assertIn("2 fitted wall planes", why)
        self.assertIn("anchored", why)
        self.assertEqual(ev["walls"], 2)
        self.assertEqual(ev["floor_coherent_share"], 0.786)

    def test_the_real_scenes_decide_the_way_they_were_measured_to(self):
        """work/rocks is an outdoor orbit; work/room_w_jsonl is a walked room."""
        for scene, expect in (("rocks", False), ("room_w_jsonl", True),
                              ("roomscan", False)):
            path = ROOT / "work" / scene
            if not (path / "viewer_assets" / "rooms.json").is_file():
                continue          # another lane may not have exported assets yet
            apply, why, ev = self.gate(path)
            self.assertIs(apply, expect, f"{scene}: {why}")
            self.assertTrue(why, "a decision without a reason is not a gate")

    def test_the_gate_threshold_is_a_count_not_a_length(self):
        self.assertIsInstance(NS["ROOM_PRIOR_MIN_WALLS"], int)
        self.assertGreaterEqual(NS["ROOM_PRIOR_MIN_WALLS"], 2)


class ExtractionIntegrityTests(unittest.TestCase):
    """The geometry half stays importable without torch: that is how it is tested."""

    def test_the_pinned_functions_came_from_a_torch_free_execution(self):
        # NS was built by exec'ing the selected AST nodes with no torch in the
        # namespace, so if any of them reached for torch this would have failed at
        # import. What this asserts is that the set did not quietly shrink.
        for name in CPU_FUNCTIONS:
            self.assertTrue(callable(NS.get(name)), f"{name} is not extractable")
        self.assertNotIn("torch", NS)
        self.assertNotIn("gsplat", NS)

    def test_nothing_torch_dependent_was_added_to_the_extracted_set(self):
        """A function in CPU_FUNCTIONS may not name torch, gsplat or robust."""
        tree = ast.parse((SCRIPTS / "train_splat.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in CPU_FUNCTIONS:
                names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                mods = {n.value.id for n in ast.walk(node)
                        if isinstance(n, ast.Attribute)
                        and isinstance(n.value, ast.Name)}
                bad = (names | mods) & {"torch", "gsplat", "rb", "robust",
                                        "rendering", "DefaultStrategy"}
                self.assertFalse(bad, f"{node.name} uses {bad}: it would stop the "
                                      f"CPU lane from pinning the geometry")

    @unittest.skipUnless(importlib.util.find_spec("torch") is None,
                         "CPU lane only: this asserts torch stays absent, which the "
                         "GPU lane is allowed to fail and the .venv lane does not")
    def test_this_lane_really_has_no_torch(self):
        self.assertIsNone(importlib.util.find_spec("torch"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
