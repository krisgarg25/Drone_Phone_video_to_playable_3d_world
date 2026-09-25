"""CPU tests for the layered occlusion/hole accounting in survey_occlusion.

Synthetic single-pass scene: a drone flies one straight line on the x < 0 side of
a 3 m wall and looks toward +x. The wall shadows the floor behind it, and the
facade near the start of the pass is in no frame at all.

    floor  z = 0   x in [-10, 10]  y in [0, 6]
    wall   x = 5   z in [0, 3]     y in [0, 6]   <- the occluder
    roof   z = 4   x in [5, 10]    y in [0, 6]   <- planar object, partly observed
    back   x = -8  z in [0, 4]     y in [0, 6]   <- never inside any frame

Camera convention is the one survey_visibility uses: x_cam = R @ x_world + t, the
camera looks along +Z, and a depth map stores distance along the *unit ray*.
"""
import json
import unittest

import numpy as np

from scripts import survey_occlusion as occ
from scripts import survey_visibility as visibility

FX = 150.0
ROWS, COLS = 192, 256
CX, CY = (COLS - 1) / 2.0, (ROWS - 1) / 2.0
CAMERA_Y = np.linspace(0.6, 5.4, 5)
CELL = 1.0

# (name, origin, in-plane axis u, in-plane axis v, u range, v range), all axis aligned.
RECTS = [
    ("floor", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
     (-10.0, 10.0), (0.0, 6.0)),
    ("wall", np.array([5.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
     (0.0, 6.0), (0.0, 3.0)),
    ("roof", np.array([5.0, 0.0, 4.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]),
     (0.0, 6.0), (0.0, 5.0)),
    ("back", np.array([-8.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
     (0.0, 6.0), (0.0, 4.0)),
]


def viewmat(centre, target):
    """World->camera matrix looking from centre at target with +Z up (det = +1)."""
    centre = np.asarray(centre, float)
    forward = np.asarray(target, float) - centre
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-9:  # looking straight up or straight down
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    rotation = np.stack([right, np.cross(forward, right), forward])
    assert abs(np.linalg.det(rotation) - 1.0) < 1e-9, "camera basis must be a rotation"
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = -rotation @ centre
    return matrix


def render_depth(matrix):
    """Ray-trace the scene rectangles into a distance-along-the-unit-ray map."""
    rotation = matrix[:3, :3]
    centre = -matrix[:3, :3].T @ matrix[:3, 3]
    pixel_u = ((np.arange(COLS) - CX) / FX)[None, :].repeat(ROWS, 0)
    pixel_v = ((np.arange(ROWS) - CY) / FX)[:, None].repeat(COLS, 1)
    directions = np.stack([pixel_u, pixel_v, np.ones_like(pixel_u)])
    directions = directions / np.linalg.norm(directions, axis=0, keepdims=True)
    rays = (rotation.T @ directions.reshape(3, -1)).T
    nearest = np.full(ROWS * COLS, np.inf)
    for _, origin, axis_u, axis_v, range_u, range_v in RECTS:
        normal = np.cross(axis_u, axis_v)
        normal = normal / np.linalg.norm(normal)
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = ((origin - centre) @ normal) / (rays @ normal)
        hit = centre[None, :] + distance[:, None] * rays
        along_u = (hit - origin) @ axis_u
        along_v = (hit - origin) @ axis_v
        inside = ((distance > 1e-9) & np.isfinite(distance)
                  & (along_u >= range_u[0]) & (along_u <= range_u[1])
                  & (along_v >= range_v[0]) & (along_v <= range_v[1]))
        nearest = np.minimum(nearest, np.where(inside, distance, np.inf))
    return np.where(np.isfinite(nearest), nearest, 0.0).reshape(ROWS, COLS).astype(np.float32)


def lattice(*coordinate_lists):
    """Every combination of the given coordinate lists, as Nx3 world points."""
    arrays = [np.asarray(values, float) for values in coordinate_lists]
    mesh = np.stack(np.meshgrid(*arrays, indexing="ij"), axis=-1).reshape(-1, len(arrays))
    assert mesh.shape[1] == 3
    return mesh


def scene_cloud():
    """Candidate surface samples: every rectangle of the scene, sampled at 0.5 m."""
    floor = lattice(np.arange(-9.75, 10.0, 0.5), np.arange(0.25, 6.0, 0.5), [0.0])
    wall = lattice([5.0], np.arange(0.25, 6.0, 0.5), np.arange(0.25, 3.0, 0.5))
    roof = lattice(np.arange(5.25, 10.0, 0.5), np.arange(0.25, 6.0, 0.5), [4.0])
    back = lattice([-8.0], np.arange(0.25, 6.0, 0.5), np.arange(0.25, 4.0, 0.5))
    return np.vstack([floor, wall, roof, back])


def scene_views(count=5):
    views = []
    for index, y in enumerate(CAMERA_Y[:count]):
        matrix = viewmat([-3.0, y, 7.0], [4.0, y, 1.2])
        views.append({"K": np.array([[FX, 0.0, CX], [0.0, FX, CY], [0.0, 0.0, 1.0]]),
                      "viewmat": matrix,
                      "depth": render_depth(matrix),
                      "camera_center": np.array([-3.0, y, 7.0]),
                      "name": f"frame_{index:02d}"})
    return views


def camera_centres(count=5):
    return np.array([[-3.0, y, 7.0] for y in CAMERA_Y[:count]])


def framing_support(points, views):
    """Claimed support: how many views frame a point, with occlusion disabled.

    A depth map that records surface at 10 km makes every framed point agree with the
    recorded surface, so the count is the frustum claim COLMAP would report before any
    occlusion check - which is exactly the number visible_support contradicts.
    """
    open_views = [{"K": view["K"], "viewmat": view["viewmat"], "name": view["name"],
                   "depth": np.full(view["depth"].shape, 1e4, np.float32)} for view in views]
    return visibility.visible_support(points, open_views)


def region_mask(points, name):
    """Points lying on one rectangle of the scene."""
    for rect_name, origin, axis_u, axis_v, range_u, range_v in RECTS:
        if rect_name != name:
            continue
        normal = np.cross(axis_u, axis_v)
        normal = normal / np.linalg.norm(normal)
        along_u = (points - origin) @ axis_u
        along_v = (points - origin) @ axis_v
        return ((np.abs((points - origin) @ normal) < 1e-9)
                & (along_u >= range_u[0] - 1e-9) & (along_u <= range_u[1] + 1e-9)
                & (along_v >= range_v[0] - 1e-9) & (along_v <= range_v[1] + 1e-9))
    raise KeyError(name)


def cell_index(point, cell_size=CELL):
    return tuple(int(value) for value in np.floor(np.asarray(point, float) / cell_size))


def entry_for(result, point, cell_size=CELL):
    wanted = cell_index(point, cell_size)
    for entry in result["cells"]:
        if tuple(entry["cell"]) == wanted:
            return entry
    raise AssertionError(f"cell {wanted} for {point} missing from {[e['cell'] for e in result['cells']]}")


class ClassifyTests(unittest.TestCase):
    def setUp(self):
        self.points = scene_cloud()
        self.views = scene_views()
        self.claimed = framing_support(self.points, self.views)
        self.visible = visibility.visible_support(self.points, self.views)

    def test_labels_counts_and_fractions_are_consistent(self):
        result = occ.classify(self.points, support=self.claimed, visible_support=self.visible)
        self.assertEqual(len(result["labels"]), len(self.points))
        self.assertEqual(set(np.unique(result["labels"])) - set(occ.LAYERS), set())
        self.assertEqual(sum(result["counts"].values()), len(self.points))
        self.assertEqual(set(result["counts"]), {"measured", "weak", "unobserved"})
        self.assertAlmostEqual(sum(result["fractions"].values()), 1.0, places=5)
        self.assertEqual(result["point_count"], len(self.points))
        self.assertEqual(result["min_views"], 3)
        self.assertEqual(result["min_visible"], 2)
        self.assertEqual(result["counts"]["measured"], int((result["labels"] == "measured").sum()))
        self.assertGreater(result["counts"]["measured"], 0)
        self.assertGreater(result["counts"]["weak"], 0)
        self.assertGreater(result["counts"]["unobserved"], 0)
        self.assertLessEqual(result["visible_views"], result["claimed_views"])

    def test_wall_shadowed_floor_is_never_labelled_measured(self):
        result = occ.classify(self.points, support=self.claimed, visible_support=self.visible)
        floor = region_mask(self.points, "floor")
        shadow = floor & (self.points[:, 0] > 5.0) & (self.points[:, 0] < 10.0)
        lit = floor & (self.points[:, 0] > 0.0) & (self.points[:, 0] < 4.5)
        far = region_mask(self.points, "back")
        self.assertGreater(int(shadow.sum()), 50)
        self.assertGreater(int(lit.sum()), 50)
        self.assertTrue(np.all(self.claimed[shadow] >= 3), "the reconstruction claims these")
        self.assertTrue(np.all(self.visible[shadow] == 0), "no claimed view survives the depth check")
        self.assertTrue(np.all(result["labels"][shadow] != "measured"))
        self.assertTrue(np.all(result["labels"][shadow] == "weak"))
        self.assertTrue(np.all(self.claimed[lit] >= 3) and np.all(self.visible[lit] >= 2))
        self.assertTrue(np.all(result["labels"][lit] == "measured"))
        # The facade the pass flew away from was claimed by nobody and seen by nobody.
        self.assertTrue(np.all(self.claimed[far] == 0))
        self.assertTrue(np.all(result["labels"][far] == "unobserved"))

    def test_the_three_layers_separate_support_from_observation(self):
        support = np.array([5, 5, 0, 1, 3, 0])
        visible = np.array([5, 1, 0, 1, 0, 0])
        result = occ.classify(np.zeros((6, 3)), support=support, visible_support=visible)
        self.assertEqual(result["labels"].tolist(),
                         ["measured", "weak", "unobserved", "weak", "weak", "unobserved"])
        self.assertEqual(result["counts"], {"measured": 1, "weak": 3, "unobserved": 2})
        self.assertAlmostEqual(result["fractions"]["weak"], 0.5, places=6)

    def test_thresholds_are_respected(self):
        support = np.array([4, 4, 4])
        visible = np.array([3, 2, 1])
        points = np.zeros((3, 3))
        self.assertEqual(occ.classify(points, support=support, visible_support=visible,
                                      min_views=3, min_visible=3)["labels"].tolist(),
                         ["measured", "weak", "weak"])
        self.assertEqual(occ.classify(points, support=support, visible_support=visible,
                                      min_views=4, min_visible=1)["labels"].tolist(),
                         ["measured", "measured", "measured"])
        # Claimed support below min_views is not measured even when it is visible.
        self.assertEqual(occ.classify(np.zeros((1, 3)), support=np.array([2]), visible_support=np.array([2]),
                                      min_views=3, min_visible=2)["labels"].tolist(), ["weak"])

    def test_classify_output_can_feed_the_layer_plumbing(self):
        result = occ.classify(self.points, support=self.claimed, visible_support=self.visible)
        codes = np.asarray(result["layer"])
        self.assertEqual(codes.dtype.kind, "i")
        self.assertTrue(np.array_equal(codes == occ.LAYER_CODE["measured"], result["labels"] == "measured"))
        self.assertTrue(np.array_equal(codes == occ.LAYER_CODE["weak"], result["labels"] == "weak"))
        blended = occ.blend([{"points": self.points, "labels": result["labels"]}])
        self.assertEqual(blended["counts"], result["counts"])

    def test_invalid_classification_inputs_raise(self):
        points = np.zeros((3, 3))
        cases = [
            dict(support=np.array([1, 2]), visible_support=np.ones(3)),          # length mismatch
            dict(support=np.array([1, 2, -1]), visible_support=np.ones(3)),      # negative count
            dict(support=np.array([1, 2, np.nan]), visible_support=np.ones(3)),  # not finite
            dict(support=np.array([1, 2, 1.5]), visible_support=np.ones(3)),     # not integral
            dict(support=np.array([1, 1, 1]), visible_support=np.array([2, 1, 1])),  # > support
            dict(support=np.array([1e30, 1, 1]), visible_support=np.ones(3)),        # unrepresentable
            dict(support=np.ones(3), visible_support=np.ones(3), min_views=0),
            dict(support=np.ones(3), visible_support=np.ones(3), min_visible=0),
            dict(support=np.ones(3), visible_support=np.ones(3), min_visible=2.5),
            dict(support=np.ones(3), visible_support=np.ones(3), min_visible=4),  # above min_views
            dict(support=np.ones(3), visible_support=np.ones(3), min_views=1e30),
            dict(support="yes", visible_support=np.ones(3)),
        ]
        for kwargs in cases:
            with self.subTest(case=str(kwargs)):
                with self.assertRaises(ValueError):
                    occ.classify(points, **kwargs)
        with self.assertRaises(ValueError):
            occ.classify(np.zeros((3, 2)), support=np.ones(3, int), visible_support=np.ones(3, int))
        with self.assertRaises(ValueError):
            occ.classify(np.full((3, 3), np.nan), support=np.ones(3, int), visible_support=np.ones(3, int))
        with self.assertRaises(ValueError):
            occ.classify(np.zeros((0, 3)), support=np.zeros(0, int), visible_support=np.zeros(0, int))


class HiddenRegionTests(unittest.TestCase):
    def setUp(self):
        self.points = scene_cloud()
        self.views = scene_views()
        self.centres = camera_centres()

    def regions(self, **kwargs):
        options = dict(cell_size_m=CELL, angular_threshold_deg=5.0)
        options.update(kwargs)
        return occ.hidden_regions(self.centres, self.points, self.views, **options)

    def test_wall_shadow_is_reported_as_occluded(self):
        result = self.regions()
        for x in (5.5, 6.5, 7.5, 8.5, 9.5):
            entry = entry_for(result, (x, 2.25, 0.0))
            self.assertEqual(entry["reason"], "occluded", msg=f"floor cell x={x}")
            self.assertGreater(entry["views_framing"], 0, "the cell is inside a frame")
            self.assertEqual(entry["views_observed"], 0, "nothing there escapes the occluder")
        self.assertGreater(result["counts"]["occluded"], 20)
        self.assertEqual(sum(result["counts"].values()), result["total_cells"])
        self.assertEqual(result["hidden_cells"], sum(result["counts"][r] for r in occ.HIDDEN_REASONS))
        self.assertLess(result["hidden_fraction"], 1.0)
        self.assertGreater(result["hidden_fraction"], 0.0)
        self.assertEqual(result["total_cells"], len(result["cells"]))
        self.assertGreater(result["view_count"], 1)

    def test_region_in_no_frame_is_reported_as_never_observed(self):
        result = self.regions()
        for z in (0.5, 1.5, 2.5, 3.5):
            entry = entry_for(result, (-8.0, 2.25, z))
            self.assertEqual(entry["reason"], "never_observed", msg=f"facade cell z={z}")
            self.assertEqual(entry["views_framing"], 0)
        self.assertGreater(result["counts"]["never_observed"], 10)

    def test_observed_geometry_is_not_claimed_hidden(self):
        result = self.regions()
        for x in (-0.5, 1.5, 3.5, 4.5):
            self.assertEqual(entry_for(result, (x, 2.25, 0.0))["reason"], "observed", msg=f"floor x={x}")
        wall = entry_for(result, (5.0, 2.25, 1.25))
        self.assertEqual(wall["reason"], "observed")
        self.assertGreater(wall["views_observed"], 2)
        self.assertGreater(wall["parallax_deg"], 5.0)
        self.assertGreater(result["counts"]["observed"], 10)
        self.assertEqual(len(result["hidden"]), result["hidden_cells"])
        self.assertTrue(all(entry["reason"] in occ.HIDDEN_REASONS for entry in result["hidden"]))
        self.assertEqual(result["cell_size_m"], CELL)
        self.assertEqual(result["angular_threshold_deg"], 5.0)
        self.assertAlmostEqual(result["excluded_area_m2"],
                               result["hidden_cells"] * CELL ** 2, places=6)

    def test_looser_baseline_threshold_turns_observed_cells_into_hidden_ones(self):
        strict = self.regions(angular_threshold_deg=5.0)
        relaxed = self.regions(angular_threshold_deg=45.0)
        self.assertGreater(relaxed["counts"]["unusable_baseline"], strict["counts"]["unusable_baseline"])
        self.assertEqual(entry_for(relaxed, (5.0, 2.25, 1.25))["reason"], "unusable_baseline")
        self.assertEqual(entry_for(strict, (5.0, 2.25, 1.25))["reason"], "observed")

    def test_a_single_view_observes_surfaces_with_no_baseline_at_all(self):
        views, centres = scene_views(count=1), camera_centres(count=1)
        result = occ.hidden_regions(centres, self.points, views, cell_size_m=CELL,
                                    angular_threshold_deg=5.0)
        self.assertEqual(result["counts"]["observed"], 0)
        self.assertGreater(result["counts"]["unusable_baseline"], 5)
        self.assertEqual(result["view_count"], 1)

    def test_many_views_are_bounded_rather_than_compared_pairwise(self):
        """Above the exact-pairwise limit the spread is a stated lower bound."""
        count = occ.EXACT_BASELINE_VIEW_LIMIT + 7  # odd, so no pair is exactly opposite
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        centres = np.column_stack([5.0 * np.cos(angles), 5.0 * np.sin(angles), np.zeros(count)])
        views = []
        for centre in centres:
            matrix = viewmat(centre, [0.0, 0.0, 0.0])
            views.append({"K": np.array([[100.0, 0.0, 3.5], [0.0, 100.0, 3.5], [0.0, 0.0, 1.0]]),
                          "viewmat": matrix, "depth": np.full((8, 8), 5.0, np.float32)})
        result = occ.hidden_regions(centres, np.array([[0.0, 0.0, 0.0]]), views,
                                    cell_size_m=CELL, angular_threshold_deg=5.0)
        self.assertIn("two-sweep", result["parallax_basis"])
        self.assertEqual(result["counts"]["observed"], 1)
        self.assertGreater(result["cells"][0]["parallax_deg"], 170.0)
        tight = occ.hidden_regions(centres, np.array([[0.0, 0.0, 0.0]]), views,
                                   cell_size_m=CELL, angular_threshold_deg=178.0)
        self.assertEqual(tight["counts"]["unusable_baseline"], 1)
        few = occ.hidden_regions(centres[:4], np.array([[0.0, 0.0, 0.0]]), views[:4],
                                 cell_size_m=CELL, angular_threshold_deg=5.0)
        self.assertIn("exact maximum pairwise angle", few["parallax_basis"])

    def test_hole_report_is_json_serialisable_and_explains_itself(self):
        result = self.regions()
        text = json.dumps(result)
        self.assertIn("reason", text)
        self.assertTrue({entry["reason"] for entry in result["hidden"]} <= set(occ.HIDDEN_REASONS))
        for entry in result["hidden"]:
            self.assertEqual(len(entry["centre"]), 3)
            self.assertGreaterEqual(entry["points_in_cell"], 1)
            self.assertIn(entry["reason"], occ.HIDDEN_REASONS)
        self.assertIn("cannot see", result["note"].lower())
        self.assertIn("sampled", result["limitation"].lower())

    def test_invalid_hidden_region_inputs_raise(self):
        cases = [dict(cell_size_m=0.0, angular_threshold_deg=5.0),
                 dict(cell_size_m=-1.0, angular_threshold_deg=5.0),
                 dict(cell_size_m=np.nan, angular_threshold_deg=5.0),
                 dict(cell_size_m=1.0, angular_threshold_deg=float("nan")),
                 dict(cell_size_m=1.0, angular_threshold_deg=-5.0),
                 dict(cell_size_m=1e-300, angular_threshold_deg=5.0),
                 dict(cell_size_m=1.0, angular_threshold_deg=5.0, relative_tolerance=0.0)]
        for options in cases:
            with self.subTest(case=str(options)):
                with self.assertRaises(ValueError):
                    self.regions(**options)
        with self.assertRaises(ValueError):  # centres that are not the views' own
            occ.hidden_regions(np.zeros((5, 3)), self.points, self.views,
                               cell_size_m=CELL, angular_threshold_deg=5.0)
        with self.assertRaises(ValueError):  # one centre per depth view
            occ.hidden_regions(camera_centres(count=4), self.points, self.views,
                               cell_size_m=CELL, angular_threshold_deg=5.0)
        with self.assertRaises(ValueError):  # no cameras at all
            occ.hidden_regions(np.zeros((0, 3)), self.points, self.views,
                               cell_size_m=CELL, angular_threshold_deg=5.0)
        with self.assertRaises(ValueError):  # no depth maps
            occ.hidden_regions(self.centres, self.points, [],
                               cell_size_m=CELL, angular_threshold_deg=5.0)
        with self.assertRaises(ValueError):  # a depth view with a mirrored basis
            broken = dict(self.views[0])
            broken["viewmat"] = np.diag([1.0, -1.0, 1.0, 1.0])
            occ.hidden_regions(camera_centres(count=1), self.points, [broken],
                               cell_size_m=CELL, angular_threshold_deg=5.0)
        for points in (np.zeros((4, 2)), np.full((4, 3), np.inf), np.zeros((0, 3))):
            with self.subTest(shape=points.shape):
                with self.assertRaises(ValueError):
                    occ.hidden_regions(self.centres, points, self.views,
                                       cell_size_m=CELL, angular_threshold_deg=5.0)


class PlaneCompletionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        ground = lattice(np.arange(0.0, 4.01, 0.05), np.arange(0.0, 4.01, 0.05), [0.0])
        roof = lattice(np.arange(5.0, 7.01, 0.1), np.arange(0.0, 6.01, 0.1), [4.0])
        ceiling = lattice(np.arange(0.0, 6.01, 0.1), np.arange(0.0, 6.01, 0.1), [8.0])
        self.ground = ground + rng.normal(0.0, 0.004, ground.shape)
        self.roof = roof + rng.normal(0.0, 0.004, roof.shape)
        self.ceiling = ceiling + rng.normal(0.0, 0.004, ceiling.shape)
        self.points = np.vstack([self.ground, self.roof, self.ceiling])
        self.labels = np.array(["measured"] * (len(self.ground) + len(self.roof))
                               + ["weak"] * len(self.ceiling))

    def planes(self, **kwargs):
        options = dict(ransac_distance=0.05, min_inliers=200, max_planes=4)
        options.update(kwargs)
        return occ.plane_complete(self.points, self.labels, **options)

    def roof_plane(self):
        return [plane for plane in self.planes(max_planes=2)
                if abs(plane["normal"][2]) > 0.9 and abs(-plane["offset"] / plane["normal"][2] - 4.0) < 0.05][0]

    def test_planes_are_fitted_to_measured_points_only(self):
        planes = self.planes(max_planes=4)
        self.assertEqual(len(planes), 2, "only the two measured planes may be fitted")
        self.assertEqual([plane["inlier_count"] for plane in planes],
                         sorted([plane["inlier_count"] for plane in planes], reverse=True))
        used = np.concatenate([plane["inlier_indices"] for plane in planes])
        self.assertTrue(np.all(self.labels[used] == "measured"))
        for plane in planes:
            self.assertAlmostEqual(float(np.linalg.norm(plane["normal"])), 1.0, places=9)
            self.assertGreaterEqual(plane["inlier_count"], 200)
            self.assertLess(plane["residual_m"], 0.05)
            self.assertLess(plane["rms_residual_m"], 0.05)
            self.assertEqual(plane["layer"], "measured")
            self.assertEqual(plane["inlier_indices"].dtype, np.int64)
            residual = np.abs(self.points[plane["inlier_indices"]] @ plane["normal"] + plane["offset"])
            self.assertLess(float(residual.mean()), 0.05)
            self.assertEqual(len(plane["inlier_indices"]), plane["inlier_count"])
        # The weak-only ceiling at z = 8 is never fitted: that would be invented geometry.
        for plane in planes:
            self.assertFalse(abs(plane["normal"][2]) > 0.9
                             and abs(-plane["offset"] / plane["normal"][2] - 8.0) < 0.5,
                             msg=f"the weak ceiling was fitted: {plane['offset']}")
        self.assertGreater(planes[0]["inlier_count"], planes[1]["inlier_count"])

    def test_a_classify_result_is_accepted_as_labels(self):
        labels = {"labels": self.labels, "counts": {}, "fractions": {}}
        self.assertEqual([plane["inlier_count"] for plane in occ.plane_complete(self.points, labels)],
                         [plane["inlier_count"] for plane in self.planes()])

    def test_fitting_is_deterministic(self):
        first, second = self.planes(max_planes=2), self.planes(max_planes=2)
        self.assertEqual([plane["inlier_count"] for plane in first],
                         [plane["inlier_count"] for plane in second])
        self.assertEqual([np.round(plane["normal"], 12).tolist() for plane in first],
                         [np.round(plane["normal"], 12).tolist() for plane in second])
        self.assertEqual([float(plane["offset"]) for plane in first],
                         [float(plane["offset"]) for plane in second])

    def test_partly_observed_roof_yields_a_constrained_patch(self):
        plane = self.roof_plane()
        hole = {"min": (7.0, 0.0, 3.0), "max": (10.0, 6.0, 5.0)}
        patch = occ.constrained_extension(plane, hole)
        self.assertEqual(patch["layer"], "constrained")
        self.assertEqual(patch["kind"], "constrained_patch")
        self.assertEqual(patch["plane_inlier_count"], plane["inlier_count"])
        self.assertGreater(patch["plane_inlier_count"], 200)
        self.assertAlmostEqual(patch["plane_residual_m"], plane["residual_m"], places=9)
        self.assertLess(patch["plane_residual_m"], 0.05)
        self.assertEqual(patch["point_count"], len(patch["points"]))
        self.assertEqual(patch["layer"], "constrained")
        residual = np.abs(patch["points"] @ patch["plane_normal"] + patch["plane_offset"])
        self.assertLess(float(residual.max()), 1e-9, "the patch lies exactly on the fitted plane")
        self.assertAlmostEqual(float(np.linalg.norm(patch["plane_normal"])), 1.0, places=9)
        self.assertGreater(patch["area_m2"], 0.0)
        self.assertGreater(patch["outside_inlier_extent_m"], 0.0)
        self.assertEqual(patch["basis"], "measured plane fit")
        self.assertIn("not a measurement", patch["note"].lower())
        self.assertTrue(np.all(patch["points"][:, 0] >= 7.0 - 1e-9))
        self.assertLess(patch["outside_inlier_extent_m"], 4.0)

    def test_patch_also_accepts_a_polygon_region(self):
        plane = self.roof_plane()
        polygon = occ.constrained_extension(plane, {"polygon": [(7.0, 0.0, 4.0), (9.0, 0.0, 4.0),
                                                                (9.0, 4.0, 4.0), (7.0, 4.0, 4.0)]})
        self.assertEqual(polygon["layer"], "constrained")
        self.assertGreater(len(polygon["points"]), 0)
        self.assertTrue(np.all(polygon["points"][:, 1] <= 4.0 + 1e-9))

    def test_constrained_patch_cannot_be_relabelled_measured(self):
        plane = self.roof_plane()
        patch = occ.constrained_extension(plane, {"min": (7.0, 0.0, 3.0), "max": (10.0, 6.0, 5.0)})
        with self.assertRaisesRegex(ValueError, "constrained"):
            occ.blend([{"points": self.ground[:10], "layer": "measured"}, dict(patch, layer="measured")])
        stripped = {key: value for key, value in patch.items() if key != "kind"}
        with self.assertRaisesRegex(ValueError, "constrained"):
            occ.blend([dict(stripped, points=patch["points"], layer="measured")])
        # The guard reports what it checked rather than silently trusting the label.
        report = occ.assert_no_inferred(occ.blend([{"points": patch["points"], "layer": "constrained"}]))
        self.assertEqual(report["constrained"], len(patch["points"]))
        self.assertEqual(report["measured"], 0)

    def test_invalid_plane_inputs_raise(self):
        cases = [dict(ransac_distance=0.0), dict(ransac_distance=-1.0), dict(ransac_distance=np.nan),
                 dict(min_inliers=2), dict(min_inliers=0), dict(min_inliers=-5),
                 dict(max_planes=0), dict(max_planes=-1)]
        for options in cases:
            with self.subTest(case=str(options)):
                with self.assertRaises(ValueError):
                    self.planes(**options)
        with self.assertRaises(ValueError):  # no measured points at all: nothing to fit
            occ.plane_complete(self.points, np.full(len(self.points), "weak"))
        with self.assertRaises(ValueError):  # label count mismatch
            occ.plane_complete(self.points, self.labels[:-1])
        with self.assertRaises(ValueError):  # unknown layer name
            occ.plane_complete(self.points, np.full(len(self.points), "guessed"))
        with self.assertRaises(ValueError):  # malformed labels
            occ.plane_complete(self.points, np.arange(len(self.points)))
        with self.assertRaises(ValueError):
            occ.plane_complete(np.zeros((3, 2)), np.array(["measured"] * 3))
        with self.assertRaises(ValueError):  # fewer measured points than a plane needs
            occ.plane_complete(self.points[:2], np.array(["measured", "measured"]))
        with self.assertRaises(ValueError):  # a plane that is not unit-normalised evidence
            occ.plane_complete(self.points, {"no_labels": 1})

    def test_invalid_constrained_extension_inputs_raise(self):
        plane = self.roof_plane()
        with self.assertRaises(ValueError):  # no region geometry
            occ.constrained_extension(plane, {})
        with self.assertRaises(ValueError):  # inverted bounds
            occ.constrained_extension(plane, {"min": (9.0, 0.0, 3.0), "max": (7.0, 6.0, 5.0)})
        with self.assertRaises(ValueError):  # a region the plane does not cross
            occ.constrained_extension(plane, {"min": (0.0, 0.0, 1.0), "max": (1.0, 1.0, 2.0)})
        with self.assertRaises(ValueError):  # unknown region kind
            occ.constrained_extension(plane, {"sphere": 3.0})
        with self.assertRaises(ValueError):
            occ.constrained_extension(plane, {"min": (7.0, 0.0, 3.0), "max": (10.0, 6.0, 5.0)},
                                      cell_size_m=0.0)
        with self.assertRaises(ValueError):  # a plane without its inlier count is not evidence
            occ.constrained_extension({k: v for k, v in plane.items() if k != "inlier_count"},
                                      {"min": (7.0, 0.0, 3.0), "max": (10.0, 6.0, 5.0)})
        with self.assertRaises(ValueError):  # a plane fitted from too few points to constrain anything
            occ.constrained_extension(dict(plane, inlier_count=2),
                                      {"min": (7.0, 0.0, 3.0), "max": (10.0, 6.0, 5.0)})
        with self.assertRaises(ValueError):
            occ.constrained_extension({"normal": [0.0, 0.0, 2.0], "offset": -8.0,
                                       "inlier_count": 500, "residual_m": 0.01},
                                      {"min": (7.0, 0.0, 3.0), "max": (10.0, 6.0, 5.0)})
        with self.assertRaises(ValueError):
            occ.constrained_extension("not a plane", {"min": (0, 0, 0), "max": (1, 1, 1)})


class BlendAndGuardTests(unittest.TestCase):
    def setUp(self):
        self.measured = np.random.default_rng(3).normal(0.0, 1.0, (40, 3))
        self.weak = np.random.default_rng(4).normal(0.0, 1.0, (12, 3))
        self.constrained = np.random.default_rng(5).normal(0.0, 1.0, (8, 3))
        self.inferred = np.random.default_rng(6).normal(0.0, 1.0, (5, 3))

    def layer(self, points, name):
        return {"points": points, "layer": name}

    def test_blend_keeps_the_layers_apart_with_integer_codes(self):
        result = occ.blend([self.layer(self.measured, "measured"),
                            self.layer(self.weak, "weak"),
                            self.layer(self.constrained, "constrained")])
        self.assertEqual(len(result["points"]), 60)
        self.assertEqual(result["layer"].dtype.kind, "i")
        self.assertEqual(result["layer"].tolist(),
                         [occ.LAYER_CODE["measured"]] * 40 + [occ.LAYER_CODE["weak"]] * 12
                         + [occ.LAYER_CODE["constrained"]] * 8)
        self.assertEqual(result["counts"], {"measured": 40, "weak": 12, "constrained": 8})
        self.assertEqual(np.asarray(result["layer_names"]).tolist(),
                         ["measured"] * 40 + ["weak"] * 12 + ["constrained"] * 8)
        self.assertTrue(np.array_equal(result["points"][:40], self.measured))
        self.assertTrue(np.array_equal(result["points"][-8:], self.constrained))
        self.assertEqual([item["count"] for item in result["provenance"]], [40, 12, 8])
        self.assertEqual([item["layer"] for item in result["provenance"]],
                         ["measured", "weak", "constrained"])
        self.assertEqual([item["start"] for item in result["provenance"]], [0, 40, 52])
        self.assertIn("constrained", result["note"].lower())

    def test_blend_accepts_per_point_labels(self):
        labels = np.array(["measured"] * 30 + ["unobserved"] * 10)
        result = occ.blend([{"points": self.measured, "labels": labels}])
        self.assertEqual(result["counts"], {"measured": 30, "unobserved": 10})
        self.assertEqual(result["layer"].tolist(),
                         [occ.LAYER_CODE["measured"]] * 30 + [occ.LAYER_CODE["unobserved"]] * 10)

    def test_assert_no_inferred_accepts_a_measured_plus_constrained_export(self):
        result = occ.blend([self.layer(self.measured, "measured"),
                            self.layer(self.constrained, "constrained")])
        report = occ.assert_no_inferred(result)
        self.assertEqual(report["inferred"], 0)
        self.assertEqual(report["measured"], 40)
        self.assertEqual(report["constrained"], 8)
        self.assertEqual(report["points"], 48)
        self.assertTrue(report["clean"])
        self.assertIn("constrained", report["note"].lower())

    def test_assert_no_inferred_raises_as_soon_as_any_point_is_inferred(self):
        result = occ.blend([self.layer(self.measured, "measured"),
                            self.layer(self.inferred, "inferred")])
        with self.assertRaisesRegex(ValueError, "inferred"):
            occ.assert_no_inferred(result)
        with self.assertRaisesRegex(ValueError, "inferred"):
            occ.assert_no_inferred({"layer": np.array([0, 0, occ.LAYER_CODE["inferred"]])})
        with self.assertRaisesRegex(ValueError, "inferred"):
            occ.assert_no_inferred(["measured", "inferred", "measured"])
        with self.assertRaisesRegex(ValueError, "inferred"):
            occ.assert_no_inferred(np.array([0, 4]))

    def test_measured_and_constrained_stay_separable_after_blending(self):
        result = occ.blend([self.layer(self.measured, "measured"),
                            self.layer(self.constrained, "constrained")])
        occ.assert_no_inferred(result)
        codes = np.asarray(result["layer"])
        measured_only = result["points"][codes == occ.LAYER_CODE["measured"]]
        self.assertEqual(len(measured_only), 40)
        self.assertTrue(np.array_equal(measured_only, self.measured))

    def test_invalid_blend_inputs_raise(self):
        cases = [
            [],
            None,
            "measured",
            [{"points": self.measured}],
            [{"layer": "measured"}],
            [{"points": self.measured, "layer": "guessed"}],
            [{"points": np.zeros((3, 2)), "layer": "measured"}],
            [{"points": np.full((3, 3), np.nan), "layer": "measured"}],
            [{"points": self.measured, "labels": np.array(["measured"] * 5)}],
            [{"points": self.measured, "layer": "measured", "labels": np.full(40, "weak")}],
            [{"points": self.measured, "layer": occ.LAYER_CODE["measured"]}],
            [{"points": self.measured, "labels": np.full(40, 0)}],
            [{"points": np.zeros((0, 3)), "layer": "measured"}],
        ]
        for layers in cases:
            with self.subTest(case=str(type(layers)) + str(len(layers) if isinstance(layers, list) else "")):
                with self.assertRaises(ValueError):
                    occ.blend(layers)


class CoverageDenominatorTests(unittest.TestCase):
    def setUp(self):
        self.points = scene_cloud()
        self.hidden = occ.hidden_regions(camera_centres(), self.points, scene_views(),
                                         cell_size_m=CELL, angular_threshold_deg=5.0)

    def test_denominator_is_at_least_one_and_reports_the_excluded_area(self):
        result = occ.coverage_denominator(self.hidden, self.hidden["total_cells"])
        self.assertGreaterEqual(result["denominator"], 1)
        self.assertEqual(result["denominator"],
                         max(self.hidden["total_cells"] - self.hidden["hidden_cells"], 1))
        self.assertEqual(result["total_cells"], self.hidden["total_cells"])
        self.assertEqual(result["excluded_cells"], self.hidden["hidden_cells"])
        self.assertEqual(result["visible_cells"], result["denominator"])
        self.assertAlmostEqual(result["excluded_area_m2"],
                               self.hidden["hidden_cells"] * self.hidden["cell_size_m"] ** 2, places=6)
        self.assertGreater(result["excluded_area_m2"], 1.0)
        self.assertAlmostEqual(result["excluded_fraction"] + result["kept_fraction"], 1.0, places=6)
        self.assertTrue(result["publish_together"])
        self.assertIn("excluded", result["note"].lower())
        self.assertGreater(result["excluded_cells"], 1)

    def test_denominator_never_collapses_even_when_nothing_is_visible(self):
        total = self.hidden["total_cells"]
        result = occ.coverage_denominator(total, total, cell_size_m=CELL)
        self.assertEqual(result["denominator"], 1)
        self.assertEqual(result["visible_cells"], 0)
        self.assertEqual(result["excluded_cells"], total)
        self.assertAlmostEqual(result["excluded_area_m2"], total * CELL ** 2, places=6)
        self.assertAlmostEqual(result["excluded_fraction"], 1.0, places=6)
        self.assertTrue(result["denominator_was_floored"])
        empty = occ.coverage_denominator(0, 0, cell_size_m=0.5)
        self.assertEqual(empty["denominator"], 1)
        self.assertEqual(empty["excluded_area_m2"], 0.0)

    def test_excluding_cells_cannot_shrink_the_reported_excluded_area(self):
        half = occ.coverage_denominator(20, 100, cell_size_m=1.0)
        whole = occ.coverage_denominator(95, 100, cell_size_m=1.0)
        self.assertGreater(whole["excluded_area_m2"], half["excluded_area_m2"])
        self.assertLess(whole["denominator"], half["denominator"])
        self.assertEqual(half["denominator"], 80)

    def test_cell_size_may_be_carried_by_the_hidden_regions_result(self):
        result = occ.coverage_denominator(self.hidden, self.hidden["total_cells"])
        self.assertEqual(result["cell_size_m"], self.hidden["cell_size_m"])
        explicit = occ.coverage_denominator(self.hidden["hidden_cells"], self.hidden["total_cells"],
                                           cell_size_m=2.0)
        self.assertEqual(explicit["cell_size_m"], 2.0)
        self.assertAlmostEqual(explicit["excluded_area_m2"], explicit["excluded_cells"] * 4.0, places=6)

    def test_denominator_rejects_impossible_or_unquantified_input(self):
        total = self.hidden["total_cells"]
        cases = [((total + 1, total), dict(cell_size_m=CELL)),
                 ((5.5, 10), dict(cell_size_m=CELL)),
                 ((-1, 10), dict(cell_size_m=CELL)),
                 ((5, 10), dict(cell_size_m=0.0)),
                 ((5, 10), dict(cell_size_m=-CELL)),
                 ((5, 10), dict(cell_size_m=np.nan)),
                 ((5, 0), dict(cell_size_m=CELL)),
                 ((5, np.nan), dict(cell_size_m=CELL)),
                 ((5, 1e300), dict(cell_size_m=CELL)),
                 (("lots", 10), dict(cell_size_m=CELL)),
                 (({"hidden_cells": -3, "cell_size_m": 1.0}, 10), {}),
                 (({"cell_size_m": 1.0}, 10), {})]
        for args, options in cases:
            with self.subTest(case=str(args) + str(options)):
                with self.assertRaises(ValueError):
                    occ.coverage_denominator(*args, **options)
        with self.assertRaises(ValueError):  # without a cell size there is no excluded area
            occ.coverage_denominator(5, 10)
        with self.assertRaises(ValueError):
            occ.coverage_denominator([{"reason": "occluded"}], 10, cell_size_m=CELL)


class InferencePolicyTests(unittest.TestCase):
    def test_generative_completion_is_excluded_from_measured_products(self):
        policy = occ.inference_policy()
        self.assertIs(policy["generative_completion_allowed_in_measured_products"], False)
        self.assertIs(policy["has_inpainting_entry_point"], False)
        self.assertIn("measured", policy["permitted_layers"])
        self.assertIn("constrained", policy["permitted_layers"])
        self.assertIn("inferred", policy["excluded_layers"])
        self.assertEqual(policy["measured_export_layers"], occ.MEASURED_EXPORT_LAYERS)
        self.assertTrue(any("never" in line.lower() for line in policy["reasons"]))
        self.assertTrue(any("occlu" in line.lower() for line in policy["reasons"]))
        self.assertTrue(all(not callable(value) for value in policy.values()))
        self.assertEqual(occ.inference_policy(), policy, "the policy is a statement, not a switch")

    def test_module_offers_no_inpainting_entry_point(self):
        self.assertFalse(hasattr(occ, "inpaint_holes"))
        self.assertEqual([name for name in dir(occ) if "inpaint" in name.lower()], [])
        self.assertEqual([name for name in dir(occ) if name.lower().startswith("generate")], [])

    def test_single_pass_limits_are_stated_plainly(self):
        limits = occ.inference_policy()["single_pass_limits"]
        self.assertTrue(any("behind" in line.lower() or "never" in line.lower() for line in limits))
        self.assertGreaterEqual(len(limits), 3)


if __name__ == "__main__":
    unittest.main()
