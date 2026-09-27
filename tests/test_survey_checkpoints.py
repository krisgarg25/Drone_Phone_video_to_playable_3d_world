"""F1/F2: checkpoint upload and the one-click survey path, on a synthetic aligned scene.

The scene is the same kind survey_workflow's own tests build: a telemetry track, camera
centres that are an exact similarity of it, and a two-point sparse model. Checkpoints are
generated from known ENU points with a known error added, written out in the frames an
operator actually has (degrees, UTM, EGM96 heights), and must come back as that error.
These tests prove the conversion and the wiring, not any real scene's accuracy.
"""
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np  # noqa: E402

import survey_checkpoints as checkpoints  # noqa: E402
import survey_crs as crs  # noqa: E402
import survey_geoid as geoid  # noqa: E402
import survey_workflow as survey  # noqa: E402

METADATA = {"schema_version": 1, "time_reference": "video", "time_offset_s": 0,
            "altitude_datum": "ellipsoidal", "position_reference": "camera_center",
            "single_pass": True, "video_duration_s": 600}
TELEMETRY = ("t_sec,latitude_deg,longitude_deg,altitude_m,horizontal_std_m,vertical_std_m\n"
             "0,28,77,100,1,2\n1,28.0001,77,100,1,2\n"
             "2,28.0001,77.0001,101,1,2\n3,28,77.0001,102,1,2\n")


class Scene:
    def __init__(self, root):
        self.root = root
        source = root / "videos/flight"
        source.mkdir(parents=True)
        (source / "flight.mp4").write_bytes(b"test-video-not-decoded")
        self.work = root / "work/flight"

    def inputs(self):
        survey.save_inputs(self.root, "flight", TELEMETRY, METADATA)

    def cameras(self):
        preparation = survey.read_json(self.work / "survey/preparation.json")
        rows = []
        for i, sample in enumerate(preparation["telemetry"]["samples"]):
            center = (np.asarray(sample["position"]) - [10, 20, 30]) / 2.0
            rows.append({"file": f"{i}.jpg", "t_sec": sample["t_sec"],
                         "camera": {"R_rowmajor": np.eye(3).ravel().tolist(),
                                    "t": (-center).tolist()}})
        (self.work / "keyframes_poses.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        sparse = self.work / "colmap/sparse/txt/points3D.txt"
        sparse.parent.mkdir(parents=True, exist_ok=True)
        sparse.write_text("1 0 0 0 10 20 30 0.1\n2 1 2 3 40 50 60 0.1\n")


def as_csv(header, rows):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


class CheckpointUploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.scene = Scene(Path(self.tmp.name))
        self.scene.inputs()
        survey.prepare_scene(self.scene.root, "flight")
        self.scene.cameras()
        state = survey.align_scene(self.scene.root, "flight")
        self.alignment = survey.read_json(self.scene.work / "survey/georeference.json")
        self.frame = state["alignment"]["coordinate_frame"]
        origin = self.frame["origin"]
        self.origin = (origin["latitude_deg"], origin["longitude_deg"], origin["altitude_m"])
        rng = np.random.default_rng(3)
        self.model_enu = rng.uniform([-40, -40, -5], [40, 40, 5], (10, 3))
        self.error = np.array([0.30, -0.40, 0.12])      # |xy| = 0.5 m, z = 0.12 m
        self.surveyed_enu = self.model_enu - self.error  # model = surveyed + error

    def geodetic(self, enu):
        return crs.enu_to_geodetic(enu, *self.origin)

    def upload(self, text, **options):
        return survey.save_checkpoints(self.scene.root, "flight", text, **options)

    def accuracy(self, state):
        return next(c for c in state["evaluation"]["criteria"] if c["id"] == "accuracy")

    def test_degrees_against_the_utm_product_recover_the_known_error(self):
        ref = self.geodetic(self.surveyed_enu)
        model_geo = self.geodetic(self.model_enu)
        zone, hemisphere, _ = crs.utm_zone_for(*self.origin[:2])
        east, north = crs.utm_forward(model_geo[:, 0], model_geo[:, 1], zone, hemisphere)
        rows = [[f"CP{i}", f"{a[0]:.10f}", f"{a[1]:.10f}", f"{a[2]:.4f}",
                 f"{e:.4f}", f"{n:.4f}", f"{b[2]:.4f}"]
                for i, (a, b, e, n) in enumerate(zip(ref, model_geo, east, north))]
        state = self.upload(as_csv(checkpoints.TEMPLATE.strip().split(","), rows), withheld=True)
        metrics = self.accuracy(state)["metrics"]
        self.assertEqual(state["status"], "evaluated")
        # Ten points are split into controls and hold-outs; the headline is the hold-out
        # set's unaligned error, which a constant offset leaves at exactly that offset.
        split = state["evaluation"]["accuracy_split"]
        self.assertEqual(split["counts"]["rows"], 10)
        self.assertEqual(metrics["count"], 10 - split["counts"]["controls"])
        self.assertAlmostEqual(metrics["horizontal_rmse_m"], 0.5, delta=0.002)
        self.assertAlmostEqual(metrics["vertical_rmse_m"], 0.12, delta=0.002)
        source = state["evaluation"]["checkpoint_upload"]
        self.assertEqual((source["reference_form"], source["model_form"]), ("geodetic", "utm"))
        self.assertTrue(source["withheld_from_reconstruction"])
        # The operator's declaration reached the accuracy report instead of being assumed.
        self.assertNotIn("independence is unverified",
                         " ".join(split["accuracy_validation_reasons"]))

    @unittest.skipUnless(geoid.available(), "EGM96 grid not installed")
    def test_msl_heights_on_a_gcp_sheet_are_converted_not_mixed(self):
        ref = self.geodetic(self.surveyed_enu)
        msl = geoid.to_orthometric(ref[:, 0], ref[:, 1], ref[:, 2])
        rows = [[f"G{i}", f"{a[0]:.10f}", f"{a[1]:.10f}", f"{h:.4f}", *map("{:.4f}".format, m)]
                for i, (a, h, m) in enumerate(zip(ref, msl, self.model_enu))]
        header = ["id", "ref_lat_deg", "ref_lon_deg", "ref_height_m",
                  "model_e_m", "model_n_m", "model_u_m"]
        right = self.accuracy(self.upload(as_csv(header, rows),
                                          reference_height_datum="egm96"))["metrics"]
        self.assertAlmostEqual(right["vertical_rmse_m"], 0.12, delta=0.002)
        # The same sheet read as ellipsoidal is off by the geoid (~53 m at Delhi) - the
        # error a mixed datum would have shipped as "accuracy".
        wrong = self.accuracy(self.upload(as_csv(header, rows)))["metrics"]
        self.assertGreater(wrong["vertical_rmse_m"], 40.0)

    def test_without_the_declaration_independence_stays_unverified(self):
        header = ["id", "ref_e_m", "ref_n_m", "ref_u_m", "model_e_m", "model_n_m", "model_u_m"]
        rows = [[f"P{i}", *a, *b] for i, (a, b) in enumerate(zip(self.surveyed_enu,
                                                                  self.model_enu))]
        state = self.upload(as_csv(header, rows))
        self.assertIn("independence is unverified",
                      " ".join(state["evaluation"]["accuracy_split"]["accuracy_validation_reasons"]))

    def test_a_hand_edit_drops_the_declaration(self):
        header = ["id", "ref_e_m", "ref_n_m", "ref_u_m", "model_e_m", "model_n_m", "model_u_m"]
        rows = [[f"P{i}", *a, *b] for i, (a, b) in enumerate(zip(self.surveyed_enu,
                                                                  self.model_enu))]
        self.upload(as_csv(header, rows), withheld=True)
        path = self.scene.work / "survey/checkpoints.json"
        data = survey.read_json(path)
        data["checkpoints"][0]["reference"][0] += 5.0
        survey.write_json(path, data)
        state = survey.evaluate_scene(self.scene.root, "flight")
        self.assertEqual(state["evaluation"]["checkpoint_upload"]["status"], "hand_written")

    def test_bad_sheets_are_refused_with_the_row(self):
        header = ["id", "ref_e_m", "ref_n_m", "ref_u_m", "model_e_m", "model_n_m", "model_u_m"]
        cases = {
            "duplicate id": [["A", 0, 0, 0, 0, 0, 0], ["A", 1, 1, 1, 1, 1, 1]],
            "must be a number": [["A", 0, "x", 0, 0, 0, 0]],
            "different frames": [["A", 0, 0, 0, 500000, 3100000, 0]],
        }
        for message, rows in cases.items():
            with self.assertRaisesRegex(ValueError, message):
                self.upload(as_csv(header, rows))
        with self.assertRaisesRegex(ValueError, "exactly one column set"):
            self.upload("id,x,y,z\nA,1,2,3\n")
        with self.assertRaisesRegex(ValueError, "does not apply"):
            self.upload(as_csv(header, [["A", 0, 0, 0, 0, 0, 0]]),
                        reference_height_datum="egm96")


class OneClickTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.scene = Scene(Path(self.tmp.name))

    def stages(self, result):
        return [(row["stage"], row["status"]) for row in result["trace"]]

    def test_advance_runs_every_cpu_step_and_stops_at_the_gpu(self):
        result = survey.advance(self.scene.root, "flight")
        self.assertEqual(self.stages(result), [("inputs", "needs_you")])
        self.scene.inputs()
        result = survey.advance(self.scene.root, "flight")
        self.assertEqual(self.stages(result), [("prepare", "done"), ("reconstruct", "needs_you")])
        steps = {row["id"]: row["status"] for row in result["state"]["steps"]}
        self.assertEqual(steps, {"inputs": "done", "prepare": "done", "reconstruct": "needs_you",
                                 "align": "waiting", "checkpoints": "waiting",
                                 "evaluate": "waiting"})
        self.scene.cameras()   # what an approved reconstruction leaves behind
        result = survey.advance(self.scene.root, "flight")
        self.assertEqual(self.stages(result), [("align", "done"), ("evaluate", "done")])
        steps = {row["id"]: row["status"] for row in result["state"]["steps"]}
        self.assertEqual(steps["checkpoints"], "optional")
        self.assertEqual(steps["evaluate"], "done")
        # Nothing left to do: a second click changes nothing and says so.
        self.assertEqual(self.stages(survey.advance(self.scene.root, "flight")), [])

    def test_changed_inputs_are_prepared_again(self):
        self.scene.inputs()
        self.scene_ready = survey.advance(self.scene.root, "flight")
        telemetry = self.scene.root / "videos/flight/telemetry.csv"
        telemetry.write_text(TELEMETRY + "4,28,77,103,1,2\n")
        result = survey.advance(self.scene.root, "flight")
        self.assertEqual(result["trace"][0], {"stage": "prepare", "status": "done"})


if __name__ == "__main__":
    unittest.main()
