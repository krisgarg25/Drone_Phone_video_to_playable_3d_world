"""CPU tests for turning survey GPS into COLMAP bundle-adjustment priors."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import survey_priors as priors


def telemetry():
    return {"schema_version": 1,
            "coordinate_frame": {"type": "ENU", "units": "m", "geodetic_crs": "EPSG:4979",
                                 "altitude_datum": "ellipsoidal",
                                 "origin": {"latitude_deg": 28.0, "longitude_deg": 77.0,
                                            "altitude_m": 100.0}},
            "samples": [{"t_sec": float(i), "position": [i * 2.0, 0.0, 1.0],
                         "horizontal_std_m": 1.5, "vertical_std_m": 2.5} for i in range(5)]}


class SurveyPriorsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.keyframes = self.root / "keyframes.jsonl"
        self.keyframes.write_text("\n".join(json.dumps(
            {"file": f"clip/{i:05d}.jpg", "clip": "clip", "t_sec": float(i)})
            for i in range(5)) + "\n", encoding="utf-8")
        self.preparation = self.root / "preparation.json"
        self.preparation.write_text(json.dumps({"id": "abc", "telemetry": telemetry()}),
                                    encoding="utf-8")

    def read(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_writes_cartesian_priors_for_every_anchored_keyframe(self):
        out = self.root / "pose_priors.jsonl"
        summary = priors.write_priors(self.keyframes, self.preparation, out)
        rows = self.read(out)
        self.assertEqual([r["file"] for r in rows], [f"clip/{i:05d}.jpg" for i in range(5)])
        for index, row in enumerate(rows):
            self.assertEqual(set(row), {"file", "position", "std"})
            np.testing.assert_allclose(row["position"], [index * 2.0, 0.0, 1.0], atol=1e-9)
            self.assertGreater(row["std"], 0)
        self.assertEqual(summary["written"], 5)
        self.assertEqual(summary["coordinate_frame"], telemetry()["coordinate_frame"])

    def test_keyframes_without_a_bracket_are_skipped_not_extrapolated(self):
        self.keyframes.write_text("\n".join(json.dumps({"file": f"f{i}.jpg", "t_sec": t})
                                            for i, t in enumerate([0.0, 1.0, 10.0])) + "\n",
                                  encoding="utf-8")
        out = self.root / "p2.jsonl"
        summary = priors.write_priors(self.keyframes, self.preparation, out)
        self.assertEqual([r["file"] for r in self.read(out)], ["f0.jpg", "f1.jpg"])
        self.assertEqual(summary["skipped"], 1)

    def test_interval_wider_than_max_gap_s_is_skipped(self):
        # Samples are 1 s apart: an exact hit still anchors, a mid-interval query
        # does not when the allowed gap is smaller than the interval.
        self.keyframes.write_text(json.dumps({"file": "exact.jpg", "t_sec": 1.0}) + "\n"
                                  + json.dumps({"file": "mid.jpg", "t_sec": 0.5}) + "\n",
                                  encoding="utf-8")
        out = self.root / "p3.jsonl"
        summary = priors.write_priors(self.keyframes, self.preparation, out, max_gap_s=0.4)
        self.assertEqual([r["file"] for r in self.read(out)], ["exact.jpg"])
        self.assertEqual(summary["skipped"], 1)

    def test_no_anchored_keyframe_raises_instead_of_writing_an_empty_prior_set(self):
        self.keyframes.write_text(json.dumps({"file": "x.jpg", "t_sec": 99.0}) + "\n",
                                  encoding="utf-8")
        out = self.root / "p4.jsonl"
        with self.assertRaises(ValueError):
            priors.write_priors(self.keyframes, self.preparation, out)
        self.assertFalse(out.exists())

    def test_output_carries_no_absolute_paths(self):
        out = self.root / "p5.jsonl"
        priors.write_priors(self.keyframes, self.preparation, out)
        self.assertNotIn(str(self.root), out.read_text(encoding="utf-8"))

    def test_invalid_inputs_raise(self):
        for keyframes, preparation in ((self.keyframes, self.root / "missing.json"),
                                       (self.root / "missing.jsonl", self.preparation)):
            with self.subTest(keyframes=str(keyframes)), self.assertRaises((ValueError, OSError)):
                priors.write_priors(keyframes, preparation, self.root / "p6.jsonl")
        bad = self.root / "bad.jsonl"
        bad.write_text(json.dumps({"file": "x.jpg", "t_sec": "not-a-time"}) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            priors.write_priors(bad, self.preparation, self.root / "p7.jsonl")


if __name__ == "__main__":
    unittest.main()
