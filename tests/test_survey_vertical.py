"""CPU-only tests for takeoff-anchored height fusion (M12) on synthetic flights.

The GNSS error model is deliberately correlated (a first-order Gauss-Markov process
with a 60 s time constant), because white noise would make any averaging look far
better than it is on a real receiver.
"""
import importlib
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

TAKEOFF = 212.4  # ellipsoidal metres


def flight(duration=600.0, rate=1.0, *, sigma=2.0, tau=60.0, baro_drift=0.0, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / rate)
    profile = np.clip(t * 2.0, 0, 60) + 5 * np.sin(t / 50.0)          # climb, then undulate
    phi = math.exp(-1.0 / (rate * tau))
    noise = np.zeros_like(t)
    noise[0] = rng.normal(0, sigma)
    for i in range(1, len(t)):
        noise[i] = phi * noise[i - 1] + rng.normal(0, sigma * math.sqrt(1 - phi ** 2))
    truth = TAKEOFF + profile
    gnss = [dict(t_sec=float(ti), latitude_deg=28.6, longitude_deg=77.2,
                 altitude_m=float(h + n), horizontal_std_m=1.5, vertical_std_m=3.0)
            for ti, h, n in zip(t, truth, noise)]
    baro = profile + rng.normal(0, 0.05, len(t)) + baro_drift * t / duration
    relative = [dict(t_sec=float(ti), relative_altitude_m=float(b)) for ti, b in zip(t, baro)]
    return gnss, relative, truth


class VerticalTests(unittest.TestCase):
    def setUp(self):
        self.v = importlib.import_module("survey_vertical")

    def rmse(self, rows, truth):
        return float(np.sqrt(np.mean((np.array([r["altitude_m"] for r in rows]) - truth) ** 2)))

    def test_fusion_beats_raw_gnss_height_over_many_flights(self):
        raw, better, errors, stds = [], [], [], []
        for seed in range(60):
            gnss, relative, truth = flight(seed=seed)
            fused, report = self.v.fuse_heights(gnss, relative)
            raw.append(self.rmse(gnss, truth))
            better.append(self.rmse(fused, truth))
            errors.append(abs(report["takeoff_height_m"] - TAKEOFF))
            stds.append(report["offset_std_m"])
            self.assertEqual(report["samples_fused"], len(gnss))
        self.assertLess(np.mean(better), 0.5 * np.mean(raw), (np.mean(raw), np.mean(better)))
        # The stated uncertainty covers the error actually made (~95% expected at 2 sigma).
        coverage = np.mean(np.array(errors) < 2 * np.array(stds))
        self.assertGreaterEqual(coverage, 0.85, coverage)

    def test_offset_std_uses_correlation_not_sample_count(self):
        gnss, relative, _ = flight()
        _, slow = self.v.fuse_heights(gnss, relative, correlation_s=60.0)
        _, naive = self.v.fuse_heights(gnss, relative, correlation_s=1.0)
        self.assertGreater(slow["offset_std_m"], 5 * naive["offset_std_m"])

    def test_declared_takeoff_height_is_used_exactly(self):
        gnss, relative, truth = flight()
        fused, report = self.v.fuse_heights(gnss, relative, takeoff_height_m=TAKEOFF,
                                            takeoff_std_m=0.05)
        self.assertEqual(report["basis"], "declared")
        self.assertLess(self.rmse(fused, truth), 0.1)
        with self.assertRaises(ValueError):
            self.v.fuse_heights(gnss, relative, takeoff_height_m=TAKEOFF)

    def test_baro_drift_is_reported_and_widens_uncertainty(self):
        gnss, relative, _ = flight(baro_drift=6.0, sigma=0.3)
        _, report = self.v.fuse_heights(gnss, relative)
        self.assertAlmostEqual(report["drift_m"], -6.0, delta=0.6)
        self.assertGreater(report["fused_vertical_std_m"], 2.5)
        self.assertIn("warnings", report)

    def test_rows_without_baro_keep_their_gnss_height(self):
        gnss, relative, _ = flight(duration=120)
        relative = [r for r in relative if r["t_sec"] < 60]
        fused, report = self.v.fuse_heights(gnss, relative)
        self.assertEqual(report["samples_kept_gnss"], 60)
        self.assertEqual(fused[-1]["altitude_m"], gnss[-1]["altitude_m"])

    def test_too_little_overlap_is_refused(self):
        gnss, relative, _ = flight(duration=60)
        with self.assertRaises(ValueError):
            self.v.fuse_heights(gnss, [dict(t_sec=500.0 + i, relative_altitude_m=1.0)
                                       for i in range(30)])

    def test_csv_round_trip(self):
        gnss, relative, _ = flight(duration=30)
        fused, _ = self.v.fuse_heights(gnss, relative)
        rows = self.v.read_telemetry_csv(self.v.write_telemetry_csv(fused))
        self.assertEqual(len(rows), 30)
        self.assertAlmostEqual(float(rows[5]["altitude_m"]), fused[5]["altitude_m"], places=9)


if __name__ == "__main__":
    unittest.main()
