"""MISB ST 0601 KLV in a STANAG 4609 transport stream (MIL-16): encode, wrap, read back."""
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def track(n=30, hae=True):
    rows = []
    for k in range(n):
        f = {"time_us": 1_758_000_000_000_000 + k * 100_000,
             "sensor_latitude_deg": 30.7333 + k * 1e-5, "sensor_longitude_deg": 76.7794 + k * 2e-5,
             "platform_heading_deg": 45.0, "platform_pitch_deg": 1.5, "platform_roll_deg": -2.0,
             "sensor_rel_azimuth_deg": 90.0, "sensor_rel_elevation_deg": -60.0, "sensor_rel_roll_deg": 0.0,
             "sensor_hfov_deg": 30.0, "sensor_vfov_deg": 17.0}
        f["sensor_hae_m" if hae else "sensor_msl_m"] = 420.0 + k * 0.5
        rows.append(f)
    return rows


class KlvTests(unittest.TestCase):
    def setUp(self):
        self.k = importlib.import_module("survey_klv")
        self.inputs = importlib.import_module("survey_inputs")

    def test_local_set_round_trip_and_checksum(self):
        f = track(1)[0]
        ls = self.k.encode_local_set(f)
        out = self.k.decode_local_set(ls)
        self.assertAlmostEqual(out["sensor_latitude_deg"], f["sensor_latitude_deg"], places=6)
        self.assertAlmostEqual(out["sensor_longitude_deg"], f["sensor_longitude_deg"], places=6)
        self.assertAlmostEqual(out["sensor_hae_m"], 420.0, delta=0.2)          # 16-bit over 19.9 km
        self.assertAlmostEqual(out["platform_heading_deg"], 45.0, delta=0.01)
        self.assertEqual(out["time_us"], f["time_us"])
        bad = bytearray(ls)
        bad[25] ^= 0x01
        with self.assertRaises(ValueError):
            self.k.decode_local_set(bytes(bad))

    def test_crc16_reference(self):
        self.assertEqual(self.k.crc16_ccitt(b"123456789"), 0x29B1)              # CCITT-FALSE check value

    def test_transport_stream_to_telemetry_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "isr.ts"
            path.write_bytes(self.k.write_ts([self.k.encode_local_set(f) for f in track()], fps=10.0))
            self.assertEqual(self.inputs.sniff_format(path).kind, "klv_ts")
            rows = self.inputs.from_flight_log(path, horizontal_std_m=5.0, vertical_std_m=8.0)
            self.assertEqual(len(rows), 30)
            self.assertAlmostEqual(rows[10]["t_sec"], 1.0, places=3)                   # PTS, video clock
            self.assertAlmostEqual(rows[10]["latitude_deg"], 30.7333 + 10e-5, places=6)
            self.assertEqual(rows[0]["horizontal_std_m"], 5.0)
            info = self.inputs.source_info(path)
            self.assertEqual(info["altitude_datum"], "ellipsoidal")
            self.assertAlmostEqual(info["gimbal"][0]["pitch_deg"], -58.5, delta=0.01)
            with self.assertRaises(ValueError):
                self.inputs.from_flight_log(path)                                       # no uncertainty declared

    def test_msl_heights_are_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "isr.ts"
            path.write_bytes(self.k.write_ts([self.k.encode_local_set(f) for f in track(5, hae=False)]))
            self.inputs.from_flight_log(path, horizontal_std_m=5.0, vertical_std_m=8.0)
            info = self.inputs.source_info(path)
            self.assertEqual(info["altitude_datum"], "mean_sea_level")
            self.assertTrue(any("geoid" in w for w in info["warnings"]))

    def test_not_a_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.ts"
            path.write_bytes(b"\x00" * 1000)
            with self.assertRaises(ValueError):
                self.k.parse_klv(path, horizontal_std_m=1, vertical_std_m=1)


if __name__ == "__main__":
    unittest.main()
