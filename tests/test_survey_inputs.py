"""CPU-only tests for normalising real drone telemetry into the survey_georef contract.

Inputs are written to a temporary directory; nothing here touches work/, videos/
or results/. scripts/survey_georef is imported for real and used as the oracle, so
these tests prove the produced rows and metadata are actually consumable by
normalize_telemetry / validate_metadata rather than merely self-consistent.
"""
import csv
import datetime
import importlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import survey_georef as georef


FIELDS = ("t_sec", "latitude_deg", "longitude_deg", "altitude_m",
          "horizontal_std_m", "vertical_std_m")
POSITIONAL = ("t_sec", "latitude_deg", "longitude_deg", "altitude_m")

# A u-blox-style GPX: <ele> (datum unspecified by the GPX schema) plus a namespaced
# dilution extension and wall-clock times.
GPX_FLIGHT = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1"
     xmlns:ubx="http://www.u-blox.com/sbt/2.0.0">
  <metadata><time>2026-09-22T08:00:00Z</time></metadata>
  <trk><name>flight</name><trkseg>
    <trkpt lat="47.3769" lon="8.5415"><ele>488.2</ele><time>2026-09-22T08:00:00Z</time>
      <extensions><ubx:hdop>1.4</ubx:hdop></extensions></trkpt>
    <trkpt lat="47.3770" lon="8.5416"><ele>489.2</ele><time>2026-09-22T08:00:02Z</time>
      <extensions><ubx:hdop>1.6</ubx:hdop></extensions></trkpt>
  </trkseg></trk>
</gpx>
"""

SRT_FLIGHT = """1
00:00:00,000 --> 00:00:00,500
Time:0 Latitude:22.5752 Longitude:113.9449 AbsoluteAltitude:53.81 RelativeAltitude:1.20 GimbalPitch:-25.00 GimbalRoll:0.00 GimbalYaw:-82.40

2
00:00:01,000 --> 00:00:01,500
Time:1000 Latitude:22.5753 Longitude:113.9450 AbsoluteAltitude:54.81 RelativeAltitude:2.20 GimbalPitch:-24.00 GimbalRoll:0.50 GimbalYaw:-80.00
"""

SRT_NO_RELATIVE = """1
00:00:00,000 --> 00:00:00,500
Time:0 Latitude:22.5752 Longitude:113.9449 AbsoluteAltitude:53.81

2
00:00:01,000 --> 00:00:01,500
Time:1000 Latitude:22.5753 Longitude:113.9450 AbsoluteAltitude:54.81
"""

DJI_CSV = """dji_telemetry.time,dji_telemetry.latitude,dji_telemetry.longitude,dji_telemetry.absolute_altitude,dji_telemetry.relative_altitude,dji_telemetry.gimbal_pitch,dji_telemetry.rtk_quality,dji_telemetry.gps_number
1758528000000,22.5752,113.9449,53.81,1.20,-25.00,4,14
1758528001000,22.5753,113.9450,54.81,2.20,-24.00,4,13
"""

DJI_CSV_ASL_ONLY = """dji_telemetry.time,dji_telemetry.latitude,dji_telemetry.longitude,dji_telemetry.absolute_altitude,dji_telemetry.rtk_fix
2026-09-22 08:00:00,22.5752,113.9449,53.81,RTK Fixed
2026-09-22 08:00:02,22.5753,113.9450,54.81,Single
"""

CONTRACT_CSV = """t_sec,latitude_deg,longitude_deg,altitude_m,horizontal_std_m,vertical_std_m
0.0,47.3769,8.5415,488.2,0.4,1.0
1.0,47.3770,8.5416,489.2,0.4,1.0
"""


def metadata(**updates):
    value = dict(schema_version=1, time_reference="video", time_offset_s=0.0,
                 altitude_datum="ellipsoidal", position_reference="camera_center",
                 single_pass=True, video_duration_s=30.0)
    value.update(updates)
    return value


class SurveyInputsTests(unittest.TestCase):
    """Parsers, the uncertainty model, optional sensors, and the metadata merge."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.io = importlib.import_module("scripts.survey_inputs")
        except ModuleNotFoundError as exc:
            if exc.name != "scripts.survey_inputs":
                raise
            raise unittest.SkipTest("scripts.survey_inputs has not been implemented")

    def setUp(self):
        keep = tempfile.TemporaryDirectory()
        self.addCleanup(keep.cleanup)
        self.root = Path(keep.name)

    def file(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def draft(self, times=(0.0, 1.0, 2.0)):
        """Positional rows with no uncertainty at all: what a bare track yields."""
        return [dict(zip(POSITIONAL, (t, 47.3769 + 1e-5 * i, 8.5415 + 1e-5 * i, 488.0 + i)))
                for i, t in enumerate(times)]

    def normalise(self, rows, meta=None):
        """Feed rows through the real survey_georef reader to prove contract fit."""
        path = self.root / "contract_under_test.csv"
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)
        path.write_text(stream.getvalue(), encoding="utf-8")
        return georef.normalize_telemetry(path, metadata() if meta is None else meta)

    # --------------------------------------------------------------- format sniffing

    def test_sniff_identifies_each_source_by_content_not_extension(self):
        cases = [("a.bin", GPX_FLIGHT, "gpx"), ("b.txt", SRT_FLIGHT, "srt"),
                 ("c.log", DJI_CSV, "dji_csv"), ("d.dat", CONTRACT_CSV, "telemetry_csv"),
                 ("e.txt", "just prose, no telemetry at all\n", "unknown")]
        for name, text, expected in cases:
            detected = self.io.sniff_format(self.file(name, text))
            self.assertEqual(detected.kind, expected, name)
            self.assertEqual(detected.via, "content", name)

    def test_from_flight_log_dispatches_and_notes_the_extension(self):
        path = self.file("flight_0001.srt.txt", SRT_FLIGHT)
        detected = self.io.sniff_format(path)
        self.assertEqual(detected.kind, "srt")
        self.assertEqual(detected.extension, ".txt")
        self.assertEqual(detected.extension_agrees, False)
        # A subtitle carries no precision evidence, so the caller must declare some.
        self.assertEqual(len(self.io.from_flight_log(
            path, altitude_source="relative", horizontal_std_m=2.0, vertical_std_m=3.0)), 2)
        self.assertTrue(self.io.sniff_format(self.file("t.gpx", GPX_FLIGHT)).extension_agrees)
        self.assertEqual(len(self.io.from_flight_log(self.file("t.gpx", GPX_FLIGHT))), 2)
        self.assertEqual(len(self.io.from_flight_log(self.file("d.csv", DJI_CSV))), 2)
        self.assertEqual(len(self.io.from_flight_log(self.file("c.csv", CONTRACT_CSV))), 2)

    def test_from_flight_log_refuses_unknown_content(self):
        with self.assertRaisesRegex(ValueError, "unrecognised"):
            self.io.from_flight_log(self.file("x.txt", "hello there\n"))

    def test_from_flight_log_rejects_garbage_rows_naming_the_line(self):
        broken = CONTRACT_CSV.replace("1.0,47.3770", "1.0,oops")
        with self.assertRaisesRegex(ValueError, "line 3"):
            self.io.from_flight_log(self.file("bad.csv", broken), on_malformed="reject")

    # ------------------------------------------------------------------------- GPX

    def test_parse_gpx_yields_contract_rows_and_records_provenance(self):
        path = self.file("track.gpx", GPX_FLIGHT)
        rows = self.io.parse_gpx(path)
        self.assertEqual([set(r) for r in rows], [set(FIELDS)] * 2)
        self.assertEqual([r["t_sec"] for r in rows], [0.0, 2.0])
        self.assertAlmostEqual(rows[0]["latitude_deg"], 47.3769)
        self.assertAlmostEqual(rows[0]["altitude_m"], 488.2)
        # Documented dilution model: HDOP x 2.0 m UERE, vertical x 1.5.
        self.assertAlmostEqual(rows[0]["horizontal_std_m"], 2.8)
        self.assertAlmostEqual(rows[1]["horizontal_std_m"], 3.2)
        self.assertAlmostEqual(rows[0]["vertical_std_m"], 4.2)
        info = self.io.source_info(path)
        self.assertEqual(info["time_reference"], "gnss_wall_clock")
        self.assertEqual(info["altitude_datum"], "unspecified_by_gpx")
        self.assertEqual(info["first_time_s"], 0.0)
        self.assertTrue(any("datum" in w for w in info["warnings"]))
        self.assertEqual(self.normalise(rows)["samples"][1]["t_sec"], 2.0)

    def test_parse_gpx_needs_evidence_of_precision_or_an_explicit_declaration(self):
        text = GPX_FLIGHT.replace("<ubx:hdop>1.4</ubx:hdop>", "").replace(
            "<ubx:hdop>1.6</ubx:hdop>", "")
        path = self.file("nofix.gpx", text)
        with self.assertRaises(ValueError) as caught:
            self.io.parse_gpx(path)
        message = str(caught.exception)
        self.assertIn("nofix.gpx", message)
        self.assertIn("line", message)
        self.assertIn("survey_inputs.attach_quality", message)
        self.assertIn("uncertainty", message)
        with self.assertRaisesRegex(ValueError, "vertical_std_m"):
            self.io.parse_gpx(path, horizontal_std_m=2.0)
        rows = self.io.parse_gpx(path, horizontal_std_m=2.0, vertical_std_m=4.0)
        self.assertEqual([r["horizontal_std_m"] for r in rows], [2.0, 2.0])
        self.assertEqual(self.io.source_info(path)["uncertainty_source"], "declared_by_caller")
        draft = self.io.parse_gpx(path, strict=False)
        self.assertEqual([set(r) for r in draft], [set(POSITIONAL)] * 2)
        filled = self.io.attach_quality(draft, hdop_by_time=[(0.0, 1.4), (2.0, 1.6)])
        self.assertEqual([r["horizontal_std_m"] for r in filled.rows], [2.8, 3.2])

    def test_parse_gpx_rejects_malformed_values_naming_line_and_attribute(self):
        bad = GPX_FLIGHT.replace('lat="47.3770"', 'lat="47.37xx0"')
        with self.assertRaises(ValueError) as caught:
            self.io.parse_gpx(self.file("bad.gpx", bad))
        message = str(caught.exception)
        self.assertIn("bad.gpx", message)
        self.assertIn("lat", message)
        self.assertIn("line 8", message)

    def test_parse_gpx_rejects_non_gpx_out_of_range_and_non_increasing(self):
        with self.assertRaisesRegex(ValueError, "not a GPX"):
            self.io.parse_gpx(self.file("x.csv", CONTRACT_CSV))
        dup = GPX_FLIGHT.replace("08:00:02Z", "08:00:00Z")
        with self.assertRaisesRegex(ValueError, "increasing"):
            self.io.parse_gpx(self.file("dup.gpx", dup))
        wide = GPX_FLIGHT.replace('lat="47.3769"', 'lat="94.5"')
        with self.assertRaisesRegex(ValueError, "latitude"):
            self.io.parse_gpx(self.file("wide.gpx", wide))
        wallless = GPX_FLIGHT.replace("<time>2026-09-22T08:00:00Z</time>", "")
        with self.assertRaisesRegex(ValueError, "time"):
            self.io.parse_gpx(self.file("nowall.gpx", wallless))

    def test_parse_gpx_refuses_ambiguous_dilution_provenance(self):
        both = GPX_FLIGHT.replace("<ubx:hdop>1.4</ubx:hdop>", "<hdop>1.4</hdop><hdop>2.0</hdop>")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.io.parse_gpx(self.file("mix.gpx", both))

    def test_parse_gpx_offsets_relative_times_by_a_declared_recording_origin(self):
        origin = datetime.datetime(2026, 9, 22, 8, 0, 1, tzinfo=datetime.timezone.utc).timestamp()
        rows = self.io.parse_gpx(self.file("o.gpx", GPX_FLIGHT), time_origin_epoch=origin)
        self.assertEqual([r["t_sec"] for r in rows], [-1.0, 1.0])
        self.assertEqual(self.io.source_info(self.root / "o.gpx")["time_reference"], "video")

    # ------------------------------------------------------------------ telemetry CSV

    def test_parse_telemetry_csv_accepts_the_exact_contract_fields(self):
        rows = self.io.parse_telemetry_csv(self.file("tel.csv", CONTRACT_CSV))
        self.assertEqual([set(r) for r in rows], [set(FIELDS)] * 2)
        self.assertEqual(rows[1]["vertical_std_m"], 1.0)
        self.assertEqual(self.normalise(rows)["schema_version"], 1)

    def test_parse_telemetry_csv_rejects_missing_and_extra_columns_by_name(self):
        with self.assertRaisesRegex(ValueError, "vertical_std_m"):
            self.io.parse_telemetry_csv(self.file(
                "m.csv", "t_sec,latitude_deg,longitude_deg,altitude_m,horizontal_std_m\n"
                         "0,47.3,8.5,488,0.4\n"))
        with self.assertRaisesRegex(ValueError, "unexpected extra"):
            self.io.parse_telemetry_csv(self.file(
                "e.csv", ",".join(FIELDS) + ",gimbal_pitch\n0,47.3,8.5,488,0.4,1.0,0\n"))

    def test_parse_telemetry_csv_names_the_offending_row_or_keeps_counting(self):
        text = CONTRACT_CSV + "2.0,47.3771,8.5417,not_a_number,0.4,1.0\n"
        with self.assertRaises(ValueError) as caught:
            self.io.parse_telemetry_csv(self.file("r.csv", text))
        self.assertIn("r.csv", str(caught.exception))
        self.assertIn("line 4", str(caught.exception))
        dropped = self.io.parse_telemetry_csv(self.file("k.csv", text), on_malformed="drop",
                                             max_dropped_fraction=0.5)
        self.assertEqual(len(dropped), 2)
        self.assertEqual(self.io.source_info(self.root / "k.csv")["rejected"], 1)
        doubled = text + "3.0,47.3772,8.5418,bad,0.4,1.0\n"
        with self.assertRaisesRegex(ValueError, "dropped"):
            self.io.parse_telemetry_csv(self.file("a.csv", doubled), on_malformed="drop",
                                        max_dropped_fraction=0.1)

    def test_no_parser_ever_emits_nan_or_none(self):
        for name, text, kwargs in (("a.csv", CONTRACT_CSV, {}),
                                   ("b.srt", SRT_FLIGHT, {"altitude_source": "relative",
                                                          "horizontal_std_m": 2.0,
                                                          "vertical_std_m": 3.0}),
                                   ("c.csv", DJI_CSV, {}), ("d.gpx", GPX_FLIGHT, {})):
            for row in self.io.from_flight_log(self.file(name, text), **kwargs):
                self.assertEqual(len(row), 6, row)
                self.assertEqual(tuple(row), FIELDS, row)
                self.assertTrue(all(v is not None and
                                    not (isinstance(v, float) and math.isnan(v))
                                    for v in row.values()), row)
                self.assertTrue(all(math.isfinite(float(v)) for v in row.values()), row)

    # -------------------------------------------------------------------------- SRT

    def test_parse_srt_reads_dji_subtitle_telemetry(self):
        path = self.file("FLIGHT_0001.srt", SRT_FLIGHT)
        rows = self.io.parse_srt(path, altitude_source="relative",
                                 horizontal_std_m=2.0, vertical_std_m=3.0)
        self.assertEqual([set(r) for r in rows], [set(FIELDS)] * 2)
        self.assertEqual([r["t_sec"] for r in rows], [0.0, 1.0])
        self.assertAlmostEqual(rows[0]["latitude_deg"], 22.5752)
        self.assertAlmostEqual(rows[1]["altitude_m"], 2.20)
        info = self.io.source_info(path)
        self.assertEqual(info["time_reference"], "video")
        self.assertEqual(info["altitude_datum"], "relative_to_takeoff")
        self.assertEqual(info["gimbal"][0], {"pitch_deg": -25.0, "roll_deg": 0.0,
                                             "yaw_deg": -82.4})
        self.assertEqual(self.normalise(rows)["coordinate_frame"]["type"], "ENU")

    def test_parse_srt_names_the_block_and_field_when_malformed(self):
        broken = SRT_FLIGHT.replace("Longitude:113.9450", "Longitude:11x3.9450")
        with self.assertRaises(ValueError) as caught:
            self.io.parse_srt(self.file("bad.srt", broken), altitude_source="relative")
        message = str(caught.exception)
        self.assertIn("bad.srt", message)
        self.assertIn("block 2", message)
        self.assertIn("Longitude", message)

    def test_parse_srt_absolute_altitude_is_mean_sea_level_not_ellipsoidal(self):
        path = self.file("abs.srt", SRT_NO_RELATIVE)
        rows = self.io.parse_srt(path, altitude_source="absolute",
                                 horizontal_std_m=2.0, vertical_std_m=3.0)
        self.assertAlmostEqual(rows[0]["altitude_m"], 53.81)
        info = self.io.source_info(path)
        self.assertEqual(info["altitude_datum"], "mean_sea_level")
        self.assertTrue(any("geoid" in w for w in info["warnings"]))

    def test_parse_srt_requires_a_declared_altitude_source(self):
        with self.assertRaises(ValueError) as caught:
            self.io.parse_srt(self.file("x.srt", SRT_FLIGHT))
        message = str(caught.exception)
        for token in ("altitude_source", "relative", "absolute"):
            self.assertIn(token, message)
        with self.assertRaisesRegex(ValueError, "altitude_source"):
            self.io.parse_srt(self.file("y.srt", SRT_FLIGHT), altitude_source="truthy")

    def test_parse_srt_rejects_plain_subtitles_and_missing_fields(self):
        with self.assertRaisesRegex(ValueError, "not telemetry"):
            self.io.parse_srt(self.file("cap.srt",
                                        "1\n00:00:01,000 --> 00:00:02,000\nHello.\n"),
                              altitude_source="relative")
        sparse = SRT_FLIGHT.replace(" Longitude:113.9449", "").replace(" Longitude:113.9450", "")
        with self.assertRaisesRegex(ValueError, "Longitude"):
            self.io.parse_srt(self.file("sparse.srt", sparse), altitude_source="relative")

    def test_parse_srt_requires_strictly_increasing_block_times(self):
        with self.assertRaisesRegex(ValueError, "increasing"):
            self.io.parse_srt(self.file("dup.srt", SRT_FLIGHT.replace("Time:1000", "Time:0")),
                              altitude_source="relative")

    # --------------------------------------------------------------------- DJI CSV

    def test_parse_dji_csv_maps_prefixed_and_bare_columns(self):
        path = self.file("d1.csv", DJI_CSV)
        rows = self.io.parse_dji_csv(path)
        bare = self.io.parse_dji_csv(self.file("d2.csv", DJI_CSV.replace("dji_telemetry.", "")))
        self.assertEqual(rows, bare)
        self.assertEqual([set(r) for r in rows], [set(FIELDS)] * 2)
        self.assertEqual([r["t_sec"] for r in rows], [0.0, 1.0])    # ms since epoch, made relative
        self.assertEqual(rows[0]["horizontal_std_m"], 0.03)         # RTK fixed, documented floor
        self.assertEqual(rows[0]["vertical_std_m"], 0.05)
        self.assertEqual(rows[0]["altitude_m"], 1.20)               # relative preferred
        info = self.io.source_info(path)
        self.assertEqual(info["altitude_datum"], "relative_to_takeoff")
        self.assertEqual(info["time_reference"], "gnss_wall_clock")
        self.assertEqual(info["gimbal"][0]["pitch_deg"], -25.0)
        self.assertEqual(info["satellites"], [14, 13])

    def test_parse_dji_csv_reads_rtk_tokens_and_time_strings_and_asl_height(self):
        path = self.file("asl.csv", DJI_CSV_ASL_ONLY)
        rows = self.io.parse_dji_csv(path)
        self.assertEqual([r["t_sec"] for r in rows], [0.0, 2.0])
        self.assertEqual(rows[0]["horizontal_std_m"], 0.03)         # RTK Fixed
        self.assertEqual(rows[1]["horizontal_std_m"], 2.0)          # Single
        self.assertEqual(rows[1]["vertical_std_m"], 4.0)
        info = self.io.source_info(path)
        self.assertEqual(info["altitude_datum"], "mean_sea_level")
        self.assertAlmostEqual(info["wall_clock_s"][1] - info["wall_clock_s"][0], 2.0)
        self.assertAlmostEqual(
            info["wall_clock_s"][0],
            datetime.datetime(2026, 9, 22, 8, 0, 0, tzinfo=datetime.timezone.utc).timestamp(),
            delta=1e-6)
        self.assertTrue(any("geoid" in w for w in info["warnings"]))

    def test_parse_dji_csv_needs_position_and_a_height_column(self):
        with self.assertRaisesRegex(ValueError, "latitude"):
            self.io.parse_dji_csv(self.file("none.csv", "dji_telemetry.time\n1\n"))
        with self.assertRaisesRegex(ValueError, "altitude"):
            self.io.parse_dji_csv(self.file(
                "noalt.csv", "dji_telemetry.latitude,dji_telemetry.longitude\n1,2\n"))

    # ---------------------------------------------------------------- attach_quality

    def test_fix_quality_table_and_token_parsing(self):
        self.assertEqual(self.io.FIX_QUALITY_STD_M["no_fix"], None)
        self.assertEqual(self.io.FIX_QUALITY_STD_M["unknown"], (5.0, 10.0))
        self.assertEqual(self.io.FIX_QUALITY_STD_M["fixed"], (0.03, 0.05))
        self.assertEqual(self.io.DOP_TO_METRE_UERE, 2.0)
        self.assertEqual(self.io.VERTICAL_DOP_RATIO, 1.5)
        for token, expected in ((4, "fixed"), ("4", "fixed"), ("RTK Fixed", "fixed"),
                                ("rtk_float", "float"), (3, "float"), ("Float", "float"),
                                (2, "psdiff"), ("waas", "psdiff"), (1, "single"),
                                ("3D", "single"), (0, "no_fix"), ("no solution", "no_fix"),
                                ("none", "no_fix")):
            self.assertEqual(self.io.parse_fix_quality(token), expected, token)
        for token in ("wat", "-1", "9", None, "3D fix"):
            with self.assertRaisesRegex(ValueError, "unknown_fix_token"):
                self.io.parse_fix_quality(token)

    def test_attach_quality_maps_fix_quality_conservatively(self):
        result = self.io.attach_quality(self.draft(), fix_quality_by_time=[(0.0, "no_fix"),
                                                                          (1.0, "single"),
                                                                          (2.0, "fixed")])
        self.assertEqual([r["t_sec"] for r in result.rows], [1.0, 2.0])
        self.assertEqual([set(r) for r in result.rows], [set(FIELDS)] * 2)
        self.assertEqual(result.rows[-1]["horizontal_std_m"], 0.03)
        self.assertEqual(result.rows[-1]["vertical_std_m"], 0.05)
        self.assertEqual(result.dropped[0]["reason"], "unusable_fix_quality")
        self.assertEqual(result.fix_quality_basis, "per_sample_rtk_or_ppk")
        self.assertEqual(result.uncertainty_source, "fix_quality")
        self.assertTrue(any("correlated GNSS bias" in w for w in result.warnings))

    def test_attach_quality_takes_the_worse_side_of_a_fix_transition(self):
        result = self.io.attach_quality(self.draft(),
                                        fix_quality_by_time=[(0.0, "fixed"), (2.0, "single")])
        self.assertEqual(result.rows[0]["horizontal_std_m"], 0.03)      # exact sample
        self.assertEqual(result.rows[1]["horizontal_std_m"], 2.0)       # bracket -> worse
        self.assertEqual(result.rows[2]["horizontal_std_m"], 2.0)
        self.assertTrue(any("transition" in w for w in result.warnings))

    def test_attach_quality_degrades_a_gapped_fix_series_to_unknown(self):
        result = self.io.attach_quality(self.draft(), fix_quality_by_time=[(0.0, "fixed"),
                                                                          (20.0, "fixed")])
        self.assertEqual(result.rows[0]["horizontal_std_m"], 0.03)      # exact sample
        self.assertEqual(result.rows[1]["horizontal_std_m"], 5.0)       # 2 and 20 are a gap
        self.assertEqual(result.rows[1]["vertical_std_m"], 10.0)
        self.assertEqual(result.rows[2]["horizontal_std_m"], 5.0)
        self.assertEqual(result.fix_quality_basis, "unknown")
        self.assertTrue(any("no fix-quality sample" in w for w in result.warnings))

    def test_attach_quality_dilution_model_and_degraded_geometry_inflation(self):
        rows = self.draft()
        result = self.io.attach_quality(rows, hdop_by_time=[(0.0, 1.4), (2.0, 1.6)])
        self.assertEqual([r["horizontal_std_m"] for r in result.rows], [2.8, 3.2, 3.2])
        self.assertEqual(result.rows[0]["vertical_std_m"], 4.2)
        self.assertEqual(result.uncertainty_source, "hdop_model")
        self.assertEqual(result.fix_quality_basis, "unknown")
        self.assertTrue(any("satellite count" in w for w in result.warnings))
        rtk = self.io.attach_quality(rows, fix_quality_by_time=[(0.0, "fixed"), (2.0, "fixed")],
                                     hdop_by_time=[(0.0, 5.0), (2.0, 5.0)])
        self.assertAlmostEqual(rtk.rows[0]["horizontal_std_m"], 0.075)   # 0.03 x (5/2)
        self.assertTrue(any("cannot be confirmed" in w for w in rtk.warnings))

    def test_attach_quality_refuses_to_invent_precision(self):
        with self.assertRaisesRegex(ValueError, "attach_quality") as caught:
            self.io.attach_quality(self.draft())
        self.assertIn("uncertainty", str(caught.exception))

    def test_attach_quality_keeps_the_larger_of_declared_and_modelled(self):
        declared = [dict(row, horizontal_std_m=0.02, vertical_std_m=0.03)
                    for row in self.draft()]
        result = self.io.attach_quality(declared, fix_quality_by_time=[(0.0, "single"),
                                                                      (2.0, "single")])
        self.assertEqual(result.rows[0]["horizontal_std_m"], 2.0)
        self.assertEqual(result.rows[0]["vertical_std_m"], 4.0)
        self.assertTrue(any("larger of" in w for w in result.warnings))
        trusted = self.io.attach_quality(declared)
        self.assertEqual(trusted.rows[0]["horizontal_std_m"], 0.02)
        self.assertEqual(trusted.rows[0]["vertical_std_m"], 0.03)
        self.assertEqual(trusted.uncertainty_source, "declared_by_caller")
        self.assertFalse(any("larger of" in w for w in trusted.warnings))

    def test_attach_quality_validates_its_inputs(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.io.attach_quality([])
        with self.assertRaisesRegex(ValueError, "longitude"):
            self.io.attach_quality([dict(t_sec=0.0, latitude_deg=1.0, altitude_m=2.0)])
        with self.assertRaisesRegex(ValueError, "increasing"):
            self.io.attach_quality(self.draft(), fix_quality_by_time=[(2.0, "fixed"),
                                                                      (0.0, "single")])
        with self.assertRaisesRegex(ValueError, "hdop"):
            self.io.attach_quality(self.draft(), hdop_by_time=[(0.0, 0.0), (1.0, 1.4)])

    def test_attach_quality_result_rows_feed_survey_georef(self):
        result = self.io.attach_quality(self.draft(), hdop_by_time=[(0.0, 1.4), (2.0, 1.6)])
        self.assertEqual(len(self.normalise(result.rows)["samples"]), 3)

    # ------------------------------------------------------------- IMU and barometer

    def test_read_imu_returns_si_units_and_reports_rejections(self):
        path = self.file("imu.csv", "t_ms,ax,ay,az,gx,gy,gz\n"
                                    "0,0.1,0.2,9.81,10,-5,0\n"
                                    "20,0.12,0.21,9.80,11,-5,0.2\n"
                                    "bad,row,row,row,row,row,row\n")
        series = self.io.read_imu(path)
        self.assertEqual(series.units, {"ax": "m/s^2", "ay": "m/s^2", "az": "m/s^2",
                                        "gx": "rad/s", "gy": "rad/s", "gz": "rad/s"})
        self.assertEqual(set(series.units), set(series.columns))
        self.assertEqual(series.columns, ("ax", "ay", "az", "gx", "gy", "gz"))
        self.assertEqual(series.n_rows, 2)
        self.assertEqual(series.rejected, 1)
        self.assertEqual(series.rejected_detail[0]["line"], 4)
        np.testing.assert_allclose(series.values[0, 2], 9.81)
        np.testing.assert_allclose(series.values[0, 3], math.radians(10.0))
        self.assertAlmostEqual(series.dt_median_s, 0.02)
        self.assertEqual(series.source_time_unit, "ms")
        self.assertEqual(series.source_units, {"acceleration": "m/s^2",
                                               "angular_velocity": "deg/s"})
        claims = series.summarise()
        self.assertEqual(claims["can"], [])
        self.assertTrue(any("dead reckoning" in c for c in claims["cannot"]))
        self.assertFalse(claims["provides_absolute_position"])
        self.assertEqual(claims["units"], series.units)
        self.assertIn("imu.csv", claims["source"])
        self.assertEqual(claims["rejected"], 1)

    def test_read_imu_alternative_column_names_and_declared_gyro_unit(self):
        path = self.file("imu2.csv",
                         "timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n"
                         "0,0,0,9.81,0.1,0,0\n1,0,0,9.81,0.1,0,0\n")
        series = self.io.read_imu(path, gyro_unit="rad/s")
        np.testing.assert_allclose(series.values[1, 3], 0.1)
        self.assertEqual(series.source_time_unit, "s")
        self.assertAlmostEqual(series.dt_median_s, 1.0)
        with self.assertRaisesRegex(ValueError, "columns"):
            self.io.read_imu(self.file("imu3.csv", "a,b\n1,2\n"))
        with self.assertRaisesRegex(ValueError, "time column"):
            self.io.read_imu(self.file("imu4.csv", "ax,ay,az,gx,gy,gz\n0,0,9.8,0,0,0\n"))

    def test_read_barometer_separates_pressure_from_height(self):
        path = self.file("baro.csv", "t_sec,pressure_hpa\n0,1013.25\n0.5,1013.10\n1.0,1012.95\n")
        series = self.io.read_barometer(path)
        self.assertEqual(series.columns, ("pressure_hpa",))
        self.assertEqual(series.units, {"pressure_hpa": "hPa"})
        self.assertEqual(series.n_rows, 3)
        claims = series.summarise()
        self.assertTrue(any("not converted to height" in c for c in claims["cannot"]))
        self.assertTrue(any("not a substitute for GNSS" in c for c in claims["cannot"]))
        self.assertEqual(claims["can"], [])
        with self.assertRaisesRegex(ValueError, "pressure or altitude"):
            self.io.read_barometer(self.file("alt.csv", "t_sec,altitude_m\n0,10\n1,11\n"))
        both = self.io.read_barometer(self.file(
            "mixed.csv", "t_sec,pressure_hpa,baro_altitude_m\n0,1013.2,7.0\n1,1013.1,8.0\n"))
        self.assertEqual(both.columns, ("pressure_hpa", "baro_altitude_m"))
        self.assertTrue(any("never combined" in c for c in both.summarise()["cannot"]))

    def test_sensor_readers_require_monotone_finite_time(self):
        for name, body, pattern in (
                ("back.csv", "1,0,0,9.8,0,0,0\n0,0,0,9.8,0,0,0\n", "monotonic"),
                ("same.csv", "1,0,0,9.8,0,0,0\n1,0,0,9.8,0,0,0\n", "monotonic"),
                ("nan.csv", "0,nan,0,9.8,0,0,0\n1,0,0,9.8,0,0,0\n", "finite"),
                ("empty.csv", "0,,0,9.8,0,0,0\n1,0,0,9.8,0,0,0\n", "finite")):
            with self.assertRaisesRegex(ValueError, pattern):
                self.io.read_imu(self.file(name, "t_sec,ax,ay,az,gx,gy,gz\n" + body))

    # ---------------------------------------------------------------------- metadata

    def test_merge_metadata_is_georef_compatible_only_when_honestly_declared(self):
        bare = self.io.merge_metadata(overrides=dict(video_duration_s=30.0))
        self.assertEqual({b.code for b in bare["blockers"]},
                         {"altitude_datum_undeclared", "position_reference_gnss_antenna"})
        self.assertEqual(bare["schema_version"], 1)
        self.assertEqual(bare["time_reference"], "video")
        self.assertEqual(bare["time_offset_s"], 0.0)
        self.assertTrue(bare["single_pass"])
        declared = self.io.merge_metadata(overrides=dict(
            video_duration_s=30.0, altitude_datum="ellipsoidal",
            altitude_datum_basis="receiver logged ellipsoidal height; no geoid applied",
            position_reference="camera_center", antenna_offset_applied=True,
            lever_arm_uncertainty_m=0.005))
        self.assertEqual(declared["blockers"], [])
        self.assertEqual(georef.validate_metadata(declared), declared)
        self.assertEqual(self.io.merge_metadata()["blockers"][0].code,
                         "missing_video_duration_s")
        self.assertEqual(self.io.merge_metadata()["video_duration_s"], 0.0)

    def test_merge_metadata_refuses_a_multi_pass_flight(self):
        meta = self.io.merge_metadata(overrides=dict(video_duration_s=30.0, single_pass=False))
        self.assertEqual(meta["blockers"][0].code, "not_single_pass")
        with self.assertRaisesRegex(ValueError, "must declare"):
            georef.validate_metadata(meta)

    def test_merge_metadata_reads_duration_from_the_container_when_probeable(self):
        try:
            import cv2
        except ImportError:  # pragma: no cover
            self.skipTest("cv2 unavailable")
        path = self.root / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (32, 32))
        if not writer.isOpened():  # pragma: no cover
            writer.release()
            self.skipTest("no video encoder available on this machine")
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        for _ in range(50):
            writer.write(frame)
        writer.release()
        capture = cv2.VideoCapture(str(path))
        frames, fps = capture.get(cv2.CAP_PROP_FRAME_COUNT), capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if not frames or not fps:  # pragma: no cover
            self.skipTest("this OpenCV build cannot probe a written container")
        meta = self.io.merge_metadata(video_path=path)
        self.assertAlmostEqual(meta["video_duration_s"], frames / fps, places=6)
        self.assertEqual(meta["video_duration_source"], "container_frames_and_fps")
        self.assertNotIn("missing_video_duration_s", {b.code for b in meta["blockers"]})
        self.assertEqual({b.code for b in meta["blockers"]},
                         {"altitude_datum_undeclared", "position_reference_gnss_antenna"})
        self.assertEqual(meta["video_probe"]["native_frames"], int(frames))
        over = self.io.merge_metadata(video_path=path, overrides=dict(video_duration_s=99.0))
        self.assertEqual(over["video_duration_s"], 99.0)
        self.assertEqual(over["video_duration_source"], "caller_override")
        self.assertEqual(self.io.merge_metadata(video_path=self.root / "missing.mp4")
                         ["blockers"][0].code, "missing_video_duration_s")

    def test_merge_metadata_blocks_agl_and_antenna_and_non_video_clocks(self):
        meta = self.io.merge_metadata(overrides=dict(
            video_duration_s=30.0, altitude_datum="AGL-relative",
            position_reference="gnss_antenna", time_reference="gnss_wall_clock"))
        self.assertEqual({b.code for b in meta["blockers"]},
                         {"altitude_datum_agl_relative", "position_reference_gnss_antenna",
                          "time_reference_not_video"})
        self.assertTrue(all(b.severity == "blocker" and b.unblocking_evidence and
                            len(b.message) > 30 for b in meta["blockers"]))
        with self.assertRaisesRegex(ValueError, "time_reference must declare"):
            georef.validate_metadata(meta)
        fixed = dict(meta, position_reference="camera_center", altitude_datum="ellipsoidal",
                     time_reference="video")
        self.assertEqual(georef.validate_metadata(fixed), fixed)

    def test_merge_metadata_treats_non_ellipsoidal_datums_as_blockers(self):
        sea = self.io.merge_metadata(overrides=dict(video_duration_s=30.0,
                                                   altitude_datum="mean_sea_level"))
        self.assertEqual(sea["altitude_datum"], "mean_sea_level")
        self.assertEqual(sea["blockers"][0].code, "altitude_datum_not_ellipsoidal")
        self.assertTrue(any("EGM" in b.unblocking_evidence
                            for b in sea["blockers"] if b.code == "altitude_datum_not_ellipsoidal"))
        ell = self.io.merge_metadata(overrides=dict(video_duration_s=30.0,
                                                   altitude_datum="ellipsoidal"))
        self.assertEqual([b.code for b in ell["blockers"] if b.field == "altitude_datum"], [])
        self.assertEqual(ell["altitude_datum_source"], "asserted_by_caller")
        self.assertEqual(ell["altitude_datum_basis"], "")
        self.assertTrue(any("basis" in w for w in ell["warnings"]))

    def test_merge_metadata_calibration_never_claims_camera_center(self):
        calib = self.file("c.json", json.dumps({
            "camera_model": "OPENCV", "width": 1920, "height": 1080, "fx": 1600.0,
            "fy": 1600.0, "cx": 960.0, "cy": 540.0,
            "antenna_offset_m": [0.0, 0.05, 0.03]}))
        meta = self.io.merge_metadata(calibration_json=calib, overrides=dict(video_duration_s=30.0))
        self.assertEqual(meta["position_reference"], "gnss_antenna")
        self.assertEqual(meta["intrinsics"]["fx"], 1600.0)
        self.assertEqual(meta["intrinsics"]["camera_model"], "OPENCV")
        self.assertFalse(meta["antenna_offset_applied"])
        self.assertIn("position_reference_gnss_antenna", {b.code for b in meta["blockers"]})
        self.assertEqual(meta["antenna_offset_m"], [0.0, 0.05, 0.03])
        applied = self.io.merge_metadata(calibration_json=calib, overrides=dict(
            video_duration_s=30.0, altitude_datum="ellipsoidal",
            altitude_datum_basis="receiver logged ellipsoidal height",
            position_reference="camera_center",
            antenna_offset_applied=True, lever_arm_uncertainty_m=0.01))
        self.assertEqual(applied["blockers"], [])
        self.assertEqual(applied["position_reference"], "camera_center")
        self.assertEqual(applied["lever_arm_uncertainty_m"], 0.01)
        self.assertTrue(applied["antenna_offset_applied"])
        unbacked = self.io.merge_metadata(overrides=dict(video_duration_s=30.0,
                                                        altitude_datum="ellipsoidal",
                                                        position_reference="camera_center"))
        self.assertEqual([b.code for b in unbacked["blockers"]], ["camera_center_unasserted"])
        with self.assertRaisesRegex(ValueError, "intrinsics"):
            self.io.merge_metadata(calibration_json=self.file(
                "bad.json", json.dumps({"camera_model": "OPENCV", "width": 1920})))

    def test_merge_metadata_rejects_unknown_or_overreaching_declarations(self):
        with self.assertRaisesRegex(ValueError, "no_video"):
            self.io.merge_metadata(overrides=dict(video_duration_s=30.0,
                                                  time_reference="no_video"))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.io.merge_metadata(overrides=dict(video_duration_s=30.0, single_pass="stookey"))
        with self.assertRaisesRegex(ValueError, "unknown override"):
            self.io.merge_metadata(overrides=dict(video_duration_s=30.0, mood="serious"))
        with self.assertRaisesRegex(ValueError, "overreach"):
            self.io.merge_metadata(overrides=dict(video_duration_s=30.0, measured_accuracy_m=0.2))

    def test_merge_metadata_uses_source_provenance_and_offset_bounds(self):
        path = self.file("w.gpx", GPX_FLIGHT)
        self.io.parse_gpx(path, horizontal_std_m=2.0, vertical_std_m=4.0)
        info = self.io.source_info(path)
        meta = self.io.merge_metadata(source_info=info, overrides=dict(video_duration_s=30.0))
        self.assertEqual(meta["time_reference"], "video")
        self.assertEqual(meta["time_offset_s"], 0.0)
        self.assertEqual({b.code for b in meta["blockers"]},
                         {"telemetry_anchor_assumed", "altitude_datum_unspecified",
                          "position_reference_gnss_antenna"})
        self.assertEqual(meta["unresolved"], [b.code for b in meta["blockers"]])
        self.assertEqual(meta["video_duration_source"], "caller_override")
        self.assertEqual(meta["uncertainty_declared_by_caller"], [2.0, 4.0])
        shifted = self.io.merge_metadata(source_info=dict(info, first_time_s=1.5),
                                         overrides=dict(video_duration_s=30.0))
        self.assertEqual(shifted["time_offset_s"], 1.5)
        self.assertTrue(any("duration" in w for w in shifted["warnings"]))
        self.assertTrue(any("30.0" in w for w in shifted["warnings"]))
        with self.assertRaisesRegex(ValueError, "together"):
            self.io.merge_metadata(overrides=dict(video_duration_s=30.0, horizontal_std_m=1.0))

    def test_merge_metadata_time_reference_follows_the_rows_not_the_source_clock(self):
        # Rows emitted by the parsers are seconds since the declared origin, so the
        # declaration can honestly be "video"; a caller that keeps an absolute clock
        # must say so, and survey_georef will (rightly) reject it.
        meta = self.io.merge_metadata(overrides=dict(
            video_duration_s=10.0, time_reference="gnss_wall_clock", time_offset_s=-1.0e9))
        self.assertEqual(meta["time_reference"], "gnss_wall_clock")
        self.assertEqual(meta["blockers"][0].code, "time_reference_not_video")
        self.assertIn("offset", meta["blockers"][0].message.lower())


if __name__ == "__main__":
    unittest.main()
