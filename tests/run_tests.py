"""Self-contained verification for the v2 pipeline additions. No framework.

  .venv\\Scripts\\python.exe tests\\run_tests.py
"""
from __future__ import annotations

import contextlib
import functools
import http.server
import importlib
import io
import json
import mmap
import sqlite3
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import poses_lib as pl  # noqa: E402
import capture_diagnostics as cd  # noqa: E402
import run_colmap as rc  # noqa: E402
import robust as _rb  # noqa: E402

_rb.configure_streams()

PASS = 0
FAILS = []


def ok(cond, label, detail=""):
    """Record and continue. Aborting on the first failure hid every check after
    it, so a suite could not say how much of the surface was still healthy."""
    global PASS
    if not cond:
        FAILS.append(label)
        print(f"FAIL {label} {('  ' + str(detail).replace(chr(10), ' | ')[:240]) if detail else ''}")
        return
    PASS += 1
    print(f"pass {label}")


def report_exit(title="checks"):
    print()
    if FAILS:
        print(f"{len(FAILS)} of {PASS + len(FAILS)} {title} FAILED: {FAILS}")
        sys.exit(1)
    print(f"ALL {PASS} {title} PASSED")
    sys.exit(0)


# ---------------------------------------------------------------- quaternions
Rz90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
q_z90 = pl.R_to_quat(Rz90)
ok(np.allclose(pl.quat_to_R(q_z90), Rz90, atol=1e-9), "quat<->R roundtrip")
ok(abs(np.linalg.norm(q_z90) - 1) < 1e-12 and q_z90[3] > 0, "scalar-last normalized w>0")
qs = pl.quat_slerp([0, 0, 0, 1], [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)], 0.5)
ok(abs(pl.quat_slerp(qs, qs, 0.7)[3] - qs[3]) < 1e-12, "slerp identity")
half = pl.quat_slerp([0, 0, 0, 1], [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)], 0.5)
ang = 2 * np.arccos(np.clip(half[3], -1, 1))
ok(abs(ang - np.pi / 4) < 1e-9, f"slerp half-angle ({np.degrees(ang):.3f}deg)")
neg = pl.quat_slerp([0, 0, 0, -1], [0, 0, 0, 1], 0.5)
ok(neg[3] >= 0, "slerp shortest arc sign fix")

# ------------------------------------------------------------------- loading
tmp = Path(tempfile.mkdtemp(prefix="posestest_"))

canon = tmp / "walk.jsonl"
rows = [(0.0, [0, 0, 0]), (1.0, [1, 0, 0]), (2.0, [2, 0.5, 0])]
canon.write_text("\n".join(json.dumps({
    "t": t, "pxyz": p,
    "qxyzw": list(pl.R_to_quat(Rz90))}) for t, p in rows), encoding="utf-8")
s, fmt = pl.load_any(canon)
ok(fmt == "jsonl" and len(s) == 3 and abs(s[1][0] - 1.0) < 1e-12, "jsonl loader")

csv_hdr = tmp / "log.csv"
csv_hdr.write_text("timestamp,qx,qy,qz,qw,tx,ty,tz\n" +
                   "\n".join(f"{t},{pl.R_to_quat(Rz90)[0]},{pl.R_to_quat(Rz90)[1]},"
                             f"{pl.R_to_quat(Rz90)[2]},{pl.R_to_quat(Rz90)[3]},{p[0]},{p[1]},{p[2]}"
                             for t, p in rows), encoding="utf-8")
s2, fmt2 = pl.load_any(csv_hdr)
ok(fmt2 == "csv" and np.allclose(s2[2][1], [2, 0.5, 0]), "csv header loader")
csv_bare = tmp / "log_bare.csv"
q = pl.R_to_quat(Rz90)
csv_bare.write_text("\n".join(f"{t},{q[0]},{q[1]},{q[2]},{q[3]},{p[0]},{p[1]},{p[2]}"
                              for (t, p) in rows), encoding="utf-8")
s2b, _ = pl.load_any(csv_bare)
ok(len(s2b) == 3 and np.allclose(s2b[0][1], [0, 0, 0]), "csv bare loader")

r3d = tmp / "metadata.json"
r3d.write_text(json.dumps({"poses": [[0, 0, 0, 1, i * 1.0, 0, 0] for i in range(30)],
                           "K": [1, 0, 0, 0, 1, 0, 0, 0, 1], "w": 8, "h": 6}), encoding="utf-8")
s3, fmt3 = pl.load_any(r3d)
ok(fmt3 == "record3d" and len(s3) == 30 and s3[10][0] == 10 / 30.0, "record3d loader")

# ------------------------------------------------------------- interpolation
ip = pl.interpolate_pose(s, 1.5)
ok(ip is not None and np.allclose(ip[0], [1.5, 0.25, 0]), "position lerp mid")
ok(pl.interpolate_pose(s, 99.0) is None, "outside coverage -> None")
m = pl.match_to_frames([(t - 0.02, p, pl.R_to_quat(Rz90)) for t, p in rows],
                       [0.0, 1.0, 2.0], max_gap=0.25)
ok(all(x is not None and x["log_dt"] <= 0.05 for x in m), "match_to_frames near hits")
m2 = pl.match_to_frames(rows_as := [(t, p, pl.R_to_quat(Rz90)) for t, p in rows],
                        [10.0], max_gap=0.25)
ok(m2 == [None], "far timestamp -> no prior")

# --------------------------------------------------------------- diagnostics
style = cd.classify(med_rot=8.0, med_trn=0.005, rot_dom_pct=70, weak_pct=10,
                    blur_p={"p25": 60})
ok(style == "rotation_dominant", f"classify rotation_dominant got {style}")
style2 = cd.classify(med_rot=0.4, med_trn=0.08, rot_dom_pct=0, weak_pct=5,
                     blur_p={"p25": 60})
ok(style2 == "translation_sweep", f"classify sweep got {style2}")

# ------------------------------------------- challenge D3: measure the light, then decide
# The trigger has to separate three things the probe cannot confuse. A scene whose
# lighting is steady, a whole-frame gain (auto-exposure riding the sun - which
# flattening provably cannot remove, because it divides by the field and re-anchors on
# the field's own mean, so a uniform gain cancels), and a shadow that moves ACROSS the
# frame (the case it exists for). Same texture, same pan, same resolution throughout:
# only the light differs, so a threshold that could not tell these apart would be
# measuring the scene, not the illumination.
import cv2 as _cv2  # noqa: E402
import survey_photometry as _ph  # noqa: E402
sys.path.insert(0, str(ROOT))            # pipeline.py lives at the repo root
import pipeline  # noqa: E402


def lit_frames(kind, n=26):
    """A panned slice of one fixed random scene, with `kind` applied to the light only.

    The scene is deliberately mid-tone and evenly reflective (140 +/- 22 grey, so no
    block of it sits under the 0.62x-of-lit-level rule on its own): otherwise the
    "steady" control would be a scene whose own albedo is a low-frequency dark region,
    which is the one thing this module cannot distinguish from a shadow.
    """
    rng = np.random.default_rng(11)

    def noise(sigma):
        b = _cv2.GaussianBlur((rng.random((256, 512)) * 255).astype(np.uint8), (0, 0),
                              sigma).astype(np.float64)
        return (b - b.mean()) / b.std()

    scene = np.clip(140 + 22 * noise(1.2) + 16 * noise(1.2), 0, 255).astype(np.uint8)
    out = []
    for i in range(n):
        u = i / (n - 1)
        frame = np.roll(scene, 4 * i, axis=1)[8:248, 8:328].copy()   # a translating camera
        w = frame.shape[1]
        x = np.arange(w, dtype=np.float64)[None, :]
        if kind == "gain":                       # sun behind cloud: uniform, whole frame
            frame = frame * (0.5 + 1.4 * u)
        elif kind == "shadow":                   # a cast shadow edge sweeping the scene
            frame = frame * (0.42 + 0.58 / (1 + np.exp(-(x - (0.25 + 0.6 * u) * w) / (w / 12))))
        elif kind == "darkwall":                 # permanent dark material, never a shadow
            frame = frame * np.where(x < 0.45 * w, 0.42, 1.0)
        out.append(np.clip(frame, 0, 255).astype(np.uint8))
    return out


lit = {k: cd.probe_illumination(lit_frames(k)) for k in ("steady", "gain", "shadow", "darkwall")}
ok(not lit["steady"]["variable_lighting"] and lit["steady"]["shadow_area_p90"] < 0.05,
   "steady lighting does not ask for normalisation", lit["steady"]["shadow_area_p90"])
ok(lit["shadow"]["variable_lighting"],
   "a shadow sweeping the frame asks for normalisation",
   {k: lit["shadow"][k] for k in ("shadow_area_p90", "shadow_area_spread")})
ok(not lit["gain"]["variable_lighting"] and lit["gain"]["exposure_drift_ratio"] > 1.8,
   "a whole-frame gain is REPORTED as drift but does not trigger flattening, which cannot fix it",
   {k: lit["gain"][k] for k in ("exposure_drift_ratio", "shadow_area_p90", "variable_lighting")})
ok(not lit["darkwall"]["variable_lighting"],
   "a permanently dark surface is albedo, not variable light",
   {k: lit["darkwall"][k] for k in ("shadow_area_p90", "shadow_area_spread", "variable_lighting")})
ok(lit["gain"]["shadow_area_median"] <= lit["steady"]["shadow_area_median"] + 0.02,
   "the trigger is gain-invariant by construction: a 3x whole-frame gain does not raise "
   "the suspect area at all",
   f"{lit['steady']['shadow_area_median']} -> {lit['gain']['shadow_area_median']}")
ok(cd.probe_illumination([])["frames_probed"] == 0
   and not cd.probe_illumination([])["variable_lighting"],
   "no frames sampled reads as 'not measured', never as 'stable'")
ok(cd.probe_illumination([np.full((24, 24), 128, np.uint8)] * 3)["frames_probed"] == 0,
   "frames under the block grid report as unmeasured instead of raising")

agg_lit = pipeline.aggregate_diag({"clips": [{"clip": "a", "illumination": lit["steady"]},
                                             {"clip": "b", "illumination": lit["shadow"]}]})
ok(agg_lit["lighting_measured"] and agg_lit["variable_lighting"],
   "one clip with moving light is enough to owe the pass to the whole scene")
ok(pipeline.aggregate_diag({"clips": [{"clip": "a"}]})["lighting_measured"] is False,
   "a cached probe from before the illumination block is not read as 'stable'")
ok(pipeline.diag_uptodate(tmp, {"videos": [], "poses": {}}) is False,
   "and a scene with no diagnostics.json is simply stale")

vals = {}
pipeline._resolve_photometric(vals, agg_lit, "auto", False)
ok(vals["photometric"] == "on" and vals["image_dir"] == "frames_match",
   "auto + measured variable light -> COLMAP is pointed at the flattened copy", vals)
forced = {}
pipeline._resolve_photometric(forced, agg_lit, "off", False)
ok(forced["image_dir"] == "frames_train",
   "--photometric off overrides the measurement and keeps the raw frames", forced)
none = {}
pipeline._resolve_photometric(none, pipeline.aggregate_diag(None), "auto", False)
ok(none["photometric"] == "off" and none["image_dir"] == "frames_train",
   "no measurement at all falls back to the raw frames, never to a pass nobody measured")

base_cfg = {"name": "t", "work": Path("work/t"), "preset": "drone", "variant": "v",
            "target": 400, "width": 640, "steps": 100, "cap": 100, "voxel": "0.3",
            "sources": {"videos": [], "poses": {}, "frames_dirs": {}},
            "cull": pipeline.CULL_CANOPY}


def _steps(**extra):
    steps = pipeline.build_steps(dict(base_cfg, **extra))
    return [s["name"] for s in steps], {s["name"]: s for s in steps}


names, by_name = _steps(photometric="on", image_dir="frames_match")
ok(names.index("keyframes") < names.index("frames") < names.index("colmap"),
   "the frames step sits between keyframes and colmap", names[:4])
ok("survey_frames.py" in str(by_name["frames"]["argv"][1])
   and "--flatten" in by_name["frames"]["argv"] and "--keep-all" in by_name["frames"]["argv"],
   "and it is the built capability, run pose-free, with the frame budget left alone")
ok(by_name["frames"]["outputs"] and all(str(o).endswith(("frames_match.json", "frames_match"))
   for o in by_name["frames"]["outputs"]),
   "its declared outputs are the report and the directory it writes")
off_names, off_steps = _steps(photometric="off", image_dir="frames_train")
ok("frames" not in off_names, "off pays for nothing: no frames step at all")
_, plain_steps = _steps()                      # a cfg from before this wiring existed
plan_digest = lambda step: step["argv"][step["argv"].index("--plan-hash") + 1]  # noqa: E731
ok(plan_digest(off_steps["colmap"]) == plan_digest(plain_steps["colmap"]),
   "choosing the raw directory hashes to exactly the pre-wiring plan, so no finished "
   "COLMAP step went stale because image_dir joined the whitelist")
ok(plan_digest(by_name["colmap"]) != plan_digest(plain_steps["colmap"]),
   "and choosing the flattened one changes the plan hash: the decision sits inside what "
   "staleness is judged by, so a run cannot skip normalisation and still look current")

# And the step itself, run the way the pipeline runs it: no GPS, no reconstruction,
# no telemetry. It used to be unable to start at all without one of the two.
import survey_frames as _sf  # noqa: E402


def lit_scene(root, n=3, size=(320, 240)):
    """frames_full + frames_train + keyframes.jsonl, as extract_keyframes leaves them."""
    frames = lit_frames("shadow", n=max(n, 3))
    rows = []
    for i in range(n):
        colour = np.dstack([frames[i]] * 3)
        for folder, img in (("frames_full", colour),
                            ("frames_train", _cv2.resize(colour, size))):
            d = root / folder / "clip"
            d.mkdir(parents=True, exist_ok=True)
            _cv2.imwrite(str(d / f"{i:05d}.jpg"), img, [_cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"file": f"clip/{i:05d}.jpg", "clip": "clip", "frame_index": i,
                     "t_sec": i * 0.5, "sharpness": 100.0})
    (root / "keyframes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                          encoding="utf-8")
    return [r["file"] for r in rows]


with tempfile.TemporaryDirectory() as sfd:
    run = Path(sfd) / "keepall"
    names = lit_scene(run)
    train_before = {run / "frames_train" / n: (run / "frames_train" / n).read_bytes()
                    for n in names}
    rc_flat = _sf.main(["--work", str(run), "--flatten", "--keep-all",
                        "--out", str(run / "frames_match.json")])
    rep = json.loads((run / "frames_match.json").read_text(encoding="utf-8"))
    ok(rc_flat == 0 and rep["frames_out"] == len(names),
       "survey_frames flattens with no pose source at all", rep.get("position_source"))
    ok(rep["operations"] == {"select": False, "flatten": True, "mask": False}
       and all(s.get("matching_copy") == ["illumination_flattened"] for s in rep["frame_scores"]),
       "and the report says what ran, per frame")
    ok(sorted(p.relative_to(run / "frames_match").as_posix()
              for p in (run / "frames_match").rglob("*.jpg")) == names,
       "the matching copy holds exactly the keyframes, no more and no fewer")
    ok(all(_cv2.imread(str(run / "frames_match" / n)).shape[:2] == (240, 320) for n in names),
       "written at the trained frames' pixel size, so masks and cameras still line up")
    ok(all(p.read_bytes() == b for p, b in train_before.items()),
       "frames_train is byte-identical after the pass: flattening never reaches the colour copy")
    copy = rep["colour_copy"]
    ok(copy["unchanged"] and copy["digest_before"] == copy["digest_after"]
       and copy["digest_before"],
       "and the step records that fact as a digest either side of itself, so a finished "
       "run can be audited without re-deriving it", copy)
    flat_field = _ph.illumination_field(_cv2.imread(str(run / "frames_match" / names[0]),
                                                    _cv2.IMREAD_GRAYSCALE), sigma_cells=2.0)
    raw_field = _ph.illumination_field(_cv2.imread(str(run / "frames_train" / names[0]),
                                                   _cv2.IMREAD_GRAYSCALE), sigma_cells=2.0)
    ok((np.percentile(flat_field, 95) - np.percentile(flat_field, 5)) / flat_field.mean()
       < (np.percentile(raw_field, 95) - np.percentile(raw_field, 5)) / raw_field.mean(),
       "the low-frequency field really is smaller in the copy COLMAP reads",
       f"field std {raw_field.std():.1f} -> {flat_field.std():.1f}")

    # --keep-all has to be a real difference and not luck about easy frames: fail every
    # quality score, and only the flag keeps the directory one-for-one with the manifest.
    real_assess = _sf.capture.assess_frame
    _sf.capture.assess_frame = lambda gray, **kw: dict(real_assess(gray, **kw),
                                                       keep=False, weight=0.0)
    try:
        _sf.main(["--work", str(run), "--flatten", "--keep-all",
                  "--out", str(run / "forced.json")])
        forced = json.loads((run / "forced.json").read_text(encoding="utf-8"))
        dropper = Path(sfd) / "dropper"
        lit_scene(dropper)
        _sf.main(["--work", str(dropper), "--flatten", "--out", str(dropper / "gone.json")])
        dropped = json.loads((dropper / "gone.json").read_text(encoding="utf-8"))
    finally:
        _sf.capture.assess_frame = real_assess
    ok(forced["frames_out"] == len(names) and forced["dropped_by_quality"] == []
       and all("kept_despite" in s for s in forced["frame_scores"]),
       "--keep-all writes a matching copy for a frame the score would drop, and says so",
       forced["frames_out"])
    ok(dropped["frames_out"] == 0 and len(dropped["dropped_by_quality"]) == len(names)
       and not any((dropper / "frames_match").rglob("*.jpg")),
       "without it the survey lane still drops them, and leaves nothing on disk for COLMAP "
       "to match by accident", dropped["frames_out"])

    def _cli(args):
        try:
            return _sf.main(args)
        except SystemExit as e:                    # parser.exit raises; it does not return
            return e.code

    ok(_cli(["--work", str(run), "--select", "--out", str(run / "nope.json")]) == 2
       and not (run / "nope.json").exists(),
       "--select still refuses to run without camera centres instead of guessing")
    ok(_cli(["--work", str(run), "--preparation", str(run / "a.json"),
             "--poses", str(run / "b.jsonl"), "--flatten"]) == 2,
       "and still will not take both pose sources at once")

    # A receipt that cannot fail is decoration. Fault-inject the exact bug it guards
    # against - the flattening path leaking a frame into the colourised copy mid-pass -
    # and require the step to notice, record it and refuse to continue.
    with tempfile.TemporaryDirectory() as td:
        proof = Path(td) / "scene"
        files = lit_scene(proof)
        digest = _sf._colour_copy_digest(proof / "frames_train")
        _sf.main(["--work", str(proof), "--flatten", "--out", str(proof / "clean.json")])
        clean = json.loads((proof / "clean.json").read_text(encoding="utf-8"))["colour_copy"]
        ok(clean["digest_before"] == digest and clean["unchanged"],
           "the recorded digest is the directory as it stands either side of the pass", clean)

        real_prep = _sf.capture.prepare_for_matching
        seen = []

        def prep_that_leaks(gray, **kw):
            out = real_prep(gray, **kw)
            if not seen:                      # first frame: mid-pass for the digest
                seen.append(1)
                (proof / "frames_train" / files[0]).write_bytes(b"the leak this guards")
            return out

        _sf.capture.prepare_for_matching = prep_that_leaks
        try:
            caught_rc = _cli(["--work", str(proof), "--flatten", "--out",
                              str(proof / "caught.json")])
        finally:
            _sf.capture.prepare_for_matching = real_prep
        caught = json.loads((proof / "caught.json").read_text(encoding="utf-8"))["colour_copy"]
        ok(caught_rc != 0 and not caught["unchanged"] and caught["digest_before"] == digest,
           "a colour copy that moved mid-pass fails the step and records it, rather than "
           "shipping a model with the light painted in", (caught_rc, caught))

# ------------------------------------------- priors injection (sqlite direct)
db_path = tmp / "selftest.db"
con = sqlite3.connect(str(db_path))
con.execute(rc.POSE_PRIORS_DDL_V4)
con.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, "
            "camera_id INTEGER NOT NULL)")
for i, nm in enumerate(["a/00000.jpg", "a/00001.jpg"]):
    con.execute("INSERT INTO images VALUES (?, ?, ?)", (i + 1, nm, 1))
con.commit()
con.close()

priors = tmp / "pose_priors.jsonl"
priors.write_text("\n".join(json.dumps({"file": f"a/{n}.jpg",
                                        "position": [i * 1.5, 0, 1], "std": 0.15})
                            for i, n in enumerate(["00000", "00001"])), encoding="utf-8")
written, total = rc.inject_pose_priors(db_path, priors)
ok((written, total) == (2, 2), f"inject counts {written}/{total}")
con = sqlite3.connect(str(db_path))
r = con.execute("SELECT corr_data_id, corr_sensor_id, corr_sensor_type, position, "
                "coordinate_system FROM pose_priors WHERE corr_data_id=1").fetchone()
pos = np.frombuffer(r[3], dtype="<f8")
cs, stype = r[4], r[2]
con.close()
ok(cs == rc.CARTESIAN == 1, f"CARTESIAN enum consistent ({cs})")
ok(stype == rc.SENSOR_TYPE_CAMERA == 0, f"CAMERA sensor type ({stype})")
ok(np.allclose(pos, [0, 0, 1]), f"prior blob decodes {pos}")
try:
    import pycolmap  # noqa: F401
    ok(int(py_cs := pycolmap.PosePriorCoordinateSystem.CARTESIAN) == rc.CARTESIAN,
       "pycolmap agrees on CARTESIAN value")
except ImportError:
    print("skip  pycolmap not installed (optional - the prior blob is checked directly)")

# ------------------------------------------------------- source resolution
sys.path.insert(0, str(ROOT))
import pipeline  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    fake_videos = Path(td)
    for c in ("walk1", "walk2"):
        (fake_videos / f"{c}.mp4").write_bytes(b"x")
        (fake_videos / f"{c}_poses.jsonl").write_text("{}", encoding="utf-8")
    old = pipeline.VIDEOS
    try:
        pipeline.VIDEOS = fake_videos.parent
        src = pipeline.resolve_sources(fake_videos.name, [], [])
        ok(sorted(src["videos"]) == sorted([fake_videos / "walk1.mp4",
                                            fake_videos / "walk2.mp4"]),
           "multi-video discovery")
        ok(set(src["poses"]) == {"walk1", "walk2"}, "auto pose pairing by stem prefix")
        src2 = pipeline.resolve_sources(fake_videos.name,
                                        [str(fake_videos / "walk2.mp4")],
                                        [f"walk2={fake_videos / 'walk2_poses.jsonl'}"])
        ok(len(src2["videos"]) == 1 and list(src2["poses"]) == ["walk2"], "explicit flags win")
    finally:
        pipeline.VIDEOS = old

# ------------------------------------------------- auto preset: measured evidence only
# The classifier used to have exactly one input that mattered - was there a pose log -
# because an essential matrix normalises its translation, so a drone at 12 m/s and a hand
# at 0.3 m/s produce identical numbers. Every clip on this machine classified as
# "orbit_mixed", so `room` and `drone` were the only reachable answers and six of the
# eight presets could not be chosen at all. These cases use the REAL measurements from
# videos/ (recorded in docs/GAPS_AND_OPTIMIZATIONS.md, E2).
def AGG(**kw):
    a = {"motion_measured": False, "mount_residual_px": None, "styles": [],
         "walked_m": None, "max_speed_m_per_s": None, "max_straightness": None,
         "sky_fraction_p90": None, "expansion_gain": None}
    a.update(kw)
    return a


def picks(agg, handheld):
    return pipeline.pick_preset(agg, handheld)


name, ev = picks(AGG(motion_measured=True, walked_m=43.0, max_speed_m_per_s=0.309,
                     max_straightness=0.025), True)
ok(name == "room", "room_w_jsonl's own log - 43 m at 0.31 m/s, closed - is a person in "
                   "a room")
ok(ev and all({"signal", "value", "because"} <= set(e) for e in ev),
   "every preset decision carries the number that made it, in words")
ok(picks(AGG(motion_measured=True, walked_m=200.0, max_speed_m_per_s=8.4,
             max_straightness=0.9), False)[0] == "drone",
   "a pose log that moves faster than a walk is a vehicle, not a handset")
ok(picks(AGG(motion_measured=True, walked_m=30.0, max_speed_m_per_s=0.9,
             max_straightness=0.8), True)[0] == "corridor",
   "a straight run at walking pace is the hallway preset, not the room preset")
ok(picks(AGG(motion_measured=True, walked_m=95.0, max_speed_m_per_s=0.5,
             max_straightness=0.1), True)[0] == "indoor_large",
   "95 m of slow walking in one pass is a floorplate, not a room")
ok(picks(AGG(mount_residual_px=5.86), False)[0] == "room",
   "rocks' shakier cousin: 5.9 px of frame bob is a walk even with no pose log")
ok(picks(AGG(mount_residual_px=0.008), False)[0] == "drone",
   "0.008 px of bob is a gimbal or an airframe, and that is how rocks and temple are "
   "recognised here without any telemetry")
ok(picks(AGG(mount_residual_px=0.3), False)[0] == "room",
   "between the two calibration bands it declines to claim flight and falls back")
ok(picks(AGG(sky_fraction_p90=0.386, mount_residual_px=5.86), False)[0] == "room"
   and not any("sky" in e["signal"] for e in picks(
       AGG(sky_fraction_p90=0.386, mount_residual_px=5.86), False)[1]),
   "sky area never decides the preset: a white painted ceiling measured 38.6% 'sky' on "
   "room_w_jsonl, so the signal is recorded and refused")
ok(picks(AGG(sky_fraction_p90=0.047, expansion_gain=0.9), False)[0] == "room",
   "a strong expansion signature alone is not proof of flight either")
name, ev = picks(AGG(), False)
ok(name == "room", "no measurements at all falls back to the indoor preset, not aerial")
ok(any("not a finding" in e["because"] for e in ev),
   "and it says out loud that a default is not a diagnosis")
# The bug this ordering was written for, still guarded: routing a handheld orbit at the
# aerial presets switched the canopy cull on against a painted ceiling.
for style in ("orbit_mixed", "translation_sweep", "rotation_dominant", "low_texture_or_blur"):
    hand = picks(AGG(styles=[style], mount_residual_px=3.9), True)[0]
    ok(pipeline.PRESETS[hand]["cull"] == pipeline.CULL_NONE,
       f"a handheld {style} clip never lands on a preset that culls the ceiling")
ok(pipeline.PRESETS[picks(AGG(mount_residual_px=0.01), False)[0]]["cull"]
   == pipeline.CULL_CANOPY, "an aerial pass does get the canopy cull it needs")
AERIAL = [p for p, v in pipeline.PRESETS.items()
          if v.get("cull") == pipeline.CULL_CANOPY and p != "auto"]
ok(AERIAL and set(AERIAL) <= {"drone", "drone_mapping", "outdoor_building", "sky_heavy"},
   f"only the open-sky presets cull a canopy: {AERIAL}")

# ------------------------------------------------------------ runner: --only is a restriction
# Stubbing run_step keeps this a test of the pass itself: which steps do_run
# chooses to execute, in order, with no subprocess and no GPU.
import types  # noqa: E402

srcs = {"videos": [], "frames_dirs": {}, "poses": {}}
rargs = types.SimpleNamespace(cmd="run", name="unitrunner", preset="room",
                              quality="standard", variant="default")


def runner_pass(only, from_step=None, fresh=False):
    ran = []
    cfg = pipeline.build_config(rargs, srcs, allow_auto_diag=False)
    cfg["work"] = Path(td) / "unitrunner"
    cfg["only"], cfg["from_step"], cfg["fresh"] = set(only), from_step, fresh
    old = pipeline.run_step
    try:
        def _stub(step, *rest):
            ran.append(step["name"])
            return {"status": "done", "secs": 0.0, "attempts": 1,
                    "fallbacks": [], "log": "stub"}
        pipeline.run_step = _stub
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.do_run(cfg)
    finally:
        pipeline.run_step = old
    return ran


with tempfile.TemporaryDirectory() as td:
    ran = runner_pass(["keyframes", "colmap"])
    ok(ran == ["keyframes", "colmap"], f"--only runs exactly the listed steps: {ran}")
    full = runner_pass([])
    same_cfg = pipeline.build_config(rargs, srcs, allow_auto_diag=False)
    same_cfg["work"] = Path(td) / "unitrunner"
    declared = [s["name"] for s in pipeline.build_steps(same_cfg)]
    ok(full == declared and not {"sky", "clouds", "reexport"} & set(full),
       f"no --only runs the full declared plan in order ({len(full)} steps), "
       f"and an indoor preset carries no canopy cull",
       f"got {full} vs {declared}")
    partial = runner_pass(["keyframes"], fresh=True)
    ok(partial == ["keyframes"],
       f"--only holds under --fresh, which otherwise marks everything stale: {partial}")


# ------------------------------------------- the dashboard reads the runner's disk
# The terminal and the step table are only trustworthy if they agree with what the
# runner wrote - including runs this server never started.
import _serve as srv  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    logs = Path(td) / "work" / "demo" / "logs"
    logs.mkdir(parents=True)
    (logs / "01-keyframes.log").write_text("$ argv\nline a\nline b\n[exit 0] 3.5s\n",
                                           encoding="utf-8")
    # What the hardened runner actually writes now: a first-try success, a step
    # a fallback rung saved, and a failure that names its class.
    (logs / "02-train.log").write_text("$ argv\nstep 300/300\n[ok 1] 41s\n",
                                       encoding="utf-8")
    (logs / "03-collider.log").write_text(
        "$ argv\nboom\n[exit -1073741819 crash] 9s\n$ argv smaller voxels\n"
        "[ok 2] 31s\n", encoding="utf-8")
    (logs / "04-surface.log").write_text("$ argv\nno ground.f32\n"
                                         "[exit 3 empty-input] 12s\n", encoding="utf-8")
    # Logs are numbered by step order, so the step in flight is always the
    # highest-numbered file, and that is the one the tail has to call live.
    (logs / "05-evals.log").write_text("$ argv\nrendering\n", encoding="utf-8")
    (Path(td) / "work" / "demo" / "keyframes_poses.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (Path(td) / "work" / "demo" / "keyframes.jsonl").write_text("{}\n{}\n{}\n", encoding="utf-8")

    sc = srv.scan_scenes(td)[0]
    ok(sc["name"] == "demo" and sc["registered"] == [2, 3],
       f"scene scan reads the registered/total pair ({sc['registered']})")
    ok([s["status"] for s in sc["steps"]]
       == ["done", "done", "recovered", "failed", "running"],
       "a footer means done, a footerless fresh log means running, a rung-2 "
       "success reads as recovered and a classified exit reads as failed",
       str([s["status"] for s in sc["steps"]]))
    ok(sc["steps"][2]["attempts"] == 2,
       "a step that only passed on its second rung says so",
       str(sc["steps"][2]))
    ok(sc["steps"][3]["kind"] == "empty-input" and sc["steps"][3]["exit"] == 3,
       "the failure class reaches the monitor, not just the log",
       str(sc["steps"][3]))
    ok(sc["steps"][0]["attempts"] is None,
       "the legacy footer still parses off disk", str(sc["steps"][0]))
    ok(sc["steps"][0]["secs"] == 3.5 and sc["steps"][0]["exit"] == 0,
       "seconds and exit code come off the log footer, not out of memory")

    t1 = srv.tail_run(td, "demo", "0")
    ok(t1["lines"][:4] == ["$ argv", "line a", "line b", "[exit 0] 3.5s"],
       "the tail merges step logs in order into one stream")
    ok(t1["current"] == "05-evals.log" and t1["running"] is True,
       "and it knows which step is live", f"{t1['current']} {t1['running']}")
    t2 = srv.tail_run(td, "demo", t1["cursor"])
    ok(t2["lines"] == [], "a resumed cursor yields nothing new")
    with (logs / "05-evals.log").open("a", encoding="utf-8") as fh:
        fh.write("more\n")
    t3 = srv.tail_run(td, "demo", t1["cursor"])
    ok(t3["lines"] == ["more"], "appended lines arrive on the next poll")
    with (logs / "05-evals.log").open("a", encoding="utf-8") as fh:
        fh.write("part")
    t4 = srv.tail_run(td, "demo", t3["cursor"])
    ok(t4["lines"] == [], "a half-written last line is held back, not shown broken")
    (logs / "05-evals.log").write_text("fresh\n", encoding="utf-8")
    t5 = srv.tail_run(td, "demo", t4["cursor"])
    ok(t5["lines"] == ["fresh"], "an offset past a replaced log restarts that log")

    httpd = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(srv.H, directory=td))
    live = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{live}/api/scenes") as r:
            got = json.load(r)
        ok([s["name"] for s in got["scenes"]] == ["demo"],
           "the live /api/scenes serves the temp work dir")
        with urllib.request.urlopen(f"http://127.0.0.1:{live}/api/tail?scene=demo&cursor=0") as r:
            got2 = json.load(r)
        ok(got2["lines"][0] == "$ argv" and got2["running"] is True,
           "the live /api/tail streams a run the server did not start")
    finally:
        httpd.shutdown()


# ------------------------------------------------------------ a real handset take
# videos/test1 is the folder his phone produced: a 360 turned on the spot, then
# the same turn pointing up at the ceiling and down at the floor. It stays as the
# fixture that measures the technique rather than the code, so the numbers below
# only move when a take is captured that walks UNDER the ceiling.
take = ROOT / "videos" / "test1"
if take.exists():
    import analyze_take as at  # noqa: E402

    tr = at.analyse(take)
    d = tr.data
    ok(d["poses"] == 2591 and abs(d["seconds"] - 94.9) < 0.5,
       f"fixture take reads {d['poses']} poses over {d['seconds']:.1f} s")
    ok(abs(d["walked"] - 15.1) < 0.5, f"fixture walked {d['walked']:.1f} m in total")
    ok(abs(d["medianY"] - 1.26) < 0.05,
       f"fixture phone height median {d['medianY']:.2f} m")
    ok(d["ceilingFrac"] > 0.2 and d["ceilingSpan"] < at.BASELINE_WANTED,
       f"fixture ceiling: {d['ceilingFrac'] * 100:.0f}% of the take pointed up from "
       f"{d['ceilingSpan']:.1f} m of ground - spun in place, so no baseline")
    ok(d["roomH"] is not None and d["roomH"] < at.MIN_ROOM_H,
       f"fixture room export reads {d['roomH']:.2f} m, under the {at.MIN_ROOM_H} m "
       f"headroom the engine now requires")
    ok(tr.failed == 2, f"fixture fails exactly its two known rules ({tr.failed})")
else:
    print("skip  videos/test1 is not present")

# ------------------------------------------------- canopy culling by preset
def steps_for(preset, cull=None):
    cfg = {"name": "t", "work": Path("work/t"), "preset": preset, "variant": "cluster_shell",
           "target": 400, "width": 640, "steps": 1000, "cap": 100, "voxel": "0.3",
           "sources": {"videos": [], "poses": {}, "frames_dirs": {}},
           "cull": cull or pipeline.PRESETS[preset].get("cull", pipeline.CULL_NONE)}
    return [s["name"] for s in pipeline.build_steps(cfg)]


CULL_STEPS = ["sky", "clouds", "reexport"]
INDOOR = [p for p, v in pipeline.PRESETS.items()
          if v.get("cull") == pipeline.CULL_NONE and p != "auto"]
OUTDOOR = [p for p, v in pipeline.PRESETS.items() if v.get("cull") == pipeline.CULL_CANOPY]
ok(INDOOR and OUTDOOR, f"presets split into {INDOOR} indoors / {OUTDOOR} under open sky")
for p in INDOOR:
    got = [s for s in steps_for(p) if s in CULL_STEPS]
    ok(not got, f"'{p}' runs no canopy cull - a white ceiling is desaturated and airborne, "
                f"so strip_clouds would cut it", f"ran {got}")
for p in OUTDOOR:
    ok([s for s in steps_for(p) if s in CULL_STEPS] == CULL_STEPS,
       f"'{p}' still cuts a cloud sea or a sky crust, in the right order")
ok([s for s in steps_for("room", "canopy") if s in CULL_STEPS] == CULL_STEPS,
   "--cull canopy puts all three back on an indoor preset")
ok(not [s for s in steps_for("drone", "none") if s in CULL_STEPS],
   "--cull none takes all three off an aerial preset")

# --------------------------------------- the geometry check COLMAP now relies on
import cv2  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "frames_train"
    for folder, (w, h) in {"clip_wide": (640, 296), "clip_tall": (296, 640)}.items():
        d = root / folder
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "00000.jpg"), np.zeros((h, w, 3), np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    ok(rc.image_size(root / "clip_wide" / "00000.jpg") == (640, 296),
       "JPEG header read gives width then height, not the array's own order",
       str(rc.image_size(root / "clip_wide" / "00000.jpg")))
    ok(rc.image_size(root / "clip_tall" / "00000.jpg") == (296, 640),
       "and a portrait frame reads back tall")
    geo = rc.frame_geometries(root)
    ok(geo == {"clip_wide": (640, 296), "clip_tall": (296, 640)},
       "one geometry per clip folder, so a mixed scene is detected before frames are lost",
       str(geo))
    ok(len({s for s in geo.values()}) == 2,
       "the two geometries disagree - with a shared camera COLMAP would have dropped one "
       "clip's frames and exited 0")

# ------------------------------------------------ json a browser can actually parse
# json.dumps writes float('nan') as a bare NaN, json.loads reads it back without
# complaint, and JSON.parse in the viewer rejects the whole file - which is how
# one unmeasured spawn height stopped test2horizontal's walk test at load.
with tempfile.TemporaryDirectory() as td:
    jf = Path(td) / "collision.json"
    _rb.write_json(jf, {"route_metrics": {"spawn_above_floor_m": float("nan"),
                                          "loop_m": float("inf"), "keep": 1.25}})
    _txt = jf.read_text(encoding="utf-8")
    ok("NaN" not in _txt and "Infinity" not in _txt,
       "write_json never emits a bare NaN/Infinity token", _txt[-90:].strip())
    _back = json.loads(_txt)
    ok(_back["route_metrics"]["spawn_above_floor_m"] is None
       and _back["route_metrics"]["keep"] == 1.25,
       "a non-finite measurement reads back as null, not as a wrong number")
    ok(json.dumps(_rb.json_safe({"a": [float("nan")]}), allow_nan=False),
       "json_safe recurses into lists and dicts")

# ------------------------------------------- shifting a mask by more than it is wide
# test2horizontal's splat collapsed, so its grid came out 17x14 cells of 1.7 cm.
# The clearance radius is a metre value divided by the cell size, so it landed at
# ~21 cells on a 17-row axis. shifted_b's `min(nz, nz + di)` then went negative,
# numpy read that as a wrap-around stop, and the step died with
# "could not broadcast input array from shape (0,14) into shape (15,14)".
import walk_path_from_glb as wp  # noqa: E402


def _shift_ref(S, di, dj, fill):
    """What a shift means, cell by cell, with no slice arithmetic at all."""
    nz, nx = S.shape
    out = np.full((nz, nx), fill, dtype=float)
    for i in range(nz):
        for j in range(nx):
            si, sj = i - di, j - dj
            if 0 <= si < nz and 0 <= sj < nx:
                out[i, j] = S[si, sj]
    return out


_rng = np.random.default_rng(0)
_small = (_rng.random((9, 7)) > 0.5)
_offsets = [(di, dj) for di in range(-11, 12) for dj in range(-9, 10)]
_bad_b = sum(not np.array_equal(wp.shifted_b(_small, di, dj),
                                _shift_ref(_small, di, dj, 0.0).astype(bool))
             for di, dj in _offsets)
ok(_bad_b == 0,
   "shifted_b matches a per-cell reference at every offset, including |d| > axis",
   f"{_bad_b} mismatches over {len(_offsets)} offsets")
_bad_f = sum(not np.array_equal(np.nan_to_num(wp.shifted(_small.astype(float), di, dj)),
                                _shift_ref(_small, di, dj, 0.0))
             for di, dj in _offsets)
ok(_bad_f == 0, "shifted keeps the same bounds and NaNs only the vacated edge",
   f"{_bad_f} mismatches")
ok(wp.close_mask(_small, 21).shape == _small.shape,
   "a dilation radius wider than the grid closes instead of raising",
   f"{int(wp.close_mask(_small, 21).sum())} cells kept")
ok(int(wp.close_mask(_small, 1).sum()) >= int(_small.sum()),
   "an ordinary radius still closes the mask rather than erasing it")

# ------------------------------------------------ the router and the physics
# walk_path_from_glb demanded a fixed 0.34 m of clearance, but viewer/pc.js
# builds `radius: 0.34 * CHAR_SCALE` with CHAR_SCALE = character_height / 1.75.
# A room preset ships 0.15 m, so it walked a 0.029 m capsule past a router asking
# for 11x that — test2horizontal's 14x17 grid of 1.7 cm cells could never fit one
# and reported "nowhere has 0.02 m of clearance".
ok(abs(wp.capsule_radius({"character_height": 0.15}) - 0.34 * 0.15 / 1.75) < 1e-12,
   "the router plans the capsule the viewer actually builds",
   f"{wp.capsule_radius({'character_height': 0.15}):.4f} m")
for _missing in ({}, {"character_height": None}, {"character_height": "tall"},
                 {"character_height": 0}, {"character_height": -1},
                 {"character_height": float("nan")}):
    ok(wp.capsule_radius(_missing) == wp.CAPSULE_R,
       f"no usable character_height ({_missing!r}) falls back to the default, "
       f"never a zero-size body")
ok(wp.capsule_radius({"character_height": 0.01}) == wp.CAPSULE_R * 0.05 / 1.75,
   "the router keeps the viewer's own 0.05 m floor, so a plan is never narrower "
   "than the physics that walks it")
ok(wp.char_scale({"character_height": 0.15}) == 0.15 / 1.75
   and wp.char_scale({}) == 1.0,
   "a scene with no usable character height is judged as the full 1.75 m body")
_ratio = wp.CLEARANCE_M / wp.CAPSULE_R
ok(all(abs(wp.CLEARANCE_M * wp.char_scale({"character_height": h})
           - _ratio * wp.capsule_radius({"character_height": h})) < 1e-12
       for h in (0.05, 0.15, 0.6, 1.75, 3.0)),
   "the corridor demand stays a fixed multiple of the body at every character "
   "height, so --min-clearance needs no per-video tuning")
_res = 0.0175
ok(int(round(wp.CAPSULE_R / _res)) > 14
   and int(round(wp.capsule_radius({"character_height": 0.15}) / _res)) < 14,
   "on that 14x17 / 1.7 cm grid the unscaled radius exceeds the grid and the "
   "scaled one fits",
   f"unscaled {int(round(wp.CAPSULE_R / _res))} cells vs "
   f"scaled {int(round(wp.capsule_radius({'character_height': 0.15}) / _res))} "
   "cells on a 14-row axis")

# ------------------------------------------ a marker has to name the code that ran it
# step_uptodate compared the command, the outputs and the inputs' mtimes, never
# the source. Editing a step script therefore left that step "done" and the next
# run shipped artifacts built by code that had already changed.
with tempfile.TemporaryDirectory() as _td:
    _root = Path(_td)
    (_root / "scripts").mkdir()
    _s1 = _root / "scripts" / "one.py"
    _s2 = _root / "scripts" / "two.py"
    _s1.write_text("print(1)\n", encoding="utf-8")
    _s2.write_text("print(2)\n", encoding="utf-8")
    _d1 = pipeline.code_digest([sys.executable, str(_s1)])
    ok(_d1 == pipeline.code_digest([sys.executable, str(_s1)]),
       "unchanged source digests the same, so a finished step stays finished")
    ok(pipeline.code_digest([sys.executable, str(_s2)]) != _d1,
       "two different step scripts get two different digests", _d1)
    _s1.write_text("print(1)\n# one line added\n", encoding="utf-8")
    ok(pipeline.code_digest([sys.executable, str(_s1)]) != _d1,
       "editing the script a step runs invalidates its marker")

# The evidence and enhancement steps are the last things to fail and the least
# load-bearing: a take still ships its world if a browser cannot hand over a jpg,
# the semantic labeller cannot run, the navmesh bake produces nothing usable, the
# monocular second ruler cannot load, or the scenario audit finds nothing to read —
# the game then falls back to the heightfield instead of the run aborting.
# `gate` is deliberately NOT in here: it is the one step with a bespoke failure branch,
# because a world that fails a hard check must be reported as failed, not shrugged off.
ok(set(pipeline.ADVISORY) == {"evals", "pairs", "semantics", "nav", "rooms", "depth",
                              "audit", "texture"},
   "evidence and labelling steps are advisory, so none can abort the walk test",
   ", ".join(pipeline.ADVISORY))
ok("gate" not in pipeline.ADVISORY and "audit" not in ("gate",),
   "the world gate stays load-bearing while its audit is optional")

# ------------------------------------------- one locked screenshot is not a failed run
# Image.save() used to write straight onto the target, truncating it in place.
# A jpg a viewer had memory-mapped cannot be truncated, msvcrt reports that as
# OSError [Errno 22] Invalid argument, and the run aborted at the evidence step
# with the world already built.
from PIL import Image  # noqa: E402

_im = Image.fromarray(np.zeros((8, 8, 3), np.uint8))
with tempfile.TemporaryDirectory() as _td:
    _tgt = Path(_td) / "AB_00.jpg"
    _tgt.write_bytes(b"from the previous run")
    ok(_rb.save_image(_im, Path(_td) / "no-such-dir" / "AB_00.jpg", quality=80)
       is False, "a write the OS refuses returns False instead of raising")
    ok(_rb.save_image(_im, _tgt, quality=80), "save_image reports the image it wrote")
    with Image.open(_tgt) as _read:
        _size = _read.size
    ok(_size == (8, 8), "an existing target is replaced, not failed on", str(_size))
    ok(not list(Path(_td).glob("*.tmp*")), "no temp file left beside the images")

# The actual condition from the rocks crash, not a stand-in for it: a jpg some
# viewer mapped cannot be truncated, and the writer must come back False with the
# old file still readable rather than half-written.
with tempfile.TemporaryDirectory() as _td:
    _map = Path(_td) / "AB_08.jpg"
    _map.write_bytes(b"from the previous run")
    with open(_map, "r+b") as _fh:
        _view = mmap.mmap(_fh.fileno(), 0, access=mmap.ACCESS_READ)
        ok(_rb.save_image(_im, _map, quality=80) is False,
           "a memory-mapped jpg refuses the write instead of truncating")
        _view.close()
    ok(_map.read_bytes() == b"from the previous run",
       "the refused file still holds its previous bytes", str(_map.stat().st_size))

# One refused name must cost the name, not the rest of the stacks: the old code
# named each output len(key), so every pair after the locked one tried to write
# the same locked name again and the whole tail of the run dropped out.
with tempfile.TemporaryDirectory() as _td:
    _td = Path(_td)
    _real, _rend, _blind = _td / "real", _td / "render", _td / "out" / "blinded"
    for i in range(4):
        _real.mkdir(parents=True, exist_ok=True)
        _rend.mkdir(parents=True, exist_ok=True)
        _im.save(str(_real / f"{i:05d}.jpg"), quality=90)
        _im.save(str(_rend / f"eval_{i:02d}.png"))
    (_td / "pairs.json").write_text(json.dumps(
        [{"real_file": f"{i:05d}.jpg", "render_file": f"eval_{i:02d}.png"}
         for i in range(4)]), encoding="utf-8")
    _blind.mkdir(parents=True, exist_ok=True)
    _bad_name, _real_save = "t_AB_01.jpg", _rb.save_image

    def _refuse_one(im, path, **kw):
        return False if Path(path).name == _bad_name else _real_save(im, path, **kw)

    _rb.save_image = _refuse_one
    _mp = importlib.import_module("make_pairs")
    _out = io.StringIO()
    _argv = list(sys.argv)
    try:
        sys.argv = ["make_pairs.py", "--real-dir", str(_real),
                    "--render-dir", str(_rend),
                    "--pairs", str(_td / "pairs.json"),
                    "--out", str(_td / "out"), "--tag", "t"]
        with contextlib.redirect_stdout(_out), contextlib.redirect_stderr(_out):
            _mp.main()
    finally:
        sys.argv = _argv
        _rb.save_image = _real_save
    _made = sorted(p.name for p in _blind.glob("t_AB_*.jpg"))
    ok(len(_made) == 4, "every pair found a writable name", ", ".join(_made))
    ok(_bad_name not in _made, "the refused name was stepped over, not forced")
    ok(len(set(_made)) == 4, "no two pairs were written to the same name")
    _key = json.loads((_td / "out" / "pair_key_t.json").read_text(encoding="utf-8"))
    ok({e["pair"] for e in _key} == set(_made),
       "the key names exactly the stacks on disk", f"{len(_key)} entries")

    # A second take must not cost the first one its answer key. The key used to be
    # one shared file, so after a matrix run only the last take's stacks could be
    # unblinded and every earlier jpg sat on disk with no mapping.
    _argv = list(sys.argv)
    try:
        sys.argv = ["make_pairs.py", "--real-dir", str(_real),
                    "--render-dir", str(_rend),
                    "--pairs", str(_td / "pairs.json"),
                    "--out", str(_td / "out"), "--tag", "u"]
        with contextlib.redirect_stdout(_out), contextlib.redirect_stderr(_out):
            _mp.main()
    finally:
        sys.argv = _argv
    _keys = {p.name: json.loads(p.read_text(encoding="utf-8"))
             for p in sorted((_td / "out").glob("pair_key_*.json"))}
    ok(sorted(_keys) == ["pair_key_t.json", "pair_key_u.json"],
       "each take keeps its own key", ", ".join(sorted(_keys)))
    ok({e["pair"] for e in _keys["pair_key_t.json"]} == set(_made),
       "the first take's key still names the first take's stacks")
    ok(sum(len(v) for v in _keys.values()) == len(list(_blind.glob("*_AB_*.jpg"))),
       "every stack on disk is named by some key")

report_exit("checks")
