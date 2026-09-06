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

# ------------------------------------------------- auto preset: capture beats motion
# room_w_jsonl is an indoor phone scan whose camera walked a circle around the
# room. Its motion style reads "orbit_mixed", which used to route it at the
# aerial presets: the canopy cull switched on against a white painted ceiling,
# and clip_gap - what keeps a room's walls in the collider - was never passed.
ok(pipeline.pick_preset(["orbit_mixed"], handheld=True) == "room",
   "a handheld orbit stays indoor even though it moved like a drone orbit")
ok(pipeline.pick_preset(["orbit_mixed"], handheld=False) == "drone",
   "a drone orbit still gets the aerial preset")
ok(pipeline.pick_preset(["translation_sweep"], handheld=True) == "room",
   "a phone walking a straight line is not proof of flight")
ok(pipeline.pick_preset(["translation_sweep"], handheld=False) == "drone",
   "a straight push-forward with no pose log stays aerial")
ok(pipeline.pick_preset(["rotation_dominant"], handheld=True) == "room",
   "the rotation-heavy case the room preset was written for")
ok(pipeline.pick_preset([], handheld=True) == "room",
   "no diagnostics falls back to the general-purpose indoor preset, not an aerial one")
AERIAL = [p for p, v in pipeline.PRESETS.items()
          if v.get("cull") == pipeline.CULL_CANOPY and p != "auto"]
for st in (["orbit_mixed"], ["translation_sweep"], ["rotation_dominant"],
           ["static_or_unknown"], []):
    got = pipeline.pick_preset(st, handheld=True)
    ok(got not in AERIAL and pipeline.PRESETS[got].get("cull") == pipeline.CULL_NONE,
       f"handheld + {st or 'no style'} -> '{got}': cull off and clip_gap "
       f"{pipeline.PRESETS[got].get('clip_gap', 'MISSING')} reaches the collider")

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

# The evidence steps are the last things to fail and the least load-bearing: a
# take still ships its world if a browser cannot hand over a jpg.
ok(set(pipeline.ADVISORY) == {"evals", "pairs"},
   "the two evidence steps are advisory, so neither can abort the walk test",
   ", ".join(pipeline.ADVISORY))

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
