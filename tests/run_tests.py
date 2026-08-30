"""Self-contained verification for the v2 pipeline additions. No framework.

  .venv\\Scripts\\python.exe tests\\run_tests.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import poses_lib as pl  # noqa: E402
import capture_diagnostics as cd  # noqa: E402
import run_colmap as rc  # noqa: E402

PASS = 0


def ok(cond, label, detail=""):
    global PASS
    if not cond:
        print(f"FAIL {label} {detail}")
        sys.exit(1)
    PASS += 1
    print(f"pass {label}")


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
ok(int(py_cs := __import__("pycolmap").PosePriorCoordinateSystem.CARTESIAN) == rc.CARTESIAN,
   "pycolmap agrees on CARTESIAN value")

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

print(f"\nALL {PASS} CHECKS PASSED")
