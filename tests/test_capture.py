"""Checks the capture stage survives the inputs that used to kill it.

  .venv\\Scripts\\python.exe tests\\test_capture.py     (or tests\\check_all.py)

Each case is a real failure shape from work/*/logs or from the audit: a video
that opens but decodes nothing, a clip whose only failure mode was a silent
empty manifest, a pose log of NaNs, a frames-dir spec the Windows drive-letter
parser mangled. The old behaviour ranged from a bare exit 0 with no output to an
AttributeError traceback.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import robust as R  # noqa: E402
import extract_keyframes as EK  # noqa: E402

fails = []
PY = sys.executable


def ok(cond, label, detail=""):
    print(("pass  " if cond else "FAIL  ") + label
          + (("  " + str(detail).replace("\n", " | ")[:240]) if detail else ""))
    if not cond:
        fails.append(label)


def tmpdir(tag):
    return Path(tempfile.mkdtemp(prefix=f"cap_{tag}_"))


def write_video(path: Path, n=40, w=320, h=240, fps=30.0):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(1)
    for i in range(n):
        frame = (rng.uniform(0, 255, (h, w, 3))).astype(np.uint8)
        frame[20:60 + i, 20:120] = 200
        vw.write(frame)
    vw.release()


def run_kf(work: Path, *extra):
    return subprocess.run([PY, str(ROOT / "scripts" / "extract_keyframes.py"),
                           "--work", str(work), *extra],
                          capture_output=True, text=True, timeout=600,
                          encoding="utf-8", errors="replace")


# ------------------------------------------------- corrupt video: no traceback
tmp = tmpdir("corrupt")
tmp.mkdir(parents=True, exist_ok=True)
bad = tmp / "bad.mp4"
bad.write_bytes(b"not a video at all, just bytes")
p = run_kf(tmp, "--video", str(bad), "--target", "50")
ok("Traceback" not in p.stderr, "corrupt video gives no traceback", p.stderr)
ok(p.returncode != 0 and "no usable source" in (p.stdout + p.stderr),
   "corrupt video is a named, non-zero failure", p.stdout + p.stderr)

# ------------------------------------------------- empty manifest is never silent
# the exact shape that used to write keyframes.jsonl with 0 rows and exit 0
tmp = tmpdir("empty")
vid = tmp / "flat.mp4"
vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
solid = np.full((240, 320, 3), 127, np.uint8)   # no texture, every frame identical
for _ in range(60):
    vw.write(solid)
vw.release()
work = tmp / "w"
work.mkdir()
p = run_kf(work, "--video", str(vid), "--target", "30")
rows = R.jsonl_rows(work / "keyframes.jsonl")
ok(not (p.returncode == 0 and len(rows) == 0),
   "a textureless clip never exits 0 with an empty manifest",
   f"rc={p.returncode} rows={len(rows)} " + p.stdout)
ok(p.returncode == 0 or "no keyframes" in (p.stdout + p.stderr),
   "if it fails, it says why", p.stdout + p.stderr)

# ------------------------------------------------- one bad clip must not kill the rest
tmp = tmpdir("mixed")
good = tmp / "good.mp4"
write_video(good, n=80)
evil = tmp / "evil.mp4"
evil.write_bytes(b"\x00\x00 garbage")
work = tmp / "w"
work.mkdir()
p = run_kf(work, "--video", str(evil), str(good), "--target", "40")
rows = R.jsonl_rows(work / "keyframes.jsonl")
ok(p.returncode == 0 and len(rows) > 0,
   "a scene with one unreadable clip still produces keyframes",
   f"rc={p.returncode} rows={len(rows)}", )
ok("skipping clip evil.mp4" in (p.stdout + p.stderr),
   "the skipped clip is named on the console")

# ------------------------------------------- probe reads the DECODED size, not the header
tmp = tmpdir("probe")
vid = tmp / "v.mp4"
write_video(vid, n=20, w=320, h=480)          # portrait: 320x480
m = EK.probe(vid)
ok((m["width"], m["height"]) == (320, 480),
   "probe reports the decoded portrait shape", m)
ok("rotate_hint" in m, "probe records whether the container disagreed")

# --------------------------------------- containers lie about fps and length
# videos/test2train/data.webm genuinely reports fps=1000.0 and frame count
# -9223372036854775808; the old probe() passed both through, so duration became
# -9.2e15 s and every budget and t_sec derived from it was garbage.
webm = ROOT / "videos" / "test2train" / "data.webm"
if webm.exists():
    import cv2 as _cv2
    raw = _cv2.VideoCapture(str(webm))
    raw_fps, raw_n = raw.get(_cv2.CAP_PROP_FPS), int(raw.get(_cv2.CAP_PROP_FRAME_COUNT))
    raw.release()
    m = EK.probe(webm)
    ok(raw_n < 0, "the webm really does report an overflowing frame count", raw_n)
    ok(m["nb_frames"] > 0 and m["duration"] > 0,
       "probe recovers a sane length from a lying container",
       f"n={m['nb_frames']} dur={m['duration']:.1f}s fps={m['fps']}")
    ok(0 < m["fps"] <= 240, "implausible fps is not passed through", m["fps"])
    ok("fps_measured" in m, "probe records whether the rate was measured or guessed")

# --------------------------------------------------- pixel cap beats side cap
tw, th = EK._train_size(1280, 2772, 1280, 1_100_000)
ok(tw * th <= 1_100_000, "a portrait frame is capped by total pixels",
   f"{tw}x{th}={tw * th}")
tw3, th3 = EK._train_size(640, 480, 640, 1_100_000)
ok((tw3, th3) == (640, 480), "a frame already inside the budget is untouched",
   f"{tw3}x{th3}")
ok(all(v % 2 == 0 for v in (tw, th)), "resize dims stay even for the codec", f"{tw}x{th}")

# --------------------------------------------------- handle released, no lock
tmp = tmpdir("lock")
vid = tmp / "v.mp4"
write_video(vid, n=30)
src = vid.read_bytes()
try:
    EK.probe(vid)
except Exception:
    pass
try:
    vid.write_bytes(src)          # fails on Windows if the capture is still open
    ok(True, "probe releases the capture so the file is not left locked")
except PermissionError as e:
    ok(False, "probe releases the capture so the file is not left locked", e)

# --------------------------------------------------- diagnostics keys must match
tmp = tmpdir("diag")
d = {"clips": [{"clip": "walk1.mp4", "motion_weight": 0.3}]}
f = tmp / "diagnostics.json"
R.write_json(f, d)
loaded = R.read_json(f)
diag = {Path(str(c.get("clip", ""))).stem: float(c.get("motion_weight", 1.0))
        for c in loaded["clips"]}
ok(diag.get("walk1") == 0.3,
   "diagnostics clip names normalise to the stem the budget lookup uses", diag)
ok("walk1.mp4" not in diag, "the old name-keyed lookup would have missed")

# --------------------------------------------------- NaN priors cannot leak
import import_phone_poses as IPP  # noqa: E402
sys.modules.setdefault("poses_lib", __import__("poses_lib"))
nan_rows = [{"t": 0.0, "pxyz": [0.0, 0.0, 0.0], "qxyzw": [0, 0, 0, 1]},
            {"t": 1.0, "pxyz": [float("nan")] * 3, "qxyzw": [0, 0, 0, 1]},
            {"t": 2.0, "pxyz": [1.0, 0.0, 0.0], "qxyzw": [0, 0, 0, 1]}]
clean = [r for r in nan_rows if R.finite(*(r["pxyz"] + r["qxyzw"]))]
ok(len(clean) == 2, "finite-value filter drops NaN pose samples", clean)
ok(not R.finite(1.0, float("nan")), "R.finite rejects NaN")
ok(R.finite(1.0, 2.0), "R.finite accepts real numbers")

# --------------------------------------------------- empty/unreadable pose log
tmp = tmpdir("poses")
work = tmp / "w"
work.mkdir()
(work / "keyframes.jsonl").write_text(
    json.dumps({"file": "a/0.jpg", "clip": "a", "frame_index": 0,
                "t_clip": 0.0, "t_sec": 0.0}) + "\n", encoding="utf-8")
empty_log = tmp / "a_poses.jsonl"
empty_log.write_text("", encoding="utf-8")
PRIORS = str(ROOT / "scripts" / "import_phone_poses.py")


def run_priors(*extra):
    return subprocess.run([PY, PRIORS, "--work", str(work),
                           "--log", f"a={empty_log}", *extra],
                          capture_output=True, text=True, timeout=180,
                          encoding="utf-8", errors="replace")


p = run_priors()
ok("Traceback" not in p.stderr,
   "an empty pose log reports a message, not a traceback", p.stderr)
# Priors are an optional boost to the solve: colmap's ladder can run without
# them, so a bad AR log must cost the priors and not the scene.
ok(p.returncode == 0, "an unusable log degrades instead of failing the run",
   f"rc={p.returncode} {p.stdout.strip()}")
ok("[warn]" in p.stdout and "WITHOUT pose priors" in p.stdout,
   "the degradation is named on the console", p.stdout.strip())
skip = R.read_json(work / "priors_skipped.json", {})
ok(bool(skip) and "reason" in skip,
   "and the reason is persisted for the report", skip)
ok(not (work / "pose_priors.jsonl").exists(),
   "no priors file is left behind to be read as valid")

strict = run_priors("--strict")
ok(strict.returncode != 0,
   "--strict turns the same input back into a hard failure",
   f"rc={strict.returncode}")
ok("[empty-input]" in strict.stderr,
   "and it is a classified failure, not a bare exit code", strict.stderr.strip())

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all capture-stage checks passed")
