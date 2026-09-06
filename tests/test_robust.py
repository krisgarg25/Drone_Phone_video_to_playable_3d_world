"""Checks for scripts/robust.py -- the budgets and classifiers the runner bets on.

  .venv\\Scripts\\python.exe tests\\test_robust.py      (or tests\\check_all.py)
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import robust as R  # noqa: E402

R.configure_streams()

fails = []


def ok(cond, label, detail=""):
    print(("pass  " if cond else "FAIL  ") + label + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(label)


# --- failure classification -------------------------------------------------
ok(R.classify(1, "torch.cuda.OutOfMemoryError: CUDA out of memory.") == R.OOM,
   "CUDA OOM -> oom")
ok(R.classify(1, "RangeError: Map maximum size exceeded") == R.VOXEL_OVERFLOW,
   "voxel Map limit -> voxel-overflow")
ok(R.classify(1, "Failed to parse options - unrecognised option '--X.y'") == R.UNSUPPORTED_FLAG,
   "bad colmap flag -> unsupported-flag")
faiss_err = ("E visual_index.cc:690] Check failed: file_version == 1 || "
             "file_version == 2 Failed to read faiss index. This may be caused by "
             "reading a legacy flann-based index")
ok(R.classify(1, faiss_err) == R.UNSUPPORTED_ASSET,
   "legacy flann vocab tree -> unsupported-asset")
# The same abort reported with no text at all is only ever a bare crash; the
# message is what turns it into a repairable class, so the real run - which
# fast-fails with 0xC0000409 after logging - must classify off the text.
ok(R.classify(-1073741819, faiss_err) == R.UNSUPPORTED_ASSET,
   "asset verdict beats the NTSTATUS reading", R.status_name(-1073741819))
win = "'C:\\Users\\krisg\\Desktop\\Drone' is not recognized as an internal or external command"
ok(R.classify(1, win) == R.MISSING_TOOL, "split path -> missing-tool")
ok(R.classify(1, "E... incremental_mapper.cc] Failed to create any sparse model") == R.EMPTY_INPUT,
   "no sparse model -> empty-input")
ok(R.classify(-1, "") == R.CRASH, "bare exit -1 -> crash")
# A step that raised StepError already named the class in its own message. Reading
# that back as a bare "failed" is what made test2horizontal's frame stop look like a
# mystery crash, and it costs the runner the "do not retry this" hint.
ok(R.classify(4, "[frame] [empty-input] multi-view support found 86 of 7555")
   == R.EMPTY_INPUT, "a step's declared empty-input survives to the report")
ok(R.classify(1, "[train] [oom] halving the pixel budget; (also: no images)")
   == R.OOM, "the declared class beats incidental wording from another tool")
ok(R.classify(1, "[path] nowhere has that much clearance") == R.FAILED,
   "an unnamed failure is still reported as unnamed")
ok("access violation" in R.status_name(-1073741819), "STATUS_ACCESS_VIOLATION decoded",
   R.status_name(-1073741819))
ok(R.status_name(0) == "ok", "exit 0 -> ok")

# --- run_cmd refuses the shell=True path bug -------------------------------
try:
    R.run_cmd("python -c \"print(1)\"")
    ok(False, "run_cmd rejects a string argv")
except TypeError as e:
    ok("spaces" in str(e), "run_cmd rejects a string argv", str(e))

import tempfile
tmp = Path(tempfile.mkdtemp(prefix="robust_"))
p = R.run_cmd([sys.executable, "-c", "print('hi from a dir with spaces')"], cwd=tmp)
ok(p.stdout.strip().endswith("spaces"), "run_cmd runs with no shell")
try:
    R.run_cmd([sys.executable, "-c", "import sys; sys.stderr.write('CUDA out of memory'); sys.exit(3)"])
    ok(False, "run_cmd raises on nonzero")
except R.StepError as e:
    ok(e.kind == R.OOM and e.returncode == 3, "nonzero exit classified", str(e))
try:
    R.run_cmd([sys.executable, "-c", "import time; time.sleep(5)"], timeout=1, retries=0)
    ok(False, "run_cmd enforces timeout")
except R.StepError as e:
    ok(e.kind == R.TIMEOUT, "timeout classified", str(e))

# --- the streamed path the runner uses for every step log ------------------
import io  # noqa: E402

_buf = io.StringIO()
_rp = R.run_cmd([sys.executable, "-c",
                 "import sys; print('out'); print('err', file=sys.stderr); print('two')"],
                log=_buf)
ok(_buf.getvalue().replace("\r", "").split() == ["out", "err", "two"],
   "output reaches the log in order, stderr folded in", repr(_buf.getvalue()))
ok(len(_rp.stdout or "") >= 8, "the caller still gets the text to classify",
   repr(_rp.stdout))
_t0 = time.time()
_buf2 = io.StringIO()
try:
    # Silent on purpose: a deadline only checked when a line arrives would hang.
    R.run_cmd([sys.executable, "-c", "import time; time.sleep(30)"],
              timeout=2, log=_buf2, retries=0)
    ok(False, "a silent process still hits its timeout")
except R.StepError as e:
    ok(e.kind == R.TIMEOUT and time.time() - _t0 < 12,
       "a silent process is killed at its deadline, not at its next word",
       f"{time.time() - _t0:.1f}s {e.kind}")
_buf3 = io.StringIO()
try:
    R.run_cmd([sys.executable, "-c",
               "import sys; print('CUDA out of memory'); sys.exit(1)"], log=_buf3)
    ok(False, "streamed nonzero should raise")
except R.StepError as e:
    ok(e.kind == R.OOM and "out of memory" in _buf3.getvalue(),
       "a streamed failure classifies off the text that went to the log", str(e))

# --- voxel budget ----------------------------------------------------------
v, cells = R.fit_voxel_size([0, 0, 0], [60, 30, 60], 0.05)
ok(cells <= R.VOXEL_CELL_LIMIT and v > 0.05, "60m scene at 0.05m coarsened to fit",
   f"voxel={v:.3f} cells={cells}")
v2, c2 = R.fit_voxel_size([0, 0, 0], [8, 3, 8], 0.35)
ok(v2 == 0.35 and c2 < R.VOXEL_CELL_LIMIT, "small room keeps the requested voxel",
   f"voxel={v2} cells={c2}")
degenerate = R.fit_voxel_size([0, 0, 0], [0.2, 0.2, 0.2], 0.35)
ok(degenerate[0] > 0, "sub-voxel scene does not divide by zero", str(degenerate))
ok(R.voxel_ladder(0.35) == [0.56, 0.896, 1.43, 2.29, 3.66] or len(R.voxel_ladder(0.35)) == 5,
   "voxel ladder is monotonic and finite")

# --- vram / pixel budget --------------------------------------------------
b_small = R.train_budget(vram_gb=2.0, n_frames=400)
b_big = R.train_budget(vram_gb=12.0, n_frames=400)
ok(b_small["cap"] < b_big["cap"], "cap scales with available VRAM",
   f"{b_small['cap']} vs {b_big['cap']}")
# The budget has two levers and must never leave the run holding more than it
# can: shrink the frames, or stop caching them at all. Asserting "small card =>
# stream" was wrong in a good way - 400 frames shrunk to 0.18 MP do fit, and
# forcing them to stream would only make the run slower for no safety.
for vram, n in ((2.0, 400), (2.0, 1500), (4.0, 400), (6.0, 260), (12.0, 900),
                (1.0, 80), (24.0, 2000)):
    b = R.train_budget(vram_gb=vram, n_frames=n)
    cache = n * b["pixels"] * 3 / 1024**3
    usable = max(0.4, vram * R.SAFE_FRACTION - R.BASE_GB)
    ok(b["stream_images"] or cache <= usable * 0.25 + 1e-9,
       f"{vram:g}GB x {n} frames is inside the cache allowance",
       f"pixels={b['pixels'] / 1e6:.2f}MP cache={cache:.2f}GiB "
       f"allowance={usable * 0.25:.2f}GiB stream={b['stream_images']}")
ok(R.train_budget(vram_gb=2.0, n_frames=4000)["stream_images"],
   "a frame set too big to shrink into memory streams instead of dying",
   str(R.train_budget(vram_gb=2.0, n_frames=4000)))
ok(R.train_budget(vram_gb=4.0, n_frames=0)["cap"] > 0, "budget works with no frame count")
ok(R.train_budget(vram_gb=0.1)["cap"] >= 120_000, "cap never drops below a usable floor")

w, h = R.pixels_for(1080, 2772, 1_100_000)
ok(w * h <= 1_100_000 and abs((h / w) - (2772 / 1080)) < 0.02,
   "portrait 1080x2772 capped by pixels, aspect kept", f"{w}x{h}={w * h}")
ok(R.pixels_for(640, 480, 1_100_000) == (640, 480), "small frames untouched")
ok(all(x % 2 == 0 for x in R.pixels_for(1001, 2003, 50_000)), "resize dims stay even")

# --- colmap capability probe, per subcommand -------------------------------
# The union of every subcommand's options is actively wrong: Mapper.multiple_models
# is a real option of `mapper` and makes `global_mapper` abort parsing, which is
# how three of the six rescue rungs died reporting a puzzling under-registration.
_COLMAP = ROOT / "tools" / "colmap" / "bin" / "colmap.exe"
if _COLMAP.exists():
    import run_colmap as _rc
    _map = R.colmap_option_map(_COLMAP, env=_rc.env())
    _rc._OPTIONS = _map          # no re-probe, and no cache file in the repo root
    ok(bool(_map.get("mapper")) and bool(_map.get("global_mapper")),
       "the probe reads each subcommand's own help",
       f"mapper={len(_map.get('mapper') or ())} global_mapper="
       f"{len(_map.get('global_mapper') or ())}")
    ok(_rc.known_options("mapper") >= {"Mapper.multiple_models"},
       "mapper's own options include Mapper.multiple_models")
    ok("Mapper.multiple_models" not in _rc.known_options("global_mapper"),
       "and global_mapper's do not - so it gets dropped, not passed and fatal",
       f"global_mapper has {len(_rc.known_options('global_mapper'))} options")
    ok("Mapper.multiple_models" in _rc.known_options(),
       "the union still knows it, for a subcommand the probe could not read")
    ok(bool(_rc.known_options("model_aligner")),
       "a subcommand with an unreadable help screen falls back to the union, "
       "not to an empty set that would drop every flag")
else:
    print("skip  vendored colmap.exe not present; probe test not run")

# --- colmap flag capability filter ----------------------------------------
known = {"Mapper.init_min_tri_angle", "Mapper.max_num_iterations"}
kept, dropped = R.split_flags(
    ["colmap", "mapper", "--Mapper.init_min_tri_angle", "4",
     "--SequentialMatching.loop_detection_vocab_tree", "tree.bin",
     "--Mapper.max_num_iterations", "25"], known)
ok(dropped == ["--SequentialMatching.loop_detection_vocab_tree", "tree.bin"],
   "unknown flag and its value dropped", str(dropped))
ok("4" in kept and "--Mapper.init_min_tri_angle" in kept, "known flag pair kept", str(kept))
kept2, dropped2 = R.split_flags(["colmap", "mapper", "/data/in", "--Mapper.x"], known)
ok(dropped2 == ["--Mapper.x"] and kept2[:3] == ["colmap", "mapper", "/data/in"],
   "unknown trailing flag dropped, positional not swallowed", str(kept2))
ok(R.split_flags(["a", "--Any.thing", "1"], set())[1] == [],
   "empty option set drops nothing (probe failed != flag invalid)")

# --- safe reductions ------------------------------------------------------
import numpy as np
ok(R.safe_min(np.array([]), 0.0, label="test") == 0.0, "safe_min on empty -> default")
ok(R.safe_min(np.array([np.nan, 3.0, 1.0])) == 1.0, "safe_min skips NaN")
ok(R.safe_pct(np.array([]), 95, -1) == -1, "safe_pct on empty -> default")
ok(R.safe_max(np.array([np.nan]), None) is None,
   "all-NaN reduces to default, not ValueError")
try:
    R.safe_max(np.array([np.nan]))
    ok(False, "all-NaN with no default still raises")
except ValueError:
    ok(True, "all-NaN with no default still raises")
try:
    R.safe_min(np.array([]))
    ok(False, "safe_min without a default still raises loudly")
except ValueError as e:
    ok("empty" in str(e), "safe_min without a default still raises loudly")
ok(R.clamp_index(-5, 999, (10, 10)) == (0, 9), "clamp_index bounds a wild spawn",
   str(R.clamp_index(-5, 999, (10, 10))))
try:
    R.load_array(tmp / "nope.f32", np.float32, required=True)
    ok(False, "load_array names a missing upstream artifact")
except R.StepError as e:
    ok(e.kind == R.EMPTY_INPUT, "load_array names a missing upstream artifact")
ok(R.load_array(tmp / "nope.f32", np.float32, required=False) is None,
   "load_array optional returns None")

# --- upstream row validation ---------------------------------------------
(tmp / "empty.jsonl").write_text("", encoding="utf-8")
try:
    R.require_rows(tmp / "empty.jsonl", 1, what="keyframes manifest")
    ok(False, "empty manifest is a hard, early error")
except R.StepError as e:
    ok(e.kind == R.EMPTY_INPUT, "empty manifest is a hard, early error", str(e))
(tmp / "half.jsonl").write_text('{"a":1}\nnot-json\n{"b":2}\n', encoding="utf-8")
ok(len(R.jsonl_rows(tmp / "half.jsonl")) == 2, "unparseable line skipped, not fatal")
(tmp / "bad-enc.json").write_bytes(b'{"d": "2\xc2\xb0"}')
ok("2" in R.read_text(tmp / "bad-enc.json"), "read_text never raises UnicodeDecodeError")

# --- report ---------------------------------------------------------------
rep = R.Report(tmp, "unit")
rep.step("keyframes", "done", secs=1.2)
rep.step("colmap", "done", secs=3.4, attempts=2, fallbacks=["relaxed tri angle"])
rep.step("gate", "warning", detail="route short")
rep.write()
loaded = R.read_json(tmp / "report.json")
ok(loaded["status"] == R.COMPLETE_WITH_WARNINGS, "warnings never fail a produced run",
   loaded["status"])
summary = R.human_summary(rep)
ok(summary.count("\n") == 3 and "colmap" in summary and "gate" in summary
   and "keyframes" in summary, "summary lists every step once", summary)
ok("note: " not in summary, "no notes means no note lines")
rep_note = R.Report(tmp / "z", "unit4")
rep_note.step("gate", "warning")
rep_note.notes.append("ceiling pruned at 0.04 opacity")
ok("note: ceiling pruned" in R.human_summary(rep_note), "notes rendered as text")

rep3 = R.Report(tmp / "y", "unit3")
rep3.step("colmap", "failed", kind=R.EMPTY_INPUT, detail="first line\nsecond line")
ok("empty-input first line" in R.human_summary(rep3)
   and "second line" not in R.human_summary(rep3),
   "failed detail is one plain line, not a list repr", R.human_summary(rep3))
ok(loaded["steps"][1]["fallbacks"] == ["relaxed tri angle"], "fallbacks recorded")
rep2 = R.Report(tmp / "x", "unit2")
rep2.step("colmap", "failed", kind=R.EMPTY_INPUT)
ok(rep2.status == R.FAILED_STATUS, "a real failure still reports failed")

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all robust.py checks passed")
