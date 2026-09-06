"""Shared hardening for every pipeline step: subprocess execution, resource
budgets, and failure classification.

This module exists because a step used to die with a bare exit code and leave
the operator to guess which number to hand-tune. Each helper here returns the
*decision* the pipeline should make next -- a coarser voxel size, a smaller
pixel budget, the flags this COLMAP build actually accepts -- rather than only
a message.

Stdlib only. It is imported from both venvs (3.12 for SfM/geometry, 3.10 for
training), and torch is never a hard dependency here.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# failure taxonomy
#
# The whole point of naming failures is that a named failure has a known repair.
# `run_cmd` classifies, and callers map a class onto the next thing to try
# instead of aborting.
# ---------------------------------------------------------------------------
OOM = "oom"                      # ran out of VRAM or RAM: shrink, then retry
VOXEL_OVERFLOW = "voxel-overflow"  # voxel grid exceeded Node's Map limit
UNSUPPORTED_FLAG = "unsupported-flag"  # binary doesn't know this option
UNSUPPORTED_ASSET = "unsupported-asset"  # binary refuses to read this data file
EMPTY_INPUT = "empty-input"      # an upstream step produced nothing usable
MISSING_TOOL = "missing-tool"    # binary/npx package not available
TIMEOUT = "timeout"
CRASH = "crash"                  # killed by the OS, no diagnostics
FAILED = "failed"                # non-zero exit, unclassified

# Signatures observed in work/*/logs across the scenes that died. Ordered most
# specific first: a CUDA OOM also contains "error", so the informative pattern
# has to win.
_PATTERNS = [
    (OOM, r"out of memory|CUDA error: no space|cudaErrorOutOfMemory|"
          r"std::bad_alloc|MemoryError|Cannot allocate memory|"
          r"array is too big|exceeds memory"),
    # RangeError: Map maximum size exceeded -- splat-transform extracting faces
    # from a grid that is too fine for the scene's extent.
    (VOXEL_OVERFLOW, r"Map maximum size exceeded|Maximum call stack|"
                     r"Array buffer allocation failed"),
    (UNSUPPORTED_FLAG, r"unrecognised option|unrecognized option|"
                       r"Failed to parse options|unknown option"),
    # COLMAP moved its retrieval index from flann to faiss in May 2025 and now
    # hard-aborts on a tree written before then. The message names the cause;
    # without this pattern the caller only sees 0xC0000409 and a dead solve.
    (UNSUPPORTED_ASSET, r"Failed to read faiss index|legacy flann-based index|"
                        r"Check failed: file_version"),
    (MISSING_TOOL, r"is not recognized as an internal or external command|"
                   r"cannot find the file|ENOENT|command not found|"
                   r"npm ERR!|npx: installed|Could not resolve"),
    (EMPTY_INPUT, r"no images|empty reconstruction|Failed to create any sparse model|"
                  r"insufficient size|Not enough images"),
]
_COMPILED = [(kind, re.compile(pat, re.I)) for kind, pat in _PATTERNS]

# A step that raised StepError already diagnosed itself, and __str__ puts the kind
# at the front of the message. The patterns above are for failures that did not
# announce themselves; without this a self-classified "empty-input" came back to the
# runner as a bare "failed", which is both the wrong report and the wrong hint about
# whether to retry.
_DECLARED_RX = re.compile(
    r"\[(" + "|".join(re.escape(k) for k in (
        OOM, VOXEL_OVERFLOW, UNSUPPORTED_FLAG, UNSUPPORTED_ASSET, EMPTY_INPUT,
        MISSING_TOOL, TIMEOUT, CRASH)) + r")\]", re.I)

# Windows NTSTATUS codes that mean "the process was killed", not "it failed".
# A trained--away negative returncode is otherwise indistinguishable from a
# script's own sys.exit(1).
_NTSTATUS = {
    0xC0000005: "access violation (segfault)",
    0xC0000374: "heap corruption",
    0xC0000409: "stack buffer overrun / fast-fail",
    0xC00000FD: "stack overflow",
    0x40010004: "debugger terminated",
    0x40010005: "process terminated",
}


def classify(returncode: int, text: str = "") -> str:
    """Name the failure so a caller can pick the repair instead of guessing."""
    blob = text or ""
    declared = _DECLARED_RX.search(blob)
    if declared:
        return declared.group(1).lower()
    for kind, rx in _COMPILED:
        if rx.search(blob):
            return kind
    if returncode < 0:
        # Terminated by signal (POSIX) -- on Windows subprocess reports the
        # NTSTATUS as a huge unsigned value wrapped into -1..-1 etc.
        return CRASH
    if returncode > 0x7FFFFFFF:
        # 0xC0000005 / 0xC0000409 and friends: a missing-identity NTSTATUS that
        # arrived as an unsigned int. The process was killed, it did not fail.
        return CRASH
    if returncode == 1 and not blob.strip():
        # A script that exits 1 having printed nothing is not reporting anything.
        return CRASH
    return FAILED


def status_name(returncode: int) -> str:
    """Human label for a raw exit code (4294967295 is meaningless to read)."""
    if returncode == 0:
        return "ok"
    u = returncode & 0xFFFFFFFF
    # Windows reports a fatal exception as a negative returncode holding the
    # NTSTATUS; an access violation arrives as -1073741819 == 0xC0000005.
    if returncode < 0 and u >= 0xC0000000:
        return f"{_NTSTATUS.get(u, f'Windows fatal exception 0x{u:08X}')} (exit {returncode})"
    if returncode < 0:
        return f"terminated by signal {-returncode}"
    if u in _NTSTATUS:
        return f"{_NTSTATUS[u]} (exit {returncode})"
    if u >= 0xC0000000:
        return f"Windows fatal exception 0x{u:08X} (exit {returncode})"
    return f"exit {returncode}"


class StepError(RuntimeError):
    """A step failure carrying the class of problem, for ladder selection."""

    def __init__(self, kind: str, message: str, *, returncode: int = 1,
                 output: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.returncode = returncode
        self.output = output

    def __str__(self) -> str:  # keep the class visible in logs
        return f"[{self.kind}] {self.message}"


# ---------------------------------------------------------------------------
# subprocess execution
#
# The repo root is "Drone to 3d mesh" -- it contains spaces. Any call built with
# shell=True, or handed to `cmd /c` unquoted, splits that path in half and the
# run dies with "'C:\\Users\\...\\Drone' is not recognized". That is not a
# random crash, it is a quoting bug, and it is why run_cmd rejects both forms.
# ---------------------------------------------------------------------------
def run_cmd(argv, *, timeout: float | None = None, env: dict | None = None,
            cwd: Path | None = None, log=None, retries: int = 0,
            retry_on: tuple = (CRASH,)) -> subprocess.CompletedProcess:
    """Run argv with no shell, capture output, classify anything that goes wrong.

    Pass a list, never a string: a string would mean shell=True and reintroduce
    the spaces-in-path bug. Retries cover the crashes that are genuinely
    transient (a GPU driver hiccup, a half-written cache); a classified
    resource/flag failure is raised immediately because retrying it identically
    is what looked like "random" behaviour.
    """
    if isinstance(argv, str):
        raise TypeError("run_cmd needs an argv list; shell strings break on the "
                        "spaces in the repo path")
    argv = [str(a) for a in argv]
    e = {**os.environ, **(env or {})}
    e.setdefault("PYTHONUNBUFFERED", "1")
    e.setdefault("PYTHONIOENCODING", "utf-8")
    attempt = 0
    while True:
        attempt += 1
        try:
            if log is None:
                p = subprocess.run(argv, cwd=str(cwd) if cwd else None, env=e,
                                   capture_output=True, text=True,
                                   encoding="utf-8", errors="replace",
                                   timeout=timeout)
                out = (p.stdout or "") + (p.stderr or "")
                returncode = p.returncode
            else:
                returncode, out = _run_streamed(argv, cwd, e, timeout, log)
            if returncode == 0:
                return subprocess.CompletedProcess(argv, returncode, stdout=out,
                                                   stderr="")
            kind = classify(returncode, out)
            if kind in retry_on and attempt <= retries:
                print(f"  [retry {attempt}/{retries}] {kind}: {argv[0]} "
                      f"{status_name(returncode)}", flush=True)
                time.sleep(min(2 * attempt, 10))
                continue
            tail = "\n".join(out.strip().splitlines()[-25:])
            raise StepError(kind,
                            f"{Path(argv[0]).name} failed: "
                            f"{status_name(returncode)}\n{tail}",
                            returncode=returncode, output=out)
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or b"") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            if attempt <= retries and TIMEOUT in retry_on:
                print(f"  [retry {attempt}/{retries}] timed out after {timeout}s", flush=True)
                continue
            raise StepError(TIMEOUT,
                            f"timed out after {timeout}s: {' '.join(argv[:3])}",
                            output=str(partial))
        except FileNotFoundError as exc:
            raise StepError(MISSING_TOOL, f"cannot launch {argv[0]}: {exc}")


def _run_streamed(argv, cwd, env, timeout, log) -> tuple[int, str]:
    """Run argv writing each output line to `log` the moment it appears.

    stderr is folded into the one stream: every consumer here concatenates the
    two anyway, and one pipe cannot deadlock the way two captured pipes can.
    """
    chunks: list[str] = []
    try:
        proc = subprocess.Popen([str(a) for a in argv],
                                cwd=str(cwd) if cwd else None, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace",
                                bufsize=1)
    except FileNotFoundError as exc:
        raise StepError(MISSING_TOOL, f"cannot launch {argv[0]}: {exc}")

    # A deadline checked only when a line arrives never fires on a silent hang,
    # which is the case a timeout exists for - so the watchdog does the killing.
    killed_by_us = threading.Event()

    def _kill():
        killed_by_us.set()
        try:
            proc.kill()
        except OSError:
            pass

    watchdog = None
    if timeout:
        watchdog = threading.Timer(timeout, _kill)
        watchdog.daemon = True
        watchdog.start()
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            chunks.append(line)
            try:
                log.write(line)
                log.flush()
            except (OSError, ValueError):
                pass                   # a closed log must not kill the run
        code = proc.wait(timeout=60)
    finally:
        if watchdog is not None:
            watchdog.cancel()
    if killed_by_us.is_set():
        raise subprocess.TimeoutExpired(argv, timeout or 0,
                                        output="".join(chunks).encode("utf-8",
                                                                      "replace"))
    return code, "".join(chunks)


def tail_text(path: Path, n: int = 40, max_bytes: int = 512_000) -> str:
    """Last n lines of a log without reading a multi-GB file into memory."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
    except OSError:
        return ""
    lines = data.decode("utf-8", "replace").splitlines()
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------------------
# resource budgets
#
# Every budget here replaces a number an operator used to set by hand after a
# crash. The defaults are sized for the measured hardware (RTX 3050 6GB) but
# read the real device when it is available.
# ---------------------------------------------------------------------------
def free_vram_gb() -> float | None:
    """Free GPU memory in GiB, or None when there is no NVIDIA GPU to ask."""
    try:
        p = subprocess.run(["nvidia-smi",
                            "--query-gpu=memory.free,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        if p.returncode != 0:
            return None
        free, total = [float(x) for x in p.stdout.strip().splitlines()[0].split(",")]
        # Leave the desktop/compositor headroom the driver does not report.
        return max(0.4, min(free, total - 512) / 1024.0)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def torch_vram_gb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free_b, _total_b = torch.cuda.mem_get_info()
        return max(0.4, free_b / 1024**3 * 0.9)
    except Exception:
        return None


def available_vram_gb() -> float:
    """Budget ceiling in GiB. Conservative 4 GiB when nothing can be probed."""
    for probe in (torch_vram_gb, free_vram_gb):
        v = probe()
        if v:
            return v
    return 4.0


# Gaussian-splat training cost, fitted on this repo's own runs: a 592x1280
# (0.76 MP) frame at 110k gaussians peaked at 2.47 GiB on the RTX 3050, while
# 1280x2772 (3.55 MP) with an 850k cap filled the card and the process was
# killed with no traceback at all. The per-step peak is dominated by render
# intermediates that scale with pixels, plus the cloud and its two Adam moments,
# plus a GPU-resident image cache that is a separate, additive term. A "width"
# limit sees none of this -- 1280x2772 is 4.7x a 640x763 budget.
BASE_GB = 0.9                       # CUDA context + gsplat working set
RENDER_GB_PER_MP = 2.0              # rasterizer intermediates + autograd
GAUSSIANS_GB_PER_M = 1.6            # params + 2 Adam states + strategy stats
SAFE_FRACTION = 0.85                # the driver never reports compositor headroom
GAUSSIAN_RESERVE_GB = 1.5           # held back before pixels are chosen
MAX_PIXELS = 2_300_000              # beyond this, detail costs more than it reads


def train_budget(vram_gb: float | None = None, max_pixels: int = 1_100_000,
                 n_frames: int = 0) -> dict:
    """Pick a per-frame pixel count, gaussian cap and cache policy that fit.

    Pixels are decided first, because that is the cost that dominates and the
    one that must always be paid; the cap is whatever survives afterwards.
    Callers resize to `pixels` (total, aspect-preserving) rather than to a side
    length, which is what makes portrait and 4K footage behave like everything
    else.
    """
    vram = vram_gb if vram_gb else available_vram_gb()
    usable = max(0.4, vram * SAFE_FRACTION - BASE_GB)
    render_allow = max(0.35, usable - GAUSSIAN_RESERVE_GB)
    pixels = int(max(96_000, min(max_pixels, MAX_PIXELS,
                                 render_allow / RENDER_GB_PER_MP * 1e6)))
    cache_gb = n_frames * pixels * 3 / 1024**3
    stream = cache_gb > usable * 0.25
    spare = usable if stream else usable - cache_gb
    cap = int(max(120_000, min((spare - pixels / 1e6 * RENDER_GB_PER_MP)
                               / GAUSSIANS_GB_PER_M * 1e6, 4_000_000)))
    return {"pixels": pixels, "cap": cap, "stream_images": stream,
            "vram_gb": round(vram, 2), "cache_gb": round(cache_gb, 2)}


# splat-transform holds the voxel grid in a JS Map; past ~16.7M entries Map itself
# throws ("RangeError: Map maximum size exceeded") after allocating gigabytes.
# That limit is what made --voxel-size a magic number people had to tune.
VOXEL_CELL_LIMIT = 6_000_000


def fit_voxel_size(box_min, box_max, voxel: float,
                   limit: int = VOXEL_CELL_LIMIT) -> tuple[float, int]:
    """Coarsen `voxel` until the box's cell count fits the budget.

    Returns (chosen_voxel, cells_at_chosen). Pure arithmetic, so a caller can
    report the change before spending 60s in the voxeliser to discover it.
    """
    span = [max(1e-3, float(hi) - float(lo)) for hi, lo in zip(box_max, box_min)]
    v = max(1e-3, float(voxel))
    for _ in range(40):
        cells = 1
        for s in span:
            cells *= max(1, int(s / v) + 1)
        if cells <= limit:
            return v, cells
        v *= 1.35
    return v, 0


def voxel_ladder(start: float, steps: int = 5) -> list[float]:
    """Increasingly coarse voxel sizes to retry a resource failure with."""
    out, v = [], float(start)
    for _ in range(steps):
        v = round(v * 1.6, 4)
        out.append(v)
    return out


def pixels_for(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """Aspect-preserving (w, h) rounded to even, capped by total pixel count."""
    w, h = max(16, int(width)), max(16, int(height))
    if w * h <= max_pixels:
        return _even(w), _even(h)
    s = (max_pixels / (w * h)) ** 0.5
    return _even(max(16, int(w * s))), _even(max(16, int(h * s)))


def _even(n: int) -> int:
    return max(16, int(n) // 2 * 2)


# ---------------------------------------------------------------------------
# COLMAP option capability probe
#
# work/auditorium/logs/02-colmap.log died on
#   "Failed to parse options - unrecognised option
#    '--SequentialMatching.loop_detection_vocab_tree'"
# The flag is valid in a newer COLMAP and absent in the vendored build, so the
# whole run aborted over an optional extra. Probe once, drop what is unknown.
# ---------------------------------------------------------------------------
_FLAG_RX = re.compile(r"--([A-Za-z0-9_]+\.[A-Za-z0-9_]+)")

_COLMAP_SUBS = ("feature_extractor", "sequential_matcher", "vocab_tree_matcher",
                "spatial_matcher", "exhaustive_matcher", "mapper", "global_mapper",
                "pose_prior_mapper", "image_undistorter", "model_aligner",
                "point_filtering")


def colmap_option_map(colmap_exe: Path, env: dict | None = None,
                      cache: Path | None = None) -> dict:
    """subcommand -> the --Section.option names *that* subcommand accepts.

    Per-subcommand matters: `Mapper.multiple_models` is a real option of `mapper`
    and an unrecognised one for `global_mapper`, so a union of the two sets let
    the rescue ladder hand it to a binary that aborts on parsing, and three of
    six rungs died reporting an unrelated-sounding failure.
    """
    if cache and cache.exists():
        try:
            blob = json.loads(cache.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            blob = None
        # An older probe cached a flat list; that shape cannot answer a
        # per-subcommand question, so re-probe rather than half-trust it.
        if isinstance(blob, dict) and all(isinstance(v, list) for v in blob.values()):
            return {k: set(v) for k, v in blob.items()}
    known: dict[str, set] = {}
    for sub in _COLMAP_SUBS:
        try:
            p = subprocess.run([str(colmap_exe), sub, "-h"], env=env,
                               capture_output=True, text=True, timeout=60)
            known[sub] = set(_FLAG_RX.findall((p.stdout or "") + (p.stderr or "")))
        except (OSError, subprocess.TimeoutExpired):
            known[sub] = set()
    if cache:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({k: sorted(v) for k, v in known.items()}),
                             encoding="utf-8")
        except OSError:
            pass
    return known


def colmap_known_options(colmap_exe: Path, env: dict | None = None,
                         cache: Path | None = None) -> set:
    """Every --Section.option this binary mentions anywhere in its help output."""
    per_sub = colmap_option_map(colmap_exe, env=env, cache=cache)
    out: set = set()
    for opts in per_sub.values():
        out |= opts
    return out


def split_flags(argv: list, known: set) -> tuple[list, list]:
    """Partition into (supported, dropped) by --Section.option name.

    Only exact `--X.y value` pairs are considered droppable, and a bare `--X.y`
    never swallows the next token, so positional arguments survive.
    """
    kept, dropped, i = [], [], 0
    while i < len(argv):
        a = str(argv[i])
        m = re.fullmatch(r"--([A-Za-z0-9_]+\.[A-Za-z0-9_]+)(=.*)?", a)
        if m and known and m.group(1) not in known:
            dropped.append(a)
            i += 1
            # consume a separate value only when the next token is not another flag
            if i < len(argv) and not str(argv[i]).startswith("--") and not m.group(2):
                if i + 1 < len(argv) and str(argv[i + 1]).startswith("--"):
                    dropped.append(argv[i])
                    i += 1
            continue
        kept.append(a)
        i += 1
    return kept, dropped


# ---------------------------------------------------------------------------
# input validation
#
# Most "random" downstream failures are an upstream step that produced an empty
# or half-written file while exiting 0. These helpers make that a hard, local
# error at the step that caused it.
# ---------------------------------------------------------------------------
def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return default


def json_safe(payload, _path: str = ""):
    """Replace non-finite floats with None, recursively.

    json.dumps writes float('nan') as a bare `NaN`, which is NOT valid JSON - so
    the viewer's JSON.parse rejects the entire file and the browser never reports
    ready. work/test2horizontal shipped `"spawn_above_floor_m": NaN` in
    collision.json that way, and one unmeasured number took down a 300 KB file.
    None becomes `null`, which every consumer here already reads as "not measured".
    """
    import math
    if isinstance(payload, dict):
        return {k: json_safe(v, f"{_path}.{k}" if _path else str(k))
                for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [json_safe(v, f"{_path}[{i}]") for i, v in enumerate(payload)]
    if isinstance(payload, float) and not math.isfinite(payload):
        _note(_path or "float", f"{payload} -> null")
        return None
    return payload


def write_json(path, payload, indent: int | None = 2) -> Path:
    """Atomic: a run killed mid-write must not leave a truncated file that the
    next run reads as corrupt. Non-finite floats become null; see json_safe.

    `indent=None` stays compact - sparse_points.json holds tens of thousands of
    coordinates and is downloaded by the viewer on every load.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(json_safe(payload), indent=indent, default=str,
                              allow_nan=False), encoding="utf-8")
    os.replace(tmp, p)
    return p


def write_text(path, text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, p)
    return p


def save_image(im, path, *, quality: int | None = None, tries: int = 2) -> bool:
    """Write a PIL image the same way write_text writes a file; False if refused.

    `im.save(path)` truncates the target in place, and a jpg/png that a browser or
    image viewer has memory-mapped cannot be truncated -- Windows reports that as
    OSError [Errno 22] Invalid argument. That is how one preview in
    results/blinded/ aborted a run whose world was already built. Renaming over
    the directory entry does not need the target to be free, and a reader never
    sees a half-written image.
    """
    p = Path(path)
    # The extension has to stay last or PIL cannot infer the format to encode.
    tmp = p.with_name(f"{p.stem}.tmp{os.getpid()}{p.suffix}")
    kw = {"quality": quality} if quality else {}
    for attempt in range(1, tries + 1):
        try:
            im.save(tmp, **kw)
            os.replace(tmp, p)
            return True
        except OSError as e:
            _note(p.name, f"{type(e).__name__} ({e.strerror}) on write "
                          f"attempt {attempt} of {tries}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt < tries:
                time.sleep(0.4 * attempt)   # a transient scanner handle clears
    return False


def read_text(path) -> str:
    """Always utf-8. Bare open() on Windows defaults to cp1252 and raises
    UnicodeDecodeError on a log that contains a degree sign."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def jsonl_rows(path, required: tuple = ()) -> list:
    """Parse a .jsonl, skipping unparseable lines instead of aborting the run.

    Returns [] for a missing/empty file; callers decide whether that is fatal.
    """
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for ln, line in enumerate(read_text(p).splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            print(f"[warn] {p.name}:{ln} is not JSON, skipped", flush=True)
            continue
        if required and not isinstance(r, dict):
            continue
        if required and any(k not in r for k in required):
            print(f"[warn] {p.name}:{ln} missing {set(required) - set(r)}, skipped",
                  flush=True)
            continue
        rows.append(r)
    return rows


def require_rows(path, minimum: int = 1, what: str = "manifest") -> list:
    """Hard, early failure when an upstream step silently produced nothing."""
    rows = jsonl_rows(path)
    if len(rows) < minimum:
        raise StepError(
            EMPTY_INPUT,
            f"{what} has {len(rows)} rows, needs {minimum}: {path}\n"
            f"  The step that writes it exited successfully but produced nothing "
            f"usable. Fix that step's input (or its filters) before blaming this one.",
            returncode=3)
    return rows


def require_file(path, what: str = "input") -> Path:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise StepError(EMPTY_INPUT, f"{what} missing or empty: {p}", returncode=3)
    return p


def finite(*values) -> bool:
    try:
        import math
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


def digest(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]


# ---------------------------------------------------------------------------
# safe reductions
#
# The single most common traceback in work/*/logs is a numpy zero-size
# reduction: H[cov > 0].min() when no grid cell has support, np.percentile of an
# empty slice, argmin of nothing. None of those are random -- the subset just
# came out empty for this scene's scale -- and all of them aborted a run that
# had already produced usable geometry. These helpers turn each one into a
# stated default plus a warning the operator can read.
# ---------------------------------------------------------------------------
_MISSING = object()


def _clean(a):
    import numpy as np
    a = np.asarray(a)
    if a.size == 0:
        return a
    if a.dtype.kind == "f":
        a = a[np.isfinite(a)]
    return a


def safe_min(a, default=_MISSING, *, label: str = ""):
    import numpy as np
    a = _clean(a)
    if a.size == 0:
        return _fallback(default, label, "min")
    return float(np.min(a))


def safe_max(a, default=_MISSING, *, label: str = ""):
    import numpy as np
    a = _clean(a)
    if a.size == 0:
        return _fallback(default, label, "max")
    return float(np.max(a))


def safe_mean(a, default=_MISSING, *, label: str = ""):
    import numpy as np
    a = _clean(a)
    if a.size == 0:
        return _fallback(default, label, "mean")
    return float(np.mean(a))


def safe_median(a, default=_MISSING, *, label: str = ""):
    import numpy as np
    a = _clean(a)
    if a.size == 0:
        return _fallback(default, label, "median")
    return float(np.median(a))


def safe_pct(a, q, default=_MISSING, *, label: str = ""):
    """Percentile that returns `default` on an empty/NaN slice instead of
    ValueError. `a` may be an axis-0 sequence of vectors; pass a `q` tuple."""
    import numpy as np
    a = _clean(a)
    if a.size == 0:
        return _fallback(default, label, f"p{q}")
    out = np.percentile(a, q, axis=0)
    return float(out) if np.ndim(out) == 0 else out


def safe_argmax(a, default: int = -1, *, label: str = "") -> int:
    import numpy as np
    a = _clean(a)
    if a.size == 0:
        _note(label or "argmax", "empty -> default")
        return default
    return int(np.argmax(a))


def _fallback(default, label: str, what: str):
    if default is _MISSING:
        raise ValueError(f"cannot take {what} of an empty array"
                         + (f" ({label})" if label else "")
                         + "; pass a default to make this scene survivable")
    _note(label or what, f"empty -> {default}")
    return default


def _note(what: str, why: str) -> None:
    print(f"[warn] {what}: {why}", flush=True)


def clamp_index(i: int, j: int, shape) -> tuple[int, int]:
    """Grid lookup for a spawn/cell coordinate that may fall outside the grid.

    H[i, j] with an unclamped spawn index is an IndexError that reads like a
    corrupt scene; clamping keeps the run alive and the caller can report the
    out-of-range value.
    """
    nz, nx = shape[-2], shape[-1]
    return (min(max(int(i), 0), nz - 1), min(max(int(j), 0), nx - 1))


def load_array(path, dtype, shape=None, *, label: str = "array", required: bool = True):
    """np.fromfile with the missing/size-mismatch cases named instead of raised.

    Returns None when the file is absent or empty and `required` is False, so a
    caller can downgrade a check to a warning rather than crash on it.
    """
    import numpy as np
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        if required:
            raise StepError(EMPTY_INPUT, f"{label} missing or empty: {p}",
                            returncode=3)
        _note(label, "missing")
        return None
    a = np.fromfile(p, dtype)
    if shape is not None:
        want = int(shape[0]) * int(shape[1]) if len(shape) > 1 else int(shape[0])
        if a.size != want:
            if required:
                raise StepError(
                    EMPTY_INPUT,
                    f"{label} holds {a.size} values, expected {want} for shape "
                    f"{tuple(shape)}: {p}\n  An upstream step wrote a grid for a "
                    f"different extent; re-run it rather than this one.",
                    returncode=3)
            _note(label, f"size {a.size} != {want}")
            return None
        a = a.reshape(shape)
    return a


# ---------------------------------------------------------------------------
# per-scene result report


def is_within_budget(n_bytes: int, limit_gb: float) -> bool:
    return n_bytes <= limit_gb * 1024**3


# ---------------------------------------------------------------------------
# per-scene result report
#
# The dashboard, the tests and the operator all need one answer: did this scene
# produce a usable world, and what had to give to get there.
# ---------------------------------------------------------------------------
COMPLETE = "complete"
COMPLETE_WITH_WARNINGS = "complete-with-warnings"
PARTIAL = "partial"
FAILED_STATUS = "failed"


class Report:
    """Accumulates step outcomes into work/<scene>/report.json."""

    def __init__(self, work: Path, scene: str) -> None:
        self.work = Path(work)
        self.scene = scene
        self.steps: list = []
        self.notes: list = []
        self.started = time.time()

    def step(self, name: str, status: str, *, kind: str = "", detail: str = "",
             secs: float = 0.0, attempts: int = 1, fallbacks=()) -> dict:
        rec = {"name": name, "status": status, "secs": round(secs, 1),
               "attempts": attempts}
        if kind:
            rec["kind"] = kind
        if detail:
            rec["detail"] = detail[:2000]
        if fallbacks:
            rec["fallbacks"] = list(fallbacks)
        self.steps.append(rec)
        return rec

    def note(self, text: str) -> None:
        self.notes.append(text)
        print(f"[note] {text}", flush=True)

    @property
    def status(self) -> str:
        ran = {s["name"]: s["status"] for s in self.steps}
        if any(v == "failed" for v in ran.values()):
            return PARTIAL if any(v in ("done", "skipped", "warning")
                                  for v in ran.values()) else FAILED_STATUS
        if any(v == "warning" for v in ran.values()) or self.notes:
            return COMPLETE_WITH_WARNINGS
        return COMPLETE

    @property
    def produced(self) -> bool:
        """Did a viewable world actually come out of this run?"""
        asset = self.work / "viewer_assets"
        return (asset / "scene.ply").exists() and (asset / "heights.f32").exists()

    def write(self) -> Path:
        payload = {
            "scene": self.scene,
            "status": self.status,
            "produced_assets": self.produced,
            "secs": round(time.time() - self.started, 1),
            "steps": self.steps,
            "notes": self.notes,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            return write_json(self.work / "report.json", payload)
        except OSError:
            return self.work / "report.json"


def human_summary(report: Report) -> str:
    """The part an operator reads after a long run: what shipped, what gave up."""
    lines = [f"{report.scene}: {report.status.upper()}"
             + ("" if report.produced else "  (no world produced)")]
    width = max([len(s["name"]) for s in report.steps] or [4])
    for s in report.steps:
        extra = ""
        if s.get("attempts", 1) > 1:
            extra += f"  x{s['attempts']}"
        for f in s.get("fallbacks", []):
            extra += f"  [{f}]"
        if s["status"] == "failed":
            first = str(s.get("detail", "")).splitlines()
            extra += f"  {s.get('kind', '')} {first[0] if first else ''}".rstrip()
        mark = {"done": "ok  ", "skipped": "skip", "warning": "WARN",
                "failed": "FAIL", "running": "...."}.get(s["status"], s["status"])
        lines.append(f"  [{mark}] {s['name']:<{width}} {s['secs']:>6.0f}s{extra}")
    for n in report.notes:
        lines.append(f"  note: {n}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# tool resolution
# ---------------------------------------------------------------------------
def find_first(candidates) -> Path | None:
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def node_tool_argv(pkg: str, local: Path | None = None) -> list:
    """Prefer a vendored install over `npx -y`, which needs the registry.

    The collider step ran `npx -y @playcanvas/splat-transform` on every build,
    so a flaky network or a cold npm cache failed a scene that had already cost
    20 minutes of GPU time.
    """
    if local and Path(local).exists():
        return [str(local), *([] if True else []), pkg]
    return ["npx", "-y", pkg]


def warn(msg: str) -> None:
    print(f"[warn] {msg}", flush=True)


def die(kind: str, msg: str, code: int = 1):
    """Exit in a way that names the problem class for the parent runner."""
    raise StepError(kind, msg, returncode=code)


def configure_streams() -> None:
    """utf-8 on both venvs; a degree sign in a print used to kill a run on cp1252.

    Line buffering matters as much as the encoding: redirected into a log file,
    stdout is block-buffered, so a step that dies mid-way looks like it never
    said anything, and a long quiet step looks like a hang.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace",
                               line_buffering=True)
        except (AttributeError, OSError, ValueError):
            pass


# Runs on import, not on request. The scripts here print em dashes, degree signs
# and set-membership prose they cannot avoid, and on a cp1252 console that is a
# UnicodeEncodeError halfway through a 7-minute step - the exact "random crash
# for no reason" this module exists to remove. Explicit calls stay valid;
# reconfigure is idempotent.
configure_streams()
