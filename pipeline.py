"""One-command pipeline: phone/drone video -> trained splat -> gated walkable scene.

V2: multi-clip scenes, capture presets + smart defaults, AR pose priors.

  python pipeline.py run room                     # videos/room/*.mp4 (any count)
  python pipeline.py run temple                   # legacy videos/temple.mp4
  python pipeline.py run room --preset auto       # diagnose footage, tune params
  python pipeline.py run room --video a.mp4 b.mp4 --poses a_poses.jsonl b_poses.csv
  python pipeline.py scan room                    # capture diagnostics only
  python pipeline.py doctor                       # toolchain health check
  python pipeline.py capture room                 # print capture checklist
  python pipeline.py status|view room

Presets set SfM/training parameters per capture style; every value remains
overridable (--target, --width, --steps, --cap, --voxel, --init-tri-angle,
--overlap, ...). The "auto" preset runs a quick motion/blur diagnostic pass
first and picks parameters from what your footage actually did.

Every step appends to work/<name>/logs/<nn>-<step>.log. A step is skipped only
when its command completed successfully, its declared inputs have not changed
since, and nothing upstream was re-run this session. Re-running a step
invalidates everything downstream.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PY310 = ROOT / ".venv310" / "Scripts" / "python.exe"
VIDEOS = ROOT / "videos"

VIDEO_EXTS = (".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv")
POSE_EXTS = (".jsonl", ".csv")

QUALITY = {
    # RTX 3050 6GB measured: 265k gaussians @ 640px peaked at 0.72 GiB.
    "standard": dict(target=None, width=640, steps=12000, cap=350_000, voxel="0.35"),
    # high: 1280px + 30k steps + antialiased rasterization + blur-aware sampling;
    # ~3M gaussians peak ~4.0-4.5 GiB on the 6GB card (expandable_segments on).
    "high": dict(target=None, width=1280, steps=15000, cap=3_000_000, voxel="0.25"),
    # ultra: for 8GB+ GPUs or short clips; 1440px, 45k steps, 4M cap.
    "ultra": dict(target=None, width=1440, steps=45000, cap=4_000_000, voxel="0.25"),
}

# Capture-style presets. None = let smart defaults / other layers decide.
PRESETS = {
    "auto": {},
    "room": dict(
        label="Handheld Indoor Room / Apartment",
        target=500, overlap=20,
        sift_peak_threshold=0.002,
        sift_edge_threshold=16,
        cross_clip="auto", loop_detection=True, prior_std=0.15,
        advice="Move in smooth arcs/orbits. Make three passes: waist-height, tilted up (ceiling), tilted down (floor)."),
    "indoor_large": dict(
        label="Large Indoor (Offices, Halls, Warehouses)",
        target=650, overlap=25,
        sift_peak_threshold=0.003,
        sift_edge_threshold=15,
        cross_clip="spatial", loop_detection=True, prior_std=0.25,
        advice="Walk serpentine grid paths with cross-ties every 10 meters to prevent drift across large rooms."),
    "outdoor_building": dict(
        label="Outdoor Building / House Facade",
        target=600, overlap=20,
        sift_peak_threshold=0.004,
        sift_edge_threshold=14,
        cross_clip="spatial", loop_detection=True, prior_std=0.5,
        advice="Circle the structure at multiple elevations (ground, mid-height, roof line) facing toward center."),
    "drone": dict(
        label="Aerial Drone Orbit / Push-Forward",
        target=400, overlap=15,
        sift_peak_threshold=0.004,
        sift_edge_threshold=12,
        cross_clip="auto", loop_detection=True, prior_std=1.0,
        advice="Fly continuous orbits at constant speed and radius with 60-70% overlap between adjacent frames."),
    "drone_mapping": dict(
        label="Drone Nadir & Terrain Mapping",
        target=500, overlap=15,
        sift_peak_threshold=0.005,
        sift_edge_threshold=10,
        cross_clip="spatial", loop_detection=True, prior_std=1.5,
        advice="Fly lawnmower grid pattern with 75% forward and 65% side overlap at constant altitude."),
    "object": dict(
        label="Close-Up Object / Turntable 360°",
        target=300, overlap=15,
        sift_peak_threshold=0.003,
        sift_edge_threshold=15,
        cross_clip="exhaustive", loop_detection=False, prior_std=0.05,
        advice="Circle the object twice at 45° and 15° elevation angles. Keep the subject centered."),
    "sky_heavy": dict(
        label="Outdoor with Bright Open Sky",
        target=450, overlap=20,
        sift_peak_threshold=0.004,
        sift_edge_threshold=12,
        cross_clip="auto", loop_detection=True, prior_std=0.8,
        advice="Angle camera slightly below horizontal to maximize ground feature density and avoid overexposed sky."),
    "corridor": dict(
        label="Linear Street / Long Hallway Path",
        target=600, overlap=30,
        sift_peak_threshold=0.003,
        sift_edge_threshold=16,
        cross_clip="auto", loop_detection=True, prior_std=0.3,
        advice="Walk slowly in forward straight lines; turn around at the end and walk back along opposite wall for loop closure."),
}

GPU_ENV = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


# --------------------------------------------------------------------------- 
# scene / source resolution
# ---------------------------------------------------------------------------
def scene_dir(name: str) -> Path:
    return VIDEOS / name


def resolve_sources(name: str, video_args, pose_args) -> dict:
    """Returns {'videos': [Path], 'poses': {clip: Path}, 'frames_dirs': {clip: Path}}."""
    sdir = scene_dir(name)
    if video_args:
        videos = []
        for v in video_args:
            p = Path(v) if Path(v).is_absolute() else ROOT / v
            if not p.exists():
                sys.exit(f"video not found: {p}")
            videos.append(p)
    elif sdir.is_dir():
        all_vids = [p for p in sdir.iterdir() if p.suffix.lower() in VIDEO_EXTS]
        by_stem = {}
        for p in all_vids:
            stem = p.stem
            if stem not in by_stem or (p.suffix.lower() == ".mp4" and by_stem[stem].suffix.lower() == ".webm"):
                by_stem[stem] = p
        videos = sorted(by_stem.values())
    else:
        legacy = VIDEOS / f"{name}.mp4"
        videos = [legacy] if legacy.exists() else []

    poses, frames_dirs = {}, {}
    stems = [v.stem for v in videos]

    def pair_clip(path: Path) -> str:
        hit = next((s for s in stems if path.stem.startswith(s)), None)
        if hit is None:
            sys.exit(f"pose log '{path.name}' matches no clip stem in {stems}; "
                     "use CLIP=path form")
        return hit

    if pose_args:
        for spec in pose_args:
            p = Path(spec)
            if "=" in spec:
                clip, _, tail = spec.partition("=")
                p = Path(tail) if Path(tail).is_absolute() else ROOT / tail
            else:
                p = p if p.is_absolute() else ROOT / p
                clip = pair_clip(p)
            if not p.exists():
                sys.exit(f"pose log not found: {p}")
            poses[clip] = p
    elif sdir.is_dir():
        stems = {v.stem for v in videos}
        for f in sorted(sdir.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in POSE_EXTS:
                continue
            hit = next((s for s in stems if f.stem.startswith(s)), None)
            if hit:
                poses[hit] = poses.get(hit) or f
        # Record3D exports: videos/<scene>/<clip>/rgbd + metadata
        for d in sorted(p for p in sdir.iterdir() if p.is_dir()):
            rgb = d / "rgbd"
            meta = d / "metadata"
            if rgb.is_dir() and meta.exists():
                frames_dirs[d.name] = rgb
                poses[d.name] = meta

    return {"videos": videos, "poses": poses, "frames_dirs": frames_dirs}


def sources_fingerprint(sources: dict) -> list:
    fp = []
    for v in sources["videos"]:
        st = v.stat()
        fp.append([str(v), st.st_size, int(st.st_mtime)])
    for c, p in sources["poses"].items():
        st = p.stat()
        fp.append([c, str(p), st.st_size])
    return fp


def diag_uptodate(work: Path, sources: dict) -> bool:
    f = work / "diagnostics.json"
    if not f.exists():
        return False
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return d.get("fingerprint") == sources_fingerprint(sources)


def run_diagnostics(sources: dict, work: Path) -> dict | None:
    vids = sources["videos"]
    if not vids:
        return None
    sys.path.insert(0, str(ROOT / "scripts"))
    from capture_diagnostics import probe_video
    reports = []
    for v in vids:
        print(f"[scan] probing {v.name} ...", flush=True)
        reports.append(probe_video(v))
    out = {"clips": reports, "fingerprint": sources_fingerprint(sources)}
    work.mkdir(parents=True, exist_ok=True)
    (work / "diagnostics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- 
# config resolution: preset <- smart <- quality <- cli overrides
# ---------------------------------------------------------------------------
SMART_RULES = [
    # (condition(diag_aggregate), param, value, why)
    (lambda a: a["weak_geometry_pct"] > 40,
     "overlap", 30, "weak pairwise geometry - widen matching window"),
    (lambda a: a["blur_p25"] is not None and a["blur_p25"] < 30,
     "max_image_size", 1600, "blurry footage - no benefit beyond 1600px SIFT"),
]


def aggregate_diag(diag: dict | None) -> dict:
    if not diag:
        return {}
    clips = diag["clips"]
    n = max(len(clips), 1)
    blur_p25s = [c.get("sharpness", {}).get("p25") for c in clips]
    blur_p25s = [b for b in blur_p25s if b is not None]
    return {
        "rotation_dominant_pct": round(sum(c.get("rotation_dominant_pct", 0) for c in clips) / n),
        "weak_geometry_pct": round(sum(c.get("weak_geometry_pct", 100) for c in clips) / n),
        "blur_p25": min(blur_p25s) if blur_p25s else None,
        "warnings": [w for c in clips for w in c.get("warnings", [])],
        "styles": [c.get("style") for c in clips],
    }


def build_config(args, sources: dict, allow_auto_diag: bool = True) -> dict:
    q = dict(QUALITY[args.quality])
    preset_name = args.preset
    diag = None
    agg = {}
    if preset_name == "auto":
        work_probe = ROOT / "work" / args.name
        if diag_uptodate(work_probe, sources):
            diag = json.loads((work_probe / "diagnostics.json").read_text(encoding="utf-8"))
            print("[auto] using cached diagnostics")
        elif allow_auto_diag:
            diag = run_diagnostics(sources, work_probe)
        else:
            print("[auto] diagnostics skipped for this command")
        agg = aggregate_diag(diag)
        # pick closest preset by dominant style
        styles = agg.get("styles") or []
        if any("orbit" in (s or "") for s in styles):
            preset_name = "object" if "static" in styles else "drone"
        elif any("rotation" in (s or "") or "low_texture" in (s or "") for s in styles):
            preset_name = "room"
        elif styles and all("translation_sweep" == s for s in styles):
            preset_name = "drone"
        else:
            preset_name = "room"
        print(f"[auto] capture style -> preset '{preset_name}' "
              f"(styles: {sorted(set(styles))})")

    cfg_vals = dict(PRESETS[preset_name])
    cfg_vals.pop("label", None)
    cfg_vals.pop("advice", None)

    # smart rules on top
    smart_notes = []
    for cond, key, val, why in SMART_RULES:
        try:
            if cond(agg):
                if key in cfg_vals and cfg_vals[key] != val:
                    smart_notes.append(f"{key}={val} ({why})")
                cfg_vals[key] = val
        except (KeyError, TypeError):
            pass

    # quality layer fills Nones
    for k, v in q.items():
        if v is None:
            q[k] = {"target": PRESETS[preset_name].get("target", 400)}.get(k, v)
        cfg_vals.setdefault(k, v)
    for k, v in q.items():
        if v is not None:
            cfg_vals[k] = v

    # explicit CLI overrides win over everything
    for k in ("target", "width", "steps", "cap", "voxel"):
        v = getattr(args, k, None)
        if v is not None:
            cfg_vals[k] = v
    for k in ("init_min_tri_angle", "overlap", "prior_std", "cross_clip", "vocab_tree"):
        v = getattr(args, k, None)
        if v is not None:
            cfg_vals[k] = v
    cfg_vals.setdefault("mapper", "auto")
    if getattr(args, "mapper", None):
        cfg_vals["mapper"] = args.mapper

    if smart_notes:
        print("[smart] adjusted: " + "; ".join(smart_notes))
    for w in agg.get("warnings", [])[:6]:
        print(f"[capture-warning] {w}")

    cfg = dict(cmd=args.cmd, name=args.name, work=ROOT / "work" / args.name,
               variant=args.variant, preset=preset_name,
               sources=sources, **cfg_vals)
    if preset_name == "auto" and diag is not None:
        cfg["_diag_path"] = ROOT / "work" / args.name / "diagnostics.json"
    return cfg


# --------------------------------------------------------------------------- 
# steps
# ---------------------------------------------------------------------------
def build_steps(cfg: dict) -> list[dict]:
    name, work, asset = cfg["name"], cfg["work"], cfg["work"] / "viewer_assets"
    src = cfg["sources"]
    steps = []

    kf_argv = [PY, ROOT / "scripts/extract_keyframes.py", "--work", work,
               "--target", cfg["target"], "--train-width", cfg["width"]]
    kf_inputs = []
    for v in src["videos"]:
        kf_argv += ["--video", v]
        kf_inputs.append(v)
    for clip, rgbdir in src["frames_dirs"].items():
        kf_argv += ["--frames-dir", f"{clip}={rgbdir}"]
    if cfg.get("_diag_path"):
        kf_argv += ["--diagnostics", cfg["_diag_path"]]
    steps.append(dict(
        name="keyframes", py=PY, argv=kf_argv, inputs=kf_inputs,
        clean=[work / "frames_train", work / "frames_full", work / "frames_undist"],
        outputs=[work / "keyframes.jsonl"]))

    if src["poses"]:
        pr_argv = [PY, ROOT / "scripts/import_phone_poses.py", "--work", work,
                   "--std", str(cfg.get("prior_std", 0.15))]
        for clip, log in src["poses"].items():
            pr_argv += ["--log", f"{clip}={log}"]
        steps.append(dict(
            name="priors", py=PY, argv=pr_argv,
            inputs=[work / "keyframes.jsonl"] + list(src["poses"].values()),
            outputs=[work / "pose_priors.jsonl"]))

    plan = {k: cfg[k] for k in (
        "camera_model", "per_folder_camera", "max_image_size", "max_features",
        "overlap", "quadratic_overlap", "loop_detection", "cross_clip",
        "exhaustive_max", "mapper", "prior_std", "init_min_tri_angle")
        if k in cfg}
    # fill the rest from the runner's defaults so the hash covers everything
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_colmap import DEFAULT_PLAN
    full_plan = {**DEFAULT_PLAN, **plan}
    digest = hashlib.sha1(json.dumps(full_plan, sort_keys=True).encode()).hexdigest()[:10]
    plan_path = work / "plan.json"
    col_inputs = [work / "keyframes.jsonl"]
    if (work / "pose_priors.jsonl").exists():
        col_inputs.append(work / "pose_priors.jsonl")
    steps.append(dict(
        name="colmap", py=PY,
        argv=[PY, ROOT / "scripts/run_colmap.py", work,
              "--plan", plan_path, "--plan-hash", digest],
        inputs=col_inputs,
        outputs=[work / "colmap" / "sparse" / "txt" / "images.txt"],
        pre=lambda: write_plan(plan_path, full_plan, digest)))

    steps += [
        dict(name="poses", py=PY,
             argv=[PY, ROOT / "scripts/parse_colmap.py", "--work", work],
             outputs=[work / "keyframes_poses.jsonl"]),
        dict(name="train", py=PY310, env=GPU_ENV,
             argv=[PY310, ROOT / "scripts/train_splat.py", "--work", work,
                   "--steps", cfg["steps"], "--cap", cfg["cap"],
                   "--refine-stop", int(cfg["steps"] * 0.75)],
             outputs=[work / "splat.ply"]),
        dict(name="frame", py=PY,
             argv=[PY, ROOT / "scripts/solve_frame.py", "--work", work,
                   "--ply", work / "splat.ply"],
             outputs=[work / "frame.json"]),
        dict(name="export", py=PY,
             argv=[PY, ROOT / "scripts/export_viewer_assets.py", "--work", work,
                   "--ply", work / "splat.ply"],
             outputs=[asset / "scene.ply", asset / "heights.f32"]),
        dict(name="sky", py=PY,
             argv=[PY, ROOT / "scripts/strip_sky.py", "--asset", asset], outputs=[]),
        dict(name="clouds", py=PY,
             argv=[PY, ROOT / "scripts/strip_clouds.py", "--asset", asset], outputs=[]),
        dict(name="reexport", py=PY,
             argv=[PY, ROOT / "scripts/export_viewer_assets.py", "--work", work,
                   "--ply", asset / "scene.ply", "--from-scene"], outputs=[]),
        dict(name="colors", py=PY,
             argv=[PY, ROOT / "scripts/export_ground_colors.py", "--work", work],
             outputs=[asset / "ground_colors.rgb"]),
        dict(name="collider", py=PY,
             argv=[PY, ROOT / "scripts/build_collider.py", "--work", work,
                   "--variant", cfg["variant"], "--voxel-size", cfg["voxel"]],
             outputs=[work / "pc" / "collision.collision.glb"]),
        dict(name="route", py=PY,
             argv=[PY, ROOT / "scripts/walk_path_from_glb.py", "--asset", asset,
                   "--glb", work / "pc/collision.collision.glb",
                   "--smooth", 3 if cfg.get("preset") in ("room", "object") else 1,
                   "--pick", "largest" if cfg.get("preset") in ("room", "object") else "best"],
             outputs=[]),
        dict(name="gate", py=PY,
             argv=[PY, ROOT / "scripts/check_world.py", "--asset", asset,
                   "--work", work], outputs=[]),
        dict(name="evals", py=PY310, env=GPU_ENV,
             argv=[PY310, ROOT / "scripts/render_evals_offline.py", "--work", work],
             outputs=[work / "eval_renders"]),
        dict(name="pairs", py=PY,
             argv=[PY, ROOT / "scripts/make_pairs.py",
                   "--real-dir", work / "frames_full",
                   "--render-dir", work / "eval_renders",
                   "--pairs", work / "eval_pairs.json",
                   "--out", ROOT / "results", "--tag", name], outputs=[]),
        dict(name="walktest", py=PY,
             argv=[PY, ROOT / "scripts/drive_viewer.py", "walk",
                   "--asset", asset, "--out", work / "walktest"], outputs=[]),
    ]
    return steps


def write_plan(path: Path, plan: dict, digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(plan)
    payload["plan_hash"] = digest
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# step execution (unchanged semantics + input freshness)
# ---------------------------------------------------------------------------
def marker_file(work: Path, step: dict):
    return work / ".pipeline" / f"{step['name']}.json"


def step_uptodate(step: dict, work: Path) -> tuple[bool, str]:
    mk = marker_file(work, step)
    if not mk.exists():
        return False, "no marker"
    try:
        saved = json.loads(mk.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "corrupt marker"
    now = [str(a) for a in step["argv"]]

    def norm(xs):
        # ignore the interpreter prefix (absolute venv path may move)
        return xs[1:] if xs and xs[0].endswith(("python.exe", "python")) else xs

    if norm(saved.get("argv", [])) != norm(now):
        return False, "command changed"
    missing = [o for o in step.get("outputs", []) if not o.exists()]
    if missing:
        return False, f"missing outputs: {missing[0]}"
    newest_input = 0.0
    for i in step.get("inputs", []):
        if i.exists():
            newest_input = max(newest_input, i.stat().st_mtime)
    if newest_input and newest_input > mk.stat().st_mtime:
        return False, "inputs changed since last run"
    return True, "done"


def run_step(step: dict, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    if step.get("pre"):
        step["pre"]()
    argv = [str(a) for a in step["argv"]]
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"\n$ {' '.join(argv)}\n")
        lf.flush()
        env = {**os.environ, **step.get("env", {})}
        t0 = time.time()
        r = subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT, env=env)
        dt = time.time() - t0
        lf.write(f"[exit {r.returncode}] {dt:.0f}s\n")
    tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-6:]
    for line in tail:
        print(f"    {line}")
    if r.returncode != 0:
        print(f"\nFAILED: {step['name']} (exit {r.returncode}) — full log: {log}")
        sys.exit(1)
    mk = marker_file(Path(log.parent.parent), step)
    mk.parent.mkdir(parents=True, exist_ok=True)
    mk.write_text(json.dumps({"argv": argv, "exit": 0, "secs": round(dt, 1)}),
                  encoding="utf-8")


def do_run(cfg: dict) -> None:
    steps = build_steps(cfg)
    dirty = bool(cfg["fresh"])
    from_idx = next((i for i, s in enumerate(steps) if s["name"] == cfg["from_step"]), None)
    times = {}
    for i, step in enumerate(steps, 1):
        ok, why = step_uptodate(step, cfg["work"])
        forced = (step["name"] in cfg["only"]) or (from_idx is not None and i - 1 >= from_idx)
        skip = ok and not dirty and not forced and not cfg["only"]
        tag = "SKIP (done)" if skip else f"RUN ({why})" if ok else "RUN"
        print(f"[{i:02d}/{len(steps)}] {step['name']}: {tag}")
        if skip:
            continue
        dirty = True
        for d in step.get("clean", []):
            if d.is_dir():
                shutil.rmtree(d)
            elif d.exists():
                d.unlink()
        if step["name"] == "export" and (cfg["work"] / "viewer_assets" / "scene.full.ply").exists():
            (cfg["work"] / "viewer_assets" / "scene.full.ply").unlink()
        log = cfg["work"] / "logs" / f"{i:02d}-{step['name']}.log"
        t0 = time.time()
        run_step(step, log)
        times[step["name"]] = time.time() - t0

    print("\n=== pipeline summary ===")
    for k, v in times.items():
        print(f"  {k:<10} {v:6.0f}s")
    total = sum(times.values())
    print(f"  total     {total:6.0f}s ({total / 60:.1f} min)")
    print(f"Evidence: results\\blinded\\, {cfg['work']}\\walktest\\, {cfg['work']}\\train_progress\\")
    print(f"View:     python pipeline.py view {cfg['name']}")


def do_status(cfg: dict) -> None:
    for i, step in enumerate(build_steps(cfg), 1):
        ok, why = step_uptodate(step, cfg["work"])
        print(f"[{i:02d}] {step['name']:<10} {'done' if ok else 'stale/missing'} ({why})")


def do_scan(cfg: dict) -> None:
    diag = run_diagnostics(cfg["sources"], cfg["work"]) if not diag_uptodate(
        cfg["work"], cfg["sources"]) else \
        json.loads((cfg["work"] / "diagnostics.json").read_text(encoding="utf-8"))
    agg = aggregate_diag(diag)
    for r in diag["clips"]:
        print(f"\n=== {r['clip']} ===  {r['duration_s']}s @ {r['fps']}fps  style={r['style']}")
        print(f"  sharpness {r['sharpness']}  ORB {r['orb_features']}")
        print(f"  rot-dominant {r['rotation_dominant_pct']}%  weak-pairs "
              f"{r['weak_geometry_pct']}%")
        for w in r["warnings"]:
            print(f"  ! {w}")
    if agg.get("warnings"):
        print("\n[verdict] fix these before burning GPU hours:")
        for w in dict.fromkeys(agg["warnings"]):
            print(f"  - {w}")
    else:
        print("\n[verdict] footage looks healthy - go ahead: "
              f"python pipeline.py run {cfg['name']}")


def do_doctor(_cfg: dict) -> None:
    checks = []

    def check(ok, label, fix=""):
        checks.append((ok, label, fix))

    ff = next(iter(sorted((ROOT / "tools").glob("**/ffmpeg.exe"))), None)
    check(bool(ff), "ffmpeg", "unzip tools/ffmpeg.zip into tools/")
    colmap = ROOT / "tools/colmap/bin/colmap.exe"
    check(colmap.exists(), "colmap binary", "unzip tools/colmap.zip into tools/")
    if colmap.exists():
        envk = {**os.environ, "PATH": f"{ROOT / 'tools/colmap/bin'};{os.environ.get('PATH', '')}"}
        out = subprocess.run([str(colmap), "-h"], capture_output=True, text=True,
                             env=envk).stdout
        ver = next((l for l in out.splitlines() if l.startswith("COLMAP")), "?")
        print(f"  colmap: {ver}")
        for cmd in ("pose_prior_mapper", "global_mapper", "spatial_matcher",
                    "vocab_tree_matcher", "model_aligner"):
            check(cmd in out, f"colmap {cmd}", "update tools/colmap.zip")
    try:
        import pycolmap  # noqa
        cs = int(pycolmap.PosePriorCoordinateSystem.CARTESIAN)
        check(True, f"pycolmap {pycolmap.__version__} (CARTESIAN={cs})")
        assert cs == 1, "run_colmap.py CARTESIAN constant mismatch!"
    except ImportError:
        check(False, "pycolmap", "pip install pycolmap==4.1.1 (optional; sqlite fallback used)")
    except AssertionError as e:
        check(False, "pycolmap enum", str(e))
    try:
        import cv2  # noqa
        check(True, f"opencv {cv2.__version__}")
    except ImportError:
        check(False, "opencv", "pip install opencv-python")
    vt = ROOT / "tools/vocab_tree.bin"
    if vt.exists():
        check(True, f"vocab tree ({vt.stat().st_size // 1024 // 1024}MB)")
    else:
        check(None, "vocab tree missing - loop detection disabled",
              "download https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin -> tools/vocab_tree.bin")
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True)
        check(q.returncode == 0, f"gpu: {q.stdout.strip() or 'none'}")
    except FileNotFoundError:
        check(None, "nvidia-smi not found (CPU-only?)")

    print()
    bad = 0
    for ok, label, fix in checks:
        mark = "PASS" if ok is True else "WARN" if ok is None else "FAIL"
        bad += ok is False
        print(f"  [{mark}] {label}" + (f"\n         fix: {fix}" if ok is not True else ""))
    if bad:
        sys.exit(f"\ndoctor: {bad} hard failure(s)")
    print("\ndoctor: all good")


CAPTURE_GUIDE = """\
capture checklist ({label})
  1. NEVER spin in place - rotation without translation cannot be reconstructed.
     Step sideways / walk arcs so every pan segment has parallax.
  2. Slow, steady speed; lock exposure/focus; bright even light.
  3. Overlap >= 60-70% between consecutive views; every surface in >= 3 views.
  4. Multiple angles welcome: record several clips, put them all in
     videos/{name}/ - the pipeline links them automatically.
{advice}
  5. Have ARCore/ARKit logging? Drop the per-clip pose log (jsonl/csv) next to
     each clip (similar filename) and COLMAP gets metric position priors -
     see docs/PHONE_CAPTURE.md. Target ~100-400 keyframes total (auto).
"""


def do_capture(cfg: dict) -> None:
    p = PRESETS[cfg["preset"]]
    print(CAPTURE_GUIDE.format(label=p.get("label", cfg["preset"]),
                               advice=p.get("advice", "") + "\n" if p.get("advice") else "",
                               **{"name": cfg["name"]}))


def do_coverage(cfg: dict) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    subprocess.run([str(PY), str(ROOT / "scripts/check_coverage.py"), "--work", str(cfg["work"])])


def do_view(cfg: dict) -> None:
    asset = cfg["work"] / "viewer_assets"
    if not (asset / "scene.ply").exists():
        sys.exit(f"no scene at {asset} — run: python pipeline.py run {cfg['name']}")
    url = f"http://localhost:8137/viewer/pc.html?asset=/work/{cfg['name']}/viewer_assets"
    webbrowser.open(url)
    subprocess.run([str(PY), str(ROOT / "_serve.py"), "8137", str(ROOT)])


# --------------------------------------------------------------------------- 
def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_name=True):
        if need_name:
            p.add_argument("name")
        p.add_argument("--quality", choices=tuple(QUALITY), default="high")
        p.add_argument("--preset", choices=tuple(PRESETS), default="room",
                       help="capture style; 'auto' diagnoses your footage first")
        p.add_argument("--video", action="append", nargs="+", default=None,
                       help="clip(s); space-separated, repeatable; default: videos/<name>/*.mp4")
        p.add_argument("--poses", action="append", nargs="+", default=None,
                       help="CLIP=path per-clip AR pose log; repeatable")
        p.add_argument("--target", type=int, default=None)
        p.add_argument("--width", type=int, default=None)
        p.add_argument("--steps", type=int, default=None)
        p.add_argument("--cap", type=int, default=None)
        p.add_argument("--voxel", default=None)
        p.add_argument("--variant", default="cluster_shell")
        p.add_argument("--init-min-tri-angle", dest="init_min_tri_angle",
                       type=float, default=None)
        p.add_argument("--overlap", type=int, default=None)
        p.add_argument("--prior-std", dest="prior_std", type=float, default=None)
        p.add_argument("--cross-clip", dest="cross_clip", default=None,
                       choices=("auto", "spatial", "vocab", "exhaustive", "none"))
        p.add_argument("--mapper", default=None,
                       choices=("auto", "incremental", "pose_prior", "global"))
        p.add_argument("--vocab-tree", dest="vocab_tree", default=None)

    common(r := sub.add_parser("run", help="full pipeline"))
    g = r.add_mutually_exclusive_group()
    g.add_argument("--fresh", action="store_true")
    g.add_argument("--from", dest="from_step", metavar="STEP")
    r.add_argument("--only", metavar="STEPS")

    common(s := sub.add_parser("scan", help="diagnose footage, no reconstruction"))
    common(cov := sub.add_parser("coverage", help="analyze 3D multi-view coverage & camera frustums"))
    common(st := sub.add_parser("status", help="per-step completion state"))
    sub.add_parser("doctor", help="toolchain health check")
    capp = sub.add_parser("capture", help="print capture checklist for a preset")
    common(capp, need_name=False)
    capp.add_argument("name", nargs="?", default="_")
    common(v := sub.add_parser("view", help="serve + open the walkable viewer"))

    args = ap.parse_args()

    if args.cmd == "doctor":
        do_doctor({})
        return

    if args.cmd == "capture":
        cfg = dict(cmd=args.cmd, name=args.name or "_",
                   preset=getattr(args, "preset", "room"))
        do_capture(cfg)
        return

    sources = resolve_sources(args.name,
                              [v for grp in (args.video or []) for v in grp],
                              [p for grp in (args.poses or []) for p in grp])
    if args.cmd in ("run", "scan", "coverage") and not sources["videos"] and not sources["frames_dirs"]:
        sys.exit(f"no sources for '{args.name}': expected videos/{args.name}/*.mp4 "
                 f"or videos/{args.name}.mp4, or pass --video")

    cfg = build_config(args, sources,
                       allow_auto_diag=args.cmd in ("run", "scan"))
    if args.cmd == "view":
        do_view(cfg)
        return
    if args.cmd == "scan":
        do_scan(cfg)
        return
    if args.cmd == "coverage":
        do_coverage(cfg)
        return
    if args.cmd == "status":
        do_status(cfg)
        return
    if args.only:
        args.only = set(args.only.split(","))
    known = {s["name"] for s in build_steps(cfg)}
    if args.only and not args.only <= known:
        sys.exit(f"unknown steps: {args.only - known}; known: {sorted(known)}")
    if args.from_step and args.from_step not in known:
        sys.exit(f"unknown step {args.from_step}; known: {sorted(known)}")
    cfg["fresh"] = getattr(args, "fresh", False)
    cfg["from_step"] = getattr(args, "from_step", None)
    cfg["only"] = getattr(args, "only", None) or set()
    do_run(cfg)


if __name__ == "__main__":
    main()
