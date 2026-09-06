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
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PY310 = ROOT / ".venv310" / "Scripts" / "python.exe"
VIDEOS = ROOT / "videos"
sys.path.insert(0, str(ROOT / "scripts"))

import robust as rb  # noqa: E402

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
    # smoke: every step, on the real reconstruction, in minutes.
    #
    # "Skip the train step to save time" is not a test of this pipeline. frame,
    # export, collider, surface, gate, evals, pairs and walktest all consume
    # splat.ply, so skipping train leaves most of the graph unexercised and the
    # run passes vacuously. 300 steps on the actual solve keeps every step honest
    # -- the splat is blurry, everything downstream is real.
    "smoke": dict(target=None, width=640, steps=300, cap=150_000, voxel="0.4"),
}

# Canopy culling: the strip_sky + strip_clouds pair and the re-export they force.
#
# This is a property of the footage, not a step every run owes. strip_clouds
# calls a gaussian fog when it is desaturated AND more than a metre above the
# local ground, judged only over the walkable footprint -- which is precisely a
# white painted ceiling in a room, at 2.5 m, spanning every column. Run indoors
# it deletes the one surface the `room` advice tells the operator to point the
# phone up at, and does it quietly: the step prints a percentage and the gate
# downstream only asks whether the FLOOR is a floor.
CULL_CANOPY = "canopy"
CULL_NONE = "none"

# Capture-style presets. None = let smart defaults / other layers decide.
PRESETS = {
    "auto": {},
    "room": dict(
        label="Handheld Indoor Room / Apartment",
        cull=CULL_NONE,
        target=500, overlap=20,
        sift_peak_threshold=0.002,
        sift_edge_threshold=16,
        cross_clip="auto", loop_detection=True, prior_std=0.15,
        # Tight multi-view support: 400+ cameras in a 3-5m room, a point 6m from
        # any camera (the default 4x AGL) is a floater not a wall. 2x AGL keeps
        # the walls but drops the floaters the triangulator put through them.
        max_range_mult=2.0, min_views=6,
        drop_backdrop=True,
        # Do not let camera_ground paint a floor across empty grid cells the
        # splat never observed. 0.6m reaches under the phone's own footprint
        # and stops there; the old 1.2m filled most of the room with an
        # imaginary surface.
        camera_ground=0.6,
        # A ceiling is trained as a large, obliquely-seen, low-opacity sheet:
        # at 0.15 it was being pruned out at export and the room lost its roof
        # (and every soft-focus detail the camera only grazed). 0.04 keeps the
        # semi-transparent surfaces the multi-view support already vouched for.
        prune_opacity=0.04,
        # A 2.5m-tall ceiling cannot take a 6m wall or a 3m skirt: the collider
        # becomes a box. These values hug the room floor.
        collider_wall=0.3, collider_skirt=0.15,
        # Room scans are boxes: the air gap between floor and ceiling is a
        # room's height, not "airborne crust". clip_collider's default gap of
        # 1.4 m was tuned to strip sky haze on outdoor captures; here it drops
        # every wall panel and the ceiling (95% of tris). Widen it past the
        # room's relief so the shell keeps its vertical surfaces.
        clip_gap=4.0,
        no_clip=False,
        # Heightfield cells and mesh smoothing are left at their defaults here
        # (auto ~8cm cells, 3-tap filter). The previous 15cm / 9-tap setting
        # made the ground so uniformly flat that the bed and every piece of
        # furniture lost their edges in the collider - which is exactly what
        # the user was complaining about ("can't identify the plain bed
        # surface"). Better a slightly sawtooth floor with real object shapes
        # than a plate.
        # Hamster: the room's true walkable footprint is 2-4m^2, a 1.75m
        # human cannot turn around. 0.15m matches what the room's own geometry
        # can host.
        character_height=0.15,
        # A room's walk loop is 3-5m by construction; the 15m outdoor threshold
        # fails a correct room. Coverage threshold also drops to 5% because a
        # room scan never sees through walls or under furniture.
        min_perimeter=3.0, min_coverage=0.05,
        advice="Move in smooth arcs/orbits. Make three passes: waist-height, tilted up (ceiling), tilted down (floor)."),
    "indoor_large": dict(
        label="Large Indoor (Offices, Halls, Warehouses)",
        cull=CULL_NONE,
        target=650, overlap=25,
        sift_peak_threshold=0.003,
        sift_edge_threshold=15,
        cross_clip="spatial", loop_detection=True, prior_std=0.25,
        advice="Walk serpentine grid paths with cross-ties every 10 meters to prevent drift across large rooms."),
    "outdoor_building": dict(
        label="Outdoor Building / House Facade",
        cull=CULL_CANOPY,
        target=600, overlap=20,
        sift_peak_threshold=0.004,
        sift_edge_threshold=14,
        cross_clip="spatial", loop_detection=True, prior_std=0.5,
        advice="Circle the structure at multiple elevations (ground, mid-height, roof line) facing toward center."),
    "drone": dict(
        label="Aerial Drone Orbit / Push-Forward",
        cull=CULL_CANOPY,
        target=400, overlap=15,
        sift_peak_threshold=0.004,
        sift_edge_threshold=12,
        cross_clip="auto", loop_detection=True, prior_std=1.0,
        advice="Fly continuous orbits at constant speed and radius with 60-70% overlap between adjacent frames."),
    "drone_mapping": dict(
        label="Drone Nadir & Terrain Mapping",
        cull=CULL_CANOPY,
        target=500, overlap=15,
        sift_peak_threshold=0.005,
        sift_edge_threshold=10,
        cross_clip="spatial", loop_detection=True, prior_std=1.5,
        advice="Fly lawnmower grid pattern with 75% forward and 65% side overlap at constant altitude."),
    "object": dict(
        label="Close-Up Object / Turntable 360°",
        cull=CULL_NONE,
        target=300, overlap=15,
        sift_peak_threshold=0.003,
        sift_edge_threshold=15,
        cross_clip="exhaustive", loop_detection=False, prior_std=0.05,
        advice="Circle the object twice at 45° and 15° elevation angles. Keep the subject centered."),
    "sky_heavy": dict(
        label="Outdoor with Bright Open Sky",
        cull=CULL_CANOPY,
        target=450, overlap=20,
        sift_peak_threshold=0.004,
        sift_edge_threshold=12,
        cross_clip="auto", loop_detection=True, prior_std=0.8,
        advice="Angle camera slightly below horizontal to maximize ground feature density and avoid overexposed sky."),
    "corridor": dict(
        label="Linear Street / Long Hallway Path",
        cull=CULL_NONE,
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


def pick_preset(styles, handheld: bool) -> str:
    """Choose a capture preset from how the footage was made, then how it moved.

    Order matters. An AR pose log (ARCore / ARKit / Record3D / the WebXR
    recorder) is a declaration that a handset made this footage - no drone emits
    one. Motion style alone cannot carry that decision: walking a circle around
    your own living room is an orbit, and routing it at the aerial presets turns
    the canopy cull on, which - per the note above CULL_CANOPY - reads a white
    painted ceiling as a cloud sea and deletes it, and drops the clip_gap that
    keeps a room's walls in the collider.
    """
    s = set(styles or [])
    orbit = any(x.startswith("orbit") for x in s)
    rotation_or_thin = any("rotation" in x or "low_texture" in x for x in s)
    sweep_only = bool(s) and s == {"translation_sweep"}
    if orbit or rotation_or_thin or sweep_only:
        return "room" if handheld else "drone"
    return "room"


def build_config(args, sources: dict, allow_auto_diag: bool = True) -> dict:
    q = dict(QUALITY[args.quality])
    preset_name = args.preset
    diag = None
    agg = {}
    narrate = args.cmd in ("run", "scan")
    if preset_name == "auto":
        work_probe = ROOT / "work" / args.name
        if diag_uptodate(work_probe, sources):
            diag = json.loads((work_probe / "diagnostics.json").read_text(encoding="utf-8"))
            if narrate:
                print("[auto] using cached diagnostics")
        elif allow_auto_diag:
            diag = run_diagnostics(sources, work_probe)
        elif narrate:
            print("[auto] diagnostics skipped for this command")
        agg = aggregate_diag(diag)
        handheld = bool(sources.get("poses")) or bool(sources.get("frames_dirs"))
        styles = agg.get("styles") or []
        preset_name = pick_preset(styles, handheld)
        if narrate:
            print(f"[auto] capture style -> preset '{preset_name}' "
                  f"(styles: {sorted(set(styles))}, "
                  f"{'handheld: AR pose logs present' if handheld else 'no pose logs'})")

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
    for k in ("target", "width", "steps", "cap", "voxel", "grow_grad"):
        v = getattr(args, k, None)
        if v is not None:
            cfg_vals[k] = v
    for k in ("init_min_tri_angle", "overlap", "prior_std", "cross_clip",
              "vocab_tree", "speed_anchor", "height_anchor"):
        v = getattr(args, k, None)
        if v is not None:
            cfg_vals[k] = v
    if getattr(args, "cull", "auto") != "auto":
        cfg_vals["cull"] = args.cull
    cfg_vals.setdefault("cull", CULL_CANOPY)
    cfg_vals.setdefault("mapper", "auto")
    if getattr(args, "mapper", None):
        cfg_vals["mapper"] = args.mapper
    # loop_detection=True is a promise three presets make; the retrieval stage
    # cannot run without a tree, so honour the one `doctor` says to put in tools/.
    if not cfg_vals.get("vocab_tree") and (ROOT / "tools/vocab_tree.bin").exists():
        cfg_vals["vocab_tree"] = str(ROOT / "tools/vocab_tree.bin")
        if args.cmd in ("run", "scan"):
            smart_notes.append("vocab_tree=tools/vocab_tree.bin (loop closure needs a "
                               "retrieval index; found one, so it will actually run)")

    if smart_notes:
        print("[smart] adjusted: " + "; ".join(smart_notes))
    for w in agg.get("warnings", [])[:6]:
        print(f"[capture-warning] {w}")

    cfg = dict(cmd=args.cmd, name=args.name, work=ROOT / "work" / args.name,
               variant=args.variant, preset=preset_name, quality=args.quality,
               timeout_scale=getattr(args, "timeout_scale", 1.0) or 1.0,
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
        "overlap", "quadratic_overlap", "loop_detection", "vocab_tree", "cross_clip",
        "exhaustive_max", "mapper", "prior_std", "init_min_tri_angle")
        if k in cfg}
    # calibration.json from the WebXR capture tool: seeds COLMAP with the real
    # focal length so it doesn't guess 1.2×width (45 % error on a phone lens).
    if src["videos"]:
        _cal = src["videos"][0].parent / "calibration.json"
        if _cal.exists():
            plan["calibration_json"] = str(_cal)
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

    frame_argv = [PY, ROOT / "scripts/solve_frame.py", "--work", work,
                  "--ply", work / "splat.ply"]
    for key, flag in (("speed_anchor", "--speed-anchor"),
                      ("height_anchor", "--height-anchor"),
                      ("max_range_mult", "--max-range-mult"),
                      ("min_views", "--min-views")):
        if cfg.get(key) is not None:
            frame_argv += [flag, cfg[key]]
    train_argv = [PY310, ROOT / "scripts/train_splat.py", "--work", work,
                  "--steps", cfg["steps"], "--cap", cfg["cap"],
                  "--refine-stop", int(cfg["steps"] * 0.75)]
    if cfg.get("grow_grad"):
        train_argv += ["--grow-grad", cfg["grow_grad"]]
    steps += [
        dict(name="poses", py=PY,
             argv=[PY, ROOT / "scripts/parse_colmap.py", "--work", work],
             outputs=[work / "keyframes_poses.jsonl"]),
        dict(name="train", py=PY310, env=GPU_ENV,
             argv=train_argv,
             outputs=[work / "splat.ply"]),
        dict(name="frame", py=PY,
             argv=frame_argv,
             inputs=[work / "splat.ply", work / "keyframes_poses.jsonl"]
                    + ([work / "pose_priors.jsonl"]
                       if (work / "pose_priors.jsonl").exists() else []),
             outputs=[work / "frame.json"]),
        dict(name="export", py=PY,
             argv=[PY, ROOT / "scripts/export_viewer_assets.py", "--work", work,
                   "--ply", work / "splat.ply",
                   *(["--camera-ground", cfg["camera_ground"]]
                     if cfg.get("camera_ground") is not None else []),
                   *(["--character-height", cfg["character_height"]]
                     if cfg.get("character_height") is not None else []),
                   *(["--prune-opacity", cfg["prune_opacity"]]
                     if cfg.get("prune_opacity") is not None else []),
                   *(["--cell-meters", cfg["cell_meters"]]
                     if cfg.get("cell_meters") is not None else []),
                   *(["--drop-backdrop"] if cfg.get("drop_backdrop") else [])],
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
                   "--variant", cfg["variant"], "--voxel-size", cfg["voxel"],
                   *(["--no-clip"] if cfg.get("no_clip") else []),
                   *(["--clip-gap", cfg["clip_gap"]]
                     if cfg.get("clip_gap") is not None else [])],
             outputs=[work / "pc" / "collision.collision.glb"]),
        dict(name="objects", py=PY,
             argv=[PY, ROOT / "scripts/build_objects.py", "--asset", asset],
             outputs=[asset / "objects.json"]),
        dict(name="surface", py=PY,
             # Builds BOTH candidate grounds (hf and shell), routes on each, ships
             # the one the autopilot walks further on, and writes ground.f32 so the
             # physics mesh, the route and the browser's underlay are literally one
             # array (the router used to plan on a smoothed surface while ammo hit the
             # raw one). Runs the router itself, so there is no route step.
             # --src is the clipped voxel shell written by build_collider; it is NOT
             # collision.collision.glb (which build_collider overwrites with the hf
             # ground mesh after this point). tune_collider needs the shell geometry
             # to do a meaningful A/B against the heightfield candidate.
             argv=[PY, ROOT / "scripts/tune_collider.py", "--work", work, "--asset", asset,
                   "--src", work / "pc" / "clipped.collision.glb",
                   "--smooth", 3 if cfg.get("preset") in ("room", "object") else 1,
                   *((["--mesh-smooth", cfg["mesh_smooth"]]
                      if cfg.get("mesh_smooth") is not None else [])),
                   "--pick", "largest" if cfg.get("preset") in ("room", "object") else "best",
                   *(["--wall", cfg["collider_wall"]]
                     if cfg.get("collider_wall") is not None else []),
                   *(["--skirt", cfg["collider_skirt"]]
                     if cfg.get("collider_skirt") is not None else [])],
             outputs=[asset / "ground.f32"]),
        dict(name="gate", py=PY,
             argv=[PY, ROOT / "scripts/check_world.py", "--asset", asset,
                   "--work", work,
                   *(["--min-coverage", cfg["min_coverage"]]
                     if cfg.get("min_coverage") is not None else []),
                   *(["--min-perimeter", cfg["min_perimeter"]]
                     if cfg.get("min_perimeter") is not None else [])],
             outputs=[]),
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
    if cfg.get("cull") != CULL_CANOPY:
        # `reexport` only exists because the cull changed what scene.ply holds,
        # so it goes with the pair rather than re-deriving the same assets.
        culled = {"sky", "clouds", "reexport"}
        steps = [s for s in steps if s["name"] not in culled]
    return steps


def write_plan(path: Path, plan: dict, digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(plan)
    payload["plan_hash"] = digest
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# step execution
#
# A child process failing used to be a bare sys.exit(1) in the middle of a
# multi-hour run, and the only way forward was to re-invoke by hand with a
# tweaked number. That is what looked like random failure. Every step now has a
# time budget, and the steps whose knob is known have a ladder of safer settings
# to fall back through.
# ---------------------------------------------------------------------------
TIMEOUTS = {
    # generous, scaled to what the step does on a 6GB card. A step that hangs is
    # worse than one that fails: it blocks the fifteen behind it.
    "keyframes": 45 * 60,
    "priors": 10 * 60,
    "colmap": 6 * 3600,        # matching + mapping on hundreds of frames
    "poses": 5 * 60,
    "train": 8 * 3600,
    "frame": 15 * 60,
    "export": 30 * 60,
    "sky": 20 * 60,
    "clouds": 20 * 60,
    "reexport": 30 * 60,
    "colors": 10 * 60,
    "collider": 40 * 60,
    "objects": 10 * 60,
    "surface": 30 * 60,
    "gate": 5 * 60,
    "evals": 30 * 60,
    "pairs": 10 * 60,
    "walktest": 30 * 60,
}


def _halve(v, floor):
    try:
        return max(floor, int(float(v) / 2))
    except (TypeError, ValueError):
        return floor


def _scale(v, factor, floor):
    try:
        return max(floor, round(float(v) * factor, 4))
    except (TypeError, ValueError):
        return floor


# Rungs per step: each is (what to tell the operator, {flag: new value}). A value
# of None drops the flag entirely. Applied in order, only on a failure whose
# class a smaller number can actually fix.
RETRIES = {
    "keyframes": [
        ("half the keyframes", {"--target": lambda cfg, s: _halve(cfg["target"], 60)}),
        ("quarter the keyframes, smaller frames",
         {"--target": lambda cfg, s: _halve(cfg["target"], 40),
          "--train-width": lambda cfg, s: _scale(cfg["width"], 0.75, 320)}),
    ],
    "train": [
        ("half the gaussians", {"--cap": lambda cfg, s: _halve(s["--cap"], 80_000)}),
        ("half the gaussians, half the steps",
         {"--cap": lambda cfg, s: _halve(s["--cap"] if "--cap" in s else cfg["cap"], 60_000),
          "--steps": lambda cfg, s: _halve(s["--steps"] if "--steps" in s else cfg["steps"], 300)}),
        ("no antialiasing, small cloud",
         {"--cap": lambda cfg, s: 100_000, "--no-antialias": True}),
    ],
    "collider": [
        ("coarser voxel grid", {"--voxel-size": lambda cfg, s: _scale(s["--voxel-size"], 2.0, 0.4)}),
    ],
    "export": [
        ("prune harder", {"--prune-opacity": lambda cfg, s: 0.2}),
    ],
    "surface": [
        ("less smoothing", {"--smooth": lambda cfg, s: 1}),
        ("no wall/skirt extrusion", {"--wall": None, "--skirt": None}),
    ],
    # colmap carries its own multi-rung rescue ladder in scripts/run_colmap.py,
    # because re-running it from here would redo a matching pass that already cost
    # the better part of an hour.
    "colmap": [],
}

# Failure classes a different number can fix. EMPTY_INPUT means an upstream step
# produced nothing, so retrying this one identically wastes an hour.
RETRYABLE = (rb.OOM, rb.VOXEL_OVERFLOW, rb.CRASH, rb.TIMEOUT, rb.FAILED)

# Steps whose only job is to turn finished assets into evidence. Nothing
# downstream reads their output, so one that dies on a locked screenshot must
# not also cost the walk test: the run records it and carries on. The scene's
# status is still partial because of the failed step.
ADVISORY = ("evals", "pairs")


def marker_file(work: Path, step: dict) -> Path:
    return work / ".pipeline" / f"{step['name']}.json"


def _norm(xs):
    # ignore the interpreter prefix (the absolute venv path may move)
    return xs[1:] if xs and str(xs[0]).endswith(("python.exe", "python")) else xs


def code_digest(argv) -> str:
    """Hash the source a step runs, so editing a script invalidates its marker.

    A marker used to compare the command, the outputs and the inputs' mtimes --
    never the code. An edit to scripts/walk_path_from_glb.py therefore left every
    downstream step "done" and the pipeline shipped a route planned by source
    that no longer exists: the same defect as a stale ground.f32 beside a rebuilt
    collider, one level up. Covers the .py files the command names plus
    robust.py, which every step script imports and which has changed a step's
    behaviour on its own more than once.
    """
    files = {(ROOT / "scripts" / "robust.py").resolve()}
    files |= {(ROOT / a).resolve() if not Path(str(a)).is_absolute()
              else Path(str(a)).resolve()
              for a in argv if str(a).endswith(".py")}
    h = hashlib.sha1()
    for f in sorted(files):
        h.update(f.name.encode("utf-8"))
        try:
            h.update(f.read_bytes())
        except OSError as e:
            # Unreadable is not "unchanged": fail to a digest that never matches.
            h.update(f"<unreadable {type(e).__name__}>".encode("utf-8"))
    return h.hexdigest()[:10]


def step_uptodate(step: dict, work: Path) -> tuple[bool, str]:
    mk = marker_file(work, step)
    if not mk.exists():
        return False, "no marker"
    saved = rb.read_json(mk)
    if not saved:
        return False, "corrupt marker"
    # Compare the BASE argv, not what was actually run: a step that succeeded on
    # a fallback rung is still done, and comparing the mutated command would make
    # every later invocation re-run the whole ladder.
    if _norm(saved.get("argv", [])) != _norm([str(a) for a in step["argv"]]):
        return False, "command changed"
    saved_code = saved.get("code")
    if saved_code != code_digest(step["argv"]):
        return False, ("step code changed since it ran" if saved_code
                       else "marker predates the code digest")
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


def flag_values(argv: list) -> dict:
    """Current value of each --flag in a command line, for rungs to build on."""
    out = {}
    for i, a in enumerate(argv):
        s = str(a)
        if s.startswith("--") and i + 1 < len(argv) \
                and not str(argv[i + 1]).startswith("--"):
            out[s] = argv[i + 1]
    return out


def _apply_rung(argv: list, rung: dict) -> list:
    """Rewrite one command line against a fallback rung."""
    present = flag_values(argv)
    out = list(argv)
    for flag, val in rung.items():
        if val is None:
            # drop the flag and its value
            if flag in present:
                _rm(out, present[flag])
                _rm(out, flag)
            elif flag in out:
                _rm(out, flag)
            continue
        if isinstance(val, bool):
            if val and flag not in out:
                out.append(flag)
            continue
        new = str(val)
        if flag in present:
            out[out.index(flag) + 1] = new
        else:
            out += [flag, new]
    return out


def _rm(lst: list, item) -> None:
    try:
        lst.remove(item)
    except ValueError:
        pass


def rescue_train(work: Path) -> tuple[bool, str]:
    """Reuse the newest training checkpoint after the trainer died.

    The kill that mattered was the one with no traceback: a card that fills on
    Windows can take the process outright, and no in-process except can catch
    that. The checkpoint is already written every --save-every steps, so the
    choice is between shipping a less-converged splat and shipping nothing.
    """
    cands = sorted((work / "train_progress").glob("splat*.ply"),
                   key=lambda p: p.stat().st_mtime, reverse=True) if \
        (work / "train_progress").is_dir() else []
    cands = [c for c in cands if c.stat().st_size > 0]
    if not cands:
        return False, "no checkpoint to rescue"
    best = cands[0]
    shutil.copyfile(best, work / "splat.ply")
    return True, f"trained on a partial cloud: {best.name} " \
                 f"({best.stat().st_size // 1024} KB) - lower PSNR than requested"


def run_step(step: dict, work: Path, log_dir: Path, idx: int,
             cfg: dict) -> dict:
    """Run one step, walking its fallback rungs. Never exits the process."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{idx:02d}-{step['name']}.log"
    base_argv = [str(a) for a in step["argv"]]
    if step.get("pre"):
        step["pre"]()
    env = {**os.environ, **step.get("env", {})}
    timeout = step.get("timeout", TIMEOUTS.get(step["name"], 3600)) \
        * float(cfg.get("timeout_scale", 1.0))
    plan = [(base_argv, [])]
    for label, mut in RETRIES.get(step["name"], []):
        prev = plan[-1][0]
        resolved = {f: (fn(cfg, flag_values(prev)) if callable(fn) else fn)
                    for f, fn in mut.items()}
        plan.append((_apply_rung(prev, resolved), plan[-1][1] + [label]))

    outcome = None
    # "w", not "a": run_step is only reached for a step that actually executes,
    # so this file should say what THIS run did. The rung ladder below still
    # accumulates inside one invocation, which is the history worth keeping.
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        for attempt, (argv, used) in enumerate(plan, 1):
            lf.write(f"\n$ {' '.join(argv)}\n")
            lf.flush()
            t0 = time.time()
            try:
                rb.run_cmd(argv, env=env, timeout=timeout, cwd=ROOT,
                           retries=0, log=lf)
                dt = time.time() - t0
                lf.write(f"[ok {attempt}] {dt:.0f}s\n")
                outcome = {"status": "done", "secs": dt, "attempts": attempt,
                           "fallbacks": used, "argv": argv}
                break
            except rb.StepError as e:
                dt = time.time() - t0
                lf.write(f"[exit {e.returncode} {e.kind}] {dt:.0f}s\n")
                lf.flush()
                for line in rb.tail_text(log, 6).splitlines():
                    print(f"    {line}")
                if e.kind not in RETRYABLE or attempt == len(plan):
                    outcome = {"status": "failed", "secs": dt, "attempts": attempt,
                               "kind": e.kind, "detail": str(e), "fallbacks": used,
                               "argv": argv}
                    break
                nxt = plan[attempt][1] or [f"rung {attempt + 1}"]
                print(f"  [{step['name']}] {e.kind} after {dt:.0f}s - retrying: "
                      f"{', '.join(nxt)}", flush=True)

    if outcome["status"] == "done":
        # The train rescue can also apply to a step that only half-finished, but
        # a success is a success; record the base argv so a fallback rung does
        # not make the step look stale on the next run.
        marker = {"argv": base_argv, "exit": 0, "secs": round(outcome["secs"], 1),
                  "effective": outcome["argv"], "fallbacks": outcome["fallbacks"],
                  "code": code_digest(base_argv)}
        marker_file(work, step).parent.mkdir(parents=True, exist_ok=True)
        rb.write_json(marker_file(work, step), marker)
    outcome["log"] = str(log)
    return outcome


def _gate_verdict(work: Path) -> dict:
    return rb.read_json(work / "viewer_assets" / "world_check.json", {}) or {}


def do_run(cfg: dict) -> int:
    steps = build_steps(cfg)
    work = cfg["work"]
    print(f"[{cfg['name']}] preset={cfg['preset']} cull={cfg['cull']} "
          f"quality={cfg['quality']} -> {len(steps)} steps")
    report = rb.Report(work, cfg["name"])
    log_dir = work / "logs"
    dirty = bool(cfg["fresh"])
    only = cfg["only"]
    from_idx = next((i for i, s in enumerate(steps)
                     if s["name"] == cfg["from_step"]), None)
    aborted = None

    for i, step in enumerate(steps, 1):
        name = step["name"]
        if only and name not in only:
            print(f"[{i:02d}/{len(steps)}] {name}: SKIP (not in --only)")
            report.step(name, "skipped")
            continue
        ok, why = step_uptodate(step, work)
        forced = bool(only) or (from_idx is not None and i - 1 >= from_idx)
        skip = ok and not dirty and not forced
        tag = "SKIP (done)" if skip else f"RUN ({why})" if ok else "RUN"
        print(f"[{i:02d}/{len(steps)}] {name}: {tag}")
        if skip:
            report.step(name, "skipped", detail=why)
            continue
        dirty = True
        for d in step.get("clean", []):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
            elif d.exists():
                try:
                    d.unlink()
                except PermissionError as e:
                    # Windows holds a file open if a viewer or an aborted run
                    # still has it; that must not stop a rebuild.
                    rb.warn(f"could not clear {d.name}: {e}")
        if name == "export" and (work / "viewer_assets" / "scene.full.ply").exists():
            (work / "viewer_assets" / "scene.full.ply").unlink(missing_ok=True)

        out = run_step(step, work, log_dir, i, cfg)
        if out["status"] == "done":
            report.step(name, "done", secs=out["secs"], attempts=out["attempts"],
                        fallbacks=out["fallbacks"])
            continue

        # ---- the step failed. Is there a usable result to carry on with? ----
        if name == "train" and not (work / "splat.ply").exists():
            rescued, note = rescue_train(work)
            if rescued:
                report.step(name, "warning", secs=out["secs"], kind=out["kind"],
                            detail=note, attempts=out["attempts"],
                            fallbacks=[note])
                report.note(f"train: {note}")
                continue
        if name == "gate":
            # check_world exits 0 unless something is structurally wrong, so any
            # non-zero here is a real defect; record its severity from the file.
            v = _gate_verdict(work)
            hard = v.get("hard_failures") or []
            if hard:
                # world_check.json names the rules that failed; the log tail does
                # not, and it was landing in the report as unexplained text.
                detail = "world gate: " + "; ".join(hard)
                report.step(name, "failed", secs=out["secs"], detail=detail,
                            kind="world-gate", fallbacks=[f"{len(hard)} hard"])
                # Deliberately not a break. evals, pairs and walktest are the
                # evidence for WHY this world is unwalkable, they only read the
                # assets, and the status is already carried by the failed step:
                # a take too degenerate to walk still gets an output and an
                # explanation instead of a run that stops and says nothing.
                print(f"\nWORLD GATE FAILED ({len(hard)} hard): " + "; ".join(hard))
                print(f"  shipping the evaluation renders and the walk test anyway "
                      f"- verdict: {work / 'world_check.json'}")
                continue
            report.step(name, "warning", secs=out["secs"],
                        kind=out.get("kind", "gate"))
            continue

        if name in ADVISORY:
            report.step(name, "failed", secs=out["secs"], kind=out.get("kind", ""),
                        detail=out.get("detail", ""), attempts=out.get("attempts", 1),
                        fallbacks=out.get("fallbacks", []))
            print(f"\n{name.upper()} FAILED ({out.get('kind', 'error')}) — the world "
                  f"is still shipped; continuing to the next evidence step. "
                  f"Full log: {out['log']}")
            if out.get("detail"):
                print("  " + out["detail"].replace("\n", "\n  "))
            continue

        report.step(name, "failed", secs=out["secs"], kind=out.get("kind", ""),
                    detail=out.get("detail", ""), attempts=out.get("attempts", 1),
                    fallbacks=out.get("fallbacks", []))
        print(f"\nFAILED: {name} ({out.get('kind', 'error')}) — full log: {out['log']}")
        if out.get("detail"):
            print("  " + out["detail"].replace("\n", "\n  "))
        aborted = name
        break

    verdict = _gate_verdict(work)
    if verdict.get("warnings"):
        report.note("world gate: " + ", ".join(verdict["warnings"]))
    report.write()
    print("\n=== pipeline summary ===")
    print(rb.human_summary(report))
    print(f"\nStatus: {report.status}")
    print(f"Report: {work / 'report.json'}")
    if report.produced:
        print(f"Evidence: results\\blinded\\, {work}\\walktest\\, "
              f"{work}\\train_progress\\")
        print(f"View:     python pipeline.py view {cfg['name']}")
    if aborted:
        print(f"Stopped at: {aborted}"
              + (f" (no world produced)" if not report.produced else
                 " (partial output on disk)"))
        return 0 if report.produced else 1
    return 0


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
        # The bytes cannot be trusted to say whether this build can read it -
        # flann and faiss trees look alike. A solved scene can: run_colmap drops
        # the tree and records why when the binary refuses it.
        rejected = sorted((ROOT / "work").glob("*/vocab_tree_skipped.json"),
                          key=lambda p: p.stat().st_mtime)
        if rejected:
            why = rb.read_json(rejected[-1], {}) or {}
            check(None, f"vocab tree REJECTED by this colmap build "
                        f"(last seen in {rejected[-1].parent.name})",
                  "COLMAP 4.x reads faiss indices only; this file is the legacy "
                  "flann format. Re-download "
                  "https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin "
                  "-> tools/vocab_tree.bin. The solve continues without vocab "
                  "cross-clip matching, which costs loop closures, not the scene. "
                  f"({str(why.get('reason', ''))[:120]})")
    else:
        check(None, "vocab tree missing - loop detection disabled",
              "download https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin -> tools/vocab_tree.bin")
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True)
        check(q.returncode == 0, f"gpu: {q.stdout.strip() or 'none'}")
    except FileNotFoundError:
        check(None, "nvidia-smi not found (CPU-only?)")

    # ---- the two interpreters this pipeline is split across ----
    # train runs on .venv310 and everything else on .venv. Discovering that a
    # venv is missing an hour into a run - after COLMAP finished - is the single
    # most expensive class of "random failure" doctor exists to prevent.
    check(PY310.exists(), f"train interpreter ({PY310.name})",
          f"create it: py -3.10 -m venv {PY310.parent.parent.name} then "
          "pip install torch gsplat numpy plyfile")
    if PY310.exists():
        try:
            t = subprocess.run([str(PY310), "-c",
                                "import torch,gsplat;print(torch.__version__,"
                                "torch.cuda.is_available())"],
                               capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as e:
            t = None
            check(False, f"train interpreter (.venv310): {e}",
                  "the venv exists but its python will not start")
        if t is not None:
            if t.returncode == 0:
                fields = (t.stdout.strip().splitlines() or [""])[-1].split()
                if len(fields) < 2:
                    check(False, "train stack probe returned an unreadable line",
                          repr(t.stdout.strip()[:200]))
                else:
                    ver, have_cuda = fields[0], fields[-1]
                    check(True, f"train stack: torch {ver}")
                    if have_cuda != "True":
                        check(None, "CUDA is not visible to .venv310",
                              "train will run on CPU and take hours - reinstall "
                              "torch with a cuda build")
            else:
                last = (t.stderr or "").strip().splitlines()
                check(False, "train stack (torch + gsplat in .venv310)",
                      last[-1] if last else
                      "pip install torch gsplat into .venv310")

    # ---- the collider's voxeliser: node plus the vendored package ----
    node = shutil.which("node")
    check(bool(node), "node on PATH (splat-transform is a Node CLI)",
          "install Node 18+ and re-run doctor")
    st = next(iter((ROOT / "tools" / "node_modules" / "@playcanvas")
                   .glob("splat-transform/package.json")), None) \
        if (ROOT / "tools" / "node_modules" / "@playcanvas").exists() else None
    check(bool(st), "splat-transform vendored in tools/",
          "npm install @playcanvas/splat-transform --prefix tools  "
          "(without it every collider build needs the network and npx)")

    # ---- does the flag probe work against this exact COLMAP build? ----
    # A flag this checkout passes but the vendored binary does not know is the
    # auditorium failure: the mapper exited 1 and the run had no idea why.
    if colmap.exists():
        known = rb.colmap_known_options(str(colmap), env=envk)
        check(len(known) > 50, f"colmap flag probe ({len(known)} options read)",
              "run_colmap.py cannot tell supported flags from unsupported ones - "
              "an unknown flag will fail the solve instead of being skipped")

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


def do_ui() -> None:
    """The whole product behind one command: serve the pages, open the dashboard.

    Nothing heavy starts here. The server is stdlib HTTP over the repo, presets
    are imported lazily by the one endpoint that needs them, and torch/gsplat
    live in the child process a run spawns - so opening the dashboard costs a
    directory listing and a browser tab.
    """
    url = "http://localhost:8137/viewer/pipeline_gui.html"
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8137)) == 0:
            print(f"[ui] a server already answers on 8137 — opening {url}")
            webbrowser.open(url)
            return
    proc = subprocess.Popen([str(PY), str(ROOT / "_serve.py"), "8137", str(ROOT)])
    for _ in range(50):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8137)) == 0:
                break
        time.sleep(0.1)
    print(f"[ui] dashboard: {url}")
    webbrowser.open(url)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


# --------------------------------------------------------------------------- 
def main() -> None:
    rb.configure_streams()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=False)

    def common(p, need_name=True):
        if need_name:
            p.add_argument("name")
        p.add_argument("--quality", choices=tuple(QUALITY), default="high",
                       help="'smoke' runs every step, including a real but short "
                            "train, in minutes - use it to verify a scene end to end")
        p.add_argument("--preset", choices=tuple(PRESETS), default="auto",
                       help="capture style; the default diagnoses your footage and "
                            "picks one, so nothing needs to be set by hand")
        p.add_argument("--timeout-scale", dest="timeout_scale", type=float,
                       default=1.0, metavar="FACTOR",
                       help="multiply every step's time budget; >1 for slower "
                            "hardware, <1 for a shakedown run")
        p.add_argument("--video", action="append", nargs="+", default=None,
                       help="clip(s); space-separated, repeatable; default: videos/<name>/*.mp4")
        p.add_argument("--poses", action="append", nargs="+", default=None,
                       help="CLIP=path per-clip AR pose log; repeatable")
        p.add_argument("--target", type=int, default=None)
        p.add_argument("--width", type=int, default=None)
        p.add_argument("--steps", type=int, default=None)
        p.add_argument("--cap", type=int, default=None)
        p.add_argument("--grow-grad", dest="grow_grad", type=float, default=None,
                       help="densification threshold — the number that actually decides "
                            "how many gaussians a scene gets. The auditorium plateaued at "
                            "247k with an 850k cap and 18k steps unused; 0.0002 is the "
                            "detail setting, at the cost of VRAM and browser sort time. "
                            "Default 0.0006")
        p.add_argument("--voxel", default=None)
        p.add_argument("--variant", default="cluster_shell")
        p.add_argument("--init-min-tri-angle", dest="init_min_tri_angle",
                       type=float, default=None)
        p.add_argument("--overlap", type=int, default=None)
        p.add_argument("--prior-std", dest="prior_std", type=float, default=None)
        p.add_argument("--cross-clip", dest="cross_clip", default=None,
                       choices=("auto", "spatial", "vocab", "exhaustive", "none"))
        p.add_argument("--cull", default="auto", choices=("auto", CULL_NONE, CULL_CANOPY),
                       help="'canopy' runs strip_sky + strip_clouds and the re-export "
                           "they force; 'none' skips all three. auto takes the preset's "
                           "answer — indoor presets say none, because a white ceiling is "
                           "desaturated and airborne and reads as a cloud sea")
        p.add_argument("--mapper", default=None,
                       choices=("auto", "incremental", "pose_prior", "global"))
        p.add_argument("--vocab-tree", dest="vocab_tree", default=None)
        anch = p.add_mutually_exclusive_group()
        anch.add_argument("--speed-anchor", dest="speed_anchor", type=float, default=None,
                          metavar="M_PER_S",
                          help="scale ruler A: how fast the camera moved along the "
                               "ground, in m/s (a drone flies ~5). "
                               "scale = speed x clip duration / reconstructed path len")
        anch.add_argument("--height-anchor", dest="height_anchor", type=float,
                          default=None, metavar="M",
                          help="scale ruler B: how high the camera sat above the ground "
                               "it filmed, in m (a walked phone is ~1.6). Use this "
                               "instead of ruler A for anything not flown: the speed of "
                               "a dolly is a guess, its height is a tape measure, and a "
                               "wrong scale inflates the collider voxel grid until it "
                               "crashes")

    common(r := sub.add_parser("run", help="full pipeline"))
    g = r.add_mutually_exclusive_group()
    g.add_argument("--fresh", action="store_true")
    g.add_argument("--from", dest="from_step", metavar="STEP")
    r.add_argument("--only", metavar="STEPS")

    common(s := sub.add_parser("scan", help="diagnose footage, no reconstruction"))
    common(cov := sub.add_parser("coverage", help="analyze 3D multi-view coverage & camera frustums"))
    common(st := sub.add_parser("status", help="per-step completion state"))
    sub.add_parser("doctor", help="toolchain health check")
    sub.add_parser("benchmark", help="profile GPU 3DGS training speed and VRAM throughput")
    capp = sub.add_parser("capture", help="print capture checklist for a preset")
    common(capp, need_name=False)
    capp.add_argument("name", nargs="?", default="_")
    common(v := sub.add_parser("view", help="serve + open the walkable viewer"))
    sub.add_parser("ui", help="one command: serve + open the dashboard "
                              "(run the pipeline, view a model, capture)")

    args = ap.parse_args()

    if args.cmd in (None, "ui"):
        do_ui()
        return

    if args.cmd == "benchmark":
        py310 = ROOT / ".venv310/Scripts/python.exe"
        py_run = str(py310) if py310.exists() else str(PY)
        subprocess.run([py_run, str(ROOT / "scripts/benchmark_hardware.py")])
        return

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
    sys.exit(do_run(cfg))


if __name__ == "__main__":
    main()
