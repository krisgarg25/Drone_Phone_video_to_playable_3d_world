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
import re
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PY310 = ROOT / ".venv310" / "Scripts" / "python.exe"
NODE = shutil.which("node") or "node"
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

# Which directory COLMAP reads, and who is allowed to touch the other one.
FRAMES_RAW = "frames_train"
"""The extracted keyframes as they came off the clip. This is also the copy the splat is
trained on (train_splat reads it by name), so nothing that removes light is ever written
here."""
FRAMES_FLATTENED = "frames_match"
"""``survey_frames --flatten``'s copy: same names, same pixel sizes, low-frequency
illumination divided out. A matching input and nothing else — challenge D3."""

# Capture-style presets. None = let smart defaults / other layers decide.
#
# WHY EVERY PRESET NOW DECLARES THE SAME KNOB SET
#
# Only `room` used to. The other seven silently inherited each script's own default for
# the collider clip, the body height, the grid resolution and the gate thresholds - which
# is how a drone scene ended up with a 4.12 m walk cell and no character height at all, so
# the gate assumed 1.75 m and the nav bake assumed 1.7 m and the preset assumed neither.
# `scripts/scenario_audit.py --all` found that by comparing files; a preset that leaves a
# knob undeclared is a knob nobody chose.
#
# THE BODY BELONGS TO THE SCENARIO, NOT TO THE COVERAGE
#
# `character_height: 0.15` on `room` was a workaround: at the measured scale of a real
# flat (`room_w_jsonl`, 1.0422 m per unit, walls at 2.69 m) the router still refused to
# find a human route, because its lateral clearance demand was 0.9 m - half a shoulder
# width per metre of height - and only about 6 m2 of a scanned room's floor is ever
# measured. Shrinking the person to 15 cm made the numbers pass and made walk mode a
# hamster diorama: bots stayed 1.75 m, the nav bake stayed 1.7 m, and the eye line came
# out 13 cm off the floor. The fix is the clearance rule (walk_path_from_glb), not the
# body size. Every walkable preset is a human again; `object` keeps a token body because
# nothing walks around a turntable.
#
# `cell_meters` is deliberately NOT set on the aerial presets. It was tried, measured
# against their actual cloud density (`temple`: ~7k near-ground gaussians over 111 m, so a
# 4.12 m cell is what the data supports), and a forced-finer grid only manufactures holes
# the coverage check then honestly reports as unmeasured. `object` declares one because a
# 20 cm subject needs it - and no close-up footage has ever been scanned here to prove it.
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
        # A person, in a room whose own walls measure 2.69 m tall. See the note above:
        # this used to be 0.15, which was the clearance bug wearing a body's clothes.
        character_height=1.75,
        # 0.3 m is a doorstep. A human walks up one without being told; the 0.8 m
        # outdoor default lets the walker mount a table as if it were a kerb.
        max_step=0.3,
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
        # The auditorium is the only scene of this kind on disk and it measures 26.9 m
        # across with cameras 1.5 m above the filmed floor - a human walking a big room.
        character_height=1.75, max_step=0.4,
        # A hall's relief is its furniture and services, not terrain: 0.25 m resolves a
        # pallet rack, and the density-derived cell over a 27 m span is coarser.
        advice="Walk serpentine grid paths with cross-ties every 10 meters to prevent drift across large rooms."),
    "outdoor_building": dict(
        label="Outdoor Building / House Facade",
        cull=CULL_CANOPY,
        target=600, overlap=20,
        sift_peak_threshold=0.004,
        sift_edge_threshold=14,
        cross_clip="spatial", loop_detection=True, prior_std=0.5,
        character_height=1.75, max_step=0.4,
        advice="Circle the structure at multiple elevations (ground, mid-height, roof line) facing toward center."),
    "drone": dict(
        label="Aerial Drone Orbit / Push-Forward",
        cull=CULL_CANOPY,
        target=400, overlap=15,
        sift_peak_threshold=0.004,
        sift_edge_threshold=12,
        cross_clip="auto", loop_detection=True, prior_std=1.0,
        # rocks measured 12.1 m above the ground it filmed across a 72 m grid, at a
        # 2.18 m cell - the cell is the finding, not the height. Nothing smaller than a
        # car exists in that collider, so the walk model cannot see an obstacle a person
        # would trip on. 1 m is the coarsest cell that still resolves a doorway.
        character_height=1.75, max_step=0.8,
        advice="Fly continuous orbits at constant speed and radius with 60-70% overlap between adjacent frames."),
    "drone_mapping": dict(
        label="Drone Nadir & Terrain Mapping",
        cull=CULL_CANOPY,
        target=500, overlap=15,
        sift_peak_threshold=0.005,
        sift_edge_threshold=10,
        cross_clip="spatial", loop_detection=True, prior_std=1.5,
        character_height=1.75, max_step=1.0,
        advice="Fly lawnmower grid pattern with 75% forward and 65% side overlap at constant altitude."),
    "object": dict(
        label="Close-Up Object / Turntable 360°",
        cull=CULL_NONE,
        target=300, overlap=15,
        # A turntable is the one scenario where the pixels are the product: the subject
        # is 10-40 cm, so the collider voxel has to resolve a filigree rather than a
        # facade. QUALITY used to overwrite this with 0.25 m - see QUALITY_DEFER.
        sift_peak_threshold=0.003,
        sift_edge_threshold=15,
        cross_clip="exhaustive", loop_detection=False, prior_std=0.05,
        voxel="0.01", cell_meters=0.01, max_step=0.02,
        # Nothing walks around a scanned mug. A token body keeps the router and the
        # camera orbit meaningful without pretending the scene is a room.
        character_height=0.15,
        advice="Circle the object twice at 45° and 15° elevation angles. Keep the subject centered."),
    "sky_heavy": dict(
        label="Outdoor with Bright Open Sky",
        cull=CULL_CANOPY,
        target=450, overlap=20,
        sift_peak_threshold=0.004,
        sift_edge_threshold=12,
        cross_clip="auto", loop_detection=True, prior_std=0.8,
        character_height=1.75, max_step=0.8,
        advice="Angle camera slightly below horizontal to maximize ground feature density and avoid overexposed sky."),
    "corridor": dict(
        label="Linear Street / Long Hallway Path",
        cull=CULL_NONE,
        target=600, overlap=30,
        sift_peak_threshold=0.003,
        sift_edge_threshold=16,
        cross_clip="auto", loop_detection=True, prior_std=0.3,
        character_height=1.75, max_step=0.35,
        # A street's occlusion relief is kerbs and parked cars, not a room's furniture,
        # so the clip gap that keeps a room's walls in the shell is far too tight here.
        clip_gap=2.5,
        advice="Walk slowly in forward straight lines; turn around at the end and walk back along opposite wall for loop closure."),
}

QUALITY_DEFER = ("voxel",)
"""Knobs that belong to the SCENE, not to the pixel budget.

The quality layer applies its values over the preset unconditionally, which is right for
width/steps/cap (they are a GPU-spend decision) and wrong for the collider voxel size: a
`high` run of a 20 cm object used the same 0.25 m voxel as a 100 m temple, and the object
preset had no way to say otherwise. An explicit `--voxel` on the command line still beats
both layers."""

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
    if d.get("fingerprint") != sources_fingerprint(sources):
        return False
    # Same footage is only the same question if the probe measured what is now being
    # asked of it. The D3 illumination block post-dates older cache files, and reading
    # "absent" as "lighting is stable" would freeze a scene out of normalisation for
    # good, so a probe that never measured it is treated as stale and re-run. Same rule
    # now for the E2 scene signature: a cache without it cannot support a preset choice.
    clips = d.get("clips") or []
    if not clips or any(c.get("illumination") is None or c.get("signature") is None
                        for c in clips):
        return False
    logs = d.get("motion_logs") or {}
    return all(clip in logs for clip in (sources.get("poses") or {}))


def run_diagnostics(sources: dict, work: Path) -> dict | None:
    vids = sources["videos"]
    sys.path.insert(0, str(ROOT / "scripts"))
    from capture_diagnostics import probe_pose_motion, probe_video
    if not vids and not sources.get("poses"):
        return None
    reports = []
    for v in vids:
        print(f"[scan] probing {v.name} ...", flush=True)
        reports.append(probe_video(v))
    # Metric motion is the one thing a video frame cannot give up and a pose log can, so
    # it is probed here and cached beside the visual report rather than re-read per
    # command. A log that fails to parse is recorded as `measured: False` with the reason:
    # "no evidence" and "evidence says nothing" are different answers.
    motions = {}
    for clip, log in sorted((sources.get("poses") or {}).items()):
        try:
            motions[clip] = probe_pose_motion(log)
        except Exception as e:                        # noqa: BLE001 - a bad log is a
            motions[clip] = {"measured": False,       # reading matter, not a crash
                             "reason": f"{type(e).__name__}: {e}"}
        print(f"[scan] {clip}: pose log -> "
              + (f"{motions[clip].get('path_m')} m walked at "
                 f"{motions[clip].get('mean_speed_m_per_s')} m/s"
                 if motions[clip].get("measured") else
                 f"unreadable ({motions[clip].get('reason')})"), flush=True)
    out = {"clips": reports, "motion_logs": motions,
           "fingerprint": sources_fingerprint(sources)}
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


def _worst(clips, key):
    """The worst clip's value of a measured illumination number, or None if no clip
    measured it. Every illumination number here gets worse as it grows, so worst is max."""
    vals = [c.get("illumination", {}).get(key) for c in clips]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def aggregate_diag(diag: dict | None) -> dict:
    if not diag:
        return {}
    clips = diag["clips"]
    n = max(len(clips), 1)
    blur_p25s = [c.get("sharpness", {}).get("p25") for c in clips]
    blur_p25s = [b for b in blur_p25s if b is not None]
    sigs = [c.get("signature") or {} for c in clips]
    probed = [c for c in clips if c.get("illumination", {}).get("frames_probed")]

    def _sig(*keys, worse=max):
        """Read a (possibly nested) signature value off every clip and keep the one that
        constrains the answer most. No numpy: pipeline.py is imported by the HTTP server,
        which must not pay for a scientific stack to answer a list of projects."""
        vals = []
        for s in sigs:
            v = s
            for k in keys:
                v = v.get(k) if isinstance(v, dict) else None
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(float(v))
        if not vals:
            return None
        return float(median(vals)) if worse is median else worse(vals)

    motions = {k: v for k, v in (diag.get("motion_logs") or {}).items()
               if isinstance(v, dict)}
    measured = [m for m in motions.values() if m.get("measured")]
    return {
        "rotation_dominant_pct": round(sum(c.get("rotation_dominant_pct", 0) for c in clips) / n),
        "weak_geometry_pct": round(sum(c.get("weak_geometry_pct", 100) for c in clips) / n),
        "blur_p25": min(blur_p25s) if blur_p25s else None,
        "warnings": [w for c in clips for w in c.get("warnings", [])],
        "styles": [c.get("style") for c in clips],
        # The E2 scene signature. `worse` is chosen per quantity as "the clip that most
        # constrains the answer": the shakiest clip for mount, the sparsest for texture.
        "signature_measured": any((s.get("mount") or {}).get("measured") for s in sigs),
        "mount_residual_px": _sig("mount", "vertical_residual_px"),
        "expansion_gain": _sig("expansion_gain", worse=median),
        "sky_fraction_p90": _sig("sky_fraction_p90"),
        "texture_per_10k_px": _sig("texture_features_per_10k_px", worse=min),
        "motion": motions,
        "motion_measured": bool(measured),
        "walked_m": round(sum(m.get("path_m", 0) for m in measured), 1) if measured else None,
        "max_speed_m_per_s": max((m.get("mean_speed_m_per_s", 0) for m in measured),
                                 default=None) if measured else None,
        "max_straightness": max((m.get("straightness", 0) for m in measured),
                                default=None) if measured else None,
        # Challenge D3. One clip with moving light is enough to owe the pass, because
        # the flattening is per frame and the alternative is a scene matched on two
        # different photometric conventions. Every number is a fraction of frame area,
        # measured against each frame's own lit level - see capture_diagnostics.
        "lighting_measured": bool(probed),
        "variable_lighting": any(c["illumination"].get("variable_lighting")
                                 for c in probed),
        "shadow_area_p90": _worst(probed, "shadow_area_p90"),
        "shadow_area_spread": _worst(probed, "shadow_area_spread"),
        "exposure_drift_ratio": _worst(probed, "exposure_drift_ratio"),
        "lighting_clips": [(c["clip"], c["illumination"]) for c in probed],
    }


# Thresholds for the scenario classifier, each one a statement about measured footage.
# Calibration is the five real clips in videos/ probed on 2026-09-26; the numbers are in
# docs/GAPS_AND_OPTIMIZATIONS.md under E2 so a future reader can see how narrow the
# evidence is rather than only how tidy the rule looks.
MOUNT_SHAKY_PX = 0.5
"""Vertical frame-to-frame displacement left over after the smooth trajectory is removed,
on a 40-frame burst. Measured: room_w_jsonl 5.86, roomscan 3.94, test1 1.44 px (all three
a person walking); rocks 0.008, temple 0.010 px (both a drone). The rule sits two orders
of magnitude below the nearest handheld reading and above the aerial ones, because a
gimbal really does remove the walk bob and there is nothing in between here."""

MOUNT_STABLE_PX = 0.15
"""The other side of the same band. Between the two values the mount is undecided and the
classifier says so instead of guessing; on the measured clips nothing landed there, which
is the gap's only real evidence."""

GROUND_SPEED_M_PER_S = 1.0
"""A person walking a phone is 0.16-0.31 m/s across every AR log in this repo. Anything
faster than a walking pace is a vehicle, and a vehicle at height is the aerial family.
Only usable when a pose log exists: video frames cannot give a speed at all, since an
essential matrix normalises its translation."""

CORRIDOR_STRAIGHTNESS = 0.5
"""Net displacement divided by path walked. The indoor logs measure 0.02-0.10 (they loop
back to where they started). A hall walked forward and back is >0.5 by construction, and
that is the whole difference the `corridor` preset exists for."""

INDOOR_WALK_M = 60.0
"""A single continuous loop of more than ~60 m at walking pace is not a room: it is a
floorplate, a hall or a warehouse, and it wants the large-indoor matching window and the
larger pose prior the `indoor_large` preset carries."""


ORBIT_TURN_DEG = 270.0
ORBIT_RADIUS_CV = 0.35
GRID_REVERSAL_DEG = 150.0
GRID_AXIS_FRACTION = 0.7
CORRIDOR_GPS_STRAIGHTNESS = 0.85


def flight_pattern(positions) -> dict:
    """Name the flight pattern a GPS track draws in plan view, or say it cannot.

    The frames cannot tell an orbit from a mapping grid (E2b), but the flight log can,
    because the pattern IS the path: an orbit turns through most of a circle at a
    roughly constant distance from its centre; a nadir mapping mission flies parallel
    legs joined by near-reversals; a corridor runs out along a line. Anything else is
    reported as unresolved rather than forced into the nearest bucket.
    """
    import numpy as np
    xy = np.asarray(positions, dtype=float)[:, :2]
    # A consumer receiver jitters by a metre or two per fix, which at 1-10 Hz is the same
    # size as a step: unsmoothed, a straight run reads as a zigzag twice its length.
    window = int(min(5, max(1, len(xy) // 10)))
    if window > 1:
        kernel = np.ones(window) / window
        xy = np.column_stack([np.convolve(xy[:, axis], kernel, mode="valid") for axis in (0, 1)])
    steps = np.diff(xy, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    moving = lengths > max(0.05, 0.002 * float(lengths.sum()))
    if moving.sum() < 5:
        return {"pattern": None, "reason": "the track barely moves in plan view"}
    steps, lengths = steps[moving], lengths[moving]
    path = float(lengths.sum())
    heading = np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))
    turn = np.degrees(np.diff(heading))
    total = float(abs(np.degrees(heading[-1] - heading[0])))
    straightness = float(np.linalg.norm(xy[-1] - xy[0]) / path)
    radius = np.linalg.norm(xy - xy.mean(axis=0), axis=1)
    radius_cv = float(radius.std() / max(radius.mean(), 1e-9))
    # A reversal: the heading swings by >150 degrees within a short stretch (a grid turn).
    window = max(1, len(turn) // 12)
    swings = np.abs(np.convolve(turn, np.ones(window), mode="valid"))
    reversals, index = 0, 0
    while index < len(swings):
        if swings[index] >= GRID_REVERSAL_DEG:
            reversals += 1
            index += window * 2
        else:
            index += 1
    # Parallel legs put most of the path on one axis (either direction along it): the
    # share of length within 20 deg of the length-weighted dominant axis.
    doubled = np.exp(2j * np.arctan2(steps[:, 1], steps[:, 0]))
    axis = np.angle((lengths * doubled).sum()) / 2
    off = np.abs(np.angle(np.exp(1j * (np.arctan2(steps[:, 1], steps[:, 0]) - axis)) ** 2)) / 2
    axis_fraction = float(lengths[np.degrees(off) <= 20].sum() / path)
    facts = {"path_m": round(path, 1), "net_turn_deg": round(total, 1),
             "straightness": round(straightness, 3), "radius_cv": round(radius_cv, 3),
             "reversals": reversals, "axis_fraction": round(axis_fraction, 3)}
    if straightness >= CORRIDOR_GPS_STRAIGHTNESS:
        return {"pattern": "corridor", **facts,
                "because": f"the track ends {straightness:.0%} of its length from where it "
                           "began: a run along a line"}
    if total >= ORBIT_TURN_DEG and radius_cv <= ORBIT_RADIUS_CV and reversals == 0:
        return {"pattern": "orbit", **facts,
                "because": f"the heading turns through {total:.0f} deg at a near-constant "
                           f"distance from the centre (spread {radius_cv:.0%})"}
    if reversals >= 2 and axis_fraction >= GRID_AXIS_FRACTION:
        return {"pattern": "grid", **facts,
                "because": f"{reversals} near-reversals of heading with {axis_fraction:.0%} of "
                           "the path on one axis: parallel legs, a mapping grid"}
    return {"pattern": None, **facts,
            "reason": "neither an orbit, a grid nor a line; the track does not decide"}


def gps_flight_pattern(name: str):
    """flight_pattern() for a scene that carries the survey lane's telemetry, else None."""
    telemetry = ROOT / "videos" / name / "telemetry.csv"
    meta = ROOT / "videos" / name / "flight_metadata.json"
    if not (telemetry.is_file() and meta.is_file()):
        return None
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        import survey_georef
        track = survey_georef.normalize_telemetry(
            telemetry, json.loads(meta.read_text(encoding="utf-8-sig")))
        return flight_pattern([s["position"] for s in track["samples"]])
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"pattern": None, "reason": f"telemetry unreadable: {error}"}


GPS_PATTERN_PRESET = {"orbit": "drone", "grid": "drone_mapping", "corridor": "corridor"}


def pick_preset(agg: dict, handheld: bool) -> tuple[str, list]:
    """Choose a capture preset from what the probe measured, and say which number did it.

    Order matters, and it is ordered by how much confidence each signal earns.

    1. A pose log is a declaration in metres. ARCore/ARKit/Record3D recorded where the
       handset was, so the path walked, the speed and how straight it ran are measurements
       rather than inferences - and no drone emits one.
    2. Without a log, mount stability is the only aerial-vs-handheld line the pixels can
       draw: an essential matrix normalises translation, so nothing in a video pair knows
       whether the camera moved at 0.3 m/s or 12 m/s.
    3. Sky area and horizon strength were tried and REFUTED on this footage: a white
       painted ceiling reads 16% "sky" and a room's wall-ceiling junction scores a
       stronger horizon than an oblique drone pass does. They are recorded, never trusted.

    Where the evidence cannot separate the aerial sub-presets, the answer is `drone` and
    the record says which candidates it cannot tell apart, rather than inventing a reason.
    """
    ev: list = []
    motion_ok = bool(agg.get("motion_measured"))
    resid = agg.get("mount_residual_px")
    speed = agg.get("max_speed_m_per_s")
    path = agg.get("walked_m")
    straight = agg.get("max_straightness")

    if motion_ok:
        ev.append({"signal": "pose_log_motion", "value": f"{path} m at {speed} m/s, "
                       f"straightness {straight}",
                   "because": "an AR pose log records metres, so the walk is measured "
                              "rather than inferred"})
        if speed is not None and speed > GROUND_SPEED_M_PER_S:
            ev.append({"signal": "speed", "value": speed,
                       "because": f"faster than a walking pace ({GROUND_SPEED_M_PER_S} m/s)"
                                  " - this is a vehicle, not a hand"})
            return "drone", ev
        if straight is not None and path is not None \
                and straight > CORRIDOR_STRAIGHTNESS and path > 15:
            ev.append({"signal": "straightness", "value": straight,
                       "because": f"it went {straight:.0%} as far as it walked: a run down "
                                  "a line, not a loop around a subject"})
            return "corridor", ev
        if path is not None and path > INDOOR_WALK_M:
            ev.append({"signal": "walked_m", "value": path,
                       "because": f"{path:.0f} m in one continuous pass is a floorplate, "
                                  f"not a room (the room scans here are 10-45 m)"})
            return "indoor_large", ev
        ev.append({"signal": "walked_m", "value": path,
                   "because": "slow, closed, tens of metres: a person inside a room"})
        return "room", ev

    if resid is not None and resid >= MOUNT_SHAKY_PX:
        ev.append({"signal": "mount_residual_px", "value": resid,
                   "because": f"the frame bobs by {resid:.2f} px between consecutive "
                              f"frames - a walk, not a gimbal (>= {MOUNT_SHAKY_PX} px)"})
        styles = set(agg.get("styles") or [])
        if styles == {"low_texture_or_blur"}:
            ev.append({"signal": "style", "value": sorted(styles),
                       "because": "the probe found no usable texture in any clip, which "
                                  "is a blank-wall interior signature rather than a size "
                                  "measurement"})
        return "room", ev
    gps = agg.get("gps_pattern")
    if resid is not None and resid <= MOUNT_STABLE_PX and gps and gps.get("pattern"):
        ev.append({"signal": "mount_residual_px", "value": resid,
                   "because": f"the frame is dead steady ({resid:.3f} px of bob) - a "
                              "gimbal or an airframe, not a hand"})
        ev.append({"signal": "gps_flight_pattern", "value": gps["pattern"],
                   "because": gps["because"] + " (from the flight's GPS log, which the "
                              "frames alone cannot supply)"})
        return GPS_PATTERN_PRESET[gps["pattern"]], ev
    if resid is not None and resid <= MOUNT_STABLE_PX:
        ev.append({"signal": "mount_residual_px", "value": resid,
                   "because": f"the frame is dead steady ({resid:.3f} px of bob) - a "
                              "gimbal or an airframe, not a hand"})
        if gps:
            ev.append({"signal": "gps_flight_pattern", "value": None,
                       "because": gps.get("reason", "the GPS track did not decide")})
        ev.append({"signal": "unresolved", "value": None,
                   "because": "orbit vs nadir-mapping vs facade-circling cannot be told "
                              "apart from these frames: sky area and horizon strength were "
                              "both refuted on measured footage, and no pose log gives a "
                              "flight path. `drone` is the aerial default; pick "
                              "drone_mapping or outdoor_building if you know the mission."})
        return "drone", ev
    if handheld:
        ev.append({"signal": "pose_logs_present", "value": True,
                   "because": "a handset made this and logged poses, but the log could "
                              "not be read for motion - room is the safe indoor default"})
        return "room", ev
    ev.append({"signal": "none", "value": None,
               "because": "no pose log, and the mount probe measured nothing "
                          f"(residual {resid!r}); `room` is the default of last resort, "
                          "not a finding"})
    return "room", ev


def _resolve_photometric(cfg_vals: dict, agg: dict, requested: str, narrate: bool) -> None:
    """Challenge D3: decide the matching directory from the footage, before the plan hash.

    This has to be settled here rather than inside a step because `image_dir` goes into
    the COLMAP plan, and the plan hash is what staleness is judged by: a run that
    decided to skip normalisation after the hash was taken would still look up to date,
    and the whole point of wiring this in is that the choice is auditable.

    `on` and `off` are the operator's word over the measurement; `off` matching on the
    raw frames is a legitimate call when the pass dulls a scene the probe could not
    separate from its own albedo. Neither is silent: the reason is printed and stored.
    """
    measured = bool(agg.get("lighting_measured"))
    choice = requested
    if requested == "auto":
        choice = "on" if agg.get("variable_lighting") else "off"
    # Three states, not two: "measured and stable" and "never measured" both leave the
    # frames raw, but only one of them is a finding. Same distinction D5 had to learn
    # the hard way, where 0% observed and 0% measurable came out of the same code path.
    if requested != "auto":
        reason = "operator override"
    elif measured:
        reason = "measured on the footage"
    elif agg:
        reason = ("the probe ran but could not measure the light here, so nothing is "
                  "removed")
    else:
        reason = "no illumination measurement: this command skipped the capture probe"
    cfg_vals["photometric"] = choice
    cfg_vals["image_dir"] = FRAMES_FLATTENED if choice == "on" else FRAMES_RAW
    cfg_vals["photometric_reason"] = (
        f"{choice} for COLMAP ({reason}"
        + (f": shadow area p90 {agg.get('shadow_area_p90'):.1%} of frame, spread across "
           f"clip {agg.get('shadow_area_spread'):.1%}, whole-frame level drift "
           + ("n/a" if agg.get("exposure_drift_ratio") is None
              else f"{agg['exposure_drift_ratio']:.2f}x") + ")"
           if measured else ")"))
    if narrate:
        if measured:
            p90, spread = agg.get("shadow_area_p90"), agg.get("shadow_area_spread")
            drift = agg.get("exposure_drift_ratio")
            print(f"[photometric] shadow area p90 {p90:.1%} of frame, spread across clip "
                  f"{spread:.1%}, whole-frame level drift "
                  + ("n/a" if drift is None else f"{drift:.2f}x")
                  + f" -> {choice} ({reason})")
        else:
            print(f"[photometric] {choice} ({reason})")


def poses_of(sources: dict) -> set:
    """Clip names that came with a pose log - the only pre-run evidence of metric motion."""
    return {c for c in (sources.get("poses") or {})}


def build_config(args, sources: dict, allow_auto_diag: bool = True) -> dict:
    q = dict(QUALITY[args.quality])
    preset_name = args.preset
    diag = None
    agg = {}
    evidence = []
    """Every parameter change this function makes, in order, with what it overrode and
    why. It is the body of the scenario record, and it is the only way a finished run
    directory can answer "was 0.002 the preset's number or the smart rule's"."""
    handheld = bool(sources.get("poses")) or bool(sources.get("frames_dirs"))
    narrate = args.cmd in ("run", "scan")
    photometric = getattr(args, "photometric", "auto")
    # The probe answers for two decisions now: which preset the footage looks like,
    # and whether the light moves enough to pay for normalising it. So it runs when
    # either one is left to it, not only under `--preset auto`.
    if preset_name == "auto" or photometric == "auto":
        work_probe = ROOT / "work" / args.name
        if diag_uptodate(work_probe, sources):
            diag = json.loads((work_probe / "diagnostics.json").read_text(encoding="utf-8"))
            if narrate:
                print("[auto] using cached diagnostics")
        elif allow_auto_diag:
            diag = run_diagnostics(sources, work_probe)
        elif narrate and preset_name == "auto":
            print("[auto] diagnostics skipped for this command")
        agg = aggregate_diag(diag)
        pattern = gps_flight_pattern(args.name)
        if pattern is not None:
            agg["gps_pattern"] = pattern
    preset_evidence = []
    if preset_name == "auto":
        styles = agg.get("styles") or []
        preset_name, preset_evidence = pick_preset(agg, handheld)
        evidence += preset_evidence
        if narrate:
            print(f"[auto] capture probe -> preset '{preset_name}'")
            for line in (e.get("because", "") for e in preset_evidence):
                if line:
                    print(f"[auto]   {line}")
            print(f"[auto]   {'handheld: pose logs present' if handheld else 'no pose logs'}"
                  f"; styles {sorted(set(styles)) or 'none measured'}")

    application = getattr(args, "application", None)
    app_record = None
    if application:
        import applications as _apps
        app = _apps.get(application)
        app_record = {"id": application, "label": app["label"], "set_by": "operator",
                      "workspace_tabs": app["workspace_tabs"], "required_products": app["required_products"]}
        if args.preset == "auto":
            chosen, why = _apps.preset_for(application, preset_name)
            if chosen != preset_name:
                evidence.append({"param": "preset", "from": preset_name, "to": chosen,
                                 "set_by": f"application:{application}", "because": why})
                if narrate:
                    print(f"[auto] application '{application}' -> preset '{chosen}' ({why})")
            preset_name = chosen
    elif agg.get("gps_pattern"):
        import applications as _apps
        suggestions = _apps.suggest(agg["gps_pattern"])
        if suggestions:
            app_record = {"id": None, "suggested": suggestions,
                          "note": "no application was given; these are suggestions from the GPS flight pattern"}

    cfg_vals = {}
    origin = {}
    """`origin[param]` records which layer last set it - `preset:room`, `smart`,
    `quality:high`, `cli`, `operator`. A finished run directory has to be able to answer
    "why is the collider built this way" without re-deriving the layer precedence from
    the source, and every E-block audit below reads a parameter it cannot otherwise
    attribute."""
    for k, v in PRESETS[preset_name].items():
        if k not in ("label", "advice"):
            cfg_vals[k] = v
            origin[k] = f"preset:{preset_name}"
    label = PRESETS[preset_name].get("label", preset_name)
    advice = PRESETS[preset_name].get("advice", "")

    def setv(key, val, src):
        cfg_vals[key] = val
        origin[key] = src

    # smart rules on top
    smart_notes = []
    for cond, key, val, why in SMART_RULES:
        try:
            if cond(agg):
                if key in cfg_vals and cfg_vals[key] != val:
                    smart_notes.append(f"{key}={val} ({why})")
                    evidence.append({"param": key, "from": cfg_vals[key], "to": val,
                                     "set_by": "smart", "because": why})
                setv(key, val, "smart")
        except (KeyError, TypeError):
            pass

    # quality layer fills Nones
    for k, v in q.items():
        if v is None:
            q[k] = {"target": PRESETS[preset_name].get("target", 400)}.get(k, v)
        cfg_vals.setdefault(k, v)
        origin.setdefault(k, f"quality:{args.quality}")
    for k, v in q.items():
        if v is None:
            continue
        # A scene-scale knob stays with the scenario that chose it. `high` quality on a
        # 20 cm object is still a 20 cm object, and 0.25 m of collider voxel cannot see
        # one. `--voxel` on the command line still overrides both, below.
        if k in QUALITY_DEFER and origin.get(k, "").startswith("preset:"):
            evidence.append({"param": k, "from": cfg_vals[k], "to": v,
                             "set_by": f"preset:{preset_name}",
                             "because": f"scene scale beats the {args.quality} quality "
                                        "tier's resolution setting"})
            continue
        if k in cfg_vals and cfg_vals[k] != v:
            evidence.append({"param": k, "from": cfg_vals[k], "to": v,
                             "set_by": f"quality:{args.quality}",
                             "because": "processing quality tier"})
        setv(k, v, f"quality:{args.quality}")

    # explicit CLI overrides win over everything
    for k in ("target", "width", "steps", "cap", "voxel", "grow_grad",
              "init_min_tri_angle", "overlap", "prior_std", "cross_clip",
              "vocab_tree", "speed_anchor", "height_anchor"):
        v = getattr(args, k, None)
        if v is not None:
            if k in cfg_vals and cfg_vals[k] != v:
                evidence.append({"param": k, "from": cfg_vals[k], "to": v,
                                 "set_by": "operator", "because": "command line"})
            setv(k, v, "cli")
    if getattr(args, "cull", "auto") != "auto":
        setv("cull", args.cull, "cli")
    elif "cull" not in cfg_vals:
        setv("cull", CULL_CANOPY, "default")
    if getattr(args, "dynamics", "auto") != "auto":
        setv("dynamics", args.dynamics, "cli")
    elif "dynamics" not in cfg_vals:
        setv("dynamics", "auto", "default")
    _resolve_photometric(cfg_vals, agg, photometric, narrate)
    for k in ("photometric", "image_dir", "photometric_reason"):
        origin[k] = "measurement" if photometric == "auto" else "operator"
    if "mapper" not in cfg_vals:
        setv("mapper", "auto", "default")
    if getattr(args, "mapper", None):
        setv("mapper", args.mapper, "cli")
    # loop_detection=True is a promise three presets make; the retrieval stage
    # cannot run without a tree, so honour the one `doctor` says to put in tools/.
    if not cfg_vals.get("vocab_tree") and (ROOT / "tools/vocab_tree.bin").exists():
        setv("vocab_tree", str(ROOT / "tools/vocab_tree.bin"), "auto-discovered")
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
    # E0: the scenario decision, its evidence and every applied value, in one object
    # the run writes to `work/<scene>/scenario.json`. Before this, the resolved preset
    # existed only in `cfg` and one console line: nothing on disk said which scenario a
    # scene was built under, so no audit could compare a preset's promises against what
    # the same scene's gate later measured.
    cfg["_scenario"] = {
        "application": app_record,
        "preset": preset_name,
        "label": label,
        "advice": advice,
        "decided_by": ("operator selection" if args.preset != "auto"
                       else "automatic from the capture probe" if preset_evidence
                       else "default: no probe result to classify"),
        "evidence": evidence,
        "quality": args.quality,
        "cull": cfg_vals.get("cull"),
        "origin": origin,
        "applied": {k: (str(v) if isinstance(v, Path) else v)
                    for k, v in sorted(cfg_vals.items())},
        "capture": {k: agg.get(k) for k in
                    ("styles", "weak_geometry_pct", "blur_p25", "rotation_dominant_pct",
                     "shadow_area_p90", "shadow_area_spread", "exposure_drift_ratio",
                     "lighting_measured", "variable_lighting") if agg.get(k) is not None},
        "handheld": handheld,
        "sources": {"clips": [v.name for v in sources["videos"]],
                    "pose_logs": sorted(poses_of(sources)),
                    "depth_frames": sorted(sources.get("frames_dirs") or {})},
    }
    return cfg


# --------------------------------------------------------------------------- 
# steps
# ---------------------------------------------------------------------------
def nav_params(cfg: dict) -> list:
    """The navmesh agent, taken from the SAME body the preset gave the physics floor.

    This function used to return a drone-specific override and nothing at all for every
    other preset, which left `bake.mjs`'s own defaults in charge: a 1.7 m agent with a
    0.4 m radius and a 0.5 m climb, in a scene whose collision.json had just declared
    1.75 m and whose router was dilating obstacles by 0.34 m. `scenario_audit` caught it
    as three separate disagreements inside one scene - physics, router and nav each
    walking a different person - and an autopilot that plans for a larger walker than the
    physics has is merely conservative, while the reverse drives the bot into a wall.

    Large aerial scenes still need a coarser cell or Recast chokes on the extent; only
    the grid resolution is preset-specific now. The body is one number from one place.
    """
    char_h = float(cfg.get("character_height") or 1.75)
    scale = max(0.05, char_h) / 1.75
    argv = ["--height", f"{char_h:g}",
            "--radius", f"{0.34 * scale:.3f}",
            "--climb", f"{float(cfg.get('max_step') or 0.5):g}"]
    if cfg.get("preset") in ("drone", "drone_mapping", "sky_heavy", "outdoor_building"):
        argv += ["--cell", "0.25", "--slope", "55", "--region", "0.6"]
    return argv


def export_argv(cfg: dict, ply, from_scene: bool = False) -> list:
    """The one export invocation, used by both `export` and `reexport`.

    A scenario's heightfield and collider geometry is decided by five knobs. When the
    second pass forwarded none of them, a `room` run rebuilt the grid at the script
    default cell size and dropped its character height, camera ground height, opacity
    prune and backdrop decision - so the gate, the router, the viewer and the walk test
    all read a world the preset never asked for. One helper, both passes, no drift.
    """
    work, asset = cfg["work"], cfg["work"] / "viewer_assets"
    argv = [PY, ROOT / "scripts/export_viewer_assets.py", "--work", work,
            "--ply", ply,
            *(["--camera-ground", cfg["camera_ground"]]
              if cfg.get("camera_ground") is not None else []),
            *(["--character-height", cfg["character_height"]]
              if cfg.get("character_height") is not None else []),
            *(["--prune-opacity", cfg["prune_opacity"]]
              if cfg.get("prune_opacity") is not None else []),
            *(["--cell-meters", cfg["cell_meters"]]
              if cfg.get("cell_meters") is not None else []),
            *(["--max-step", cfg["max_step"]]
              if cfg.get("max_step") is not None else []),
            *(["--drop-backdrop"] if cfg.get("drop_backdrop") else [])]
    return argv + (["--from-scene"] if from_scene else [])


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

    if cfg.get("photometric") == "on":
        # Challenge D3, and the reason `image_dir` is in the plan below: COLMAP matches
        # a copy of the keyframes with the low-frequency illumination field divided
        # out, while frames_train - the copy train_splat colourises the splat from -
        # stays byte-identical to the extraction. `--keep-all` because the frame budget
        # was already spent by extract_keyframes: a matching directory shorter than
        # keyframes.jsonl means images COLMAP registered that the trainer never reads.
        steps.append(dict(
            name="frames", py=PY,
            argv=[PY, ROOT / "scripts/survey_frames.py", "--work", work,
                  "--flatten", "--keep-all", "--out", work / "frames_match.json"],
            inputs=[work / "keyframes.jsonl"],
            outputs=[work / "frames_match.json", work / "frames_match"]))

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
        "exhaustive_max", "mapper", "prior_std", "init_min_tri_angle",
        # Which frame directory COLMAP reads. Without this key the plan could never
        # select frames_match, so the photometric pass had no way to reach the run.
        "image_dir",
        # E2a. Both keys have been in the preset table since it was written and both
        # are read by run_colmap's feature extractor - but neither was ever in this
        # whitelist, so neither reached plan.json, and `plan.get(key, fallback)` at
        # run_colmap.py:512 always returned the fallback. Net effect: every capture
        # style ran the `room` detector (peak 0.002 / edge 16) and the SIFT tuning
        # recorded against the aerial, warehouse and object presets was inert. A
        # preset that cannot reach the mapper is a comment, not a configuration.
        "sift_peak_threshold", "sift_edge_threshold")
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
    # The brief makes GPS a mandatory input. When the scene carries the survey lane's
    # telemetry, scale comes from fitting the camera track to it (ruler D) instead of an
    # assumed flight speed; the files are in argv, so adding them makes `frame` stale.
    telemetry = ROOT / "videos" / name / "telemetry.csv"
    flight_meta = ROOT / "videos" / name / "flight_metadata.json"
    if telemetry.is_file() and flight_meta.is_file():
        frame_argv += ["--telemetry", telemetry, "--flight-metadata", flight_meta]
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
    # Spelled out rather than inherited, because train_splat's own default is armed
    # (0.05) and that would put an unvalidated cost into every production run. The
    # depth-consistency term measurably improves geometry self-consistency
    # (cv-depth p50 4.8-5.3x lower) but showed no colour improvement on `rocks` -
    # which is 72/72 registered, texture-rich, and therefore the opposite of the
    # textureless/few-view scene it exists for - at +12-14% step time. It stays off
    # until it wins on a scene that actually has the problem.
    train_argv += ["--depth-weight", cfg.get("depth_weight", 0.0)]
    steps += [
        dict(name="poses", py=PY,
             argv=[PY, ROOT / "scripts/parse_colmap.py", "--work", work],
             outputs=[work / "keyframes_poses.jsonl"]),
        # Challenge D1. Masking needs to know how the camera moved, so it can only
        # run against a reconstruction that already exists - which makes it a second
        # matching pass, and the step itself decides whether that pass is worth its
        # cost and reverts if it did not register more frames. Advisory: the pass-1
        # model on disk is valid on its own, so a failed mask experiment must not
        # throw away the reconstruction with it.
        *([] if cfg.get("dynamics") == "off" else [dict(
            name="dynamics", py=PY,
            argv=[PY, ROOT / "scripts/dynamics_pass.py", "--work", work,
                  "--below", str(cfg.get("rescue_below", 0.6))]
                 + (["--force"] if cfg.get("dynamics") == "force" else []),
            inputs=[work / "keyframes_poses.jsonl",
                    work / "colmap" / "sparse" / "txt" / "images.txt"],
            outputs=[work / "dynamics.json"])]),
        dict(name="train", py=PY310, env=GPU_ENV,
             argv=train_argv,
             outputs=[work / "splat.ply"]),
        dict(name="frame", py=PY,
             argv=frame_argv,
             inputs=[work / "splat.ply", work / "keyframes_poses.jsonl"]
                    + ([work / "pose_priors.jsonl"]
                       if (work / "pose_priors.jsonl").exists() else []),
             outputs=[work / "frame.json"]),
        dict(name="depth", py=PY310, env=GPU_ENV,
             # MoGe-2 as an INDEPENDENT SECOND RULER. Advisory and read-only by
             # construction: it compares its own metric estimate against frame.json
             # and writes nothing the reconstruction consumes, so a disagreement is
             # evidence to act on rather than a silent rewrite of the scene's scale.
             # This is what caught rocks' "flight speed x clip duration" anchor
             # disagreeing by 38% where the AR-prior interior agreed within 15%.
             argv=[PY310, ROOT / "scripts/depth_prior.py", "--work", work],
             inputs=[work / "frame.json", work / "keyframes_poses.jsonl",
                     work / "colmap" / "sparse" / "txt" / "images.txt"],
             outputs=[work / "depth" / "summary.json"]),
        dict(name="export", py=PY,
             argv=export_argv(cfg, work / "splat.ply"),
             outputs=[asset / "scene.ply", asset / "heights.f32"]),
        dict(name="sky", py=PY,
             argv=[PY, ROOT / "scripts/strip_sky.py", "--asset", asset], outputs=[]),
        dict(name="clouds", py=PY,
             argv=[PY, ROOT / "scripts/strip_clouds.py", "--asset", asset], outputs=[]),
        # Re-export over the culled model. Same helper as `export`, deliberately: this
        # step used to forward nothing, and because `--from-scene` rewrites
        # collision.json from scratch, a `room` run's character height, camera ground
        # height, cell size, opacity prune and backdrop decision were all applied at
        # `export` and then silently replaced by script defaults two steps later.
        # Every interior scene on disk has the defaults in its collision.json for this
        # reason. E1/E3/E4 are unmeasurable while the preset's own knobs evaporate
        # before the gate reads them.
        dict(name="reexport", py=PY,
             argv=export_argv(cfg, asset / "scene.ply", from_scene=True), outputs=[]),
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
        dict(name="nav", py=NODE,
             # Bake a navigation mesh from the shipped collider so game/combat mode
             # and bot pathing work straight from a run — no manual `node
             # tools/navbake/bake.mjs <scene>` per scene. Runs after `surface` because
             # tune_collider rewrites collision.collision.glb there. Advisory: nothing
             # downstream in the run reads nav.json; the viewer loads it at runtime and
             # falls back to a heightfield nav if it is absent.
             argv=[NODE, ROOT / "tools/navbake/bake.mjs", name, *nav_params(cfg)],
             inputs=[work / "pc" / "collision.collision.glb"],
             outputs=[work / "pc" / "nav.json"]),
        dict(name="semantics", py=PY,
             # Heuristic ground/road/building/vegetation/obstacle labelling of the
             # sparse cloud. Advisory: it reads finished assets and never gates the
             # run, so a scene still ships if labelling is skipped or fails.
             argv=[PY, ROOT / "scripts/label_semantics.py", "--work", work],
             inputs=[asset / "sparse_points.json"],
             outputs=[asset / "semantics.json"]),
        dict(name="rooms", py=PY,
             # Floor / wall / ceiling detection for interiors, and the only place a
             # ceiling height or a real wall clearance can come from. Advisory like
             # `semantics`: it reads finished assets and never gates the run, so a
             # drone scene still ships - with zero rooms and the geometry that made it
             # refuse, rather than a floor it did not measure.
             argv=[PY, ROOT / "scripts/detect_rooms.py", "--asset", asset],
             inputs=[asset / "scene.ply", asset / "collision.json"],
             outputs=[asset / "rooms.json"]),
        dict(name="texture", py=PY,
             # Challenge G1: the frames baked into a UV texture on COLMAP's CPU Delaunay
             # surface of the sparse model, written in the viewer's world frame. Advisory:
             # a coarse mesh the splat already outclasses must never cost a scene its walk.
             argv=[PY, ROOT / "scripts/survey_texture.py", "--work", work],
             inputs=[work / "colmap" / "sparse" / "txt" / "images.txt", work / "frame.json"],
             outputs=[work / "textured" / "texture_report.json"]),
        dict(name="gate", py=PY,
             argv=[PY, ROOT / "scripts/check_world.py", "--asset", asset,
                   "--work", work,
                   *(["--min-coverage", cfg["min_coverage"]]
                     if cfg.get("min_coverage") is not None else []),
                   *(["--min-perimeter", cfg["min_perimeter"]]
                     if cfg.get("min_perimeter") is not None else [])],
             outputs=[]),
        # E2: compare the scenario this run chose against the world it produced, while
        # both are still true. Advisory because it changes nothing - it is a reading of
        # artefacts the previous steps already wrote - but it has to be a STEP, because
        # an audit nobody runs is how six of these presets went a full year without ever
        # being looked at, and how a `room` scene with an 11 m walk grid stayed green.
        dict(name="audit", py=PY,
             argv=[PY, ROOT / "scripts/scenario_audit.py", name, "--root", ROOT,
                   "--json", work / "scenario_audit.json"],
             inputs=[work / "scenario.json", asset / "world_check.json"],
             outputs=[work / "scenario_audit.json"]),
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
    "frames": 20 * 60,       # one flatten + JPEG write per keyframe: ~0.5 s at 4K
    "priors": 10 * 60,
    "colmap": 6 * 3600,        # matching + mapping on hundreds of frames
    "poses": 5 * 60,
    "dynamics": 7 * 3600,      # a mask pass plus a whole second matching run
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
    "nav": 10 * 60,
    "semantics": 10 * 60,
    "rooms": 10 * 60,
    "texture": 20 * 60,        # CPU mesh + bake: 15 s for rocks' 17k faces x 72 views
    "depth": 20 * 60,          # MoGe inference: measured 0.19 s/frame, 1.0 GiB peak
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
ADVISORY = ("evals", "pairs", "semantics", "nav", "rooms", "depth", "audit", "texture")


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
    collider, one level up. Covers the .py files the command names, every sibling
    module they import, and robust.py, which every step script imports and which
    has changed a step's behaviour on its own more than once.
    """
    scripts = (ROOT / "scripts").resolve()
    files = {scripts / "robust.py"}
    files |= {(ROOT / a).resolve() if not Path(str(a)).is_absolute()
              else Path(str(a)).resolve()
              for a in argv if str(a).endswith(".py")}
    pending = [f for f in files if f.parent == scripts]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # `import robust as rb` and `from camera_intrinsics import f` both name a
        # sibling module; either one changes what the step computes.
        for name in re.findall(r"^(?:from|import)\s+([a-z_][a-z0-9_]*)\b", source, re.M):
            sibling = scripts / (name + ".py")
            if sibling.exists() and sibling not in files:
                files.add(sibling)
                pending.append(sibling)
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


def write_scenario(cfg: dict, steps: list | None = None) -> Path:
    """Persist the scenario decision where a finished run directory can still be asked
    what it was built as.

    `work/<scene>/scenario.json` is E0, and the reason it did not exist before is
    instructive: the resolved preset lived in `cfg`, was printed once to a console
    nobody scrolls back through, and was never written down. So no audit could compare a
    preset's promises with what the same scene's gate later measured, no UI could say
    "this was built as a room scan", and re-running with a different `--preset` left no
    trace that it had. The record carries the applied values AND who set each one, which
    is the only way a parameter is reviewable after the fact.
    """
    work = cfg["work"]
    rec = dict(cfg.get("_scenario") or {})
    rec.update({"scene": cfg["name"], "variant": cfg.get("variant"),
                "decided_at": int(time.time())})
    if steps is not None:
        rec["steps"] = [s["name"] for s in steps]
    # No machine path leaves this file: it is served to the browser through the project
    # API, and an absolute filesystem path in a public payload is both a leak and, on a
    # LAN demo, an invitation. A workspace-local API has refused absolute paths in
    # request bodies since D3; it should not put one in a response either. Rewritten on
    # the way out (not by string-replacing the serialised JSON, where backslashes are
    # escaped and the pattern silently fails to match).
    root = str(ROOT) + os.sep

    def _relativise(v):
        if isinstance(v, dict):
            return {k: _relativise(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_relativise(x) for x in v]
        if isinstance(v, str) and v.startswith(root):
            return v[len(root):]
        return v

    rec = _relativise(rec)
    work.mkdir(parents=True, exist_ok=True)
    out = work / "scenario.json"
    rb.write_json(out, rec)
    return out


def do_run(cfg: dict) -> int:
    steps = build_steps(cfg)
    work = cfg["work"]
    print(f"[{cfg['name']}] preset={cfg['preset']} cull={cfg['cull']} "
          f"quality={cfg['quality']} -> {len(steps)} steps")
    write_scenario(cfg, steps)
    report = rb.Report(work, cfg["name"])
    if cfg.get("photometric_reason"):
        # Which directory COLMAP was pointed at is in plan.json; why is the part a
        # reviewer of a finished run cannot recover, and it is the part that decides
        # whether the model was matched on flattened frames or on the raw ones.
        report.note("photometric (D3): " + cfg["photometric_reason"])
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
                      f"- verdict: {work / 'viewer_assets' / 'world_check.json'}")
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
        s = r.get("signature") or {}
        m = s.get("mount") or {}
        print("  scene signature: "
              f"sky {s.get('sky_fraction_median')} (p90 {s.get('sky_fraction_p90')}), "
              f"horizon {s.get('horizon_score_median')}, expansion "
              f"{s.get('expansion_gain')}, texture "
              f"{s.get('texture_features_per_10k_px')}/10k px")
        print("  mount: " + (f"{m.get('vertical_residual_px')} px vertical residual over "
                             f"{m.get('frames')} consecutive frames "
                             f"(roughness {m.get('roughness')})"
                             if m.get("measured") else
                             f"not measured - {m.get('reason', 'no burst taken')}"))
        i = r.get("illumination") or {}
        if i.get("frames_probed"):
            print(f"  illumination: {i['frames_probed']} frames, shadow area median "
                  f"{i['shadow_area_median']:.1%} / p90 {i['shadow_area_p90']:.1%} / "
                  f"spread {i['shadow_area_spread']:.1%}, low-freq contrast "
                  f"{i['low_freq_contrast_median']:.2f}, level drift "
                  + ("n/a" if i.get("exposure_drift_ratio") is None
                     else f"{i['exposure_drift_ratio']:.2f}x")
                  + f" -> variable: {bool(i.get('variable_lighting'))}")
            print(f"    {i['basis']}")
        elif i.get("reason"):
            print(f"  illumination: not measured - {i['reason']}")
        for w in r["warnings"]:
            print(f"  ! {w}")
    for clip, mo in sorted((agg.get("motion") or {}).items()):
        print(f"\n=== pose log {clip} === " + (
            f"{mo['path_m']} m walked, net {mo['net_displacement_m']} m, straightness "
            f"{mo['straightness']}, {mo['mean_speed_m_per_s']} m/s over "
            f"{mo['duration_s']} s ({mo['samples']} samples, "
            f"{mo['teleports_dropped']} relocalisation jumps dropped)"
            if mo.get("measured") else f"unreadable - {mo.get('reason')}"))
    # The decision the probe buys, printed in full. A one-word "preset=room" cannot be
    # argued with, so every signal that decided it is named with its number, and so is
    # the case where nothing measured enough to decide.
    handheld = bool(cfg["sources"].get("poses")) or bool(cfg["sources"].get("frames_dirs"))
    pattern = gps_flight_pattern(cfg["name"])
    if pattern is not None:
        agg["gps_pattern"] = pattern
    chosen, ev = pick_preset(agg, handheld)
    print(f"\n[scenario] {chosen}  ({PRESETS[chosen].get('label', chosen)})")
    for e in ev:
        print(f"  - {e['signal']}={e['value']}: {e['because']}")
    if PRESETS[chosen].get("advice"):
        print(f"  - capture advice: {PRESETS[chosen]['advice']}")
    write_scenario({**cfg, "preset": chosen, "_scenario": {
        **(cfg.get("_scenario") or {}), "preset": chosen,
        "label": PRESETS[chosen].get("label", chosen),
        "advice": PRESETS[chosen].get("advice", ""),
        "evidence": (cfg.get("_scenario") or {}).get("evidence") or ev}})
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
          f"{PY310} -m pip install -r requirements-train.txt "
          "(or run python scripts/bootstrap.py --with-train)")
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
    do_ui(cfg["name"])


def do_ui(scene: str | None = None) -> None:
    from urllib.parse import quote
    from _serve import terminate_process_tree

    def listening(port):
        with socket.socket() as probe:
            return probe.connect_ex(("127.0.0.1", port)) == 0

    frontend = ROOT / "groundcontrol"
    next_cli = frontend / "node_modules/next/dist/bin/next"
    node = shutil.which("node")
    if not listening(3000) and (not node or not next_cli.is_file()):
        sys.exit("The workspace needs Node.js and its packages. Run npm install in groundcontrol, then try again.")
    children = []
    process_options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                       if os.name == "nt" else {"start_new_session": True})
    url = "http://127.0.0.1:3000" + ("/projects/" + quote(scene, safe="") if scene else "")
    try:
        if not listening(8137):
            server = ("import functools,http.server,_serve; "
                      "http.server.ThreadingHTTPServer(('127.0.0.1',8137),"
                      "functools.partial(_serve.H,directory=str(_serve.ROOT))).serve_forever()")
            children.append(subprocess.Popen([str(PY), "-c", server], cwd=str(ROOT), **process_options))
        if not listening(3000):
            children.append(subprocess.Popen([node, str(next_cli), "dev", "--hostname", "127.0.0.1", "--port", "3000"],
                                             cwd=str(frontend), **process_options))
        for port in (8137, 3000):
            deadline = time.monotonic() + 60
            while not listening(port):
                if any(child.poll() is not None for child in children) or time.monotonic() > deadline:
                    raise RuntimeError(f"Workspace service on port {port} did not start. Check the output above.")
                time.sleep(0.2)
        print(f"[ui] workspace: {url}")
        webbrowser.open(url)
        if children:
            print("[ui] Ctrl+C stops the services started here and their child jobs.")
            while all(child.poll() is None for child in children):
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for child in reversed(children):
            if child.poll() is None:
                terminate_process_tree(child)


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
        import applications as _apps
        p.add_argument("--application", choices=tuple(_apps.APPLICATIONS), default=None,
                       help="what the scan is for (playbook §4); it only selects existing knobs "
                            "and is recorded in scenario.json")
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
        p.add_argument("--dynamics", default="auto", choices=("auto", "off", "force"),
                       help="challenge D1. 'auto' masks moving objects and re-matches only "
                            "when registration is already weak, because the mask pass needs "
                            "camera centres a first reconstruction has to supply. 'force' "
                            "pays for it regardless; 'off' never masks and says so in "
                            "work/<name>/dynamics.json")
        p.add_argument("--photometric", default="auto", choices=("auto", "on", "off"),
                       help="challenge D3. 'auto' asks the capture probe: when the "
                            "shadow-suspect area of the frame is large AND moves across "
                            "the clip, a 'frames' step flattens the matching copy to "
                            "work/<name>/frames_match and COLMAP reads that. 'on' pays for "
                            "the pass regardless; 'off' matches on the raw frames_train "
                            "even where the measurement says lighting varies. Either way "
                            "the trained colours come from frames_train, never the "
                            "flattened copy.")
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
