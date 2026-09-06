"""Export trained-splat assets for the browser viewer — one pass, final coords.

Pipeline position: train -> splat.ply (COLMAP frame) -> solve_frame.py -> THIS.

What it does:
  1. Load INRIA PLY, prune low-opacity gaussians.
  2. Read the world frame from frame.json: gravity from the drone's own gimbal,
     metric scale from clip duration. See solve_frame.py for why neither is
     derived from the splat's bounds any more.
  3. Split the scene into the walkable REGION (multi-view supported, near the
     flight path) and the BACKDROP (distant mountains, sparse floaters). Both
     get rendered; only the region defines geometry.
  4. Reorient+rescale poses from keyframes_poses.jsonl the same way.
  5. Write viewer_assets/scene.ply, poses.json, heights.f32, coverage.u8,
     collision.json, heightfield preview; pick a spawn on supported ground.
  6. Write eval_pairs.json telling the render step which real frames to match.

Failure policy: an odd scene must still produce assets, or a named reason. A
hamster-scale room, an outdoor take with no walkable floor, a cloud with almost
no multi-view support and a cloud that --prune-opacity emptied are all handled:
every reduction over a possibly-empty selection goes through scripts/robust.py,
absolute metre thresholds are derived from the scene's own extent, and what
genuinely cannot proceed raises a classified StepError that names the upstream
step owing the missing input.

Usage:
  python solve_frame.py --work work/rocks           # writes frame.json first
  python export_viewer_assets.py --work work/rocks --ply work/rocks/splat.ply
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402

from build_heightfield import rasterize_ground, save_heightfield, camera_ground
from solve_frame import multiview_support

# What solve_frame's multiview_support and our own pose export need from a
# keyframes_poses.jsonl camera record. A row missing one of these is not a
# crash waiting in a nested dict lookup, it is one unusable frame.
CAM_KEYS = ("R_rowmajor", "t", "fx", "fy", "cx", "cy")


def _num(doc: dict, key: str, default: float) -> float:
    """Read a finite number out of frame.json.

    Absent, non-numeric and NaN/inf all collapse to `default`. Every caller pairs
    it with its own range check, so a fallback gets reported rather than quietly
    steering the export.
    """
    try:
        v = float(doc.get(key))
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _complete_camera(c) -> bool:
    """A camera this step can project with: intrinsics that are real numbers.

    multiview_support() does float(c["fx"]) and a matmul on R, and the pose
    export writes the same four numbers into poses.json; a null or a string is
    one unusable frame, not a traceback.
    """
    return (isinstance(c, dict) and all(k in c for k in CAM_KEYS)
            and all(_num(c, k, 0.0) > 0 for k in ("fx", "fy", "cx", "cy")))


def usable_poses(path: Path) -> list:
    """Pose rows this step can actually use: a file name and a complete camera."""
    rows = rb.jsonl_rows(path)
    ok = [r for r in rows
          if isinstance(r, dict) and isinstance(r.get("file"), str)
          and _complete_camera(r.get("camera"))]
    if len(ok) < len(rows):
        rb.warn(f"{path.name}: {len(rows) - len(ok)} of {len(rows)} rows have no "
                f"'file' or an incomplete camera; ignored")
    return ok


def frame_size(work: Path, name: str) -> tuple[int, int] | None:
    """Pixel size of an extracted frame, or None when it is not readable.

    poses.json has to carry width/height for the eval renderer, and the frames
    this step asked for are the full-resolution ones — which a scene that never
    finished extracting, or a clip whose frames were cleaned up, does not have.
    One missing jpg used to abort a run whose geometry was already done.
    """
    from PIL import Image
    for sub in ("frames_full", "frames_train"):
        try:
            with Image.open(work / sub / name) as im:
                return int(im.size[0]), int(im.size[1])
        except (OSError, ValueError):
            continue
    return None


def fallback_region(P: np.ndarray, fr: dict, min_views: int, max_range: float,
                    n_cams: int) -> np.ndarray:
    """Which gaussians define geometry when the multi-view test saw nothing.

    Typical for a hamster-scale room, a take with very few keyframes, or a
    `--max-range-mult` tighter than the scene. The empty mask used to reach
    `P[region].min(axis=0)` and abort the step; the repair has to be chosen
    here, because the whole-cloud answer is a different kind of wrong: the
    backdrop would then set the bounds, the heightfield footprint and the
    collider box, which is exactly what the region test exists to prevent.
    """
    box = fr.get("region_box_m") if isinstance(fr.get("region_box_m"), dict) else {}
    try:
        b_lo = np.array(box.get("min", []), np.float64)
        b_hi = np.array(box.get("max", []), np.float64)
    except (TypeError, ValueError):
        b_lo = b_hi = np.zeros((0,))
    usable = (b_lo.shape == (3,) and b_hi.shape == (3,)
              and np.all(np.isfinite(b_lo)) and np.all(np.isfinite(b_hi))
              and bool(np.all(b_hi[[0, 2]] > b_lo[[0, 2]])))
    rb.warn(f"multi-view region is EMPTY ({min_views}+ views within "
            f"{max_range:.3f} units of {n_cams} cameras). The cloud is "
            f"{len(P)} gaussians; something here is at a scale the support test "
            f"was not tuned for.")
    if usable:
        # solve_frame wrote this box from the flight path in the same world
        # frame, so it is a measured zone rather than a guessed one.
        inside = np.all((P >= b_lo) & (P <= b_hi), axis=1)
        if inside.any():
            rb.warn(f"  falling back to frame.json's region_box_m: "
                    f"{int(inside.sum())}/{len(P)} gaussians")
            return inside
    rb.warn("  no region box to fall back on either: every gaussian now votes on "
            "the bounds, heightfield and collider box, so the backdrop will "
            "inflate them. Re-run solve_frame.py with a larger "
            "--max-range-mult / smaller --min-views to fix the real cause.")
    return np.ones(len(P), dtype=bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--ply", required=True, type=Path)
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--cell-meters", type=float, default=0.0,
                    help="heightfield cell size in m; 0 = derive from point density")
    ap.add_argument("--camera-ground", dest="camera_ground", type=float,
                    default=1.2,
                    help="radius (m) around each camera in which the pose, not the "
                         "cloud, defines the floor; 0 disables it. Measured on the "
                         "auditorium: 0.75 m left seat cushions reading 0.22 m above "
                         "the floor, 1.2-1.5 m gives the true ~0.42 m")
    ap.add_argument("--percentile", type=float, default=20.0)
    ap.add_argument("--max-eval-cams", type=int, default=10)
    ap.add_argument("--prune-opacity", type=float, default=0.15)
    ap.add_argument("--thicken", type=float, default=0.30,
                    help="min vertical thickness as fraction of mean horizontal scale")
    ap.add_argument("--max-step", type=float, default=0.8)
    ap.add_argument("--character-height", type=float, default=None,
                    help="metres the viewer's player capsule should stand; when "
                         "set, it is written into collision.json so pc.js sizes "
                         "the character to the scene (a phone-scan of a 3 m room "
                         "cannot host a 1.75 m human without clipping through walls).")
    ap.add_argument("--from-scene", action="store_true",
                    help="the input PLY is an ALREADY-EXPORTED viewer scene "
                         "(e.g. a cloud-culled scene.ply): coordinates, quats and "
                         "scales are left untouched and only the derived assets are "
                         "regenerated — heightfield, coverage, spawn hint, walk "
                         "rectangle, collider box. The multi-view region test still "
                         "needs COLMAP-frame points, so they are inverted back for "
                         "that test only.")
    ap.add_argument("--drop-backdrop", action="store_true",
                    help="remove every gaussian outside the multi-view-supported "
                         "region from scene.ply before writing. For outdoor "
                         "footage this destroys the distant ridgelines that make "
                         "the scene read as a real place, so it is off by default. "
                         "For a room scan the backdrop is floaters the camera "
                         "triangulated 5-15 m past the walls, and it is precisely "
                         "what makes the finished splat look like a glowing cube.")
    args = ap.parse_args()

    out = args.work / "viewer_assets"
    out.mkdir(parents=True, exist_ok=True)

    frame_file = args.work / "frame.json"
    if not frame_file.exists():
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"missing {frame_file} - run solve_frame.py first. Every threshold in "
            f"this step is in metres, and only frame.json says which way is up and "
            f"what a metre is.", returncode=3)
    fr = rb.read_json(frame_file)
    if not isinstance(fr, dict) or not fr:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{frame_file} is empty or is not JSON - re-run solve_frame.py",
            returncode=3)
    try:
        Rg = np.array(fr.get("rotation_rowmajor", []), np.float64)
    except (TypeError, ValueError):
        Rg = np.zeros((0,))
    if Rg.shape != (3, 3) or not np.all(np.isfinite(Rg)):
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{frame_file}: 'rotation_rowmajor' is absent or not a finite 3x3 - "
            f"re-run solve_frame.py", returncode=3)
    s = _num(fr, "scale_m_per_unit", 0.0)
    if not rb.finite(s) or s <= 0:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{frame_file}: 'scale_m_per_unit' is {fr.get('scale_m_per_unit')!r}, "
            f"needs a positive finite number. solve_frame.py could not anchor the "
            f"take metricly - re-run it with --speed-anchor / --height-anchor.",
            returncode=3)
    gi = fr.get("gravity_info") if isinstance(fr.get("gravity_info"), dict) else {}
    try:
        up = np.array(fr.get("up_in_colmap", [0.0, 1.0, 0.0]), np.float64).reshape(-1)
    except (TypeError, ValueError):
        up = np.array([np.nan])
    print(f"[export] frame: up={np.round(up, 3).tolist()} "
          f"(source: {gi.get('used', 'unknown')}), "
          f"scale {s:.3f} m/unit from {fr.get('scale_source', 'unknown')}")

    poses_file = args.work / "keyframes_poses.jsonl"

    if not args.ply.exists() or args.ply.stat().st_size == 0:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"input PLY missing or empty: {args.ply}\n"
            f"  'train' writes work/<scene>/splat.ply, the cloud-cull steps write "
            f"viewer_assets/scene.ply. Whichever step is named by --ply produced "
            f"nothing.", returncode=3)
    try:
        ply = PlyData.read(str(args.ply))
        arr = ply["vertex"].data.copy()
    except Exception as e:
        # plyfile's own PlyParseError family inherits from nothing useful, and a
        # truncated or half-written cloud is exactly the "upstream produced
        # garbage" case this step cannot repair.
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"cannot read {args.ply}: {type(e).__name__}: {e}\n"
            f"  Truncated or unparseable PLY - re-run the step that wrote it.",
            returncode=3) from e
    names = set(arr.dtype.names or ())
    if not {"x", "y", "z"} <= names:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{args.ply} has no x/y/z vertex properties (properties: "
            f"{sorted(names)[:6]}{'...' if len(names) > 6 else ''}) - that is not a "
            f"gaussian splat PLY.", returncode=3)
    if len(arr) == 0:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{args.ply} holds 0 vertices - the step that wrote it exported an "
            f"empty cloud", returncode=3)

    if args.prune_opacity > 0 and "opacity" in names:
        # INRIA PLY stores logits; compare in sigmoid space, keep storage as logits
        op_logit = np.asarray(arr["opacity"], np.float64)
        op = 1.0 / (1.0 + np.exp(-op_logit))
        keep = op >= args.prune_opacity
        print(f"[export] pruning low-opacity gaussians: kept {keep.sum()}/{len(keep)}")
        if not keep.any():
            brightest = rb.safe_max(op, 0.0, label="brightest gaussian opacity")
            raise rb.StepError(
                rb.EMPTY_INPUT,
                f"--prune-opacity {args.prune_opacity} removed every one of the "
                f"{len(op)} gaussians in {args.ply.name}: the most opaque of them "
                f"is {brightest:.4f}. Lower --prune-opacity below {brightest:.4f} "
                f"or pass 0 to disable the filter. If the whole take really is "
                f"that faint, it is 'train' that needs another look.",
                returncode=3)
        arr = arr[keep]

    P = np.stack([np.asarray(arr["x"], np.float64),
                  np.asarray(arr["y"], np.float64),
                  np.asarray(arr["z"], np.float64)], axis=1)
    # A gaussian with a NaN centre cannot be bounded, rasterized or indexed; it
    # can only make every min/max/percentile downstream NaN as well.
    solid = np.all(np.isfinite(P), axis=1)
    if not solid.all():
        rb.warn(f"{int((~solid).sum())}/{len(P)} gaussians have a non-finite centre "
                f"and are dropped")
        arr, P = arr[solid], P[solid]
    if len(P) == 0:
        raise rb.StepError(rb.EMPTY_INPUT,
                           f"no gaussian in {args.ply.name} has a finite centre - "
                           f"there is no geometry here to export", returncode=3)

    # ---- which gaussians are allowed to define geometry ----
    # Computed in the COLMAP frame because that is where the camera matrices
    # live. Everything outside `region` still gets exported and drawn — the
    # distant ridgelines are most of why the scene reads as a real place — it
    # just cannot vote on bounds, scale, the heightfield or the collider box.
    cams = [r["camera"] for r in usable_poses(poses_file)] if poses_file.exists() else []
    if not cams:
        rb.warn(f"no usable camera poses in {poses_file.name}; the multi-view "
                f"region test has nothing to vote with")
    min_views = int(_num(fr, "region_min_views", 4))
    max_range = _num(fr, "region_max_range_units", 0.0)
    if max_range <= 0:
        # solve_frame's own rule: 4x the camera height above the ground.
        agl_u = _num(fr, "camera_agl_m", 0.0) / s
        max_range = 4.0 * agl_u if agl_u > 0 else math.inf
        rb.warn(f"frame.json has no usable 'region_max_range_units'; using "
                f"{max_range:.3f} units instead")
    if args.from_scene:
        P_test = (P / s) @ Rg          # world -> COLMAP, exact inverse of below
    else:
        P_test = P
    try:
        region, _, _ = multiview_support(P_test, cams, min_views, max_range)
    except (ValueError, TypeError, KeyError) as e:
        rb.warn(f"the multi-view support test could not run against "
                f"{poses_file.name}: {type(e).__name__}: {e}")
        region = np.zeros(len(P_test), dtype=bool)
    del P_test
    print(f"[export] geometry region: {region.sum()}/{len(P)} gaussians "
          f"({100 * region.mean():.1f}%), rest is backdrop")
    if not region.any():
        region = fallback_region(P, fr, min_views, max_range, len(cams))

    # ---- gravity + metric scale, both from frame.json ----
    if not args.from_scene:
        P = (P @ Rg.T) * s
    a_lo, a_hi = P.min(axis=0), P.max(axis=0)
    if not (np.all(np.isfinite(a_lo)) and np.all(np.isfinite(a_hi))):
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"the world-frame extent is not finite: scale {s:.6g} m/unit applied to "
            f"{len(P)} gaussians overflows. Check 'scale_m_per_unit' in {frame_file}.",
            returncode=3)
    # The region is never empty by the time we get here, but its bounds are the
    # one number every asset is cut from, so an empty slice falls back to the
    # whole cloud instead of raising on the first axis it cannot reduce.
    Rpts = P[region]
    lo = np.array([rb.safe_min(Rpts[:, k], a_lo[k], label=f"region min {ax}")
                   for k, ax in enumerate("xyz")])
    hi = np.array([rb.safe_max(Rpts[:, k], a_hi[k], label=f"region max {ax}")
                   for k, ax in enumerate("xyz")])
    print(f"[export] region extent x[{lo[0]:.1f}..{hi[0]:.1f}] "
          f"y[{lo[1]:.1f}..{hi[1]:.1f}] z[{lo[2]:.1f}..{hi[2]:.1f}] m")
    print(f"[export] full extent   x[{a_lo[0]:.0f}..{a_hi[0]:.0f}] "
          f"y[{a_lo[1]:.0f}..{a_hi[1]:.0f}] z[{a_lo[2]:.0f}..{a_hi[2]:.0f}] m")

    if not args.from_scene:
        arr["x"] = P[:, 0].astype(np.float32)
        arr["y"] = P[:, 1].astype(np.float32)
        arr["z"] = P[:, 2].astype(np.float32)

    if not args.from_scene and {"rot_0", "rot_1", "rot_2", "rot_3"} <= names:
        q = np.stack([np.asarray(arr[f"rot_{k}"], np.float64) for k in range(4)], axis=1)
        qn = np.linalg.norm(q, axis=1, keepdims=True)
        if not np.all(qn > 0):
            rb.warn(f"{int((qn[:, 0] <= 0).sum())} gaussians have a zero-length "
                    f"quaternion; they keep an identity rotation")
        q /= np.maximum(qn, 1e-12)
        w, x, y, z = q.T
        Rm = np.empty((len(q), 3, 3))
        Rm[:, 0, 0] = 1 - 2 * (y * y + z * z); Rm[:, 0, 1] = 2 * (x * y - w * z); Rm[:, 0, 2] = 2 * (x * z + w * y)
        Rm[:, 1, 0] = 2 * (x * y + w * z); Rm[:, 1, 1] = 1 - 2 * (x * x + z * z); Rm[:, 1, 2] = 2 * (y * z - w * x)
        Rm[:, 2, 0] = 2 * (x * z - w * y); Rm[:, 2, 1] = 2 * (y * z + w * x); Rm[:, 2, 2] = 1 - 2 * (x * x + y * y)
        Rn = np.broadcast_to(Rg, Rm.shape) @ Rm
        wn = np.sqrt(np.clip(1.0 + Rn[:, 0, 0] + Rn[:, 1, 1] + Rn[:, 2, 2], 0.0, None)) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = (Rn[:, 2, 1] - Rn[:, 1, 2]) / (4 * wn)
            yn = (Rn[:, 0, 2] - Rn[:, 2, 0]) / (4 * wn)
            zn = (Rn[:, 1, 0] - Rn[:, 0, 1]) / (4 * wn)
        bad = ~np.isfinite(wn) | (wn < 1e-6)
        wn[bad], xn[bad], yn[bad], zn[bad] = 1.0, 0.0, 0.0, 0.0
        arr["rot_0"] = wn.astype(np.float32)
        arr["rot_1"] = xn.astype(np.float32)
        arr["rot_2"] = yn.astype(np.float32)
        arr["rot_3"] = zn.astype(np.float32)

        # ---- vertical thickening + metric rescale of the gaussians ----
        # Training saw the scene only from above, so ground gaussians are flat
        # horizontal discs; at eye level they read as knife streaks with sky
        # between them. Thicken each gaussian along its most-vertical local
        # axis (disc -> pancake): solid from the side, ~unchanged from above.
        #
        # Both operations are RELATIVE. An earlier version clamped the new
        # thickness to an absolute [0.02, 0.5] and never applied `s` to the
        # scales at all, so gaussians ended up 3.6x too fat for the scene they
        # sat in — which is what made the ground render as a white smear.
        if {"scale_0", "scale_1", "scale_2"} <= names:
            S = np.stack([np.asarray(arr[f"scale_{k}"], np.float64) for k in range(3)], 1)
            # exp() overflows to inf past ~709 and passes NaN straight through;
            # both would be written back into the PLY as an unsplattable gaussian.
            s_lin = np.exp(np.clip(S, -30.0, 30.0))
            if not np.all(np.isfinite(s_lin)):
                rb.warn(f"{int((~np.isfinite(s_lin)).sum())} non-finite gaussian "
                        f"sizes collapsed to 0")
                s_lin = np.nan_to_num(s_lin, nan=0.0, posinf=0.0, neginf=0.0)
            if args.thicken > 0:
                vert = np.abs(Rn[:, 1, :])  # [N,3] world-Y component of each local axis
                jmax = vert.argmax(1)
                other = np.where(np.arange(3)[None, :] == jmax[:, None], np.nan, s_lin)
                ref = np.nanmean(other, axis=1)
                if not np.all(np.isfinite(ref)):
                    # A row with no usable horizontal scale has nothing to be
                    # thickened against; leave its vertical axis as trained.
                    rb.warn(f"{int((~np.isfinite(ref)).sum())} gaussians have no "
                            f"horizontal scale to thicken against; left alone")
                    ref = np.where(np.isfinite(ref), ref, 0.0)
                rows = np.arange(len(S))
                new_lin = np.maximum(s_lin[rows, jmax], args.thicken * ref)
                grew = float((new_lin > s_lin[rows, jmax]).mean())
                s_lin[rows, jmax] = new_lin
                print(f"[export] vertical thickening: min {args.thicken}x of mean "
                      f"horizontal scale, {grew * 100:.0f}% gaussians grew")
            # positions were multiplied by s; sizes must follow or the splats
            # no longer match the geometry they represent
            s_lin *= s
            for k in range(3):
                arr[f"scale_{k}"] = np.log(np.maximum(s_lin[:, k], 1e-9)).astype(np.float32)
            print(f"[export] gaussian scales x{s:.4f} to match the rescale; median "
                  f"size {rb.safe_median(s_lin, 0.0, label='median gaussian size'):.4f} m")

        # ---- clean unrotated higher-order spherical harmonics ----
        # In world coordinates (rotated by Rg), unrotated SH bands create
        # artificial iridescent green and magenta/purple color banding.
        # Zeroing f_rest restores pure, true-to-life diffuse colors.
        rest_count = 0
        for name in arr.dtype.names:
            if name.startswith("f_rest_"):
                arr[name] = 0.0
                rest_count += 1
        if rest_count > 0:
            print(f"[export] neutralized {rest_count} unrotated SH rest channels (eliminated green/purple iridescent tint)")

    if args.drop_backdrop:
        n_before = len(arr)
        arr = arr[region]
        P = P[region]
        # Downstream code indexes `arr`/`P` through the region mask (spawn,
        # heightfield, collider box). Once the cloud IS the region, the mask
        # becomes identity — otherwise a boolean of length n_before would
        # raise IndexError on the shortened arrays.
        region = np.ones(len(arr), dtype=bool)
        print(f"[export] --drop-backdrop: kept {len(arr)}/{n_before} gaussians "
              f"inside the multi-view region ({100 * len(arr) / max(n_before, 1):.0f}%)")
        if len(arr) == 0:
            raise rb.StepError(
                rb.EMPTY_INPUT,
                f"--drop-backdrop removed all {n_before} gaussians: the multi-view "
                f"region ({min_views}+ views within {max_range:.3f} units of "
                f"{len(cams)} cameras) matched none of them. Re-run without the "
                f"flag, or fix solve_frame.py's --min-views / --max-range-mult.",
                returncode=3)

    scene_ply = out / "scene.ply"
    if args.from_scene:
        print(f"[export] from-scene: leaving {scene_ply.name} untouched")
    else:
        # PlyData.read mmaps its source; on Windows writing back to the same
        # path is Errno 22. In from_scene mode we do not rewrite at all.
        if scene_ply.exists() and scene_ply.resolve() == args.ply.resolve():
            raise rb.StepError(
                rb.FAILED,
                f"--ply {args.ply} is the file this step writes ({scene_ply}). "
                f"Re-exporting an already-exported scene needs --from-scene.",
                returncode=2)
        try:
            PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(scene_ply))
        except (OSError, ValueError) as e:
            raise rb.StepError(
                rb.FAILED, f"cannot write {scene_ply}: {type(e).__name__}: {e}",
                returncode=2) from e
    print(f"[export] scene.ply: {len(arr)} gaussians, up=+Y, meters")

    # ---- ground heightfield, region only ----
    # Feeding the whole cloud here is what produced a 110 x 64 m grid of which
    # only 4% of cells had real support: the backdrop set the footprint and the
    # walkable core became a handful of pixels in the corner.
    cell_req = args.cell_meters if args.cell_meters > 0 else None
    if args.cell_meters < 0:
        rb.warn(f"--cell-meters {args.cell_meters:g} m is not a length; using the "
                f"cell derived from gaussian density instead")
    h_span_x, h_span_z = float(hi[0] - lo[0]), float(hi[2] - lo[2])
    try:
        H, hlo, cell, cover = rasterize_ground(P[region], args.res, args.percentile,
                                               cell=cell_req)
    except (ValueError, ZeroDivisionError, FloatingPointError) as e:
        if cell_req:
            repair = (f"--cell-meters {args.cell_meters:g} m is finer than this "
                      f"scene can support")
        elif min(h_span_x, h_span_z) <= 0:
            repair = (f"the region has no horizontal extent (x {h_span_x:.3f} m by z "
                      f"{h_span_z:.3f} m), so there is no floor to grid; pass "
                      f"--cell-meters to force a cell size")
        else:
            repair = f"--percentile {args.percentile:g} must lie within 0-100"
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"the ground heightfield could not be rasterized from "
            f"{int(region.sum())} region gaussians: {type(e).__name__}: {e}\n"
            f"  {repair}", returncode=3) from e
    if not rb.finite(cell) or cell <= 0:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"the heightfield cell size came out as {cell!r} m; every grid index, "
            f"spawn cell and window in this step divides by it. --cell-meters "
            f"{args.cell_meters:g}", returncode=3)
    nz, nx = H.shape
    # How far above the measured surface a gaussian still counts as ground. The
    # old bare 2.5 was tuned on human-scale takes: on a hamster diorama it
    # swallows the ceiling, and on a coarse outdoor grid it can sit under the
    # heightfield's own noise. Half this scene's relief, never finer than a few
    # cells, and capped at 2.5 m so any scene with >= 5 m of relief (every scene
    # the constant was measured on) keeps the old band exactly.
    relief = max(float(hi[1]) - float(lo[1]), 0.0)
    near_band = float(min(2.5, max(4.0 * cell, 0.5 * relief)))
    print(f"[export] ground band: a gaussian within {near_band:.2f} m of the "
          f"surface counts as floor (region relief {relief:.2f} m, cell "
          f"{cell:.3f} m; 2.5 m is the human-scale cap)")

    # ---- ground the take actually stood on ----
    # A percentile of the cloud sees the top of whatever is in the column, so in a
    # seat bank the "ground" is upholstery and the collider is built on chairs.
    # Each camera pose is a direct sample of the floor it filmed: held
    # camera_agl_m above it. Only trusted at person scale — the drone-default
    # scenes carry anchors of 11-23 m that would flatten real terrain.
    # The 0.8-3.0 m band is deliberately still absolute: it judges how high the
    # capture rig was held, which is a property of the take and not of the
    # scene's modelled scale, so shrinking it for a small world would be wrong.
    agl = _num(fr, "camera_agl_m", 0.0)
    if args.camera_ground > 0 and poses_file.exists() and 0.8 <= agl <= 3.0:
        C = []
        for row in usable_poses(poses_file):
            c = row["camera"]
            try:
                Rc = np.array(c["R_rowmajor"], np.float64).reshape(3, 3)
                C.append(-(Rc.T @ np.array(c["t"], np.float64)))
            except (ValueError, TypeError):
                continue
        if C:
            C = (np.array(C) @ Rg.T) * s                  # COLMAP -> world metres
            C = C[np.all(np.isfinite(C), axis=1)]
        if len(C) == 0:
            rb.warn("camera-measured ground skipped: no finite camera position in "
                    f"{poses_file.name}")
        else:
            before = float((cover > 0).mean())
            H, cover, filled, confirmed, lowered = camera_ground(
                H, cover, cell, hlo, C, agl, args.camera_ground)
            print(f"[export] camera-measured ground from {len(C)} poses at {agl:.2f} m: "
                  f"{filled} empty cells filled, {confirmed} confirmed, {lowered} lowered "
                  f"off furniture; ground coverage {before * 100:.0f}% -> "
                  f"{(cover > 0).mean() * 100:.0f}% of the grid")
    elif args.camera_ground > 0 and poses_file.exists():
        print(f"[export] camera-measured ground skipped: anchor is {agl:.1f} m, "
              f"outside the 0.8-3.0 m person-scale band it is valid for")

    save_heightfield(H, hlo, cell, np.eye(3), out, args.max_step, cover)

    def pick_spawn(pxz: np.ndarray, py: np.ndarray) -> tuple[float, float]:
        """Density centroid of near-ground gaussians, snapped to a flat cell.

        Restricted to cells with real gaussian support: the flattest patch of a
        diffused-hole region is perfectly flat and perfectly imaginary.
        """
        hlo_x, hlo_z = float(hlo[0]), float(hlo[2])
        if len(pxz) == 0:
            rb.warn("no region points to spawn on; using the grid centre")
            return hlo_x + nx * cell / 2, hlo_z + nz * cell / 2
        fb_x = rb.safe_median(pxz[:, 0], hlo_x + nx * cell / 2, label="spawn x default")
        fb_z = rb.safe_median(pxz[:, 1], hlo_z + nz * cell / 2, label="spawn z default")
        gz = np.clip((pxz[:, 1] - hlo_z) / cell, 0, nz - 1.001).astype(int)
        gx = np.clip((pxz[:, 0] - hlo_x) / cell, 0, nx - 1.001).astype(int)
        near = np.abs(py - H[gz, gx]) < near_band
        if not near.any():
            # Nothing sits on the measured surface: a scan of a wall, or a floor
            # the heightfield invented everywhere. Prefer supported cells, then
            # any region point, over an empty selection that cannot be median'd.
            rb.warn(f"spawn: no gaussian within {near_band:.2f} m of the surface; "
                    f"widening to the {int((cover[gz, gx] > 0).sum())} supported "
                    f"points of {len(pxz)}")
            near = cover[gz, gx] > 0
        if not near.any():
            rb.warn("spawn: no ground-supported cell at all; centroiding every "
                    f"{len(pxz)} region point instead")
            near = np.ones(len(pxz), dtype=bool)
        cx = rb.safe_median(pxz[near, 0], fb_x, label="spawn median x")
        cz = rb.safe_median(pxz[near, 1], fb_z, label="spawn median z")
        # snap to the flattest supported 5x5 neighbourhood near the centroid
        ci, cj = rb.clamp_index(int((cz - hlo_z) / cell),
                                int((cx - hlo_x) / cell), (nz, nx))
        best, bij = None, (ci, cj)
        for i in range(max(2, ci - 12), min(nz - 2, ci + 13)):
            for j in range(max(2, cj - 12), min(nx - 2, cj + 13)):
                if not cover[i - 2:i + 3, j - 2:j + 3].all():
                    continue
                slope = float(np.ptp(H[i - 2:i + 3, j - 2:j + 3]))
                if best is None or slope < best:
                    best, bij = slope, (i, j)
        if best is None:
            rb.warn(f"no flat, fully supported 5x5 patch within the search window "
                    f"of grid cell ({ci},{cj}); spawning there untested")
        i, j = rb.clamp_index(bij[0], bij[1], (nz, nx))
        # cell centre, not corner: every rasterizer here indexes with floor(), so
        # cell j spans [hlo_x + j*cell, hlo_x + (j+1)*cell). Returning the corner
        # puts the spawn half a cell (0.32 m) diagonally outside the flat 5x5
        # neighbourhood that was just chosen for being flat.
        x = hlo_x + (j + 0.5) * cell
        z = hlo_z + (i + 0.5) * cell
        # A cell centre is only a meaningful answer while the grid is finer than
        # the scene. At a --cell-meters coarse enough to cover the room in one
        # cell it sits metres outside the ground it was picked from, so hold the
        # spawn inside the region it came from - a no-op for any real cell size.
        if not (lo[0] <= x <= hi[0] and lo[2] <= z <= hi[2]):
            rb.warn(f"spawn cell centre ({x:.2f}, {z:.2f}) is outside the region "
                    f"x[{lo[0]:.2f}..{hi[0]:.2f}] z[{lo[2]:.2f}..{hi[2]:.2f}] - a "
                    f"{cell:.2f} m cell is coarse next to this scene; clamped")
            x = min(max(x, float(lo[0])), float(hi[0]))
            z = min(max(z, float(lo[2])), float(hi[2]))
        return x, z

    # ---- eval cameras (same reorientation + scale) ----
    pairs = []
    spawn = None
    walk_path = None
    rows = usable_poses(poses_file) if poses_file.exists() else []
    if not rows:
        # Without poses there is no eval camera set and no measured floor to pick
        # a spawn cell on, but the geometry assets are already written, so put
        # the player a third of the way into the region's own bounds.
        rb.warn("no camera poses to export: "
                + (f"{poses_file} does not exist yet - run parse_colmap first"
                   if not poses_file.exists()
                   else f"{poses_file.name} holds no complete camera record"))
        spawn = {"x": float(lo[0] + (hi[0] - lo[0]) * 0.3),
                 "z": float(lo[2] + (hi[2] - lo[2]) * 0.3),
                 "face_xz": [rb.safe_mean(P[:, 0], float(lo[0]), label="cloud center x"),
                             rb.safe_mean(P[:, 2], float(lo[2]), label="cloud center z")]}
    else:
        step = max(1, len(rows) // max(1, args.max_eval_cams))
        chosen = rows[::step][:args.max_eval_cams]
        eval_cams = []
        for row in chosen:
            c = row["camera"]
            try:
                Rw = np.array(c["R_rowmajor"], np.float64).reshape(3, 3) @ Rg.T
                t = np.array(c["t"], np.float64) * s        # scale about origin
            except (ValueError, TypeError) as e:
                rb.warn(f"pose {row['file']}: unreadable camera matrix ({e}); skipped")
                continue
            eval_cams.append({
                "name": row["file"], "t_sec": row.get("t_sec"),
                "R_rowmajor": [list(map(float, r)) for r in Rw],
                "t": list(map(float, t)),
                **{k: float(c[k]) for k in ("fx", "fy", "cx", "cy")},
            })
        # The eval renderer reads width/height straight out of poses.json, so a
        # camera with no size is as useless as one with no frame. The extracted
        # jpg is the authority; video_meta and the COLMAP principal point are
        # this scene's own measurements of the same rectangle.
        meta = rb.read_json(args.work / "video_meta.json", {}) or {}
        guessed = []
        for cam in eval_cams:
            size = frame_size(args.work, cam["name"])
            if size is None:
                w, h = meta.get("width"), meta.get("height")
                if rb.finite(w, h) and w > 0 and h > 0:
                    size = (int(w), int(h))
                else:
                    # COLMAP puts the principal point at the frame centre.
                    size = (int(round(2 * cam["cx"])), int(round(2 * cam["cy"])))
                guessed.append(cam["name"])
            cam["width"], cam["height"] = size
        if guessed:
            rb.warn(f"{len(guessed)}/{len(eval_cams)} eval frames are not on disk "
                    f"(e.g. {guessed[0]}); their size came from the scene's own "
                    f"metadata instead: {', '.join(guessed[:3])}")
        rb.write_json(out / "poses.json", eval_cams, indent=1)
        pairs = [{"render_file": f"eval_{i:02d}.png", "real_file": cam["name"],
                  "t_sec": cam["t_sec"]} for i, cam in enumerate(eval_cams)]
        rb.write_json(args.work / "eval_pairs.json", pairs, indent=1)
        print(f"[export] {len(eval_cams)} eval cameras -> poses.json; eval_pairs.json")

        # Export ALL camera frustums, sparse tie-points, and coverage grid for 3D viewer
        try:
            from check_coverage import compute_camera_frustums, parse_points3d, analyze_coverage_grid
            all_cams = compute_camera_frustums(rows, Rg, s, frustum_depth=0.22)
            rb.write_json(out / "cameras.json", all_cams, indent=1)
            print(f"[export] {len(all_cams)} camera frustums -> cameras.json")

            pts_colmap, rgbs, tracks = parse_points3d(args.work / "colmap" / "sparse" / "txt" / "points3D.txt")
            if len(pts_colmap) > 0:
                pts_w = (pts_colmap @ Rg.T) * s
                max_pts = 60000
                if len(pts_w) > max_pts:
                    st = len(pts_w) // max_pts
                    pts_sub, rgb_sub, tr_sub = pts_w[::st], rgbs[::st], tracks[::st]
                else:
                    pts_sub, rgb_sub, tr_sub = pts_w, rgbs, tracks
                sparse_exp = {
                    "count": len(pts_sub),
                    "points": [[round(float(v), 4) for v in p] for p in pts_sub],
                    "colors": [[int(v) for v in c] for c in rgb_sub],
                    "tracks": [int(v) for v in tr_sub],
                }
                rb.write_json(out / "sparse_points.json", sparse_exp, indent=None)
                print(f"[export] {len(pts_sub)} sparse points -> sparse_points.json")

                cam_p = np.array([c["pos"] for c in all_cams], np.float64)
                if cam_p.size == 0:
                    raise ValueError("no camera frustums to bound the coverage grid")
                cam_p = cam_p.reshape(-1, 3)
                near_m = np.min(np.linalg.norm(pts_w[:, None, :] - cam_p[None, ::10, :], axis=2), axis=1) < 5.0
                pts_rm = pts_w[near_m] if np.sum(near_m) > 100 else pts_w
                c_lo = np.array([rb.safe_min(cam_p[:, k], 0.0, label=f"camera min {k}")
                                 for k in range(3)])
                c_hi = np.array([rb.safe_max(cam_p[:, k], 0.0, label=f"camera max {k}")
                                 for k in range(3)])
                b_lo = np.minimum(np.percentile(pts_rm, 2, axis=0), c_lo - 0.5)
                b_hi = np.maximum(np.percentile(pts_rm, 98, axis=0), c_hi + 0.5)
                cov_data = analyze_coverage_grid(all_cams, pts_w, b_lo, b_hi)
                rb.write_json(out / "coverage_grid.json", cov_data, indent=1)
                print(f"[export] coverage grid -> coverage_grid.json ({cov_data['covered_pct']}% covered)")
        except Exception as e:
            print(f"[export] WARNING: could not generate coverage assets: {e}")

        # spawn: in the dense ground-data zone (flat cell), facing region center
        Preg = P[region]
        sx, sz = pick_spawn(Preg[:, [0, 2]], Preg[:, 1])
        center_x = rb.safe_mean(Preg[:, 0], float(sx), label="region center x")
        center_z = rb.safe_mean(Preg[:, 2], float(sz), label="region center z")
        spawn = {"x": float(sx), "z": float(sz),
                 "face_xz": [center_x, center_z]}

        # walk loop: rectangle around the supported ground zone (5-95 pct),
        # inset so the autopilot stays inside reconstructed terrain
        pxz = Preg[:, [0, 2]]
        gz2 = np.clip((pxz[:, 1] - float(hlo[2])) / cell, 0, H.shape[0] - 1.001).astype(int)
        gx2 = np.clip((pxz[:, 0] - float(hlo[0])) / cell, 0, H.shape[1] - 1.001).astype(int)
        near2 = (np.abs(Preg[:, 1] - H[gz2, gx2]) < near_band) & (cover[gz2, gx2] > 0)
        sel = near2
        if not sel.any():
            sel = cover[gz2, gx2] > 0
            if sel.any():
                rb.warn(f"walk loop: nothing sits within {near_band:.2f} m of the "
                        f"surface, using all {int(sel.sum())} ground-supported "
                        f"points instead")
        if not sel.any():
            walk_path = None
            rb.warn(f"walk loop: no walkable floor - of {len(pxz)} region points, "
                    f"0 lie on a ground-supported heightfield cell. The scene has "
                    f"no floor to loop around (an outdoor scan past the last wall, "
                    f"or a --percentile that read only facade), so collision.json "
                    f"is written without a walk_path.")
        else:
            x5, x95 = rb.safe_pct(pxz[sel, 0], (5, 95), (lo[0], hi[0]),
                                  label="walk loop x")
            z5, z95 = rb.safe_pct(pxz[sel, 1], (5, 95), (lo[2], hi[2]),
                                  label="walk loop z")
            # 2.0 m is the inset the outdoor drone takes were tuned to. On a
            # scene whose supported zone is only a few metres across it inverts
            # the rectangle and puts every corner in ground that was measured as
            # outside the loop, so cap it at a fifth of this loop's own span.
            ins = max(0.0, min(2.0, 0.2 * min(float(x95) - float(x5),
                                              float(z95) - float(z5))))
            corners = [(x5 + ins, z5 + ins), (x95 - ins, z5 + ins),
                       (x95 - ins, z95 - ins), (x5 + ins, z95 - ins)]
            k0 = min(range(4), key=lambda k: (corners[k][0] - sx) ** 2 + (corners[k][1] - sz) ** 2)
            walk_path = [list(map(float, corners[(k0 + d) % 4])) for d in range(4)]
            perim = sum(np.hypot(corners[a][0] - corners[(a + 1) % 4][0],
                                 corners[a][1] - corners[(a + 1) % 4][1]) for a in range(4))
            print(f"[export] walk loop x[{x5 + ins:.1f}..{x95 - ins:.1f}] "
                  f"z[{z5 + ins:.1f}..{z95 - ins:.1f}], perimeter {perim:.0f} m "
                  f"(inset {ins:.2f} m, {int(sel.sum())} supported points)")

    # stash spawn + walk path + the derived collider box into collision.json
    col = rb.read_json(out / "collision.json")
    if not isinstance(col, dict) or not col:
        # save_heightfield wrote this a few lines above. If it is not there now,
        # the grid still exists in memory and heights.f32 is already on disk, so
        # describe the grid from the values in hand instead of dying on a read.
        rb.warn(f"{(out / 'collision.json').name} is not what save_heightfield "
                f"should have left; writing the grid description from this run")
        col = {"origin_xz": [float(hlo[0]), float(hlo[2])], "cell": float(cell),
               "nx": int(nx), "nz": int(nz),
               "rotation_rowmajor": np.eye(3).tolist(),
               "max_step": float(args.max_step), "has_coverage": True}
    col["spawn"] = spawn
    if walk_path:
        col["walk_path"] = walk_path

    # Collider box, published so the voxeliser stops needing a hand-tuned
    # constant. Tight around the walkable region, in this file's frame
    # (up=+Y, metres). Deliberately NOT padded far above the terrain: the grid
    # ceiling is where --voxel-floor-fill dumps its slab, and the old hardcoded
    # box is why the character spawned on an invisible plateau. The generous pad
    # is on the BOTTOM, where a solid floor under the terrain is harmless.
    # 15% of the region's own span, capped at the 1.0 m the human-scale takes
    # were tuned to and floored so a sub-decimetre cloud still gets a real box:
    # a bare 1.0 m on a hamster-scale room doubles the collider and buries the
    # character in the floor-fill slab.
    span = float(np.max(hi - lo))
    pad = float(min(1.0, max(0.05, 0.15 * span)))
    col["content_bounds"] = {
        "min": [float(lo[0]), float(lo[1]), float(lo[2])],
        "max": [float(hi[0]), float(hi[1]), float(hi[2])],
    }
    col["collider_box"] = {
        "min": [float(lo[0] - pad), float(lo[1] - pad), float(lo[2] - pad)],
        "max": [float(hi[0] + pad), float(hi[1] + pad), float(hi[2] + pad)],
    }
    col["region_box"] = col["collider_box"]
    col["scale_m_per_unit"] = s
    if args.character_height is not None:
        ch = float(args.character_height)
        if rb.finite(ch) and ch > 0:
            col["character_height"] = ch
        else:
            rb.warn(f"--character-height {ch} is not a height; leaving it out so "
                    f"the viewer keeps its own default")
    try:
        rb.write_json(out / "collision.json", col)
    except (OSError, TypeError, ValueError) as e:
        raise rb.StepError(
            rb.FAILED,
            f"cannot write {out / 'collision.json'}: {type(e).__name__}: {e}\n"
            f"  Every step after this one reads it, so the assets in {out} are "
            f"incomplete.", returncode=2) from e
    print(f"[export] spawn at x={spawn['x']:.1f} z={spawn['z']:.1f} facing scene center")
    cb = col["collider_box"]
    print(f"[export] collider box -B "
          f"{cb['min'][0]:.2f},{cb['min'][1]:.2f},{cb['min'][2]:.2f},"
          f"{cb['max'][0]:.2f},{cb['max'][1]:.2f},{cb['max'][2]:.2f} "
          f"(pad {pad:.2f} m)")


if __name__ == "__main__":
    rb.configure_streams()
    try:
        main()
    except rb.StepError as e:
        print(f"\n[export] {e}", file=sys.stderr, flush=True)
        sys.exit(e.returncode)
