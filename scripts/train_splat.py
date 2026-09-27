"""Turbocharged 3D Gaussian Splatting trainer for high quality & fast convergence.

Optimizations:
  1. Pre-cached GPU tensors: all viewmats, Ks, and images reside directly in CUDA VRAM.
  2. Native PyTorch Adam optimizer with seamless parameter resizing via DefaultStrategy.
  3. Fast channel-grouped vectorized SSIM in PyTorch.
  4. Random background regularization: destroys smoky floaters in unobserved air/ceiling.
  5. AbsGrad strategy (absgrad=True) + opacity reset intervals for razor-sharp geometric detail.
  6. Numerical stability clamps on scales, quats, and opacities.
  7. Sparse-depth consistency (challenge D2): the rendered depth the rasterizer already
     produces is pulled onto the depth the COLMAP cloud already observed, so a textureless
     wall has a constraint even where it has no gradient. See "sparse-depth supervision".
     Measured on work/rocks: held-out cross-view depth disagreement 4.8-5.3x lower against
     a 5% control repeat spread, +12-14% step time, no change in peak VRAM. Held-out colour
     is NOT resolved: two paired blocks gave -0.24/-0.17 dB, which is inside the 0.69 dB
     spread of three identical control runs, so the term demonstrably buys geometry and
     demonstrably does not buy colour. --depth-weight 0 --no-depth-measure adds no term to
     the loss and renders no depth channel on any step. That was measured on the WRONG
     scene class (rocks is texture-rich and fully registered); see "D2 on the textureless
     class" below for the measurement on the scenes it is for.

Two measurement contracts this file has had to grow, because it was shipping numbers
that were not about the model:

  A. A metric or an exported cloud may never describe an opacity-reset step. gsplat's
     DefaultStrategy clamps EVERY opacity to `prune_opa * 2` = 0.04 at the end of every
     RESET_EVERY-th step below refine_stop (gsplat 1.5.3 strategy/default.py:162,195-199),
     so any --steps that is a multiple of 3000 under the refine stop ENDED on one.
     Reproduced on work/rocks, seed 0, 640x360, 1-in-6 held out, depth term off:
       --steps 6000  held-out PSNR 16.14 dB, SSIM 0.5545  <- the reset, not the model
       --steps 5999  held-out PSNR 34.16 dB, SSIM 0.8929  <- one step earlier
     and splat.ply from the 6000 run has all 167,609 opacities sitting exactly on the
     0.04 clamp, so the shipped artefact was the demoted cloud too. The trainer now
     captures the pre-clamp opacities on the steps gsplat resets (ResetCapture), exports
     and scores with them, and REFUSES to report at all if the capture did not happen
     (artefact_reset_state). The run is not lengthened, gsplat's schedule is not altered,
     and train_report.json's `final_eval` says which of the three branches ran, with the
     max opacity as gsplat left it next to the max opacity that was reported.
  B. Nothing in here is a metre. Scene units are arbitrary until solve_frame anchors
     them (work/rocks/frame.json: scale_m_per_unit 4.734, scale_source "flight speed x
     clip duration"), and the room-anchor seeds used to be placed at "1.5 / 2.5 / 3.5
     metres" in front of every camera -- in scene units, so 7.1 / 11.8 / 16.6 m on rocks,
     where the cloud the cameras could actually see sits at a median depth of 9.04 units
     = 42.8 m. The seeds were 26-36 m of empty air in front of the rock, and the run
     still printed "seeded 180 spatial anchor points on room walls" over an outdoor
     orbit. Anchor distances are now measured per camera from this scene's own cloud
     (anchor_seed_distances: the depth quantiles of the points inside that camera's
     frustum, in scene units, with a camera-hull fallback that is also a measurement),
     the jitter is a fraction of that distance, and the whole prior is gated on what the
     room detector measured (room_prior_decision reads viewer_assets/rooms.json from
     scripts/detect_rooms.py): work/rocks and work/auditorium decline it with the
     detector's own floor-fragmentation text, work/roomscan declines it with one wall
     plane where two are needed, and work/room_w_jsonl applies it at its measured
     p25/p50/p75 = 1.20/1.47/1.72 units = 1.25/1.54/1.79 m.

Usage:
  python train_splat.py --work work/room_w_jsonl --steps 12000 --cap 650000
"""
import argparse
import hashlib
import json
import math
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_intrinsics import camera_matrix_and_distortion  # noqa: E402
from parse_colmap import qvec2rot  # noqa: E402
import robust as rb  # noqa: E402

SH_DEG = 3

# Densification starts here, and the depth term's warmup is measured from the same
# step: arming a geometry prior against a cloud that has not been placed yet is how
# these runs diverge.
REFINE_START = 500

# gsplat's DefaultStrategy clamps EVERY opacity down to `prune_opa * 2` (0.04
# here) on this interval -- .venv310/.../gsplat/strategy/default.py:195-199 --
# inside a callback that returns early from `refine_stop_iter` onward (same file,
# line 162). The trainer declares the interval itself instead of passing a bare
# literal, because the reporting has to know exactly when a reset lands: a metric
# or an exported cloud taken on the end of such a step is a measurement of the
# reset, not of the model (see reset_lands_on).
RESET_EVERY = 3000          # steps between opacity resets
# How many steps after a reset an in-flight reading is still worth annotating on
# the log. Measured, work/rocks, the single-view training PSNR in the step line
# across one reset at step 3000: 31.85 at 3000 (rendered before the clamp), 31.59
# at 3200, 32.44 at 3400, 33.71 at 3600 against 30-32 either side -- the dip is
# real and it is gone inside a few hundred steps. 600 is that, rounded up: it
# annotates the log, and no number that gets reported comes from inside it.
RESET_RECOVERY_TAG_STEPS = 600   # steps

# Sparse-depth term constants. Every one of these is a count, a fraction or a
# pixel patch -- deliberately not a length, because the scene's units are
# arbitrary until solve_frame anchors them.
#
# A point triangulated from this many images is worth half of one triangulated
# from infinitely many. COLMAP's own mean track length on a good outdoor solve is
# 8-15 views, so 4 puts a well-constrained point near 0.7 and a 2-view point at 0.33.
POINT_VIEWS_HALF = 4.0
# Below these, the term refuses to run and says why, rather than training on
# noise: a per-view median of 100 projected points over a 640x360 frame is one
# point per ~2300 pixels, and 2% frame coverage is about the point where the
# supervision stops being a few scattered stamps.
MIN_POINTS_PER_VIEW = 100
MIN_FRAME_COVERAGE = 0.02
# Per-pixel tolerance floor, as a multiple of the cloud's own measured 3-nearest
# neighbour spacing: disagreement smaller than the granularity of the evidence is
# not evidence of an error. 1.0 = one spacing.
NN_SPACING_FLOOR = 1.0
# A rendered pixel below this accumulated opacity has no surface to compare a
# depth against, so it is left unsupervised rather than pulled toward whatever
# the expected-depth sum of an empty pixel happens to be.
ALPHA_MIN_FOR_DEPTH = 0.5


# ---------------- reset-aware step scheduling (defect: the 15 dB reading) ----------------
# `--steps 6000` with the default `--refine-stop 9000` used to end the run on an
# opacity-reset step. Two things went wrong at once and only one of them was
# visible: the final held-out PSNR came out ~15 dB, and the exported splat.ply was
# the 0.04-opacity cloud that reading describes. Every A/B at such a step count
# therefore measured the reset, silently, and read as a 24 dB regression.
#
# These helpers are pure arithmetic on step counts (no torch, no GPU) so a CPU test
# can pin them against gsplat's own source.
def reset_lands_on(step: int, reset_every: int = RESET_EVERY,
                   refine_stop: int | None = None) -> bool:
    """True when gsplat resets every opacity at the END of `step`.

    Mirrors the installed strategy, not a guess about it: default.py:195 fires on
    `step % reset_every == 0 and step > 0`, and default.py:162 returns before that
    for `step >= refine_stop_iter`, so a reset at or past the refine stop never
    happens. A `refine_stop` of None means "this run does not stop refining", which
    is the same as a number above `step`.
    """
    if reset_every <= 0 or step <= 0:
        return False
    if refine_stop is not None and step >= refine_stop:
        return False
    return step % reset_every == 0


def artefact_reset_state(step: int, reset_every: int = RESET_EVERY,
                         refine_stop: int | None = None,
                         snapshot_ready: bool = True):
    """(is the cloud demoted now, restore the trained opacities, refuse to report).

    The rule the trainer has to follow at a reset step, as a pure decision so a CPU
    test can pin all three branches:

      * not a reset step -> nothing special;
      * reset step, and the pre-reset opacities were captured -> export and
        measure the captured cloud, and say that is what happened;
      * reset step, no capture (a gsplat whose internals moved, or a cap trim that
        changed the row count after the capture) -> REFUSE. Reporting the demoted
        cloud as a model is what produced the 16 dB reading; refusing is the only
        answer that cannot be misread.
    """
    demoted = reset_lands_on(step, reset_every, refine_stop)
    if not demoted:
        return False, False, False
    if snapshot_ready:
        return True, True, False
    return True, False, True


class ResetCapture:
    """Keep the opacities gsplat is about to clamp away, for one step.

    DefaultStrategy.step_post_backward ends a reset step with
    `reset_opa(params=..., optimizers=..., state=..., value=self.prune_opa * 2.0)`
    (gsplat 1.5.3, default.py:195-199), which replaces `params["opacities"]` with a
    copy clamped to 0.04 and zeroes the Adam state under it. Wrapping that call for
    the duration of one `step_post_backward` is the only place the pre-clamp vector
    exists at the post-refine row ordering: the densify/prune in the same call has
    already added and removed rows by then, so a snapshot taken before it would not
    line up. Nothing here reimplements gsplat's schedule, and the patch is on the
    module attribute for exactly the one call it is entered around.

    `.opas` stays None if the parent did not reset, which the caller turns into a
    refusal to report rather than a quiet miss.
    """

    def __init__(self):
        self.opas = None
        self.value = None
        self._restore = None

    def __enter__(self):
        import gsplat.strategy.default as gd
        orig = gd.reset_opa
        rec = self

        def wrapped(*args, **kw):
            params = kw.get("params")
            if params is None and args:
                params = args[0]
            if params is not None and "opacities" in params:
                rec.opas = params["opacities"].detach().clone()
                rec.value = kw.get("value")
            return orig(*args, **kw)

        gd.reset_opa = wrapped
        self._restore = lambda: setattr(gd, "reset_opa", orig)
        return self

    def __exit__(self, *exc):
        if self._restore is not None:
            self._restore()
        return False


@contextmanager
def trained_opacities(params, snapshot):
    """Hold the pre-reset opacities in `params` for the duration of a block.

    In-place on `.data`, so the Parameter object the optimizers are bound to never
    changes identity and the demoted values come straight back on exit: the
    training schedule a mid-run capture restores into is untouched. Yields True when
    it actually restored, and a mismatched row count restores nothing -- guessing
    which rows moved is how a wrong cloud gets shipped.
    """
    if snapshot is None:
        yield False
        return
    live = params["opacities"]
    if snapshot.shape != live.detach().shape:
        yield False
        return
    demoted = live.detach().clone()
    with torch.no_grad():
        live.data.copy_(snapshot)
    try:
        yield True
    finally:
        with torch.no_grad():
            live.data.copy_(demoted)


# ---------------- is this scene a room at all? ----------------
# The room-anchor seeding below is a prior about interiors: walls in front of the
# camera, at wall distance. It used to run unconditionally and used to print
# "seeded N spatial anchor points on room walls" over an outdoor rock orbit.
# The finished detector (scripts/detect_rooms.py -> work/<scene>/viewer_assets/
# rooms.json) is what says whether that prior applies, so the decision is read
# from its measurement instead of being assumed.
#
# Every threshold here is a COUNT of things the detector already measured; none is
# a length, and none is a metre. What a wall or a floor has to be to count is
# decided in detect_rooms.py, in metres, and stated in that file's own `thresholds`
# block -- which is why this gate can stay scale-free.
ROOM_PRIOR_MIN_WALLS = 2        # fitted wall planes; one wall cannot place "walls"


def room_prior_decision(work: Path) -> tuple[bool, str, dict]:
    """(apply the room prior, why, evidence) from viewer_assets/rooms.json.

    Deliberately dependency-free (json + Path, no robust, no torch) so the CPU lane
    can exec it. It reports the detector's own words when it declines, because "no"
    without a reason is how a gate becomes a coin flip. An absent rooms.json is an
    absent measurement, not a negative one: it declines too, and says the file was
    not there rather than that the scene was measured to be outdoors.
    """
    path = Path(work) / "viewer_assets" / "rooms.json"
    if not path.is_file():
        return False, ("no viewer_assets/rooms.json - detect_rooms.py has not run on "
                       "this scene, so nothing has measured whether it is an "
                       "interior; an unmeasured scene is not a room"), {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        return False, f"viewer_assets/rooms.json is unreadable ({exc})", {}
    rooms = doc.get("rooms") or []
    units = doc.get("units") or {}
    ev = {"path": str(path), "n_rooms": len(rooms),
          "metric_confidence": units.get("metric_confidence"),
          "scale_source": units.get("scale_source")}
    if not rooms:
        refusal = doc.get("refusal") or {}
        why = refusal.get("floor") or refusal.get("rooms_found")
        return False, ("rooms.json says this scene has no room: "
                       f"{why}" if why else
                       "rooms.json reports rooms: [] with no refusal text"), ev
    room = rooms[0]
    walls = room.get("walls") or []
    floor = room.get("floor") or {}
    enclosure = room.get("enclosure") or {}
    n_walls = int(enclosure.get("wall_count", len(walls)))
    ev.update(walls=n_walls,
              floor_coherent_share=floor.get("coherent_share_of_floor"),
              floor_area_m2=floor.get("observed_area_m2"),
              wall_planes=[{"azimuth_deg": w.get("azimuth_deg"),
                           "length_m": w.get("length_m"),
                           "inlier_splats": w.get("inlier_splats")} for w in walls])
    if n_walls < ROOM_PRIOR_MIN_WALLS:
        return False, (f"rooms.json detected an interior with only {n_walls} fitted "
                       f"wall plane(s); a room prior needs >= {ROOM_PRIOR_MIN_WALLS}"
                       + (f" ({enclosure.get('note')})" if enclosure.get("note")
                          else "")), ev
    return True, (f"rooms.json: {len(rooms)} room(s), floor coherent share "
                  f"{ev['floor_coherent_share']}, {n_walls} fitted wall planes "
                  f"(metric_confidence {ev['metric_confidence']}, "
                  f"scale_source {ev['scale_source']})"), ev



# ---------------- COLMAP TXT parsing ----------------
def load_colmap(txt_dir: Path):
    cams = {}
    for line in (txt_dir / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        # p[2], p[3] are the width/height COLMAP solved at — the only way to know
        # whether the images on disk still match these intrinsics.
        cams[int(p[0])] = dict(model=p[1], params=list(map(float, p[4:])),
                               w0=int(p[2]), h0=int(p[3]))

    imgs = []
    for line in (txt_dir / "images.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if len(p) != 10:
            continue
        q = np.array(list(map(float, p[1:5])))
        t = np.array(list(map(float, p[5:8])))
        imgs.append(dict(name=p[9], R=qvec2rot(q), t=t, cam_id=int(p[8])))
    return cams, imgs


def load_points3d(txt_dir: Path):
    """(xyz, rgb, err_px, n_views) for every point the solve triangulated.

    ERROR and TRACK[] are loaded alongside the geometry because the depth term
    has to know how hard to believe each point, and COLMAP already reports both
    in scene-scale-free units: the mean reprojection error in PIXELS and the
    track length in VIEWS. A wall with three 4-view points and a wall with three
    thousand 15-view points are not the same evidence.
    """
    xyz, rgb, err, nviews = [], [], [], []
    for line in (txt_dir / "points3D.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 8:
            continue
        try:
            e = float(p[7])
        except ValueError:
            e = float("nan")
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
        rgb.append([int(p[4]), int(p[5]), int(p[6])])
        err.append(e)
        nviews.append(max(0, (len(p) - 8) // 2))
    return (np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.float32),
            np.asarray(err, dtype=np.float32), np.asarray(nviews, dtype=np.float32))


# ---------------- data preparation ----------------
def prepare_dataset(work: Path, max_images: int | None = None):
    txt = work / "colmap" / "sparse" / "txt"
    cams, imgs = load_colmap(txt)

    und_dir = work / "frames_undist"
    und_dir.mkdir(exist_ok=True)

    data = []
    src_sizes = set()
    for im in imgs:
        c = cams[im["cam_id"]]
        src_path = work / "frames_train" / im["name"]
        src = cv2.imread(str(src_path))
        hs, ws = src.shape[:2]
        # Undistortion must use intrinsics in the decoded source pixel grid.
        K, dist = camera_matrix_and_distortion(
            c["model"], c["params"], (c["w0"], c["h0"]), (ws, hs))
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                training_K = K.astype(np.float32)
        except FloatingPointError as exc:
            raise ValueError("Camera matrix K is not representable in float32") from exc
        if not np.isfinite(training_K).all() or training_K[0, 0] <= 0 or training_K[1, 1] <= 0:
            raise ValueError("Float32 camera matrix K must be finite with positive focal lengths")
        if np.any(dist != 0):
            # Both calibration and decoded content determine the undistorted pixels.
            source_hash = hashlib.sha256(src.tobytes()).hexdigest()
            calibration = json.dumps(
                [c["model"], c["params"], c["w0"], c["h0"], ws, hs, source_hash],
                separators=(",", ":"))
            calibration_key = hashlib.sha256(calibration.encode("utf-8")).hexdigest()[:20]
            cache = und_dir / f"{ws}x{hs}__{calibration_key}" / im["name"]
            if cache.exists():
                img_path = cache
                img = cv2.cvtColor(cv2.imread(str(cache)), cv2.COLOR_BGR2RGB)
            else:
                img = cv2.undistort(src, K, dist)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cache), img)
                img_path = cache
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_path = src_path
            img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        K = training_K
        sx, sy = w / c["w0"], h / c["h0"]
        if abs(sx - 1) > 1e-6 or abs(sy - 1) > 1e-6:
            src_sizes.add((c["w0"], c["h0"], w, h))
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = im["R"]
        viewmat[:3, 3] = im["t"]
        sharpness = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

        data.append(dict(name=im["name"], K=K, viewmat=viewmat, path=img_path, width=w, height=h,
                         sharpness=sharpness, img=img))
        if max_images and len(data) >= max_images:
            break
    print(f"[train] dataset: {len(data)} images {data[0]['width']}x{data[0]['height']}")
    for w0, h0, w, h in sorted(src_sizes):
        print(f"[train] COLMAP solved at {w0}x{h0}, training images are {w}x{h}: "
              f"intrinsics scaled x{w / w0:.3f} to match")
    sharp_arr = sorted(d["sharpness"] for d in data)
    print(f"[train] frame sharpness (Laplacian var): min {sharp_arr[0]:.1f} "
          f"median {sharp_arr[len(sharp_arr)//2]:.1f} max {sharp_arr[-1]:.1f}")
    fit_to_vram(data)
    return data, load_points3d(txt)


def fit_to_vram(data: list) -> int:
    """Cap per-frame pixels so the rasterizer can actually render this set.

    A "width" limit cannot see a portrait frame coming: 1280x2772 is 3.5 MP,
    nearly 4x a 640x763 budget, and the card was filled mid-run with the process
    killed before it could print anything. The keyframe step sizes from video
    metadata; this is the last place that knows the real decoded size.

    Returns the chosen per-frame pixel count. K scales with the resize because
    the projected gaussian positions are only right if the intrinsics follow the
    pixel grid.
    """
    from robust import available_vram_gb, pixels_for, train_budget
    w0, h0 = data[0]["width"], data[0]["height"]
    want = min(w0 * h0, 1_100_000)
    budget = train_budget(vram_gb=available_vram_gb(), max_pixels=want,
                          n_frames=len(data))
    nw, nh = pixels_for(w0, h0, budget["pixels"])
    if (nw, nh) == (w0, h0):
        return w0 * h0
    sx, sy = nw / w0, nh / h0
    for d in data:
        d["K"] = np.array([[d["K"][0, 0] * sx, 0, d["K"][0, 2] * sx],
                           [0, d["K"][1, 1] * sy, d["K"][1, 2] * sy],
                           [0, 0, 1]], dtype=np.float32)
        if d.get("img") is not None:
            d["img"] = cv2.resize(d["img"], (nw, nh), interpolation=cv2.INTER_AREA)
        d["width"], d["height"] = nw, nh
    print(f"[train] {w0}x{h0} = {w0 * h0 / 1e6:.2f} MP/frame x {len(data)} frames does "
          f"not fit {budget['vram_gb']:.1f} GiB free: re-sized to {nw}x{nh} "
          f"({nw * nh / 1e6:.2f} MP), intrinsics scaled with it")
    return nw * nh


# ---------------- fast vectorized ssim ----------------
_W_CACHE = {}


def fast_ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Vectorized PyTorch SSIM across RGB channels using grouped 2D convolution."""
    ch = img1.shape[1]
    dev = img1.device
    if ch not in _W_CACHE or _W_CACHE[ch].device != dev:
        coords = torch.arange(11, device=dev, dtype=torch.float32) - 5.0
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        _W_CACHE[ch] = (g[:, None] @ g[None, :]).expand(ch, 1, 11, 11) / (g.sum() ** 2)
    w = _W_CACHE[ch]

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.conv2d(img1, w, groups=ch, padding=5)
    mu2 = F.conv2d(img2, w, groups=ch, padding=5)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, w, groups=ch, padding=5) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, w, groups=ch, padding=5) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, w, groups=ch, padding=5) - mu12
    ssim_val = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_val.mean()


# ---------------- sparse-depth supervision (challenge D2) ----------------
# Textureless / few-view / blur: a wall with no texture produces no photometric
# gradient, so nothing tells the gaussians where it is and it collapses into
# mush. But the solve already knows where the wall is -- the sparse COLMAP cloud
# is observed geometry, and gsplat already renders a depth map as a by-product of
# the pass that produces the image. Pulling one onto the other is FSGS' and
# DNGaussian's idea (recorded independently at
# 06_Splat_Completion_Research.md:874-875, "needs no depth model"), and it costs
# one extra render channel, no download and no new dependency.
#
# UNITS, because this project has been burned by absolute constants applied to
# scale-estimated geometry: the scene's lengths are in COLMAP units, which are
# arbitrary until solve_frame anchors them (work/<scene>/frame.json carries
# scale_m_per_unit; nothing below reads it, so the loss cannot inherit a wrong
# scale). The knobs are therefore all dimensionless (weights, fractions of depth,
# point counts, pixel patches) except `spacing`, which is one measured length in
# scene units and is printed with its units.
#
# gsplat API used (installed source, gsplat 1.5.3+pt24cu124):
#   rendering.py:51   render_mode accepts "RGB+ED"
#   rendering.py:614-615  the per-gaussian camera-space z is appended as an extra
#                         colour channel, so it is alpha-composited in the SAME
#                         kernel launch as the image -- there is no second render
#   rendering.py:617-623  the depth channel's background is forced to 0, so the
#                         random-background floater trick cannot corrupt it
#   rendering.py:760-768  for "RGB+ED" the last channel is divided by render_alphas,
#                         i.e. it is the EXPECTED depth sum(w_i z_i)/sum(w_i)
#   rendering.py:230-233  return shape [..., C, H, W, D+1], depth is the last channel
#
# MEASURED, work/rocks (72 frames, 640x360, 12k steps, cap 350k, 1-in-6 held out,
# paired blocks at seed 0 and seed 1; the held-out views supervise neither colour
# nor depth, and the cross-view-depth figure never reads the sparse cloud, so it
# is not this term's own objective being re-measured):
#   rendered-vs-observed disagreement   raw 0.434 -> 0.137, err 2.249u -> 0.900u
#   held-out cross-view depth p50       0.184-0.203u -> 0.0385u  (4.8-5.3x,
#                                        p90 2.00-2.08u -> 0.72-0.78u;
#                                        1.66-1.72% -> 0.33% of the pixel's depth)
#   gaussians on observed geometry      58.3-59.6% -> 62.0-62.5%
#   cost                                +12-14% step time (2.3 ms per armed step),
#                                        peak VRAM unchanged at 640x360 (1.28 GiB
#                                        allocated / 1.33 GiB reserved in both arms),
#                                        14 MB of projected cloud + scratch resident.
#                                        Re-measured on a 1080p-class set (work/
#                                        auditorium: 395 views resized by the VRAM
#                                        budget to 1136x638 = 0.72 MP, 300 steps,
#                                        quality numbers meaningless that short):
#                                        19.9 vs 22.6 it/s (+13.6%), 2.32 GiB
#                                        allocated in BOTH arms, 3.31 vs 3.12 GiB
#                                        reserved (+0.19), 158.4 MB resident
#                                        (144.6 of it projected points) - no OOM on
#                                        this 6 GiB card at either resolution.
#   HELD-OUT COLOUR: NOT RESOLVED. Point estimates say psnr -0.24 dB (seed 0) and
#   -0.17 dB (seed 1), ssim -0.0008/-0.0007 -- but THREE identical control repeats
#   at the same seed and config gave 39.38 / 38.71 / 38.69 dB (spread 0.69 dB, N
#   184.6k / 194.8k / 194.9k gaussians), because gsplat's backward accumulates with
#   atomics and the run is therefore not bit-reproducible. A 0.2 dB effect is inside
#   that spread: this data says the term does not IMPROVE colour, and is consistent
#   with a small colour cost, but does not measure one. The depth-consistency gain
#   is 4.8x against a control spread of 5%, so that half is established.
# So on a scene that is already registered and already textured the term is a bias
# toward observed geometry, and the only thing it demonstrably buys there is geometry.
# Two couplings a reader must not miss: gsplat's absgrad (rendering.py:145-147,
# 330) accumulates |dL/d(means2d)| over EVERY rendered channel, so the fourth
# channel feeds the densifier as well as the surface - the armed cloud was 8-9%
# larger (201,082 vs 184,646) - and with 6.1% frame coverage, 12.7% of the
# supervised pixels hold more than one sparse point (occluded geometry projected
# onto the surface in front of it), which is what the per-pixel spread term in
# `sigma` is there to absorb.
#
# How big the ignored disagreement actually is, in the scene's own units and then
# in metres as work/rocks/frame.json anchors them (scale_m_per_unit 4.734, whose
# scale_source is "flight speed x clip duration" - an estimate, so the metre values
# are reported and never used by the loss): cloud 3-nearest spacing 0.0958u = 0.45 m
# as the tolerance floor, and tau x depth at the median supervised pixel
# 0.1 x 9.58u = 0.96u = 4.5 m. The disagreement the term is willing to overlook is
# thus ~10x the granularity of the evidence it is checking itself against, which is
# what "relative, scale-free" costs: it binds on gross geometry, not on detail.


def nn_spacing(xyz: torch.Tensor, n_sample: int = 20_000, k: int = 3,
               gen: torch.Generator | None = None) -> float:
    """Median distance to the k nearest neighbours of the cloud, in SCENE UNITS.

    The same statistic init_from_points() uses to set the starting gaussian
    scales, so the depth term's tolerance floor and the model's own granularity
    are measured with one ruler instead of one being guessed. Pass a private
    generator: sampling off the global stream would shift the random-background
    draws and put a seed difference into a weight-only A/B.
    """
    if len(xyz) < k + 1:
        return 0.0
    sub = xyz[torch.randperm(len(xyz), device=xyz.device, generator=gen)
              [:min(n_sample, len(xyz))]]
    d = torch.cdist(sub, sub)
    d.fill_diagonal_(float("inf"))
    return float(d.topk(min(k, len(sub) - 1), largest=False).values.mean(1).median())


def sparse_depth_targets(data: list, xyz, err_px, nviews, near: float, far: float,
                         err_ref_px: float, support_pt: float):
    """Project the cloud the trainer already loaded into every training view.

    No new file format and no new reader: this is the same pinhole projection
    gsplat applies to the gaussian centres, using the same per-view K (already
    rescaled by fit_to_vram) and the same viewmat, so an observed pixel and a
    rendered pixel mean the same thing.

    Returns (targets, report). targets[i] is None when view i has no usable point
    (nothing behind the near plane, nothing inside the frame), else the flat
    pixel indices, the observed z-depth in scene units, and each point's own
    confidence in [0, 1] -- 1/(1+err/err_ref) for the solve's reported
    reprojection error in pixels, times nviews/(nviews+support_pt) for how many
    images actually voted for the point.
    """
    idxs, zs, confs, counts = [], [], [], []
    pts = np.ascontiguousarray(xyz, dtype=np.float64)
    e = np.asarray(err_px, dtype=np.float64)
    nv = np.asarray(nviews, dtype=np.float64)
    eq = 1.0 / (1.0 + np.nan_to_num(e, nan=err_ref_px * 1e6) / max(err_ref_px, 1e-6))
    vq = nv / (nv + max(support_pt, 1e-6))
    conf_all = np.clip(eq * vq, 0.0, 1.0).astype(np.float32)

    for d in data:
        K = d["K"]
        vm = d["viewmat"]
        H, W = d["height"], d["width"]
        Xc = pts @ vm[:3, :3].T + vm[:3, 3]
        z = Xc[:, 2]
        u = K[0, 0] * Xc[:, 0] / np.where(np.abs(z) < 1e-9, 1e-9, z) + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / np.where(np.abs(z) < 1e-9, 1e-9, z) + K[1, 2]
        keep = ((z > near) & (z < far)
                & (u >= 0) & (u < W - 1e-6) & (v >= 0) & (v < H - 1e-6))
        if not keep.any():
            idxs.append(None)
            counts.append(0)
            continue
        pi = (np.floor(v[keep]).astype(np.int64) * W + np.floor(u[keep]).astype(np.int64))
        idxs.append(pi.astype(np.int32))
        zs.append(z[keep].astype(np.float32))
        confs.append(conf_all[keep])
        counts.append(int(keep.sum()))

    hit = [c for c in counts if c > 0]
    # One shared H x W grid holds every view's observed depth, so a set with
    # mixed frame sizes would silently index the wrong pixels -- and a wrong
    # pixel is worse than no pixel, because it supervises real geometry in the
    # wrong place. fit_to_vram normally equalises the set; a scene that skips it
    # does not get a depth term, and says so.
    sizes = {(int(d["width"]), int(d["height"])) for d in data}
    report = {
        "n_views": len(data),
        "n_points": int(len(pts)),
        "views_with_points": len(hit),
        "points_per_view_median": int(np.median(hit)) if hit else 0,
        "points_per_view_min": int(min(hit)) if hit else 0,
        "distinct_sizes": len(sizes),
        "coverage_median": (float(np.median([c / (d["width"] * d["height"])
                                             for c, d in zip(counts, data)]))
                            if counts else 0.0),
        "conf_median": float(np.median(conf_all)) if len(conf_all) else 0.0,
    }
    return (idxs, zs, confs), report


class DepthTerm:
    """One view's observed-depth map, built on the GPU each step it is needed.

    Deliberately recomputed rather than cached: three H x W float maps per view
    (0.9 MB each at 640x360) would be ~170 MB for 60 views and ~2 GB for a
    465-frame 1.4 MP set, and the card this runs on has 6. Rebuilding costs a
    handful of scatter kernels over a few thousand points on a fixed set of
    pre-allocated buffers.

    Two claims about its footprint, because they are not the same one:
      * the scratch is a constant times pixels and does NOT grow with views or
        points -- five W x H float maps, 13.8 MB at the 1136x638 a 1080p/395-frame
        set was resized to, 4.4 MB at 640x360;
      * the projected cloud stored in self.views DOES grow with views x points per
        view (12 bytes each): 14 MB for rocks' 60 x ~14k, but 144.6 MB for the
        auditorium set's 385 x ~30k, i.e. 158.4 MB resident in total there, which
        is 3.5% of that run's 4.5 GiB budget. Both figures are reported per run in
        train_report.json under depth.cost, so the cost is measured, not asserted.
    """

    def __init__(self, targets, H: int, W: int, device: str, spacing: float,
                 tau: float, support: float, window: int, nn_floor: float,
                 alpha_min: float):
        idxs, zs, confs = targets
        self.H, self.W, self.n = H, W, H * W
        self.device = device
        self.tau, self.support = tau, support
        self.window = max(1, int(window) | 1)
        self.alpha_min = alpha_min
        self.sigma_floor = max(0.0, float(nn_floor) * float(spacing))
        self.views = []
        for i in range(len(idxs)):
            if idxs[i] is None:
                self.views.append(None)
                continue
            self.views.append((
                torch.from_numpy(idxs[i]).to(device).long(),
                torch.from_numpy(zs[i]).to(device),
                torch.from_numpy(confs[i]).to(device),
            ))
        n_ok = sum(1 for v in self.views if v is not None)
        # 12 bytes per stored point: int32 pixel + float32 depth + float32 conf.
        self.pts_mb = sum(float(v[1].numel()) for v in self.views if v) * 12 / 1024**2
        self.buf = dict(
            conf=torch.zeros(self.n, device=device),
            zc=torch.zeros(self.n, device=device),
            zmin=torch.full((self.n,), float("inf"), device=device),
            zmax=torch.full((self.n,), float("-inf"), device=device),
        )
        # Four persistent W x H scratch maps plus one transient for the pooling:
        # the whole thing is a constant times pixels, never times views or points.
        self.stats = dict(n_views_ok=n_ok, n_views=len(self.views),
                          scratch_mb=round(5 * self.n * 4 / 1024**2, 2),
                          pts_mb=round(self.pts_mb, 2))

    def observed(self, view: int):
        """(observed depth, per-pixel weight, per-pixel tolerance) in scene units."""
        v = self.views[view]
        if v is None:
            return None
        idx, z, conf = v
        b = self.buf
        cf = b["conf"].zero_()
        cf.index_add_(0, idx, conf)
        zc = b["zc"].zero_()
        zc.index_add_(0, idx, conf * z)
        # index_reduce_ with include_self=False ignores the +inf / -inf fill, so
        # an unobserved pixel keeps inf/-inf and is dropped by the weight below.
        b["zmin"].index_reduce_(0, idx, z, reduce="amin", include_self=False)
        b["zmax"].index_reduce_(0, idx, z, reduce="amax", include_self=False)
        seen = cf > 0
        obs = torch.where(seen, zc / cf.clamp(min=1e-12), torch.zeros_like(zc))
        spread = torch.where(seen, (b["zmax"] - b["zmin"]).clamp(min=0.0),
                             torch.zeros_like(zc))
        # Support: how much observed geometry lands in this neighbourhood at all.
        # Pooled confidence SUM over a window x window patch -- a COUNT, so a thin
        # wall (3 points) is held loosely and a dense one tightly, and the number
        # that decides it is scale-free. avg_pool2d returns the mean, hence the
        # x window^2 back to a sum; count_include_pad=False stops the border from
        # reading its own padding as thin cloud.
        dens = F.avg_pool2d(cf.view(1, 1, self.H, self.W), self.window, stride=1,
                            padding=self.window // 2, count_include_pad=False)
        dens = dens.view(-1) * float(self.window * self.window)
        w = torch.where(seen, dens / (dens + self.support), torch.zeros_like(zc))
        # Tolerance: 10% of the pixel's own depth, or the spread of the cloud
        # inside the pixel (two surfaces projected onto one splat pixel genuinely
        # disagree), floored at the cloud's neighbour spacing. All scene units.
        sigma = self.tau * obs + torch.maximum(spread, torch.full_like(zc, self.sigma_floor))
        return obs.view(1, self.H, self.W), w.view(1, self.H, self.W), sigma.view(1, self.H, self.W)

    def loss(self, view: int, depth_pred: torch.Tensor, alpha: torch.Tensor):
        """Bounded disagreement between rendered and observed depth, in [0, 1].

        r(d) = |d| / (|d| + sigma) is FSGS' robust form: a pixel that agrees
        contributes 0, one that disagrees hugely contributes 1 and no gradient,
        so a stale or mis-triangulated sparse point cannot drag the surface off.
        Averaged as a confidence-weighted mean, which keeps the term in [0, 1]
        whatever the coverage is -- so --depth-weight means the same thing in a
        cloud-rich scene and a cloud-poor one.
        """
        got = self.observed(view)
        if got is None:
            return None
        obs, w, sigma = got
        d = (depth_pred - obs).abs()
        r = d / (d + sigma.clamp(min=1e-9))
        m = (w > 0) & (alpha > self.alpha_min) & torch.isfinite(d)
        rw, wm = r * w, (w * m.to(w.dtype))
        tot = wm.sum().clamp(min=1e-12)
        term = (rw * m.to(w.dtype)).sum() / tot
        err = (d * m.to(w.dtype) * w).sum() / tot  # mean |depth error|, scene units
        # mean_weight is the share of the frame the term actually constrains:
        # 0.0 with no coverage is "nothing was measured", never "it was perfect".
        return (term, err, m.to(torch.float32).mean(), w.mean())


# ---------------- held-out evaluation ----------------
# The A/B instrument for the depth term. Training views are the ones the depth
# term was fitted on, so a PSNR measured against them answers the wrong question
# twice over. --holdout N withholds every Nth frame from BOTH arms: its pixels
# never supervise colour or depth, and these are the numbers that say whether the
# geometry generalised.
#
# The geometric number that does not use the sparse cloud at all is
# cross-view depth consistency (cv_depth_*): render depth from one held-out
# camera, walk those pixels onto a second held-out camera, and ask whether that
# camera agrees. A wall that collapsed into mush cannot be in two places at once,
# so it fails this; a surface that is wrong but the same wrong way from both
# sides passes. It is a consistency measure, not an accuracy one, and is labelled
# as such in the report.


def render_views(params, views: list, W: int, H: int, rendering, device: str,
                 near: float, far: float, antialias: bool, with_depth: bool = True):
    """Render a list of view dicts -> (rgb [N,3,H,W] in 0..1, depth [N,H,W], alpha [N,H,W]).

    depth is None when with_depth is False: the extra channel is the only thing
    the depth term costs at render time, so a run without it never pays.
    """
    quats = F.normalize(params["quats"], dim=-1)
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    scales, opas = torch.exp(params["scales"]), torch.sigmoid(params["opacities"])
    mode = "RGB+ED" if with_depth else "RGB"
    rgbs, deps, alps = [], [], []
    with torch.no_grad():
        for d in views:
            vm = torch.tensor(d["viewmat"], dtype=torch.float32, device=device)[None]
            K = torch.tensor(d["K"], dtype=torch.float32, device=device)[None]
            out, alpha, _ = rendering(
                params["means"], quats, scales, opas, colors, vm, K, W, H,
                near_plane=near, far_plane=far, render_mode=mode, sh_degree=SH_DEG,
                packed=True, rasterize_mode="antialiased" if antialias else "classic")
            rgbs.append(out[..., :3].permute(0, 3, 1, 2).clamp(0, 1)[0])
            alps.append(alpha[0, :, :, 0])
            if with_depth:
                deps.append(out[..., 3][0])
    return (torch.stack(rgbs), torch.stack(deps) if with_depth else None,
            torch.stack(alps))


def depth_transfer(view_a: dict, da: torch.Tensor, aa: torch.Tensor,
                   view_b: dict, db: torch.Tensor, ab: torch.Tensor,
                   W: int, H: int, stride: int, near: float, far: float):
    """Absolute z disagreement, in scene units, between two rendered depth maps.

    Pixels of view A are un-projected to world with A's intrinsics and pose, re-
    projected into B, and compared with B's own rendered depth there. Nothing in
    this uses the sparse cloud, so it is not the depth term's own objective being
    re-measured: it asks whether the surface is in one place or in several.
    """
    dev = da.device
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    lin = torch.tensor(vs.ravel() * W + us.ravel(), dtype=torch.long, device=dev)
    pix = torch.tensor(np.stack([us.ravel(), vs.ravel()], 1).astype(np.float32),
                       dtype=torch.float32, device=dev)
    z, a = da.view(-1)[lin], aa.view(-1)[lin]
    ok = (a > 0.5) & torch.isfinite(z) & (z > near) & (z < far)
    if int(ok.sum()) < 50:
        return None
    pix, z = pix[ok], z[ok]
    Ka = view_a["K"]
    Xc = torch.stack([(pix[:, 0] - Ka[0, 2]) / Ka[0, 0] * z,
                      (pix[:, 1] - Ka[1, 2]) / Ka[1, 1] * z, z], 1)
    # viewmat is world->cam (R | t), so cam->world is Xw = R^T (Xc - t).
    Ra, ta = torch.tensor(view_a["viewmat"], dtype=torch.float32, device=dev)[:3, :3], \
        torch.tensor(view_a["viewmat"], dtype=torch.float32, device=dev)[:3, 3]
    Rb, tb = torch.tensor(view_b["viewmat"], dtype=torch.float32, device=dev)[:3, :3], \
        torch.tensor(view_b["viewmat"], dtype=torch.float32, device=dev)[:3, 3]
    Kb = view_b["K"]
    Xw = (Xc - ta) @ Ra                        # cam-A -> world  (= Ra.T @ (Xc-ta))
    Yc = Xw @ Rb.T + tb                        # world -> cam-B
    zb = Yc[:, 2]
    okb = torch.isfinite(zb) & (zb > near) & (zb < far)
    ub = torch.where(okb, Yc[:, 0] / zb.clamp(min=1e-9) * Kb[0, 0] + Kb[0, 2],
                     torch.full_like(zb, -1.0))
    vb = torch.where(okb, Yc[:, 1] / zb.clamp(min=1e-9) * Kb[1, 1] + Kb[1, 2],
                     torch.full_like(zb, -1.0))
    okb = (ub >= 0) & (ub < W - 1e-6) & (vb >= 0) & (vb < H - 1e-6)
    if int(okb.sum()) < 50:
        return None
    qi = (vb[okb].round().clamp(0, H - 1) * W + ub[okb].round().clamp(0, W - 1)).long()
    zb = zb[okb]
    tgt, at = db.view(-1)[qi], ab.view(-1)[qi]
    good = (at > 0.5) & torch.isfinite(tgt) & ((tgt - zb).abs() < 0.5 * far)
    if int(good.sum()) < 50:
        return None
    e = (tgt[good] - zb[good]).abs()
    return dict(n=int(good.sum()), mean=float(e.mean()), p50=float(e.median()),
                p90=float(e.quantile(0.9)),
                rel=float((e / zb[good].clamp(min=1e-6)).median()))


def cloud_support(means: torch.Tensor, xyz: torch.Tensor, reach_units: float,
                  n_gs: int = 4000, chunk: int = 500,
                  gen: torch.Generator | None = None):
    """Share of gaussian centres with observed geometry within reach_units.

    reach_units is a length in SCENE units (built from the cloud's own measured
    neighbour spacing, never a metric constant), so this transfers between
    scenes. Floaters in unobserved air are what the term is meant to stop.
    """
    if len(means) == 0 or len(xyz) == 0:
        return None
    g = means[torch.randperm(len(means), device=means.device, generator=gen)[:n_gs]]
    sup, dist = 0, []
    for i in range(0, len(g), chunk):
        block = g[i:i + chunk]
        d = torch.cdist(block, xyz).min(1)
        sup += int((d.values <= reach_units).sum())
        dist.append(d.values)
    return dict(fraction=sup / max(len(g), 1),
                nn_median=float(torch.cat(dist).median()),
                reach_units=float(reach_units))


def dump_pair(dir_path: Path, name: str, gt: torch.Tensor, rgb: torch.Tensor,
              depth, alpha):
    """Evidence on disk: what was seen, what was rendered, where they differ.

    The depth png is stretched between its own 2nd and 98th percentile of the
    pixels the render actually covered, so one stray floater depth cannot wash
    the picture out; it is a picture, and the numbers say the same thing.
    """
    def save(t, suffix):
        a = (t.detach().clamp(0, 1) * 255).byte().cpu().numpy()
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        return rb.save_image(Image.fromarray(a), dir_path / f"{name}{suffix}.png")
    save(gt.permute(1, 2, 0), "_gt")
    save(rgb.permute(1, 2, 0), "_render")
    save((gt - rgb).abs().mean(0), "_diff")
    if depth is not None and alpha is not None:
        seen = depth[alpha > 0.5]
        if seen.numel() > 100:
            lo, hi = torch.quantile(seen, 0.02), torch.quantile(seen, 0.98)
            save(((depth - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1), "_depth")
    return True


def evaluate(params, eval_data, gt, W, H, rendering, device, args, extent,
             out_dir: Path | None = None):
    """Held-out PSNR / SSIM / cross-view depth consistency, plus picture dumps.

    `gt` is the [N, 3, H, W] ground truth of exactly these views in 0..1.
    A missing measurement is reported as `None` in cv_depth, never as a 0.
    """
    rgb, dep, alp = render_views(params, eval_data, W, H, rendering, device,
                                 args.near_plane, 10 * extent, args.antialias)
    psnr, ssim = [], []
    for i in range(len(eval_data)):
        a = gt[i]
        m = F.mse_loss(rgb[i], a)
        psnr.append(float(-10.0 * torch.log10(m.clamp_min(1e-10))))
        ssim.append(float(fast_ssim(rgb[i:i + 1], a[None])))
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            dump_pair(out_dir, f"{i:02d}_{Path(str(eval_data[i]['name'])).stem}",
                      a, rgb[i], dep[i], alp[i])
    near, far = args.near_plane, 10 * extent
    acc = {k: [] for k in ("mean", "p50", "p90", "rel", "n")}
    for i in range(len(eval_data) - 1):
        for (a, b) in ((i, i + 1), (i + 1, i)):
            got = depth_transfer(eval_data[a], dep[a], alp[a], eval_data[b], dep[b],
                                 alp[b], W, H, args.eval_stride, near, far)
            if got is None:
                continue
            for k in acc:
                acc[k].append(got[k])
    n_pairs = len(acc["n"])
    # status/measurable/unmeasurable_reason, the pattern D5 established: an
    # unmeasurable geometric number has to be able to say it was unmeasurable.
    cvd = dict(pairs=n_pairs,
               status="measured" if n_pairs else "not_measurable")
    for k in ("mean", "p50", "p90", "rel"):
        cvd[k] = float(np.mean(acc[k])) if n_pairs else None
    cvd["pixels"] = int(np.sum(acc["n"])) if n_pairs else 0
    if not n_pairs:
        cvd["unmeasurable_reason"] = (
            f"no pair of the {len(eval_data)} held-out views had >=50 mutually "
            f"visible covered pixels to compare at stride {args.eval_stride}")
    return dict(psnr=float(np.mean(psnr)), psnr_min=float(np.min(psnr)),
                ssim=float(np.mean(ssim)), n_views=len(eval_data), cv_depth=cvd)


# A gaussian centre counts as "on observed geometry" if a sparse point is within
# this many cloud-neighbour spacings. A multiple of a measured length, so it
# transfers between scenes; 3 spacings on the rocks cloud is ~0.29 scene units.
NN_SUPPORT_SPACINGS = 3.0
# How often the depth disagreement is sampled when it is not already in the loss
# (see --depth-measure). One extra depth-channel render every this steps.
DEPTH_PROBE_EVERY = 50


def fmt_cvd(cvd: dict) -> str:
    """One line for a cross-view depth result, with the units and the no-measurement
    case spelled out. An unmeasured geometric number prints as 'not measurable',
    never as 0."""
    if cvd.get("status") != "measured":
        return f"not measurable ({cvd.get('unmeasurable_reason', 'no pairs')})"
    return (f"p50 {cvd['p50']:.4g}u p90 {cvd['p90']:.4g}u "
            f"(mean {cvd['mean']:.4g}u, {cvd['rel'] * 100:.1f}% of depth, "
            f"{cvd['pixels']} px over {cvd['pairs']} view pairs)")


def ground_truth(views: list, device: str, W: int, H: int) -> torch.Tensor:
    """[N, 3, H, W] float in 0..1 for these view dicts, from cache or from disk.

    The cached copy is the one fit_to_vram re-sized, so it is the same picture the
    renderer is being asked to reproduce; a streamed set falls back to reading the
    frame and resizing it exactly the way the training loop does.
    """
    out = []
    for d in views:
        a = d.get("img")
        if a is None:
            bgr = cv2.imread(str(d["path"]))
            if bgr is None:
                raise FileNotFoundError(d["path"])
            a = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (a.shape[1], a.shape[0]) != (W, H):
            a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
        out.append(torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
                   .to(device, dtype=torch.float32) * (1.0 / 255.0))
    return torch.stack(out)


# ---------------- initialization ----------------
# The room-anchor seeds used to sit at "1.5 / 2.5 / 3.5 metres" in front of every
# camera. Scene units are not metres: work/rocks/frame.json carries
# scale_m_per_unit 4.734 from scale_source "flight speed x clip duration", so those
# three distances seeded 7.1 / 11.8 / 16.6 m out -- past the far wall of any room,
# and inside no rock orbit either. The distances below are therefore quantiles of
# the depth THIS scene's own cloud is at, in front of THIS camera, in scene units.
#
# ANCHOR_DEPTH_QUANTILES are probabilities (dimensionless), ANCHOR_JITTER_FRACTION
# is a fraction of the seed's own distance (dimensionless), and
# MIN_POINTS_PER_CAMERA is a count. None of them is a length.
ANCHOR_DEPTH_QUANTILES = (0.25, 0.50, 0.75)
ANCHOR_JITTER_FRACTION = 0.05     # of each seed's own measured distance
MIN_POINTS_PER_CAMERA = 20        # a camera with fewer points has no distribution


def anchor_seed_distances(data: list, xyz, quantiles=ANCHOR_DEPTH_QUANTILES,
                          n_sample: int = 20_000, rng=None):
    """Per-camera seeding distances, in SCENE UNITS, measured from this cloud.

    For each camera the cloud is projected with that camera's own (already
    rescaled) K and viewmat, and the requested quantiles of the z-depth of the
    points that actually land inside the frame are used: the distribution of what
    that camera can see is where its walls are. Returns (distances
    [n_cameras, n_quantiles], evidence).

    Two fallbacks, both still measured, both named in the evidence so the log can
    say which one ran:
      * a camera that sees fewer than MIN_POINTS_PER_CAMERA points takes the
        pooled quantile over the cameras that do;
      * if NO camera clears that bar the whole cloud is unsuitable, so the
        distances come from the camera hull (the box the cameras themselves
        occupy), which is a measured length and not a metre.
    A scene with neither a cloud nor a camera spread gets None and no anchors: an
    absent measurement is never turned into a distance of 0.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    pts = np.asarray(xyz, dtype=np.float64)
    if len(pts) > n_sample:
        pts = pts[rng.permutation(len(pts))[:n_sample]]
    cam_centers, cam_fwds, own, n_ok = [], [], [], 0
    for d in data:
        vm = np.asarray(d["viewmat"], dtype=np.float64)
        R, t = vm[:3, :3], vm[:3, 3]
        cam_centers.append(-R.T @ t)
        cam_fwds.append(R[2, :])
        if len(pts):
            Xc = pts @ R.T + t
            z = Xc[:, 2]
            K = np.asarray(d["K"], dtype=np.float64)
            W, H = float(d["width"]), float(d["height"])
            zi = np.where(np.abs(z) < 1e-9, np.nan, z)
            u = K[0, 0] * Xc[:, 0] / zi + K[0, 2]
            v = K[1, 1] * Xc[:, 1] / zi + K[1, 2]
            seen = ((z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H))
            zs = z[seen]
            if len(zs) >= MIN_POINTS_PER_CAMERA:
                own.append(np.quantile(zs, quantiles))
                n_ok += 1
                continue
        own.append(None)
    cam_centers = np.asarray(cam_centers, dtype=np.float32)
    cam_fwds = np.asarray(cam_fwds, dtype=np.float32)
    ev = {"n_cameras": len(data), "cloud_points_sampled": int(len(pts)),
          "cameras_with_own_distribution": n_ok,
          "quantiles": list(quantiles), "units": "scene units (COLMAP), NOT metres"}
    good = [o for o in own if o is not None]
    if good:
        pooled = np.median(np.stack(good), axis=0).astype(np.float32)
        ev["source"] = ("cloud depth inside the frustum, per camera" if n_ok == len(data)
                        else f"cloud depth inside the frustum for {n_ok}/{len(data)} "
                             f"cameras; the rest took the pooled quantile")
        ev["pooled_scene_units"] = [round(float(x), 4) for x in pooled]
        D = np.stack([o if o is not None else pooled for o in own]).astype(np.float32)
    else:
        hull = cam_centers.max(0) - cam_centers.min(0)
        span = float(np.linalg.norm(hull))
        if not np.isfinite(span) or span <= 0:
            return None, dict(ev, source="none: no cloud in any frustum and no "
                                        "camera spread to measure", span_units=0.0)
        ev["source"] = ("camera hull: no camera saw enough cloud for its own depth "
                        "distribution, so the seeding distance is a fraction of the "
                        f"{span:.4g}-unit box the cameras occupy")
        ev["camera_hull_diag_units"] = round(span, 4)
        D = np.tile((np.asarray(quantiles) * span).astype(np.float32), (len(data), 1))
    ev["distance_scene_units_p25_50_75"] = [round(float(x), 4)
                                            for x in np.median(D, axis=0)]
    return (cam_centers, cam_fwds, D), ev


def init_from_points(xyz: np.ndarray, rgb: np.ndarray, data: list, device: str, n_init: int = 150_000, room_anchors: bool = True, anchor_note: str = ""):
    pts = torch.tensor(xyz, dtype=torch.float32)
    if len(pts) > n_init:
        sel = torch.randperm(len(pts))[:n_init]
        pts = pts[sel]
        rgb = rgb[sel.cpu().numpy()]
    
    # Inject spatial anchor seeds for indoor room walls and corners, but only on a
    # scene the room detector measured as a room, and only at distances this
    # scene's own cloud says the walls are at.
    if room_anchors and data and len(data) > 5:
        seeds, aev = anchor_seed_distances(data, xyz)
        if seeds is None:
            print("[train] room anchors: SKIPPED, no measurable distance to place them "
                  f"by ({aev['source']}) - not a distance of 0, no distance at all")
        else:
            cam_centers, cam_fwds, D = seeds
            jitter = D * ANCHOR_JITTER_FRACTION
            wall_pts, wall_rgbs = [], []
            mean_c = rgb.mean(axis=0) if len(rgb) else np.array([160.0, 160.0, 160.0])
            for k in range(D.shape[1]):
                w = cam_centers + cam_fwds * D[:, k:k + 1]
                w = w + (np.random.randn(*w.shape).astype(np.float32) * jitter[:, k:k + 1])
                wall_pts.append(w)
                wall_rgbs.append(np.tile(mean_c, (len(w), 1)))
            wall_pts = np.vstack(wall_pts)
            wall_rgbs = np.vstack(wall_rgbs)
            pts = torch.cat([pts, torch.tensor(wall_pts, dtype=torch.float32)], dim=0)
            rgb = np.vstack([rgb, wall_rgbs])
            print(f"[train] seeded {len(wall_pts)} spatial anchor points on room walls "
                  f"at each camera's own measured cloud depth (median "
                  f"{'/'.join(f'{v:.4g}' for v in np.median(D, axis=0))} scene units "
                  f"= p25/p50/p75, jitter {ANCHOR_JITTER_FRACTION} x that distance). "
                  f"Evidence: {anchor_note}")

    pts = pts.to(device)
    rgb_t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).clamp(1e-4, 1 - 1e-4)

    sub = pts[torch.randperm(len(pts), device=device)[:min(20000, len(pts))]]
    d = torch.cdist(sub, sub)
    d.fill_diagonal_(float("inf"))
    nn3 = d.topk(min(3, len(sub) - 1), largest=False).values.mean(1)
    med = float(nn3.median().clamp(min=1e-6))
    scales0 = torch.full((len(pts),), math.log(med), dtype=torch.float32, device=device)

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts),
        "quats": torch.nn.Parameter(torch.cat([torch.ones(len(pts), 1, device=device),
                                               torch.zeros(len(pts), 3, device=device)], dim=1)),
        "scales": torch.nn.Parameter(scales0[:, None].repeat(1, 3)),
        "opacities": torch.nn.Parameter(torch.full((len(pts),), 0.1, device=device)),
        "sh0": torch.nn.Parameter(_rgb_to_sh0(rgb_t)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(len(pts), (SH_DEG + 1) ** 2 - 1, 3, device=device)),
    })
    extent = float((pts.max(0).values - pts.min(0).values).max())
    return params, extent


def _rgb_to_sh0(rgb):
    return (rgb - 0.5) / 0.28209479177387814


def export_ply(means, quats, scales, opacities, sh0, shN, path: Path, note: str = ""):
    """Write the cloud. `note` goes on the log line, because a splat whose step is
    unknown is a splat whose opacity schedule state is unknown -- after a reset
    boundary the same count of gaussians renders 24 dB worse."""
    N = len(means)
    f_dc = sh0[:, 0, :]
    n_rest = (SH_DEG + 1) ** 2 - 1
    f_rest = torch.zeros(N, n_rest * 3, dtype=torch.float32)
    if shN is not None and shN.shape[1] > 0:
        k = min(shN.shape[1], n_rest)
        for i in range(k):
            f_rest[:, i * 3:(i + 1) * 3] = shN[:, i, :]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")] + \
            [(f"f_rest_{i}", "f4") for i in range(n_rest * 3)] + \
            [("opacity", "f4"), ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
             ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    arr = np.zeros(N, dtype=dtype)
    m = means.detach().cpu().numpy()
    arr["x"], arr["y"], arr["z"] = m[:, 0], m[:, 1], m[:, 2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc.detach().cpu().numpy().T
    fr = f_rest.detach().numpy()
    for i in range(n_rest * 3):
        arr[f"f_rest_{i}"] = fr[:, i]
    arr["opacity"] = opacities.detach().cpu().numpy()
    sc = scales.detach().cpu().numpy()
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = sc[:, 0], sc[:, 1], sc[:, 2]
    qt = quats.detach().cpu().numpy()
    qt = qt / np.linalg.norm(qt, axis=1, keepdims=True)
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = qt[:, 0], qt[:, 1], qt[:, 2], qt[:, 3]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))
    print(f"[train] wrote {path} ({N} gaussians){' at ' + note if note else ''}")


# ---------------- main trainer ----------------
def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--cap", type=int, default=650_000, help="hard max gaussian count")
    ap.add_argument("--init-pts", type=int, default=150_000)
    ap.add_argument("--ssim-weight", type=float, default=0.2)
    ap.add_argument("--grow-grad", dest="grow_grad", type=float, default=0.0006,
                    help="2D positional gradient a gaussian must exceed to be cloned or "
                         "split. This, not --cap, is what decides the splat count: the "
                         "auditorium run plateaued at 247k with an 850k cap and 18k steps "
                         "left. Lower it for detail, at the cost of VRAM and sort time.")
    ap.add_argument("--refine-stop", type=int, default=9000)
    ap.add_argument("--save-every", type=int, default=3000)
    ap.add_argument("--antialias", action=argparse.BooleanOptionalAction, default=True,
                    help="gsplat antialiased rasterization (big sharpness win when "
                         "training resolution differs from capture resolution)")
    ap.add_argument("--opa-reg", type=float, default=0.002,
                    help="L1 opacity regularization weight (kills semi-transparent floaters)")
    ap.add_argument("--near-plane", dest="near_plane", type=float, default=0.01,
                    help="rasterizer near plane in SCENE UNITS (the solve's own units, "
                         "which are not metres until frame.json anchors them). Shared with "
                         "the depth term's visibility gate, so they cannot drift apart.")
    # ---- challenge D2: sparse-depth consistency ----------------------------------
    # All six numbers below are dimensionless or pixel/count based on purpose. The
    # only length the term touches is the cloud's own measured neighbour spacing,
    # derived per scene; nothing here is a metric constant, because the scene's
    # scale is arbitrary until solve_frame anchors it.
    ap.add_argument("--depth-weight", dest="depth_weight", type=float, default=0.05,
                    help="weight of the rendered-vs-observed sparse-depth term (dimensionless). "
                         "The term itself is a confidence-weighted mean of |d|/(|d|+sigma), so "
                         "it lives in [0,1] whatever the coverage or the scene scale is and this "
                         "weight means the same thing in every scene. 0 is the A/B control and "
                         "renders no depth channel at all. FSGS and DNGaussian publish 0.05 for "
                         "this role, so that is the default; measured on work/rocks (640x360, "
                         "12k steps, 1-in-6 held out, paired at seed 0 and seed 1) that is raw "
                         "depth disagreement 0.434 -> 0.137 and held-out cross-view depth p50 "
                         "~0.19 -> ~0.039 scene units (4.8-5.3x) for ~+12-14%% step time and no "
                         "change in peak ALLOCATED VRAM (+0.19 GiB reserved at 1080p-class, "
                         "where the projected cloud alone costs 158 MB resident). The held-out "
                         "COLOUR change is not resolved by "
                         "that measurement: -0.24/-0.17 dB against a 0.69 dB spread between "
                         "three identical control runs. It is a bias TOWARD OBSERVED GEOMETRY, "
                         "so on footage whose colour is already well fitted it buys geometry, "
                         "not sharpness.")
    ap.add_argument("--depth-warmup", dest="depth_warmup", type=float, default=0.15,
                    help="FRACTION OF --steps over which the depth term ramps 0 -> full weight, "
                         "starting where the densifier starts (refine_start_iter). Depth from "
                         "step 0 fights the photometric loss while the cloud is still being "
                         "placed, which is the standard way to make these diverge. 0.15 x 12000 "
                         "= 500 -> 2300.")
    ap.add_argument("--depth-tau", dest="depth_tau", type=float, default=0.10,
                    help="disagreement scale, as a FRACTION OF THE PIXEL'S OWN DEPTH: a pixel "
                         "ignores render-vs-observe disagreement below tau x depth (plus the "
                         "depth spread of the cloud inside that pixel, floored at one cloud "
                         "neighbour spacing). Where the two views genuinely disagree, this is "
                         "what stops a stale point from being obeyed.")
    ap.add_argument("--depth-support", dest="depth_support", type=float, default=3.0,
                    help="points of pooled confidence at which a pixel reaches HALF strength "
                         "(a COUNT, not a length). A textureless wall held by 3 sparse points "
                         "is constrained far more loosely than one held by 3000, which is the "
                         "whole reason this term does not flatten a thin cloud into a lie.")
    ap.add_argument("--depth-window", dest="depth_window", type=int, default=9,
                    help="PIXEL patch the support count above is pooled over. Independent of "
                         "scene scale, but it is resolution dependent: it is pooled on the "
                         "training grid, so it means ~9x9 of whatever --cap/pixel budget won.")
    ap.add_argument("--depth-err-px", dest="depth_err_px", type=float, default=1.5,
                    help="PIXELS. COLMAP reports a mean reprojection ERROR per sparse point, "
                         "and a point is worth 1/(1+err/this) of a constraint: the solve's own "
                         "uncertainty, in a unit it already speaks. Median here is printed at "
                         "startup so the default can be checked against the scene.")
    ap.add_argument("--depth-measure", dest="depth_measure",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="sample the rendered-vs-observed depth error on the log even when "
                         "--depth-weight is 0, so the control arm of an A/B reports the same "
                         "number as the armed arm. Costs one extra depth-channel render every "
                         f"{DEPTH_PROBE_EVERY} steps (about {100.0 / DEPTH_PROBE_EVERY:.0f}%% of "
                         "the steps) and keeps the projected cloud resident on the GPU. "
                         "With --no-depth-measure AND --depth-weight 0 no step renders the "
                         "fourth channel and nothing is added to the loss, which is the "
                         "control arm; the cloud's neighbour spacing is still measured on "
                         "both arms, because the report quotes it (see nn_spacing above).")
    ap.add_argument("--holdout", type=int, default=0,
                    help="withhold every Nth training frame from colour AND depth supervision "
                         "and report PSNR/SSIM/cross-view depth consistency on it. Off (0) for "
                         "production runs; this is the A/B instrument, and both arms of a "
                         "comparison must be given the same value.")
    ap.add_argument("--eval-stride", dest="eval_stride", type=int, default=4,
                    help="PIXELS: subsampling of the held-out depth consistency check "
                         "(1 = every pixel; 4 is 16x cheaper and the surface is smooth).")
    ap.add_argument("--eval-images", action=argparse.BooleanOptionalAction, default=True,
                    help="write held-out gt/render/diff/depth PNGs next to the checkpoint")
    ap.add_argument("--random-bkgd", action="store_true", default=True,
                    help="use random background color to kill floaters in unobserved air/ceiling")
    ap.add_argument("--room-prior", dest="room_prior", choices=("auto", "on", "off"),
                    default="auto",
                    help="whether to seed the room-wall anchor points at all. auto asks "
                         "the finished detector (viewer_assets/rooms.json from "
                         "detect_rooms.py) and declines, with its reason, on a scene it "
                         "did not measure as a room; on applies it anyway (overruling the "
                         "detector, which is printed); off never applies it. All three "
                         "place the seeds at distances MEASURED from this scene's own "
                         "cloud, never at a metre literal.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    # Hardware acceleration flags for Ampere (RTX 30-series) Tensor Cores
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    from gsplat import rasterization as rendering, DefaultStrategy
    from gsplat.strategy.ops import remove as _remove_gs

    data, (p_xyz, p_rgb, p_err, p_nviews) = prepare_dataset(args.work)
    if len(data) < 3:
        sys.exit(f"[train] only {len(data)} frames are registered in this model - "
                 f"training needs at least 3. See {args.work / 'logs'} for the "
                 f"colmap registration count.")

    W, H = data[0]["width"], data[0]["height"]

    # --holdout splits the set BEFORE anything is cached or seeded, so a held-out
    # frame is unseen by the colour loss, the depth maps and the room anchors
    # alike. Nothing about the split depends on the scene.
    eval_data = []
    if args.holdout and args.holdout > 1:
        cand = list(range(0, len(data), args.holdout))
        keep = max(1, min(len(cand), len(data) // 4))   # never starve the trainer
        ev_idx = set(cand[:keep])
        eval_data = [d for i, d in enumerate(data) if i in ev_idx]
        data = [d for i, d in enumerate(data) if i not in ev_idx]
        print(f"[train] holdout 1-in-{args.holdout}: {len(data)} views train, "
              f"{len(eval_data)} views are scored only")

    # Pre-cache camera viewmats and Ks in GPU VRAM
    gpu_viewmats = torch.stack([torch.tensor(d["viewmat"], dtype=torch.float32, device=device) for d in data])
    gpu_Ks = torch.stack([torch.tensor(d["K"], dtype=torch.float32, device=device) for d in data])

    # One budget for the whole card, from measured free VRAM: pixels, image
    # cache and gaussian cloud are three claims on the same 6 GiB, and the two
    # heuristics this replaces each looked at only one of them. The cap clamp in
    # particular could not express what killed a run -- 465 frames at 1280x2772
    # is 3.5 MP of render intermediates per step, which no change to the
    # gaussian count touches, so the card filled and the process died with a
    # bare exit -1 and nothing printed.
    from robust import train_budget
    _free_b, _total_b = torch.cuda.mem_get_info()
    vram_gb = _free_b / 1024**3
    budget = train_budget(vram_gb=vram_gb, max_pixels=W * H, n_frames=len(data))
    img_bytes = sum(d["img"].nbytes for d in data if d.get("img") is not None)
    # A frame count alone says nothing about VRAM pressure once the images are
    # not resident, so streaming is decided against the budget, not a constant.
    stream_images = budget["stream_images"] or any(d.get("img") is None for d in data)
    if args.cap > budget["cap"]:
        print(f"[train] cap {args.cap} -> {budget['cap']}: {W}x{H} frames "
              f"x {len(data)} on {vram_gb:.1f} GiB free of {_total_b / 1024**3:.1f} GiB")
        args.cap = budget["cap"]
    if stream_images:
        print(f"[train] image set {img_bytes / 1024**2:.0f} MB will cost the "
              f"gaussians their room on a {vram_gb:.1f} GiB budget "
              f"— streaming frames from disk instead")
        for d in data:
            d["img"] = None
        gpu_imgs_u8 = None
    else:
        print(f"[train] pre-caching {img_bytes / 1024**2:.1f} MB images directly in GPU VRAM")
        gpu_imgs_u8 = torch.stack([torch.from_numpy(np.array(d.pop("img"))).to(device).permute(2, 0, 1) for d in data])

    # ---- is this a room? the detector answers, not the caller ----------------
    # The room-anchor prior below used to run on every scene, so an outdoor rock
    # orbit printed "seeded 180 spatial anchor points on room walls" and got seeds
    # placed by a room guess. detect_rooms.py has now measured whether there is a
    # coherent floor and walls to have a prior about; when it says no, this says so
    # and why, on the log, instead of declining silently. --room-prior on/off lets a
    # caller overrule the measurement, and the log and the report both say that is
    # what happened: the default, auto, is the detector's answer.
    want_anchors, anchor_note, anchor_ev = room_prior_decision(args.work)
    anchor_ev = dict(anchor_ev)
    if args.room_prior == "on" and not want_anchors:
        want_anchors = True
        anchor_ev["overruled"] = "--room-prior on"
        anchor_note = f"OVERRULED to apply by --room-prior on; detector said: {anchor_note}"
    elif args.room_prior == "off" and want_anchors:
        want_anchors = False
        anchor_ev["overruled"] = "--room-prior off"
        anchor_note = f"OVERRULED to skip by --room-prior off; detector said: {anchor_note}"
    elif args.room_prior in ("on", "off"):
        anchor_ev["overruled"] = f"--room-prior {args.room_prior} (agrees)"
    print(f"[train] room prior: {'APPLIED' if want_anchors else 'DECLINED'} - "
          f"{anchor_note}")

    params, extent = init_from_points(p_xyz, p_rgb, data, device, args.init_pts,
                                      room_anchors=want_anchors,
                                      anchor_note=anchor_note)
    print(f"[train] init {len(params['means'])} gaussians, scene extent {extent:.2f} "
          f"COLMAP units (arbitrary until solve_frame anchors them)")
    if not len(params["means"]):
        # An empty cloud is not a slow start: the rasterizer's CUDA kernel
        # fail-fasts on zero gaussians, and the process dies with a bare
        # 0xC0000409 that looks like the GPU or the video format failing.
        sys.exit("[train] the reconstruction has no 3D points to seed from, so there "
                 "is nothing to train. The colmap solve is the problem, not this step.")
    # ---- challenge D2: is there observed geometry to be consistent with? ----
    # The whole premise is that the solve already paid for the answer: the sparse
    # cloud is measured depth and gsplat already renders depth in the same pass as
    # the image (see "sparse-depth supervision" above), so this costs no model, no
    # download and one extra output channel.
    #
    # When there is not enough measured geometry the term is a NO-OP and SAYS SO on
    # the log, because a regulariser that quietly contributes nothing reads exactly
    # like a regulariser that is working. The three bars below are all counts or
    # fractions of the frame -- no length, so none of them can be wrong because the
    # scene scale is arbitrary.
    #
    # The spacing is measured on both arms, weight 0 included, because the report
    # quotes it and an A/B must not differ in what it computes for free. A private
    # generator: sampling the subset off the global stream would shift the random
    # background draws and put a seed difference into a weight-only comparison.
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 1)
    pts_gpu = torch.as_tensor(p_xyz, dtype=torch.float32, device=device)
    spacing = nn_spacing(pts_gpu, gen=gen)
    depth_term = None
    depth_note = ("no-op (--depth-weight 0 --no-depth-measure): photometric loss only, "
                  "the depth channel is never rendered")
    if args.depth_weight > 0 or args.depth_measure:
        targets, srep = sparse_depth_targets(
            data, p_xyz, p_err, p_nviews, args.near_plane, 10 * extent,
            args.depth_err_px, POINT_VIEWS_HALF)
        too_thin = []
        if srep["distinct_sizes"] > 1:
            too_thin.append(f"the {srep['n_views']} views have "
                            f"{srep['distinct_sizes']} different pixel sizes, so no "
                            f"single observed-depth grid fits them")
        if srep["views_with_points"] < max(3, int(0.2 * len(data))):
            too_thin.append(f"only {srep['views_with_points']}/{len(data)} views have "
                            f"any observed point in frustum")
        if srep["points_per_view_median"] < MIN_POINTS_PER_VIEW:
            too_thin.append(f"median {srep['points_per_view_median']} points per view "
                            f"< {MIN_POINTS_PER_VIEW}")
        if srep["coverage_median"] < MIN_FRAME_COVERAGE:
            too_thin.append(f"the cloud covers {srep['coverage_median'] * 100:.4g}% of a "
                            f"frame < {MIN_FRAME_COVERAGE * 100:.0f}%")
        depth_note = (
            f"{srep['n_points']} sparse points over {srep['n_views']} views: median "
            f"{srep['points_per_view_median']} points/view (min "
            f"{srep['points_per_view_min']}), covering "
            f"{srep['coverage_median'] * 100:.4g}% of the frame; median point "
            f"confidence {srep['conf_median']:.2f}; cloud 3-nearest spacing "
            f"{spacing:.4g} scene units")
        if too_thin:
            args.depth_weight = 0.0
            args.depth_measure = False
            depth_note = ("NO-OP, depth term DISABLED: " + "; ".join(too_thin)
                          + ". Not a zero disagreement - nothing was measured. "
                          + f"[{depth_note}]")
        else:
            depth_term = DepthTerm(targets, H, W, device, spacing, args.depth_tau,
                                   args.depth_support, args.depth_window,
                                   NN_SPACING_FLOOR, ALPHA_MIN_FOR_DEPTH)
            depth_note += (f"; built for {depth_term.stats['n_views_ok']}/{len(data)} "
                           f"views (scratch {depth_term.stats['scratch_mb']} MB + "
                           f"{depth_term.stats['pts_mb']} MB of points resident)")
            if args.depth_weight > 0:
                depth_note += (f"; ARMED with tolerance floor {NN_SPACING_FLOOR} x "
                               f"spacing, tau {args.depth_tau} x depth, half strength "
                               f"at {args.depth_support} points per "
                               f"{args.depth_window}x{args.depth_window} px")
            else:
                depth_note += (f"; MEASURING ONLY (--depth-weight 0), sampled every "
                               f"{DEPTH_PROBE_EVERY} steps and never added to the loss")

    # Warmup spans [densifier start, densifier start + warmup x steps]. A smoke
    # run never reaches step 500, so on a short run the start falls back to 10% of
    # the steps - otherwise the term would be inert on exactly the tier that is
    # supposed to exercise every step.
    depth_start_at = max(REFINE_START, 1) if args.steps > 2 * REFINE_START \
        else max(1, int(0.1 * args.steps))
    depth_full_at = depth_start_at + max(1.0, args.depth_warmup * args.steps)

    lrs = {
        "means": 1.6e-4 * extent, "quats": 1e-3, "scales": 5e-3,
        "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20,
    }
    optimizers = {
        name: torch.optim.Adam([{"params": params[name], "lr": lrs[name], "name": name}],
                               eps=1e-15, betas=(0.9, 0.999))
        for name in params.keys()
    }

    strategy = DefaultStrategy(
        verbose=False,
        absgrad=True,
        grow_grad2d=args.grow_grad,
        prune_opa=0.02,
        prune_scale3d=0.10 * extent,
        reset_every=RESET_EVERY,
        refine_start_iter=REFINE_START,
        refine_stop_iter=args.refine_stop,
        refine_every=150
    )
    strategy_state = strategy.initialize_state()

    # ---- the reset, and what an artefact or a metric is allowed to describe ----
    # gsplat clamps EVERY opacity down to `prune_opa * 2` (0.04 here) at the end of
    # every RESET_EVERY-th step below refine_stop, and the run used to be able to
    # END on one. Two things went wrong at once and only one was visible: the final
    # held-out PSNR read ~16 dB instead of ~34 dB, and the exported splat.ply was
    # the demoted cloud that reading describes (measured 2026-09-26, work/rocks at
    # --steps 6000: 16.14 dB held-out, 0.5545 SSIM, every one of 167,609 opacities
    # sitting exactly at the 0.04 clamp; the same run at --steps 5999: 34.16 dB,
    # 0.8929). Every A/B at such a step count measured the reset, silently.
    #
    # The fix keeps gsplat's schedule exactly -- it just does not let the demotion
    # become the answer. The opacities gsplat is about to clamp are captured on the
    # reset step (ResetCapture), and any cloud that is exported or measured on that
    # step is exported with the captured values. If the capture did not happen, the
    # run REFUSES to report rather than print the artefact (artefact_reset_state).
    resets_crossed = [s for s in range(1, args.steps + 1)
                      if reset_lands_on(s, RESET_EVERY, args.refine_stop)]
    capture = None          # live only on the step gsplat resets on

    if len(params["means"]) > args.cap:
        # Seeding can overshoot a cap that was only just settled, and the first
        # refine would then push straight into an out-of-memory. Trim to the most
        # opaque seeds rather than starting over-budget.
        print(f"[train] seeds {len(params['means'])} > cap {args.cap}: trimming "
              f"to the most opaque")
        opa = torch.sigmoid(params["opacities"]).reshape(-1)
        drop = torch.ones_like(opa, dtype=torch.bool)
        drop[opa.topk(args.cap).indices] = False
        _remove_gs(params, optimizers, strategy_state, drop)

    prog = args.work / "train_progress"
    prog.mkdir(exist_ok=True)
    log = open(args.work / "train_log.txt", "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    # Checkpoint the seeds immediately. A card that fills, or a driver that
    # takes the process with a bare 0xC0000409, is only survivable if something
    # exists to rescue; with --save-every 3000 a death in the first minutes left
    # the runner nothing but a failure to report.
    export_ply(params["means"], F.normalize(params["quats"], dim=1),
               params["scales"], params["opacities"],
               params["sh0"], params["shN"], prog / "splat.partial.ply",
               note="step 0 (seeds)")

    say(f"[train] start steps={args.steps} cap={args.cap} {W}x{H} "
        f"ssim_w={args.ssim_weight} absgrad=True grow_grad2d={args.grow_grad} "
        f"refine_stop={args.refine_stop} "
        f"antialias={args.antialias} opa_reg={args.opa_reg}")
    # Say what the reset schedule is and where the reported numbers will be taken,
    # before the first one lands, so a 16 dB reading can never be read as a model.
    say(f"[train] opacity reset every {RESET_EVERY} steps while step < refine_stop "
        f"({args.refine_stop}) (gsplat clamps every opacity to 0.04 at the end of "
        f"those steps), at steps "
        f"{resets_crossed if resets_crossed else 'none'} of this run")
    if artefact_reset_state(args.steps, RESET_EVERY, args.refine_stop)[0]:
        say(f"[train] NOTE: the requested last step {args.steps} IS a reset step, so "
            f"the cloud as it stands there is a demoted one and a metric taken from it "
            f"measures the reset, not the model (work/rocks measured 16.14 dB against "
            f"34.16 dB one step earlier). This run will export and report the "
            f"opacities as they were trained at step {args.steps}, before the reset "
            f"that follows it, and the training itself continues from the reset "
            f"exactly as gsplat schedules it.")
    say(f"[train] depth term: "
        f"{'ARMED' if depth_term is not None and args.depth_weight > 0 else 'OFF'}"
        f"{' (measuring)' if depth_term is not None and args.depth_weight == 0 else ''}, "
        f"weight={args.depth_weight} warmup {depth_start_at} -> {int(depth_full_at)} "
        f"of {args.steps} steps. {depth_note}")
    if eval_data:
        say(f"[train] holdout: {len(eval_data)} of {len(data) + len(eval_data)} frames "
            f"never supervised, scored at the end (PSNR/SSIM/cross-view depth)")

    # Reporting only: the scene's own units are arbitrary until solve_frame anchors
    # them, so the numbers on the log are printed with what the anchor says when it
    # exists and with no claim at all when it does not. Nothing in the loss reads it.
    mpu = rb.read_json(args.work / "frame.json", None)
    mpu = (mpu or {}).get("scale_m_per_unit")
    if isinstance(mpu, (int, float)) and mpu > 0:
        say(f"[train] unit anchor: frame.json scale_m_per_unit={mpu:.4g} - 1.0 scene "
            f"depth unit = {mpu:.3g} m in the metrics below")
    else:
        say("[train] no scale anchor in frame.json: every length below is in arbitrary "
            "COLMAP units, NOT metres")

    # Blur-aware sampling: sharper frames train more often (directly attacks the
    # "splat looks soft" problem caused by motion-blurred frames polluting SH colors)
    sharp_w = np.clip(np.array([d["sharpness"] for d in data], dtype=np.float64), 15.0, None) ** 0.5
    sample_p = sharp_w / sharp_w.sum()
    say(f"[train] blur-aware sampling: effective weight range "
        f"{sample_p.min() * len(data):.2f}x - {sample_p.max() * len(data):.2f}x per frame")

    def means_lr(step):
        return max(1.6e-6 * extent, 1.6e-4 * extent * math.exp(-step * 0.00023))

    n_images = len(data)
    t0 = time.time()

    def depth_weight_at(step):
        """0 until the densifier starts, then a linear ramp to full weight.

        Depth regularisation from step 0 fights the photometric loss -- the
        gaussians have not been placed yet, and a prior applied to an unformed
        cloud is the standard way to make the two diverge. The ramp's end is a
        fraction of --steps so a smoke run and a production run warm up over the
        same share of themselves.
        """
        if depth_term is None or args.depth_weight <= 0:
            return 0.0
        if step <= depth_start_at:
            return 0.0
        return args.depth_weight * min(1.0, (step - depth_start_at)
                                       / max(depth_full_at - depth_start_at, 1.0))

    # Running means, kept on the GPU so reporting costs one device sync per window
    # instead of one per term per step. `acc`/`tot` are the loss budget, sampled
    # every step; `acc_m`/`tot_m` are the depth measurement, sampled every
    # DEPTH_PROBE_EVERY steps whenever the term is not in the loss (so the control
    # arm reports the same number as the armed arm) and every step when it is.
    KEYS = ("l1", "ssim", "opa", "dep", "w")
    MKEYS = ("raw", "err", "cov", "wsup")
    acc = {k: torch.zeros((), device=device) for k in KEYS}
    tot = {k: torch.zeros((), device=device) for k in KEYS}
    acc_m = {k: torch.zeros((), device=device) for k in MKEYS}
    tot_m = {k: torch.zeros((), device=device) for k in MKEYS}
    n_acc = n_tot = n_m = n_m_tot = 0
    # How many steps actually paid for the fourth output channel. Cost of the term
    # is not a guess: this counts it, and the report quotes it next to the it/s.
    n_depth_renders = 0

    for step in range(1, args.steps + 1):
        optimizers["means"].param_groups[0]["lr"] = means_lr(step)

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            params["scales"].clamp_(max=math.log(10.0 * extent), min=-15.0)

        img_idx = int(np.random.choice(n_images, p=sample_p))
        viewmat = gpu_viewmats[img_idx:img_idx + 1]
        K = gpu_Ks[img_idx:img_idx + 1]
        if not stream_images:
            img_gt = gpu_imgs_u8[img_idx:img_idx + 1].to(dtype=torch.float32) * (1.0 / 255.0)
        else:
            # The renderer is asked for W x H, so the target must be W x H. The
            # frame on disk is not necessarily that size: fit_to_vram may have
            # re-sized the set, and a multi-clip scene can carry mixed native
            # resolutions in the first place.
            frame_bgr = cv2.imread(str(data[img_idx]["path"]))
            if frame_bgr is None:
                sys.exit(f"[train] cannot read {data[img_idx]['path']} - the keyframe "
                         f"step listed a file that is not decodable.")
            if (frame_bgr.shape[1], frame_bgr.shape[0]) != (W, H):
                frame_bgr = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_gt = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32) * (1.0 / 255.0)

        if args.random_bkgd and np.random.rand() < 0.5:
            bkgd = torch.rand(3, device=device)
        else:
            bkgd = None

        quats_n = F.normalize(params["quats"], dim=-1)
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)

        # The depth channel is requested only on the steps that will use it: an
        # un-armed, un-measuring run renders three channels exactly as it always
        # did and pays nothing. gsplat alpha-composites the per-gaussian z in the
        # same kernel launch (rendering.py:614-615) and divides it by alpha for
        # "RGB+ED" (rendering.py:760-768), so this is one pass, not two.
        #
        # gsplat 1.5.3 cannot take a `backgrounds` tensor alongside a depth
        # render_mode: rendering.py:617-623 appends a per-camera zero channel to
        # what the packed path requires to be a flat [D] tensor (cuda/_wrapper.py
        # :597-598 asserts shape image_dims + (channels,), and image_dims is empty
        # when packed=True) and the torch.cat raises. So on those steps the
        # background is composited here instead, with the kernel's own formula
        # (render + background * (1 - alpha), cuda/_torch_impl.py:712-714). No CPU
        # test can pin that equivalence - the lint lane has no torch, and
        # test_camera_intrinsics.py deliberately execs only the numpy half of this
        # file - so it is pinned by the two things that do not need a GPU: the
        # formula above is the kernel's, and the --no-depth-measure arm of the A/B
        # renders three channels with the kernel's own composite exactly as before.
        w_depth = depth_weight_at(step)
        want_depth = w_depth > 0 or (depth_term is not None and args.depth_measure
                                     and step % DEPTH_PROBE_EVERY == 0)
        n_depth_renders += int(want_depth)
        render_mode = "RGB+ED" if want_depth else "RGB"

        sh_deg_cur = min(step // 1000, SH_DEG)
        out = rendering(
            params["means"], quats_n, torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), colors, viewmat, K,
            W, H, near_plane=args.near_plane, far_plane=10 * extent,
            render_mode=render_mode, sh_degree=sh_deg_cur,
            packed=True, absgrad=True,
            backgrounds=None if want_depth else bkgd,
            rasterize_mode="antialiased" if args.antialias else "classic",
        )
        rgb_pred, alpha, info = out
        rendered_depth = None
        if render_mode == "RGB+ED":
            rendered_depth = rgb_pred[..., 3:4]
            rgb_pred = rgb_pred[..., :3]
            if bkgd is not None:
                rgb_pred = rgb_pred + bkgd.view(1, 1, 1, -1) * (1.0 - alpha)
        rgb_pred = rgb_pred.permute(0, 3, 1, 2)  # [1, 3, H, W]
        target_comp = img_gt

        l1 = F.l1_loss(rgb_pred, target_comp)
        ssim_loss = 1.0 - fast_ssim(rgb_pred.clamp(0, 1), target_comp) if args.ssim_weight > 0 else 0.0
        loss = (1 - args.ssim_weight) * l1 + args.ssim_weight * ssim_loss
        terms = dict(l1=(1 - args.ssim_weight) * l1.detach(),
                     ssim=(args.ssim_weight * ssim_loss).detach() if args.ssim_weight > 0
                     else torch.zeros((), device=device),
                     opa=torch.zeros((), device=device),
                     dep=torch.zeros((), device=device),
                     raw=torch.zeros((), device=device),
                     err=torch.zeros((), device=device),
                     cov=torch.zeros((), device=device),
                     wsup=torch.zeros((), device=device),
                     w=torch.full((), w_depth, device=device))
        if args.opa_reg > 0:
            # weak global pull on opacity: prevents soft semi-transparent floater
            # crusts while letting real surfaces stay fully opaque
            l_opa = args.opa_reg * torch.sigmoid(params["opacities"]).mean()
            loss = loss + l_opa
            terms["opa"] = l_opa.detach()

        if rendered_depth is not None and depth_term is not None:
            # D2: how far the rendered surface sits from the depth the solve
            # already measured, everywhere it measured it, weighted by how much it
            # measured there.
            #
            # This is also the A/B instrument. With --depth-measure the SAME
            # number is sampled on every DEPTH_PROBE_EVERY-th step of a run whose
            # weight is 0, so the control arm reports the depth error the term is
            # meant to reduce instead of being silent about it -- and neither arm's
            # figure comes from a different ruler.
            with torch.no_grad() if w_depth <= 0 else nullcontext():
                got = depth_term.loss(
                    img_idx, torch.nan_to_num(rendered_depth[0, :, :, 0], nan=0.0,
                                              posinf=0.0, neginf=0.0),
                    alpha[0, :, :, 0])
            if got is not None:
                l_depth, d_err, d_cov, d_w = got
                terms["raw"] = l_depth.detach()
                terms["err"] = d_err.detach()
                terms["cov"] = d_cov.detach()
                terms["wsup"] = d_w.detach()
                for key in ("raw", "err", "cov", "wsup"):
                    acc_m[key] += terms[key]
                    tot_m[key] += terms[key]
                n_m += 1
                n_m_tot += 1
                if w_depth > 0:
                    loss = loss + w_depth * l_depth
                    terms["dep"] = (w_depth * l_depth).detach()
        # Kept on the GPU and read once per report window: calling .item() on
        # every term every step would add device syncs to the hot loop.
        for key in ("l1", "ssim", "opa", "dep", "w"):
            acc[key] += terms[key]
            tot[key] += terms[key]
        n_acc += 1
        n_tot += 1

        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        loss.backward()
        # On the steps gsplat resets on, capture what it is about to clamp away:
        # the refine in the same call has already re-shaped the cloud, so this is
        # the only moment the rows line up (see ResetCapture).
        capture = ResetCapture() if reset_lands_on(step, RESET_EVERY,
                                                   args.refine_stop) else None
        if capture is None:
            strategy.step_post_backward(params, optimizers, strategy_state, step,
                                        info, packed=True)
        else:
            with capture:
                strategy.step_post_backward(params, optimizers, strategy_state,
                                            step, info, packed=True)
        if not len(params["means"]):
            # The densifier prunes on opacity; when it takes the last gaussian
            # the next backward pass fail-fasts inside CUDA with a bare
            # 0xC0000409 and no traceback. Say which step emptied it instead.
            say(f"[train] COLLAPSED at step {step}: every gaussian was pruned "
                f"(loss {loss.item():.4f}). The seeds cannot support a surface - "
                f"this is a solve-quality problem, not a GPU one.")
            sys.exit(1)
        # gsplat 1.5.3's DefaultStrategy has no max_gaussians, so --cap used to be
        # a number that got printed and never enforced: this cloud grew to 1.05 M
        # against a stated 850 k and kept climbing until the card filled.
        n = len(params["means"])
        if n > args.cap:
            opa = torch.sigmoid(params["opacities"]).reshape(-1)
            drop = torch.ones_like(opa, dtype=torch.bool)
            drop[opa.topk(args.cap).indices] = False
            _remove_gs(params, optimizers, strategy_state, drop)
            if capture is not None and capture.opas is not None:
                # Keep the captured cloud row-aligned with the live one, or the
                # restore below would write the wrong gaussians' opacities.
                capture.opas = capture.opas[~drop]
            say(f"[train] --cap {args.cap} reached at step {step}: "
                f"{n} -> {len(params['means'])} on lowest opacity")

        for opt in optimizers.values():
            opt.step()

        if step % 200 == 0:
            with torch.no_grad():
                mse = F.mse_loss(rgb_pred.clamp(0, 1), img_gt)
                psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
            it_s = step / max(time.time() - t0, 1e-4)
            a = {k: float(v) / max(n_acc, 1) for k, v in acc.items()}
            # An in-flight PSNR logged just after a reset is the demoted cloud
            # re-opacifying, not the model regressing; say so on the line, because
            # a dip of this size has already been misread as a defect once.
            recent = [s for s in resets_crossed
                      if 0 < step - s <= RESET_RECOVERY_TAG_STEPS]
            reset_tag = (f" [recovering from the step-{max(recent)} opacity reset: "
                         f"not a quality reading]" if recent else "")
            # What it cost (the weighted contribution to the total) and what it
            # bought (the raw disagreement, the same error in scene depth units,
            # and the share of the frame it actually constrains) both go on the
            # line, so the term cannot vanish into the total. The depth columns
            # read "not measured" when there was nothing to measure, never 0.
            say(f"[train] step {step:6d} loss {loss.item():.4f} "
                f"| rgb {a['l1']:.4f} ssim {a['ssim']:.4f} "
                f"opa {a['opa']:.4f} depth {a['dep']:.4f} "
                + (f"[raw {float(acc_m['raw'])/max(n_m,1):.4f} "
                   f"w {a['w']:.4f} cov {float(acc_m['cov'])/max(n_m,1)*100:.1f}% "
                   f"sup {float(acc_m['wsup'])/max(n_m,1):.3f} "
                   f"err {float(acc_m['err'])/max(n_m,1):.4g}u] "
                   if n_m else "[depth not measured] ")
                + f"psnr {psnr.item():.2f} " + reset_tag + " "
                f"N {len(params['means'])} {time.time()-t0:.0f}s ({it_s:.1f} it/s) "
                f"mem {torch.cuda.max_memory_allocated()/2**30:.2f}GiB "
                # Reserved, not just allocated: this is the number that competes
                # with the driver for the 6 GiB, and on Windows nvidia-smi reports
                # less than the allocator holds, so "does it OOM" cannot be read
                # off either one alone.
                f"res {torch.cuda.max_memory_reserved()/2**30:.2f}GiB")
            for v in acc.values():
                v.zero_()
            for v in acc_m.values():
                v.zero_()
            n_acc = n_m = 0

        if step % args.save_every == 0 or step == args.steps:
            # Not splat.ply: a run that dies at step 12k would otherwise leave a
            # half-trained cloud where the last complete one was, with every
            # exported asset downstream still built from that one. A boundary that
            # is also a reset step is exported with the opacities as trained, so a
            # rescue never reads a cloud that has just been demoted.
            snap = capture.opas if capture is not None else None
            with trained_opacities(params, snap) as restored:
                export_ply(params["means"], F.normalize(params["quats"], dim=1),
                           params["scales"], params["opacities"],
                           params["sh0"], params["shN"], prog / "splat.partial.ply",
                           note=f"step {step}" + (" (opacities as trained, before "
                                                 "this step's reset)" if restored
                                                 else ""))
                with torch.no_grad():
                    rgb_eval, _, _ = rendering(
                        params["means"], F.normalize(params["quats"], dim=1),
                        torch.exp(params["scales"]),
                        torch.sigmoid(params["opacities"]),
                        torch.cat([params["sh0"], params["shN"]], dim=1),
                        viewmat, K, W, H, render_mode="RGB",
                        sh_degree=SH_DEG, packed=True,
                        rasterize_mode="antialiased" if args.antialias else "classic")
                    rgb_eval = rgb_eval.permute(0, 3, 1, 2)
                    prev = (rgb_eval[0].clamp(0, 1) * 255).byte()
                    prev = prev.permute(1, 2, 0).cpu().numpy()
                # Instrumentation, not output: the checkpoint above is what a
                # rescue reads, so a refused preview must not end the solve.
                if not rb.save_image(Image.fromarray(prev),
                                     prog / f"step_{step:06d}.jpg", quality=92):
                    say(f"[train] step {step}: preview jpg refused, continuing")

    # ---- the exported cloud, and every metric below, is the trained one ------
    # If the last step was a reset step, gsplat has just clamped every opacity to
    # 0.04 and the cloud standing here is the demoted one. Restore the captured
    # pre-reset opacities for the export and the evaluation; the program ends with
    # this block, so nothing further can train on the restored values (a mid-run
    # partial puts them back after writing, via the same context manager).
    snap = capture.opas if capture is not None else None
    demoted_end, restore_end, refuse_end = artefact_reset_state(
        args.steps, RESET_EVERY, args.refine_stop, snapshot_ready=snap is not None)
    opa_demoted_max = float(torch.sigmoid(params["opacities"]).max())
    if refuse_end:
        say(f"[train] REFUSING TO REPORT: step {args.steps} is an opacity-reset step "
            f"and the pre-reset opacities were not captured (max sigmoid opacity "
            f"{opa_demoted_max:.4f}), so this cloud is the demoted one and any metric "
            f"taken from it measures the reset. Not shipping it and not scoring it: "
            f"see artefact_reset_state and the --steps 6000 measurement in the module "
            f"docstring.")
        sys.exit(1)
    if restore_end:
        with torch.no_grad():
            params["opacities"].data.copy_(snap)
    opa_reported_max = float(torch.sigmoid(params["opacities"]).max())
    final_eval = dict(reported_at_step=args.steps, requested_step=args.steps,
                      last_step_was_reset=demoted_end,
                      opacities_restored=bool(restore_end),
                      reset_every_steps=RESET_EVERY,
                      refine_stop=args.refine_stop, resets_at_steps=resets_crossed,
                      max_opacity_as_left_by_gsplat=opa_demoted_max,
                      max_opacity_reported=opa_reported_max,
                      recovery_window_steps=RESET_RECOVERY_TAG_STEPS,
                      why=(f"step {args.steps} is an opacity-reset step: gsplat left "
                           f"every opacity at <= {opa_demoted_max:.3f}, so the "
                           f"exported cloud and every metric below are the captured "
                           f"pre-reset opacities (max {opa_reported_max:.3f}). "
                           f"Training itself continued from the reset, as gsplat "
                           f"schedules it" if demoted_end else
                           f"step {args.steps} is not an opacity-reset step, so the "
                           f"cloud and the metrics are exactly the requested run"))
    if restore_end:
        say(f"[train] reporting the cloud as trained at step {args.steps}: opacities "
            f"restored from the gsplat reset's {opa_demoted_max:.3f} max to "
            f"{opa_reported_max:.3f} (every in-run step still ran on the reset "
            f"schedule, so this is the model that step produced, not its demotion)")

    export_ply(params["means"], F.normalize(params["quats"], dim=1),
               params["scales"], params["opacities"],
               params["sh0"], params["shN"], args.work / "splat.ply",
               note=(f"step {args.steps}" + (" (opacities as trained, before the reset "
                                            "that ends this step)" if restore_end
                                             else "")))
    say(f"[train] DONE in {time.time()-t0:.0f}s "
        f"({args.steps / (time.time()-t0):.1f} it/s), final N={len(params['means'])}"
        + (f" -- exported and scored with the pre-reset opacities: {args.steps} is "
           f"an opacity-reset step" if restore_end else ""))

    # ---- measured report -----------------------------------------------------
    # Held-out first, because it is the only answer to "did the geometry
    # generalise"; the in-sample numbers are labelled as such.
    kn = max(n_tot, 1)
    km = max(n_m_tot, 1)
    tm = {k: float(v) / kn for k, v in tot.items()}
    tmm = {k: (float(v) / km if n_m_tot else None) for k, v in tot_m.items()}
    report = dict(
        work=str(args.work), steps=args.steps, cap=args.cap, resolution=[W, H],
        gaussians=len(params["means"]), train_views=len(data),
        seconds=round(time.time() - t0, 1),
        # What the numbers below describe. `in_sample`/`held_out` are meaningless
        # without it: a metric taken on an opacity-reset step reads ~18 dB below the
        # same model's, and an artefact taken there is a demoted cloud.
        final_eval=final_eval,
        room_anchors=dict(applied=bool(want_anchors), why=anchor_note, **anchor_ev),
        # Allocated is what the tensors hold; reserved is what the card is committed
        # to and therefore what an OOM at 6 GiB is actually decided by. nvidia-smi on
        # Windows under-reports both, so the two allocator figures are what is quoted.
        peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        peak_mem_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
        it_per_s=round(args.steps / max(time.time() - t0, 1e-6), 2),
        ssim_weight=args.ssim_weight,
        depth=dict(armed=depth_term is not None, weight=args.depth_weight,
                   warmup_start_step=depth_start_at,
                   warmup_full_step=int(depth_full_at),
                   tau_rel=args.depth_tau, support_points=args.depth_support,
                   window_px=args.depth_window, err_ref_px=args.depth_err_px,
                   nn_spacing_units=spacing,
                   nn_spacing_m=(spacing * mpu if mpu else None),
                   run_mean=dict(contrib=tm["dep"], weight=tm["w"],
                                 raw=tmm["raw"], err_units=tmm["err"],
                                 coverage=tmm["cov"], support_mass=tmm["wsup"],
                                 samples=n_m_tot,
                                 measured=n_m_tot > 0),
                   cost=dict(depth_channel_renders=n_depth_renders,
                             steps=args.steps,
                             share_of_steps=round(n_depth_renders / max(args.steps, 1), 4),
                             probe_every_steps=DEPTH_PROBE_EVERY,
                             resident_mb=(depth_term.stats["scratch_mb"]
                                          + depth_term.stats["pts_mb"])
                             if depth_term else 0.0,
                             stats=depth_term.stats if depth_term else None),
                   note=depth_note))
    say(f"[train] every metric below is from the cloud as trained at step "
        f"{args.steps}" + (", with the opacities gsplat's reset at that step had just "
                          f"clamped away put back" if restore_end else ""))

    if data:
        # In-sample: the term has seen these views, so this is a fit number, not a
        # generalisation number. Reported anyway because that is what the existing
        # log line has always measured, and a change in it must be attributable.
        sample = data[::max(1, len(data) // 12)][:12]
        ev = evaluate(params, sample, ground_truth(sample, device, W, H), W, H,
                      rendering, device, args, extent,
                      out_dir=(prog / "eval_insample") if args.eval_images else None)
        report["in_sample"] = ev
        say(f"[train] in-sample ({len(sample)} trained views): psnr {ev['psnr']:.2f} "
            f"ssim {ev['ssim']:.4f} cross-view depth p50 "
            f"{fmt_cvd(ev['cv_depth'])}")
    if eval_data:
        ev = evaluate(params, eval_data, ground_truth(eval_data, device, W, H), W, H,
                      rendering, device, args, extent,
                      out_dir=(prog / "eval_heldout") if args.eval_images else None)
        report["held_out"] = ev
        say(f"[train] HELD-OUT ({len(eval_data)} views, never supervised): "
            f"psnr {ev['psnr']:.2f} (worst view {ev['psnr_min']:.2f}) "
            f"ssim {ev['ssim']:.4f} cross-view depth {fmt_cvd(ev['cv_depth'])}")
    # Did the cloud end up holding the splat? Reported in scene units and, when
    # frame.json anchors them, in metres. Not measurable on a near-empty cloud,
    # which says so instead of printing 0%.
    if spacing > 0:
        gen2 = torch.Generator(device=device)
        gen2.manual_seed(args.seed + 2)
        sub = pts_gpu[torch.randperm(len(pts_gpu), device=device,
                                      generator=gen2)[:50_000]]
        sup = cloud_support(params["means"], sub, NN_SUPPORT_SPACINGS * spacing,
                            gen=gen2)
    else:
        sup = None
        say("[train] gaussians-vs-cloud support NOT MEASURABLE: the cloud has no "
            "measurable neighbour spacing (too few points), so no reach can be "
            "defined. This is not 0%.")
    if sup:
        report["gaussians_near_observed_geometry"] = sup
        say(f"[train] {sup['fraction'] * 100:.1f}% of gaussian centres have a sparse "
            f"point within {sup['reach_units']:.4g} scene units "
            f"({sup['nn_median']:.4g} units median nearest-point distance)"
            + (f" = {sup['reach_units'] * mpu:.3g} m" if mpu else ", no scale anchor"))
    rb.write_json(prog / "train_report.json", report)
    say(f"[train] report written to {prog / 'train_report.json'}")


if __name__ == "__main__":
    main()
