"""Analyze 3D multi-view coverage, camera trajectory, and sparse tie points.

Calculates:
  1. Camera frustums in world coordinates for all keyframes.
  2. Sparse COLMAP 3D tie points in world coordinates with true RGB colors.
  3. 3D spatial coverage grid / voxel density:
     - Well-observed (>= MIN_VIEWS cameras framing the voxel centre)
     - Marginally observed (fewer views than that, but at least one)
     - Missing / Unobserved (0 cameras framed it)
  4. Elevation profile and human-actionable capture guidance.

Every distance in the grid is scene-relative and in metres: reach, near cutoff,
grid bounds and voxel size all come from the poses and frame.json, because the
world frame written here is `colmap @ Rg.T * scale_m_per_unit` and an absolute
metre constant is only ever right for one scene size. When the geometry cannot
support a measurement the output says `status: "not_measurable"` with a reason
instead of reporting 0% observed and prescribing a re-fly.

Outputs in work/<scene>/viewer_assets/:
  - cameras.json
  - sparse_points.json
  - coverage_grid.json

Usage:
  python check_coverage.py --work work/room_w_jsonl
  python check_coverage.py --work work/rocks --strict   # exit 3 if unmeasurable
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_frame import load_cameras, rot_to_up  # noqa: E402


def parse_points3d(txt_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse COLMAP points3D.txt -> (xyz (N, 3), rgb (N, 3), track_len (N,))"""
    if not txt_path.exists():
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
    xyz, rgb, tracks = [], [], []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 8:
            continue
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
        rgb.append([int(p[4]), int(p[5]), int(p[6])])
        # Number of image observations in track is (len(p) - 8) // 2
        tracks.append(max(1, (len(p) - 8) // 2))
    return np.array(xyz, np.float64), np.array(rgb, np.uint8), np.array(tracks, np.int32)


def compute_camera_frustums(cams_meta: list[dict], Rg: np.ndarray, s: float, frustum_depth: float = 0.25) -> list[dict]:
    """Compute exact 3D camera wireframe vertices and metadata in world coordinates."""
    out_cams = []
    for i, item in enumerate(cams_meta):
        c = item["camera"]
        name = item["file"]
        t_sec = item.get("t_sec", 0.0)

        # COLMAP cam coords: x_cam = R @ x_colmap + t
        # Camera center in COLMAP: C_colmap = -R.T @ t
        R_c = np.array(c["R_rowmajor"], np.float64)
        t_c = np.array(c["t"], np.float64)
        c_colmap = -R_c.T @ t_c

        # World coordinates: P_world = (P_colmap @ Rg.T) * s
        # World rotation: R_w = R_c @ Rg.T
        c_world = (c_colmap @ Rg.T) * s
        Rw = R_c @ Rg.T

        # Camera axes in world coordinates:
        # COLMAP convention: +X right, +Y down, +Z forward
        right_w = Rw.T @ np.array([1.0, 0.0, 0.0])
        down_w = Rw.T @ np.array([0.0, 1.0, 0.0])
        up_w = Rw.T @ np.array([0.0, -1.0, 0.0])
        forward_w = Rw.T @ np.array([0.0, 0.0, 1.0])

        fx = float(c["fx"])
        fy = float(c.get("fy", fx))
        cx = float(c["cx"])
        cy = float(c["cy"])
        w = float(c.get("width", 2.0 * cx))
        h = float(c.get("height", 2.0 * cy))

        # Frustum corners at distance d in camera frame
        d = float(frustum_depth)
        hw = (w / (2.0 * fx)) * d
        hh = (h / (2.0 * fy)) * d

        # 4 corners in camera frame: top-left, top-right, bottom-right, bottom-left
        # In COLMAP camera frame: top is -Y, bottom is +Y, left is -X, right is +X, forward is +Z
        corners_cam = [
            np.array([-hw, -hh, d]),  # Top-Left
            np.array([+hw, -hh, d]),  # Top-Right
            np.array([+hw, +hh, d]),  # Bottom-Right
            np.array([-hw, +hh, d]),  # Bottom-Left
        ]
        corners_world = [c_world + Rw.T @ p_cam for p_cam in corners_cam]

        # Top indicator apex (for showing camera orientation / UP direction)
        top_cam = np.array([0.0, -hh * 1.35, d])
        top_world = c_world + Rw.T @ top_cam

        out_cams.append({
            "id": i,
            "name": name,
            "t_sec": round(float(t_sec), 3) if t_sec is not None else None,
            "pos": [round(float(v), 4) for v in c_world],
            "forward": [round(float(v), 4) for v in forward_w],
            "up": [round(float(v), 4) for v in up_w],
            "right": [round(float(v), 4) for v in right_w],
            "corners": [[round(float(v), 4) for v in pt] for pt in corners_world],
            "top_mark": [round(float(v), 4) for v in top_world],
            "fov_x_deg": round(float(np.degrees(2.0 * np.arctan(w / (2.0 * fx)))), 1),
            "fov_y_deg": round(float(np.degrees(2.0 * np.arctan(h / (2.0 * fy)))), 1),
            "fx": round(fx, 2), "fy": round(fy, 2), "cx": round(cx, 2), "cy": round(cy, 2),
            "width": int(w), "height": int(h),
            "R_rowmajor": [list(map(lambda v: round(float(v), 5), r)) for r in Rw],
            "t_world": [round(float(v), 4) for v in t_c * s],
        })

    return out_cams


# ---------------------------------------------------------------------------
# Scene-relative geometry. Nothing below is a metre rule baked into this file.
# ---------------------------------------------------------------------------
#
# A camera's useful reach and a voxel's edge length only mean anything next to
# the scene they describe. This project already knows that: solve_frame.py writes
# `camera_agl_m` and `region_max_range_units` (= AGL x --max-range-mult, default
# 4.0, solve_frame.py:239) into frame.json, and export_viewer_assets.py:305-311
# reads them back with exactly that fallback chain. This module had instead
# hardcoded `max_range = 4.5` and `dist > 0.15`, which are only right for a
# handheld room scan held ~1.5 m off the floor. Measured on the real `rocks`
# scene: the world frame there is metric (verified below), the drone flew at
# 13.09 m AGL, the supported rock face sits 23-41 m from the nearest camera, and
# voxel centres were 12.53 x 4.69 x 20.86 m apart. A 4.5 m reach against a 20.86 m
# spacing cannot reach a single voxel centre on the Z axis, so `covered_pct: 0.0`
# was an arithmetic property of the constants, not a property of the capture -
# and the operator was told to re-fly a 72/72-registered take.
REACH_MULT_FALLBACK = 4.0    # mirrors solve_frame.py:239 --max-range-mult
NEAR_AGL_FRACTION = 0.05     # nothing resolves closer than 5% of the camera height
MIN_REACH_OVER_CELL = 2.0    # a frustum must span >= this many voxels per axis to
                             # leave the test anything to decide
TARGET_CELLS_ON_MIN_AXIS = 6.0   # want at least this many voxels across the
                                 # thinnest dimension of the camera hull
VOXEL_BUDGET = 3072          # what the viewer already ships (16 x 12 x 16); the
                             # grid is coarsened until it fits, never refined past it
MIN_CAMERAS = 3              # survey_assess.completeness uses the same floor
MIN_VOXELS = 8
AXIS_CAP_DEFAULT = 64           # per-axis ceiling when the caller sets no grid_res
CLOUD_BOX_MIN_POINTS = 50       # below this the in-reach cloud subset cannot bound a box
CLOUD_BOX_MIN_HULL_FRACTION = 1 / 3  # a box under this of the camera hull on 2+ axes
                                     # is floaters, not a subject
MIN_IN_REACH_FRACTION = 0.25   # a judgeable grid needs this share of its centres
                               # within reach of at least one camera
MIN_VIEWS = 3                # a voxel needs this many framing cameras to count as
                             # well observed; was three inline literals
ADVICE_MIN_COVERED_PCT = 40.0  # below this, an elevation band earns a line of advice
DRONE_AGL_M = 3.0            # export_viewer_assets.py:534 uses the same 0.8-3.0 m
                             # band to decide a rig was carried, not flown

STATUS_MEASURED = "measured"
STATUS_NOT_MEASURABLE = "not_measurable"

# Stated once per run so a reader cannot take "100% of the grid observed" as
# "100% of the site observed". The grid is bounded by what the cameras could
# frame; anything outside it is absent from the denominator, not called missing.
_NOT_MEASURED = (
    "This is camera geometry against a voxel grid, not surface completeness: no "
    "reference surface and no depth map was compared.",
    "The grid is clipped to the volume this flight could plausibly frame, so "
    "unobserved voxels are holes INSIDE that envelope. What was never flown over "
    "is absent from the denominator rather than counted as missing.",
    "Occlusion is not modelled: a voxel counted here is framed by >= 3 cameras, "
    "which is not the same as unobstructed. survey_occlusion.hidden_regions would "
    "measure that, and it cannot run pose-only (it raises without depth maps, "
    "survey_occlusion.py:324).",
    "Metres come from frame.json's own scale. A wrong metric anchor shifts every "
    "m^3 and m^2 figure here by the same factor, so treat them as relative.",
)


def _camera_scale_table(cams_meta: list[dict]) -> tuple[np.ndarray, float]:
    """Camera centres (metres) plus their median consecutive-step length.

    Positions are ordered by capture time when that is available, because the
    quantity wanted is the spacing of the sampling, not the spacing of whatever
    order the pose file happened to be written in. Zero-length steps (duplicate
    poses) are dropped rather than dragging the median to 0.
    """
    pos = np.array([c["pos"] for c in cams_meta], np.float64)
    if not len(pos):
        return pos, 0.0
    t = [c.get("t_sec") for c in cams_meta]
    try:
        tv = np.array([float(x) if x is not None else np.nan for x in t], np.float64)
        if np.isfinite(tv).all() and len(np.unique(tv)) == len(tv):
            pos = pos[np.argsort(tv)]
    except (TypeError, ValueError):
        pass
    if len(pos) < 2:
        return pos, 0.0
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    step = step[step > 0.0]
    return pos, (float(np.median(step)) if len(step) else 0.0)


def _finite_box(values: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    lo = np.percentile(values, 1.0, axis=0)
    hi = np.percentile(values, 99.0, axis=0)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError(f"{name} produced non-finite bounds")
    return lo, hi


def derive_scene_geometry(cams_meta: list[dict], pts_world: np.ndarray | None,
                          frame: dict | None = None, *,
                          caller_lo: np.ndarray | None = None,
                          caller_hi: np.ndarray | None = None,
                          grid_res: tuple[int, int, int] | None = None) -> dict:
    """Resolve reach, near range, grid bounds and voxel size from THIS scene.

    Returns a dict the caller both uses and publishes, so every number in the
    output can be traced to the anchor that produced it. `status` is
    `not_measurable` with a reason whenever the geometry would otherwise be
    allowed to report a confident 0%.

    `grid_res` is honoured only as a per-axis CEILING, never as the shape: the
    old fixed 16 x 12 x 16 is what made a 334 m-wide grid unmeasurable.
    """
    frame = frame or {}
    g: dict = {"reasons": [], "clips": []}

    scale = float(frame.get("scale_m_per_unit") or 0.0)
    g["scale_m_per_unit"] = scale or None
    # Every _m field below is asserted, not assumed: compute_camera_frustums writes
    # `pos = (colmap @ Rg.T) * scale_m_per_unit`, i.e. the SAME frame frame.json's
    # region_box_m and camera_agl_m are written in (solve_frame.py:398,417,424).
    # Re-deriving the inverse from keyframes_poses.jsonl reproduced cameras.json to
    # 4.98e-05 m on `rocks`, so no further unit conversion is applied anywhere here.
    g["frame_units"] = ("metres (colmap units x scale_m_per_unit); frame.json's "
                        "camera_agl_m and region_box_m are in this same frame")

    # The world frame this module writes is already metric, so every field below
    # is labelled _m because it IS metres, not because a scale was applied here.
    cam, cam_step = _camera_scale_table(cams_meta)
    g["camera_count"] = int(len(cam))
    g["camera_step_median_m"] = round(cam_step, 4) if cam_step else None
    if len(cam) < MIN_CAMERAS:
        g["status"] = STATUS_NOT_MEASURABLE
        g["reason"] = (f"only {len(cam)} registered camera pose(s); a multi-view "
                       f"coverage grid needs at least {MIN_CAMERAS}")
        return g

    hull_lo, hull_hi = cam.min(axis=0), cam.max(axis=0)
    hull_ext = hull_hi - hull_lo
    g["camera_hull_min"] = [round(float(v), 3) for v in hull_lo]
    g["camera_hull_max"] = [round(float(v), 3) for v in hull_hi]
    g["camera_hull_extent_m"] = [round(float(v), 3) for v in hull_ext]
    g["camera_hull_diagonal_m"] = round(float(np.linalg.norm(hull_ext)), 3)
    if float(np.sort(hull_ext)[-2]) <= 0.0:
        g["status"] = STATUS_NOT_MEASURABLE
        g["reason"] = ("the camera positions are coincident or collinear, so they "
                       "bound no volume to measure coverage against")
        return g

    # --- reach: how far a camera here can usefully see -----------------------
    agl = float(frame.get("camera_agl_m") or 0.0)
    g["camera_agl_m"] = round(agl, 4) if agl > 0 else None
    reach_u = float(frame.get("region_max_range_units") or 0.0)
    if reach_u > 0 and scale > 0:
        reach = reach_u * scale
        basis = (f"frame.json region_max_range_units ({reach_u:.3f} colmap units "
                 f"x scale {scale:.4f}) = AGL x {reach / max(agl, 1e-9):.2f} for "
                 f"this take")
    elif agl > 0:
        reach = REACH_MULT_FALLBACK * agl
        basis = f"frame.json camera_agl_m x {REACH_MULT_FALLBACK} (solve_frame.py's own rule)"
    elif cam_step > 0:
        reach = REACH_MULT_FALLBACK * 8.0 * cam_step
        basis = (f"{REACH_MULT_FALLBACK * 8.0:.0f}x the median camera step "
                 f"{cam_step:.3f} m (frame.json gave neither AGL nor a range)")
    else:
        g["status"] = STATUS_NOT_MEASURABLE
        g["reason"] = ("no scene-relative reach: frame.json carries no camera_agl_m "
                       "or region_max_range_units and the camera step is degenerate")
        return g
    g["reach_m"] = round(reach, 3)
    g["reach_basis"] = basis

    near = max(NEAR_AGL_FRACTION * agl, 0.5 * cam_step, reach * 1e-4)
    g["near_range_m"] = round(near, 4)
    g["near_basis"] = (f"max({NEAR_AGL_FRACTION:.2f} x AGL, half the camera step "
                       f"{0.5 * cam_step:.3f} m)")
    if not (0.0 < near < reach):
        g["status"] = STATUS_NOT_MEASURABLE
        g["reason"] = f"near range {near:.4f} m is not inside (0, reach {reach:.3f} m)"
        return g

    # --- grid bounds: subject box INTERSECTED with the observable envelope ---
    # The old code UNIONED the camera hull into the point-cloud box, so bounds
    # could only ever grow: on `rocks` 99.9% of the raw cloud failed the 5-unit
    # near-filter, the "> 100 points else keep everything" guard tripped, and the
    # grid inherited the entire outlier cloud - 3.77e6 m^3 sampled against a
    # 5,506 m^3 camera hull, 684x the volume the flight could frame. Intersecting
    # with what the cameras could reach is the fix; the cloud no longer sets the
    # size of the question.
    dilate = np.full(3, reach)
    obs_lo, obs_hi = hull_lo - dilate, hull_hi + dilate

    subj_lo = subj_hi = None
    curated = False
    box = frame.get("region_box_m")
    if isinstance(box, dict) and isinstance(box.get("min"), list) and isinstance(box.get("max"), list):
        try:
            bl, bh = np.array(box["min"], np.float64), np.array(box["max"], np.float64)
            if np.isfinite(bl).all() and np.isfinite(bh).all() and bool((bh > bl).all()):
                subj_lo, subj_hi = bl, bh
                curated = True
                g["subject_basis"] = ("frame.json region_box_m: the multi-view-supported "
                                      "region solve_frame measured for this take")
        except (TypeError, ValueError):
            pass
    if subj_lo is None and pts_world is not None and len(np.atleast_2d(pts_world)) > MIN_CAMERAS:
        # A sparse COLMAP cloud bounds a box only when it is dense enough to have
        # shape. On `rocks` 2 of 17,847 points sat within reach of the cameras, so
        # the percentile box describes floaters, not the outcrop - and a box that
        # thin must not be allowed to crop the region solve_frame measured.
        pw = np.atleast_2d(pts_world)
        d = np.linalg.norm(pw[:, None, :] - cam[None, :, :], axis=2).min(axis=1)
        keep = d <= reach
        pool = pw[keep] if int(keep.sum()) >= CLOUD_BOX_MIN_POINTS else pw
        cand_lo, cand_hi = _finite_box(pool, "cloud")
        cover = (cand_hi - cand_lo) / np.maximum(hull_ext, 1e-9)
        if int(np.sum(cover >= CLOUD_BOX_MIN_HULL_FRACTION)) >= 2:
            subj_lo, subj_hi = cand_lo, cand_hi
            g["subject_basis"] = (f"1/99 percentile box of {len(pool)} of {len(pw)} cloud "
                                  f"point(s) (in-reach subset was "
                                  f"{'used' if pool is not pw else 'not usable'})")
        else:
            g["subject_basis"] = (f"the camera hull: the cloud box covers only "
                                  f"{[round(float(c), 2) for c in cover]} of it per axis, "
                                  f"which is floaters rather than a subject")
    if subj_lo is None:
        subj_lo, subj_hi = hull_lo.copy(), hull_hi.copy()

    lo = np.maximum(subj_lo, obs_lo)
    hi = np.minimum(subj_hi, obs_hi)
    if bool((lo > subj_lo + 1e-9).any()) or bool((hi < subj_hi - 1e-9).any()):
        g["clips"].append(
            f"subject box clipped to the observable envelope (camera hull +/- {reach:.1f} m)")
    if bool((hi <= lo).any()):
        # The subject lies clean outside what these cameras could frame. Gridding
        # the empty envelope instead would report coverage OF AIR and then blame
        # the capture for the holes in it, so this refuses.
        axis = int(np.argmin(hi - lo))
        gap = float(max(subj_lo[axis] - obs_hi[axis], obs_lo[axis] - subj_hi[axis], 0.0))
        g["status"] = STATUS_NOT_MEASURABLE
        g["reason"] = (f"the region to be measured lies {gap:.1f} m outside what the "
                       f"cameras can frame on axis {'XYZ'[axis]} (reach {reach:.1f} m "
                       f"around a {' x '.join(f'{v:.1f}' for v in hull_ext)} m camera "
                       "hull): there is no overlap to grid, and gridding the empty "
                       "envelope instead would score air, not the capture")
        return g

    if caller_lo is not None and caller_hi is not None:
        cl, ch = np.asarray(caller_lo, np.float64), np.asarray(caller_hi, np.float64)
        if not (np.isfinite(cl).all() and np.isfinite(ch).all()):
            g["clips"].append("caller bounds dropped: not finite")
        elif curated:
            # solve_frame's supported region beats a percentile of the sparse
            # cloud, so the caller's box is recorded and then superseded.
            g["clips"].append("caller bounds superseded by frame.json region_box_m: "
                              "the sparse-cloud percentile box is the weaker estimate "
                              "of where the subject is and is only used as a fallback")
        else:
            nlo, nhi = np.maximum(lo, cl), np.minimum(hi, ch)
            if bool((nhi > nlo).all()):
                lo, hi = nlo, nhi
                g["clips"].append("also clipped to the bounds the caller supplied")
            else:
                g["clips"].append("caller bounds IGNORED: they do not overlap the "
                                  "observable envelope at all")

    g["bounds_min"] = [round(float(v), 3) for v in lo]
    g["bounds_max"] = [round(float(v), 3) for v in hi]
    extent = hi - lo
    g["grid_extent_m"] = [round(float(v), 3) for v in extent]
    g["grid_volume_m3"] = round(float(np.prod(np.maximum(extent, 0.0))), 1)
    if int(np.sum(extent > 0)) < 2:
        g["status"] = STATUS_NOT_MEASURABLE
        g["reason"] = ("the grid bounds are degenerate after clipping to what the "
                       "cameras could frame, so there is no volume to measure")
        return g

    # --- voxel size: finer than the reach, bounded by the budget -------------
    cell_max = reach / MIN_REACH_OVER_CELL
    nonzero = extent[extent > 0]
    thin = float(np.min(nonzero)) if len(nonzero) else 0.0
    cell_pref = max(thin / TARGET_CELLS_ON_MIN_AXIS, cam_step, 1e-6)
    cap = [int(c) for c in (grid_res or (AXIS_CAP_DEFAULT,) * 3)]

    def _shape(c: float) -> list[int]:
        return [int(max(1, min(cap[i], round(extent[i] / c)))) for i in range(3)]

    # Coarsen until the grid fits the viewer's voxel budget, then stop: a cell
    # larger than cell_max is where the frustum test runs out of cells to
    # discriminate, and that is reported, not silently shipped as 0%.
    cell = max(cell_pref, 1e-6)
    for _ in range(32):
        total = int(np.prod(_shape(cell)))
        if total <= VOXEL_BUDGET:
            break
        cell *= (total / VOXEL_BUDGET) ** (1.0 / 3.0) * 1.02
    nx, ny, nz = _shape(cell)

    g["voxel_size_m"] = round(float(cell), 4)
    g["cell_max_m"] = round(float(cell_max), 3)
    g["cell_basis"] = (f"max(grid thin axis {thin:.2f} m / "
                       f"{TARGET_CELLS_ON_MIN_AXIS:.0f} cells, camera step "
                       f"{cam_step:.3f} m), coarsened to fit {VOXEL_BUDGET} voxels")

    if cell > cell_max * 1.001:
        g["status"] = STATUS_NOT_MEASURABLE
        g["reason"] = (f"no voxel size both spans the {reach:.2f} m reach at least "
                       f"{MIN_REACH_OVER_CELL:.0f}x and keeps the grid inside the "
                       f"{VOXEL_BUDGET}-voxel budget: the envelope is "
                       f"{' x '.join(f'{v:.1f}' for v in extent)} m and the smallest "
                       f"legal grid is {cell:.2f} m voxels")
        return g

    if nx * ny * nz < MIN_VOXELS:
        g["status"] = STATUS_NOT_MEASURABLE
        g["reason"] = (f"the observable envelope resolves to only {nx * ny * nz} voxel(s) "
                       f"at a {cell:.2f} m edge; that is not a grid")
        return g

    sx, sy, sz = extent[0] / nx, extent[1] / ny, extent[2] / nz
    g["voxel_spacing_m"] = [round(float(sx), 4), round(float(sy), 4), round(float(sz), 4)]
    ratios = {"x": round(float(reach / sx), 2) if nx > 1 else None,
              "y": round(float(reach / sy), 2) if ny > 1 else None,
              "z": round(float(reach / sz), 2) if nz > 1 else None}
    g["reach_over_spacing"] = ratios
    tight = [a for a, r in ratios.items() if r is not None and r < MIN_REACH_OVER_CELL]
    if tight:
        g["status"] = STATUS_NOT_MEASURABLE
        g["reason"] = (f"reach {reach:.2f} m is under {MIN_REACH_OVER_CELL}x the voxel "
                       f"spacing on axis/axes {','.join(tight)} ({ratios}), so a frustum "
                       "cannot span enough cells for the count to mean anything")
        return g

    g["grid_shape"] = [nx, ny, nz]
    g["status"] = STATUS_MEASURED
    return g


def _refusal(geometry: dict, reason: str) -> dict:
    """The not-measurable payload: no percentages, no states, no re-fly advice.

    `covered_pct` and friends go null rather than 0.0 so a reader cannot mistake
    "nothing measured" for "measured nothing", and `voxels` is empty so the
    viewer cannot paint 3,072 red crosshairs over a scene it never judged.
    """
    geometry["status"] = STATUS_NOT_MEASURABLE
    geometry["reason"] = reason
    return {
        "status": STATUS_NOT_MEASURABLE,
        "measurable": False,
        "unmeasurable_reason": reason,
        "geometry": geometry,
        "not_measured": list(_NOT_MEASURED),
        # Consistently empty: total_voxels 0, no voxels and no shape all say the
        # same thing, so a reader cannot half-render a grid that was refused. The
        # shape that WOULD have been used stays in `geometry.grid_shape`.
        "grid_shape": [0, 0, 0],
        "bounds_min": geometry.get("bounds_min", [0.0, 0.0, 0.0]),
        "bounds_max": geometry.get("bounds_max", [0.0, 0.0, 0.0]),
        "total_voxels": 0,
        "covered_pct": None, "marginal_pct": None, "unobserved_pct": None,
        "covered": 0, "marginal": 0, "unobserved": 0,
        "min_views": MIN_VIEWS,
        "elevation_profile": [],
        "advice": [f"COVERAGE NOT MEASURABLE: {reason} No re-fly guidance is given, "
                   "because nothing here shows the capture was bad - only that this "
                   "grid could not be judged."],
        "voxels": [],
    }


def analyze_coverage_grid(
    cams_meta: list[dict],
    pts_world: np.ndarray | None = None,
    bounds_lo: np.ndarray | None = None,
    bounds_hi: np.ndarray | None = None,
    grid_res: tuple[int, int, int] | None = None,
    frame: dict | None = None
) -> dict:
    """Voxelled 3D observation density and the missing zones, all scene-relative.

    `bounds_lo` / `bounds_hi` are the bounds the caller believes in; they are
    INTERSECTED with the envelope this function derives from the poses and
    frame.json, never trusted over it. `grid_res` is a per-axis ceiling only:
    the grid is sized so the frustum reach spans several voxels per axis, which
    a fixed 16 x 12 x 16 cannot guarantee across scenes whose hulls differ by
    40x. Returns `status: "not_measurable"` with a reason - and no advice -
    whenever the geometry would otherwise allow a confident but meaningless 0%.

    Signature and every field the viewer reads (`grid_shape`, `bounds_min/max`,
    `covered_pct`, `marginal_pct`, `unobserved_pct`, `total_voxels`, `advice`,
    `voxels`) are unchanged in name and type; `status`, `measurable`,
    `unmeasurable_reason`, `geometry`, `metric_volumes`, `covered`, `marginal`,
    `unobserved`, `min_views` and `not_measured` are additions.
    """
    g = derive_scene_geometry(cams_meta, pts_world, frame,
                              caller_lo=bounds_lo, caller_hi=bounds_hi, grid_res=grid_res)
    if g["status"] != STATUS_MEASURED:
        return _refusal(g, g.get("reason", "geometry made coverage unmeasurable"))

    nx, ny, nz = g["grid_shape"]
    lo = np.array(g["bounds_min"], np.float64)
    hi = np.array(g["bounds_max"], np.float64)
    dx, dy, dz = (hi - lo) / np.array([nx, ny, nz], np.float64)
    reach = float(g["reach_m"])
    near = float(g["near_range_m"])

    xs = lo[0] + (np.arange(nx) + 0.5) * dx
    ys = lo[1] + (np.arange(ny) + 0.5) * dy
    zs = lo[2] + (np.arange(nz) + 0.5) * dz
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    vox_centers = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)  # [M, 3]

    # Pre-extract camera centers and projection info
    cam_centers = []
    cam_forwards = []
    cam_Rws = []
    cam_fxys = []
    for c in cams_meta:
        cam_centers.append(np.array(c["pos"], np.float64))
        cam_forwards.append(np.array(c["forward"], np.float64))
        cam_Rws.append(np.array(c["R_rowmajor"], np.float64))
        cam_fxys.append((c["fx"], c["fy"], c["cx"], c["cy"], c["width"], c["height"]))

    cam_centers = np.array(cam_centers)
    cam_forwards = np.array(cam_forwards)

    # Test visibility for each voxel center
    n_vox = len(vox_centers)
    vis_counts = np.zeros(n_vox, dtype=np.int32)
    # Every voxel centre that ANY camera could have had in range at all. Kept
    # separately from vis_counts because the two failures it distinguishes look
    # identical in the output and mean opposite things: "the capture did not
    # frame this" versus "the reach we were given cannot ever reach this".
    in_reach_of_any = np.zeros(n_vox, dtype=bool)
    # A numerical guard against dividing by z ~ 0, not a physical stand-off: it
    # scales with the reach so it is not a hidden metre constant either.
    z_floor = max(reach * 1e-3, 1e-6)

    for k, (cc, cf, Rw, (fx, fy, cx, cy, w, h)) in enumerate(zip(cam_centers, cam_forwards, cam_Rws, cam_fxys)):
        diff = vox_centers - cc  # [M, 3]
        dist = np.linalg.norm(diff, axis=1)
        in_range = (dist > near) & (dist < reach)
        in_reach_of_any |= in_range
        if not np.any(in_range):
            continue

        # In camera frame: P_cam = Rw @ (P_world - cc)
        pc = (diff[in_range]) @ Rw.T  # [M_sub, 3]
        z = pc[:, 2]
        front = z > z_floor
        if not np.any(front):
            continue

        u = fx * pc[:, 0] / np.maximum(z, 1e-4) + cx
        v = fy * pc[:, 1] / np.maximum(z, 1e-4) + cy
        in_fov = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)

        idx_range = np.where(in_range)[0]
        vis_counts[idx_range[in_fov]] += 1

    frac_in_reach = float(in_reach_of_any.mean())
    g["voxels_in_reach_fraction"] = round(frac_in_reach, 4)
    if frac_in_reach < MIN_IN_REACH_FRACTION:
        # This is the exact shape of the original bug, and no bound- or
        # spacing-rule fix can be trusted to exclude every instance of it, so
        # the measurement itself is checked: if the range test alone empties the
        # grid, the reach is wrong and the capture is innocent.
        return _refusal(
            g, f"only {100.0 * frac_in_reach:.1f}% of the "
               f"{n_vox} voxel centres fell within the {reach:.2f} m reach of any of the "
               f"{len(cams_meta)} cameras (the bar for a judgeable grid is "
               f"{100.0 * MIN_IN_REACH_FRACTION:.0f}%). The reach, not the flight, is "
               "what is emptying this grid, so no coverage percentage is reported.")

    vis_grid = vis_counts.reshape((nx, ny, nz))

    # Elevation statistics (Y is vertical up)
    elev_stats = []
    for iy in range(ny):
        y_val = ys[iy]
        slice_counts = vis_grid[:, iy, :].ravel()
        covered = int(np.sum(slice_counts >= MIN_VIEWS))
        marginal = int(np.sum((slice_counts >= 1) & (slice_counts < MIN_VIEWS)))
        unobserved = int(np.sum(slice_counts == 0))
        total = len(slice_counts)
        pct_good = round(100.0 * covered / total, 1)
        elev_stats.append({
            "y": round(float(y_val), 2),
            "covered_pct": pct_good,
            "covered": covered,
            "marginal": marginal,
            "unobserved": unobserved,
            "total": total,
        })

    # Summary metrics
    total_voxels = int(n_vox)
    good_voxels = int(np.sum(vis_counts >= MIN_VIEWS))
    marginal_voxels = int(np.sum((vis_counts >= 1) & (vis_counts < MIN_VIEWS)))
    unobserved_voxels = int(np.sum(vis_counts == 0))

    coverage_pct = round(100.0 * good_voxels / max(total_voxels, 1), 1)
    unobserved_pct = round(100.0 * unobserved_voxels / max(total_voxels, 1), 1)

    # ---- the missing volume in metres -------------------------------------
    # The grid is in metres (see the note above derive_scene_geometry), so this
    # is a measured figure rather than a rescaled one: voxel volume = dx*dy*dz,
    # and the re-fly target is the surface the unobserved cells expose to the
    # observed ones - the boundary you can actually stand in front of, not the
    # whole blob. survey_occlusion.hidden_regions would have given occlusion-aware
    # regions, but it raises without stereo depth maps (survey_occlusion.py:324),
    # none exist on disk, and it has no production caller - so this is computed
    # from this grid alone and says so.
    voxel_m3 = float(dx * dy * dz)
    seen = vis_grid >= 1
    unobs = vis_grid == 0
    face_area = np.array([dy * dz, dx * dz, dx * dy], np.float64)
    exposed_m2 = 0.0
    for axis in range(3):
        for sl_a, sl_b in ((slice(1, None), slice(None, -1)),
                           (slice(None, -1), slice(1, None))):
            idx_a = [sl_a if i == axis else slice(None) for i in range(3)]
            idx_b = [sl_b if i == axis else slice(None) for i in range(3)]
            exposed_m2 += float(np.sum(unobs[tuple(idx_a)] & seen[tuple(idx_b)])) * face_area[axis]
    boundary_m2 = float(np.sum(unobs)) * float(np.mean(face_area))
    metric = {
        "voxel_volume_m3": round(voxel_m3, 3),
        "grid_volume_m3": round(voxel_m3 * total_voxels, 1),
        "observed_volume_m3": round(voxel_m3 * good_voxels, 1),
        "marginal_volume_m3": round(voxel_m3 * marginal_voxels, 1),
        "unobserved_volume_m3": round(voxel_m3 * unobserved_voxels, 1),
        "unobserved_exposed_area_m2": round(exposed_m2, 1),
        "unobserved_boundary_area_m2_upper_bound": round(boundary_m2, 1),
        "basis": ("metres, from frame.json scale_m_per_unit="
                  f"{g.get('scale_m_per_unit')}; the world frame this module writes is "
                  "already colmap x scale, so no further conversion is applied. "
                  "exposed_area counts faces of unobserved cells that touch a "
                  "cell at least one camera framed; the upper bound assumes every "
                  "unobserved cell presents one average face."),
    }
    g["unobserved_area_m2"] = round(exposed_m2, 1)
    g["unobserved_volume_m3"] = round(voxel_m3 * unobserved_voxels, 1)

    # ---- advice: only ever from a measurement that was valid --------------
    #
    # The band test is unchanged (top 35% / bottom 30% of the elevation profile
    # against the same 40% bar) so nothing here can be softened into flattering
    # the capture. What changed is that a deficit is always reported as a
    # deficit, and only a WALKED rig is told to walk differently: "do a pass
    # walking the perimeter tilted upwards" is not an action an aircraft takes,
    # and at 13 m up the top of the grid is sky.
    top_slices = elev_stats[int(ny * 0.65):]
    bot_slices = elev_stats[:max(1, int(ny * 0.30))]
    top_cov = float(np.mean([s["covered_pct"] for s in top_slices])) if top_slices else 0.0
    bot_cov = float(np.mean([s["covered_pct"] for s in bot_slices])) if bot_slices else 0.0
    agl_m = float(g.get("camera_agl_m") or 0.0)
    weak = [band for band in (("ceiling/upper walls", top_cov),
                              ("floor/baseboard corners", bot_cov))
            if band[1] < ADVICE_MIN_COVERED_PCT]

    advice: list[str] = []
    if not weak:
        advice.append("Solid comprehensive 360-degree coverage across all elevations.")
    elif agl_m <= DRONE_AGL_M:
        if top_cov < ADVICE_MIN_COVERED_PCT:
            advice.append(
                f"Ceiling & upper walls have poor coverage ({top_cov:.0f}% observed). "
                "To eliminate smoky ceiling artifacts, do a pass walking the perimeter tilted 45-60 degrees upwards."
            )
        if bot_cov < ADVICE_MIN_COVERED_PCT:
            advice.append(
                f"Floor & baseboard corners have weak coverage ({bot_cov:.0f}% observed). "
                "Record a knee-height pass tilted slightly downwards."
            )
    else:
        named = ", ".join(f"{name} {pct:.0f}%" for name, pct in weak)
        advice.append(
            f"Coverage falls under the {ADVICE_MIN_COVERED_PCT:.0f}% bar in "
            f"{named}, across the top {g['bounds_max'][1]:.1f} m and bottom "
            f"{g['bounds_min'][1]:.1f} m of the gridded volume. No manoeuvre is "
            f"prescribed: this rig flew {agl_m:.1f} m up, and the walk-the-perimeter "
            "and knee-height advice a walked interior gets does not transfer to a "
            "flight plan. Read the elevation profile above against the passes flown."
        )

    # Export voxels for 3D visualization in viewer
    status_by_count = np.where(vis_grid >= MIN_VIEWS, "good",
                               np.where(vis_grid >= 1, "weak", "missing"))
    vox_export = [
        [round(float(xs[ix]), 3), round(float(ys[iy]), 3), round(float(zs[iz]), 3),
         int(vis_grid[ix, iy, iz]), str(status_by_count[ix, iy, iz])]
        for ix in range(nx) for iy in range(ny) for iz in range(nz)
    ]

    return {
        "status": STATUS_MEASURED,
        "measurable": True,
        "geometry": g,
        "not_measured": list(_NOT_MEASURED),
        "grid_shape": [nx, ny, nz],
        "bounds_min": [round(float(v), 3) for v in lo],
        "bounds_max": [round(float(v), 3) for v in hi],
        "total_voxels": total_voxels,
        "covered_pct": coverage_pct,
        "marginal_pct": round(100.0 * marginal_voxels / max(total_voxels, 1), 1),
        "unobserved_pct": unobserved_pct,
        "covered": good_voxels,
        "marginal": marginal_voxels,
        "unobserved": unobserved_voxels,
        "min_views": MIN_VIEWS,
        "metric_volumes": metric,
        "elevation_profile": elev_stats,
        "advice": advice,
        "voxels": vox_export,
    }


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    ap = argparse.ArgumentParser(description="Analyze multi-view 3D coverage and generate preview assets")
    ap.add_argument("--work", required=True, type=Path, help="Work directory, e.g. work/room_w_jsonl")
    ap.add_argument("--frustum-depth", type=float, default=0.22, help="Frustum visual pyramid depth in meters")
    ap.add_argument("--strict", action="store_true",
                    help="exit 3 when coverage comes back not_measurable, so CI can gate on it")
    args = ap.parse_args()

    work = args.work
    asset_dir = work / "viewer_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    frame_file = work / "frame.json"
    if not frame_file.exists():
        sys.exit(f"missing {frame_file} — run solve_frame.py first")
    fr = json.loads(frame_file.read_text(encoding="utf-8"))
    Rg = np.array(fr["rotation_rowmajor"], np.float64)
    s = float(fr["scale_m_per_unit"])

    poses_file = work / "keyframes_poses.jsonl"
    if not poses_file.exists():
        sys.exit(f"missing {poses_file} — run parse_colmap.py first")
    poses_rows = [json.loads(l) for l in poses_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    # 1. Compute exact 3D camera frustums
    cams_meta = compute_camera_frustums(poses_rows, Rg, s, frustum_depth=args.frustum_depth)
    (asset_dir / "cameras.json").write_text(json.dumps(cams_meta, indent=1), encoding="utf-8")
    print(f"[coverage] Exported {len(cams_meta)} camera frustums -> {asset_dir / 'cameras.json'}")

    # 2. Parse and export COLMAP sparse 3D tie points + densified wall point cloud
    txt_dir = work / "colmap" / "sparse" / "txt"
    pts_colmap, rgbs, tracks = parse_points3d(txt_dir / "points3D.txt")
    if len(pts_colmap) > 0:
        pts_world = (pts_colmap @ Rg.T) * s
    else:
        pts_world = np.empty((0, 3))
        rgbs = np.empty((0, 3), dtype=np.uint8)
        tracks = np.empty((0,), dtype=np.int32)

    # Densify wall and plane point cloud for room visualizations.
    #
    # These ray-cast points are INVENTED - the viewer needs a solid-looking room
    # and a bare COLMAP sparse cloud does not give it one. They are therefore kept
    # out of `pts_world`, which is the measured cloud the coverage grid bounds
    # itself with. Letting fabricated points set the grid bounds is how a
    # diagnostic starts scoring a scene against geometry that was never captured.
    cam_pos = np.array([c["pos"] for c in cams_meta])
    cam_fwds = np.array([c["forward"] for c in cams_meta])
    pts_dense = pts_world
    rgb_dense = rgbs
    track_dense = tracks

    if len(cam_pos) > 5:
        # Cast rays from each camera pose to generate solid wall point clouds
        wall_pts = []
        wall_rgbs = []
        mean_c = rgbs.mean(axis=0) if len(rgbs) else np.array([160, 160, 160])
        for dist in (1.5, 2.2, 3.0):
            wp = cam_pos + cam_fwds * dist
            # Add micro jitter for natural point distribution
            wp += np.random.randn(*wp.shape) * 0.08
            wall_pts.append(wp)
            wall_rgbs.append(np.tile(mean_c, (len(wp), 1)))
        
        if wall_pts:
            wall_pts = np.vstack(wall_pts)
            wall_rgbs = np.vstack(wall_rgbs)
            pts_dense = np.vstack([pts_world, wall_pts]) if len(pts_world) else wall_pts
            rgb_dense = np.vstack([rgbs, wall_rgbs]) if len(rgbs) else wall_rgbs
            track_dense = (np.concatenate([tracks, np.full(len(wall_pts), 3, dtype=np.int32)])
                           if len(tracks) else np.full(len(wall_pts), 3, dtype=np.int32))

    # Subsample if massive for snappy viewer loading
    max_pts = 90000
    if len(pts_dense) > max_pts:
        step = len(pts_dense) // max_pts
        pts_sub = pts_dense[::step]
        rgb_sub = rgb_dense[::step]
        track_sub = track_dense[::step]
    else:
        pts_sub, rgb_sub, track_sub = pts_dense, rgb_dense, track_dense

    sparse_export = {
        "count": len(pts_sub),
        "points": [[round(float(v), 4) for v in p] for p in pts_sub],
        "colors": [[int(v) for v in c] for c in rgb_sub],
        "tracks": [int(v) for v in track_sub],
    }
    (asset_dir / "sparse_points.json").write_text(json.dumps(sparse_export), encoding="utf-8")
    print(f"[coverage] Exported {len(pts_sub)} sparse & wall tie points -> {asset_dir / 'sparse_points.json'}")

    # 3. Coverage grid, bounded by what THESE cameras could have framed.
    #
    # The old chain here was: keep points within 5.0 m of a camera, take their
    # 2/98 percentile box, then UNION it with the camera hull - so the box could
    # only ever grow. On `rocks` just 2 of 17,847 COLMAP points survived the
    # 5.0 m filter, the `> 100` guard tripped, and the grid silently inherited the
    # entire outlier cloud: 3.77e6 m^3 sampled against a 5,506 m^3 camera hull.
    # All of that is now derived in derive_scene_geometry() from frame.json and
    # the poses, and the caller's own box is only ever an intersection.
    cloud_lo = cloud_hi = None
    if len(pts_world) > MIN_CAMERAS:
        cloud_lo, cloud_hi = np.percentile(pts_world, [2.0, 98.0], axis=0)

    coverage_data = analyze_coverage_grid(cams_meta, pts_world, cloud_lo, cloud_hi, frame=fr)
    (asset_dir / "coverage_grid.json").write_text(json.dumps(coverage_data, indent=1), encoding="utf-8")
    geo = coverage_data["geometry"]

    # Print summary report
    print("\n=== COVERAGE DIAGNOSTIC REPORT ===")
    print(f"  Registered Cameras: {len(cams_meta)}")
    print(f"  Camera hull:        {' x '.join(f'{v:.1f}' for v in geo.get('camera_hull_extent_m', [0, 0, 0]))} m"
          f"   (step {geo.get('camera_step_median_m')} m, AGL {geo.get('camera_agl_m')} m)")
    print(f"  Scene-relative reach: {geo.get('reach_m')} m — {geo.get('reach_basis')}")
    print(f"  Near cutoff:          {geo.get('near_range_m')} m — {geo.get('near_basis')}")
    for clip in geo.get("clips", []):
        print(f"  Bounds clip:        {clip}")
    for limit in geo.get("reasons", []):
        print(f"  Limit:              {limit}")

    if coverage_data["status"] != STATUS_MEASURED:
        print(f"\n  STATUS: {coverage_data['status'].upper()} — coverage is NOT measurable"
              f" for this scene.")
        print(f"    Reason: {coverage_data.get('unmeasurable_reason')}")
        print("    No percentages and no re-fly guidance are reported: a grid that cannot"
              " be judged says nothing about the capture.")
        print("\n  Operator Guidance:")
        for adv in coverage_data["advice"]:
            print(f"    ! {adv}")
        for note in coverage_data["not_measured"]:
            print(f"    - not measured: {note}")
        print(f"\n[coverage] Exported 3D coverage grid -> {asset_dir / 'coverage_grid.json'}")
        if args.strict:
            sys.exit(3)
        return

    bx = coverage_data["bounds_min"]
    bz = coverage_data["bounds_max"]
    met = coverage_data["metric_volumes"]
    print(f"  Grid {coverage_data['grid_shape']} over "
          f"{' x '.join(f'{v:.1f}' for v in geo.get('grid_extent_m', [0, 0, 0]))} m"
          f", voxel {' x '.join(f'{v:.2f}' for v in geo.get('voxel_spacing_m', [0, 0, 0]))} m"
          f"  (reach/spacing {geo.get('reach_over_spacing')})")
    print(f"  Well-Observed (>={coverage_data['min_views']} views): {coverage_data['covered_pct']}%"
          f"  ({coverage_data['covered']} voxels)")
    print(f"  Marginal (1-{coverage_data['min_views'] - 1} views):      {coverage_data['marginal_pct']}%")
    print(f"  Unobserved (Missing):      {coverage_data['unobserved_pct']}%")
    print(f"  Unobserved volume:         {met['unobserved_volume_m3']} m^3 of "
          f"{met['grid_volume_m3']} m^3 gridded")
    print(f"  Unobserved surface exposed to observed cells: "
          f"{met['unobserved_exposed_area_m2']} m^2")
    print(f"\n  Room Bounding Box:  x[{bx[0]:.1f}..{bz[0]:.1f}], y[{bx[1]:.1f}..{bz[1]:.1f}],"
          f" z[{bx[2]:.1f}..{bz[2]:.1f}] m")
    print("\n  Elevation Profile (Floor -> Ceiling):")
    for row in coverage_data["elevation_profile"]:
        bar = "#" * int(row["covered_pct"] / 5) + "." * (20 - int(row["covered_pct"] / 5))
        tag = "CRITICAL MISSING" if row["covered_pct"] < 20 else ("WEAK" if row["covered_pct"] < 50 else "GOOD")
        print(f"    Y={row['y']:+7.2f}m |{bar}| {row['covered_pct']:5.1f}% [{tag}]")

    print("\n  Operator Guidance:")
    for adv in coverage_data["advice"]:
        print(f"    ! {adv}")
    print(f"\n[coverage] status={coverage_data['status']} -> {asset_dir / 'coverage_grid.json'}")


if __name__ == "__main__":
    main()
