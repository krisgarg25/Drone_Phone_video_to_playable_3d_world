"""Solve the world frame (gravity + metric scale) from COLMAP cameras.

Why this exists
---------------
The old pipeline took gravity from a RANSAC plane fit over the splat means and
then forced the normal to point along +Y with

    if best_n @ [0,1,0] < 0: best_n = -best_n

A plane normal has no inherent sign, so that line is a coin flip. On this clip
it came up tails: the true up direction is ~-Y in COLMAP's frame, so the whole
scene was exported UPSIDE DOWN. Everything downstream then did the wrong thing
in a self-consistent way — `rasterize_ground` takes a LOW percentile of Y per
cell to dig under floaters, which in a flipped scene digs into the sky instead;
the voxel collider's top surface became the underside of the terrain shell; and
the character was placed standing on it, hovering over the scene.

Drone footage carries a much better gravity reference: the gimbal holds the
horizon level, so each camera's own up axis IS world up. Averaged over 114
keyframes that is a strong estimate, and it is independently corroborated by
the plane the orbit path lies in.

Scale has no ground truth without GPS, so there are three rulers:
  C  (prior path)  — when AR pose priors are available the metric path length
                     over the registered frames gives scale directly, with no
                     speed or height assumption. Always preferred when available.
  A  (speed)       — the clip is a known duration: speed x duration / path_u.
  B  (height)      — the camera is a known height above the ground it filmed.
A suits a drone, whose speed the pilot set; B suits anything held at arm's
length, whose speed nobody knows but whose height is a tape measure.
Without priors, pass exactly one — a 5 m/s anchor on a slow indoor dolly
inflated the scene 15x and the collider voxeliser then died on the grid.

  python solve_frame.py --work work/rocks
  python solve_frame.py --work work/auditorium --height-anchor 1.6
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402

# What "a drone orbit" means when the caller states no anchor at all.
DRONE_SPEED = 5.0     # m/s over the ground
DRONE_HEIGHT = 25.0   # m above it
# A numerical floor, not a quality bar: below this many points a min/max box is one
# straggler's idea of a room. Whether the support is good enough to bound the
# world is judged by how far it spreads (see the region test in main()).
MIN_SUPPORT_POINTS = 60


def load_cameras(work: Path):
    rows = [json.loads(l) for l in
            (work / "keyframes_poses.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    C, UP, FWD, T = [], [], [], []
    for r in rows:
        c = r["camera"]
        R = np.array(c["R_rowmajor"], np.float64)  # world -> cam
        t = np.array(c["t"], np.float64)
        C.append(-R.T @ t)
        UP.append(R.T @ np.array([0.0, -1.0, 0.0]))   # COLMAP cam +Y is down
        FWD.append(R.T @ np.array([0.0, 0.0, 1.0]))   # COLMAP cam +Z is forward
        T.append(r.get("t_sec", np.nan))
    cams = [r["camera"] for r in rows]
    return np.array(C), np.array(UP), np.array(FWD), np.array(T, np.float64), cams


def gravity_from_cameras(UP: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, dict]:
    """World up from the mean camera up axis, cross-checked against the orbit plane."""
    up = UP.mean(axis=0)
    coh = float(np.linalg.norm(up))  # 1.0 == every frame agreed
    up /= max(coh, 1e-12)
    c0 = C.mean(axis=0)
    _, sv, vt = np.linalg.svd(C - c0)
    n = vt[-1]
    if n @ up < 0:
        n = -n
    ang = float(np.degrees(np.arccos(np.clip(n @ up, -1, 1))))
    planar = float(sv[2] / max(sv[0], 1e-12))
    info = {"camera_up_coherence": round(coh, 4),
            "orbit_plane_normal": np.round(n, 4).tolist(),
            "orbit_plane_vs_camera_up_deg": round(ang, 2),
            "orbit_planarity": round(planar, 4),
            "orbit_singular_values": np.round(sv, 3).tolist()}
    # If the orbit really is planar and roughly agrees, average the two — the
    # orbit plane is immune to per-frame gimbal jitter.
    if planar < 0.1 and ang < 25.0:
        up = up + n
        up /= np.linalg.norm(up)
        info["used"] = "mean(camera_up, orbit_plane_normal)"
    else:
        info["used"] = "camera_up"
    return up, info


def rot_to_up(normal: np.ndarray) -> np.ndarray:
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(normal, y)
    s = np.linalg.norm(v)
    c = float(normal @ y)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0]).astype(float)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def multiview_support(P: np.ndarray, cams: list, min_views: int, max_dist: float):
    """Count how many cameras actually saw each point, and its closest range.

    A gaussian is trustworthy for GEOMETRY only if several cameras saw it from
    close enough that its ground sampling is fine.
    """
    Pf = np.ascontiguousarray(P, dtype=np.float32)
    n_views = np.zeros(len(Pf), dtype=np.int32)
    near = np.full(len(Pf), np.inf, dtype=np.float32)
    for c in cams:
        R = np.ascontiguousarray(c["R_rowmajor"], dtype=np.float32)
        t = np.ascontiguousarray(c["t"], dtype=np.float32)
        pc = Pf @ R.T + t
        z = pc[:, 2]
        front = z > 1e-6
        W, H = float(2.0 * c["cx"]), float(2.0 * c["cy"])
        with np.errstate(divide="ignore", invalid="ignore"):
            u = float(c["fx"]) * pc[:, 0] / z + float(c["cx"])
            v = float(c["fy"]) * pc[:, 1] / z + float(c["cy"])
        vis = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        n_views += vis
        # Fast vector Euclidean distance
        d = np.sqrt(pc[:, 0] * pc[:, 0] + pc[:, 1] * pc[:, 1] + pc[:, 2] * pc[:, 2])
        np.minimum(near, np.where(vis, d, np.inf), out=near)
    return (n_views >= min_views) & (near <= max_dist), n_views, near


def refine_with_ground(Rg: np.ndarray, P: np.ndarray, C: np.ndarray,
                       max_tilt_deg: float = 20.0):
    """Least-squares plane through the ground points near the flight path.

    Accepted only if it stays within max_tilt of the camera-derived up, so a
    mountainside or a mis-triangulated cloud cannot hijack gravity again.
    """
    Pr, Cr = P @ Rg.T, C @ Rg.T
    cx, cz = Cr[:, 0].mean(), Cr[:, 2].mean()
    reach = max(np.percentile(np.hypot(Cr[:, 0] - cx, Cr[:, 2] - cz), 90) * 2.0, 1e-6)
    near = np.hypot(Pr[:, 0] - cx, Pr[:, 2] - cz) < reach
    G = Pr[near]
    if len(G):
        G = G[G[:, 1] < np.percentile(G[:, 1], 40)]  # ground, not canopy/boulder
    # How few points a plane can be fitted from - a numerical floor, not a quality
    # bar. A count scaled to the cloud cannot work here: the cloud is whatever this
    # run's VRAM budget allowed, so an absolute floor would make gravity depend on
    # the machine rather than the take (and it did reject valid ones).
    min_fit = max(24, int(0.002 * len(P)))
    if len(G) < min_fit:
        return Rg, {"refined": False, "n_ground_points": int(len(G)),
                    "reason": f"too few near-path ground points ({len(G)} < {min_fit})"}
    c0 = G.mean(axis=0)
    _, _, vt = np.linalg.svd(G - c0, full_matrices=False)
    n = vt[-1]
    if n[1] < 0:
        n = -n
    tilt = float(np.degrees(np.arccos(np.clip(n[1], -1, 1))))
    if tilt > max_tilt_deg:
        return Rg, {"refined": False, "ground_tilt_deg": round(tilt, 2),
                    "reason": f"ground plane {tilt:.1f} deg off camera up, rejected"}
    return rot_to_up(n) @ Rg, {"refined": True, "ground_tilt_deg": round(tilt, 2),
                               "n_ground_points": int(len(G))}


def prior_scale(work: Path, C: np.ndarray, T: np.ndarray) -> float | None:
    """Metric path / COLMAP path from AR pose priors (ruler C).

    Reads work/pose_priors.jsonl and work/keyframes_poses.jsonl, keeps only
    frames that are both registered in COLMAP and have a prior, then computes
    the ratio of the two path lengths.  No coordinate-frame alignment is needed
    because distances are invariant under rotation and translation — the only
    remaining degree of freedom is uniform scale, which is exactly what we want.
    Returns None if priors are absent or too few frames matched.
    """
    priors_file = work / "pose_priors.jsonl"
    kf_file = work / "keyframes_poses.jsonl"
    if not priors_file.exists() or not kf_file.exists():
        return None
    # file -> prior position (meters)
    priors: dict[str, np.ndarray] = {}
    for line in priors_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            priors[row["file"]] = np.array(row["position"], np.float64)
    # registered frames in keyframes_poses.jsonl order (COLMAP image order)
    files: list[str] = []
    for line in kf_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            files.append(json.loads(line)["file"])
    if len(files) != len(C):
        return None  # file and C arrays out of sync — bail rather than guess
    # keep only frames that have both a COLMAP position and a prior
    rows = [(files[i], C[i], T[i]) for i in range(len(files))
            if files[i] in priors and np.isfinite(T[i])]
    if len(rows) < 10:
        return None
    rows.sort(key=lambda r: r[2])          # chronological order for path length
    colmap_pos = np.array([r[1] for r in rows])   # COLMAP units
    prior_pos  = np.array([priors[r[0]] for r in rows])  # meters
    colmap_path = float(np.linalg.norm(np.diff(colmap_pos, axis=0), axis=1).sum())
    prior_path  = float(np.linalg.norm(np.diff(prior_pos,  axis=0), axis=1).sum())
    if colmap_path < 0.05:
        return None
    s = prior_path / colmap_path
    print(f"[frame] ruler C (prior path): {prior_path:.2f} m / "
          f"{colmap_path:.2f} u = {s:.3f} m/unit  [{len(rows)} registered frames]")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--ply", type=Path, default=None)
    ap.add_argument("--speed-anchor", dest="speed_anchor", type=float, default=None,
                    metavar="M_PER_S",
                    help="scale ruler A: how fast the camera moved along the ground, "
                         f"in m/s. Implied: {DRONE_SPEED} (a drone)")
    ap.add_argument("--height-anchor", dest="height_anchor", type=float, default=None,
                    metavar="M",
                    help="scale ruler B: how high the camera sat above the ground it "
                         "filmed, in m. Use this one for anything handheld or on a "
                         f"dolly — you know it to 20 cm, you do not know its speed. "
                         f"Implied: {DRONE_HEIGHT}")
    ap.add_argument("--no-prior-scale", action="store_true",
                    help="skip ruler C (prior path scale) even when pose_priors.jsonl exists")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-views", type=int, default=4,
                    help="cameras that must see a gaussian for it to define geometry")
    ap.add_argument("--max-range-mult", type=float, default=4.0,
                    help="max view distance for geometry, in camera-AGL multiples")
    args = ap.parse_args()

    C, UP, FWD, T, cams = load_cameras(args.work)
    if len(C) == 0:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            "no camera poses to solve a world frame from: "
            f"{(args.work / 'keyframes_poses.jsonl').name} is empty. The colmap and "
            "poses steps must produce a registered camera set first.",
            returncode=3)
    print(f"[frame] {len(C)} cameras")
    up, info = gravity_from_cameras(UP, C)
    print(f"[frame] camera-up coherence {info['camera_up_coherence']} "
          f"(1.0 = all frames agree)")
    print(f"[frame] orbit planarity {info['orbit_planarity']}, "
          f"plane vs camera-up {info['orbit_plane_vs_camera_up_deg']} deg")
    print(f"[frame] gravity source: {info['used']}")
    print(f"[frame] world up in COLMAP frame = {np.round(up, 4).tolist()}")

    ply = args.ply or (args.work / "splat.ply")
    rb.require_file(ply, "splat.ply (written by the train step)")
    v = PlyData.read(str(ply))["vertex"]
    P = np.stack([np.asarray(v["x"], np.float64), np.asarray(v["y"], np.float64),
                  np.asarray(v["z"], np.float64)], axis=1)
    if len(P) == 0:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{ply.name} holds 0 gaussians - the train step produced an empty splat, "
            "so there is no geometry to put a world frame on.",
            returncode=3)
    if "opacity" in v.data.dtype.names:
        op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float64)))
        P_filt = P[op >= 0.15]
        # Relative for the same reason as every other count here: 500 is a large
        # share of a 7.5k cloud and a rounding error on a 30k one, so an absolute
        # floor made the frame depend on free VRAM. Keep the cut only when it does
        # not gut the cloud.
        if len(P_filt) >= max(60, 0.05 * len(P)):
            P = P_filt

    Rg = rot_to_up(up)
    Rg, rinfo = refine_with_ground(Rg, P, C)
    print(f"[frame] ground refine: {rinfo}")

    Cr = C @ Rg.T
    Pr = P @ Rg.T
    cx, cz = Cr[:, 0].mean(), Cr[:, 2].mean()
    reach = np.percentile(np.hypot(Cr[:, 0] - cx, Cr[:, 2] - cz), 90) * 2.0
    near = np.hypot(Pr[:, 0] - cx, Pr[:, 2] - cz) < reach
    ground_u = float(np.percentile(Pr[near, 1], 10)) if near.sum() > 0 else float(np.percentile(Pr[:, 1], 10))
    agl_u = float(Cr[:, 1].mean() - ground_u)
    path_u = float(np.linalg.norm(np.diff(Cr, axis=0), axis=1).sum())
    dur = float(np.nanmax(T) - np.nanmin(T)) if np.isfinite(T).any() else np.nan

    print(f"[frame] cameras sit {agl_u:.2f} units above the near-path ground "
          f"({'ABOVE, correct' if agl_u > 0 else 'BELOW - up is still flipped!'})")
    if agl_u <= 0:
        raise SystemExit("[frame] FATAL: gravity still points the wrong way; refusing "
                         "to write a frame that would put the player under the terrain")
    print(f"[frame] orbit path {path_u:.2f} units over {dur:.2f} s")

    # ---- walkable region from multi-view support ----
    mask, n_views, near_d = multiview_support(
        P, cams, args.min_views, agl_u * args.max_range_mult)
    print(f"[frame] geometry support: {mask.sum()}/{len(P)} gaussians "
          f"({100 * mask.mean():.1f}%) seen by >={args.min_views} cams within "
          f"{agl_u * args.max_range_mult:.1f} u "
          f"(median views {int(np.median(n_views))})")
    # Support judged against THIS cloud and the space it has to bound. An absolute
    # count is the wrong yardstick: a 300-step smoke cloud holds 7.5k gaussians or
    # 30k depending on what else had the GPU, and rejecting one stopped a run that
    # already had a finished splat on disk. So the count is only a numerical floor
    # - what the region actually needs is points spread across the camera path.
    sup = Pr[mask]
    ext_sup = (sup.max(axis=0)[[0, 2]] - sup.min(axis=0)[[0, 2]]
               if len(sup) else np.zeros(2))
    ext_cam = Cr.max(axis=0)[[0, 2]] - Cr.min(axis=0)[[0, 2]]
    cov = 100 * ext_sup / np.maximum(ext_cam, 1e-9)
    if len(sup) < MIN_SUPPORT_POINTS:
        thin = f"only {len(sup)} of {len(P)} gaussians pass the support test"
    elif not bool(np.all(ext_sup >= 0.5 * ext_cam)):
        thin = (f"its {len(sup)} points span {cov[0]:.0f}% x {cov[1]:.0f}% of the "
                "camera path, not enough of it to bound a room")
    else:
        thin = ""
    if thin:
        # The measured stand-in: the flight path and whatever sits under it. It
        # has to have a real vertical span, because region_box_m is the box
        # export_viewer_assets falls back to when there is no support mask, and a
        # zero-thickness slice at camera height would keep nothing.
        rb.warn(f"[frame] too little multi-view support to bound a region ({thin}) "
                "- bounding the region by the camera path and the ground under it "
                "instead; expect a coarse walkable area")
        under = Pr[near & (Pr[:, 1] < np.percentile(Cr[:, 1], 50))]
        region = np.vstack([Cr, under]) if len(under) else Cr
        region_src = "camera path + ground under it"
    else:
        region, region_src = sup, "multi-view support"
    Rlo, Rhi = region.min(axis=0), region.max(axis=0)
    print(f"[frame] walkable region (units, from {region_src}) "
          f"x[{Rlo[0]:.1f}..{Rhi[0]:.1f}] y[{Rlo[1]:.1f}..{Rhi[1]:.1f}] "
          f"z[{Rlo[2]:.1f}..{Rhi[2]:.1f}]")

    # ---- metric scale ----
    # Three rulers. C (prior path) beats A and B whenever AR priors are
    # available — it measures scale directly rather than assuming a speed or
    # height. A suits a drone, whose speed the pilot set; B suits anything held
    # at arm's length, whose speed nobody knows but whose height is a tape measure.
    prior_s: float | None = None
    if not args.no_prior_scale:
        prior_s = prior_scale(args.work, C, T)
    speed = args.speed_anchor if args.speed_anchor is not None else DRONE_SPEED
    height = args.height_anchor if args.height_anchor is not None else DRONE_HEIGHT
    if args.speed_anchor is not None and args.height_anchor is not None:
        raise SystemExit("[frame] FATAL: --speed-anchor and --height-anchor both given. "
                         "They are two independent rulers and disagreeing ones would say "
                         "which is right only by luck — pass the one you actually know.")
    scale_a = (speed * dur) / path_u if path_u > 0 and np.isfinite(dur) else np.nan
    scale_b = height / agl_u if agl_u > 0 else np.nan
    if prior_s is not None:
        print(f"[frame] ruler A, speed     ({speed} m/s x {dur:.1f} s): {scale_a:.3f} m/unit")
        print(f"[frame] ruler B, height    ({height} m above ground):    {scale_b:.3f} m/unit")
        scale, source = prior_s, "AR pose-prior metric path"
    elif args.height_anchor is not None:
        print(f"[frame] ruler A, speed     ({speed} m/s x {dur:.1f} s): {scale_a:.3f} m/unit")
        print(f"[frame] ruler B, height    ({height} m above ground):    {scale_b:.3f} m/unit")
        scale, source = float(scale_b), f"camera height {height:.1f} m above ground"
    else:
        print(f"[frame] ruler A, speed     ({speed} m/s x {dur:.1f} s): {scale_a:.3f} m/unit")
        print(f"[frame] ruler B, height    ({height} m above ground):    {scale_b:.3f} m/unit")
        scale = float(scale_a) if np.isfinite(scale_a) else float(scale_b)
        source = ("flight speed x clip duration" if np.isfinite(scale_a)
                  else f"camera height {height:.1f} m above ground")
    if not np.isfinite(scale) or scale <= 0:
        # Both rulers failed. Ruler A needs a finite clip duration and a
        # reconstructed path; ruler B needs a fitted ground under the camera.
        # Guessing 1.0 would write a NaN-free but wrong number into frame.json,
        # and every metre downstream inherits it - including the collider voxel
        # grid, which is exactly how a wrong scale turns into a crash later.
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"[frame] no metric ruler produced a finite scale "
            f"(ruler A speed x duration = {scale_a:.3g}, ruler B height / AGL = "
            f"{scale_b:.3g}; reconstructed path {path_u:.3g} units, "
            f"camera above fitted ground {agl_u:.3g} units).\n"
            f"  The solve has no measurable ground under the cameras. Pass "
            f"--height-anchor M (a walked phone is ~1.6 m, a drone orbit more) "
            f"or --speed-anchor M_PER_S, or attach an AR pose log.",
            returncode=4)
    print(f"[frame] -> {source} = {scale:.3f} m/unit")
    print(f"[frame] implied camera height {agl_u * scale:.1f} m, "
          f"speed {path_u * scale / dur if dur else float('nan'):.2f} m/s")
    if source.startswith("flight") and not 0.5 <= agl_u * scale <= 80.0:
        print(f"[frame] WARNING: that speed puts the camera {agl_u * scale:.1f} m above "
              f"the ground, which is neither a drone nor a person. Re-anchor with "
              f"--height-anchor.")

    Rlo_m, Rhi_m = Rlo * scale, Rhi * scale
    p_hi = rb.safe_pct(region[:, 1], 99.5, 0.0, label="region relief p99.5")
    p_lo = rb.safe_pct(region[:, 1], 1, 0.0, label="region relief p1")
    relief = float(p_hi - p_lo) * scale
    print(f"[frame] walkable region {Rhi_m[0] - Rlo_m[0]:.0f} x "
          f"{Rhi_m[2] - Rlo_m[2]:.0f} m, relief {relief:.1f} m "
          f"[{region_src}]")
    allm = Pr * scale
    print(f"[frame] full scene incl. backdrop x[{allm[:, 0].min():.0f}.."
          f"{allm[:, 0].max():.0f}] z[{allm[:, 2].min():.0f}.."
          f"{allm[:, 2].max():.0f}] m (kept for looks, excluded from geometry)")

    doc = {"rotation_rowmajor": [list(map(float, r)) for r in Rg],
           "up_in_colmap": up.tolist(), "gravity_info": info,
           "ground_refine": rinfo,
           "scale_m_per_unit": scale,
           "scale_anchor_speed": scale_a, "scale_anchor_agl": scale_b,
           "scale_source": source,
           "orbit_path_units": path_u, "duration_s": dur,
           "camera_agl_units": agl_u, "camera_agl_m": agl_u * scale,
           "region_min_views": args.min_views,
           "region_max_range_units": agl_u * args.max_range_mult,
           # Which points bounded the box, and how many the support test liked, so
           # a reader can tell a measured region from the fallback without the log.
           "region_source": region_src,
           "region_supported_gaussians": int(mask.sum()),
           "region_box_m": {"min": [float(x) for x in Rlo_m],
                            "max": [float(x) for x in Rhi_m]},
           "region_relief_m": relief}
    outp = args.out or (args.work / "frame.json")
    outp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"[frame] wrote {outp}")


if __name__ == "__main__":
    rb.configure_streams()
    try:
        main()
    except rb.StepError as e:
        print(f"\n[frame] {e}", file=sys.stderr, flush=True)
        sys.exit(e.returncode)
