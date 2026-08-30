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

Scale has no ground truth without GPS, so this reports the two defensible
anchors side by side (orbit path length vs. flight duration, and camera height
above the local ground) and lets the caller pick.

  python solve_frame.py --work work/rocks
"""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


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
    close enough that its ground sampling is fine. Two different failure modes
    get rejected by the two criteria:

      * the sparse training halo — floaters seen by one or two cameras;
      * the distant mountains and forest — genuinely reconstructed and seen by
        every frame, but a hundred flight-heights away, so a metre of terrain is
        a fraction of a pixel. Lovely as a backdrop, useless as a floor.

    Both stay in the render. They just stop voting on bounds, scale, gravity,
    the heightfield and the collider box, which is what wrecked earlier runs.
    """
    n_views = np.zeros(len(P), np.int32)
    near = np.full(len(P), np.inf)
    for c in cams:
        R = np.array(c["R_rowmajor"], np.float64)
        t = np.array(c["t"], np.float64)
        pc = P @ R.T + t
        z = pc[:, 2]
        front = z > 1e-6
        W, H = 2.0 * c["cx"], 2.0 * c["cy"]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = c["fx"] * pc[:, 0] / z + c["cx"]
            v = c["fy"] * pc[:, 1] / z + c["cy"]
        vis = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        n_views += vis
        d = np.linalg.norm(pc, axis=1)
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
    if near.sum() < 500:
        return Rg, {"refined": False, "reason": "too few near-path points"}
    G = Pr[near]
    G = G[G[:, 1] < np.percentile(G[:, 1], 40)]  # ground, not canopy/boulder
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--ply", type=Path, default=None)
    ap.add_argument("--drone-speed", type=float, default=5.0,
                    help="assumed drone ground speed in m/s (scale anchor A)")
    ap.add_argument("--drone-agl", type=float, default=25.0,
                    help="assumed drone height above ground in m (scale anchor B)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-views", type=int, default=4,
                    help="cameras that must see a gaussian for it to define geometry")
    ap.add_argument("--max-range-mult", type=float, default=4.0,
                    help="max view distance for geometry, in camera-AGL multiples")
    args = ap.parse_args()

    C, UP, FWD, T, cams = load_cameras(args.work)
    print(f"[frame] {len(C)} cameras")
    up, info = gravity_from_cameras(UP, C)
    print(f"[frame] camera-up coherence {info['camera_up_coherence']} "
          f"(1.0 = all frames agree)")
    print(f"[frame] orbit planarity {info['orbit_planarity']}, "
          f"plane vs camera-up {info['orbit_plane_vs_camera_up_deg']} deg")
    print(f"[frame] gravity source: {info['used']}")
    print(f"[frame] world up in COLMAP frame = {np.round(up, 4).tolist()}")

    ply = args.ply or (args.work / "splat.ply")
    v = PlyData.read(str(ply))["vertex"]
    P = np.stack([np.asarray(v["x"], np.float64), np.asarray(v["y"], np.float64),
                  np.asarray(v["z"], np.float64)], axis=1)
    if "opacity" in v.data.dtype.names:
        op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float64)))
        P_filt = P[op >= 0.15]
        if len(P_filt) >= 500:
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
    if mask.sum() < 5000:
        raise SystemExit("[frame] FATAL: multi-view support found almost nothing; "
                         "check camera intrinsics in keyframes_poses.jsonl")
    Rlo, Rhi = Pr[mask].min(axis=0), Pr[mask].max(axis=0)
    print(f"[frame] walkable region (units) x[{Rlo[0]:.1f}..{Rhi[0]:.1f}] "
          f"y[{Rlo[1]:.1f}..{Rhi[1]:.1f}] z[{Rlo[2]:.1f}..{Rhi[2]:.1f}]")

    # ---- metric scale ----
    # Only ONE real-world quantity is known without GPS: the clip duration. So
    # the flight-speed anchor is the primary, and the implied drone altitude is
    # the cross-check. (The old pipeline instead declared the scene 110 m wide,
    # measured across bounds the floater halo had inflated 4x.)
    scale_a = (args.drone_speed * dur) / path_u if path_u > 0 and np.isfinite(dur) else np.nan
    scale_b = args.drone_agl / agl_u if agl_u > 0 else np.nan
    print(f"[frame] anchor A, primary  (speed {args.drone_speed} m/s x {dur:.1f} s): "
          f"{scale_a:.3f} m/unit")
    print(f"[frame] anchor B, crosscheck (altitude {args.drone_agl} m AGL): "
          f"{scale_b:.3f} m/unit")
    scale = float(scale_a) if np.isfinite(scale_a) else float(scale_b)
    print(f"[frame] -> implied drone altitude {agl_u * scale:.1f} m AGL, "
          f"speed {path_u * scale / dur:.1f} m/s")
    if not 5.0 <= agl_u * scale <= 80.0:
        print(f"[frame] WARNING: implied altitude {agl_u * scale:.1f} m is not "
              f"plausible for this framing; revisit --drone-speed")

    Rlo_m, Rhi_m = Rlo * scale, Rhi * scale
    relief = float(np.percentile(Pr[mask, 1], 99.5)
                   - np.percentile(Pr[mask, 1], 1)) * scale
    print(f"[frame] walkable region {Rhi_m[0] - Rlo_m[0]:.0f} x "
          f"{Rhi_m[2] - Rlo_m[2]:.0f} m, relief {relief:.1f} m")
    allm = Pr * scale
    print(f"[frame] full scene incl. backdrop x[{allm[:, 0].min():.0f}.."
          f"{allm[:, 0].max():.0f}] z[{allm[:, 2].min():.0f}.."
          f"{allm[:, 2].max():.0f}] m (kept for looks, excluded from geometry)")

    doc = {"rotation_rowmajor": [list(map(float, r)) for r in Rg],
           "up_in_colmap": up.tolist(), "gravity_info": info,
           "ground_refine": rinfo,
           "scale_m_per_unit": scale,
           "scale_anchor_speed": scale_a, "scale_anchor_agl": scale_b,
           "scale_source": "flight speed x clip duration",
           "orbit_path_units": path_u, "duration_s": dur,
           "camera_agl_units": agl_u, "camera_agl_m": agl_u * scale,
           "region_min_views": args.min_views,
           "region_max_range_units": agl_u * args.max_range_mult,
           "region_box_m": {"min": [float(x) for x in Rlo_m],
                            "max": [float(x) for x in Rhi_m]},
           "region_relief_m": relief}
    outp = args.out or (args.work / "frame.json")
    outp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"[frame] wrote {outp}")


if __name__ == "__main__":
    main()
