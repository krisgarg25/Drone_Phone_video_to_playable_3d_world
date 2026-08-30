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

Usage:
  python solve_frame.py --work work/rocks           # writes frame.json first
  python export_viewer_assets.py --work work/rocks --ply work/rocks/splat.ply
"""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from build_heightfield import rasterize_ground, save_heightfield
from solve_frame import load_cameras, multiview_support


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--ply", required=True, type=Path)
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--cell-meters", type=float, default=0.0,
                    help="heightfield cell size in m; 0 = derive from point density")
    ap.add_argument("--percentile", type=float, default=20.0)
    ap.add_argument("--max-eval-cams", type=int, default=10)
    ap.add_argument("--prune-opacity", type=float, default=0.15)
    ap.add_argument("--thicken", type=float, default=0.30,
                    help="min vertical thickness as fraction of mean horizontal scale")
    ap.add_argument("--max-step", type=float, default=0.8)
    ap.add_argument("--from-scene", action="store_true",
                    help="the input PLY is an ALREADY-EXPORTED viewer scene "
                         "(e.g. a cloud-culled scene.ply): coordinates, quats and "
                         "scales are left untouched and only the derived assets are "
                         "regenerated — heightfield, coverage, spawn hint, walk "
                         "rectangle, collider box. The multi-view region test still "
                         "needs COLMAP-frame points, so they are inverted back for "
                         "that test only.")
    args = ap.parse_args()

    out = args.work / "viewer_assets"
    out.mkdir(parents=True, exist_ok=True)

    frame_file = args.work / "frame.json"
    if not frame_file.exists():
        raise SystemExit(f"missing {frame_file} — run solve_frame.py first")
    fr = json.loads(frame_file.read_text(encoding="utf-8"))
    Rg = np.array(fr["rotation_rowmajor"], np.float64)
    s = float(fr["scale_m_per_unit"])
    print(f"[export] frame: up={np.round(fr['up_in_colmap'], 3).tolist()} "
          f"(source: {fr['gravity_info']['used']}), "
          f"scale {s:.3f} m/unit from {fr['scale_source']}")

    ply = PlyData.read(str(args.ply))
    arr = ply["vertex"].data.copy()
    names = set(arr.dtype.names)

    if args.prune_opacity > 0 and "opacity" in names:
        # INRIA PLY stores logits; compare in sigmoid space, keep storage as logits
        op_logit = np.asarray(arr["opacity"], np.float64)
        op = 1.0 / (1.0 + np.exp(-op_logit))
        keep = op >= args.prune_opacity
        print(f"[export] pruning low-opacity gaussians: kept {keep.sum()}/{len(keep)}")
        arr = arr[keep]

    P = np.stack([np.asarray(arr["x"], np.float64),
                  np.asarray(arr["y"], np.float64),
                  np.asarray(arr["z"], np.float64)], axis=1)

    # ---- which gaussians are allowed to define geometry ----
    # Computed in the COLMAP frame because that is where the camera matrices
    # live. Everything outside `region` still gets exported and drawn — the
    # distant ridgelines are most of why the scene reads as a real place — it
    # just cannot vote on bounds, scale, the heightfield or the collider box.
    _, _, _, _, cams = load_cameras(args.work)
    if args.from_scene:
        P_test = (P / s) @ Rg          # world -> COLMAP, exact inverse of below
    else:
        P_test = P
    region, n_views, _ = multiview_support(
        P_test, cams, int(fr["region_min_views"]), float(fr["region_max_range_units"]))
    del P_test
    print(f"[export] geometry region: {region.sum()}/{len(P)} gaussians "
          f"({100 * region.mean():.1f}%), rest is backdrop")

    # ---- gravity + metric scale, both from frame.json ----
    if not args.from_scene:
        P = (P @ Rg.T) * s
    lo, hi = P[region].min(axis=0), P[region].max(axis=0)
    print(f"[export] region extent x[{lo[0]:.1f}..{hi[0]:.1f}] "
          f"y[{lo[1]:.1f}..{hi[1]:.1f}] z[{lo[2]:.1f}..{hi[2]:.1f}] m")
    a_lo, a_hi = P.min(axis=0), P.max(axis=0)
    print(f"[export] full extent   x[{a_lo[0]:.0f}..{a_hi[0]:.0f}] "
          f"y[{a_lo[1]:.0f}..{a_hi[1]:.0f}] z[{a_lo[2]:.0f}..{a_hi[2]:.0f}] m")

    if not args.from_scene:
        arr["x"] = P[:, 0].astype(np.float32)
        arr["y"] = P[:, 1].astype(np.float32)
        arr["z"] = P[:, 2].astype(np.float32)

    if not args.from_scene and {"rot_0", "rot_1", "rot_2", "rot_3"} <= names:
        q = np.stack([np.asarray(arr[f"rot_{k}"], np.float64) for k in range(4)], axis=1)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
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
            s_lin = np.exp(S)
            if args.thicken > 0:
                vert = np.abs(Rn[:, 1, :])  # [N,3] world-Y component of each local axis
                jmax = vert.argmax(1)
                other = np.where(np.arange(3)[None, :] == jmax[:, None], np.nan, s_lin)
                ref = np.nanmean(other, axis=1)
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
            print(f"[export] gaussian scales x{s:.4f} to match the rescale; "
                  f"median size {np.median(s_lin):.4f} m")

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

    scene_ply = out / "scene.ply"
    if args.from_scene:
        print(f"[export] from-scene: leaving {scene_ply.name} untouched")
    else:
        # PlyData.read mmaps its source; on Windows writing back to the same
        # path is Errno 22. In from_scene mode we do not rewrite at all.
        PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(scene_ply))
    print(f"[export] scene.ply: {len(arr)} gaussians, up=+Y, meters")

    # ---- ground heightfield, region only ----
    # Feeding the whole cloud here is what produced a 110 x 64 m grid of which
    # only 4% of cells had real support: the backdrop set the footprint and the
    # walkable core became a handful of pixels in the corner.
    H, hlo, cell, cover = rasterize_ground(P[region], args.res, args.percentile,
                                           cell=(args.cell_meters or None))
    save_heightfield(H, hlo, cell, np.eye(3), out, args.max_step, cover)

    def pick_spawn(pxz: np.ndarray, py: np.ndarray) -> tuple[float, float]:
        """Density centroid of near-ground gaussians, snapped to a flat cell.

        Restricted to cells with real gaussian support: the flattest patch of a
        diffused-hole region is perfectly flat and perfectly imaginary.
        """
        nz, nx = H.shape
        hlo_x, hlo_z = float(hlo[0]), float(hlo[2])
        gz = np.clip((pxz[:, 1] - hlo_z) / cell, 0, nz - 1.001).astype(int)
        gx = np.clip((pxz[:, 0] - hlo_x) / cell, 0, nx - 1.001).astype(int)
        near = np.abs(py - H[gz, gx]) < 2.5
        cx, cz = float(np.median(pxz[near, 0])), float(np.median(pxz[near, 1]))
        # snap to the flattest supported 5x5 neighbourhood near the centroid
        cj, ci = int((cx - hlo_x) / cell), int((cz - hlo_z) / cell)
        best, bij = None, (ci, cj)
        for i in range(max(2, ci - 12), min(nz - 2, ci + 13)):
            for j in range(max(2, cj - 12), min(nx - 2, cj + 13)):
                if not cover[i - 2:i + 3, j - 2:j + 3].all():
                    continue
                slope = float(np.ptp(H[i - 2:i + 3, j - 2:j + 3]))
                if best is None or slope < best:
                    best, bij = slope, (i, j)
        i, j = bij
        # cell centre, not corner: every rasterizer here indexes with floor(), so
        # cell j spans [hlo_x + j*cell, hlo_x + (j+1)*cell). Returning the corner
        # puts the spawn half a cell (0.32 m) diagonally outside the flat 5x5
        # neighbourhood that was just chosen for being flat.
        return hlo_x + (j + 0.5) * cell, hlo_z + (i + 0.5) * cell

    # ---- eval cameras (same reorientation + scale) ----
    poses_file = args.work / "keyframes_poses.jsonl"
    pairs = []
    spawn = None
    if poses_file.exists():
        rows = [json.loads(l) for l in poses_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        step = max(1, len(rows) // args.max_eval_cams)
        chosen = rows[::step][:args.max_eval_cams]
        eval_cams = []
        for row in chosen:
            c = row["camera"]
            Rw = np.array(c["R_rowmajor"], np.float64) @ Rg.T  # world->cam, reoriented
            t = np.array(c["t"], np.float64) * s               # scale about origin
            eval_cams.append({
                "name": row["file"], "t_sec": row.get("t_sec"),
                "R_rowmajor": [list(map(float, r)) for r in Rw],
                "t": list(map(float, t)),
                **{k: float(c[k]) for k in ("fx", "fy", "cx", "cy")},
            })
        from PIL import Image
        for cam in eval_cams:
            with Image.open(args.work / "frames_full" / cam["name"]) as im:
                cam["width"], cam["height"] = im.size
        (out / "poses.json").write_text(json.dumps(eval_cams, indent=1), encoding="utf-8")
        pairs = [{"render_file": f"eval_{i:02d}.png", "real_file": cam["name"],
                  "t_sec": cam["t_sec"]} for i, cam in enumerate(eval_cams)]
        (args.work / "eval_pairs.json").write_text(json.dumps(pairs, indent=1), encoding="utf-8")
        print(f"[export] {len(eval_cams)} eval cameras -> poses.json; eval_pairs.json")

        # Export ALL camera frustums, sparse tie-points, and coverage grid for 3D viewer
        try:
            from check_coverage import compute_camera_frustums, parse_points3d, analyze_coverage_grid
            all_cams = compute_camera_frustums(rows, Rg, s, frustum_depth=0.22)
            (out / "cameras.json").write_text(json.dumps(all_cams, indent=1), encoding="utf-8")
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
                (out / "sparse_points.json").write_text(json.dumps(sparse_exp), encoding="utf-8")
                print(f"[export] {len(pts_sub)} sparse points -> sparse_points.json")

                cam_p = np.array([c["pos"] for c in all_cams])
                near_m = np.min(np.linalg.norm(pts_w[:, None, :] - cam_p[None, ::10, :], axis=2), axis=1) < 5.0
                pts_rm = pts_w[near_m] if np.sum(near_m) > 100 else pts_w
                b_lo = np.minimum(np.percentile(pts_rm, 2, axis=0), cam_p.min(axis=0) - 0.5)
                b_hi = np.maximum(np.percentile(pts_rm, 98, axis=0), cam_p.max(axis=0) + 0.5)
                cov_data = analyze_coverage_grid(all_cams, pts_w, b_lo, b_hi)
                (out / "coverage_grid.json").write_text(json.dumps(cov_data, indent=1), encoding="utf-8")
                print(f"[export] coverage grid -> coverage_grid.json ({cov_data['covered_pct']}% covered)")
        except Exception as e:
            print(f"[export] WARNING: could not generate coverage assets: {e}")

        # spawn: in the dense ground-data zone (flat cell), facing region center
        Preg = P[region]
        sx, sz = pick_spawn(Preg[:, [0, 2]], Preg[:, 1])
        center = Preg[:, [0, 2]].mean(axis=0)
        spawn = {"x": float(sx), "z": float(sz),
                 "face_xz": [float(center[0]), float(center[1])]}

        # walk loop: rectangle around the supported ground zone (5-95 pct),
        # inset 2 m, so the autopilot stays inside reconstructed terrain
        pxz = Preg[:, [0, 2]]
        gz2 = np.clip((pxz[:, 1] - float(hlo[2])) / cell, 0, H.shape[0] - 1.001).astype(int)
        gx2 = np.clip((pxz[:, 0] - float(hlo[0])) / cell, 0, H.shape[1] - 1.001).astype(int)
        near2 = (np.abs(Preg[:, 1] - H[gz2, gx2]) < 2.5) & (cover[gz2, gx2] > 0)
        x5, x95 = np.percentile(pxz[near2, 0], [5, 95])
        z5, z95 = np.percentile(pxz[near2, 1], [5, 95])
        ins = 2.0
        corners = [(x5 + ins, z5 + ins), (x95 - ins, z5 + ins),
                   (x95 - ins, z95 - ins), (x5 + ins, z95 - ins)]
        k0 = min(range(4), key=lambda k: (corners[k][0] - sx) ** 2 + (corners[k][1] - sz) ** 2)
        walk_path = [list(map(float, corners[(k0 + d) % 4])) for d in range(4)]
        perim = sum(np.hypot(corners[a][0] - corners[(a + 1) % 4][0],
                             corners[a][1] - corners[(a + 1) % 4][1]) for a in range(4))
        print(f"[export] walk loop x[{x5 + ins:.1f}..{x95 - ins:.1f}] "
              f"z[{z5 + ins:.1f}..{z95 - ins:.1f}], perimeter {perim:.0f} m")
    else:
        print("[export] WARNING: no keyframes_poses.jsonl yet — run parse_colmap first")
        spawn = {"x": float(lo[0] + (hi[0] - lo[0]) * 0.3),
                 "z": float(lo[2] + (hi[2] - lo[2]) * 0.3),
                 "face_xz": [float(P[:, 0].mean()), float(P[:, 2].mean())]}
        walk_path = None

    # stash spawn + walk path + the derived collider box into collision.json
    col = json.loads((out / "collision.json").read_text(encoding="utf-8"))
    col["spawn"] = spawn
    if walk_path:
        col["walk_path"] = walk_path

    # Collider box, published so the voxeliser stops needing a hand-tuned
    # constant. Tight around the walkable region, in this file's frame
    # (up=+Y, metres). Deliberately NOT padded far above the terrain: the grid
    # ceiling is where --voxel-floor-fill dumps its slab, and the old hardcoded
    # box is why the character spawned on an invisible plateau. The generous pad
    # is on the BOTTOM, where a solid floor under the terrain is harmless.
    pad = 1.0
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
    (out / "collision.json").write_text(json.dumps(col, indent=2), encoding="utf-8")
    print(f"[export] spawn at x={spawn['x']:.1f} z={spawn['z']:.1f} facing scene center")
    cb = col["collider_box"]
    print(f"[export] collider box -B "
          f"{cb['min'][0]:.2f},{cb['min'][1]:.2f},{cb['min'][2]:.2f},"
          f"{cb['max'][0]:.2f},{cb['max'][1]:.2f},{cb['max'][2]:.2f}")


if __name__ == "__main__":
    main()
