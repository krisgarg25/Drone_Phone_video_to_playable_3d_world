"""Synthetic INRIA-format splat scene to test the viewer end-to-end
(spark loading, eval camera math, heightfield collision, autopilot)
without a trained model. Ground plane at y=0 + rock-ish clusters.

  python make_test_scene.py --out work/testscene
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

N_REST = 48  # SH deg 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--span", type=float, default=70.0)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # ---- ground: grid of flat gaussians ----
    g = 90
    xs = np.linspace(-args.span / 2, args.span / 2, g)
    zs = np.linspace(-args.span / 2, args.span / 2, g)
    gx, gz = np.meshgrid(xs, zs)
    gy = 0.15 * np.sin(gx / 9) * np.cos(gz / 7)  # gentle undulation
    n_ground = gx.size
    means = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    colors = np.zeros((n_ground, 3))
    colors[:, 0] = 0.25 + 0.1 * rng.random(n_ground)   # r
    colors[:, 1] = 0.45 + 0.15 * rng.random(n_ground)  # g
    colors[:, 2] = 0.2 + 0.1 * rng.random(n_ground)    # b
    scales = np.full((n_ground, 3), math.log(args.span / g * 1.6))
    quats = np.tile([1, 0, 0, 0], (n_ground, 1)).astype(np.float64)
    opac = np.full(n_ground, 0.9)

    # ---- rocks: random ellipsoid clusters ----
    for cx, cz, h, r in [(-8, -5, 6, 3.5), (6, -9, 4, 2.5), (2, 7, 3, 2.0), (14, 3, 2.2, 1.4)]:
        n = 2600
        u = rng.normal(0, 1, (n, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        rad = r * rng.random(n) ** 0.33
        pts = np.stack([cx + u[:, 0] * rad * 1.2,
                        u[:, 1] * rad * (h / r) * 0.5 + h * 0.35,
                        cz + u[:, 2] * rad * 1.2], 1)
        means = np.concatenate([means, pts])
        gray = 0.42 + 0.16 * rng.random((n, 1)) + np.array([0.06, 0.03, -0.02])
        colors = np.concatenate([colors, np.repeat(gray, n, axis=0) * 0 + gray * np.ones((n, 3))])
        s = np.log(np.full((n, 3), 0.35)) + rng.normal(0, 0.15, (n, 3))
        scales = np.concatenate([scales, s])
        qq = rng.normal(0, 1, (n, 4))
        qq /= np.linalg.norm(qq, axis=1, keepdims=True)
        quats = np.concatenate([quats, qq])
        opac = np.concatenate([opac, np.full(n, 0.95)])

    N = len(means)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")] + \
            [(f"f_rest_{i}", "f4") for i in range(N_REST * 3)] + \
            [("opacity", "f4")] + [(f"scale_{k}", "f4") for k in range(3)] + \
            [(f"rot_{k}", "f4") for k in range(4)]
    arr = np.zeros(N, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = means.T
    # rgb -> SH DC coeff
    f_dc = (colors - 0.5) / 0.28209479177387814
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc.T
    arr["opacity"] = np.log(opac / (1 - opac))
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales.T
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = quats.T
    (out / "scene.ply").write_bytes(b"")  # placeholder replaced below
    from plyfile import PlyData as PD
    PD([PlyElement.describe(arr, "vertex")], text=False).write(str(out / "scene.ply"))
    print(f"[testscene] {N} gaussians -> {out/'scene.ply'}")

    # ---- heightfield: sample the same analytic ground ----
    res = 200
    hs = np.linspace(-args.span / 2, args.span / 2, res)
    hx, hz = np.meshgrid(hs, hs)
    H = 0.15 * np.sin(hx / 9) * np.cos(hz / 7)
    (out / "heights.f32").write_bytes(H.astype(np.float32).tobytes())
    cell = hs[1] - hs[0]
    col = {
        "origin_xz": [float(hs[0]), float(hs[0])],
        "cell": float(cell), "nx": res, "nz": res,
        "rotation_rowmajor": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "max_step": 0.8,
        "spawn": {"x": 0.0, "z": 24.0, "face_xz": [0.0, -5.0]},
    }
    (out / "collision.json").write_text(json.dumps(col, indent=1))

    # ---- eval cameras: 3 poses circling the center rock ----
    cams = []
    for i, (az, el, d) in enumerate([(20, 35, 30), (65, 30, 34), (110, 40, 28)]):
        a, e = math.radians(az), math.radians(el)
        c = np.array([d * math.cos(e) * math.sin(a), d * math.sin(e), d * math.cos(e) * math.cos(a)])
        target = np.array([0, 2.5, 0])
        zax = c - target
        zax /= np.linalg.norm(zax)
        up = np.array([0.0, 1.0, 0.0])
        xax = np.cross(up, zax)
        xax /= np.linalg.norm(xax)
        yax = np.cross(zax, xax)
        Rcw = np.stack([xax, -yax, zax])  # COLMAP basis: x right, y down, z fwd
        t = Rcw @ c
        cams.append({
            "name": f"test_{i}", "t_sec": i,
            "R_rowmajor": [list(map(float, r)) for r in Rcw],
            "t": list(map(float, t)),
            "fx": 420.0, "fy": 420.0, "cx": 320.0, "cy": 180.0,
            "width": 640, "height": 360,
        })
    (out / "poses.json").write_text(json.dumps(cams, indent=1))
    print(f"[testscene] collision.json + heights.f32 + poses.json written")


if __name__ == "__main__":
    main()
