"""Vertex-color the collision heightfield using the trained splat's own colors.

For each heightfield cell, opacity/volume-weighted mean RGB of gaussians near
that cell (from viewer_assets/scene.ply, already reoriented+scaled). Empty
cells diffuse from neighbors. Output: viewer_assets/ground_colors.rgb
(nx*nz*3 uint8, row-major z,y,x — same layout as heights.f32).

The walk-mode viewer builds a solid ground mesh from heights.f32 + this file,
so ground-level views see continuous terrain instead of sky through splat holes.

Usage:
  python export_ground_colors.py --work work/rocks
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402

SH2RGB = 0.28209479177387814


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--radius", type=float, default=1.6, help="gather radius in cells")
    args = ap.parse_args()

    asset = args.work / "viewer_assets"
    col = rb.read_json(asset / "collision.json", None)
    if not isinstance(col, dict) or not {"nx", "nz", "cell", "origin_xz"} <= col.keys():
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{asset / 'collision.json'} is missing, unreadable or has no grid "
            "description - the export step writes it.", returncode=3)
    nx, nz = int(col["nx"]), int(col["nz"])
    ox, oz = col["origin_xz"]
    cell = float(col["cell"])
    if nx <= 0 or nz <= 0 or not rb.finite(cell, ox, oz) or cell <= 0:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{asset / 'collision.json'} describes a {nx}x{nz} grid with cell "
            f"{cell!r} and origin {ox!r},{oz!r} - not a usable heightfield. "
            "Re-run the export step.", returncode=3)
    H = rb.load_array(asset / "heights.f32", np.float32, (nz, nx),
                      label="heights.f32 (export_viewer_assets)")

    ply = asset / "scene.ply"
    rb.require_file(ply, "viewer_assets/scene.ply (written by the export step)")
    try:
        v = PlyData.read(str(ply))["vertex"].data
    except Exception as e:
        raise rb.StepError(rb.EMPTY_INPUT,
                           f"{ply.name} could not be parsed: {type(e).__name__}: {e}",
                           returncode=3)
    need = ["x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]
    missing = [p for p in need if p not in v.dtype.names]
    if missing:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{ply.name} has no {missing} - it is not a trained splat the export "
            "step wrote (a culled or hand-edited PLY?).", returncode=3)
    x = np.asarray(v["x"], np.float64)
    y = np.asarray(v["y"], np.float64)
    z = np.asarray(v["z"], np.float64)
    op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float64)))
    rgb = (np.stack([np.asarray(v[f"f_dc_{c}"], np.float64) for c in range(3)], 1)
           * SH2RGB + 0.5).clip(0, 1)
    w = op  # opacity-only: volume weighting lets a few huge sky-adjacent blobs dominate
    solid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(w) \
        & np.isfinite(rgb).all(1)
    if not solid.any():
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"every one of {len(x)} gaussians in {ply.name} has a non-finite "
            "position, colour or opacity - the train step needs to be re-run.",
            returncode=3)
    if (~solid).sum():
        rb.warn(f"{int((~solid).sum())}/{len(x)} gaussians have a non-finite "
                "position or colour and are left uncoloured")
        x, y, z, w, rgb = (a[solid] for a in (x, y, z, w, rgb))

    # keep ground-ish gaussians only (within 2.5 m of the heightfield surface)
    gx = np.clip((x - ox) / cell, 0, nx - 1.001).astype(np.int32)
    gz = np.clip((z - oz) / cell, 0, nz - 1.001).astype(np.int32)
    near_v = np.abs(y - H[gz, gx]) < 2.5

    acc = np.zeros((nz, nx, 3))
    wsum = np.zeros((nz, nx))
    m = near_v & np.isfinite(w) & (w > 0)
    np.add.at(acc, (gz[m], gx[m]), rgb[m] * w[m, None])
    np.add.at(wsum, (gz[m], gx[m]), w[m])

    R = int(np.ceil(args.radius))
    dy, dx = np.mgrid[-R:R + 1, -R:R + 1]
    kern = np.exp(-(dy ** 2 + dx ** 2) / (2 * (args.radius) ** 2))
    acc2 = np.zeros_like(acc)
    w2 = np.zeros((nz, nx))
    for ddy in range(-R, R + 1):
        for ddx in range(-R, R + 1):
            k = kern[ddy + R, ddx + R]
            if k < 0.02:
                continue
            ys = slice(max(ddy, 0), nz + min(ddy, 0))
            yd = slice(max(-ddy, 0), nz + min(-ddy, 0))
            xs = slice(max(ddx, 0), nx + min(ddx, 0))
            xd = slice(max(-ddx, 0), nx + min(-ddx, 0))
            acc2[yd, xd] += acc[ys, xs] * k
            w2[yd, xd] += wsum[ys, xs] * k

    mean_rgb = (acc.sum((0, 1)) / max(wsum.sum(), 1e-9))
    filled = w2 > 0
    out = np.ones((nz, nx, 3)) * mean_rgb
    out[filled] = acc2[filled] / w2[filled, None]

    # smooth pass to kill single-cell outliers
    out_p = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
    out = (out_p[:-2, 1:-1] + out_p[2:, 1:-1] + out_p[1:-1, :-2] + out_p[1:-1, 2:]
           + 4 * out_p[1:-1, 1:-1]) / 8.0

    arr = (out.clip(0, 1) * 255).astype(np.uint8)
    arr.tofile(asset / "ground_colors.rgb")
    print(f"[ground-colors] {nx}x{nz}, {filled.mean() * 100:.0f}% cells had gaussians, "
          f"mean rgb {mean_rgb.round(3)} -> {asset / 'ground_colors.rgb'}")


if __name__ == "__main__":
    rb.configure_streams()
    try:
        main()
    except rb.StepError as e:
        print(f"\n[ground-colors] {e}", file=sys.stderr, flush=True)
        sys.exit(e.returncode)
