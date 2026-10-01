"""Textured FLAT ground for the demo video: planning (Plan) and placing (Place).

Like ``make_plan_scene.py`` it is synthetic and says so, but it is meant to look like
a real field rather than a test grid:

* 100 x 100 m of ground at y = 0, skinned with real CC0 aerial photo textures
  (Poly Haven, fetched into ``data/textures`` on first run): grass and rock, with
  patches of bare dirt blended in by low-frequency noise so the tiling does not read;
* a worn dirt track along the east side and a paved patio (concrete pavers) where
  furniture looks at home;
* a tree line along the north edge and a few loose trees (vegetation labels);
* a flat collision mesh with a rim wall, so measuring, snapping and walking work;
* the same synthetic GPS fit as the plan test scene, so the sunlight study has a latitude.

  python scripts/make_flat_scene.py --out work/flat_ground
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

import label_semantics
from ground_mesh import build as ground_skin
from make_plan_scene import ORIGIN
from plan_glb import glb_bytes

ROOT = Path(__file__).resolve().parents[1]
TEXTURES = ROOT / "data" / "textures"
# name -> real-world tile size in metres (Poly Haven "dimensions").
TEX = {"aerial_grass_rock": 15.0, "dirt_aerial_03": 25.0, "concrete_pavers_02": 2.0}
TEX_URL = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/4k/{0}/{0}_diff_4k.jpg"

SPAN, CELL = 100.0, 0.5            # ground extent (m) and collision/height grid cell (m)
EDGE_R, OUTER_STEP = 74.0, 0.4      # ragged outer edge radius (m) and its splat spacing; not walkable
STEP, PATIO_STEP = 0.14, 0.06       # splat spacing on the field and on the patio (m)
PATIO = (-22.0, 18.0, 14.0, 10.0)  # centre x, centre z, length x, depth z  (-z is north)
TRACK_X, TRACK_W = 34.0, 3.2        # a north-south dirt track: centre line x (wobbles), width
TREES = [(x, -44.0 + 1.5 * math.sin(x), 2.4 + 0.5 * math.cos(1.7 * x)) for x in np.arange(-44.0, 46.0, 7.5)]
TREES += [(-38.0, 30.0, 3.0), (-31.0, 36.0, 2.4), (41.0, 22.0, 2.8)]   # x, z, crown radius


def texture(name: str, metres_per_px: float) -> np.ndarray:
    """The texture as float RGB, resampled so one pixel covers ``metres_per_px``."""
    path = TEXTURES / f"{name}_diff_4k.jpg"
    if not path.is_file():
        TEXTURES.mkdir(parents=True, exist_ok=True)
        print(f"[flat-scene] fetching {name} (CC0, polyhaven.com)")
        urllib.request.urlretrieve(TEX_URL.format(name), path)
    px = max(8, int(round(TEX[name] / metres_per_px)))
    img = Image.open(path).convert("RGB").resize((px, px), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def sample(tex: np.ndarray, tile_m: float, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    n = tex.shape[0]
    i = np.floor((z / tile_m % 1.0) * n).astype(int) % n
    j = np.floor((x / tile_m % 1.0) * n).astype(int) % n
    return tex[i, j]


def smooth_noise(x, z, seed, scales=(38.0, 17.0, 7.0), weights=(0.6, 0.3, 0.1)):
    """Cheap band-limited noise in [0, 1]: a few random-phase sinusoid sums per scale."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(x)
    for s, w in zip(scales, weights):
        acc = np.zeros_like(x)
        for _ in range(6):
            a = rng.uniform(0, math.pi)
            acc += np.sin((x * math.cos(a) + z * math.sin(a)) * 2 * math.pi / s + rng.uniform(0, 2 * math.pi))
        out += w * acc / math.sqrt(6)
    return 1 / (1 + np.exp(-2.2 * out))


def track_centre(z):
    return TRACK_X + 2.5 * np.sin(z / 14.0)


def ground_rgb(x, z, step):
    """Field colour at (x, z), from textures filtered for splats ``step`` apart."""
    grass = texture("aerial_grass_rock", step / 2)
    dirt = texture("dirt_aerial_03", step / 2)
    # Grass sampled on a rotated, rescaled lattice as well, blended, so the 15 m repeat breaks up.
    c, s = math.cos(0.3), math.sin(0.3)
    g = 0.55 * sample(grass, TEX["aerial_grass_rock"], x, z) + 0.45 * sample(grass, TEX["aerial_grass_rock"] * 1.37, c * x - s * z + 7.1, s * x + c * z + 3.3)
    d = 0.6 * sample(dirt, TEX["dirt_aerial_03"], x, z) + 0.4 * sample(dirt, TEX["dirt_aerial_03"] * 0.83, s * x + c * z + 2.0, c * x - s * z)
    patch = np.clip((smooth_noise(x, z, 3) - 0.62) / 0.12, 0, 1)[:, None]          # bare-dirt patches
    wear = np.clip(1 - np.abs(x - track_centre(z)) / (TRACK_W / 2), 0, 1)[:, None] ** 0.6   # the track
    mix = np.maximum(patch * 0.85, wear)
    rgb = g * (1 - mix) + d * mix
    tone = 0.9 + 0.2 * smooth_noise(x, z, 11, scales=(55.0, 23.0), weights=(0.7, 0.3))   # light/dry variation
    return np.clip(rgb * tone[:, None], 0, 1)


def lattice(rng, lo, hi, step):
    xs = np.arange(lo + step / 2, hi, step)
    gx, gz = np.meshgrid(xs, xs)
    return (gx.ravel() + rng.uniform(-0.3, 0.3, gx.size) * step,
            gz.ravel() + rng.uniform(-0.3, 0.3, gz.size) * step)


def edge_radius(x, z):
    """An irregular outline around the work square, like the ragged edge of a real scan."""
    a = np.arctan2(z, x)
    return EDGE_R + 5 * np.sin(3 * a + 0.7) + 3 * np.sin(7 * a + 2.1) + 1.5 * np.sin(13 * a + 0.3)


def ground_splats(rng):
    px, pz, plx, pdz = PATIO
    x, z = lattice(rng, -SPAN / 2, SPAN / 2, STEP)
    keep = ~((np.abs(x - px) < plx / 2) & (np.abs(z - pz) < pdz / 2))
    x, z = x[keep], z[keep]
    field = (np.column_stack([x, np.zeros_like(x), z]), ground_rgb(x, z, STEP), np.array([STEP * 0.62, 0.004, STEP * 0.62]), None)
    # Beyond the 100 m work square: coarser splats out to a ragged edge, so the site
    # does not read as a perfect tile. Not walkable or measurable.
    ox, oz = lattice(rng, -EDGE_R - 12, EDGE_R + 12, OUTER_STEP)
    r = np.hypot(ox, oz)
    m = ((np.abs(ox) > SPAN / 2) | (np.abs(oz) > SPAN / 2)) & (r < edge_radius(ox, oz))
    # Thin the last few metres so the edge frays instead of ending on a clean line.
    m &= rng.random(ox.size) < np.clip((edge_radius(ox, oz) - r) / 4.0, 0, 1) ** 0.5
    ox, oz = ox[m], oz[m]
    outer = (np.column_stack([ox, np.full(ox.size, -0.006), oz]), ground_rgb(ox, oz, OUTER_STEP), np.array([OUTER_STEP * 0.62, 0.004, OUTER_STEP * 0.62]), None)

    pavers = texture("concrete_pavers_02", PATIO_STEP / 2)
    qx, qz = np.meshgrid(np.arange(-plx / 2 + PATIO_STEP / 2, plx / 2, PATIO_STEP) + px,
                         np.arange(-pdz / 2 + PATIO_STEP / 2, pdz / 2, PATIO_STEP) + pz)
    qx, qz = qx.ravel(), qz.ravel()
    prgb = sample(pavers, TEX["concrete_pavers_02"], qx, qz) * (0.95 + 0.1 * smooth_noise(qx, qz, 5, scales=(6.0,), weights=(1.0,)))[:, None]
    patio = (np.column_stack([qx, np.full_like(qx, 0.012), qz]), np.clip(prgb, 0, 1), np.array([PATIO_STEP * 0.62, 0.003, PATIO_STEP * 0.62]), None)
    # The renderer drops sub-pixel splats, so the fine paving vanishes from far away.
    # A coarse layer just under it carries the average colour at any distance.
    cx2, cz2 = np.meshgrid(np.arange(-plx / 2 + 0.15, plx / 2, 0.3) + px, np.arange(-pdz / 2 + 0.15, pdz / 2, 0.3) + pz)
    base = np.tile(prgb.mean(axis=0), (cx2.size, 1))
    patio_base = (np.column_stack([cx2.ravel(), np.full(cx2.size, 0.006), cz2.ravel()]), np.clip(base, 0, 1), np.array([0.2, 0.003, 0.2]), None)
    return [field, outer, patio_base, patio]


def tree_splats(rng, trees, label="vegetation", density=1.0, size=0.2):
    parts = []
    for tx, tz, r in trees:
        trunk_h = 2.2 + 0.4 * rng.random()
        k = int(160 * density)
        trunk = np.column_stack([tx + rng.normal(0, 0.1, k), rng.uniform(0, trunk_h, k), tz + rng.normal(0, 0.1, k)])
        parts.append((trunk, np.tile([0.30, 0.22, 0.15], (k, 1)) + rng.normal(0, 0.02, (k, 3)), np.array([0.09, 0.09, 0.09]), label))
        # A crown made of a few overlapping clumps, lit from above, reads as foliage.
        hue = rng.normal(0, 0.025, 3)
        for _ in range(rng.integers(4, 7)):
            cr = r * rng.uniform(0.45, 0.7)
            off = rng.normal(0, 0.45 * r, 3) * np.array([1.0, 0.5, 1.0])
            centre = np.array([tx, trunk_h + r * 0.85, tz]) + off
            n = int(700 * cr * density) + 40
            u = rng.normal(size=(n, 3))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            pts = centre + u * cr * rng.random((n, 1)) ** 0.2
            light = (0.5 + 0.5 * (pts[:, 1] - trunk_h) / (2.0 * r) + 0.15 * u[:, 1]).clip(0.35, 1.15)[:, None]
            green = (np.column_stack([0.17 + 0.07 * rng.random(n), 0.31 + 0.12 * rng.random(n), 0.11 + 0.05 * rng.random(n)]) + hue) * light
            parts.append((pts, green.clip(0, 1), np.array([size, size, size]), label))
        # A soft shadow disc under the crown, so the trees sit on the ground.
        m = int(900 * density)
        a, rr = rng.uniform(0, 2 * math.pi, m), r * 1.1 * np.sqrt(rng.random(m))
        sh = np.column_stack([tx + 0.8 + rr * np.cos(a), np.full(m, 0.008), tz + 0.5 + rr * np.sin(a)])
        parts.append((sh, np.tile([0.12, 0.14, 0.08], (m, 1)), np.array([0.35, 0.003, 0.35]), None))
    return parts


def write_ply(path, parts, opacity=0.97):
    """SH degree 0 splats (no f_rest), so a half-million-splat field stays small."""
    xyz = np.vstack([p[0] for p in parts]).astype(np.float32)
    rgb = np.vstack([p[1] for p in parts])
    scale = np.vstack([np.tile(np.log(p[2]), (len(p[0]), 1)) for p in parts])
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.zeros(len(xyz), dtype=[(k, "f4") for k in names])
    arr["x"], arr["y"], arr["z"] = xyz.T
    dc = (rgb - 0.5) / 0.28209479177387814
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = dc.T
    arr["opacity"] = math.log(opacity / (1 - opacity))
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scale.T
    arr["rot_0"] = 1.0
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
    return len(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "work" / "flat_ground")
    args = ap.parse_args()
    work = args.out
    va, pc = work / "viewer_assets", work / "pc"
    va.mkdir(parents=True, exist_ok=True)
    pc.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(26158)
    parts = ground_splats(rng) + tree_splats(rng, TREES)
    count = write_ply(va / "scene.ply", parts)

    n = int(round(SPAN / CELL))
    centres = -SPAN / 2 + (np.arange(n) + 0.5) * CELL
    xx, zz = np.meshgrid(centres, centres)
    ground = np.zeros((n, n), dtype="<f4")
    top = ground.copy()
    for tx, tz, r in TREES:
        dd = np.hypot(xx - tx, zz - tz)
        top = np.maximum(top, np.where(dd <= r, 2.4 + r + np.sqrt(np.clip(r * r - dd * dd, 0, None)), 0)).astype("<f4")
    ground.tofile(va / "ground.f32")
    top.tofile(va / "heights.f32")
    np.ones((n, n), dtype=np.uint8).tofile(va / "coverage.u8")
    (va / "collision.json").write_text(json.dumps({
        "origin_xz": [-SPAN / 2, -SPAN / 2], "cell": CELL, "nx": n, "nz": n,
        "rotation_rowmajor": np.eye(3).tolist(), "max_step": 0.8, "has_coverage": True,
        "spawn": {"x": 0.0, "z": 40.0, "face_xz": [0.0, 0.0]},
        # Every ground splat is ~9 cm and flat: under the engine's default cull (2 px,
        # contribution 3) the far half of the field drops out at low viewing angles.
        "splat_cull": {"min_pixel_size": 0.5, "min_contribution": 0.5},
        "note": "synthetic textured flat ground (make_flat_scene.py)"}, indent=1))

    coarse = 2.0
    m = int(round(SPAN / coarse))
    skin = ground_skin(np.zeros((m, m)), -SPAN / 2, -SPAN / 2, coarse, wall=2.0, skirt=1.0)
    verts = skin.reshape(-1, 3)
    (pc / "collision.collision.glb").write_bytes(glb_bytes(
        [{"name": "collider", "positions": verts, "indices": np.arange(len(verts))}], generator="make_flat_scene.py"))

    coords, rgb = [], []
    for xyz, _, _, label in parts:
        if label:
            coords += xyz[::4].round(3).tolist()
            rgb += [list(label_semantics.CLASS_RGB[label])] * len(xyz[::4])
    counts = {k: sum(1 for c in rgb if tuple(c) == v) for k, v in label_semantics.CLASS_RGB.items()}
    (va / "semantics.json").write_text(json.dumps({
        "schema_version": 1, "classes": list(label_semantics.CLASS_RGB), "counts": counts,
        "note": "synthetic labels written with the scene", "coords": coords, "rgb": rgb}))
    (va / "cameras.json").write_text("[]")

    (work / "frame.json").write_text(json.dumps({
        "rotation_rowmajor": np.eye(3).tolist(), "scale_m_per_unit": 1.0,
        "scale_source": "GPS telemetry similarity fit (synthetic test origin, not a flight)",
        "scale_anchor_gps": {"scale": 1.0, "alignment": {
            "schema_version": 1, "status": "aligned", "method": "synthetic", "scale": 1.0,
            "rotation": [[1, 0, 0], [0, 0, -1], [0, 1, 0]], "translation": [0.0, 0.0, 0.0],
            "fit_rmse_m": 0.0,
            "coordinate_frame": {"type": "ENU", "units": "m", "geodetic_crs": "EPSG:4979",
                                 "altitude_datum": "ellipsoidal", "origin": ORIGIN}}}}, indent=1))
    (work / "project.json").write_text(json.dumps({
        "name": "Open ground", "workflow": "general",
        "notes": "Synthetic 100 m flat ground with CC0 aerial textures (Poly Haven) for planning and placement. Not a scan."}, indent=1))
    print(f"[flat-scene] {count} splats, {n}x{n} grid, {len(skin)} collider triangles -> {work}")


if __name__ == "__main__":
    main()
