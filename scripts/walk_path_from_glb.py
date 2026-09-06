"""Derive the walk loop + spawn candidates from the COLLISION MESH itself.

The export's walk_path is a rectangle around the splat's dense ground zone —
but the collision mesh built from those same gaussians has boulders and pits
the rectangle ignores, and the autopilot spends its 50 m thrashing against a
wall. This script rasterizes the GLB's top surface, keeps cells whose local
slope a capsule could traverse, takes the largest connected walkable region,
inscribes the biggest axis-aligned rectangle in it, and writes that loop back
into viewer_assets/collision.json (plus the region centroid as spawn hint).

  python walk_path_from_glb.py --asset work/rocks/viewer_assets \
      --glb work/rocks/pc/collision.collision.glb
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402

CAPSULE_R = 0.34   # m — Ammo capsule radius of a 1.75 m mover; the viewer scales
                   # this by character_height, so route against the scaled value
CLEARANCE_M = 0.9  # m of clearance a 1.75 m mover wants in a corridor
FULL_HEIGHT = 1.75   # m — the character height viewer/pc.js divides by
MIN_CHAR_H = 0.05    # m — the floor viewer/pc.js puts under character_height


def char_scale(col: dict) -> float:
    """How big the mover is, as a fraction of the 1.75 m default.

    Must stay in step with viewer/pc.js (`CHAR_SCALE = max(0.05, CHAR_H) / 1.75`).
    A scene with no usable character height walks the default body, so this
    returns 1.0 rather than a guess.
    """
    ch = col.get("character_height")
    if isinstance(ch, (int, float)) and rb.finite(ch) and ch > 0:
        return max(MIN_CHAR_H, float(ch)) / FULL_HEIGHT
    return 1.0


def capsule_radius(col: dict) -> float:
    """The capsule radius the mover will actually have in this scene.

    Routing a body larger than the one that walks throws away corridors the
    physics would have taken; routing one smaller sends the autopilot into walls.
    """
    return CAPSULE_R * char_scale(col)


def read_glb_tris(path: Path) -> np.ndarray:
    data = path.read_bytes()
    clen, _ = struct.unpack("<II", data[12:20])
    js = json.loads(data[20:20 + clen])
    bin_off = 20 + clen + 8
    out = []
    for mesh in js["meshes"]:
        for prim in mesh["primitives"]:
            acc = js["accessors"][prim["attributes"]["POSITION"]]
            bv = js["bufferViews"][acc["bufferView"]]
            voff = bin_off + bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            pts = np.frombuffer(data, np.uint8, count=acc["count"] * 12, offset=voff)
            pts = pts.copy().view(np.float32).reshape(-1, 3)
            iacc = js["accessors"][prim["indices"]]
            ibv = js["bufferViews"][iacc["bufferView"]]
            ioff = bin_off + ibv.get("byteOffset", 0) + iacc.get("byteOffset", 0)
            n = iacc["count"]
            if iacc["componentType"] == 5125:
                idx = np.frombuffer(data, np.uint32, count=n, offset=ioff)
            else:
                idx = np.frombuffer(data, np.uint16, count=n, offset=ioff)
            out.append(pts[idx.reshape(-1, 3)])
    return np.concatenate(out, axis=0)


def top_surface(tris: np.ndarray, ref: np.ndarray, ox: float, oz: float,
                cell: float, band: float = 2.5) -> np.ndarray:
    """Highest collider surface per cell, bounded to the reference ground.

    Rasterized onto the EXPORTED heightfield's own grid, so the result is
    directly comparable with heights.f32 and coverage.u8 — no resampling, and
    the loop gets planned on the same surface the spawn is chosen from.

    `band` is the fix for the defect that put the character on an invisible
    plateau: an unqualified "max triangle height in this cell" happily elects a
    floater crust or a voxel-fill slab, because both sit metres above the
    terrain and both are dead flat, so the slope test downstream sees nothing
    wrong. Samples further than `band` from the reference surface are not
    candidates for standing on.
    """
    V = tris.reshape(-1, 3)
    nz, nx = ref.shape
    j = np.floor((V[:, 0] - ox) / cell).astype(int)
    i = np.floor((V[:, 2] - oz) / cell).astype(int)
    ok = (i >= 0) & (i < nz) & (j >= 0) & (j < nx)
    i, j, y = i[ok], j[ok], V[ok, 1]
    ok = np.abs(y - ref[i, j]) <= band
    i, j, y = i[ok], j[ok], y[ok]
    S = np.full((nz, nx), -np.inf)
    np.maximum.at(S, (i, j), y)
    S[~np.isfinite(S)] = np.nan
    return S


def largest_region(mask: np.ndarray) -> np.ndarray:
    """Connected-component (4-neighbour) flood; keep the biggest region."""
    return regions(mask)[0]


def regions(mask: np.ndarray) -> list[np.ndarray]:
    """All connected components (4-neighbour), biggest area first."""
    from collections import deque
    nz, nx = mask.shape
    seen = np.zeros_like(mask, bool)
    comps = []
    for sz in range(nz):
        for sx in range(nx):
            if not mask[sz, sx] or seen[sz, sx]:
                continue
            comp = []
            q = deque([(sz, sx)])
            seen[sz, sx] = True
            while q:
                z, x = q.popleft()
                comp.append((z, x))
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = z + dz, x + dx
                    if 0 <= a < nz and 0 <= b < nx and mask[a, b] and not seen[a, b]:
                        seen[a, b] = True
                        q.append((a, b))
            out = np.zeros_like(mask, bool)
            for z, x in comp:
                out[z, x] = True
            comps.append(out)
    comps.sort(key=lambda m: int(m.sum()), reverse=True)
    return comps


def clearance(mask: np.ndarray, cell: float) -> np.ndarray:
    """Metres from each walkable cell to the nearest unwalkable one.

    Replaces a largest-inscribed-rectangle search. On this terrain the walkable
    region is 26% of the grid but so ragged that its biggest axis-aligned
    rectangle is under 7 m on a side — the rectangle was measuring how convex
    the region is, not how much of it you can walk. Clearance answers the
    question that actually matters: where is there room to move, and how much.

    Two-pass chamfer transform with (1, sqrt2) weights — accurate to a few
    percent, and it avoids a scipy dependency for twenty lines of work.
    """
    D = np.where(mask, np.inf, 0.0)
    nz, nx = mask.shape
    d1, d2 = 1.0, np.sqrt(2.0)
    for i in range(nz):
        for j in range(nx):
            if D[i, j] == 0:
                continue
            best = D[i, j]
            for di, dj, w in ((-1, 0, d1), (0, -1, d1), (-1, -1, d2), (-1, 1, d2)):
                a, b = i + di, j + dj
                if 0 <= a < nz and 0 <= b < nx:
                    best = min(best, D[a, b] + w)
            D[i, j] = best
    for i in range(nz - 1, -1, -1):
        for j in range(nx - 1, -1, -1):
            if D[i, j] == 0:
                continue
            best = D[i, j]
            for di, dj, w in ((1, 0, d1), (0, 1, d1), (1, 1, d2), (1, -1, d2)):
                a, b = i + di, j + dj
                if 0 <= a < nz and 0 <= b < nx:
                    best = min(best, D[a, b] + w)
            D[i, j] = best
    return np.where(np.isfinite(D), D * cell, 0.0) * mask


def largest_patch(mask: np.ndarray, ref: np.ndarray) -> int:
    """Biggest single connected piece of `ref` still inside `mask`, in cells."""
    return max((int((r & ref).sum()) for r in regions(mask & ref)), default=0)


def restrict_to(walk: np.ndarray, cand: np.ndarray, what: str, cell: float,
                keep: float) -> tuple[np.ndarray, bool]:
    """Take cells away from the walkable set, but not if it leaves no floor.

    Returns the mask to use and whether the cut was actually applied, because the
    caller has to tell the viewer which of its decisions survived.

    Two separate cuts have to pass through here — the furniture footprints, and
    the floor the camera derived but the render never drew. Either one is right
    most of the time and catastrophic when the surface underneath is mis-measured,
    and the failure looks the same from both: the walkable grid breaks into scraps
    no loop fits on. So the price is judged as a ratio on the largest surviving
    patch, never as an absolute area, and the cut is refused if it does not pay.
    """
    before = largest_patch(walk, walk)
    after = largest_patch(cand, cand)
    if after >= keep * before:
        print(f"[path] {what}: walkable area {walk.sum() * cell * cell:.0f} -> "
              f"{cand.sum() * cell * cell:.0f} m2, largest patch "
              f"{before * cell * cell:.0f} -> {after * cell * cell:.0f} m2 "
              f"({100 * after / max(before, 1):.0f}% kept)")
        return cand, True
    print(f"[path] {what} NOT applied: it would cut the largest walkable patch from "
          f"{before * cell * cell:.0f} m2 to {after * cell * cell:.0f} m2 "
          f"({100 * after / max(before, 1):.0f}%), below the {keep * 100:.0f}% that "
          f"leaves a floor to route on. The surface under it is mis-measured.")
    return walk, False


def object_cells(boxes: list, ox: float, oz: float, res: float,
                 nx: int, nz: int) -> np.ndarray:
    """Rasterize objects.json footprints onto the routing grid.

    Each box is a rotated rectangle in the ground plane, so a cell is blocked
    when its centre falls inside |u|,|v| after the footprint is turned by -yaw.
    The margin around an object is not added here — the clearance pass already
    shrinks the walkable set by the capsule radius, and baking it in twice would
    wall off the aisles people actually walk down.
    """
    j, i = np.meshgrid(np.arange(nx), np.arange(nz))
    X = (j + 0.5) * res + ox
    Z = (i + 0.5) * res + oz
    blocked = np.zeros((nz, nx), bool)
    for b in boxes:
        cx, cz = b["center_xz"]
        a = np.radians(b["yaw_deg"])
        dx, dz = X - cx, Z - cz
        u = dx * np.cos(a) + dz * np.sin(a)
        v = -dx * np.sin(a) + dz * np.cos(a)
        blocked |= (np.abs(u) <= b["size"][0] / 2) & (np.abs(v) <= b["size"][2] / 2)
    return blocked


def _shift_bounds(n: int, d: int) -> tuple[int, int]:
    """Destination rows/cols that receive a shift of `d` on an axis of length `n`.

    Both ends must be clamped into [0, n]. `min(n, n + d)` alone can go negative
    when |d| > n - which is normal here, because a clearance radius is converted
    to cells by dividing by the cell size, and a collapsed scene can have a 1.7 cm
    grid. A negative stop is a wrap-around to numpy, so the destination slice kept
    15 rows while the source slice kept 0: "could not broadcast input array from
    shape (0,14) into shape (15,14)".
    """
    return min(n, max(0, d)), min(n, max(0, n + d))


def shifted(S: np.ndarray, di: int, dj: int) -> np.ndarray:
    """S moved by (di,dj) cells, vacated edge filled with NaN."""
    out = np.full_like(S, np.nan)
    nz, nx = S.shape
    z0, z1 = _shift_bounds(nz, di)
    x0, x1 = _shift_bounds(nx, dj)
    out[z0:z1, x0:x1] = S[z0 - di:z1 - di, x0 - dj:x1 - dj]
    return out


def shifted_b(S: np.ndarray, di: int, dj: int) -> np.ndarray:
    """Boolean S moved by (di,dj) cells, vacated edge filled with False."""
    out = np.zeros_like(S, bool)
    nz, nx = S.shape
    z0, z1 = _shift_bounds(nz, di)
    x0, x1 = _shift_bounds(nx, dj)
    out[z0:z1, x0:x1] = S[z0 - di:z1 - di, x0 - dj:x1 - dj]
    return out


def _disk(r: int) -> list[tuple[int, int]]:
    return [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
            if di * di + dj * dj <= r * r]


def close_mask(mask: np.ndarray, r: int) -> np.ndarray:
    """Dilate then erode: fills unwalkable gaps narrower than 2r cells.

    Clearance must be measured on this, not on the raw walkable set. On stepped
    ground the unwalkable cells are one-cell riser faces the capsule steps over
    anyway, and against the raw mask every tread sits within 20 cm of a "cliff",
    so nothing clears the margin and the router has no corridor to follow.
    """
    offs = _disk(r)
    dil = np.zeros_like(mask)
    for di, dj in offs:
        dil |= shifted_b(mask, di, dj)
    out = np.ones_like(mask)
    for di, dj in offs:
        out &= shifted_b(dil, di, dj)
    return out


def camera_floor(H: np.ndarray, centres: np.ndarray, agl: float, ox: float,
                 oz: float, res: float, radius: float, tol: float):
    """Cells whose surface is what the take passed over at `agl` metres.

    The heightfield's low percentile votes for a seat back when a column is full
    of upholstery, so in a raked auditorium the walkable set becomes the furniture
    and the spawn ends up two metres above the floor. The footage already knows
    the answer: the camera held at `agl` above the ground it filmed, so the ground
    is the surface `agl` below each camera position.
    """
    nz, nx = H.shape
    mask = np.zeros((nz, nx), bool)
    used = 0
    r = max(1, int(round(radius / res)))
    for cx, cy, cz in centres:
        gi, gj = int(np.floor((cz - oz) / res)), int(np.floor((cx - ox) / res))
        if not (0 <= gi < nz and 0 <= gj < nx):
            continue
        target = float(cy) - agl
        a0, a1 = max(0, gi - r), min(nz, gi + r + 1)
        b0, b1 = max(0, gj - r), min(nx, gj + r + 1)
        sub_H = H[a0:a1, b0:b1]
        ii, jj = np.mgrid[a0:a1, b0:b1]
        hit = np.isfinite(sub_H) & (np.abs(sub_H - target) <= tol) \
            & ((ii - gi) ** 2 + (jj - gj) ** 2 <= r * r)
        if hit.any():
            mask[ii[hit], jj[hit]] = True
            used += 1
    return mask, used


def floor_field(H: np.ndarray, res: float, base: float, coarse: float = 0.5,
                percentile: float = 10.0) -> np.ndarray:
    """The low surface around each cell, over a window of `base` metres.

    Estimated on a coarse grid on purpose: at cell resolution a seat row is a
    plateau with no floor anywhere in the window, so the cell that IS a seat back
    looks identical to the cell that is the aisle beside it. Blocks of `coarse`
    first, then a low percentile across the window, then back to the fine grid.
    """
    nz, nx = H.shape
    ds = max(1, int(round(coarse / res)))
    cz, cx = (nz + ds - 1) // ds, (nx + ds - 1) // ds
    lo = np.full((cz, cx), np.nan)
    for a in range(cz):
        for b in range(cx):
            v = H[a * ds:min(nz, (a + 1) * ds),
                  b * ds:min(nx, (b + 1) * ds)]
            v = v[np.isfinite(v)]
            if v.size:
                lo[a, b] = np.percentile(v, percentile)
    r = max(1, int(round(base / (ds * res))) // 2)
    r = min(r, (min(cz, cx) - 1) // 2)
    if r >= 1:
        # edge-pad first: sliding_window_view returns valid-mode output and would
        # otherwise shrink the coarse grid by r on every side, and the upsampled
        # field would come back smaller than H.
        W = np.lib.stride_tricks.sliding_window_view(
            np.pad(lo, r, mode="edge"), (2 * r + 1, 2 * r + 1))
        with np.errstate(invalid="ignore"):
            low = np.nanpercentile(W, percentile, axis=(-2, -1))
    else:
        low = lo
    return np.repeat(np.repeat(low, ds, axis=0), ds, axis=1)[:nz, :nx]


def smooth_surface(A: np.ndarray, k: int = 3) -> np.ndarray:
    """Box-filter the surface over finite cells only, preserving the hole mask.

    Required before any slope test. The collider is a voxel shell, so its top
    steps in exact multiples of --voxel-size: measured raw on 0.64 m cells the
    neighbour differences come out quantized to 0.35 / 0.70 / 1.05 m (p50 / p75
    / p90), i.e. a "29 degree median grade" that is one voxel of stair-step and
    no terrain at all. That artifact alone consumed the whole 32 degree budget
    and left 3% of the grid connected. Filtered, the same terrain reads p50=10,
    p90=36 degrees, which is what the ridge actually is.

    Holes must not average in as zero, hence the parallel weight accumulator.
    """
    F = np.nan_to_num(A, nan=0.0)
    W = np.isfinite(A).astype(np.float64)
    fs, ws = np.zeros_like(F), np.zeros_like(W)
    r = k // 2
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            fs += np.roll(np.roll(F, di, 0), dj, 1)
            ws += np.roll(np.roll(W, di, 0), dj, 1)
    out = np.where(ws > 0, fs / np.maximum(ws, 1e-9), np.nan)
    return np.where(np.isfinite(A), out, np.nan)


def geodesic_far_path(passable: np.ndarray, start: tuple[int, int]) -> list:
    """Cell path from `start` to the geodesically farthest reachable cell.

    8-connected BFS with unit steps — the exact metric does not matter, only
    that "farthest" means farthest *through the passages* rather than farthest
    in a straight line, which on a rocky web is usually across a cliff.
    """
    from collections import deque
    nz, nx = passable.shape
    par = np.full((nz, nx, 2), -1, np.int32)
    seen = np.zeros_like(passable, bool)
    seen[start] = True
    q, last = deque([start]), start
    while q:
        i, j = q.popleft()
        last = (i, j)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if (di or dj) and 0 <= a < nz and 0 <= b < nx \
                        and passable[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    par[a, b] = (i, j)
                    q.append((a, b))
    path = [last]
    while tuple(par[path[-1]]) != (-1, -1):
        path.append(tuple(int(v) for v in par[path[-1]]))
    path.reverse()
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, type=Path)
    ap.add_argument("--glb", required=True, type=Path)
    ap.add_argument("--band", type=float, default=2.5,
                    help="max distance from the exported ground to count as standable (m)")
    ap.add_argument("--angle", type=float, default=40.0,
                    help="max sustained ground slope over --grade-base (degrees)")
    ap.add_argument("--grade-base", dest="grade_base", type=float, default=3.0,
                    help="baseline --angle is measured over (m); a metre of a "
                         "voxel-shell collider measures the sampling, not the slope")
    ap.add_argument("--floor-radius", dest="floor_radius", type=float, default=1.5,
                    help="how far around each camera's ground point to look (m)")
    ap.add_argument("--floor-tol", dest="floor_tol", type=float, default=0.45,
                    help="how close to that ground height a cell must be (m)")
    ap.add_argument("--floor-base", dest="floor_base", type=float, default=2.5,
                    help="window (m) the local floor is read over for the spawn")
    ap.add_argument("--furniture", type=float, default=0.4,
                    help="how far above that floor a cell may sit and still count "
                         "as a place to stand (m). Terrain relief is metres, so "
                         "this excludes a seat back without excluding a ridge")
    ap.add_argument("--climb", type=float, default=0.9,
                    help="max one-cell rise (m); defaults to the 0.9 m step the "
                         "mover itself accepts, so the route plans a path the "
                         "capsule will actually follow")
    ap.add_argument("--surface", choices=("hf", "shell"), default="hf",
                    help="hf = plan on ground.f32, the exact surface the collider "
                         "was built from, so the map and the world cannot disagree. "
                         "shell = the old behaviour, re-deriving a surface from the "
                         "voxel shell's top face, kept to measure against")
    ap.add_argument("--smooth", type=int, default=3,
                    help="box-filter width (cells) applied before the slope test")
    ap.add_argument("--min-clearance", type=float, default=None,
                    help=f"corridor half-width the capsule needs (m). Default is "
                         f"{CLEARANCE_M} m scaled by the scene's character height, "
                         f"because the mover the viewer builds is that same fraction "
                         f"of a 1.75 m person")
    ap.add_argument("--near-floor", dest="near_floor", type=float, default=0.0,
                    help="keep the route this far (m) from a surface the gaussians "
                         "actually measured, instead of floor the camera merely "
                         "derived. Off, because measured on the auditorium it costs "
                         "more than it buys: it cut the walk from 230 m2 to one "
                         "22 m2 scrap in the stands, a SHORTER loop than the "
                         "unrestricted route, spawning 2.8 m above its own floor. "
                         "A raked room's floor is mostly under its seats, so this "
                         "is a property of the take, not of the estimator")
    ap.add_argument("--keep-patch", dest="keep_patch", type=float, default=0.35,
                    help="how much of the largest walkable patch a cut may take "
                         "away. High = keep the long tour and accept that it runs "
                         "over floor only the camera derived; low = pay in route "
                         "length to stay on surface the splat actually drew")
    ap.add_argument("--objects", dest="objects", type=int, default=1,
                    help="1 = treat every box in objects.json as an obstacle and "
                         "keep the route off it, 0 = ignore that file. A detected "
                         "set that leaves under 35% of the largest walkable patch "
                         "standing is rejected as a mis-segmented floor, not obeyed")
    ap.add_argument("--pick", choices=("largest", "best"), default="best",
                    help="which walkable region gets the loop: 'largest' takes the "
                         "biggest by area; 'best' prefers the HIGHEST region that is "
                         "at least --pick-min-frac of the biggest, so a low cloud/fog "
                         "bench cannot outvote the courtyard on the hill")
    ap.add_argument("--pick-min-frac", type=float, default=0.15,
                    help="'best' mode: regions smaller than this fraction of the "
                         "largest are never chosen")
    args = ap.parse_args()

    col_file = args.asset / "collision.json"
    col = json.loads(col_file.read_text(encoding="utf-8"))
    nx, nz, res = col["nx"], col["nz"], col["cell"]
    ox, oz = col["origin_xz"]
    # Route the body the viewer actually walks. A room preset ships
    # character_height 0.15, and the unscaled 0.34 m demanded 11x the clearance
    # that mover needs — test2horizontal's 14x17 grid of 0.017 m cells fit none.
    scale = char_scale(col)
    capsule_r = capsule_radius(col)
    explicit = args.min_clearance is not None
    args.min_clearance = args.min_clearance if explicit else CLEARANCE_M * scale
    if scale != 1.0:
        print(f"[path] character_height {col['character_height']} m -> "
              f"{scale:.2f}x body: routing a {capsule_r:.3f} m capsule through a "
              f"{args.min_clearance:.2f} m corridor"
              + (" (--min-clearance given, kept as asked)" if explicit else
                 ", both scaled by the viewer's own CHAR_SCALE"))
    args.min_clearance = max(args.min_clearance, capsule_r)
    ref = np.fromfile(args.asset / "heights.f32", np.float32).reshape(nz, nx).astype(np.float64)
    cov = np.fromfile(args.asset / "coverage.u8", np.uint8).reshape(nz, nx)

    shipped = args.asset / "ground.f32"
    H = None
    if args.surface == "hf" and shipped.exists():
        g = np.fromfile(shipped, np.float32)
        if g.size == nz * nx:
            H = g.astype(np.float64).reshape(nz, nx)
            src = "ground.f32 (the collider's own surface)"
            # ...but "its own" is a claim about provenance this file cannot check
            # on its own. A stale ground.f32 beside a rebuilt collider sends the
            # autopilot walking a map of a different world, which is the exact
            # bug the shared array exists to prevent, so verify it against the
            # mesh the physics will actually use. The rim wall is extruded up, so
            # the median is the statistic that ignores it.
            shell = top_surface(read_glb_tris(args.glb), ref, ox, oz, res, args.band)
            m = np.isfinite(shell) & np.isfinite(H)
            agree = float(np.median(np.abs(H[m] - shell[m]))) if m.any() else float("nan")
            print(f"[path] ground.f32 vs the collider mesh: median "
                  f"{agree:.2f} m apart over {100 * m.mean():.0f}% shared cells")
            if not agree < 0.5:
                print(f"[path] WARNING: they disagree by {agree:.2f} m — the route would "
                      f"be planned on a surface nothing collides with. Rebuild the "
                      f"collider with scripts/tune_collider.py.")
    if H is None:
        H = smooth_surface(top_surface(read_glb_tris(args.glb), ref, ox, oz, res,
                                       args.band), args.smooth)
        src = f"voxel shell top face, smoothed {args.smooth}"
    print(f"[path] surface grid {H.shape} cell {res:.3f} m, "
          f"x[{ox:.1f}..], z[{oz:.1f}..], covered {np.isfinite(H).mean() * 100:.0f}%, from {src}")

    # Traversability is a STEP test, not an angle test at one cell. This scene's
    # ground is an amphitheatre: 0.4 m risers, 0.6 m treads, an effective 34
    # degree slope that is real geometry a person walks up. Measured per cell it
    # reads 74 degrees — one riser inside one 0.12 m cell — so the old test
    # called every tier a cliff and left 1% of the grid walkable, then routed a
    # 0.4 sq m tour. What the capsule may actually do is decided by the mover
    # (combat.js accepts a rise of max(0.9, step length)), so the route uses the
    # same number and keeps a metre-baseline angle test for the surfaces that
    # are genuinely too steep to stand a walk on.
    base = max(1, int(round(args.grade_base / res)))
    rise = np.tan(np.radians(args.angle)) * base * res
    walk = np.isfinite(H) & (cov > 0)
    steep = np.zeros_like(H)   # one-cell rise: the step a foot has to clear
    grade = np.zeros_like(H)   # rise over --grade-base: the slope of the surface
    for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        # max, not min: a cell with one gentle neighbour and three cliffs is a
        # ledge, and min() called it walkable
        steep = np.fmax(steep, np.nan_to_num(np.abs(H - shifted(H, di, dj))))
    for di, dj in ((0, base), (0, -base), (base, 0), (-base, 0)):
        grade = np.fmax(grade, np.nan_to_num(np.abs(H - shifted(H, di, dj))))
    walk &= steep <= args.climb
    walk &= grade <= rise
    gr = np.degrees(np.arctan(steep[np.isfinite(H) & (cov > 0)] / res))
    ga = np.degrees(np.arctan(grade[np.isfinite(H) & (cov > 0)] / (base * res)))
    nan3 = (float("nan"),) * 3
    srp = rb.safe_pct(gr, (50, 75, 90), nan3, label="one-cell rise")
    ggp = rb.safe_pct(ga, (50, 75, 90), nan3, label="grade")
    print(f"[path] one-cell rise: "
          + " ".join(f"p{q}={v:.0f}deg" for q, v in zip((50, 75, 90), srp))
          + f" | grade over {base * res:.1f} m: "
          + " ".join(f"p{q}={v:.0f}deg" for q, v in zip((50, 75, 90), ggp)))
    print(f"[path] walkable cells: {walk.mean() * 100:.0f}% "
          f"(one-cell rise <= {args.climb:.2f} m, "
          f"grade over {base * res:.1f} m <= {args.angle:.0f}deg)")

    # The take itself says where the floor is. The camera was held
    # --camera-agl metres above the ground it filmed, so the surface that far
    # below each camera position is the ground an operator actually stood on.
    # Without this a raked auditorium votes its seat backs in as floor — a
    # column of upholstery has no floor sample in it — and the tour ends up
    # walking on the furniture, which is what the walk-test frames showed.
    cf = None
    work = args.asset.parent
    kp, fj = work / "keyframes_poses.jsonl", work / "frame.json"
    if kp.exists() and fj.exists():
        fr = json.loads(fj.read_text(encoding="utf-8"))
        agl = float(fr.get("camera_agl_m") or 0.0)
        Rg = np.array(fr["rotation_rowmajor"], np.float64)
        sc = float(fr["scale_m_per_unit"])
        C = []
        for line in kp.read_text(encoding="utf-8").splitlines():
            c = json.loads(line).get("camera")
            if c:
                R = np.array(c["R_rowmajor"], np.float64)
                C.append(-(R.T @ np.array(c["t"], np.float64)))
        if C and agl > 0.0:
            C = (np.array(C) @ Rg.T) * sc
            cf, used = camera_floor(H, C, agl, ox, oz, res,
                                    args.floor_radius, args.floor_tol)
            on = walk & cf
            print(f"[path] filmed ground = camera {agl:.1f} m above it, "
                  f"{used}/{len(C)} cameras matched: {int(on.sum())} of "
                  f"{int(walk.sum())} walkable cells sit on it "
                  f"({on.sum() * res * res:.0f} m2) — used to choose where to "
                  f"route and spawn, not to cut the walkable set")

    # Furniture is an obstacle, not terrain. The heightfield can only store one
    # height per column, so a seat bank arrives as a 0.4 m bump in the ground —
    # which the step test happily calls walkable, and the player ends up striding
    # across upholstery. objects.json carries the same footprints as primitives,
    # so cut them out of the walkable set here and the route keeps the aisles.
    #
    # But a 2.5D floor that has been mis-measured makes every box a slab of the
    # aisle itself, and obeying that strands the player with no floor. So the
    # block is only applied if a walkable patch worth routing survives it.
    obj_file = args.asset / "objects.json"
    objects_applied = 0
    if args.objects and obj_file.exists():
        boxes = json.loads(obj_file.read_text(encoding="utf-8")).get("boxes", [])
        if boxes:
            blocked = object_cells(boxes, ox, oz, res, nx, nz)
            hit = walk & blocked
            walk, applied = restrict_to(
                walk, walk & ~blocked,
                f"{len(boxes)} objects ({int(hit.sum())} cells, "
                f"{hit.sum() * res * res:.0f} m2 of floor)", res, args.keep_patch)
            objects_applied = len(boxes) if applied else 0

    # Optional: keep the route on floor the gaussians actually drew, rather than
    # floor only the camera's height implies. Off by default — on the auditorium
    # the measured floor is 15% of the grid because a raked room's floor is under
    # its seats, and forcing the cut left one 22 m2 fragment in the stands, a
    # shorter loop, and a spawn 2.8 m off the ground it stood on.
    meas = cov == 1
    if args.near_floor > 0 and int(meas.sum()):
        d = clearance(~meas, res)          # metres to the nearest measured cell
        far = walk & (d > args.near_floor)
        walk, _ = restrict_to(
            walk, walk & (d <= args.near_floor),
            f"floor within {args.near_floor:.2f} m of a measurement "
            f"({int(far.sum())} cells, "
            f"{far.sum() * res * res:.0f} m2 painted by pose only)",
            res, args.keep_patch)

    # Gaps narrower than the capsule are things it steps over, not walls, so
    # clearance is measured on the closed mask.
    rc = int(round(capsule_r / res))
    regs = regions(walk)
    if len(regs) > 1:
        print("[path] walkable regions (area, median height):")
        for r in regs[:6]:
            print(f"[path]   {int(r.sum())} cells, {r.sum() * res * res:.0f} m2, "
                  f"median y {np.nanmedian(H[r]):+.1f} m")
    if args.pick == "best" and len(regs) > 1:
        biggest = int(regs[0].sum())
        cands = [r for r in regs if r.sum() >= args.pick_min_frac * biggest]
        # Prefer candidates that have usable clearance; among those, prefer highest.
        cands_with_clearance = []
        for r in cands:
            D_r = clearance(close_mask(r, rc), res)
            n_cf = int((r & cf).sum()) if cf is not None else 0
            cands_with_clearance.append(
                (r, float(D_r.max()), float(np.nanmedian(H[r])), n_cf))
        # Keep candidates with at least minimum clearance if any exist
        valid = [c for c in cands_with_clearance if c[1] >= args.min_clearance]
        if valid:
            walk = max(valid, key=lambda c: (c[3] > 0, c[2], int(c[0].sum())))[0]
            why = "highest with clearance"
        else:
            # Fallback: pick candidate with the largest clearance
            walk = max(cands_with_clearance,
                       key=lambda c: (c[3] > 0, c[1], c[2], int(c[0].sum())))[0]
            why = "best clearance fallback"
        print(f"[path] picked '{why}' region: {int(walk.sum())} cells, "
              f"{walk.sum() * res * res:.0f} m2 at median y {np.nanmedian(H[walk]):+.1f} m "
              f"(--pick best, min frac {args.pick_min_frac})")
    else:
        if cf is not None and len(regs) > 1:
            walk = max(regs, key=lambda r: (int((r & cf).sum()) > 0, int(r.sum())))
        else:
            walk = regs[0]
    print(f"[path] selected region: {walk.mean() * 100:.0f}% of grid "
          f"({int(walk.sum())} cells, {walk.sum() * res * res:.0f} m2)")

    # Indexing here MUST match top_surface's floor(), not round(): an earlier
    # round() put samples half a cell off, so an edge one cell inside the region
    # tested cells one cell outside it and nothing ever passed.
    def in_walk(x: float, z: float) -> bool:
        i = int(np.floor((z - oz) / res))
        j = int(np.floor((x - ox) / res))
        return 0 <= i < walk.shape[0] and 0 <= j < walk.shape[1] and bool(walk[i, j])

    def cell_h(x: float, z: float) -> float:
        i = int(np.floor((z - oz) / res))
        j = int(np.floor((x - ox) / res))
        if 0 <= i < H.shape[0] and 0 <= j < H.shape[1]:
            return H[i, j]
        return np.nan

    D = clearance(close_mask(walk, rc), res)
    # Relief is judged over the space the capsule actually occupies, not a fixed
    # cell count: 5 cells is 1.5 m on a coarse scan and 0.6 m on a fine one.
    rad = max(2, int(round(capsule_r * 1.5 / res)))   # a disc 1.5x the capsule radius
    flat = 0.25   # m — rocks measures 0.09 and room_w_jsonl 0.15 at a good spawn

    F = floor_field(H, res, args.floor_base)
    obj = np.where(np.isfinite(F), H - F, np.nan)   # metres above the local floor

    def best_cell(mask: np.ndarray):
        """Flattest, best-cleared, most-supported, least-furniture cell in `mask`.

        A footprint cell with no ground under it outranks any amount of relief —
        the capsule needs the whole footprint — and standing on an object outranks
        relief but not a hole. Rejecting unsupported candidates outright sent the
        caller to a widest-point fallback that parked the spawn in a hole.
        """
        hole_cost = 1000.0   # one footprint cell without ground
        best_score, best_ij, best_obj = float('inf'), None, float('nan')
        for ci_cand, cj_cand in np.argwhere(mask):
            if not (rad <= ci_cand < nz - rad and rad <= cj_cand < nx - rad):
                continue
            sl = (slice(ci_cand - rad, ci_cand + rad + 1),
                  slice(cj_cand - rad, cj_cand + rad + 1))
            sub_cov = cov[sl] > 0
            sub_H, sub_ref = H[sl], ref[sl]
            good = sub_cov & np.isfinite(sub_H) & np.isfinite(sub_ref)
            holes = int(sub_cov.size - good.sum())
            relief = max(float(np.ptp(sub_ref[good])), float(np.ptp(sub_H[good]))) \
                if good.sum() >= 2 else float('inf')
            stand = np.nanmax(obj[sl]) if np.isfinite(obj[sl]).any() else 0.0
            penalty = 0.0 if relief < flat else (relief - flat) * 100.0
            above = 0.0 if stand <= args.furniture \
                else (stand - args.furniture) * 200.0
            off = 0.0 if (cf is None or cf[ci_cand, cj_cand]) else 50.0
            score = relief + penalty + above + hole_cost * holes + off \
                - 0.2 * D[ci_cand, cj_cand]
            if score < best_score:
                best_score, best_ij, best_obj = score, (int(ci_cand), int(cj_cand)), stand
        return best_ij, (best_score >= hole_cost), best_obj

    # A clearance target the scene cannot meet is relaxed, but only as far as it
    # must be to get a corridor at all, and never below the capsule's own radius.
    # Relaxing to 95% of the best cell's clearance — what this used to do — is
    # self-defeating by construction: only the peak cell passes, so a 220 m2
    # walkable region in room_multi_video routed a one-cell tour.
    want = args.min_clearance
    floor = capsule_r
    # ~10 m of corridor, but no scene can give more cells than it has: a 0.25 m
    # take wants a proportion of what it walked, not an impossible absolute.
    min_cells = max(8, min(int(round(10.0 / res)), int(walk.sum() // 4)))
    t, comps = want, []
    while True:
        passable = walk & (D >= t)
        comps = [r for r in regions(passable) if int(r.sum()) >= min_cells]
        if comps or t <= floor:
            break
        t = max(floor, t * 0.7)
    if not comps:
        max_d = float(D.max())
        if max_d < res:
            rb.die(rb.EMPTY_INPUT,
                   f"[path] nowhere has {t:.2f} m of clearance (best {max_d:.2f} m, "
                   f"cell {res:.2f} m) — the walkable set is too fragmented to route "
                   f"a capsule of radius {capsule_r:.3f} m")
        print(f"[path] warning: no corridor of {min_cells} cells at {floor:.2f} m "
              f"clearance; routing the single widest spot "
              f"({max_d:.2f} m, {int(walk.sum()) * res * res:.0f} m2 walkable)")
        passable = walk & (D >= max_d * 0.95)
        comps = [passable]
        t = max_d * 0.95
    else:
        comps.sort(key=lambda r: int(r.sum()), reverse=True)
    if t < want:
        print(f"[path] clearance target {want:.2f} m unreachable here; relaxed to "
              f"{t:.2f} m to get a corridor at all (capsule radius {capsule_r:.3f} m)")
    args.min_clearance = t
    comp = comps[0]
    ij, hole_spawn, stand = best_cell(comp)
    if ij is None:
        ij = tuple(int(v) for v in np.unravel_index(
            int((D * comp).argmax()), D.shape))
        print(f"[path] no cell in the routed region has a full capsule footprint "
              f"inside the grid; using its widest point")
    ci, cj = ij
    cx = ox + (cj + 0.5) * res
    cz = oz + (ci + 0.5) * res
    print(f"[path] routed region: {int(comp.sum())} cells, "
          f"{comp.sum() * res * res:.0f} m2 at {t:.2f} m clearance "
          f"({len(comps)} regions of >= {min_cells} cells)")
    print(f"[path] spawn {cx:.1f} {cz:.1f}, clearance {float(D[ci, cj]):.2f} m, "
          f"{stand:.2f} m above the floor around it"
          + (" [warning: ground missing under part of the spawn footprint]"
             if hole_spawn else "")
          + (" [warning: spawn stands on an object, not on the floor]"
             if stand > args.furniture else ""))

    # A CORRIDOR, not a ring or a rectangle. This terrain's walkable set is
    # 728 m2 but nowhere has more than 2.8 m of clearance: it is a web of 2-4 m
    # passages through a rocky ridge, so neither an inscribed rectangle nor an
    # inscribed circle fits anything worth walking. The geodesically farthest
    # reachable cell, string-pulled back to waypoints and traversed out and
    # back, uses the passages as they are. The autopilot indexes walk_path
    # cyclically, so out-and-back is already a closed loop.
    cells = geodesic_far_path(passable, (int(ci), int(cj)))
    print(f"[path] corridor: {len(cells)} cells, "
          f"{sum(1 for _ in cells) * res:.0f} m of grid steps (before pulling)")

    def clear_line(a: tuple[int, int], b: tuple[int, int]) -> bool:
        ax, az = ox + (a[1] + 0.5) * res, oz + (a[0] + 0.5) * res
        bx, bz = ox + (b[1] + 0.5) * res, oz + (b[0] + 0.5) * res
        gap = res / 2.0
        n = max(int(np.hypot(bx - ax, bz - az) / gap), 1)
        for f in np.linspace(0, 1, n + 1):
            x = ax + (bx - ax) * f
            z = az + (bz - az) * f
            if not in_walk(x, z):
                return False
            h = cell_h(x, z)
            if np.isnan(h):
                return False
        return True

    # string-pulling: from the current waypoint take the FARTHEST corridor cell
    # a straight line reaches, so the loop has few legs. The autopilot arrives
    # within 1.5 m, so a leg shorter than 2 m is issued and satisfied in the same
    # frame and the walk stalls — prefer the farthest clear cell that is also
    # far enough. Every leg published here is a leg clear_line has checked: an
    # earlier version dropped waypoints on distance alone and re-tested nothing,
    # which put 46 of 272 auditorium loop samples outside the walkable region.
    keep, i = [cells[0]], 0
    while i < len(cells) - 1:
        picked = i + 1
        for j in range(len(cells) - 1, i, -1):
            if clear_line(cells[i], cells[j]):
                picked = j
                break
        keep.append(cells[picked])
        i = picked
    pulled = keep
    if len(pulled) < 2:
        pulled = [keep[0], keep[-1]]

    out = [(ox + (j + 0.5) * res, oz + (i + 0.5) * res) for i, j in pulled]
    span = sum(np.hypot(out[k + 1][0] - out[k][0], out[k + 1][1] - out[k][1])
               for k in range(len(out) - 1))
    # out and back, so the cyclic waypoint list closes without a teleport leg
    corners = out + out[-2:0:-1]
    perim = 2 * span
    print(f"[path] loop from ({cx:.1f}, {cz:.1f}), one-way {span:.0f} m, "
          f"perimeter {perim:.0f} m, {len(corners)} waypoints")

    # final honesty check on what we are about to publish
    gap = res / 2.0
    max_step = args.climb   # samples are half a cell apart; a riser is legal
    pts = []
    for k in range(len(corners)):
        a, b = corners[k], corners[(k + 1) % len(corners)]
        m = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) / gap), 1)
        for f in np.linspace(0, 1, m + 1):
            pts.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    hs = np.array([cell_h(x, z) for x, z in pts])
    off = sum(not in_walk(x, z) for x, z in pts)
    bad = off + int(np.isnan(hs).sum() + (np.abs(np.diff(hs)) > max_step).sum())
    print(f"[path] loop check: {len(pts)} samples, {off} outside region, "
          f"{bad} bad total ({100 * bad / len(pts):.1f}%)")
    if bad > max(2, int(0.02 * len(pts))):
        raise SystemExit("[path] the pulled loop leaves the walkable region — "
                         "raise --min-clearance so the corridor is wider")

    col["walk_path"] = [list(map(float, c)) for c in corners]

    # Spawn must live on the SAME surface as the path — the export's dense-zone
    # hint can sit in a walled pit the corridor never touches (that cost a full
    # run: the capsule spawned in a pit while the loop was on the crust plain
    # above it). The clearance maximum is the roomiest cell in the region, and
    # it is where the corridor starts.
    if "spawn" not in col or not isinstance(col.get("spawn"), dict):
        col["spawn"] = {}
    col["spawn"]["x"] = float(cx)
    col["spawn"]["z"] = float(cz)
    col["spawn"]["face_xz"] = [float(corners[1][0]), float(corners[1][1])]
    # Which boxes the viewer may treat as solid: the ones the router agreed to
    # route around. Boxes it rejected are a mis-measured floor, and colliding with
    # those would wall the player into invisible geometry nobody planned for.
    col["object_colliders"] = objects_applied
    # Published as data, not just printed, because the collider surface is chosen
    # by which one this route walks better (scripts/tune_collider.py) — a judge
    # that reads stdout would break the first time a message is reworded.
    col["route_metrics"] = {
        "surface": src,
        "walkable_pct": round(float(walk.mean() * 100), 1),
        "routed_m2": round(float(comp.sum() * res * res), 1),
        "perimeter_m": round(float(perim), 1),
        "waypoints": int(len(corners)),
        "loop_samples": int(len(pts)),
        "loop_bad_pct": round(100 * bad / len(pts), 2),
        "spawn_clearance_m": round(float(D[ci, cj]), 2),
        "spawn_above_floor_m": round(float(stand), 2),
        "spawn_hole": bool(hole_spawn),
        "spawn_on_object": bool(stand > args.furniture),
    }
    rb.write_json(col_file, col)
    print(f"[path] wrote walk_path ({len(corners)} waypoints) + "
          f"spawn ({cx:.1f}, {cz:.1f}) -> {col_file}")


if __name__ == "__main__":
    main()
