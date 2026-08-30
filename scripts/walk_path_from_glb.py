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
from pathlib import Path

import numpy as np


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
    ap.add_argument("--angle", type=float, default=32.0,
                    help="max ground slope the capsule can walk (degrees)")
    ap.add_argument("--smooth", type=int, default=3,
                    help="box-filter width (cells) applied before the slope test")
    ap.add_argument("--min-clearance", type=float, default=0.9,
                    help="corridor half-width the capsule (r=0.34 m) needs (m)")
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
    args.min_clearance = max(args.min_clearance, res * 1.1)
    ref = np.fromfile(args.asset / "heights.f32", np.float32).reshape(nz, nx).astype(np.float64)
    cov = np.fromfile(args.asset / "coverage.u8", np.uint8).reshape(nz, nx)

    tris = read_glb_tris(args.glb)
    raw = top_surface(tris, ref, ox, oz, res, args.band)
    H = smooth_surface(raw, args.smooth)
    print(f"[path] surface grid {H.shape} cell {res:.3f} m, "
          f"x[{ox:.1f}..], z[{oz:.1f}..], covered {np.isfinite(H).mean() * 100:.0f}%")

    # Walkable: collider surface present, real gaussian support under it, and
    # every existing neighbour within the grade. Coverage matters because the
    # heightfield diffuses across holes: a hole is perfectly smooth and
    # perfectly invented, and an autopilot sent across one walks on nothing.
    rise = np.tan(np.radians(args.angle)) * res
    walk = np.isfinite(H) & (cov > 0)
    steep = np.zeros_like(H)
    dz_pairs = np.abs(np.diff(H, axis=0))  # (nz-1, nx) steps between z rows
    dx_pairs = np.abs(np.diff(H, axis=1))  # (nz, nx-1) steps between x cols
    for sl, d in (((slice(None, -1), slice(None)), dz_pairs),
                  ((slice(1, None), slice(None)), dz_pairs),
                  ((slice(None), slice(None, -1)), dx_pairs),
                  ((slice(None), slice(1, None)), dx_pairs)):
        # max, not min: a cell with one gentle neighbour and three cliffs is a
        # ledge, and min() called it walkable
        steep[sl] = np.fmax(steep[sl], np.nan_to_num(d, nan=0.0))
    walk &= steep < rise
    gr = np.degrees(np.arctan(steep[np.isfinite(H) & (cov > 0)] / res))
    print(f"[path] grade over supported cells: "
          + " ".join(f"p{p}={np.percentile(gr, p):.0f}deg" for p in (50, 75, 90, 95)))
    print(f"[path] walkable cells: {walk.mean() * 100:.0f}% "
          f"(step < {rise:.2f} m / {res:.2f} m cell = {args.angle:.0f}deg)")

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
            D_r = clearance(r, res)
            cands_with_clearance.append((r, float(D_r.max()), float(np.nanmedian(H[r]))))
        # Keep candidates with at least minimum clearance if any exist
        valid = [c for c in cands_with_clearance if c[1] >= args.min_clearance]
        if valid:
            walk = max(valid, key=lambda c: (c[2], int(c[0].sum())))[0]
            why = "highest with clearance"
        else:
            # Fallback: pick candidate with the largest clearance
            walk = max(cands_with_clearance, key=lambda c: (c[1], c[2], int(c[0].sum())))[0]
            why = "best clearance fallback"
        print(f"[path] picked '{why}' region: {int(walk.sum())} cells, "
              f"{walk.sum() * res * res:.0f} m2 at median y {np.nanmedian(H[walk]):+.1f} m "
              f"(--pick best, min frac {args.pick_min_frac})")
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

    D = clearance(walk, res)
    cand_ijs = np.argwhere(D >= min(args.min_clearance, 0.3))
    best_ci, best_cj = None, None
    best_score = float('inf')
    for ci_cand, cj_cand in cand_ijs:
        if 2 <= ci_cand < nz - 2 and 2 <= cj_cand < nx - 2:
            sub_cov = cov[ci_cand - 2:ci_cand + 3, cj_cand - 2:cj_cand + 3]
            sub_H = H[ci_cand - 2:ci_cand + 3, cj_cand - 2:cj_cand + 3]
            sub_ref = ref[ci_cand - 2:ci_cand + 3, cj_cand - 2:cj_cand + 3]
            if sub_cov.all() and np.isfinite(sub_H).all() and np.isfinite(sub_ref).all():
                relief_ref = float(np.ptp(sub_ref))
                relief_H = float(np.ptp(sub_H))
                relief = max(relief_ref, relief_H)
                penalty = 0.0 if relief < 1.0 else (relief - 1.0) * 100.0
                score = relief + penalty - 0.2 * D[ci_cand, cj_cand]
                if score < best_score:
                    best_score = score
                    best_ci, best_cj = ci_cand, cj_cand
    if best_ci is not None:
        ci, cj = best_ci, best_cj
    else:
        ci, cj = np.unravel_index(int(D.argmax()), D.shape)

    cx = ox + (cj + 0.5) * res
    cz = oz + (ci + 0.5) * res
    max_d = float(D[ci, cj])
    print(f"[path] best clearance {max_d:.1f} m at x={cx:.1f} z={cz:.1f}")
    if max_d < args.min_clearance:
        if max_d >= res:
            print(f"[path] warning: best clearance {max_d:.2f} m is below target "
                  f"{args.min_clearance:.2f} m; adaptively relaxing clearance to {max_d:.2f} m")
            args.min_clearance = max_d * 0.95
        else:
            raise SystemExit(f"[path] nowhere has {args.min_clearance} m of clearance "
                             f"(best {max_d:.1f} m) — the walkable set is too "
                             f"fragmented to route a capsule of radius 0.34 m")

    # A CORRIDOR, not a ring or a rectangle. This terrain's walkable set is
    # 728 m2 but nowhere has more than 2.8 m of clearance: it is a web of 2-4 m
    # passages through a rocky ridge, so neither an inscribed rectangle nor an
    # inscribed circle fits anything worth walking. The geodesically farthest
    # reachable cell, string-pulled back to waypoints and traversed out and
    # back, uses the passages as they are. The autopilot indexes walk_path
    # cyclically, so out-and-back is already a closed loop.
    passable = walk & (D >= args.min_clearance)
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

    # string-pulling: keep the farthest waypoint still reachable in a straight
    # line, so the loop has few legs but never cuts across an impassable cell
    keep, i = [cells[0]], 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not clear_line(cells[i], cells[j]):
            j -= 1
        keep.append(cells[j])
        i = j
    # the autopilot arrives within 1.5 m, so waypoints closer than that are
    # satisfied the instant they are issued and the walk stalls
    pulled = [keep[0]]
    for c in keep[1:]:
        dist = np.hypot(c[0] - pulled[-1][0], c[1] - pulled[-1][1]) * res
        if dist >= 2.0 or not clear_line(pulled[-1], c):
            pulled.append(c)
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
    max_step = np.tan(np.radians(args.angle)) * res * 1.5
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
    col_file.write_text(json.dumps(col, indent=2), encoding="utf-8")
    print(f"[path] wrote walk_path ({len(corners)} waypoints) + "
          f"spawn ({cx:.1f}, {cz:.1f}) -> {col_file}")


if __name__ == "__main__":
    main()
