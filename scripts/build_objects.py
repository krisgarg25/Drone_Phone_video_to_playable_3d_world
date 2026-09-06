"""Turn the stuff standing on the floor into box colliders the engine can use.

The collision mesh is a heightfield skin: one number per column, so a seat bank
becomes a bump in the ground and the player ends up walking on upholstery. The
fix is the one games have always used — keep the 2.5D field for the FLOOR and
give everything that stands on it a real primitive.

Method, and why it is not proximity clustering: the gaussians are 5-15 cm fuzzy
discs that overlap every gap, so 3D nearest-neighbour merging fuses the whole
room into one blob (measured: 76% of above-floor points in a single 25 x 13 m
box). What separates objects instead is a height slice of the flat stuff — keep
the near-horizontal discs between 0.1 and 1.2 m above the floor, cluster their
footprints in the ground plane, and only merge neighbouring cells whose surface
height agrees to within a hand span. Adjacent seat rows are 0.4 m apart in
height where they meet, so they stay apart.

"Nothing found" is a legitimate answer for this step: a sparse reconstruction,
a scene with no supported ground cell, or a diorama whose furniture lift is
2 cm all produce an objects.json with zero boxes and a printed reason. What it
must not produce is a traceback, because the box list is an enhancement to a
collider that already shipped. The lift band itself stays in absolute metres:
scaling it off a scene's own relief would start calling a quarry's boulders
furniture, and the --min-lift/--max-lift flags are the honest way to retune it.

  python build_objects.py --asset work/auditorium/viewer_assets
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402

VOX = 0.10        # m — footprint grid; finer than the 0.25 m that fused rows
LIFT_MIN = 0.10   # m above the floor: off it
LIFT_MAX = 1.20   # m: above this it is wall décor or a ceiling
FLAT = 0.82       # cos(35 deg) — splat normal vs up; a flat-ish top surface
HT_TOL = 0.10     # m — how close two footprint cells must be to be one surface
MIN_PTS = 80      # gaussians in a cluster before it is an object
MIN_CELLS = 4     # 0.1 m cells, so ~0.2 m across
MIN_SIDE = 0.30   # m — a 30 cm sliver is noise, not furniture

VERTEX_PROPS = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3")


def require_props(arr, names, src: Path) -> None:
    """A vertex table without these columns is not a gaussian splat."""
    missing = [n for n in names if n not in (arr.dtype.names or ())]
    if missing:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{src.name} has no {', '.join(missing)} property, so neither the "
            f"surface normal nor the footprint of a splat can be measured.\n"
            f"  export_viewer_assets.py owns the vertex table; re-run train + export.",
            returncode=3)


def cluster_flat_tops(P, lift, flat, reject: dict = None):
    """Group horizontal surfaces into objects. One tuple per object, below."""
    Q, L = P[flat], lift[flat]
    if len(Q) == 0:
        return []

    def drop(reason):
        if reject is not None:
            reject[reason] = reject.get(reason, 0) + 1

    x0 = rb.safe_min(Q[:, 0], 0.0, label="flat-top x min")
    z0 = rb.safe_min(Q[:, 2], 0.0, label="flat-top z min")
    kx = np.floor((Q[:, 0] - x0) / VOX).astype(np.int64)
    kz = np.floor((Q[:, 2] - z0) / VOX).astype(np.int64)
    SX = int(rb.safe_max(kx, 0.0, label="flat-top x extent")) + 1
    SZ = int(rb.safe_max(kz, 0.0, label="flat-top z extent")) + 1
    uk, binv = np.unique(kz * SX + kx, return_inverse=True)
    n = len(uk)
    order = np.argsort(binv, kind="stable")
    b_s, L_s = binv[order], L[order]
    st = np.searchsorted(b_s, np.arange(n), side="left")
    en = np.searchsorted(b_s, np.arange(n), side="right")
    hmed = np.array([float(np.median(L_s[a:b])) if b > a else np.inf
                     for a, b in zip(st, en)])

    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cz, cx = np.unravel_index(uk, (SZ, SX))
    idx = {int(k): i for i, k in enumerate(uk)}
    for dz in (0, 1):
        for dx in (-1, 0, 1):
            if dz == dx == 0:
                continue
            b, c = cz + dz, cx + dx
            ok = (b >= 0) & (b < SZ) & (c >= 0) & (c < SX)
            ii = np.nonzero(ok)[0]
            for i, kk in zip(ii, b[ii] * SX + c[ii]):
                j = idx.get(int(kk))
                # merge only at equal height, so a cushion does not weld to the
                # seat back behind it — that is what separates one row from the next
                if j is not None and abs(hmed[i] - hmed[j]) <= HT_TOL:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
    lab = np.array([find(i) for i in range(n)])
    plab = lab[binv]
    roots, rc = np.unique(lab, return_counts=True)
    out = []
    for r, c in zip(roots, rc):
        if c < MIN_CELLS:
            drop("cells")
            continue
        sel = plab == r
        pts = Q[sel]
        if len(pts) < MIN_PTS:
            drop("points")
            continue
        cen = pts.mean(axis=0)
        # lift = point y - floor y, so floor = y - lift. The median over the
        # cluster is the ground this object stands on, which the box extrudes from.
        h0 = float(np.median(pts[:, 1] - L[sel]))
        v2 = np.stack([pts[:, 0] - cen[0], pts[:, 2] - cen[2]], axis=1)
        _, s, vt = np.linalg.svd(v2, full_matrices=False)
        e = v2 @ vt.T
        dl, dw = float(e[:, 0].max() - e[:, 0].min()), float(e[:, 1].max() - e[:, 1].min())
        if min(dl, dw) < MIN_SIDE:
            drop("side")
            if reject is not None:
                reject["widest sliver m"] = max(reject.get("widest sliver m", 0.0),
                                                min(dl, dw))
            continue
        out.append((np.nonzero(sel)[0], (dl, dw), float(np.median(L[sel])), cen,
                    float(np.arctan2(vt[0, 1], vt[0, 0])), h0,
                    float(pts[:, 1].max())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, type=Path)
    ap.add_argument("--min-lift", type=float, default=LIFT_MIN)
    ap.add_argument("--max-lift", type=float, default=LIFT_MAX)
    args = ap.parse_args()

    a = args.asset
    col = rb.read_json(a / "collision.json")
    if not isinstance(col, dict) or not {"nx", "nz", "cell", "origin_xz"} <= set(col):
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{(a / 'collision.json').name} is missing or has no grid header "
            f"(nx/nz/cell/origin_xz), so there is no floor to measure a lift against.\n"
            f"  export_viewer_assets.py writes it; re-run export.",
            returncode=3)
    nx, nz, res = int(col["nx"]), int(col["nz"]), float(col["cell"])
    ox, oz = (float(v) for v in col["origin_xz"])
    if nx <= 0 or nz <= 0 or not rb.finite(res, ox, oz) or res <= 0:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"collision.json declares an unusable grid (nx={nx}, nz={nz}, "
            f"cell={res!r}, origin_xz=({ox!r}, {oz!r})).\n"
            f"  export_viewer_assets.py wrote it; re-run export.",
            returncode=3)
    shape = (nz, nx)
    H = rb.load_array(a / "heights.f32", np.float32, shape,
                      label="heights.f32 (export_viewer_assets)").astype(np.float64)
    cov = rb.load_array(a / "coverage.u8", np.uint8, shape,
                        label="coverage.u8 (export_viewer_assets)")
    sup = cov > 0
    if not sup.any():
        rb.warn("coverage.u8 marks no supported cell: every floor height in the "
                "grid is guesswork, so nothing can be called furniture. Writing an "
                "empty objects.json.")
    scene_ply = a / "scene.ply"
    if not scene_ply.exists():
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{scene_ply} does not exist.\n"
            f"  export_viewer_assets.py writes it; run export before this step.",
            returncode=3)
    d = PlyData.read(str(scene_ply))["vertex"].data
    if len(d) == 0:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{scene_ply.name} holds 0 gaussians — train or export produced an "
            f"empty splat, so there is nothing to cluster into objects.",
            returncode=3)
    require_props(d, VERTEX_PROPS, scene_ply)

    op = 1.0 / (1.0 + np.exp(-np.asarray(d["opacity"], np.float64)))
    keep = op >= 0.15
    P = np.stack([np.asarray(d[k], np.float64) for k in ("x", "y", "z")], axis=1)[keep]
    SC = np.exp(np.stack([np.asarray(d[k], np.float64) for k in
                          ("scale_0", "scale_1", "scale_2")], axis=1))[keep]
    q = np.asarray(np.stack([np.asarray(d[k], np.float64) for k in
                             ("rot_0", "rot_1", "rot_2", "rot_3")], axis=1))[keep]
    # A quaternion with a zero (or NaN) norm has no axis at all; calling its
    # shortest axis a surface normal would invent a flat top out of nothing.
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    qn = np.where(np.isfinite(qn) & (qn > 1e-9), qn, np.nan)
    q = q / qn
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
    ], axis=1)
    ax = SC.argmin(axis=1)
    nrm = R[np.arange(len(R)), :, ax]          # the shortest axis is the surface normal
    gi = np.clip(((P[:, 2] - oz) / res).astype(int), 0, nz - 1)
    gj = np.clip(((P[:, 0] - ox) / res).astype(int), 0, nx - 1)
    lift = P[:, 1] - H[gi, gj]
    flat = (np.abs(nrm @ np.array([0.0, 1.0, 0.0])) >= FLAT) \
        & (lift > args.min_lift) & (lift < args.max_lift) & (cov[gi, gj] > 0)
    flat = flat & np.isfinite(lift)
    n_flat = int(flat.sum())
    print(f"[objects] {len(P)} gaussians; {n_flat} are flat tops "
          f"{args.min_lift:.2f}-{args.max_lift:.2f} m above the floor "
          f"on supported ground "
          f"({100 * rb.safe_mean(flat, 0.0, label='flat fraction'):.0f}%)")

    if n_flat == 0 and len(P):
        report_no_flat_tops(P, lift, sup, H, args)

    reject = {}
    groups = cluster_flat_tops(P, lift, flat, reject)
    boxes = []
    for k, (sel, (dl, dw), lh, cen, yaw, h0, ymax) in enumerate(groups):
        boxes.append({
            "id": k,
            "center_xz": [round(float(cen[0]), 3), round(float(cen[2]), 3)],
            "center_y": round((h0 + ymax) / 2.0, 3),
            "size": [round(max(dl, 0.2), 3), round(max(ymax - h0, 0.15), 3),
                     round(max(dw, 0.2), 3)],
            "yaw_deg": round(float(np.degrees(yaw)), 1),
            "lift_m": round(lh, 3),
            "floor_y": round(h0, 3),
            "points": int(len(sel)),
        })
    area = sum(b["size"][0] * b["size"][2] for b in boxes)
    print(f"[objects] {len(boxes)} boxes, {area:.0f} m2 of footprint, "
          f"top heights p50 {rb.safe_median([b['lift_m'] for b in boxes], 0.0, label='box lift p50'):.2f} m"
          if boxes else "[objects] nothing found")
    if n_flat and not boxes and reject:
        tally = ", ".join(f"{v} {k}" for k, v in sorted(reject.items())
                          if k != "widest sliver m")
        widest = reject.get("widest sliver m", 0.0)
        rb.warn(f"[objects] {sum(v for k, v in reject.items() if k != 'widest sliver m')} "
                f"flat-top clusters were found and every one was rejected ({tally}"
                + (f", the widest sliver measured {widest:.2f} m against the "
                   f"{MIN_SIDE:.2f} m minimum" if widest else "")
                + "). Those thresholds — a 0.10 m footprint grid, "
                  f"{MIN_CELLS} cells, {MIN_PTS} gaussians, {MIN_SIDE:.2f} m minimum "
                  "side — are fixed and have no flag, so a scene below human "
                  "furniture scale finds no objects here by construction. Zero boxes "
                  "is the honest answer, not a lost object.")
    for b in sorted(boxes, key=lambda b: -b["size"][0] * b["size"][2])[:8]:
        print(f"  #{b['id']:02d} {b['size'][0]:.2f} x {b['size'][2]:.2f} m "
              f"x {b['size'][1]:.2f} tall at ({b['center_xz'][0]:+.1f}, "
              f"{b['center_xz'][1]:+.1f}) yaw {b['yaw_deg']:+.0f}")
    payload = {"cell": res, "origin_xz": [ox, oz], "nx": nx, "nz": nz,
               "count": len(boxes), "footprint_m2": round(area, 1), "boxes": boxes}
    if not boxes:
        payload["note"] = ("no flat top in the lift band on supported ground; the "
                           "heightfield collider is the whole story for this scene")
    rb.write_json(a / "objects.json", payload)
    print(f"[objects] wrote {a / 'objects.json'}")


def report_no_flat_tops(P, lift, sup, H, args) -> None:
    """Say WHY the band caught nothing, in terms of this scene's own geometry."""
    if not sup.any():
        rb.warn("[objects] the grid has no supported cell, so the lift test has no "
                "floor to measure against")
        return
    relief = rb.safe_max(H[sup], 0.0, label="supported ground max") \
        - rb.safe_min(H[sup], 0.0, label="supported ground min")
    hi = rb.safe_max(lift[np.isfinite(lift)] if lift.size else [], None,
                     label="lift max")
    span = rb.safe_max(P[:, 1], 0.0, label="splat y max") \
        - rb.safe_min(P[:, 1], 0.0, label="splat y min")
    rb.warn(f"[objects] nothing reaches the {args.min_lift:.2f}-{args.max_lift:.2f} m "
            f"lift band. This splat spans {span:.2f} m of height and its floor spans "
            f"{relief:.2f} m of the grid; the highest measured lift is "
            + (f"{hi:.2f} m" if rb.finite(hi) else "undefined")
            + ".")
    if rb.finite(hi) and hi < args.min_lift:
        rb.warn("[objects] that is a scene below human furniture scale (a diorama, a "
                "tabletop, a very sparse solve). Retune with --min-lift/--max-lift; "
                "shipping zero boxes is correct if nothing here is furniture.")


if __name__ == "__main__":
    rb.configure_streams()
    try:
        main()
    except rb.StepError as e:
        print(f"\n[objects] {e}", file=sys.stderr, flush=True)
        sys.exit(e.returncode)
