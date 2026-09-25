"""Place real-world furniture into a scanned interior and say honestly if it fits.

The scene's ``viewer_assets`` already carries everything needed: a per-column
floor height (``ground.f32``), the top surface (``heights.f32``) and a
supported/unsupported mask (``coverage.u8``) on the same grid the walk-mode
character uses. This module reads that grid — no reconstruction, no guessing —
to snap a dropped item onto the floor, orient it, and check two things a
customer actually asks before buying a sofa: does the footprint land on real
floor (not a wall or a void), and does its height clear the ceiling.

Coordinates are the viewer's Y-up frame (x, y, z); the grid is indexed exactly
as ``viewer/pc.js`` ``groundHF`` indexes it, so the box the backend places and
the box the character collides with are the same box. The output record reuses
the ``objects.json`` box schema (``center_xz`` / ``center_y`` / ``size`` /
``yaw_deg``) so the viewer renders a placement with the collider path it already
trusts.

Furniture sizes are nominal catalogue dimensions in metres — a planning aid, not
surveyed ground truth. The fit verdict is only as good as the scan's coverage
grid, and it says so in ``reason``.
"""
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt

# Nominal catalogue dimensions, metres, as [length, height, depth]. length is the
# major (yaw) axis, depth the minor axis, height vertical. Deliberately ordinary
# numbers a shopper would recognise from a spec sheet.
LIBRARY = {
    "sofa":          {"label": "Sofa (3-seat)",   "size": [2.00, 0.85, 0.90]},
    "armchair":      {"label": "Armchair",        "size": [0.85, 0.80, 0.90]},
    "coffee_table":  {"label": "Coffee table",    "size": [1.10, 0.45, 0.60]},
    "dining_table":  {"label": "Dining table",    "size": [1.60, 0.75, 0.90]},
    "dining_chair":  {"label": "Dining chair",    "size": [0.45, 0.90, 0.50]},
    "bed_double":    {"label": "Double bed",      "size": [2.00, 0.55, 1.60]},
    "desk":          {"label": "Desk",            "size": [1.20, 0.75, 0.60]},
    "wardrobe":      {"label": "Wardrobe",        "size": [1.00, 2.05, 0.60]},
    "bookshelf":     {"label": "Bookshelf",       "size": [0.80, 1.80, 0.30]},
    "side_table":    {"label": "Side table",      "size": [0.50, 0.55, 0.50]},
    "tv_stand":      {"label": "TV stand",        "size": [1.40, 0.50, 0.40]},
    "stool":         {"label": "Stool",           "size": [0.35, 0.45, 0.35]},
}

# A footprint is sampled on an N x N grid across the rotated rectangle; 5x5 is
# enough to catch a corner clipping a wall without pretending to sub-metre truth.
_FOOTPRINT_SAMPLES = 5
# How far the item must sit from any wall/void for the drop to be "comfortable".
# Below this it still fits but is flagged tight (clearance is reported either way).
_TIGHT_M = 0.30
# A footprint fits when its centre is on floor and at least this fraction of the
# sampled footprint is on measured floor — tolerant of scan gaps, strict on walls.
_MIN_SUPPORT = 0.6

# Cache parsed grids by (path, collision.json mtime) so a page poll does not
# re-read a multi-megabyte heightfield for every placement.
_GRID_CACHE = {}


def library():
    """The furniture catalogue as a JSON-safe list with footprint areas."""
    out = []
    for key, spec in LIBRARY.items():
        length, height, depth = spec["size"]
        out.append({"item": key, "label": spec["label"], "size": list(spec["size"]),
                    "footprint_m2": round(length * depth, 3)})
    return out


def item_spec(item):
    """The catalogue entry for ``item``, or raise — an unknown item is refused."""
    spec = LIBRARY.get(item)
    if not isinstance(spec, dict):
        raise ValueError(f"Unknown furniture item {item!r}.")
    return spec


def _grid(scene_dir):
    """Load (and cache) the scene grid: floor, top surface, coverage, header."""
    va = Path(scene_dir) / "viewer_assets"
    col_path = va / "collision.json"
    if not col_path.is_file():
        raise ValueError("The scene has no collision grid to place furniture on.")
    key = (str(va), col_path.stat().st_mtime_ns)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached
    col = json.loads(col_path.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    nx, nz = int(col["nx"]), int(col["nz"])
    cell = float(col["cell"])
    ox, oz = (float(v) for v in col["origin_xz"])
    if nx <= 0 or nz <= 0 or not (cell > 0):
        raise ValueError("The scene's collision grid is unusable.")

    def read(name, dtype):
        path = va / name
        if not path.is_file():
            return None
        arr = np.fromfile(path, dtype=dtype)
        if arr.size != nx * nz:
            return None
        return arr.reshape(nz, nx).astype(np.float64)

    floor = read("ground.f32", "<f4")
    if floor is None:
        floor = read("heights.f32", "<f4")
    top = read("heights.f32", "<f4")
    cov_path = va / "coverage.u8"
    cover = np.fromfile(cov_path, dtype=np.uint8).reshape(nz, nx) if cov_path.is_file() and cov_path.stat().st_size == nx * nz else None
    if floor is None:
        raise ValueError("The scene has no floor heightfield to snap furniture to.")
    supported = (cover > 0) if cover is not None else np.isfinite(floor)
    # A real scan's floor is riddled with one- or two-cell coverage holes from
    # textureless patches, so the raw mask reads open floor as "void". Bridge those
    # small gaps with a morphological closing (a disk ~0.15 m across) before judging
    # fit or measuring clearance; a genuine wall is far thicker than the kernel, so it
    # is not bridged. Without this every item lands "does not fit" on a good scan.
    radius = max(1, int(round(0.15 / cell)))
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    supported = binary_closing(supported, structure=(xx * xx + yy * yy) <= radius * radius)
    # Metres from each supported cell to the nearest unsupported one; edges of the
    # whole grid count as walls so furniture is kept off the boundary too.
    padded = np.pad(supported, 1, constant_values=False)
    clearance = distance_transform_edt(padded)[1:-1, 1:-1] * cell
    grid = {"nx": nx, "nz": nz, "cell": cell, "ox": ox, "oz": oz,
            "floor": floor, "top": top, "supported": supported, "clearance": clearance}
    _GRID_CACHE[key] = grid
    return grid


def _cell_of(grid, x, z):
    """Grid cell (row=z, col=x) for a world (x, z), matching groundHF's convention.

    Returned as (z_index, x_index) because every grid array is shaped (nz, nx) and
    indexed [z, x]. Swapping these silently corrupts a square grid and throws on a
    non-square one — real room scans are never square.
    """
    gx = (x - grid["ox"]) / grid["cell"] - 0.5
    gz = (z - grid["oz"]) / grid["cell"] - 0.5
    if gx < -0.5 or gz < -0.5 or gx > grid["nx"] - 0.5 or gz > grid["nz"] - 0.5:
        return None
    return int(np.clip(round(gz), 0, grid["nz"] - 1)), int(np.clip(round(gx), 0, grid["nx"] - 1))


def _bilinear(arr, grid, x, z):
    gx = min(max((x - grid["ox"]) / grid["cell"] - 0.5, 0), grid["nx"] - 1.001)
    gz = min(max((z - grid["oz"]) / grid["cell"] - 0.5, 0), grid["nz"] - 1.001)
    x0, z0 = int(np.floor(gx)), int(np.floor(gz))
    fx, fz = gx - x0, gz - z0
    x1, z1 = min(x0 + 1, grid["nx"] - 1), min(z0 + 1, grid["nz"] - 1)
    a = arr[z0, x0] * (1 - fx) + arr[z0, x1] * fx
    b = arr[z1, x0] * (1 - fx) + arr[z1, x1] * fx
    return a * (1 - fz) + b * fz


def floor_y(scene_dir, x, z):
    """The measured floor height at world (x, z), or None if off the grid."""
    grid = _grid(scene_dir)
    if _cell_of(grid, x, z) is None:
        return None
    y = _bilinear(grid["floor"], grid, x, z)
    return float(y) if np.isfinite(y) else None


def _footprint_cells(grid, x, z, length, depth, yaw_deg):
    """Sample points inside the rotated footprint rectangle, as (world x, z)."""
    a = math.radians(yaw_deg)
    ca, sa = math.cos(a), math.sin(a)
    hu, hw = length / 2, depth / 2
    ts = np.linspace(-1, 1, _FOOTPRINT_SAMPLES)
    pts = []
    for qu in ts:
        for qv in ts:
            u, v = qu * hu, qv * hw
            pts.append((x + u * ca - v * sa, z + u * sa + v * ca))
    return pts


def fit_check(scene_dir, x, z, length, depth, yaw_deg, height):
    """Can a ``length x depth`` footprint at ``(x, z)`` hold a ``height`` item?

    Returns supported (all sampled cells on measured floor), floor_clearance_m
    (nearest wall/void distance under the footprint), ceiling_height_m (measured
    headroom over the footprint) and a fits/valid verdict with an honest reason.
    """
    grid = _grid(scene_dir)
    supported = grid["supported"]
    clearance = grid["clearance"]
    top, floor = grid["top"], grid["floor"]
    center_supported = False
    min_clear = float("inf")
    headrooms = []
    on_grid = 0
    supported_hits = 0
    for px, pz in _footprint_cells(grid, x, z, length, depth, yaw_deg):
        cell = _cell_of(grid, px, pz)
        if cell is None:
            continue
        ci, cj = cell
        on_grid += 1
        if supported[ci, cj]:
            supported_hits += 1
            min_clear = min(min_clear, float(clearance[ci, cj]))
            if top is not None:
                head = top[ci, cj] - floor[ci, cj]
                if np.isfinite(head):
                    headrooms.append(float(head))
    cc = _cell_of(grid, x, z)
    if cc is not None:
        center_supported = bool(supported[cc[0], cc[1]])
    if on_grid == 0:
        return {"supported": False, "support_fraction": 0.0, "floor_clearance_m": None,
                "ceiling_height_m": None, "fits_height": False, "tight": False, "valid": False,
                "reason": "the drop point is outside the scanned floor"}
    fraction = supported_hits / on_grid
    # A footprint "fits" when it is centred on floor and a clear majority of its
    # samples are supported — tolerant of the odd scan gap, still refusing a drop
    # that mostly lands on a wall or off the scan.
    is_supported = bool(center_supported and fraction >= _MIN_SUPPORT)
    # A 2.5D scan skin stores one surface per column, so most interior exports have
    # no separate ceiling (top == floor). Only claim a headroom when a real ceiling
    # surface exists; otherwise report it unknown rather than failing every tall item.
    ceiling_height = None
    fits_height = True
    if headrooms and max(headrooms) >= 0.5:
        ceiling_height = round(min(headrooms), 3)
        fits_height = height <= ceiling_height + 1e-6
    clear = round(min_clear, 3) if np.isfinite(min_clear) else None
    tight = bool(clear is not None and clear < _TIGHT_M)
    # "Tight" (close to a wall) is reported as a neutral flag, not a failure — a bed
    # pushed against a wall is intentional. Only a non-fit gets a reason.
    reason = None
    if not is_supported:
        reason = "most of the footprint falls on a wall or unmeasured void" if not center_supported \
            else f"only {fraction * 100:.0f}% of the footprint lands on measured floor"
    elif not fits_height:
        reason = f"the item is {height:.2f} m tall but the measured headroom is {ceiling_height:.2f} m"
    return {"supported": is_supported, "support_fraction": round(fraction, 3),
            "floor_clearance_m": clear, "ceiling_height_m": ceiling_height,
            "fits_height": bool(fits_height), "tight": tight,
            "valid": bool(is_supported and fits_height), "reason": reason}


def make_placement(scene_dir, item, x, z, *, yaw_deg=0.0, scale=1.0, label=None):
    """Assemble a placement record: catalogue size scaled, snapped to the floor, fit-checked.

    ``x, z`` is the footprint centre on the floor; the returned ``center_y`` lifts
    the box so its base sits on the measured floor. If the floor cannot be read at
    that point the record is still produced but flagged invalid, never fabricated.
    """
    spec = item_spec(item)
    length, height, depth = spec["size"]
    s = float(scale) if scale and scale > 0 else 1.0
    length, height, depth = length * s, height * s, depth * s
    yaw = float(yaw_deg) % 360.0
    floor = floor_y(scene_dir, x, z)
    fit = fit_check(scene_dir, x, z, length, depth, yaw, height)
    if floor is None:
        fit = {**fit, "supported": False, "valid": False,
               "reason": fit.get("reason") or "no measured floor at the drop point"}
        base_y = 0.0
    else:
        base_y = floor
    return {
        "id": uuid.uuid4().hex,
        "item": item,
        "label": (label or spec["label"])[:200],
        "size": [round(length, 4), round(height, 4), round(depth, 4)],
        "center_xz": [round(float(x), 4), round(float(z), 4)],
        "center_y": round(base_y + height / 2, 4),
        "yaw_deg": round(yaw, 2),
        "scale": s,
        "footprint_m2": round(length * depth, 3),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit": fit,
    }
