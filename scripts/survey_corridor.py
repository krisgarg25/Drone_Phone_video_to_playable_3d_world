"""Corridor products for border and linear-asset flights (BOR-02, BOR-04).

* ``tile_products``: cut a long corridor into fixed ground tiles (default 1 km) and write
  a DSM GeoTIFF + LAS per tile plus ``tiles.json``. Each tile is processed on its own, so
  raster memory is bounded by the tile, not the corridor; the cloud itself is still held
  in memory (streaming the cloud from disk is not done here). Tiles are plain GeoTIFF,
  not Cloud-Optimised (no overviews / internal tiling), and LAS, not COPC-LAZ.
* ``profile``: heights sampled along a polyline from a raster, with an ``observed`` flag,
  so a profile across an unseen gully is drawn as a gap, not a straight line.
* ``blind_spots``: which stretches of a line (a fence, the LoC trace) each observation
  post can see over the DSM, within range and field of view; merged coverage and the
  list of blind stretches with their chainage.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage


def _bilinear(grid, transform, xy):
    a, b, _, d, _, f = transform
    col = (xy[:, 0] - a) / b - 0.5
    row = (xy[:, 1] - d) / f - 0.5
    valid = np.isfinite(grid)
    z = ndimage.map_coordinates(np.where(valid, grid, 0.0), [row, col], order=1,
                                mode="nearest")
    w = ndimage.map_coordinates(valid.astype(float), [row, col], order=1, mode="constant",
                                cval=0.0)
    return np.where(w > 0.999, z, np.nan)


def _densify(line, step):
    line = np.asarray(line, dtype=np.float64)[:, :2]
    if len(line) < 2:
        raise ValueError("a line needs at least two vertices")
    seg = np.diff(line, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    chain = np.r_[0.0, np.cumsum(length)]
    s = np.arange(0.0, chain[-1] + 1e-9, step)
    if s[-1] < chain[-1]:
        s = np.r_[s, chain[-1]]
    x = np.interp(s, chain, line[:, 0])
    y = np.interp(s, chain, line[:, 1])
    return s, np.column_stack([x, y])


def profile(grid, transform, line, *, step_m=1.0, observed=None):
    """Chainage, x, y, z along ``line``; z is NaN where the raster has no value."""
    s, xy = _densify(line, step_m)
    z = _bilinear(np.asarray(grid, float), transform, xy)
    seen = np.isfinite(z)
    if observed is not None:
        a, b, _, d, _, f = transform
        rr = np.clip(((xy[:, 1] - d) / f).astype(int), 0, observed.shape[0] - 1)
        cc = np.clip(((xy[:, 0] - a) / b).astype(int), 0, observed.shape[1] - 1)
        seen &= np.asarray(observed, bool)[rr, cc]
    ok = np.isfinite(z)
    climb = float(np.clip(np.diff(z[ok]), 0, None).sum()) if ok.sum() > 1 else 0.0
    return {"chainage_m": s, "xy": xy, "z": z, "observed": seen,
            "summary": {"length_m": round(float(s[-1]), 2),
                        "min_z": None if not ok.any() else round(float(np.nanmin(z)), 2),
                        "max_z": None if not ok.any() else round(float(np.nanmax(z)), 2),
                        "climb_m": round(climb, 2),
                        "observed_fraction": round(float(seen.mean()), 4)}}


def _visible(grid, transform, eye, targets, *, samples_per_m=1.0, clearance_m=0.1):
    """True where the straight line eye -> target clears the surface."""
    out = np.zeros(len(targets), dtype=bool)
    for i, target in enumerate(targets):
        dist = float(np.hypot(*(target[:2] - eye[:2])))
        n = max(2, int(dist * samples_per_m))
        t = np.linspace(0.0, 1.0, n)[1:-1]
        if len(t) == 0:
            out[i] = True
            continue
        pts = eye[None, :] + t[:, None] * (target - eye)[None, :]
        ground = _bilinear(grid, transform, pts[:, :2])
        # Unobserved ground along the ray is treated as blocking: we cannot claim sight
        # across terrain we never saw.
        blocked = ~np.isfinite(ground) | (ground > pts[:, 2] - clearance_m)
        out[i] = not blocked.any()
    return out


def blind_spots(dsm, transform, line, posts, *, step_m=5.0, target_height_m=1.7,
                min_blind_m=10.0):
    """Coverage of ``line`` by ``posts`` over ``dsm``.

    ``posts``: list of dicts {id, x, y, height_m (eye above ground), range_m,
    bearing_deg (0 = +y north, clockwise; optional), fov_deg (optional, 360 default)}.
    """
    dsm = np.asarray(dsm, dtype=np.float64)
    s, xy = _densify(line, step_m)
    ground = _bilinear(dsm, transform, xy)
    targets = np.column_stack([xy, ground + target_height_m])
    seen_by = np.zeros((len(posts), len(s)), dtype=bool)
    report_posts = []
    for k, post in enumerate(posts):
        base = _bilinear(dsm, transform, np.array([[post["x"], post["y"]]], float))[0]
        if not np.isfinite(base):
            report_posts.append(dict(id=post.get("id", f"P{k + 1}"), error="post is off the DSM"))
            continue
        eye = np.array([post["x"], post["y"], base + float(post.get("height_m", 2.0))])
        vec = xy - eye[:2]
        dist = np.hypot(vec[:, 0], vec[:, 1])
        candidate = (dist <= float(post.get("range_m", 1000.0))) & np.isfinite(ground)
        fov = float(post.get("fov_deg", 360.0))
        if fov < 360.0 and post.get("bearing_deg") is not None:
            bearing = np.degrees(np.arctan2(vec[:, 0], vec[:, 1])) % 360.0
            off = np.abs((bearing - float(post["bearing_deg"]) + 180.0) % 360.0 - 180.0)
            candidate &= off <= fov / 2.0
        idx = np.nonzero(candidate)[0]
        seen_by[k, idx] = _visible(dsm, transform, eye, targets[idx])
        report_posts.append(dict(id=post.get("id", f"P{k + 1}"),
                                 covered_m=round(float(seen_by[k].sum() * step_m), 1),
                                 eye_z=round(float(eye[2]), 2)))
    covered = seen_by.any(0)
    unknown = ~np.isfinite(ground)
    stretches, start = [], None
    for i in range(len(s) + 1):
        blind = i < len(s) and not covered[i]
        if blind and start is None:
            start = i
        if not blind and start is not None:
            a, b = s[start], s[i - 1] + (step_m if i - 1 < len(s) - 1 else 0.0)
            length = b - a
            if length >= min_blind_m:
                stretches.append(dict(from_m=round(float(a), 1), to_m=round(float(min(b, s[-1])), 1),
                                      length_m=round(float(min(b, s[-1]) - a), 1),
                                      unobserved_ground=bool(unknown[start:i].any()),
                                      centre_xy=[round(float(v), 2) for v in xy[(start + i - 1) // 2]]))
            start = None
    return {"chainage_m": s, "seen_by": seen_by, "covered": covered,
            "summary": {"line_length_m": round(float(s[-1]), 1),
                        "covered_fraction": round(float(covered.mean()), 4),
                        "posts": report_posts, "blind_stretches": stretches,
                        "target_height_m": target_height_m, "step_m": step_m,
                        "notes": ["Sight is straight-line over the DSM (trees count as opaque).",
                                  "Ground never observed along a ray blocks it; such stretches "
                                  "are marked unobserved_ground."]}}


def write_cog(grid, path, *, transform, crs_wkt, nodata=-9999.0):
    """Cloud-Optimised GeoTIFF through GDAL's COG driver (tiles, overviews, deflate); None without GDAL."""
    try:
        import rasterio
        import rasterio.shutil
        from rasterio.io import MemoryFile
        from rasterio.transform import Affine
    except ImportError:
        return None
    import re
    from rasterio.crs import CRS
    try:
        crs = CRS.from_wkt(crs_wkt)
    except Exception:  # noqa: BLE001 - GDAL refuses partial WKT; fall back to its EPSG code
        code = re.findall(r'AUTHORITY\["EPSG",\s*"(\d+)"\]', crs_wkt or "")
        if not code:
            return None
        crs = CRS.from_epsg(int(code[-1]))
    a, b, c, d, e, f = transform
    data = np.where(np.isfinite(grid), grid, nodata).astype(np.float32)
    profile = dict(driver="GTiff", width=data.shape[1], height=data.shape[0], count=1, dtype="float32",
                   crs=crs, transform=Affine(b, c, a, e, f, d), nodata=nodata)
    with MemoryFile() as mem:
        with mem.open(**profile) as tmp:
            tmp.write(data, 1)
        with mem.open() as src:
            rasterio.shutil.copy(src, str(path), driver="COG", COMPRESS="DEFLATE", BLOCKSIZE=256, OVERVIEWS="AUTO")
    with rasterio.open(path) as check:
        layout = check.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")
        return {"layout": layout, "overviews": check.overviews(1), "blocksize": check.block_shapes[0]}


def tile_products(enu, out_dir, *, tile_m=1000.0, cell_m=0.5, crs_wkt=None, offset=(0.0, 0.0),
                  colors=None, min_points=100):
    """Write ``tile_<i>_<j>/dsm.tif`` + ``points.las`` and ``tiles.json``.

    ``offset`` is added to x/y before writing (e.g. the UTM coordinates of the ENU origin)
    so the tiles land in the declared CRS.
    """
    import survey_change
    import survey_formats
    enu = np.asarray(enu, dtype=np.float64)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ox, oy = offset
    world = enu[:, :2] + np.array([ox, oy])
    ti = np.floor(world[:, 0] / tile_m).astype(np.int64)
    tj = np.floor(world[:, 1] / tile_m).astype(np.int64)
    keys = np.unique(np.column_stack([ti, tj]), axis=0)
    index = []
    for i, j in keys.tolist():
        pick = (ti == i) & (tj == j)
        if pick.sum() < min_points:
            index.append(dict(tile=f"{i}_{j}", skipped=f"{int(pick.sum())} points"))
            continue
        bounds = (i * tile_m, j * tile_m, (i + 1) * tile_m, (j + 1) * tile_m)
        pts = np.column_stack([world[pick], enu[pick, 2]])
        grid = survey_change.grid_surface(pts, cell_m=cell_m, bounds=bounds)
        folder = out_dir / f"tile_{i}_{j}"
        folder.mkdir(exist_ok=True)
        entry = dict(tile=f"{i}_{j}", bounds=list(bounds), points=int(pick.sum()),
                     observed_fraction=round(float(np.isfinite(grid["zmax"]).mean()), 4))
        if crs_wkt:
            survey_formats.write_geotiff(grid["zmax"].astype(np.float32), folder / "dsm.tif",
                                         transform=grid["transform"], crs_wkt=crs_wkt,
                                         nodata=-9999.0)
            cog = write_cog(grid["zmax"], folder / "dsm_cog.tif", transform=grid["transform"], crs_wkt=crs_wkt)
            if cog:
                entry["cog"] = cog
            cols = dict(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2])
            if colors is not None:
                c = np.asarray(colors)[pick]
                cols.update(red=c[:, 0], green=c[:, 1], blue=c[:, 2])
            survey_formats.write_las(cols, folder / "points.las", scale=(0.001,) * 3,
                                     offsets=(bounds[0], bounds[1], 0.0), srs_wkt=crs_wkt)
            entry["files"] = ["dsm.tif", "points.las"] + (["dsm_cog.tif"] if entry.get("cog") else [])
        else:
            np.save(folder / "dsm.npy", grid["zmax"].astype(np.float32))
            entry["files"] = ["dsm.npy"]
            entry["note"] = "no CRS given: local grid, not a GeoTIFF"
        index.append(entry)
    manifest = dict(tile_m=tile_m, cell_m=cell_m, crs=bool(crs_wkt), tiles=index,
                    notes=["dsm.tif is the plain GeoTIFF our own reader verifies; dsm_cog.tif is the same raster as a "
                           "Cloud-Optimised GeoTIFF (GDAL COG driver: internal tiles + overviews) when GDAL is present.",
                           "Points are LAS 1.4, not COPC-LAZ: no LAZ compressor is installed here."])
    (out_dir / "tiles.json").write_text(json.dumps(manifest, indent=2))
    return manifest
