"""Building damage grade from a single post-event cloud (DIS-02). Heuristic, and says so.

A standing roof - flat, gable or hip - is locally planar and stands at its full height
over its whole footprint. A collapse is low, rough rubble; a partial collapse keeps part
of the roof and adds rubble. Per building this module measures, on a north-up grid:

* ``observed``: footprint cells that hold any point (below 50% -> grade ``unknown``);
* ``roof_cover``: observed cells standing more than ``min_height_m`` above the DTM;
* ``rough``: roof cells whose 3x3 least-squares plane residual exceeds ``rough_m`` (a
  ridge line is one row of cells; rubble is everywhere);
* ``height_m``: 90th percentile nDSM; with a pre-event reference height (survey, OSM,
  earlier DSM) also ``height_ratio``.

Grades (EMS-98 has five; three are all a 2.5D surface can separate honestly):
``collapsed`` - roof_cover < 0.3, or height_ratio < 0.5;
``partial``   - roof_cover < 0.75, rough > 0.3, or height_ratio < 0.8;
``intact``    - otherwise (no visible roof damage; walls and interiors are not seen);
``unknown``   - under half the footprint observed.

Footprints should come from *before* the event (cadastre, OSM, earlier flight): a building
that collapsed flat leaves no nDSM blob to find. Without them, footprints are derived from
the post-event nDSM and a flattened building is missed - the report says which was used.
"""
import math

import numpy as np
from scipy import ndimage

GRADES = ("intact", "partial", "collapsed", "unknown")


def plane_residual(z, cell_m):
    """RMS residual of a least-squares plane over each 3x3 window (NaN-aware)."""
    valid = np.isfinite(z)
    zf = np.where(valid, z, 0.0)
    k = np.ones((3, 3))
    n = ndimage.correlate(valid.astype(float), k, mode="constant")
    s = ndimage.correlate(zf, k, mode="constant")
    s2 = ndimage.correlate(zf * zf, k, mode="constant")
    kx = np.array([[-1, 0, 1]] * 3, float) * cell_m
    ky = kx.T
    sx = ndimage.correlate(zf, kx, mode="constant")
    sy = ndimage.correlate(zf, ky, mode="constant")
    # For a full 3x3 window sum(x^2) = sum(y^2) = 6 cell^2 and x, y, 1 are orthogonal.
    full = n == 9
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / 9.0
        gx, gy = sx / (6 * cell_m ** 2), sy / (6 * cell_m ** 2)
        var = s2 / 9.0 - mean ** 2 - (gx ** 2 + gy ** 2) * (6 * cell_m ** 2) / 9.0
    return np.where(full & valid, np.sqrt(np.maximum(var, 0.0)), np.nan)


def _footprints_from_ndsm(ndsm, cell_m, min_height_m, min_area_m2):
    tall = np.nan_to_num(ndsm, nan=0.0) > min_height_m
    tall = ndimage.binary_opening(tall, np.ones((3, 3)))
    labels, count = ndimage.label(tall, np.ones((3, 3)))
    masks = []
    for index in range(1, count + 1):
        mask = labels == index
        if mask.sum() * cell_m * cell_m < min_area_m2:
            continue
        rr, cc = np.nonzero(mask)
        long_side = cell_m * max(np.ptp(rr), np.ptp(cc)) + cell_m
        # Walls, embankments and ridges stand above the DTM too; a footprint more than
        # 150 m long, or eight times longer than it is wide, is a linear feature, not a
        # building to grade.
        width = mask.sum() * cell_m * cell_m / long_side
        if long_side > 150.0 or long_side > 8.0 * width:
            continue
        masks.append(ndimage.binary_dilation(mask, iterations=1))
    return masks


def footprints_from(enu, *, cell_m=0.5, min_height_m=2.5, min_area_m2=25.0, max_window_m=24.0):
    """Building footprints (convex hulls) and heights from a *pre-event* cloud.

    Feeding these to ``assess`` on the post-event cloud is what lets a building that
    collapsed flat be found at all: its footprint comes from before, its state from after.
    """
    import survey_terrain
    from scipy.spatial import ConvexHull
    terrain = survey_terrain.classify_ground(np.asarray(enu, float), cell_m=cell_m,
                                             max_window_m=max_window_m)
    ndsm, transform = terrain["ndsm"].astype(float), terrain["transform"]
    a, b, _, d, _, f = transform
    rings, heights = {}, {}
    for i, mask in enumerate(_footprints_from_ndsm(ndsm, cell_m, min_height_m, min_area_m2)):
        rr, cc = np.nonzero(mask)
        corners = []
        for dr in (0.0, 1.0):
            for dc in (0.0, 1.0):
                corners.append(np.column_stack([a + (cc + dc) * b, d + (rr + dr) * f]))
        pts = np.vstack(corners)
        hull = ConvexHull(pts)
        key = f"B{i + 1}"
        rings[key] = pts[hull.vertices].round(3).tolist()
        tall = mask & (np.nan_to_num(ndsm) > min_height_m)
        heights[key] = round(float(np.percentile(ndsm[tall], 90)), 2) if tall.any() else None
    return rings, heights


def _mask_of(ring, transform, shape):
    from matplotlib.path import Path as MplPath
    a, b, _, d, _, f = transform
    rows, cols = shape
    cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
    centres = np.column_stack([a + (cc.ravel() + 0.5) * b, d + (rr.ravel() + 0.5) * f])
    return MplPath(np.asarray(ring, float)[:, :2]).contains_points(centres).reshape(rows, cols)


def grade(observed, roof_cover, rough, height_ratio=None):
    if observed < 0.5:
        return "unknown", f"only {observed:.0%} of the footprint observed"
    if roof_cover < 0.3:
        return "collapsed", f"roof stands over {roof_cover:.0%} of the footprint"
    if height_ratio is not None and height_ratio < 0.5:
        return "collapsed", f"height is {height_ratio:.0%} of the reference"
    reasons = []
    if roof_cover < 0.75:
        reasons.append(f"roof over {roof_cover:.0%} of the footprint")
    if rough > 0.3:
        reasons.append(f"{rough:.0%} of the roof is rough (not planar)")
    if height_ratio is not None and height_ratio < 0.8:
        reasons.append(f"height is {height_ratio:.0%} of the reference")
    if reasons:
        return "partial", "; ".join(reasons)
    return "intact", "full-height planar roof over the footprint"


def assess(enu, *, footprints=None, reference_heights=None, cell_m=0.5, min_height_m=2.0,
           rough_m=0.15, min_area_m2=25.0, max_window_m=24.0, frame=None):
    """Grade buildings in an ENU cloud. ``footprints``: {id: ring (x, y)} or None."""
    import survey_terrain
    enu = np.asarray(enu, dtype=np.float64)
    terrain = survey_terrain.classify_ground(enu, cell_m=cell_m, max_window_m=max_window_m)
    ndsm, dsm, transform = terrain["ndsm"].astype(float), terrain["dsm"].astype(float), terrain["transform"]
    residual = plane_residual(dsm, cell_m)
    if footprints:
        items = [(str(k), _mask_of(v, transform, ndsm.shape)) for k, v in footprints.items()]
        source = "given (pre-event)"
    else:
        masks = _footprints_from_ndsm(ndsm, cell_m, min_height_m, min_area_m2)
        items = [(f"B{i + 1}", m) for i, m in enumerate(masks)]
        source = "derived from the post-event nDSM: flattened buildings are not found"
    a, b, _, d, _, f = transform
    area = cell_m * cell_m
    buildings, counts = [], {g: 0 for g in GRADES}
    for key, mask in items:
        total = int(mask.sum())
        if total == 0:
            buildings.append(dict(id=key, grade="unknown", reason="footprint outside the scan"))
            counts["unknown"] += 1
            continue
        seen = mask & np.isfinite(ndsm)
        observed = float(seen.sum() / total)
        roof = seen & (ndsm > min_height_m)
        roof_cover = float(roof.sum() / max(seen.sum(), 1))
        judged = roof & np.isfinite(residual)
        rough = float((residual[judged] > rough_m).mean()) if judged.any() else 0.0
        height = float(np.percentile(ndsm[roof], 90)) if roof.any() else 0.0
        ref = None if reference_heights is None else reference_heights.get(key)
        ratio = None if not ref else height / float(ref)
        label, why = grade(observed, roof_cover, rough, ratio)
        rr, cc = np.nonzero(mask)
        centre = [round(float(a + (cc.mean() + 0.5) * b), 2), round(float(d + (rr.mean() + 0.5) * f), 2)]
        entry = dict(id=key, grade=label, reason=why, footprint_m2=round(total * area, 1),
                     observed=round(observed, 3), roof_cover=round(roof_cover, 3),
                     rough=round(rough, 3), height_m=round(height, 2),
                     reference_height_m=ref,
                     height_ratio=None if ratio is None else round(ratio, 3),
                     centre_enu=centre)
        if frame is not None:
            import survey_change
            entry.update(survey_change._where(centre, frame))
        buildings.append(entry)
        counts[label] += 1
    return {"buildings": buildings, "counts": counts, "footprints": source,
            "method": "roof cover, 3x3 plane residual and height vs reference on the nDSM",
            "parameters": dict(cell_m=cell_m, min_height_m=min_height_m, rough_m=rough_m),
            "notes": ["Heuristic triage, not an engineering assessment: walls, interiors and "
                      "soft-storey failures under an intact roof are not visible.",
                      "Grades follow roof geometry only; confirm on the ground."],
            "grids": {"residual": residual.astype(np.float32), "ndsm": ndsm.astype(np.float32),
                      "transform": transform}}
