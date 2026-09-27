"""A workspace scene seen by the operations analyses (Phase 4).

The workspace speaks the viewer frame (x, y up, z; -z is scene north). The survey
modules speak a north-up map raster (x east, y north, z up). This adapter holds the one
conversion between them and everything the analyses need from a scene:

* ``map = (x, -z, y)`` of the viewer; the workspace ground grid is already a north-up
  raster in that frame (row 0 = smallest z = north), so rasters line up cell for cell
  and overlays go straight back onto the scan;
* ``dtm`` = the scene's measured floor (``ground.f32``), ``dsm`` = its top surface
  (``heights.f32``), ``observed`` = coverage; unobserved cells are NaN, never filled;
* ``surface_points`` = one point per observed cell of the top surface, the "cloud" the
  change / damage / cut-fill modules consume (the heightfield every workspace scene has,
  so a real scan and the synthetic test site take the same path);
* georeferenced scenes also convert map points to WGS84 / MGRS / UTM through
  ``scene_frames``; local scenes say so and never invent Earth coordinates.

The map frame is *scene* north: for a georeferenced scene whose viewer frame is rotated
against ENU, bearings on the map are not true bearings - Earth coordinates are still
exact because they go through the registry.
"""
import json
from pathlib import Path

import numpy as np

import plan_shadow
import scene_frames
import workspace_place
import workspace_proposals as proposals

MAX_QUADS = 30_000


class Scene:
    def __init__(self, work):
        self.work = Path(work)
        self.grid = workspace_place._grid(self.work)
        g = self.grid
        self.cell = float(g["cell"])
        self.ox, self.oz = float(g["ox"]), float(g["oz"])
        self.shape = (int(g["nz"]), int(g["nx"]))
        self.transform = (self.ox, self.cell, 0.0, -self.oz, 0.0, -self.cell)
        observed = np.asarray(g["supported"], bool)
        self.observed = observed
        floor = np.asarray(g["floor"], float)
        top = g.get("top")
        top = floor if top is None else np.asarray(top, float)
        top = np.maximum(top, floor)
        self.floor_raw, self.top_raw = floor, top
        self.dtm = np.where(observed, floor, np.nan)
        self.dsm = np.where(observed, top, np.nan)
        try:
            self.registry = scene_frames.load(self.work)
        except (OSError, ValueError):
            self.registry = {"status": "local_relative", "scale_status": "relative"}
        self.georeferenced = self.registry.get("status") == "georeferenced"
        self.metric = self.registry.get("scale_status") == "metric"

    # ---- frames ---------------------------------------------------------------
    @property
    def bounds(self):
        a, b, _, d, _, f = self.transform
        return (a, d + self.shape[0] * f, a + self.shape[1] * b, d)

    @staticmethod
    def viewer_to_map(points):
        p = np.atleast_2d(np.asarray(points, float))
        return np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])

    @staticmethod
    def map_to_viewer(points):
        p = np.atleast_2d(np.asarray(points, float))
        return np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])

    def xz_to_map(self, xz):
        xz = np.atleast_2d(np.asarray(xz, float))
        return np.column_stack([xz[:, 0], -xz[:, 1]])

    def map_to_xz(self, xy):
        xy = np.atleast_2d(np.asarray(xy, float))
        return np.column_stack([xy[:, 0], -xy[:, 1]])

    def height_at_map(self, xy, *, top=True):
        xy = np.atleast_2d(np.asarray(xy, float))
        col = np.clip(((xy[:, 0] - self.ox) / self.cell).astype(int), 0, self.shape[1] - 1)
        row = np.clip(((-xy[:, 1] - self.oz) / self.cell).astype(int), 0, self.shape[0] - 1)
        grid = self.top_raw if top else self.floor_raw
        return grid[row, col]

    def viewer_point(self, xy, *, top=True, lift=0.0):
        """Map (x, y) -> viewer [x, y, z] on the surface."""
        xy = np.atleast_2d(np.asarray(xy, float))
        h = self.height_at_map(xy, top=top) + lift
        return [[round(float(x), 3), round(float(y), 3), round(float(-n), 3)]
                for (x, n), y in zip(xy, h)]

    def where(self, xy):
        """Earth coordinates of a map point, or None for a local scene."""
        if not self.georeferenced:
            return None
        import survey_coords
        v = self.viewer_point(xy)[0]
        lat, lon, h = scene_frames.viewer_to_geodetic([v], self.registry)[0]
        return {"lat": round(float(lat), 7), "lon": round(float(lon), 7), "ellipsoidal_height_m": round(float(h), 2),
                "mgrs": survey_coords.to_mgrs(float(lat), float(lon))}

    def origin_frame(self):
        """survey_measure ENU frame dict for modules that attach MGRS themselves (or None)."""
        return None

    def to_utm(self, map_points):
        """Map (x, y, z) -> UTM (E, N, h) and the CRS dict; refuses on a local scene."""
        if not self.georeferenced:
            raise ValueError("the scene has no GPS fit, so it has no UTM coordinates")
        import survey_crs
        enu = scene_frames.viewer_to_enu(self.map_to_viewer(map_points), self.registry)
        utm, _, crs = survey_crs.enu_to_crs(enu, self.registry["alignment"])
        return utm, crs

    # ---- data -----------------------------------------------------------------
    def surface_points(self, *, top=True):
        rows, cols = np.nonzero(self.observed)
        a, b, _, d, _, f = self.transform
        z = (self.top_raw if top else self.floor_raw)[rows, cols]
        return np.column_stack([a + (cols + 0.5) * b, d + (rows + 0.5) * f, z])

    def labels(self, cls):
        """Map-frame cells carrying the semantic class ``cls`` (from semantics.json)."""
        path = self.work / "viewer_assets" / "semantics.json"
        mask = np.zeros(self.shape, bool)
        if not path.is_file():
            return mask
        import label_semantics
        data = json.loads(path.read_text(encoding="utf-8"))
        rgb = np.asarray(data.get("rgb", []), float).reshape(-1, 3)
        xyz = np.asarray(data.get("coords", []), float).reshape(-1, 3)
        want = np.asarray(label_semantics.CLASS_RGB[cls], float)
        pick = np.all(np.abs(rgb - want) < 1, axis=1)
        if not pick.any():
            return mask
        col = ((xyz[pick, 0] - self.ox) / self.cell).astype(int)
        row = ((xyz[pick, 2] - self.oz) / self.cell).astype(int)
        ok = (row >= 0) & (row < self.shape[0]) & (col >= 0) & (col < self.shape[1])
        mask[row[ok], col[ok]] = True
        from scipy import ndimage
        return ndimage.binary_closing(mask, iterations=2) & self.observed

    def project(self):
        path = self.work / "project.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    # ---- overlays -------------------------------------------------------------
    def overlay(self, layers, *, transform=None, surface=None, budget=MAX_QUADS):
        """Coloured quads over the scan for boolean masks on a map raster.

        ``layers`` = [(mask, [r, g, b], opacity, part)]. ``transform`` defaults to the
        scene grid; a coarser raster on the same origin is fine.
        """
        transform = transform or self.transform
        a, b, _, d, _, _ = transform
        work = {"cell": b, "ox": a, "oz": -d}
        meshes = []
        for mask, color, opacity, part in layers:
            if surface is None:
                ground = self._resample_top(mask.shape, b)
            else:
                ground = np.where(np.isfinite(surface), surface, np.nanmedian(surface))
            mesh, used = plan_shadow._quads(np.asarray(mask, bool), ground, work, color, opacity, part, budget)
            budget -= used
            if mesh:
                meshes.append(mesh)
        return meshes

    def _resample_top(self, shape, cell):
        if shape == self.shape and abs(cell - self.cell) < 1e-9:
            return self.top_raw
        rows = np.clip(((np.arange(shape[0]) + 0.5) * cell / self.cell).astype(int), 0, self.shape[0] - 1)
        cols = np.clip(((np.arange(shape[1]) + 0.5) * cell / self.cell).astype(int), 0, self.shape[1] - 1)
        return self.top_raw[np.ix_(rows, cols)]

    def line_mesh(self, xy, color, *, width=0.8, lift=0.4, part="line"):
        """A ribbon along a map polyline, draped on the top surface."""
        pts = np.asarray(xy, float)
        if len(pts) < 2:
            return None
        positions, indices = [], []
        for p, q in zip(pts[:-1], pts[1:]):
            seg = q - p
            length = float(np.hypot(*seg))
            if length < 1e-6:
                continue
            steps = max(1, int(length / 2.0))
            n = np.array([-seg[1], seg[0]]) / length * width / 2
            for k in range(steps):
                a = p + seg * k / steps
                b = p + seg * (k + 1) / steps
                ya = float(self.height_at_map(a)[0]) + lift
                yb = float(self.height_at_map(b)[0]) + lift
                base = len(positions)
                for (x, y), h in (((a + n)), ya), (((a - n)), ya), (((b - n)), yb), (((b + n)), yb):
                    positions.append([float(x), h, float(-y)])
                indices += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
        if not indices:
            return None
        return proposals._mesh("overlay", positions, indices, color, 0.95, part=part)


def load_epoch(work):
    return Scene(work)


def epoch_points_in(target, source, *, top=True):
    """``source`` scene's surface points expressed in ``target``'s map frame.

    Both georeferenced: through WGS84 (exact). Both local: assumed to share one frame
    (only true for scenes built in one reconstruction or the synthetic pair) - the caller
    reports this. One of each: refused.
    """
    pts = source.surface_points(top=top)
    if target.georeferenced and source.georeferenced:
        viewer = scene_frames.enu_to_viewer(
            scene_frames.viewer_to_enu(source.map_to_viewer(pts), source.registry), target.registry)
        return target.viewer_to_map(viewer), "both epochs georeferenced: aligned through WGS84, then co-registered"
    if target.georeferenced != source.georeferenced:
        raise ValueError("one epoch is georeferenced and the other is not; there is no common frame")
    return pts, ("neither epoch is georeferenced: their frames are assumed identical and only a "
                 "shift is corrected by co-registration (rotation or scale differences are not)")


def viewer_points_between(target, source, points):
    """Viewer-frame points of ``source`` expressed in ``target``'s viewer frame (see epoch_points_in)."""
    pts = np.atleast_2d(np.asarray(points, float))
    if target.georeferenced and source.georeferenced:
        return scene_frames.enu_to_viewer(scene_frames.viewer_to_enu(pts, source.registry), target.registry),             "both epochs georeferenced: aligned through WGS84"
    if target.georeferenced != source.georeferenced:
        raise ValueError("one epoch is georeferenced and the other is not; there is no common frame")
    return pts, "neither epoch is georeferenced: their frames are assumed identical"
