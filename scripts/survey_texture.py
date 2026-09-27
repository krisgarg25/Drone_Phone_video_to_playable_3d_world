"""Bake the source video's frames into a UV texture for a reconstructed mesh (G1).

Until now every mesh this project delivered carried per-vertex colour at best
(``claims_textured_mesh: False``): a 17k-face surface over a 100 m site gets ~8k
colours, one per vertex, and everything between them is a smear. The frames that
built the mesh hold far more detail than the geometry does, so this module projects
them back onto it - the same idea as every photogrammetry package's texturing stage:

1. **Visibility.** Each registered camera renders a coarse depth buffer of the mesh
   (faces splatted as barycentric point lattices sized to their on-screen area), and a
   face counts as seen by a view only if it is in front of that buffer, fully inside
   the frame, and faces the camera at a usable angle.
2. **View selection.** Each face takes the view that sees it largest and most
   head-on (``|cos| * projected area``, the data term of Waechter et al. 2014 without
   the photo-consistency part), then two passes of neighbour-majority smoothing trade
   a sliver of resolution for fewer seams.
3. **Atlas.** Faces are paired into equal square cells of a single texture, each
   face one half of a cell, with the cell edge sized to the 95th-percentile on-screen
   face so the atlas is no larger than the frames can fill.
4. **Sampling.** Every texel is mapped to its 3D point by barycentrics, projected
   through the chosen camera *including its lens distortion*, and bilinearly
   sampled from the original frame. Padding texels clamp to their face's edge, so
   mip-mapping in a viewer does not bleed a neighbour's colour across a seam.

What it does not do, and says so in its report: no global colour adjustment between
views (a seam between two differently exposed frames stays visible), no
photo-consistency test (a car that moved between frames can be painted onto the
road), and a face no camera saw cleanly keeps its vertex colour or neutral grey and is
counted, never filled in from a guess.

Camera convention is COLMAP's: ``x_cam = R @ X + t`` with R, t from images.txt/.bin,
pixels from cameras.txt/.bin. The mesh must be in the same frame as that model.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

MAX_ATLAS_PX = 8192
MIN_CELL_PX, MAX_CELL_PX = 6, 64
DEPTH_BUFFER_WIDTH = 480
MIN_COS = 0.15
"""A face seen at more than ~81 degrees from its normal is smeared across a few pixels, so
such a view only wins when nothing better sees the face (GRAZING_WEIGHT)."""
GRAZING_COS, GRAZING_WEIGHT = 0.02, 0.05
"""Below ~89 degrees a face is edge-on and refused outright. Between that and MIN_COS a
view scores at 5%: on a sparse Delaunay mesh the steep sliver faces are otherwise left
grey (measured on rocks: 73% of faces textured with a hard 81-degree cut, 94% without),
and a smeared sample of the right place beats a flat grey. The report counts them."""
DEPTH_TOLERANCE = 0.02
EDGE_MARGIN_PX = 2.0
SMOOTHING_PASSES = 2

# COLMAP camera models this module can project through, with their parameter names.
CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
                 3: ("RADIAL", 5), 4: ("OPENCV", 8)}
_MODEL_IDS = {name: number for number, (name, _count) in CAMERA_MODELS.items()}


# ------------------------------------------------------------------------ COLMAP model
def _quat_to_matrix(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])


def read_model(model_dir):
    """(cameras, images) from a COLMAP sparse model folder, text or binary."""
    model_dir = Path(model_dir)
    if (model_dir / "cameras.txt").is_file() and (model_dir / "images.txt").is_file():
        return _read_text(model_dir)
    if (model_dir / "cameras.bin").is_file() and (model_dir / "images.bin").is_file():
        return _read_binary(model_dir)
    raise ValueError(f"{model_dir} holds no COLMAP cameras/images (.txt or .bin)")


def _camera(model, width, height, params):
    if model not in _MODEL_IDS:
        raise ValueError(f"camera model {model} is not supported for texturing; supported: "
                         + ", ".join(_MODEL_IDS))
    return {"model": model, "width": int(width), "height": int(height),
            "params": [float(v) for v in params]}


def _read_text(model_dir):
    cameras = {}
    for line in (model_dir / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            fields = line.split()
            cameras[int(fields[0])] = _camera(fields[1], fields[2], fields[3], fields[4:])
    images = []
    lines = [line for line in (model_dir / "images.txt").read_text(encoding="utf-8").splitlines()
             if not line.startswith("#")]
    for header in lines[0::2]:
        fields = header.split()
        if len(fields) < 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, fields[1:8])
        images.append({"name": " ".join(fields[9:]), "camera_id": int(fields[8]),
                       "R": _quat_to_matrix(qw, qx, qy, qz), "t": np.array([tx, ty, tz])})
    return cameras, images


def _read_binary(model_dir):
    cameras = {}
    data = (model_dir / "cameras.bin").read_bytes()
    count, offset = struct.unpack_from("<Q", data, 0)[0], 8
    for _ in range(count):
        camera_id, model_id, width, height = struct.unpack_from("<iiQQ", data, offset)
        offset += 24
        if model_id not in CAMERA_MODELS:
            raise ValueError(f"camera model id {model_id} is not supported for texturing")
        name, n = CAMERA_MODELS[model_id]
        params = struct.unpack_from("<" + "d" * n, data, offset)
        offset += 8 * n
        cameras[camera_id] = _camera(name, width, height, params)
    images = []
    data = (model_dir / "images.bin").read_bytes()
    count, offset = struct.unpack_from("<Q", data, 0)[0], 8
    for _ in range(count):
        values = struct.unpack_from("<i7di", data, offset)
        offset += 64
        end = data.index(b"\x00", offset)
        name = data[offset:end].decode("utf-8")
        offset = end + 1
        points = struct.unpack_from("<Q", data, offset)[0]
        offset += 8 + points * 24
        qw, qx, qy, qz, tx, ty, tz = values[1:8]
        images.append({"name": name, "camera_id": values[8],
                       "R": _quat_to_matrix(qw, qx, qy, qz), "t": np.array([tx, ty, tz])})
    return cameras, images


def project(points, image, camera):
    """World points -> (pixel xy, depth) through COLMAP's camera model, distortion included."""
    cam = np.asarray(points, dtype=np.float64) @ image["R"].T + image["t"]
    depth = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = cam[:, 0] / depth, cam[:, 1] / depth
    p, model = camera["params"], camera["model"]
    if model == "SIMPLE_PINHOLE":
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    elif model == "PINHOLE":
        fx, fy, cx, cy = p
    elif model in ("SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        r2 = x * x + y * y
        radial = 1 + p[3] * r2 + (p[4] * r2 * r2 if model == "RADIAL" else 0.0)
        x, y = x * radial, y * radial
    else:  # OPENCV
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        x, y = (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    return np.column_stack([fx * x + cx, fy * y + cy]), depth


def camera_centre(image):
    return -image["R"].T @ image["t"]


# --------------------------------------------------------------------------- visibility
def _lattice(k):
    """Barycentric points of a k-subdivided triangle, (m, 3)."""
    rows = [(i / k, j / k, 1 - (i + j) / k) for i in range(k + 1) for j in range(k + 1 - i)]
    return np.array(rows, dtype=np.float64)


LATTICE_MAX_PX = 48


def _rasterise(buffer, corners, depths, width, height):
    """Write one triangle's per-pixel depth into ``buffer`` (flat, min), bbox-clipped."""
    x0, y0 = np.floor(corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width - 1), min(y1, height - 1)
    if x0 > x1 or y0 > y1:
        return
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
    (ax, ay), (bx, by), (cx, cy) = corners
    area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    if abs(area) < 1e-9:
        return
    w0 = ((bx - xs) * (cy - ys) - (cx - xs) * (by - ys)) / area
    w1 = ((cx - xs) * (ay - ys) - (ax - xs) * (cy - ys)) / area
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not inside.any():
        return
    z = w0 * depths[0] + w1 * depths[1] + w2 * depths[2]
    index = (ys[inside] - 0.5).astype(np.int64) * width + (xs[inside] - 0.5).astype(np.int64)
    np.minimum.at(buffer, index, z[inside])


def _depth_buffer(tri_px, tri_depth, valid, width, height):
    """Min-depth buffer: small faces splatted as lattices, large ones rasterised.

    Lattice spacing stays below 2/3 px, so a face leaves no hole in the buffer (a sparser
    splat, capped at 48 subdivisions, let a roof 120 px across fail to hide the ground
    under it). A face longer than ``LATTICE_MAX_PX`` on screen is instead rasterised over
    its bounding box clipped to the buffer, so a hull face reaching far off-screen costs
    at most one buffer's worth of pixels rather than a lattice of millions.
    """
    buffer = np.full(height * width, np.inf)
    edges = np.max(np.linalg.norm(tri_px - np.roll(tri_px, 1, axis=1), axis=2), axis=1)
    subdivisions = np.clip(np.ceil(edges * 1.5), 1, None).astype(np.int64)
    subdivisions[~valid] = 0
    large = subdivisions > 1.5 * LATTICE_MAX_PX
    for face in np.flatnonzero(large):
        _rasterise(buffer, tri_px[face], tri_depth[face], width, height)
    subdivisions[large] = 0
    for k in np.unique(subdivisions[subdivisions > 0]):
        members = np.flatnonzero(subdivisions == k)
        bary = _lattice(int(k))
        step = max(1, 4_000_000 // len(bary))
        for start in range(0, len(members), step):
            faces = members[start:start + step]
            xy = np.einsum("mk,fkd->fmd", bary, tri_px[faces]).reshape(-1, 2)
            z = (tri_depth[faces] @ bary.T).reshape(-1)
            col, row = np.floor(xy[:, 0]).astype(np.int64), np.floor(xy[:, 1]).astype(np.int64)
            keep = (col >= 0) & (col < width) & (row >= 0) & (row < height) & (z > 0)
            np.minimum.at(buffer, row[keep] * width + col[keep], z[keep])
    return buffer.reshape(height, width)


def visibility(vertices, faces, cameras, images):
    """(score[F, V], grazing[F, V]): score is |cos| * area_px (0 = not seen), grazing
    marks a view that sees the face at more than the MIN_COS angle."""
    tri = vertices[faces]
    centroid = tri.mean(axis=1)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normal, axis=1)
    normal = normal / np.where(length > 0, length, 1)[:, None]
    scores = np.zeros((len(faces), len(images)), dtype=np.float32)
    grazing = np.zeros((len(faces), len(images)), dtype=bool)
    for index, image in enumerate(images):
        camera = cameras[image["camera_id"]]
        width, height = camera["width"], camera["height"]
        px, depth = project(vertices, image, camera)
        tri_px, tri_depth = px[faces], depth[faces]
        in_front = (tri_depth > 1e-6).all(axis=1) & np.isfinite(tri_px).all(axis=(1, 2))
        inside = in_front & (tri_px[..., 0] >= EDGE_MARGIN_PX).all(1) & (
            tri_px[..., 0] <= width - 1 - EDGE_MARGIN_PX).all(1) & (
            tri_px[..., 1] >= EDGE_MARGIN_PX).all(1) & (
            tri_px[..., 1] <= height - 1 - EDGE_MARGIN_PX).all(1)
        ray = camera_centre(image) - centroid
        cos = np.abs(np.einsum("fd,fd->f", normal, ray)) / np.maximum(
            np.linalg.norm(ray, axis=1), 1e-12)
        a, b, c = tri_px[:, 0], tri_px[:, 1], tri_px[:, 2]
        area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                            - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
        candidate = inside & (cos >= GRAZING_COS) & (area > 0.05)
        if not candidate.any():
            continue
        shrink = min(1.0, DEPTH_BUFFER_WIDTH / width)
        bw, bh = max(8, int(round(width * shrink))), max(8, int(round(height * shrink)))
        buffer = _depth_buffer(tri_px * shrink, tri_depth, in_front, bw, bh)
        c_px, c_depth = project(centroid[candidate], image, camera)
        col = np.clip((c_px[:, 0] * shrink).astype(np.int64), 0, bw - 1)
        row = np.clip((c_px[:, 1] * shrink).astype(np.int64), 0, bh - 1)
        # The median of the 3x3 neighbourhood absorbs the buffer's own discretisation at
        # a depth edge (a min would let a neighbouring face hide this one, a max would let
        # an occluder hide nothing), so an occluder must cover most of it to hide the face.
        padded = np.pad(buffer, 1, constant_values=np.inf)
        nearest = np.median(np.stack([padded[row + dy, col + dx] for dy in range(3)
                                      for dx in range(3)]), axis=0)
        unoccluded = c_depth <= nearest * (1 + DEPTH_TOLERANCE) + 1e-9
        rows = np.flatnonzero(candidate)[unoccluded]
        weight = np.where(cos[rows] >= MIN_COS, 1.0, GRAZING_WEIGHT)
        scores[rows, index] = (weight * cos[rows] * area[rows]).astype(np.float32)
        grazing[rows, index] = cos[rows] < MIN_COS
    return scores, grazing


def _face_neighbours(faces):
    """(F, 3) adjacent face per edge, -1 on a boundary or a non-manifold edge."""
    count = len(faces)
    edges = np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1)
    edges = np.sort(edges.reshape(-1, 2), axis=1)
    owner = np.repeat(np.arange(count), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner, slot = edges[order], owner[order], order
    same = np.all(edges[1:] == edges[:-1], axis=1)
    neighbours = np.full(count * 3, -1, dtype=np.int64)
    pairs = np.flatnonzero(same)
    # Only exact pairs: an edge shared by three faces is non-manifold and left alone.
    unique = pairs[~np.isin(pairs - 1, pairs) & ~np.isin(pairs + 1, pairs)]
    neighbours[slot[unique]] = owner[unique + 1]
    neighbours[slot[unique + 1]] = owner[unique]
    return neighbours.reshape(count, 3)


def select_views(scores, faces, passes=SMOOTHING_PASSES):
    """Best view per face (-1 when none), then neighbour-majority smoothing."""
    choice = np.where(scores.max(axis=1) > 0, np.argmax(scores, axis=1), -1)
    neighbours = _face_neighbours(faces)
    rows = np.arange(len(faces))
    for _ in range(passes):
        views = np.where(neighbours >= 0, choice[np.maximum(neighbours, 0)], -1)
        a, b, c = views[:, 0], views[:, 1], views[:, 2]
        majority = np.where((a >= 0) & ((a == b) | (a == c)), a,
                            np.where((b >= 0) & (b == c), b, -1))
        # Switch only to a view that genuinely sees this face, and not for a worse
        # than half-as-good image of it: smoothing trades seams, not coverage.
        ok = (majority >= 0) & (majority != choice) & (choice >= 0)
        better = np.zeros(len(faces), dtype=bool)
        better[ok] = scores[rows[ok], majority[ok]] >= 0.5 * scores[rows[ok], choice[ok]]
        choice = np.where(better, majority, choice)
    return choice


# --------------------------------------------------------------------------------- atlas
def atlas_layout(face_count, cell_px):
    cells = (face_count + 1) // 2
    grid = max(1, math.ceil(math.sqrt(cells)))
    size = grid * cell_px
    if size > MAX_ATLAS_PX:
        cell_px = max(MIN_CELL_PX, MAX_ATLAS_PX // grid)
        size = grid * cell_px
        if size > MAX_ATLAS_PX:
            raise ValueError(f"{face_count} faces do not fit a {MAX_ATLAS_PX} px atlas even at "
                             f"{MIN_CELL_PX} px per cell; decimate the mesh first")
    return grid, cell_px, size


def _cell_coordinates(face_ids, grid, cell_px):
    cell = face_ids // 2
    return (cell % grid) * cell_px, (cell // grid) * cell_px, (face_ids % 2).astype(bool)


def face_uvs(face_count, grid, cell_px, size):
    """(F, 3, 2) UVs in [0, 1], image origin top-left (glTF convention)."""
    ids = np.arange(face_count)
    x0, y0, upper = _cell_coordinates(ids, grid, cell_px)
    pad, span = 1.0, cell_px - 2.0
    lower_corners = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    upper_corners = np.array([[1, 1], [0, 1], [1, 0]], dtype=np.float64)
    corners = np.where(upper[:, None, None], upper_corners, lower_corners)
    uv_px = np.stack([x0, y0], axis=1)[:, None, :] + pad + corners * span
    return uv_px / size


def _texels(face_ids, grid, cell_px):
    """For each face: texel pixel indices and their barycentric coordinates."""
    x0, y0, upper = _cell_coordinates(face_ids, grid, cell_px)
    local = (np.arange(cell_px) + 0.5 - 1.0) / (cell_px - 2.0)
    a, b = np.meshgrid(local, local)                      # a along x, b along y
    a, b = np.clip(a.ravel(), 0, 1), np.clip(b.ravel(), 0, 1)
    ix, iy = np.meshgrid(np.arange(cell_px), np.arange(cell_px))
    ix, iy = ix.ravel(), iy.ravel()
    lower_mask, upper_mask = (a + b) <= 1.0, (a + b) > 1.0
    result = []
    for index, face in enumerate(face_ids):
        if upper[index]:
            mask, fa, fb = upper_mask, 1 - a, 1 - b
        else:
            mask, fa, fb = lower_mask, a, b
        s = np.clip(fa[mask] + fb[mask], 0, None)
        scale = np.where(s > 1, 1 / np.maximum(s, 1e-12), 1.0)  # clamp padding onto the face
        u, v = fa[mask] * scale, fb[mask] * scale
        bary = np.column_stack([1 - u - v, u, v])
        result.append((x0[index] + ix[mask], y0[index] + iy[mask], bary))
    return result


def _sample(frame, px, width=1024):
    """Bilinear RGB at float pixel positions. cv2.remap caps a map at 32767 rows, so the
    positions go through as a (rows, 1024) block, padded and then trimmed."""
    import cv2
    count = len(px)
    rows = max(1, -(-count // width))
    grid = np.zeros((rows * width, 2), dtype=np.float32)
    grid[:count] = px
    colour = cv2.remap(frame, grid[:, 0].reshape(rows, width), grid[:, 1].reshape(rows, width),
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return colour.reshape(-1, 3)[:count]


def bake(vertices, faces, cameras, images, load_image, *, vertex_colors=None,
         cell_px=None, progress=None):
    """Texture ``faces`` from ``images``. Returns (texture HxWx3 uint8, uvs F x 3 x 2, report).

    ``load_image(name)`` returns an HxWx3 uint8 RGB frame whose size matches its camera.
    """
    import cv2
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite Nx3")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 \
            or faces.max() >= len(vertices):
        raise ValueError("faces must be a non-empty Fx3 index array into vertices")
    images = [image for image in images if image["camera_id"] in cameras]
    if not images:
        raise ValueError("no registered image has a camera to project through")
    scores, grazing = visibility(vertices, faces, cameras, images)
    choice = select_views(scores, faces)
    seen = choice >= 0
    if cell_px is None:
        # The on-screen area of a well-seen face, as a square's side, is the resolution the
        # frames can actually supply; texels beyond it would only upsample.
        best = scores.max(axis=1)[seen]
        side = math.sqrt(2 * float(np.percentile(best, 95))) if seen.any() else 8
        cell_px = int(np.clip(math.ceil(side) + 2, MIN_CELL_PX, MAX_CELL_PX))
    grid, cell_px, size = atlas_layout(len(faces), int(cell_px))
    texture = np.full((size, size, 3), 128, dtype=np.uint8)
    uvs = face_uvs(len(faces), grid, cell_px, size)
    sampled = 0
    for view in np.unique(choice[seen]):
        image = images[int(view)]
        frame = load_image(image["name"])
        camera = cameras[image["camera_id"]]
        if frame is None or frame.shape[:2] != (camera["height"], camera["width"]):
            raise ValueError(f"{image['name']}: frame is missing or not "
                             f"{camera['width']}x{camera['height']} as its camera says")
        members = np.flatnonzero(choice == view)
        for start in range(0, len(members), 4096):
            chunk = members[start:start + 4096]
            texels = _texels(chunk, grid, cell_px)
            xs = np.concatenate([t[0] for t in texels])
            ys = np.concatenate([t[1] for t in texels])
            points = np.concatenate([t[2] @ vertices[faces[face]]
                                     for t, face in zip(texels, chunk)])
            px, _ = project(points, image, camera)
            texture[ys, xs] = _sample(frame, px)
            sampled += len(chunk)
        if progress:
            progress(sampled, int(seen.sum()))
    unseen = np.flatnonzero(~seen)
    if len(unseen) and vertex_colors is not None:
        colors = np.asarray(vertex_colors, dtype=np.float64)
        for start in range(0, len(unseen), 4096):
            chunk = unseen[start:start + 4096]
            texels = _texels(chunk, grid, cell_px)
            for (xs, ys, bary), face in zip(texels, chunk):
                texture[ys, xs] = np.clip(bary @ colors[faces[face]], 0, 255).astype(np.uint8)
    used = np.bincount(choice[seen], minlength=len(images))
    report = {"schema_version": 1, "method": "best-view per face, neighbour-majority smoothed",
              "faces": int(len(faces)), "textured_faces": int(seen.sum()),
              "untextured_faces": int((~seen).sum()),
              "grazing_faces": int(grazing[np.flatnonzero(seen), choice[seen]].sum()),
              "untextured_fill": "vertex colour" if vertex_colors is not None else "grey (128)",
              "views_available": len(images), "views_used": int((used > 0).sum()),
              "atlas_px": int(size), "cell_px": int(cell_px),
              "texels_per_face": int(cell_px * cell_px // 2),
              "not_done": ["no colour adjustment between views: exposure seams remain",
                           "no photo-consistency test: moving objects can be painted onto the "
                           "surface behind them",
                           "occlusion is tested against a coarse depth buffer at the face "
                           "centroid, so a face partly hidden in its chosen view can carry a "
                           "sliver of its occluder"]}
    return texture, uvs, report


# ------------------------------------------------------------------------------- writers
def write_obj(directory, vertices, faces, uvs, texture, *, name="textured", quality=92):
    """OBJ + MTL + JPEG. OBJ's texture origin is bottom-left, so v is flipped here."""
    import cv2
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image_name = f"{name}.jpg"
    if not cv2.imwrite(str(directory / image_name), texture[:, :, ::-1],
                       [cv2.IMWRITE_JPEG_QUALITY, int(quality)]):
        raise ValueError("could not write the texture image")
    (directory / f"{name}.mtl").write_text(
        f"newmtl {name}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd {image_name}\n",
        encoding="ascii")
    flat = uvs.reshape(-1, 2)
    lines = [f"mtllib {name}.mtl", f"o {name}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"vt {u:.7f} {1.0 - v:.7f}" for u, v in flat]
    lines.append(f"usemtl {name}")
    lines += [f"f {a + 1}/{3 * i + 1} {b + 1}/{3 * i + 2} {c + 1}/{3 * i + 3}"
              for i, (a, b, c) in enumerate(faces)]
    (directory / f"{name}.obj").write_text("\n".join(lines) + "\n", encoding="ascii")
    return [f"{name}.obj", f"{name}.mtl", image_name]


def write_gltf(directory, vertices, faces, uvs, *, name="textured", image_name=None):
    """glTF 2.0: one textured primitive, corners unshared so each carries its own UV."""
    directory = Path(directory)
    image_name = image_name or f"{name}.jpg"
    positions = np.asarray(vertices, dtype=np.float32)[np.asarray(faces).reshape(-1)]
    texcoords = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
    indices = np.arange(len(positions), dtype=np.uint32)
    blobs = [positions.tobytes(), texcoords.tobytes(), indices.tobytes()]
    offsets = np.cumsum([0] + [len(b) for b in blobs])
    (directory / f"{name}.bin").write_bytes(b"".join(blobs))
    document = {
        "asset": {"version": "2.0", "generator": "scripts/survey_texture.py"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{
            "attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0,
            "mode": 4}]}],
        "materials": [{"name": name, "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "doubleSided": True}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"uri": image_name, "mimeType": "image/jpeg"}],
        "buffers": [{"uri": f"{name}.bin", "byteLength": int(offsets[-1])}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": int(offsets[0]), "byteLength": len(blobs[0]),
             "target": 34962},
            {"buffer": 0, "byteOffset": int(offsets[1]), "byteLength": len(blobs[1]),
             "target": 34962},
            {"buffer": 0, "byteOffset": int(offsets[2]), "byteLength": len(blobs[2]),
             "target": 34963}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
             "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(texcoords), "type": "VEC2"},
            {"bufferView": 2, "componentType": 5125, "count": len(indices), "type": "SCALAR"}],
    }
    (directory / f"{name}.gltf").write_text(json.dumps(document), encoding="utf-8")
    return [f"{name}.gltf", f"{name}.bin"]


def write_glb(directory, vertices, faces, uvs, *, name="textured", image_name=None):
    """One self-contained binary glTF: geometry, UVs and the JPEG atlas in a single file.

    The .gltf above needs its .bin and .jpg beside it; a .glb is what a viewer, Blender or
    a customer's inbox wants. Chunks are JSON then BIN, 4-byte aligned, as the spec orders.
    """
    import struct as _struct
    directory = Path(directory)
    image = (directory / (image_name or f"{name}.jpg")).read_bytes()
    positions = np.asarray(vertices, dtype=np.float32)[np.asarray(faces).reshape(-1)]
    texcoords = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
    indices = np.arange(len(positions), dtype=np.uint32)
    blobs = [positions.tobytes(), texcoords.tobytes(), indices.tobytes(), image]
    views, body = [], b""
    for blob in blobs:
        views.append({"buffer": 0, "byteOffset": len(body), "byteLength": len(blob)})
        body += blob + b"\x00" * (-len(blob) % 4)
    views[0]["target"] = views[1]["target"] = 34962
    views[2]["target"] = 34963
    document = {
        "asset": {"version": "2.0", "generator": "scripts/survey_texture.py"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{
            "attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0,
            "mode": 4}]}],
        "materials": [{"name": name, "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "doubleSided": True}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": 3, "mimeType": "image/jpeg"}],
        "buffers": [{"byteLength": len(body)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
             "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(texcoords), "type": "VEC2"},
            {"bufferView": 2, "componentType": 5125, "count": len(indices), "type": "SCALAR"}],
    }
    text = json.dumps(document, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    total = 12 + 8 + len(text) + 8 + len(body)
    with (directory / f"{name}.glb").open("wb") as stream:
        stream.write(_struct.pack("<4sII", b"glTF", 2, total))
        stream.write(_struct.pack("<I4s", len(text), b"JSON") + text)
        stream.write(_struct.pack("<I4s", len(body), b"BIN\x00") + body)
    return [f"{name}.glb"]


def frame_loader(image_dir):
    """load_image for a COLMAP image folder: names are paths relative to it."""
    import cv2
    root = Path(image_dir).resolve()

    def load(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"image name {name!r} leaves the image folder")
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return None if frame is None else frame[:, :, ::-1].copy()
    return load


# ----------------------------------------------------------------------------- checking
def render(vertices, faces, image, camera, *, uvs=None, texture=None, vertex_colors=None,
           shrink=1.0):
    """Render the mesh into one camera: (rgb HxWx3 uint8, covered HxW bool).

    Faces are splatted as barycentric lattices at >= 1 sample per pixel and the nearest
    sample wins per pixel. Colour comes from the texture through the UVs, or from
    interpolated vertex colours. It exists to check a bake against a real frame, not to
    be a viewer.
    """
    width = max(8, int(round(camera["width"] * shrink)))
    height = max(8, int(round(camera["height"] * shrink)))
    px, depth = project(vertices, image, camera)
    tri_px, tri_depth = px[faces] * shrink, depth[faces]
    ok = (tri_depth > 1e-6).all(1) & np.isfinite(tri_px).all(axis=(1, 2))
    edges = np.max(np.linalg.norm(tri_px - np.roll(tri_px, 1, axis=1), axis=2), axis=1)
    subdivisions = np.clip(np.ceil(edges * 1.5), 1, 256).astype(np.int64)
    subdivisions[~ok] = 0
    pixel_all, depth_all, colour_all = [], [], []
    for k in np.unique(subdivisions[subdivisions > 0]):
        members = np.flatnonzero(subdivisions == k)
        bary = _lattice(int(k))
        for start in range(0, len(members), max(1, 2_000_000 // len(bary))):
            chunk = members[start:start + max(1, 2_000_000 // len(bary))]
            xy = np.einsum("mk,fkd->fmd", bary, tri_px[chunk]).reshape(-1, 2)
            z = (tri_depth[chunk] @ bary.T).reshape(-1)
            col, row = np.floor(xy[:, 0]).astype(np.int64), np.floor(xy[:, 1]).astype(np.int64)
            keep = (col >= 0) & (col < width) & (row >= 0) & (row < height)
            if texture is not None:
                uv = np.einsum("mk,fkd->fmd", bary, uvs[chunk]).reshape(-1, 2)
                tx = np.clip((uv[:, 0] * texture.shape[1]).astype(np.int64), 0,
                             texture.shape[1] - 1)
                ty = np.clip((uv[:, 1] * texture.shape[0]).astype(np.int64), 0,
                             texture.shape[0] - 1)
                colour = texture[ty[keep], tx[keep]]
            else:
                colour = np.einsum("mk,fkd->fmd", bary,
                                   np.asarray(vertex_colors, float)[faces[chunk]]).reshape(-1, 3)[keep]
            pixel_all.append(row[keep] * width + col[keep])
            depth_all.append(z[keep])
            colour_all.append(np.asarray(colour, dtype=np.uint8))
    rgb = np.zeros((height * width, 3), dtype=np.uint8)
    covered = np.zeros(height * width, dtype=bool)
    if pixel_all:
        pixel, z = np.concatenate(pixel_all), np.concatenate(depth_all)
        colour = np.concatenate(colour_all)
        order = np.lexsort((z, pixel))
        first = order[np.r_[True, pixel[order][1:] != pixel[order][:-1]]]
        rgb[pixel[first]] = colour[first]
        covered[pixel[first]] = True
    return rgb.reshape(height, width, 3), covered.reshape(height, width)


def holdout_check(vertices, faces, cameras, images, load_image, *, views=None, shrink=0.5,
                  save_dir=None):
    """Bake without each chosen view, render it, and compare with the real frame.

    Returns per-view mean absolute error (0-255) over pixels the textured mesh covers,
    next to the same mesh coloured per vertex from the same frames - the colouring a
    mesh had before this module existed. A texture that landed on the wrong geometry
    scores no better than the vertex colours.

    ``save_dir`` additionally writes the strips those numbers came from, since scoring
    only the numbers leaves no way to see what was measured.
    """
    import cv2
    views = list(range(0, len(images), max(1, len(images) // 6)))[:6] if views is None else views
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for held in views:
        kept = [image for index, image in enumerate(images) if index != held]
        texture, uvs, report = bake(vertices, faces, cameras, kept, load_image)
        colors = _vertex_colors(vertices, faces, cameras, kept, load_image)
        image = images[held]
        camera = cameras[image["camera_id"]]
        truth = load_image(image["name"])
        truth = cv2.resize(truth, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
        textured, covered_t = render(vertices, faces, image, camera, uvs=uvs, texture=texture,
                                     shrink=shrink)
        vertex, covered_v = render(vertices, faces, image, camera, vertex_colors=colors,
                                   shrink=shrink)
        mask = covered_t & covered_v & (textured.sum(axis=2) > 0)
        grey = np.all(textured == 128, axis=2)
        mask &= ~grey
        if mask.sum() < 100:
            continue
        diff_t = np.abs(textured[mask].astype(int) - truth[mask].astype(int)).mean()
        diff_v = np.abs(vertex[mask].astype(int) - truth[mask].astype(int)).mean()
        rows.append({"view": image["name"], "pixels": int(mask.sum()),
                     "mae_textured": round(float(diff_t), 2),
                     "mae_vertex_colour": round(float(diff_v), 2),
                     "detail_textured": _detail_agreement(textured, truth, mask),
                     "detail_vertex_colour": _detail_agreement(vertex, truth, mask)})
        if save_dir is not None:
            strip = np.concatenate([_label(cv2, textured, f"textured  mae {diff_t:.1f}"),
                                    _label(cv2, vertex, f"vertex colour  mae {diff_v:.1f}"),
                                    _label(cv2, truth, "real frame (held out)")], axis=1)
            name = Path(image["name"]).stem
            cv2.imwrite(str(Path(save_dir) / f"{len(rows):02d}_{name}.png"),
                        cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    return rows


def _label(cv2, rgb, caption):
    """Draw a caption bar above a panel so a strip reads without a separate legend."""
    width = rgb.shape[1]
    bar = np.full((26, width, 3), 245, dtype=np.uint8)
    cv2.putText(bar, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1,
                cv2.LINE_AA)
    return np.concatenate([bar, rgb], axis=0)


def _detail_agreement(rendered, truth, mask):
    """Correlation of local gradient magnitude, rendered vs real, over ``mask``.

    Pixel error rewards blur: when geometry is slightly off, a smooth colour scores as
    well as sharp detail in almost the right place. This asks the other question - does
    the render have edges where the real frame has edges - which per-vertex colour on a
    coarse mesh cannot answer well, and a misplaced texture cannot either.
    """
    import cv2
    def magnitude(rgb):
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))
    inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    a, b = magnitude(rendered)[inner], magnitude(truth)[inner]
    if a.std() == 0 or b.std() == 0:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def _vertex_colors(vertices, faces, cameras, images, load_image):
    """Per-vertex colour from each vertex's best view: the pre-texture baseline."""
    scores, _ = visibility(vertices, faces, cameras, images)
    choice = np.where(scores.max(axis=1) > 0, np.argmax(scores, axis=1), -1)
    colors = np.full((len(vertices), 3), 128.0)
    view_of_vertex = np.full(len(vertices), -1)
    for corner in range(3):
        idx = faces[:, corner]
        fresh = (view_of_vertex[idx] < 0) & (choice >= 0)
        view_of_vertex[idx[fresh]] = choice[fresh]
    for view in np.unique(view_of_vertex[view_of_vertex >= 0]):
        members = np.flatnonzero(view_of_vertex == view)
        image = images[int(view)]
        px, _ = project(vertices[members], image, cameras[image["camera_id"]])
        colors[members] = _sample(load_image(image["name"]), px)
    return colors


# ------------------------------------------------------------------------ pipeline step
def _delaunay_mesh(model_dir, destination):
    """COLMAP's CPU graph-cut mesher over the sparse model: faces from measured points."""
    import os
    import subprocess
    tools = Path(__file__).resolve().parent.parent / "tools" / "colmap"
    exe = tools / "bin" / ("colmap.exe" if os.name == "nt" else "colmap")
    if not exe.is_file():
        raise ValueError(f"COLMAP is not installed at {exe}; no mesh can be built")
    env = {**os.environ, "PATH": f"{tools / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
           "QT_PLUGIN_PATH": str(tools / "plugins")}
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([str(exe), "delaunay_mesher", "--input_path", str(model_dir),
                             "--input_type", "sparse", "--output_path", str(destination)],
                            env=env, capture_output=True, text=True, timeout=1800)
    if result.returncode or not destination.is_file():
        raise ValueError("delaunay_mesher failed: " + (result.stderr or result.stdout)[-400:])


def texture_work(work, *, image_dir="frames_train", holdout=False):
    """The pipeline's ``texture`` step: mesh the sparse model, bake, write in world metres.

    The mesh is COLMAP's graph-cut Delaunay surface over the *sparse* points - coarse
    (thousands of faces for a whole site), but every vertex is a triangulated point, and
    it needs no GPU. Output lands in ``work/textured/`` in the viewer's world frame,
    ``(P @ R.T) * scale`` from frame.json, so it overlays the splat and inherits exactly
    its scale caveat.
    """
    work = Path(work)
    model = work / "colmap" / "sparse" / "txt"
    out = work / "textured"
    frame = json.loads((work / "frame.json").read_text(encoding="utf-8"))
    rotation = np.asarray(frame["rotation_rowmajor"], dtype=np.float64)
    scale = float(frame["scale_m_per_unit"])
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not scale > 0:
        raise ValueError("frame.json has no usable rotation/scale; run the frame step first")
    mesh = out / "mesh_sparse_delaunay.ply"
    _delaunay_mesh(model, mesh)
    try:
        from survey_deliver import read_mesh_ply
    except ImportError:
        from scripts.survey_deliver import read_mesh_ply
    vertices, faces, colors = read_mesh_ply(mesh)
    if faces is None or not len(faces):
        raise ValueError("the sparse model produced no surface faces")
    cameras, images = read_model(model)
    load = frame_loader(work / image_dir)
    # The graph-cut mesher closes the volume it carves, so a sparse model comes back with
    # hull walls around the site that no camera ever looked at. A face no view sees
    # carries no measured appearance and, here, no measured shape either: drop it and
    # count it, rather than ship grey invented walls.
    scores, _ = visibility(vertices, faces, cameras, images)
    seen = scores.max(axis=1) > 0
    dropped = int((~seen).sum())
    faces = faces[seen]
    texture, uvs, report = bake(vertices, faces, cameras, images, load, vertex_colors=colors)
    report["dropped_unseen_faces"] = dropped
    world = (vertices @ rotation.T) * scale
    files = write_obj(out, world, faces, uvs, texture)
    files += write_gltf(out, world, faces, uvs)
    files += write_glb(out, world, faces, uvs)
    report.update(mesh="COLMAP delaunay_mesher over the sparse model (CPU)",
                  vertices=int(len(vertices)), frame="viewer world, (P @ R.T) * scale",
                  scale_m_per_unit=scale, scale_source=frame.get("scale_source"),
                  image_dir=image_dir, files=files)
    if holdout:
        report["holdout"] = holdout_check(vertices, faces, cameras, images, load,
                                          save_dir=out / "compare")
    (out / "texture_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Bake video frames into a UV-textured mesh.")
    parser.add_argument("--work", type=Path, required=True, help="work/<scene>")
    parser.add_argument("--image-dir", default="frames_train",
                        help="frame folder the COLMAP image names are relative to")
    parser.add_argument("--holdout", action="store_true",
                        help="also bake without each of 6 views and score it on that view")
    args = parser.parse_args(argv)
    try:
        report = texture_work(args.work, image_dir=args.image_dir, holdout=args.holdout)
    except (ValueError, KeyError, OSError) as error:
        print(f"[texture] {error}")
        return 2
    print(f"[texture] {report['textured_faces']}/{report['faces']} faces textured from "
          f"{report['views_used']} views; atlas {report['atlas_px']} px "
          f"({report['untextured_faces']} left as {report['untextured_fill']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
