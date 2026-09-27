// Engine-free planning-layer logic for the viewer, shared with Node tests.
//
// The backend (scripts/workspace_proposals.py) owns every proposal's geometry; the
// viewer only draws what it is sent, picks what is under the cursor, and hides the
// splats inside demolished volumes. Nothing here invents or edits geometry.

export const MAX_PLAN_FEATURES = 2000;
export const MAX_PLAN_VERTICES = 600000;
export const MAX_CLIPS = 8;
export const PLAN_VIEWS = ["proposal", "existing"];

const finite = v => typeof v === "number" && Number.isFinite(v);
const rgb = c => Array.isArray(c) && c.length === 3 && c.every(v => Number.isInteger(v) && v >= 0 && v <= 255);

/** Validate the host's `plan` payload; throw on anything malformed or oversized. */
export function validatePlan(value) {
  if (!value || typeof value !== "object") throw new Error("plan value must be an object");
  const features = value.features;
  if (!Array.isArray(features) || features.length > MAX_PLAN_FEATURES) throw new Error("plan features must be a bounded array");
  let vertices = 0;
  const clean = features.map(f => {
    if (!f || typeof f.id !== "string" || !f.id || f.id.length > 64) throw new Error("plan feature id must be a short string");
    if (!Array.isArray(f.meshes) || f.meshes.length > 16) throw new Error("plan feature meshes must be a bounded array");
    return {
      id: f.id, kind: typeof f.kind === "string" ? f.kind.slice(0, 16) : "feature",
      meshes: f.meshes.map(m => {
        if (!m || !Array.isArray(m.positions) || !Array.isArray(m.indices)) throw new Error("plan mesh needs positions and indices");
        if (m.positions.length % 3 || !m.positions.every(finite)) throw new Error("plan mesh positions must be finite xyz triples");
        const count = m.positions.length / 3;
        if (m.indices.length % 3 || !m.indices.every(i => Number.isInteger(i) && i >= 0 && i < count)) throw new Error("plan mesh indices must be triangles into its positions");
        if (!rgb(m.color)) throw new Error("plan mesh color must be RGB 0..255");
        const opacity = finite(m.opacity) ? Math.min(1, Math.max(0.05, m.opacity)) : 1;
        vertices += m.indices.length;
        return { positions: m.positions, indices: m.indices, color: m.color, opacity, violation: m.violation === true, part: typeof m.part === "string" ? m.part.slice(0, 16) : "" };
      }),
    };
  });
  if (vertices > MAX_PLAN_VERTICES) throw new Error("plan geometry is too large to draw");
  const clips = Array.isArray(value.clips) ? value.clips : [];
  if (clips.length > MAX_CLIPS) throw new Error(`at most ${MAX_CLIPS} demolished volumes can be hidden at once`);
  for (const c of clips) {
    if (!c || !["cx", "cz", "hx", "hz", "angle", "y_min", "y_max"].every(k => finite(c[k]))) throw new Error("plan clip must be a finite oriented box");
  }
  const view = value.view === undefined ? "proposal" : value.view;
  if (!PLAN_VIEWS.includes(view)) throw new Error("plan view must be proposal or existing");
  const selected = value.selected === undefined || value.selected === null ? null : value.selected;
  if (selected !== null && (typeof selected !== "string" || selected.length > 64)) throw new Error("plan selected must be an id or null");
  const edit = value.edit === undefined || value.edit === null ? null : validateEdit(value.edit);
  return { features: clean, clips: clips.map(c => ({ ...c })), view, selected, edit };
}

/** The selected feature's editable outline, as the host describes it. */
export function validateEdit(edit) {
  if (!edit || typeof edit.id !== "string" || !edit.id || edit.id.length > 64) throw new Error("plan edit needs a feature id");
  const points = edit.points;
  if (!Array.isArray(points) || points.length < 1 || points.length > 512 ||
      !points.every(p => Array.isArray(p) && p.length === 2 && p.every(finite))) throw new Error("plan edit points must be [x, z] pairs");
  if (!finite(edit.y)) throw new Error("plan edit needs a base height");
  const min = edit.closed ? 3 : Math.min(2, points.length);
  return { id: edit.id, points: points.map(p => [p[0], p[1]]), y: edit.y, closed: edit.closed === true,
    vertices: edit.vertices === true && points.length >= min, rotate: edit.rotate === true,
    heading_deg: finite(edit.heading_deg) ? edit.heading_deg : null, min };
}

const centroid = pts => [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length];

/**
 * Handles for editing an outline in 3D: a dot per vertex, a "+" at each edge midpoint
 * (drag it to insert a vertex), a move grip at the centroid and a rotate grip on a stalk
 * that points along the feature's heading, so it turns with the feature.
 */
export function planHandles(edit) {
  const pts = edit.points, n = pts.length, out = [];
  if (edit.vertices) {
    pts.forEach((p, index) => out.push({ kind: "vertex", index, xz: [...p] }));
    for (let i = 0; i < (edit.closed ? n : n - 1); i++) {
      const a = pts[i], b = pts[(i + 1) % n];
      out.push({ kind: "mid", index: i, xz: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] });
    }
  }
  const c = centroid(pts);
  out.push({ kind: "move", index: -1, xz: c });
  if (edit.rotate) {
    const radius = Math.max(1.5, ...pts.map(p => Math.hypot(p[0] - c[0], p[1] - c[1])));
    const heading = edit.heading_deg !== null && edit.heading_deg !== undefined ? edit.heading_deg * Math.PI / 180
      : n >= 2 ? Math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]) : 0;
    const reach = radius + Math.max(2, 0.25 * radius);
    out.push({ kind: "rotate", index: -1, xz: [c[0] + reach * Math.cos(heading), c[1] + reach * Math.sin(heading)] });
  }
  return out;
}

const snapTo = (v, step) => (step > 0 ? Math.round(v / step) * step : v);

/**
 * The outline after dragging ``handle`` from ``from`` to ``to`` (both [x, z] on the
 * edit plane). Rotation is about the centroid and uses the same sense as the backend's
 * yaw (+x toward +z), so an object's heading changes by exactly ``rotation_deg``.
 */
export function dragShape(edit, handle, from, to, { grid = 0, snapDeg = 0 } = {}) {
  const pts = edit.points.map(p => [...p]);
  if (handle.kind === "vertex") {
    pts[handle.index] = [snapTo(to[0], grid), snapTo(to[1], grid)];
    return { points: pts, rotation_deg: 0 };
  }
  if (handle.kind === "mid") {
    pts.splice(handle.index + 1, 0, [snapTo(to[0], grid), snapTo(to[1], grid)]);
    return { points: pts, rotation_deg: 0 };
  }
  if (handle.kind === "move") {
    const dx = snapTo(to[0] - from[0], grid), dz = snapTo(to[1] - from[1], grid);
    return { points: pts.map(([x, z]) => [x + dx, z + dz]), rotation_deg: 0 };
  }
  const c = centroid(pts);
  let angle = Math.atan2(to[1] - c[1], to[0] - c[0]) - Math.atan2(from[1] - c[1], from[0] - c[0]);
  angle = Math.atan2(Math.sin(angle), Math.cos(angle));
  if (snapDeg > 0) angle = snapTo(angle * 180 / Math.PI, snapDeg) * Math.PI / 180;
  const cs = Math.cos(angle), sn = Math.sin(angle);
  return {
    points: pts.map(([x, z]) => [c[0] + (x - c[0]) * cs - (z - c[1]) * sn, c[1] + (x - c[0]) * sn + (z - c[1]) * cs]),
    rotation_deg: angle * 180 / Math.PI,
  };
}

/** The outline without one vertex, or null when that would leave too few. */
export function removeVertex(edit, index) {
  if (!edit.vertices || edit.points.length - 1 < edit.min || index < 0 || index >= edit.points.length) return null;
  return edit.points.filter((_, i) => i !== index);
}

/**
 * Ray from the camera onto the horizontal plane y = h, as [x, z], or null if it is behind
 * the camera or meets the plane at under ``minAngleDeg``: near the horizon one pixel spans
 * metres of ground, so a drag there would fling the feature across the scene.
 */
export function planeHit(origin, direction, h, minAngleDeg = 4) {
  const dy = direction[1];
  if (Math.abs(dy) < Math.sin(minAngleDeg * Math.PI / 180) * Math.hypot(...direction)) return null;
  const t = (h - origin[1]) / dy;
  if (!(t > 0)) return null;
  return [origin[0] + direction[0] * t, origin[2] + direction[2] * t];
}

/**
 * Walls the walker collides with: one thin box per footprint edge of each proposed
 * building (any polygon, convex or not), and one box per object's trunk/pole/body.
 * Returned as {center, halfExtents, yaw} with yaw in the physics convention (-angle).
 */
export function planColliders(features) {
  const boxes = [];
  for (const f of features) {
    if (f.kind !== "building" && f.kind !== "object") continue;
    for (const m of f.meshes) {
      const p = m.positions;
      let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < p.length; i += 3) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[i + k]); hi[k] = Math.max(hi[k], p[i + k]); }
      if (f.kind === "object") {
        // A tree's crown and a lamp's arm are overhead: only what stands on the ground blocks.
        if (!["trunk", "pole", "base", "body", "object", "wheel", "cabin"].includes(m.part)) continue;
        if (lo[1] > hi[1] - 0.05) continue;
        boxes.push({ center: [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2],
          halfExtents: [Math.max(0.05, (hi[0] - lo[0]) / 2), (hi[1] - lo[1]) / 2, Math.max(0.05, (hi[2] - lo[2]) / 2)], yaw: 0 });
        continue;
      }
      if (m.part !== "walls") continue;
      // Walls come as quads (bottom ring 0..n-1, top ring n..2n-1): one box per edge.
      const n = p.length / 6;
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        const ax = p[3 * i], az = p[3 * i + 2], bx = p[3 * j], bz = p[3 * j + 2];
        const len = Math.hypot(bx - ax, bz - az);
        if (len < 1e-3) continue;
        const y0 = p[3 * i + 1], y1 = p[3 * (n + i) + 1];
        boxes.push({ center: [(ax + bx) / 2, (y0 + y1) / 2, (az + bz) / 2], halfExtents: [len / 2, Math.max(0.1, (y1 - y0) / 2), 0.15],
          yaw: -Math.atan2(bz - az, bx - ax) * 180 / Math.PI });
      }
    }
  }
  return boxes;
}

/**
 * Collider triangles to drop for demolitions: those whose centroid is inside a clip box
 * and more than ``keep_m`` above the ground there (``groundAt(x, z)``), so the ground
 * the structure stood on stays walkable.
 */
export function clipTriangles(positions, indices, clips, groundAt, keep_m = 0.4) {
  const keep = [];
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t] * 3, b = indices[t + 1] * 3, c = indices[t + 2] * 3;
    const x = (positions[a] + positions[b] + positions[c]) / 3;
    const y = (positions[a + 1] + positions[b + 1] + positions[c + 1]) / 3;
    const z = (positions[a + 2] + positions[b + 2] + positions[c + 2]) / 3;
    if (insideClip(clips, [x, y, z]) && y > groundAt(x, z) + keep_m) continue;
    keep.push(indices[t], indices[t + 1], indices[t + 2]);
  }
  return keep;
}

const LIGHT = (() => { const l = [0.35, 0.85, 0.4]; const n = Math.hypot(...l); return l.map(v => v / n); })();
const mix = (a, b, t) => a.map((v, i) => Math.round(v * (1 - t) + b[i] * t));

/**
 * Flat-shaded, non-indexed geometry with baked per-face light (the placement layer's
 * unlit convention): every triangle gets its own three vertices so faces read as
 * faces. Selected features are tinted amber, rule violations red.
 */
export function planGeometry(mesh, { selected = false } = {}) {
  const p = mesh.positions, idx = mesh.indices;
  const tris = idx.length / 3;
  const positions = new Float32Array(tris * 9);
  const normals = new Float32Array(tris * 9);
  const colors = new Uint8Array(tris * 12);
  let base = mesh.color;
  if (mesh.violation) base = mix(base, [236, 64, 64], 0.55);
  if (selected) base = mix(base, [255, 196, 40], 0.4);
  const alpha = Math.round(255 * (mesh.opacity ?? 1));
  for (let t = 0; t < tris; t++) {
    const a = idx[3 * t] * 3, b = idx[3 * t + 1] * 3, c = idx[3 * t + 2] * 3;
    const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
    const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len; ny /= len; nz /= len;
    // Double-sided light: a face seen from behind is lit like its front.
    const shade = 0.58 + 0.42 * Math.abs(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]);
    const col = base.map(v => Math.min(255, Math.round(v * shade)));
    for (let k = 0; k < 3; k++) {
      const src = idx[3 * t + k] * 3, dst = 9 * t + 3 * k;
      positions[dst] = p[src]; positions[dst + 1] = p[src + 1]; positions[dst + 2] = p[src + 2];
      normals[dst] = nx; normals[dst + 1] = ny; normals[dst + 2] = nz;
      colors.set([col[0], col[1], col[2], alpha], 12 * t + 4 * k);
    }
  }
  const indices = new Uint32Array(tris * 3).map((_, i) => i);
  return { positions, normals, colors, indices, transparent: alpha < 255 };
}

/** Cheap order-sensitive checksum of a feature's geometry, so any vertex edit redraws it. */
export function planSignature(feature, selected) {
  let sum = 0;
  for (const m of feature.meshes) {
    const p = m.positions;
    for (let i = 0; i < p.length; i++) sum = (sum * 31 + Math.round(p[i] * 1000)) % 2147483647;
    sum = (sum * 31 + m.indices.length + (m.violation ? 7 : 0) + Math.round(m.opacity * 100)) % 2147483647;
    for (const c of m.color) sum = (sum * 31 + c) % 2147483647;
  }
  return `${feature.meshes.length}:${sum}:${selected ? 1 : 0}`;
}

/** Nearest proposal feature hit by a ray (Möller–Trumbore), or null. Overlays are never picked. */
export function rayPick(features, origin, direction, { kinds = null } = {}) {
  const [ox, oy, oz] = origin;
  const dl = Math.hypot(...direction) || 1;
  const [dx, dy, dz] = direction.map(v => v / dl);
  let best = null;
  for (const f of features) {
    if (f.kind === "overlay" || (kinds && !kinds.includes(f.kind))) continue;
    for (const m of f.meshes) {
      const p = m.positions, idx = m.indices;
      for (let t = 0; t < idx.length; t += 3) {
        const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
        const e1x = p[b] - p[a], e1y = p[b + 1] - p[a + 1], e1z = p[b + 2] - p[a + 2];
        const e2x = p[c] - p[a], e2y = p[c + 1] - p[a + 1], e2z = p[c + 2] - p[a + 2];
        const hx = dy * e2z - dz * e2y, hy = dz * e2x - dx * e2z, hz = dx * e2y - dy * e2x;
        const det = e1x * hx + e1y * hy + e1z * hz;
        if (Math.abs(det) < 1e-12) continue;
        const inv = 1 / det;
        const sx = ox - p[a], sy = oy - p[a + 1], sz = oz - p[a + 2];
        const u = inv * (sx * hx + sy * hy + sz * hz);
        if (u < 0 || u > 1) continue;
        const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
        const v = inv * (dx * qx + dy * qy + dz * qz);
        if (v < 0 || u + v > 1) continue;
        const dist = inv * (e2x * qx + e2y * qy + e2z * qz);
        if (dist > 1e-6 && (!best || dist < best.t)) best = { id: f.id, t: dist };
      }
    }
  }
  return best;
}

/** Is the point inside any demolished oriented box? Mirrors the shader exactly. */
export function insideClip(clips, [x, y, z]) {
  return clips.some(c => {
    const cs = Math.cos(c.angle), sn = Math.sin(c.angle);
    const dx = x - c.cx, dz = z - c.cz;
    const u = dx * cs + dz * sn, v = -dx * sn + dz * cs;
    return Math.abs(u) <= c.hx && Math.abs(v) <= c.hz && y >= c.y_min && y <= c.y_max;
  });
}

/** Uniform values for the clip shader: two vec4 per slot, unused slots never match. */
export function clipUniforms(clips) {
  const out = [];
  for (let i = 0; i < MAX_CLIPS; i++) {
    const c = clips[i];
    out.push(c ? [[c.cx, c.cz, c.hx, c.hz], [Math.cos(c.angle), Math.sin(c.angle), c.y_min, c.y_max]]
               : [[0, 0, -1, -1], [1, 0, 1, -1]]);
  }
  return out;
}

/** gsplatModifyVS replacement: splats whose centre is in a demolished box become transparent. */
export function clipShader(language) {
  const slots = [...Array(MAX_CLIPS).keys()];
  if (language === "wgsl") {
    return [
      ...slots.flatMap(i => [`uniform uPlanClipA${i}: vec4f;`, `uniform uPlanClipB${i}: vec4f;`]),
      "fn planInside(p: vec3f, a: vec4f, b: vec4f) -> bool {",
      "  let d = vec2f(p.x - a.x, p.z - a.y);",
      "  let u = d.x * b.x + d.y * b.y;",
      "  let v = -d.x * b.y + d.y * b.x;",
      "  return abs(u) <= a.z && abs(v) <= a.w && p.y >= b.z && p.y <= b.w;",
      "}",
      "fn modifySplatCenter(center: ptr<function, vec3f>) {}",
      "fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f, rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {}",
      "fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {",
      ...slots.map(i => `  if (planInside(center, uniform.uPlanClipA${i}, uniform.uPlanClipB${i})) { (*color).a = 0.0; }`),
      "}",
    ].join("\n");
  }
  return [
    ...slots.flatMap(i => [`uniform vec4 uPlanClipA${i};`, `uniform vec4 uPlanClipB${i};`]),
    "bool planInside(vec3 p, vec4 a, vec4 b) {",
    "  vec2 d = vec2(p.x - a.x, p.z - a.y);",
    "  float u = d.x * b.x + d.y * b.y;",
    "  float v = -d.x * b.y + d.y * b.x;",
    "  return abs(u) <= a.z && abs(v) <= a.w && p.y >= b.z && p.y <= b.w;",
    "}",
    "void modifySplatCenter(inout vec3 center) {}",
    "void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {}",
    "void modifySplatColor(vec3 center, inout vec4 color) {",
    ...slots.map(i => `  if (planInside(center, uPlanClipA${i}, uPlanClipB${i})) { color.a = 0.0; }`),
    "}",
  ].join("\n");
}
