// DOM/engine-free workspace contract and camera math, shared with Node tests.
import { previewMeasurement } from "./measurement_math.js";
import { validatePlan } from "./plan_core.js";
const MODES = ["orbit", "fly", "walk"];
const LAYERS = ["splats", "cameras", "points", "coverage", "collider", "semantics"];
export const MAX_PICKS = 128;
export const finitePoint = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/** Ignore unrelated/untrusted messages; reject malformed commands from the host. */
export function readCommand(event, parent, origin) {
  if (event.source !== parent || event.origin !== origin) return null;
  const data = event.data;
  if (!data || data.namespace !== "groundcontrol" || data.type !== "command") return null;
  const result = { command: data.command };
  const fields = ["namespace", "type", "command"];
  switch (data.command) {
    case "mode":
      if (!MODES.includes(data.value)) throw new Error("Invalid mode value");
      result.value = data.value; fields.push("value"); break;
    case "layer":
      if (!LAYERS.includes(data.layer)) throw new Error("Invalid layer command");
      result.layer = data.layer; fields.push("layer");
      if (typeof data.value !== "boolean") throw new Error("Layer value must be boolean");
      result.value = data.value; fields.push("value"); break;
    case "pick":
      if (typeof data.value !== "boolean") throw new Error("Pick value must be boolean");
      result.value = data.value; fields.push("value");
      if (data.limit !== undefined) {
        if (!Number.isInteger(data.limit) || data.limit < 1 || data.limit > MAX_PICKS) throw new Error("Invalid pick limit");
        result.limit = data.limit; fields.push("limit");
      }
      break;
    case "edit":
      if (typeof data.value !== "boolean") throw new Error("Edit value must be boolean");
      result.value = data.value; fields.push("value"); break;
    case "frame":
      if (!Number.isSafeInteger(data.index) || data.index < 0) throw new Error("Invalid frame index");
      result.index = data.index; fields.push("index"); break;
    case "measurement-tool":
      if (![null, "point", "distance", "height", "area", "volume"].includes(data.kind)) throw new Error("Invalid measurement kind");
      if (!["m", "units"].includes(data.unit)) throw new Error("Invalid measurement unit");
      result.kind = data.kind; result.unit = data.unit; fields.push("kind", "unit"); break;
    case "focus-placement":
      if (typeof data.id !== "string" || !data.id.trim() || data.id.length > 256) throw new Error("Invalid placement id");
      result.id = data.id; fields.push("id"); break;
    case "fit": case "clear-picks": case "snapshot": case "get-state": break;
    case "measurements":
      if (!Array.isArray(data.value) || data.value.length > 500) throw new Error("measurements value must be a bounded array");
      result.value = data.value; fields.push("value"); break;
    case "placements":
      if (!Array.isArray(data.value) || data.value.length > 500) throw new Error("placements value must be a bounded array");
      result.value = data.value; fields.push("value"); break;
    case "set-picks":
      if (!Array.isArray(data.value) || data.value.length > MAX_PICKS || !data.value.every(finitePoint)) throw new Error("set-picks value must be a bounded array of points");
      result.value = data.value; fields.push("value"); break;
    case "select":
      if (data.id !== null && typeof data.id !== "string") throw new Error("select id must be a string or null");
      result.id = data.id; fields.push("id"); break;
    case "labels":
      if (typeof data.value !== "boolean") throw new Error("labels value must be boolean");
      result.value = data.value; fields.push("value"); break;
    case "view":
      if (!["fit", "top"].includes(data.value)) throw new Error("Invalid view value");
      result.value = data.value; fields.push("value"); break;
    case "plan":
      result.value = validatePlan(data.value); fields.push("value"); break;
    case "plan-select": case "camera-follow":
      if (typeof data.value !== "boolean") throw new Error(`${data.command} value must be boolean`);
      result.value = data.value; fields.push("value"); break;
    case "camera-set": {
      const o = data.value;
      if (!o || !finitePoint(o.target) || !["distance", "yaw", "pitch"].every(k => Number.isFinite(o[k])) || !(o.distance > 0) ||
          Math.abs(o.pitch) >= Math.PI / 2 || Object.keys(o).some(k => !["target", "distance", "yaw", "pitch"].includes(k))) {
        throw new Error("camera-set value must be a finite orbit {target, distance, yaw, pitch}");
      }
      result.value = { target: [...o.target], distance: o.distance, yaw: o.yaw, pitch: o.pitch }; fields.push("value"); break;
    }
    default: throw new Error("Unknown workspace command");
  }
  if (Object.keys(data).some(key => !fields.includes(key))) throw new Error("Unexpected command field");
  return result;
}

/** Early-installed bridge: get-state is safe before loading and after failures. */
export class WorkspaceProtocol {
  constructor({ parent, origin, send, getState, getDetails, execute }) {
    Object.assign(this, { parent, origin, send, getState, getDetails, execute });
    this.ready = false;
    this.error = null;
  }
  emit(type, data = {}) { this.send({ namespace: "groundcontrol", type, ...data }); }
  state() { this.emit("state", this.getState()); }
  replay() {
    if (this.ready) {
      const { mode, layers } = this.getState();
      this.emit("ready", { mode, layers, ...this.getDetails() });
    }
    this.state();
    if (this.error) this.emit("error", { message: this.error });
  }
  markReady() { this.ready = true; this.error = null; this.replay(); }
  fail(message) {
    this.ready = false;
    this.error = String(message);
    this.emit("error", { message: this.error });
  }
  receive(event) {
    try {
      const cmd = readCommand(event, this.parent, this.origin);
      if (!cmd) return;
      if (cmd.command === "get-state") { this.replay(); return; }
      if (!this.ready) throw new Error(this.error || "Viewer is not ready; wait for the ready event");
      this.execute(cmd);
      this.state();
    } catch (e) {
      this.emit("error", { message: e.message || String(e) });
    }
  }
}

export function workspaceAssetPath(path) {
  const match = typeof path === "string" && path.match(/^\/(?:runtime\/)?work\/([A-Za-z0-9_.-]+)\/viewer_assets\/?$/);
  if (!match || match[1] === "." || match[1] === "..") throw new Error("Invalid workspace scene asset path");
  return path.replace(/\/$/, "");
}

export function layerState(layers, layer, value, capabilities) {
  if (!LAYERS.includes(layer) || typeof value !== "boolean") throw new Error("Invalid layer value");
  if (value && layer !== "splats" && !capabilities[layer]) throw new Error(`${layer} layer is unavailable`);
  return { ...layers, [layer]: value };
}

export function boundsFromPoints(points) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  let count = 0;
  for (const p of points) {
    if (!finitePoint(p)) continue;
    count++;
    for (let i = 0; i < 3; i++) { min[i] = Math.min(min[i], p[i]); max[i] = Math.max(max[i], p[i]); }
  }
  return count ? { min, max } : null;
}

/** Fit the enclosing sphere using the smaller horizontal/vertical field of view. */
export function fitOrbit(bounds, fov, aspect) {
  if (!bounds || !finitePoint(bounds.min) || !finitePoint(bounds.max)) throw new Error("Scene bounds are unavailable");
  const target = bounds.min.map((v, i) => (v + bounds.max[i]) / 2);
  const radius = Math.max(0.01, Math.hypot(...bounds.max.map((v, i) => (v - bounds.min[i]) / 2)));
  const halfFov = clamp(fov, 1, 179) * Math.PI / 360;
  const limiting = Math.min(halfFov, Math.atan(Math.tan(halfFov) * Math.max(0.01, aspect)));
  const distance = radius * 1.15 / Math.sin(limiting);
  return { target, distance, yaw: Math.PI / 4, pitch: Math.PI / 7,
    minDistance: Math.max(radius * 0.001, 0.001), maxDistance: distance * 50 };
}

export function orbitEye(s) {
  const cp = Math.cos(s.pitch);
  return [s.target[0] + s.distance * Math.sin(s.yaw) * cp,
    s.target[1] + s.distance * Math.sin(s.pitch),
    s.target[2] + s.distance * Math.cos(s.yaw) * cp];
}

/** Adopt a camera without the next orbit/fly update undoing its selected pose. */
export function orbitFromPose(eye, forward, distance) {
  if (!finitePoint(eye) || !finitePoint(forward) || !(Math.hypot(...forward) > 0) || !(distance > 0)) {
    throw new Error("Invalid camera pose or direction");
  }
  const n = Math.hypot(...forward), f = forward.map(v => v / n);
  return { target: eye.map((v, i) => v + f[i] * distance), distance,
    yaw: Math.atan2(-f[0], -f[2]), pitch: Math.asin(clamp(-f[1], -1, 1)),
    minDistance: 0.001, maxDistance: Math.max(1000, distance * 50) };
}

export function rotateOrbit(s, yawDelta, pitchDelta) {
  return { ...s, yaw: s.yaw + yawDelta, pitch: clamp(s.pitch + pitchDelta, -Math.PI / 2 + 0.001, Math.PI / 2 - 0.001) };
}

export function panOrbit(s, dx, dy, height, fov) {
  const scale = 2 * s.distance * Math.tan(fov * Math.PI / 360) / Math.max(1, height);
  const sy = Math.sin(s.yaw), cy = Math.cos(s.yaw), sp = Math.sin(s.pitch), cp = Math.cos(s.pitch);
  const right = [cy, 0, -sy], up = [-sy * sp, cp, -cy * sp];
  return { ...s, target: s.target.map((v, i) => v + scale * (-dx * right[i] + dy * up[i])) };
}

export function zoomOrbit(s, delta) {
  return { ...s, distance: clamp(s.distance * Math.exp(clamp(delta * 0.001, -50, 50)), s.minDistance, s.maxDistance) };
}

/** PlayCanvas screenToWorld consumes CSS pixels relative to its canvas. */
export function canvasPoint(clientX, clientY, rect) {
  if (!(rect.width > 0 && rect.height > 0) || !Number.isFinite(clientX) || !Number.isFinite(clientY)) return null;
  const x = clientX - rect.left, y = clientY - rect.top;
  return x >= 0 && y >= 0 && x <= rect.width && y <= rect.height ? [x, y] : null;
}

export function collisionSurfacePoint(hit, meshEntities) {
  if (!hit || !meshEntities.has(hit.entity) || hit.entity.rigidbody?.type !== "static" || hit.entity.collision?.type !== "mesh") return null;
  const point = [hit.point?.x, hit.point?.y, hit.point?.z];
  return finitePoint(point) ? point : null;
}

export function appendPick(points, point, limit = MAX_PICKS) {
  if (!finitePoint(point)) throw new Error("Invalid measurement point");
  if (points.length >= limit) throw new Error(`Measurement limit (${limit}) reached; clear picks to continue`);
  return [...points, [...point]];
}

/**
 * Horizontal distance walked in one frame, honestly.
 *
 * Only counts while the body is grounded (falling, jumping and the fall-recovery
 * teleport are not walking), rejects a frame that moved a whole body-length (a
 * respawn teleport, not a step) and a sub-millimetre floor (physics jitter against
 * a wall is not travel). prev/next are [x, z] horizontal pairs.
 */
export function walkDistance(prev, next, grounded, charScale, jitter = 1e-4) {
  if (!grounded || !prev || !next) return 0;
  const jump = Math.hypot(next[0] - prev[0], next[1] - prev[1]);
  if (!(jump < charScale) || jump < jitter) return 0;
  return jump;
}

const MEASURE_COLORS = { point: [0.98, 0.66, 0.24], distance: [0.98, 0.66, 0.24], height: [0.36, 0.78, 0.52], area: [0.36, 0.62, 0.94], volume: [0.71, 0.55, 0.85] };
export function measureColor(kind) { return MEASURE_COLORS[kind] || MEASURE_COLORS.point; }

/** How a stored measurement should be drawn: a point, an open line, or a closed ring. */
export function measureRenderable(m) {
  const pts = Array.isArray(m.points) ? m.points.filter(finitePoint) : [];
  if (pts.length < 2) return { type: "point", points: pts };
  if (m.kind === "area" || m.kind === "volume") { const ring = pts.slice(); if (JSON.stringify(ring[0]) !== JSON.stringify(ring[ring.length - 1])) ring.push(ring[0]); return { type: "polygon", points: ring }; }
  return { type: "line", points: pts };
}

/** The world point a floating label attaches to: the point, a line midpoint, or a polygon centroid. */
export function measureAnchor(m) {
  const r = measureRenderable(m);
  const pts = r.points;
  if (!pts.length) return null;
  if (r.type === "point") return pts[0];
  if (r.type === "line") { const a = pts[0], b = pts[pts.length - 1]; return a.map((v, i) => (v + b[i]) / 2); }
  const n = pts.length - 1 || 1;
  return [0, 1, 2].map(i => pts.slice(0, n).reduce((s, p) => s + p[i], 0) / n);
}

/** A compact label; unsupported geometry is an estimate, not a validity upgrade. */
export function formatMeasureLabel(m) {
  const name = String(m.label || m.text || m.kind || "Measurement").slice(0, 80);
  if (m.kind === "point" || m.kind === "annotation") return name;
  if (!Number.isFinite(m.value) || m.valid === false) {
    const points = Array.isArray(m.points) ? m.points.filter(finitePoint) : [];
    const unit = String(m.unit || "m").startsWith("units") ? "units" : "m";
    const kind = ["distance", "height", "area", "volume"].includes(m.kind) ? m.kind : null;
    const estimate = previewMeasurement(kind, points, unit);
    if (estimate.value !== null) return `${name} · ~ ${round(estimate.value)} ${estimate.unit} · surface estimate`;
    if (kind === "volume" && points.length > 1) return `${name} · ~ ${round(estimate.perimeter)} ${unit} footprint perimeter · surface estimate`;
    return `${name} · insufficient picked geometry`;
  }
  const unc = m.uncertainty && Number.isFinite(m.uncertainty.m) ? ` ±${round(m.uncertainty.m)}` : "";
  return `${name} · ${round(m.value)}${m.unit || ""}${unc}`;
}

/** Fixed picks remain visible after the last click, even before the host saves. */
export function liveMeasurement(kind, picks, cursor, picking, unit = "m") {
  if (kind === null) return null;
  const points = picks.filter(finitePoint).map(p => [...p]);
  if (picking && finitePoint(cursor) && !points.at(-1)?.every((v, i) => v === cursor[i])) points.push([...cursor]);
  if (!points.length) return null;
  const record = { kind, points };
  const preview = previewMeasurement(kind, points, unit), labels = [];
  const ring = measureRenderable(record).points;
  let total = null;
  if (preview.value !== null) {
    total = `${kind === "height" ? "Height" : kind === "area" ? "Area" : "Total"} · ${round(preview.value)} ${preview.unit}`;
  } else if (kind === "volume" && points.length > 1) {
    total = `Footprint · ${round(preview.perimeter)} ${unit} perimeter · volume unavailable`;
  } else if (kind === "point" || kind === "annotation") {
    total = `Point · ${points[0].map(round).join(", ")} ${unit}`;
  }
  // A lone segment that repeats the total is noise; one that adds information
  // (a height's slant length) still earns its own label.
  const redundant = preview.segments.length === 1 && total !== null && preview.value === preview.segments[0];
  if (!redundant) for (let i = 0; i < preview.segments.length; i++) {
    const a = ring[i], b = ring[i + 1];
    if (!a || !b) continue;
    labels.push({ id: `live:segment:${i}`, role: "segment", text: `${round(preview.segments[i])} ${unit}`,
      anchor: a.map((v, j) => (v + b[j]) / 2), priority: 80 });
  }
  if (total) labels.push({ id: "live:total", role: "total", text: total, anchor: measureAnchor(record), priority: 100 });
  return { ...record, preview, labels };
}

/** Coalesce the latest cursor sample, including null clears, to at most 15 Hz. */
export class PreviewStream {
  constructor(emit) { this.emit = emit; this.last = -Infinity; this.dirty = false; this.point = null; }
  update(point) { this.point = finitePoint(point) ? [...point] : null; this.dirty = true; }
  flush(now) {
    if (!this.dirty || now - this.last < 1000 / 15) return;
    this.last = now; this.dirty = false;
    this.emit("preview", { point: this.point });
  }
}

/** Explicit drawing-buffer to CSS conversion; X and Y scales need not match. */
export function projectedCssPoint(s, rect, bufferWidth, bufferHeight) {
  if (![s.x, s.y, s.z, rect.width, rect.height, bufferWidth, bufferHeight].every(Number.isFinite) ||
      s.z <= 0 || rect.width <= 0 || rect.height <= 0 || bufferWidth <= 0 || bufferHeight <= 0) return null;
  const x = s.x * rect.width / bufferWidth, y = s.y * rect.height / bufferHeight;
  return x >= 0 && x <= rect.width && y >= 0 && y <= rect.height ? [x, y] : null;
}

/** Priority-first, bounded label packing with a small set of anchor offsets. */
export function layoutLabels(candidates, width, height) {
  const placed = [];
  for (const c of [...candidates].sort((a, b) => b.priority - a.priority)) {
    if (placed.length >= 32) break;
    if (c.x < 0 || c.x > width || c.y < 0 || c.y > height || c.width > width - 8) continue;
    const left = Math.max(4, Math.min(width - c.width - 4, c.x - c.width / 2));
    for (const dy of [-c.height - 12, -2 * c.height - 18, 12, c.height + 18, -3 * c.height - 24]) {
      const top = c.y + dy;
      if (top < 4 || top + c.height > height - 4) continue;
      if (placed.some(p => left < p.left + p.width + 4 && left + c.width + 4 > p.left && top < p.top + p.height + 4 && top + c.height + 4 > p.top)) continue;
      placed.push({ ...c, left, top }); break;
    }
  }
  return placed;
}

/**
 * World geometry for a placed furniture box, matching the physics shape and the
 * detected-furniture outline exactly: the major (length) axis runs along
 * (cos yaw, 0, sin yaw), corners emitted bottom/top per footprint quad. Returns
 * null for a malformed record so a bad placement can never crash the render.
 */
export function placementBox(p) {
  if (!p || !Array.isArray(p.center_xz) || p.center_xz.length !== 2 || !p.center_xz.every(Number.isFinite)) return null;
  if (!Number.isFinite(p.center_y) || !Array.isArray(p.size) || p.size.length !== 3 || !p.size.every(Number.isFinite)) return null;
  if (p.size.some(v => v <= 0)) return null;
  const yaw = Number.isFinite(p.yaw_deg) ? p.yaw_deg : 0;
  const a = yaw * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
  const [cx, cz] = p.center_xz, [L, H, D] = p.size, hu = L / 2, hw = D / 2;
  const y0 = p.center_y - H / 2, y1 = p.center_y + H / 2;
  const corners = [];
  for (const [qu, qv] of [[-hu, -hw], [hu, -hw], [hu, hw], [-hu, hw]]) {
    const x = cx + qu * ca - qv * sa, z = cz + qu * sa + qv * ca;
    corners.push([x, y0, z], [x, y1, z]);
  }
  return { corners, center: [cx, p.center_y, cz], halfExtents: [hu, H / 2, hw], yaw };
}

export function placementSignature(placements) {
  return JSON.stringify(placements.map(p => [p.id, p.item, p.center_xz, p.center_y, p.size, p.yaw_deg, p.stale, p.fit]));
}

export function focusPlacement(placements, id, fov, aspect) {
  const box = placementBox(placements.find(p => p.id === id));
  if (!box) throw new Error("Selected placement is unavailable");
  return fitOrbit(boundsFromPoints(box.corners), fov, aspect);
}

/**
 * Which placed item a camera ray passes through first. Ammo answers a filtered
 * ray with nothing when the nearest body is not the one being looked for, so a
 * piece standing in front of the scan shell can never be selected through
 * physics; the boxes are known, so test the ray against them directly.
 */
export function placementRay(placements, from, to) {
  if (!finitePoint(from) || !finitePoint(to)) return null;
  const d = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
  const length = Math.hypot(...d);
  if (!(length > 1e-6)) return null;
  const dir = d.map(v => v / length);
  let best = null, bestT = Infinity;
  for (const p of placements) {
    const box = placementBox(p);
    if (!box) continue;
    const [cx, cy, cz] = box.center, [hu, hv, hw] = box.halfExtents;
    const a = -(box.yaw * Math.PI / 180), ca = Math.cos(a), sa = Math.sin(a);
    const ox = (from[0] - cx) * ca - (from[2] - cz) * sa, oz = (from[0] - cx) * sa + (from[2] - cz) * ca;
    const dx = dir[0] * ca - dir[2] * sa, dz = dir[0] * sa + dir[2] * ca;
    const o = [ox, from[1] - cy, oz], v = [dx, dir[1], dz];
    const half = [hu, hv, hw];
    let enter = 0, exit = length, ok = true;
    for (let i = 0; i < 3; i++) {
      if (Math.abs(v[i]) < 1e-9) { if (Math.abs(o[i]) > half[i]) { ok = false; break; } continue; }
      let t0 = (-half[i] - o[i]) / v[i], t1 = (half[i] - o[i]) / v[i];
      if (t0 > t1) { const swap = t0; t0 = t1; t1 = swap; }
      enter = Math.max(enter, t0); exit = Math.min(exit, t1);
      if (enter > exit) { ok = false; break; }
    }
    if (ok && enter <= length && enter < bestT) { best = p.id; bestT = enter; }
  }
  return best;
}

/** Colour a placed box by its fit verdict: green fits, red does not, blue unknown. */
export function placementColor(p) {
  const fit = p && p.fit;
  if (!fit || fit.valid === undefined) return [0.42, 0.72, 0.98];
  return fit.valid ? [0.36, 0.82, 0.52] : [0.94, 0.36, 0.32];
}

/**
 * Transform that maps an imported model's natural world AABB onto its placement
 * box, in the box's own local frame (the caller supplies yaw/centre on the
 * wrapper node). scale/offset make the model fill the conservative collider
 * exactly, so what you see is what the character bumps into. Returns null when
 * either input is unusable, so a malformed model can never divide by zero.
 */
export function importedModelFit(modelAabb, box) {
  if (!modelAabb || !box || !Array.isArray(modelAabb.min) || !Array.isArray(modelAabb.max) ||
      !finitePoint(modelAabb.min) || !finitePoint(modelAabb.max) ||
      !Array.isArray(box.halfExtents) || !box.halfExtents.every(Number.isFinite)) return null;
  const size = [0, 1, 2].map(i => modelAabb.max[i] - modelAabb.min[i]);
  const center = [0, 1, 2].map(i => (modelAabb.max[i] + modelAabb.min[i]) / 2);
  if (size.some(v => !Number.isFinite(v) || v <= 1e-6)) return null;
  const scale = [0, 1, 2].map(i => (box.halfExtents[i] * 2) / size[i]);
  if (scale.some(v => !Number.isFinite(v) || v <= 0)) return null;
  return { scale, offset: [0, 1, 2].map(i => -center[i] * scale[i]) };
}

/** "Sofa · 2×0.9 m · fits" style label; clearance is to unscanned space, never a wall. */
export function placementLabel(p) {
  const name = (p.label || p.item || "Item").toString().slice(0, 40);
  const dims = Array.isArray(p.size) ? `${round(p.size[0])}×${round(p.size[2])} m` : "";
  const fit = p.fit || {};
  // The stored reason carries the standing caveats; the 3D label keeps only the
  // verdict, and the panel is where the limitations are spelled out.
  let verdict;
  if (!fit.valid) verdict = (fit.reason || "no measured floor").split(";")[0];
  else verdict = fit.tight && fit.floor_clearance_m != null ? `fits · ${fit.floor_clearance_m} m to unscanned space` : "fits";
  return `${name} · ${dims} · ${verdict}`;
}
const round = v => Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 });

/** A single pending capture, flushed synchronously after the renderer draws. */
export class SnapshotCapture {
  constructor(emit) { this.emit = emit; this.pending = false; }
  request() {
    if (this.pending) throw new Error("A snapshot is already pending");
    this.pending = true;
  }
  postrender(capture) {
    if (!this.pending) return;
    this.pending = false;
    try {
      const dataUrl = capture();
      if (typeof dataUrl !== "string" || !/^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(dataUrl)) {
        throw new Error("Renderer did not return a PNG image");
      }
      this.emit("snapshot", { dataUrl });
    } catch (e) {
      this.emit("error", { message: `Snapshot failed: ${e.message || e}` });
    }
  }
}
