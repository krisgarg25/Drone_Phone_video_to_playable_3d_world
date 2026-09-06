/**
 * navstats.mjs -- small shared helpers for bake.mjs and verify.mjs.
 *
 * Everything here works on the nav.json triangle-soup contract: `verts` is a
 * flat [x,y,z,...] array and `tris` a flat [i0,i1,i2,...] array indexing it.
 */

import * as fs from 'node:fs';

/** Axis-aligned bounds of a flat position array. */
export function boundsOf(positions) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i], y = positions[i + 1], z = positions[i + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

/**
 * Quantise positions to `decimals` and merge vertices that land on the same
 * rounded coordinate. This is what keeps nav.json small: the raw navmesh output
 * repeats every vertex per triangle, and rounding to millimetres collapses the
 * shared corners into a real indexed mesh.
 */
export function weldRounded(positions, indices, decimals = 3) {
  const q = Math.pow(10, decimals);
  const map = new Map();
  const out = [];
  const remap = new Int32Array(positions.length / 3);
  for (let v = 0; v < positions.length / 3; v++) {
    const ix = Math.round(positions[v * 3] * q);
    const iy = Math.round(positions[v * 3 + 1] * q);
    const iz = Math.round(positions[v * 3 + 2] * q);
    const key = `${ix},${iy},${iz}`;
    let id = map.get(key);
    if (id === undefined) {
      id = out.length / 3;
      map.set(key, id);
      out.push(ix / q, iy / q, iz / q);
    }
    remap[v] = id;
  }
  const idx = new Uint32Array(indices.length);
  for (let i = 0; i < indices.length; i++) idx[i] = remap[indices[i]];
  // drop triangles that collapsed to a line/point by welding
  let kept = 0;
  for (let t = 0; t < idx.length / 3; t++) {
    const a = idx[t * 3], b = idx[t * 3 + 1], c = idx[t * 3 + 2];
    if (a === b || b === c || a === c) continue;
    idx[kept * 3] = a; idx[kept * 3 + 1] = b; idx[kept * 3 + 2] = c;
    kept++;
  }
  return { positions: new Float32Array(out), indices: idx.subarray(0, kept * 3) };
}

/** Total triangle area (3D). Pass xzOnly for the plan/footprint area. */
export function totalArea(positions, indices, xzOnly = false) {
  let sum = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t] * 3, b = indices[t + 1] * 3, c = indices[t + 2] * 3;
    const ux = positions[b] - positions[a], uy = positions[b + 1] - positions[a + 1], uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a], vy = positions[c + 1] - positions[a + 1], vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    sum += xzOnly ? Math.abs(ny) / 2 : Math.hypot(nx, ny, nz) / 2;
  }
  return sum;
}

/**
 * Connected components of the triangle soup, glued along shared vertices.
 *
 * @returns {{count: number, sizes: number[], largest: number, label: Int32Array,
 *            areas: number[]}} `sizes` and `areas` are sorted largest first,
 *          `label[t]` is the component index of triangle t (into `sizes`).
 */
export function components(positions, indices) {
  const nv = positions.length / 3;
  const nt = indices.length / 3;
  const parent = new Int32Array(nt);
  for (let i = 0; i < nt; i++) parent[i] = i;
  const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  const union = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[rb] = ra; };

  const first = new Int32Array(nv).fill(-1);
  for (let t = 0; t < nt; t++) {
    for (let k = 0; k < 3; k++) {
      const v = indices[t * 3 + k];
      if (first[v] === -1) first[v] = t;
      else union(t, first[v]);
    }
  }

  const roots = new Map(); // root -> index into sizes
  const sizes = [];
  const areas = [];
  const label = new Int32Array(nt);
  for (let t = 0; t < nt; t++) {
    const r = find(t);
    let ci = roots.get(r);
    if (ci === undefined) { ci = sizes.length; roots.set(r, ci); sizes.push(0); areas.push(0); }
    label[t] = ci;
    sizes[ci]++;
    const a = indices[t * 3] * 3, b = indices[t * 3 + 1] * 3, c = indices[t * 3 + 2] * 3;
    const ux = positions[b] - positions[a], uy = positions[b + 1] - positions[a + 1], uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a], vy = positions[c + 1] - positions[a + 1], vz = positions[c + 2] - positions[a + 2];
    areas[ci] += Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) / 2;
  }
  const order = sizes.map((s, i) => i).sort((a, b) => sizes[b] - sizes[a]);
  const remap = new Int32Array(sizes.length);
  order.forEach((old, n) => { remap[old] = n; });
  for (let t = 0; t < nt; t++) label[t] = remap[label[t]];
  return {
    count: sizes.length,
    sizes: order.map((i) => sizes[i]),
    areas: order.map((i) => +areas[i].toFixed(2)),
    largest: sizes[order[0]] ?? 0,
    label,
  };
}

/** Wavefront OBJ of a triangle soup, for eyeballing in a 3D viewer. */
export function writeObj(path, positions, indices, objectName = 'navmesh') {
  const lines = [`# navmesh debug export`, `o ${objectName}`];
  for (let i = 0; i < positions.length; i += 3) {
    lines.push(`v ${positions[i]} ${positions[i + 1]} ${positions[i + 2]}`);
  }
  for (let i = 0; i < indices.length; i += 3) {
    lines.push(`f ${indices[i] + 1} ${indices[i + 1] + 1} ${indices[i + 2] + 1}`);
  }
  fs.writeFileSync(path, lines.join('\n') + '\n');
  return lines.length;
}

// ------------------------------------------------------- XZ queries (bake+verify)
// "Can a bot stand here" is a 2D question on the XZ plane; the mesh height is
// sampled off whichever triangle wins, exactly like a runtime that walks on the
// navmesh does it.

function barycentric2D(px, pz, ax, az, bx, bz, cx, cz) {
  const v0x = bx - ax, v0z = bz - az, v1x = cx - ax, v1z = cz - az;
  const den = v0x * v1z - v1x * v0z;
  if (Math.abs(den) < 1e-12) return null; // degenerate in plan view
  const qx = px - ax, qz = pz - az;
  const u = (qx * v1z - v1x * qz) / den;
  const v = (v0x * qz - qx * v0z) / den;
  if (u < -1e-6 || v < -1e-6 || u + v > 1 + 1e-6) return null;
  return { u, v, w: 1 - u - v };
}

function pointToSegment2D(px, pz, ax, az, bx, bz) {
  const dx = bx - ax, dz = bz - az;
  const len2 = dx * dx + dz * dz;
  const t = len2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (pz - az) * dz) / len2)) : 0;
  return { d: Math.hypot(px - (ax + dx * t), pz - (az + dz * t)), t };
}

/**
 * Distance in XZ from a point to one triangle plus the mesh height there.
 * Inside a triangle the distance is 0 and the height is barycentric; outside,
 * the height comes off the nearest edge, so a point just off a polygon still
 * gets a usable floor y instead of a nonsense one.
 */
export function nearestOnTriangle(P, tri, x, z) {
  const a = tri[0] * 3, b = tri[1] * 3, c = tri[2] * 3;
  const bary = barycentric2D(x, z, P[a], P[a + 2], P[b], P[b + 2], P[c], P[c + 2]);
  if (bary) return { d: 0, y: bary.w * P[a + 1] + bary.u * P[b + 1] + bary.v * P[c + 1], inside: true };
  let best = null;
  for (const [i, j] of [[a, b], [b, c], [c, a]]) {
    const s = pointToSegment2D(x, z, P[i], P[i + 2], P[j], P[j + 2]);
    if (!best || s.d < best.d) best = { d: s.d, y: P[i + 1] + (P[j + 1] - P[i + 1]) * s.t, inside: false };
  }
  return best;
}

/** Uniform XZ grid over a navmesh triangle soup, with an exact expanding search. */
export class XzGrid {
  constructor(positions, indices, cell = 2) {
    this.P = positions;
    this.cell = cell;
    this.tris = [];
    for (let t = 0; t < indices.length; t += 3) this.tris.push([indices[t], indices[t + 1], indices[t + 2]]);
    const bb = boundsOf(positions);
    this.ox = bb.min[0];
    this.oz = bb.min[2];
    this.buckets = new Map();
    this.tris.forEach((tri, n) => {
      const xs = [], zs = [];
      for (const v of tri) { xs.push(this.P[v * 3]); zs.push(this.P[v * 3 + 2]); }
      for (let gx = Math.floor((Math.min(...xs) - this.ox) / cell); gx <= Math.floor((Math.max(...xs) - this.ox) / cell); gx++) {
        for (let gz = Math.floor((Math.min(...zs) - this.oz) / cell); gz <= Math.floor((Math.max(...zs) - this.oz) / cell); gz++) {
          const k = `${gx},${gz}`;
          let list = this.buckets.get(k);
          if (!list) this.buckets.set(k, (list = []));
          list.push(n);
        }
      }
    });
  }

  /** Nearest triangle in plan view. Stops as soon as a wider ring cannot hold
   *  anything closer, so the answer is exact even when nothing is nearby. */
  nearest(x, z, maxRadius = 400) {
    const g = this.cell;
    const cx = Math.floor((x - this.ox) / g);
    const cz = Math.floor((z - this.oz) / g);
    let best = null;
    const maxRing = Math.ceil(maxRadius / g);
    for (let ring = 0; ring <= maxRing; ring++) {
      if (best && best.d <= (ring - 1) * g) break;
      for (let ix = cx - ring; ix <= cx + ring; ix++) {
        for (let iz = cz - ring; iz <= cz + ring; iz++) {
          if (Math.abs(ix - cx) !== ring && Math.abs(iz - cz) !== ring) continue; // done in an inner ring
          const list = this.buckets.get(`${ix},${iz}`);
          if (!list) continue;
          for (const n of list) {
            const r = nearestOnTriangle(this.P, this.tris[n], x, z);
            if (!best || r.d < best.d) best = { d: r.d, y: r.y, inside: r.inside, tri: n };
          }
        }
      }
    }
    return best ?? { d: Infinity, y: null, inside: false, tri: -1 };
  }
}

/** Points along a walk loop every `step` metres, closing leg included: the
 *  viewer's autopilot indexes collision.json's walk_path cyclically. */
export function walkPathSamples(loop, step) {
  const out = [];
  if (!loop || !loop.length) return out;
  if (loop.length === 1) return [[loop[0][0], loop[0][1]]];
  for (let i = 0; i < loop.length; i++) {
    const a = loop[i], b = loop[(i + 1) % loop.length];
    const dx = b[0] - a[0], dz = b[1] - a[1];
    const n = Math.max(1, Math.round(Math.hypot(dx, dz) / step));
    for (let k = 0; k < n; k++) out.push([a[0] + (dx * k) / n, a[1] + (dz * k) / n]);
  }
  out.push([loop[loop.length - 1][0], loop[loop.length - 1][1]]);
  return out;
}
