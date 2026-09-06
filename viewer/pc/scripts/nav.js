/* Navmesh: loads a baked walkable surface, paths over it, and ranks AI spots. */

const VERT_STRIDE = 3;

function edgeKey(a, b) {
  return a < b ? a * 1e7 + b : b * 1e7 + a;
}

function triArea2(ax, az, bx, bz, cx, cz) {
  return Math.abs((bx - ax) * (cz - az) - (cx - ax) * (bz - az)) * 0.5;
}

class MinHeap {
  constructor() {
    this.items = [];
  }
  push(node, cost) {
    const items = this.items;
    items.push({ node, cost });
    let i = items.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (items[p].cost <= items[i].cost) break;
      const t = items[p]; items[p] = items[i]; items[i] = t;
      i = p;
    }
  }
  pop() {
    const items = this.items;
    const top = items[0];
    const last = items.pop();
    if (items.length) {
      items[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < items.length && items[l].cost < items[m].cost) m = l;
        if (r < items.length && items[r].cost < items[m].cost) m = r;
        if (m === i) break;
        const t = items[m]; items[m] = items[i]; items[i] = t;
        i = m;
      }
    }
    return top;
  }
  get size() {
    return this.items.length;
  }
}

export class Nav {
  constructor(verts, tris, meta = {}) {
    this.verts = verts;
    this.tris = tris;
    this.meta = meta;
    this.source = meta.source || "unknown";
    this.adjacent = [];
    this._build();
  }

  /**
   * Canonical vertex ids keyed on quantised position, so triangles that meet in
   * space are linked even when the source never welded its corners. 1 cm absorbs
   * coordinate rounding without ever merging two storeys.
   */
  _welded() {
    const { verts } = this;
    const n = verts.length / VERT_STRIDE;
    const ids = new Int32Array(n);
    const map = new Map();
    let next = 0;
    for (let i = 0; i < n; i++) {
      const key = `${Math.round(verts[i * VERT_STRIDE] * 100)}|` +
        `${Math.round(verts[i * VERT_STRIDE + 1] * 100)}|` +
        `${Math.round(verts[i * VERT_STRIDE + 2] * 100)}`;
      let id = map.get(key);
      if (id === undefined) { id = next++; map.set(key, id); }
      ids[i] = id;
    }
    return ids;
  }

  _build() {
    const { verts, tris } = this;
    this.triCount = tris.length / 3;
    this.cx = new Float32Array(this.triCount);
    this.cy = new Float32Array(this.triCount);
    this.cz = new Float32Array(this.triCount);
    this.adjacent = [];
    if (!this.triCount) {
      this.areaM2 = 0;
      this.bounds = { minX: 0, minZ: 0, maxX: 0, maxZ: 0 };
      this.cell = 1;
      this.gridW = 1;
      this.gridH = 1;
      this.grid = [];
      return;
    }
    const edgeMap = new Map();
    const weld = this._welded();
    let minX = Infinity, minZ = Infinity, maxX = -Infinity, maxZ = -Infinity, area = 0;
    for (let t = 0; t < this.triCount; t++) {
      const a = tris[t * 3] * VERT_STRIDE, b = tris[t * 3 + 1] * VERT_STRIDE, c = tris[t * 3 + 2] * VERT_STRIDE;
      const ax = verts[a], ay = verts[a + 1], az = verts[a + 2];
      const bx = verts[b], by = verts[b + 1], bz = verts[b + 2];
      const cx = verts[c], cy = verts[c + 1], cz = verts[c + 2];
      this.cx[t] = (ax + bx + cx) / 3;
      this.cy[t] = (ay + by + cy) / 3;
      this.cz[t] = (az + bz + cz) / 3;
      area += triArea2(ax, az, bx, bz, cx, cz);
      if (this.cx[t] < minX) minX = this.cx[t];
      if (this.cx[t] > maxX) maxX = this.cx[t];
      if (this.cz[t] < minZ) minZ = this.cz[t];
      if (this.cz[t] > maxZ) maxZ = this.cz[t];
      const i0 = tris[t * 3], i1 = tris[t * 3 + 1], i2 = tris[t * 3 + 2];
      const c0 = weld[i0], c1 = weld[i1], c2 = weld[i2];
      for (const [p, q] of [[c0, c1], [c1, c2], [c2, c0]]) {
        const k = edgeKey(p, q);
        const prev = edgeMap.get(k);
        if (prev === undefined) edgeMap.set(k, t);
        else if (prev >= 0) {
          (this.adjacent[prev] ||= []).push(t);
          (this.adjacent[t] ||= []).push(prev);
          edgeMap.set(k, -1);
        }
      }
    }
    this.areaM2 = area;
    this.bounds = { minX, minZ, maxX, maxZ };
    this.cell = Math.min(2, Math.max(0.5, Math.max(maxX - minX, maxZ - minZ) / 48));
    this.gridW = Math.ceil((maxX - minX) / this.cell) + 2;
    this.gridH = Math.ceil((maxZ - minZ) / this.cell) + 2;
    this.grid = new Array(this.gridW * this.gridH);
    for (let t = 0; t < this.triCount; t++) {
      const gx = Math.floor((this.cx[t] - minX) / this.cell) + 1;
      const gz = Math.floor((this.cz[t] - minZ) / this.cell) + 1;
      const i = gz * this.gridW + gx;
      (this.grid[i] ||= []).push(t);
    }
  }

  /** Drop triangles the caller rejects (unobserved scan cells, stranded islands). */
  filter(keep) {
    const outVerts = [];
    const outTris = [];
    let removed = 0;
    for (let t = 0; t < this.triCount; t++) {
      if (!keep(this.cx[t], this.cy[t], this.cz[t], t)) { removed++; continue; }
      for (let k = 0; k < 3; k++) {
        const src = this.tris[t * 3 + k] * VERT_STRIDE;
        outVerts.push(this.verts[src], this.verts[src + 1], this.verts[src + 2]);
        outTris.push(outVerts.length / VERT_STRIDE - 1);
      }
    }
    this.verts = new Float32Array(outVerts);
    this.tris = new Int32Array(outTris);
    this._build();
    this.removed = removed;
    return this;
  }

  /** Membership mask for the connected walkable region containing `tri`. */
  componentOf(tri) {
    const mask = new Uint8Array(this.triCount);
    if (tri < 0 || tri >= this.triCount) return mask;
    const stack = [tri];
    mask[tri] = 1;
    let n = 1;
    while (stack.length) {
      const cur = stack.pop();
      for (const nb of this.adjacent[cur] || []) {
        if (mask[nb]) continue;
        mask[nb] = 1;
        n++;
        stack.push(nb);
      }
    }
    mask.count = n;
    return mask;
  }

  static fromJSON(json) {
    const verts = new Float32Array(json.verts);
    const tris = new Int32Array(json.tris);
    return new Nav(verts, tris, { source: "navmesh", ...json.build, areaM2: json.stats?.areaM2 });
  }

  static async load(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`navmesh unavailable at ${url} (HTTP ${r.status})`);
    const json = await r.json();
    if (!json.tris?.length) throw new Error(`navmesh at ${url} has no triangles`);
    return Nav.fromJSON(json);
  }

  /** Fallback walkable surface from the scan heightfield, so gameplay works unbaked. */
  static fromHeightfield(HF) {
    const { nx, nz, cell, data, ox, oz } = HF;
    const verts = [], tris = [];
    let n = 0;
    for (let gz = 0; gz < nz - 1; gz++) {
      for (let gx = 0; gx < nx - 1; gx++) {
        const x = ox + gx * cell, z = oz + gz * cell;
        const h00 = data[gz * nx + gx], h10 = data[gz * nx + gx + 1];
        const h01 = data[(gz + 1) * nx + gx], h11 = data[(gz + 1) * nx + gx + 1];
        verts.push(x, h00, z, x + cell, h10, z, x, h01, z + cell, x + cell, h11, z + cell);
        tris.push(n, n + 1, n + 3, n, n + 3, n + 2);
        n += 4;
      }
    }
    return new Nav(new Float32Array(verts), new Int32Array(tris), { source: "heightfield" });
  }

  /** Nearest triangle whose 2D projection contains (x,z), falling back to nearest centroid. */
  triAt(x, z, searchR = 3) {
    const { minX, minZ } = this.bounds;
    const gx = Math.floor((x - minX) / this.cell) + 1, gz = Math.floor((z - minZ) / this.cell) + 1;
    const rad = Math.max(1, Math.ceil(searchR / this.cell));
    let best = -1, bestD = searchR * searchR;
    let exact = -1, exactBary = null;
    for (let dz = -rad; dz <= rad; dz++) {
      for (let dx = -rad; dx <= rad; dx++) {
        const idx = (gz + dz) * this.gridW + (gx + dx);
        const bucket = this.grid[idx];
        if (!bucket) continue;
        for (const t of bucket) {
          const bary = this._bary(t, x, z);
          if (bary && exact < 0) { exact = t; exactBary = bary; }
          const d = (this.cx[t] - x) ** 2 + (this.cz[t] - z) ** 2;
          if (d < bestD) { bestD = d; best = t; }
        }
      }
    }
    if (exact >= 0) this._lastBary = exactBary;
    else this._lastBary = null;
    return exact >= 0 ? exact : best;
  }

  _bary(t, x, z) {
    const { verts, tris } = this;
    const a = tris[t * 3] * VERT_STRIDE, b = tris[t * 3 + 1] * VERT_STRIDE, c = tris[t * 3 + 2] * VERT_STRIDE;
    const ax = verts[a], az = verts[a + 2], bx = verts[b], bz = verts[b + 2], cx = verts[c], cz = verts[c + 2];
    const v0x = bx - ax, v0z = bz - az, v1x = cx - ax, v1z = cz - az, v2x = x - ax, v2z = z - az;
    const den = v0x * v1z - v0z * v1x;
    if (Math.abs(den) < 1e-9) return null;
    const u = (v2x * v1z - v2z * v1x) / den;
    const v = (v0x * v2z - v0z * v2x) / den;
    return u >= -0.01 && v >= -0.01 && u + v <= 1.01 ? [u, v] : null;
  }

  heightAt(x, z, searchR = 3) {
    const t = this.triAt(x, z, searchR);
    if (t < 0) return null;
    const bary = this._lastBary;
    if (!bary) return this.cy[t];
    const { verts, tris } = this;
    const a = tris[t * 3] * VERT_STRIDE, b = tris[t * 3 + 1] * VERT_STRIDE, c = tris[t * 3 + 2] * VERT_STRIDE;
    return verts[a + 1] + bary[0] * (verts[b + 1] - verts[a + 1]) + bary[1] * (verts[c + 1] - verts[a + 1]);
  }

  /** A* over the triangle graph, then a greedy line-of-walk pull. */
  findPath(sx, sz, tx, tz, { canWalk = null, maxNodes = 4000 } = {}) {
    const start = this.triAt(sx, sz, 2.5);
    const goal = this.triAt(tx, tz, 2.5);
    if (start < 0 || goal < 0) return null;
    if (start === goal) {
      const direct = [{ x: sx, z: sz }, { x: tx, z: tz }];
      return canWalk && !canWalk(direct[0], direct[1]) ? this._route(start, goal, sx, sz, tx, tz, canWalk, maxNodes)
        : [{ x: tx, y: this.heightAt(tx, tz) ?? this.cy[goal], z: tz }];
    }
    return this._route(start, goal, sx, sz, tx, tz, canWalk, maxNodes);
  }

  _route(start, goal, sx, sz, tx, tz, canWalk, maxNodes) {
    const g = new Map([[start, 0]]);
    const came = new Map();
    const open = new MinHeap();
    const seen = new Set();
    const h = (t) => Math.hypot(this.cx[t] - tx, this.cz[t] - tz);
    open.push(start, h(start));
    let visits = 0;
    while (open.size && visits < maxNodes) {
      const { node: cur } = open.pop();
      if (seen.has(cur)) continue;
      seen.add(cur);
      visits++;
      if (cur === goal) break;
      for (const nb of this.adjacent[cur] || []) {
        if (seen.has(nb)) continue;
        const step = Math.hypot(this.cx[nb] - this.cx[cur], this.cz[nb] - this.cz[cur]);
        const tentative = (g.get(cur) ?? Infinity) + step;
        if (tentative < (g.get(nb) ?? Infinity)) {
          g.set(nb, tentative);
          came.set(nb, cur);
          open.push(nb, tentative + h(nb));
        }
      }
    }
    if (!seen.has(goal)) return null;
    const chain = [];
    for (let n = goal; n !== undefined; n = came.get(n)) chain.unshift(n);
    const pts = [{ x: sx, y: this.heightAt(sx, sz) ?? this.cy[chain[0]], z: sz }];
    for (const t of chain) pts.push({ x: this.cx[t], y: this.cy[t], z: this.cz[t] });
    pts.push({ x: tx, y: this.heightAt(tx, tz) ?? this.cy[goal], z: tz });
    const dedup = pts.filter((p, i) => i === 0 || Math.hypot(p.x - pts[i - 1].x, p.z - pts[i - 1].z) > 0.05);
    if (!canWalk || dedup.length <= 2) return dedup.slice(1);
    const out = [];
    let i = 0;
    while (i < dedup.length - 1) {
      let j = i + 1;
      for (let k = dedup.length - 1; k > i; k--) {
        if (canWalk(dedup[i], dedup[k])) { j = k; break; }
      }
      if (j === i + 1 && !canWalk(dedup[i], dedup[i + 1])) return null;
      out.push(dedup[j]);
      i = j;
    }
    return out;
  }

  /** Distance along the surface between two points, or null when unreachable. */
  distance(sx, sz, tx, tz, opts) {
    const path = this.findPath(sx, sz, tx, tz, opts);
    if (!path) return null;
    let d = 0;
    let px = sx, pz = sz;
    for (const p of path) { d += Math.hypot(p.x - px, p.z - pz); px = p.x; pz = p.z; }
    return d;
  }

  /**
   * Rank walkable triangles as AI spots by who can see them from where the player
   * is likely to come from. `los(from, to)` returns true when nothing blocks sight.
   */
  scoreSpots({ approach = [], los, eyeHeight = 1.5, chestHeight = 1.25, minSpawnDist = 4, maxRange = 26, maxSpots = 320, unseenRadius = 9 } = {}) {
    const spots = [];
    const stride = Math.max(1, Math.floor(this.triCount / maxSpots));
    for (let t = 0; t < this.triCount; t += stride) {
      const x = this.cx[t], y = this.cy[t], z = this.cz[t];
      const chest = { x, y: y + chestHeight, z };
      let seenFrom = 0, unseenNear = 0, nearestApproach = Infinity;
      for (const a of approach) {
        const d = Math.hypot(a.x - x, a.z - z);
        if (d > maxRange) continue;
        nearestApproach = Math.min(nearestApproach, d);
        const from = { x: a.x, y: (a.y ?? y) + eyeHeight, z: a.z };
        if (los(from, chest)) seenFrom++;
        else if (d < unseenRadius) unseenNear++;
      }
      if (nearestApproach < minSpawnDist) continue;
      const exits = this.adjacent[t]?.length || 0;
      spots.push({
        tri: t, x, y, z,
        seenFrom, unseenNear,
        nearestApproach: nearestApproach === Infinity ? 999 : nearestApproach,
        exits,
        ambush: unseenNear >= 2 && seenFrom >= 1 && seenFrom <= Math.max(1, Math.ceil(approach.length * 0.3)),
        overwatch: seenFrom >= Math.max(3, approach.length * 0.25),
        flank: unseenNear >= 1 && nearestApproach < 16 && exits >= 2,
      });
    }
    spots.sort((a, b) => (b.ambush - a.ambush) || (b.unseenNear - a.unseenNear));
    return spots;
  }
}
