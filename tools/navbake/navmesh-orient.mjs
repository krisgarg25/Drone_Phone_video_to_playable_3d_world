/**
 * navmesh-orient.mjs -- winding / normal repair for scan-derived triangle soups.
 *
 * Why this exists
 * ---------------
 * Recast decides "is this triangle a floor?" from the normal it derives from the
 * *vertex winding* (`rcMarkWalkableTriangles` keeps a triangle only if
 * `normal.y > cos(slopeAngle)`). It never reads a NORMAL attribute. So on a mesh
 * whose winding is arbitrary -- which is what a voxel / marching-cubes shell off
 * a gaussian scan tends to be -- floors read as ceilings and silently vanish
 * from the navmesh: you get a *valid* navmesh with holes in it.
 *
 * What we do
 * ----------
 * 1. weld coincident vertices (quantised) so shared edges can be found,
 * 2. drop zero-area / repeated-index triangles,
 * 3. make winding *locally consistent* per connected patch: BFS over edge
 *    adjacency, flipping triangles so that each shared edge is crossed in
 *    opposite directions by its two faces,
 * 4. choose the *global* side of each component:
 *      - closed manifold component with a meaningful enclosed volume
 *          -> normals out of the enclosed region (positive signed volume). For
 *             both a solid boulder and a room shell that is the side with free
 *             space, i.e. the side you can stand on.
 *      - anything else (scan shells are never quite watertight)
 *          -> the side with more horizontal surface facing +Y wins. These
 *             colliders have had their airborne crust clipped and carry no
 *             ceilings, so "more surface facing up" is the walkable side.
 *
 * Every decision and its evidence goes into the returned report, so a human can
 * see what was changed and why instead of trusting a silent fix.
 */

const WELD_SCALE_DEFAULT = 1e-4;

/** Weld coincident vertices and re-index. Returns a fresh, compacted mesh. */
export function weld(positions, indices, scale = WELD_SCALE_DEFAULT) {
  const map = new Map();
  const welded = [];
  const nv = positions.length / 3;
  const remap = new Int32Array(nv);
  for (let v = 0; v < nv; v++) {
    const x = positions[v * 3], y = positions[v * 3 + 1], z = positions[v * 3 + 2];
    const key = `${Math.round(x / scale)},${Math.round(y / scale)},${Math.round(z / scale)}`;
    let id = map.get(key);
    if (id === undefined) {
      id = welded.length / 3;
      map.set(key, id);
      welded.push(x, y, z);
    }
    remap[v] = id;
  }
  const out = new Uint32Array(indices.length);
  for (let i = 0; i < indices.length; i++) out[i] = remap[indices[i]];
  return { positions: new Float32Array(welded), indices: out };
}

/** Unnormalised triangle normal: its length is twice the triangle area. */
function crossOf(P, a, b, c) {
  const ux = P[b * 3] - P[a * 3], uy = P[b * 3 + 1] - P[a * 3 + 1], uz = P[b * 3 + 2] - P[a * 3 + 2];
  const vx = P[c * 3] - P[a * 3], vy = P[c * 3 + 1] - P[a * 3 + 1], vz = P[c * 3 + 2] - P[a * 3 + 2];
  return [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
}

/** Coarse orientation census of a triangle soup, derived from the winding. */
export function measure(positions, indices) {
  let up = 0, down = 0, side = 0, degenerate = 0, upArea = 0, downArea = 0, sideArea = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const n = crossOf(positions, indices[t], indices[t + 1], indices[t + 2]);
    const len = Math.hypot(n[0], n[1], n[2]);
    if (len < 1e-12) { degenerate++; continue; }
    const area = len / 2;
    const ny = n[1] / len;
    if (ny > 0.7) { up++; upArea += area; }
    else if (ny < -0.7) { down++; downArea += area; }
    else { side++; sideArea += area; }
  }
  return { up, down, side, degenerate, upArea: +upArea.toFixed(1), downArea: +downArea.toFixed(1), sideArea: +sideArea.toFixed(1) };
}

/**
 * @param {Float32Array} positions flat xyz
 * @param {Uint32Array} indices flat i0,i1,i2
 * @param {'auto'|'never'|'always'} [mode]
 *   auto   repair (default)
 *   never  touch nothing -- for A-B comparison / diagnostics
 *   always repair, then force-flip every component
 * @param {number} [weldScale] position quantisation used for welding
 * @returns {{positions: Float32Array, indices: Uint32Array, report: object}}
 */
export function orientForWalking(positions, indices, mode = 'auto', weldScale = WELD_SCALE_DEFAULT) {
  const before = measure(positions, indices);
  if (mode === 'never') {
    return {
      positions, indices,
      report: { mode, repaired: false, trianglesIn: indices.length / 3, trianglesOut: indices.length / 3,
        verticesOut: positions.length / 3, degenerateDropped: 0, componentCount: 0, flippedComponents: 0,
        flippedTris: 0, before, after: before, components: [] },
    };
  }

  const { positions: P, indices: welded } = weld(positions, indices, weldScale);
  const ntIn = welded.length / 3;

  // ---- drop degenerate triangles -------------------------------------------
  const trisAll = new Uint32Array(welded.length);
  let nt = 0;
  for (let t = 0; t < ntIn; t++) {
    const a = welded[t * 3], b = welded[t * 3 + 1], c = welded[t * 3 + 2];
    if (a === b || b === c || a === c) continue;
    const n = crossOf(P, a, b, c);
    if (Math.hypot(n[0], n[1], n[2]) < 1e-12) continue;
    trisAll[nt * 3] = a; trisAll[nt * 3 + 1] = b; trisAll[nt * 3 + 2] = c;
    nt++;
  }
  const tris = trisAll.subarray(0, nt * 3);

  // ---- edge adjacency: key -> list of (tri*4 + localEdge) ------------------
  const edges = new Map();
  for (let t = 0; t < nt; t++) {
    for (let e = 0; e < 3; e++) {
      const a = tris[t * 3 + e], b = tris[t * 3 + (e + 1) % 3];
      const key = a < b ? `${a}|${b}` : `${b}|${a}`;
      const hit = edges.get(key);
      if (hit) hit.push(t * 4 + e);
      else edges.set(key, [t * 4 + e]);
    }
  }

  // ---- BFS: propagate consistent winding, then choose the global side ------
  const flip = new Uint8Array(nt).fill(255); // 255 = unvisited, else 0/1 = reverse winding
  const components = [];
  const queue = new Int32Array(nt);
  for (let seed = 0; seed < nt; seed++) {
    if (flip[seed] !== 255) continue;
    flip[seed] = 0;
    let qh = 0, qt = 0;
    queue[qt++] = seed;
    let borderEdges = 0, nonManifoldEdges = 0;
    const members = [];
    while (qh < qt) {
      const t = queue[qh++];
      members.push(t);
      for (let e = 0; e < 3; e++) {
        const a = tris[t * 3 + e], b = tris[t * 3 + (e + 1) % 3];
        const key = a < b ? `${a}|${b}` : `${b}|${a}`;
        const list = edges.get(key);
        if (!list || list.length === 1) { borderEdges++; continue; }
        if (list.length > 2) { nonManifoldEdges++; continue; }
        const other = list[0] === t * 4 + e ? list[1] : list[0];
        const ot = other >> 2, oe = other & 3;
        // do the two faces run along the shared edge in the SAME direction?
        const sameDirection = tris[ot * 3 + oe] === a && tris[ot * 3 + (oe + 1) % 3] === b;
        const want = flip[t] ^ (sameDirection ? 1 : 0);
        if (flip[ot] === 255) {
          flip[ot] = want;
          queue[qt++] = ot;
        }
      }
    }

    // statistics of the patch, in the locally-repaired orientation
    let vol = 0, upArea = 0, downArea = 0, sideArea = 0;
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const t of members) {
      const a = tris[t * 3], b = tris[t * 3 + 1], c = tris[t * 3 + 2];
      const s = flip[t] ? -1 : 1;
      const n = crossOf(P, a, b, c);
      const len = Math.hypot(n[0], n[1], n[2]);
      const area = len / 2;
      vol += s * (P[a * 3] * n[0] + P[a * 3 + 1] * n[1] + P[a * 3 + 2] * n[2]) / 6;
      const ny = len > 0 ? (n[1] * s) / len : 0;
      if (ny > 0.7) upArea += area;
      else if (ny < -0.7) downArea += area;
      else sideArea += area;
      for (const v of [a, b, c]) {
        const x = P[v * 3], y = P[v * 3 + 1], z = P[v * 3 + 2];
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
      }
    }
    const closed = borderEdges === 0 && nonManifoldEdges === 0;
    const boxVolume = Math.max(0, maxX - minX) * Math.max(0, maxY - minY) * Math.max(0, maxZ - minZ);
    const volumeMeaningful = Math.abs(vol) > Math.max(1e-6, boxVolume * 0.01);

    let rule, componentFlip;
    if (closed && volumeMeaningful) {
      componentFlip = vol < 0 ? 1 : 0;
      rule = 'closed manifold: normals out of enclosed region (signed volume)';
    } else if (upArea >= downArea) {
      componentFlip = 0;
      rule = closed ? 'closed but degenerate volume: up-area majority' : 'open shell: up-area majority';
    } else {
      componentFlip = 1;
      rule = `${closed ? 'closed but degenerate volume' : 'open shell'}: up-area majority (flipped)`;
    }
    if (mode === 'always') componentFlip ^= 1;
    if (componentFlip) for (const t of members) flip[t] ^= 1;

    components.push({
      tris: members.length,
      closed,
      borderEdges,
      nonManifoldEdges,
      signedVolume: +vol.toFixed(1),
      upArea: +upArea.toFixed(1),
      downArea: +downArea.toFixed(1),
      sideArea: +sideArea.toFixed(1),
      rule,
      flipped: componentFlip === 1,
    });
  }

  // ---- rebuild the index buffer with the repaired winding ------------------
  const outIdx = new Uint32Array(tris.length);
  for (let t = 0; t < nt; t++) {
    const a = tris[t * 3], b = tris[t * 3 + 1], c = tris[t * 3 + 2];
    if (flip[t]) {
      outIdx[t * 3] = a; outIdx[t * 3 + 1] = c; outIdx[t * 3 + 2] = b;
    } else {
      outIdx[t * 3] = a; outIdx[t * 3 + 1] = b; outIdx[t * 3 + 2] = c;
    }
  }

  const after = measure(P, outIdx);
  const flippedComponents = components.filter((c) => c.flipped);
  return {
    positions: P,
    indices: outIdx,
    report: {
      mode,
      repaired: true,
      weldScale,
      trianglesIn: ntIn,
      trianglesOut: nt,
      verticesOut: P.length / 3,
      degenerateDropped: ntIn - nt,
      componentCount: components.length,
      flippedComponents: flippedComponents.length,
      flippedTris: flippedComponents.reduce((n, c) => n + c.tris, 0),
      components: components.sort((a, b) => b.tris - a.tris).slice(0, 8),
      before,
      after,
    },
  };
}
