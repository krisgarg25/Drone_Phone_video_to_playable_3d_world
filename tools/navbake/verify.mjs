#!/usr/bin/env node
/**
 * verify.mjs -- the acceptance check for a baked navmesh.
 *
 *   node verify.mjs <sceneName> [--tol 1.5] [--step 1] [--min-component 20]
 *
 * Answers three questions about work/<scene>/pc/nav.json:
 *
 * 1. SHAPE -- does it satisfy the contract the browser runtime is written
 *    against (version, field names, index range, rounded verts, area/triCount
 *    honesty, freshness against the collider GLB)?
 * 2. COVERAGE -- does the navmesh actually cover where the player walks? Every
 *    point of collision.json's spawn + walk_path, interpolated at --step metres
 *    (the closing leg counts: the autopilot indexes walk_path cyclically), is
 *    matched to its nearest navmesh triangle in XZ. Further than --tol metres
 *    is a loud failure, because that is a spot the autopilot already walks
 *    through and a bot would refuse to enter. The navmesh height there is also
 *    compared with the exported heightfield, which is the check that catches a
 *    navmesh inflated by the agent height.
 * 3. QUALITY -- how much of the mesh is stranded: connected components smaller
 *    than --min-component triangles are places a bot could path onto but never
 *    reach, so they pollute spot scoring and target selection.
 *
 * Exit code 0 = pass, 1 = fail, 2 = bad usage / missing files.
 */

import { existsSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { boundsOf, components, totalArea, walkPathSamples, XzGrid } from './navstats.mjs';
import { loadGlb } from './glb.mjs';
import { orientForWalking } from './navmesh-orient.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');

function parseArgs(argv) {
  const opts = { tol: 1.5, step: 1, nav: null, minComponent: 20, quiet: false, scene: null };
  const positional = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) { positional.push(a); continue; }
    const key = a.slice(2);
    const next = () => argv[++i];
    if (key === 'tol') opts.tol = Number(next());
    else if (key === 'step') opts.step = Number(next());
    else if (key === 'nav') opts.nav = next();
    else if (key === 'min-component') opts.minComponent = Number(next());
    else if (key === 'quiet') opts.quiet = true;
    else if (key === 'help') {
      console.log('usage: node verify.mjs <sceneName> [--tol 1.5] [--step 1] [--min-component 20] [--nav <path>]');
      process.exit(0);
    } else throw new Error(`unknown option ${a}`);
  }
  if (!positional.length) throw new Error('usage: node verify.mjs <sceneName>');
  opts.scene = positional[0];
  if (!(opts.tol > 0) || !(opts.step > 0)) throw new Error('--tol and --step must be > 0');
  return opts;
}

// The XZ query helpers (XzGrid / nearestOnTriangle / walkPathSamples) live in
// navstats.mjs so that bake.mjs can run the same coverage probe while tuning.

// ---------------------------------------------------------- heightfield reader
/**
 * The source collider's own surface, restricted to triangles steep enough to be
 * floor candidates at the bake's slope limit -- i.e. exactly what Recast was
 * allowed to turn into navmesh. Used as the reference for the "y must be feet
 * level" check, because it is the surface the navmesh was baked from and is
 * immune to the viewer_assets being regenerated after the collider.
 */
function colliderSurfaceGrid(glbPath, slopeDeg, cell = 2.5) {
  const { positions, indices } = loadGlb(glbPath);
  const { positions: P, indices: I } = orientForWalking(positions, indices, 'auto');
  const thr = Math.max(0.5, Math.cos((slopeDeg * Math.PI) / 180));
  const up = [];
  for (let t = 0; t < I.length; t += 3) {
    const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz);
    if (len < 1e-12 || ny / len <= thr) continue;
    up.push(I[t], I[t + 1], I[t + 2]);
  }
  return { grid: new XzGrid(P, up, cell), triangles: up.length / 3 };
}

/** The exported ground grid, sampled exactly the way viewer/pc.js samples it. */
function loadGround(assetDir, nx, nz) {
  for (const name of ['ground.f32', 'heights.f32']) {
    const p = path.join(assetDir, name);
    if (!existsSync(p)) continue;
    const buf = readFileSync(p);
    const data = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    if (data.length === nx * nz) return { name, data, nx, nz };
  }
  return null;
}

function sampleGround(hf, ox, oz, cell, x, z) {
  const { nx, nz, data } = hf;
  const gx = Math.min(Math.max((x - ox) / cell - 0.5, 0), nx - 1.001);
  const gz = Math.min(Math.max((z - oz) / cell - 0.5, 0), nz - 1.001);
  const x0 = Math.floor(gx), z0 = Math.floor(gz), fx = gx - x0, fz = gz - z0;
  const h00 = data[z0 * nx + x0], h10 = data[z0 * nx + x0 + 1];
  const h01 = data[(z0 + 1) * nx + x0], h11 = data[(z0 + 1) * nx + x0 + 1];
  return (h00 * (1 - fx) + h10 * fx) * (1 - fz) + (h01 * (1 - fx) + h11 * fx) * fz;
}

/** Percentile of an already-sorted array. */
function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  return sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))];
}

// ------------------------------------------------------------------------- main
function main() {
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (e) {
    console.error(e.message);
    return 2;
  }
  const log = opts.quiet ? () => {} : (...a) => console.log(...a);
  const scene = opts.scene;
  const navPath = path.resolve(REPO, opts.nav ?? path.join('work', scene, 'pc', 'nav.json'));
  const glbPath = path.join(REPO, 'work', scene, 'pc', 'collision.collision.glb');
  const assetDir = path.join(REPO, 'work', scene, 'viewer_assets');

  if (!existsSync(navPath)) {
    console.error(`[verify] missing ${navPath}\n[verify] bake it first:  node bake.mjs ${scene} --obj`);
    process.exit(2);
  }
  const nav = JSON.parse(readFileSync(navPath, 'utf8'));
  const navSize = statSync(navPath).size;
  const fails = [];
  const notes = [];

  // ---- 1. contract shape ----------------------------------------------------
  for (const k of ['version', 'source', 'generated', 'build', 'bbox', 'verts', 'tris', 'stats']) {
    if (nav[k] === undefined) fails.push(`nav.json is missing "${k}"`);
  }
  for (const k of ['cellSize', 'cellHeight', 'walkableRadius', 'walkableHeight', 'walkableClimb', 'walkableSlopeDeg']) {
    if (nav.build?.[k] === undefined) fails.push(`build block is missing "${k}"`);
  }
  for (const k of ['triCount', 'areaM2', 'largestComponentTris']) {
    if (nav.stats?.[k] === undefined) fails.push(`stats block is missing "${k}"`);
  }
  if (nav.version !== 1) fails.push(`nav.json version is ${nav.version}, runtime expects 1`);

  const rawTris = nav.tris ?? [];
  const rawVerts = nav.verts ?? [];
  const badIdx = rawTris.findIndex((v) => !Number.isInteger(v) || v < 0 || v >= rawVerts.length / 3);
  if (badIdx !== -1) fails.push(`tris[${badIdx}] = ${rawTris[badIdx]} is not a vertex index`);
  if (rawVerts.some((v) => !Number.isFinite(v))) fails.push('verts contains a non-finite number');

  const V = Float32Array.from(rawVerts);
  const T = Uint32Array.from(rawTris);
  const nt = T.length / 3;
  if (!nt) fails.push('nav.json has no triangles');
  if (V.length % 3) fails.push(`verts length ${V.length} is not a multiple of 3`);
  if (T.length % 3) fails.push(`tris length ${T.length} is not a multiple of 3`);
  if (nt !== nav.stats?.triCount) fails.push(`stats.triCount ${nav.stats?.triCount} != tris/3 = ${nt}`);
  let maxIdx = 0;
  for (let i = 0; i < T.length; i++) if (T[i] > maxIdx) maxIdx = T[i];
  if (maxIdx >= V.length / 3) fails.push(`triangle index ${maxIdx} out of range for ${V.length / 3} verts`);
  for (let i = 0; i < V.length; i++) {
    if (!Number.isFinite(V[i])) { fails.push(`verts[${i}] is not finite`); break; }
  }
  // check the rounding on the raw JSON numbers: a Float32Array copy adds
  // representation noise (-12.74 -> -12.7400007) that would trip this test
  const raw = nav.verts ?? [];
  for (let i = 0; i < raw.length; i++) {
    if (Math.abs(raw[i] * 1000 - Math.round(raw[i] * 1000)) > 1e-9) {
      notes.push(`verts[${i}] = ${raw[i]} is not rounded to 3 decimals`);
      break;
    }
  }
  const bb = boundsOf(V);
  for (let i = 0; i < 6; i++) {
    const declared = nav.bbox?.[i];
    const want = i < 3 ? bb.min[i] : bb.max[i - 3];
    if (declared === undefined || Math.abs(declared - want) > 0.002) {
      fails.push(`bbox[${i}] = ${declared} disagrees with the data (${want})`);
    }
  }
  if (navSize > 2e6) fails.push(`nav.json is ${(navSize / 1e6).toFixed(2)} MB, over the ~2 MB budget`);
  const area = totalArea(V, T);
  if (Math.abs(area - (nav.stats?.areaM2 ?? -1)) > Math.max(0.5, area * 0.01)) {
    fails.push(`stats.areaM2 ${nav.stats?.areaM2} disagrees with the geometry (${area.toFixed(2)})`);
  }
  if (existsSync(glbPath) && statSync(glbPath).mtimeMs > Date.parse(nav.generated)) {
    fails.push(`nav.json is STALE: ${path.basename(glbPath)} is newer than ${nav.generated} - re-bake`);
  }
  if (nav.source && nav.source !== path.basename(glbPath)) {
    notes.push(`source says "${nav.source}", baked from "${path.basename(glbPath)}"`);
  }

  log(`[verify] ${scene}  nav.json ${(navSize / 1024).toFixed(0)} kB, generated ${nav.generated}`);
  const b = nav.build ?? {};
  log(`[verify]   build: cellSize ${b.cellSize ?? '-'} cellHeight ${b.cellHeight ?? '-'} radius ${b.walkableRadius ?? '-'}` +
      ` height ${b.walkableHeight ?? '-'} climb ${b.walkableClimb ?? '-'} slope ${b.walkableSlopeDeg ?? '-'}deg`);
  log(`[verify]   ${nt} tris / ${V.length / 3} verts, walkable area ${area.toFixed(1)} m2 ` +
      `(plan ${totalArea(V, T, true).toFixed(1)} m2)`);
  log(`[verify]   bbox x [${bb.min[0].toFixed(1)}, ${bb.max[0].toFixed(1)}] ` +
      `y [${bb.min[1].toFixed(2)}, ${bb.max[1].toFixed(2)}] z [${bb.min[2].toFixed(1)}, ${bb.max[2].toFixed(1)}]`);
  log(`[verify]   Y range ${(bb.max[1] - bb.min[1]).toFixed(2)} m over ${nt} triangles`);

  // ---- 2. coverage of where the player actually walks -----------------------
  const colPath = path.join(assetDir, 'collision.json');
  if (!existsSync(colPath)) {
    fails.push(`no ${path.relative(REPO, colPath)} - cannot prove the navmesh covers the walked path`);
  } else if (nt) {
    const col = JSON.parse(readFileSync(colPath, 'utf8'));
    const rot = col.rotation_rowmajor;
    const identity = !rot || JSON.stringify(rot) === JSON.stringify([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    const ground = identity ? loadGround(assetDir, col.nx, col.nz) : null;
    if (!identity) notes.push('collision.json rotation_rowmajor is not identity - heights not compared');
    if (!ground) notes.push('no ground.f32/heights.f32 to compare navmesh height against');

    const grid = new XzGrid(V, T, Math.max(1, opts.tol * 1.5));
    let surf = null;
    if (existsSync(glbPath)) {
      try {
        surf = colliderSurfaceGrid(glbPath, b.walkableSlopeDeg ?? 50);
      } catch (e) {
        notes.push(`could not read the collider for a surface cross-check: ${e.message}`);
      }
    }

    const samples = [];
    if (col.spawn) samples.push({ what: 'spawn', x: col.spawn.x, z: col.spawn.z });
    else notes.push('collision.json has no spawn');
    const loop = col.walk_path ?? [];
    if (!loop.length) notes.push('collision.json has no walk_path');
    for (const [x, z] of walkPathSamples(loop, opts.step)) samples.push({ what: 'walk_path', x, z });

    const dists = [];
    const dys = [];
    const dysSurf = [];
    const misses = [];
    let inside = 0, pass = 0;
    let spawnResult = null;
    for (const s of samples) {
      const r = grid.nearest(s.x, s.z);
      const ok = r.d <= opts.tol;
      if (ok) pass++;
      if (r.inside) inside++;
      dists.push(r.d);
      const groundY = ground ? sampleGround(ground, col.origin_xz[0], col.origin_xz[1], col.cell, s.x, s.z) : null;
      const dy = groundY === null || r.y === null ? null : r.y - groundY;
      if (dy !== null) dys.push(dy);
      // the collider surface right under this point: what Recast was given
      let surfY = null;
      if (surf && ok && r.y !== null) {
        const hit = surf.grid.nearest(s.x, s.z, 4);
        if (hit.d <= 1) surfY = hit.y;
      }
      const dySurf = surfY === null || r.y === null ? null : r.y - surfY;
      if (dySurf !== null) dysSurf.push(dySurf);
      if (!ok) misses.push({ ...s, ...r, groundY, dy });
      if (s.what === 'spawn') spawnResult = { ...r, groundY, dy, surfY, dySurf };
    }
    const ratio = samples.length ? (100 * pass / samples.length) : 0;
    dists.sort((a, b) => a - b);
    log(`[verify]   coverage: ${pass}/${samples.length} walk samples within ${opts.tol} m = ${ratio.toFixed(1)}%` +
        `  (${inside}/${samples.length} directly inside a triangle)`);
    log(`[verify]   distance to navmesh: median ${quantile(dists, 0.5).toFixed(2)} m, ` +
        `p95 ${quantile(dists, 0.95).toFixed(2)} m, max ${dists[dists.length - 1].toFixed(2)} m`);
    if (spawnResult) {
      log(`[verify]   spawn (${col.spawn.x.toFixed(1)}, ${col.spawn.z.toFixed(1)}): navmesh ${spawnResult.d.toFixed(2)} m away, ` +
          `navY ${spawnResult.y === null ? '?' : spawnResult.y.toFixed(2)}, ` +
          `colliderY ${spawnResult.surfY === null ? '?' : spawnResult.surfY.toFixed(2)}, ` +
          `heightfieldY ${spawnResult.groundY === null ? '?' : spawnResult.groundY.toFixed(2)}`);
    }
    if (dys.length) {
      const a = dys.slice().sort((x, y) => x - y);
      log(`[verify]   navmesh y - ${ground ? ground.name : 'heightfield'}: median ${quantile(a, 0.5) >= 0 ? '+' : ''}${quantile(a, 0.5).toFixed(2)} m, ` +
          `p95 |${quantile(a.map(Math.abs).sort((x, y) => x - y), 0.95).toFixed(2)}| m`);
      if (Math.abs(quantile(a, 0.5)) > 0.6 && !dysSurf.length) {
        fails.push(`navmesh sits ${quantile(a, 0.5).toFixed(2)} m above the exported heightfield median, ` +
          `and there is no collider to cross-check against - y must be feet level`);
      }
    }
    if (dysSurf.length) {
      const abs = dysSurf.map(Math.abs).sort((x, y) => x - y);
      const med = dysSurf.slice().sort((x, y) => x - y)[abs.length >> 1];
      log(`[verify]   navmesh y - collider surface: median ${med >= 0 ? '+' : ''}${med.toFixed(2)} m, ` +
          `p95 |${quantile(abs, 0.95).toFixed(2)}| m, max |${abs[abs.length - 1].toFixed(2)}| m over ${dysSurf.length} samples ` +
          `(cellHeight ${b.cellHeight ?? '?'} quantisation should be well under 1 m)`);
      // this is the real feet-level test: a navmesh inflated by walkableHeight
      // would sit ~1.7 m above the surface it was baked from
      if (Math.abs(quantile(abs, 0.5)) > 0.6) {
        fails.push(`navmesh y is ${quantile(abs, 0.5).toFixed(2)} m (median) off the collider surface it was baked from - ` +
          `y must be the feet-level floor, not inflated by the agent height`);
      }
      if (Math.abs(med) > 0.9) {
        fails.push(`navmesh is systematically ${med.toFixed(2)} m ${med > 0 ? 'above' : 'below'} the collider surface - ` +
          `the walk samples are matched to the wrong shell`);
      }
    }
    if (dys.length && Math.abs(quantile(dys.slice().sort((x, y) => x - y), 0.5)) > 1.0) {
      notes.push(`the navmesh disagrees with ${ground ? ground.name : 'the heightfield'} by ` +
        `${quantile(dys.slice().sort((x, y) => x - y), 0.5).toFixed(2)} m at the median while agreeing with the collider: ` +
        `viewer_assets and ${path.basename(glbPath)} describe different surfaces. The collider is what the player ` +
        `physically stands on (ammo trimesh raycast), so the navmesh follows it, but collision.json's walk_path was ` +
        `planned on the other one - re-run scripts/walk_path_from_glb.py or rebuild the collider to agree.`);
    }

    // how badly do the two surfaces disagree in general? (evidence, not a gate)
    if (ground && surf) {
      const stepCells = Math.max(1, Math.round(2 / col.cell));
      const diffs = [];
      for (let i = 0; i < col.nz; i += stepCells) {
        for (let j = 0; j < col.nx; j += stepCells) {
          const x = col.origin_xz[0] + (j + 0.5) * col.cell;
          const z = col.origin_xz[1] + (i + 0.5) * col.cell;
          const hit = surf.grid.nearest(x, z, 3);
          if (hit.d > 0.5) continue; // no up-facing collider surface in this cell
          diffs.push(hit.y - sampleGround(ground, col.origin_xz[0], col.origin_xz[1], col.cell, x, z));
        }
      }
      if (diffs.length) {
        const srt = diffs.slice().sort((a, b) => a - b);
        const abs = diffs.map(Math.abs).sort((a, b) => a - b);
        log(`[verify]   collider top surface vs ${ground.name} over the exported grid: median ` +
            `${quantile(srt, 0.5) >= 0 ? '+' : ''}${quantile(srt, 0.5).toFixed(2)} m, ` +
            `p95 |${quantile(abs, 0.95).toFixed(2)}| m  (${diffs.length} cells)`);
      }
    }
    for (const m of misses.slice(0, 12)) {
      log(`[verify]   MISS ${m.what} (${m.x.toFixed(1)}, ${m.z.toFixed(1)}) nearest navmesh triangle ${m.tri} ` +
          `is ${Number.isFinite(m.d) ? m.d.toFixed(2) : 'far'} m away in XZ` +
          `${m.y === null ? '' : ` navY ${m.y.toFixed(2)}`}${m.groundY === null ? '' : ` groundY ${m.groundY.toFixed(2)}`}`);
    }
    if (misses.length > 12) log(`[verify]   ... ${misses.length - 12} more misses`);
    if (samples.length && pass === 0) fails.push(`NONE of the ${samples.length} walk samples landed on the navmesh`);
    else if (ratio < 100) {
      fails.push(`${samples.length - pass}/${samples.length} walk samples have no navmesh triangle within ${opts.tol} m`);
    }

    // is the spawn on the biggest piece? a bot that spawns on a 6-tri island
    // has nowhere to go even though the mesh "covers" it
    if (spawnResult && spawnResult.tri >= 0) {
      const comps0 = components(V, T);
      const label = comps0.label[spawnResult.tri];
      if (label !== 0) {
        notes.push(`spawn is on component #${label} (${comps0.sizes[label]} tris, ${comps0.areas[label]} m2), ` +
          `not the largest (${comps0.sizes[0]} tris, ${comps0.areas[0]} m2)`);
      }
    }
  }

  // ---- 3. stranded islands --------------------------------------------------
  if (nt) {
    const comps = components(V, T);
    const strandedTris = comps.sizes.reduce((n, s) => n + (s < opts.minComponent ? s : 0), 0);
    const strandedArea = comps.areas.reduce((n, a, i) => n + (comps.sizes[i] < opts.minComponent ? a : 0), 0);
    log(`[verify]   ${comps.count} connected component(s); largest ${comps.largest} tris ` +
        `(${(100 * comps.largest / nt).toFixed(1)}%) / ${comps.areas[0]} m2`);
    log(`[verify]   component sizes: ${comps.sizes.slice(0, 12).join(', ')}${comps.sizes.length > 12 ? ` ... (${comps.sizes.length} total)` : ''}`);
    log(`[verify]   stranded (< ${opts.minComponent} tris): ${strandedTris}/${nt} tris = ` +
        `${(100 * strandedTris / nt).toFixed(2)}% of triangles, ${strandedArea.toFixed(1)} m2 of the ${area.toFixed(1)} m2 total`);
    if (comps.largest !== nav.stats?.largestComponentTris) {
      fails.push(`stats.largestComponentTris ${nav.stats?.largestComponentTris} != recomputed ${comps.largest}`);
    }
    if (100 * strandedTris / nt > 20) {
      notes.push(`over a fifth of the mesh is stranded - raise --region or --cell, or lower --radius`);
    }
  }

  for (const n of notes) log(`[verify]   note: ${n}`);
  if (fails.length) {
    for (const f of fails) console.error(`[verify] FAIL: ${f}`);
    console.error(`[verify] ${scene}: FAILED (${fails.length} problem(s))`);
    return 1;
  }
  log(`[verify] ${scene}: PASS`);
  return 0;
}

process.exitCode = main();
