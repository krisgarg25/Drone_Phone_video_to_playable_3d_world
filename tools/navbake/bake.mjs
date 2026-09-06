#!/usr/bin/env node
/**
 * bake.mjs -- bake a navigation mesh from a scene's collision mesh, offline.
 *
 *   node bake.mjs <sceneName> [--cell 0.15] [--radius 0.4] [--height 1.7]
 *                            [--climb 0.5] [--slope 50] [--dry] [--obj]
 *
 * Reads   work/<scene>/pc/collision.collision.glb   (PlayCanvas collision shell)
 * Writes  work/<scene>/pc/nav.json                  (the runtime's nav contract)
 *
 * The GLB is a static trimesh shell voxelised out of the gaussian cloud, in the
 * same world space the viewer puts it in untransformed -- so nav.json's verts
 * are directly comparable with collision.json's spawn / walk_path / heightfield.
 *
 * Recast's parameters are mostly in *voxel* units, not metres; the conversion is
 * done here (see toVoxels) and the metre values are what lands in nav.json's
 * `build` block, because the browser runtime thinks in metres.
 */

import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { init, getNavMeshPositionsAndIndices, exportNavMesh } from '@recast-navigation/core';
import { generateSoloNavMesh, generateTiledNavMesh } from '@recast-navigation/generators';

import { loadGlb, GlbError } from './glb.mjs';
import { orientForWalking } from './navmesh-orient.mjs';
import { boundsOf, components, totalArea, walkPathSamples, weldRounded, writeObj, XzGrid } from './navstats.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');

const DEFAULTS = {
  cell: 0.15,
  cellHeight: null,
  radius: 0.4,
  height: 1.7,
  climb: 0.5,
  slope: 50,
  region: 1.0,
  merge: 2.0,
  maxEdge: 0, // 0 => Recast's default of 12 cells
  detail: 6, // in cells; detailSampleDist = cs * detail (generator convention)
  detailError: 1, // in ch units
  surface: 'detail',
  tileSize: 32,
  orient: 'auto',
  solo: false,
  noSlopeFilter: false,
  in: null,
  out: null,
  obj: false,
  dtnav: null,
  dry: false,
  quiet: false,
};

const USAGE = `usage: node bake.mjs <sceneName> [options]

  --cell <m>            xz cell size                      (default ${DEFAULTS.cell})
  --cell-height <m>     y cell size, defaults to --cell
  --radius <m>          agent radius / erosion            (default ${DEFAULTS.radius})
  --height <m>          agent height / headroom           (default ${DEFAULTS.height})
  --climb <m>           max step height / walkable climb   (default ${DEFAULTS.climb})
  --slope <deg>         max walkable slope                (default ${DEFAULTS.slope})
  --no-slope-filter     equivalent to --slope 89.9: let the heightfield filters decide
  --region <m>          side of the smallest region kept  (default ${DEFAULTS.region})
  --merge <m>           side under which regions merge    (default ${DEFAULTS.merge})
  --max-edge <m>        max contour edge length, 0 = Recast default (12 cells)
  --detail <cells>      height-detail sampling distance    (default ${DEFAULTS.detail}, 0 = flat)
  --detail-error <ch>   allowed detail error in cell heights (default ${DEFAULTS.detailError})
  --surface <mode>      detail | poly                      (default ${DEFAULTS.surface})
  --tile-size <cells>   detour tile edge                   (default ${DEFAULTS.tileSize})
  --solo                single-tile build (fails above 65535 verts per tile)
  --orient <mode>       auto | never | always              (default ${DEFAULTS.orient})
  --in <path>           input glb (default work/<scene>/pc/collision.collision.glb)
  --out <path>          output json (default work/<scene>/pc/nav.json)
  --obj                 also write nav.obj next to nav.json
  --dtnav [path]        also write the detour binary (default work/<scene>/pc/nav.dtnav.bin)
  --dry                 build and report, write nothing
`;

function parseArgs(argv) {
  const opts = { ...DEFAULTS };
  const positional = [];
  const num = (name, raw) => {
    const v = Number(raw);
    if (!Number.isFinite(v)) throw new Error(`--${name} needs a number, got "${raw}"`);
    return v;
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) { positional.push(a); continue; }
    const key = a.slice(2);
    const eq = key.indexOf('=');
    const name = eq === -1 ? key : key.slice(0, eq);
    const inlineVal = eq === -1 ? null : key.slice(eq + 1);
    const take = () => (inlineVal !== null ? inlineVal : argv[++i]);
    switch (name) {
      case 'cell': opts.cell = num(name, take()); break;
      case 'cell-height': opts.cellHeight = num(name, take()); break;
      case 'radius': opts.radius = num(name, take()); break;
      case 'height': opts.height = num(name, take()); break;
      case 'climb': opts.climb = num(name, take()); break;
      case 'slope': opts.slope = num(name, take()); break;
      case 'region': opts.region = num(name, take()); break;
      case 'merge': opts.merge = num(name, take()); break;
      case 'max-edge': opts.maxEdge = num(name, take()); break;
      case 'detail': opts.detail = num(name, take()); break;
      case 'detail-error': opts.detailError = num(name, take()); break;
      case 'surface': opts.surface = take(); break;
      case 'tile-size': opts.tileSize = num(name, take()); break;
      case 'orient': opts.orient = take(); break;
      case 'in': opts.in = take(); break;
      case 'out': opts.out = take(); break;
      case 'dtnav': opts.dtnav = (inlineVal === null && (argv[i + 1] === undefined || argv[i + 1].startsWith('--'))) ? '' : take(); break;
      case 'obj': opts.obj = true; break;
      case 'solo': opts.solo = true; break;
      case 'no-slope-filter': opts.noSlopeFilter = true; break;
      case 'dry': opts.dry = true; break;
      case 'quiet': opts.quiet = true; break;
      case 'help': case 'h': process.stdout.write(USAGE); process.exit(0); break;
      default: throw new Error(`unknown option --${name}\n${USAGE}`);
    }
  }
  if (!positional.length) throw new Error(`missing <sceneName>\n${USAGE}`);
  if (positional.length > 1) throw new Error(`only one scene at a time, got: ${positional.join(' ')}`);
  if (!['auto', 'never', 'always'].includes(opts.orient)) throw new Error(`--orient must be auto|never|always`);
  if (!['detail', 'poly'].includes(opts.surface)) throw new Error(`--surface must be detail|poly`);
  if (!(opts.cell > 0)) throw new Error(`--cell must be > 0 (got ${opts.cell})`);
  if (opts.cellHeight !== null && !(opts.cellHeight > 0)) throw new Error(`--cell-height must be > 0`);
  if (!(opts.height > 0)) throw new Error(`--height must be > 0`);
  if (!(opts.radius >= 0)) throw new Error(`--radius must be >= 0`);
  if (!(opts.climb >= 0)) throw new Error(`--climb must be >= 0`);
  if (!(opts.slope >= 0 && opts.slope < 90)) throw new Error(`--slope must be in [0, 90) degrees`);
  if (!(opts.region >= 0) || !(opts.merge >= 0)) throw new Error(`--region and --merge must be >= 0`);
  if (!(opts.detail >= 0)) throw new Error(`--detail must be >= 0 (0 disables the height detail mesh)`);
  if (!Number.isInteger(opts.tileSize) || opts.tileSize < 4) throw new Error(`--tile-size must be an integer >= 4 cells`);
  return { scene: positional[0], opts };
}

/**
 * Metres -> Recast voxel units. Recast's walkableHeight/Climb/Radius are counts
 * of heightfield cells (ch for the vertical ones, cs for the radius), and the
 * generator squares minRegionArea/mergeRegionArea itself, so we hand it a *side*
 * in cells.
 */
function toVoxels(opts, cs, ch) {
  return {
    walkableRadius: Math.max(1, Math.round(opts.radius / cs)),
    walkableHeight: Math.max(1, Math.ceil(opts.height / ch)),
    walkableClimb: Math.max(0, Math.floor(opts.climb / ch)),
    minRegionArea: Math.max(1, Math.round(opts.region / cs)),
    mergeRegionArea: Math.max(1, Math.round(opts.merge / cs)),
    maxEdgeLen: opts.maxEdge > 0 ? Math.max(1, Math.round(opts.maxEdge / cs)) : 12,
  };
}

/** Navmesh surface as a triangle soup: the walkable tops only. */
function extractSurface(navMesh, mode) {
  if (mode === 'detail') return getNavMeshPositionsAndIndices(navMesh);
  // fan-triangulate the detour polygons: fewer triangles, no height detail
  const positions = [];
  const indices = [];
  const maxTiles = navMesh.getMaxTiles();
  for (let ti = 0; ti < maxTiles; ti++) {
    const tile = navMesh.getTile(ti);
    const header = tile.header();
    if (!header) continue;
    for (let pi = 0; pi < header.polyCount(); pi++) {
      const poly = tile.polys(pi);
      if (poly.getType() === 1) continue; // off-mesh connection
      const n = poly.vertCount();
      const base = positions.length / 3;
      for (let vi = 0; vi < n; vi++) {
        const o = poly.verts(vi) * 3;
        positions.push(tile.verts(o), tile.verts(o + 1), tile.verts(o + 2));
      }
      for (let vi = 1; vi < n - 1; vi++) indices.push(base, base + vi, base + vi + 1);
    }
  }
  return [positions, indices];
}

async function main() {
  let parsed;
  try {
    parsed = parseArgs(process.argv.slice(2));
  } catch (e) {
    console.error(e.message);
    process.exit(2);
  }
  const { scene, opts } = parsed;
  const log = opts.quiet ? () => {} : (...a) => console.log(...a);
  const t0 = Date.now();

  // recast-navigation is a wasm module: nothing in @recast-navigation/* works
  // before this resolves. In Node the compat build carries the binary inline,
  // so no file path / locateFile hook is needed.
  await init();

  const glbPath = path.resolve(REPO, opts.in ?? path.join('work', scene, 'pc', 'collision.collision.glb'));
  const outPath = path.resolve(REPO, opts.out ?? path.join('work', scene, 'pc', 'nav.json'));
  if (!existsSync(glbPath)) {
    console.error(`[bake] no collider at ${glbPath}\n[bake] known scenes: ${knownScenes().join(', ')}`);
    process.exit(2);
  }

  // ---- 1. load --------------------------------------------------------------
  let mesh;
  try {
    mesh = loadGlb(glbPath);
  } catch (e) {
    console.error(`[bake] ${e instanceof GlbError ? 'GLB' : 'fatal'}: ${e.message}`);
    throw e;
  }
  const src = boundsOf(mesh.positions);
  log(`[bake] ${scene}: ${path.relative(REPO, glbPath)} ${(statSync(glbPath).size / 1024).toFixed(0)} kB`);
  log(`[bake]   nodes ${mesh.stats.nodes}, primitives ${mesh.stats.primitives}, ` +
      `${mesh.stats.triangles} tris / ${mesh.stats.vertices} verts`);
  log(`[bake]   source bbox x [${src.min[0].toFixed(1)}, ${src.max[0].toFixed(1)}] ` +
      `y [${src.min[1].toFixed(1)}, ${src.max[1].toFixed(1)}] z [${src.min[2].toFixed(1)}, ${src.max[2].toFixed(1)}]`);

  // ---- 2. winding / normal repair ------------------------------------------
  const oriented = orientForWalking(mesh.positions, mesh.indices, opts.orient);
  const rep = oriented.report;
  if (rep.mode === 'never') {
    log(`[bake]   winding left as-is (--orient never)`);
  } else {
    const b = rep.before, a = rep.after;
    log(`[bake]   winding: in up ${b.up} / down ${b.down} / side ${b.side} (up area ${b.upArea} m2, down ${b.downArea} m2)`);
    log(`[bake]   winding: out up ${a.up} / down ${a.down} / side ${a.side} (up area ${a.upArea} m2, down ${a.downArea} m2)`);
    log(`[bake]   ${rep.componentCount} patches, ${rep.flippedComponents} flipped ` +
        `(${rep.flippedTris} tris), ${rep.degenerateDropped} degenerate dropped`);
    for (const c of rep.components.slice(0, 3)) {
      log(`[bake]     patch ${c.tris} tris closed=${c.closed} up ${c.upArea} / down ${c.downArea} m2 -> ${c.rule}`);
    }
  }

  // ---- 3. recast config -----------------------------------------------------
  const cs = opts.cell;
  const ch = opts.cellHeight ?? opts.cell;
  const vx = toVoxels(opts, cs, ch);
  const slope = opts.noSlopeFilter ? 89.9 : opts.slope;
  const config = {
    cs,
    ch,
    walkableSlopeAngle: slope,
    walkableRadius: vx.walkableRadius,
    walkableHeight: vx.walkableHeight,
    walkableClimb: vx.walkableClimb,
    minRegionArea: vx.minRegionArea,
    mergeRegionArea: vx.mergeRegionArea,
    maxEdgeLen: vx.maxEdgeLen,
    maxSimplificationError: 1.3,
    maxVertsPerPoly: 6,
    detailSampleDist: opts.detail,
    detailSampleMaxError: opts.detailError,
    borderSize: 0,
    tileSize: opts.solo ? 0 : opts.tileSize,
  };
  const gridX = Math.floor((src.max[0] - src.min[0]) / cs) + 1;
  const gridZ = Math.floor((src.max[2] - src.min[2]) / cs) + 1;
  const gridY = Math.floor((src.max[1] - src.min[1]) / ch) + 1;
  log(`[bake]   cs ${cs} ch ${ch} -> grid ${gridX} x ${gridY} x ${gridZ} cells` +
      (opts.solo ? ' (solo)' : ` -> ${Math.ceil(gridX / opts.tileSize)} x ${Math.ceil(gridZ / opts.tileSize)} tiles`));
  log(`[bake]   radius ${vx.walkableRadius} cells, height ${vx.walkableHeight} cells, ` +
      `climb ${vx.walkableClimb} cells, slope ${slope}${opts.noSlopeFilter ? ' (slope filter bypassed)' : 'deg'}, ` +
      `minRegion ${vx.minRegionArea} cells`);

  // ---- 4. build -------------------------------------------------------------
  const tBuild = Date.now();
  const result = (opts.solo ? generateSoloNavMesh : generateTiledNavMesh)(
    oriented.positions, oriented.indices, config, false);
  if (!result.success) {
    console.error(`[bake] FAILED: ${result.error}`);
    console.error('[bake] the steps that can fail are heightfield creation, rasterisation,');
    console.error('[bake] compaction, erosion, distance field, regions, contours, polymesh,');
    console.error('[bake] detail mesh, then "Failed to create Detour navmesh data" which means');
    console.error('[bake] zero polygons survived. Try --slope 89.9, a smaller --radius, a');
    console.error('[bake] larger --climb, --orient always, or a coarser --cell.');
    process.exit(3);
  }
  log(`[bake]   built in ${((Date.now() - tBuild) / 1000).toFixed(1)}s`);

  // ---- 5. extract + compact -------------------------------------------------
  const [rawPos, rawIdx] = extractSurface(result.navMesh, opts.surface);
  const surface = weldRounded(rawPos, rawIdx, 3);
  const tris = dedupeTriangles(surface.indices);
  if (!tris.length) {
    // generateTiledNavMesh reports success even when not one polygon survived,
    // which is what an inverted shell looks like: Recast happily returns an
    // empty navmesh instead of failing.
    console.error('[bake] FAILED: the build succeeded but produced ZERO triangles.');
    console.error('[bake] no triangle survived the walkability filters. Compare');
    console.error('[bake] --orient never / auto / always to see whether it is the winding,');
    console.error('[bake] then try a smaller --radius, a bigger --climb or --slope.');
    result.navMesh.destroy?.();
    process.exit(3);
  }
  const pos3 = surface.positions;
  const bb = boundsOf(pos3);
  const comps = components(pos3, tris);
  const area = totalArea(pos3, tris);
  const stranded = comps.sizes.reduce((n, s) => n + (s < 20 ? s : 0), 0);
  const strandedAreas = comps.areas.reduce((n, a, i) => n + (comps.sizes[i] < 20 ? a : 0), 0);

  // ---- 4b. coverage smoke test ---------------------------------------------
  // The same probe verify.mjs gates the bake on, printed here so a parameter
  // sweep shows at once whether the path the player actually walks is on the
  // mesh, without a second process and a re-read of the json.
  const metaPath = path.join(REPO, 'work', scene, 'viewer_assets', 'collision.json');
  if (tris.length && existsSync(metaPath)) {
    let meta = null;
    try {
      meta = JSON.parse(readFileSync(metaPath, 'utf8'));
    } catch (e) {
      log(`[bake]   walk coverage: unreadable collision.json (${e.message})`);
    }
    if (meta) {
      const grid = new XzGrid(pos3, tris, 2.25);
      const samples = [];
      if (meta.spawn) samples.push(['spawn', meta.spawn.x, meta.spawn.z]);
      for (const [x, z] of walkPathSamples(meta.walk_path ?? [], 1)) samples.push(['walk_path', x, z]);
      const dists = [];
      let ok = 0, inside = 0, worst = { d: -1 };
      for (const [what, x, z] of samples) {
        const r = grid.nearest(x, z);
        dists.push(r.d);
        if (r.inside) inside++;
        if (r.d <= 1.5) ok++;
        if (r.d > worst.d) worst = { d: r.d, what, x, z };
      }
      dists.sort((a, b) => a - b);
      log(`[bake]   walk coverage: ${ok}/${samples.length} samples <= 1.5 m ` +
          `(${inside} inside a triangle), median ${dists[dists.length >> 1]?.toFixed(2) ?? '-'} m, ` +
          `max ${worst.d.toFixed(2)} m${worst.d > 1.5 ? `  <-- worst: ${worst.what} (${worst.x.toFixed(1)}, ${worst.z.toFixed(1)})` : ''}`);
    }
  }

  const nav = {
    version: 1,
    source: path.basename(glbPath),
    generated: new Date().toISOString(),
    build: {
      cellSize: cs,
      cellHeight: ch,
      walkableRadius: opts.radius,
      walkableHeight: opts.height,
      walkableClimb: opts.climb,
      // the angle actually used: --no-slope-filter overrides --slope
      walkableSlopeDeg: slope,
    },
    bbox: [round(bb.min[0]), round(bb.min[1]), round(bb.min[2]), round(bb.max[0]), round(bb.max[1]), round(bb.max[2])],
    verts: arrayToNumbers(pos3),
    tris: Array.from(tris),
    stats: {
      triCount: tris.length / 3,
      areaM2: +area.toFixed(2),
      largestComponentTris: comps.largest,
    },
  };

  const json = JSON.stringify(nav);
  log(`[bake]   navmesh: ${nav.stats.triCount} tris, ${pos3.length / 3} verts, ` +
      `${nav.stats.areaM2} m2 (plan ${totalArea(pos3, tris, true).toFixed(1)} m2)`);
  log(`[bake]   y range ${bb.min[1].toFixed(2)} .. ${bb.max[1].toFixed(2)} (height ${Math.max(0, bb.max[1] - bb.min[1]).toFixed(2)})`);
  log(`[bake]   ${comps.count} components, largest ${comps.largest} tris / ${comps.areas[0]} m2, ` +
      `next ${comps.sizes.slice(1, 5).join(', ') || '-'}`);
  log(`[bake]   stranded (<20 tris): ${stranded} tris (${comps.sizes.length ? (100 * stranded / (tris.length / 3)).toFixed(2) : '0'}%), ` +
      `${strandedAreas.toFixed(1)} m2`);
  log(`[bake]   nav.json would be ${(json.length / 1024 / 1024).toFixed(2)} MB` +
      (json.length > 2e6 ? '  <-- OVER THE 2 MB BUDGET: raise --cell or lower --detail' : ''));

  if (opts.dry) {
    log('[bake] --dry: nothing written');
    return;
  }

  writeFileSync(outPath, json);
  log(`[bake] wrote ${path.relative(REPO, outPath)} (${(statSync(outPath).size / 1024).toFixed(0)} kB) in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  if (opts.obj) {
    const objPath = outPath.replace(/\.json$/, '.obj');
    writeObj(objPath, pos3, tris, `nav_${scene}`);
    log(`[bake] wrote ${path.relative(REPO, objPath)} (${(statSync(objPath).size / 1024).toFixed(0)} kB)`);
  }
  if (opts.dtnav !== null) {
    const binPath = path.resolve(REPO, opts.dtnav || path.join('work', scene, 'pc', 'nav.dtnav.bin'));
    const bytes = exportNavMesh(result.navMesh);
    const buf = bytes instanceof Uint8Array
      ? Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength)
      : Buffer.from(bytes);
    writeFileSync(binPath, buf);
    log(`[bake] wrote ${path.relative(REPO, binPath)} (${(buf.length / 1024).toFixed(0)} kB, detour binary for dt pathfinding)`);
  }
  result.navMesh.destroy?.();
}

function dedupeTriangles(indices) {
  const seen = new Set();
  const out = new Uint32Array(indices.length);
  let n = 0;
  for (let t = 0; t < indices.length; t += 3) {
    let a = indices[t], b = indices[t + 1], c = indices[t + 2];
    if (a === b || b === c || a === c) continue;
    // canonicalise rotation so the same triangle emitted by two tiles is merged
    if (a > b || (a === b && b > c)) { const t2 = a; a = b; b = c; c = t2; }
    if (a > b || (a === b && b > c)) { const t2 = a; a = c; c = b; b = t2; }
    const key = `${a},${b},${c}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out[n++] = indices[t]; out[n++] = indices[t + 1]; out[n++] = indices[t + 2];
  }
  return out.subarray(0, n);
}

function arrayToNumbers(a) {
  const out = new Array(a.length);
  // re-round in double precision: the values came out of a Float32Array, so
  // -55.99 is stored as -55.9900016784668 and would break the 3-decimal contract
  for (let i = 0; i < a.length; i++) out[i] = Math.round(a[i] * 1000) / 1000;
  return out;
}

function round(v) {
  return Math.round(v * 1000) / 1000;
}

function knownScenes() {
  try {
    return readdirSync(path.join(REPO, 'work')).filter((d) =>
      existsSync(path.join(REPO, 'work', d, 'pc', 'collision.collision.glb')));
  } catch {
    return [];
  }
}

main().catch((e) => {
  console.error(e && e.stack ? e.stack : String(e));
  process.exit(1);
});
