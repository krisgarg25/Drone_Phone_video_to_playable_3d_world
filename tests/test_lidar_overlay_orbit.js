#!/usr/bin/env node
/*
 * test_lidar_overlay_orbit.js
 *
 * Drives the SHIPPED persist + reproject path (viewer/lidar_overlay.js) with a
 * synthetic cube. World points are seeded on ONE face through the shipped
 * persist (unproject + voxel hash), then a sequence of orbiting camera poses
 * is applied and the map is re-projected with the shipped project:
 *
 *   (a) projected dots stay inside that face's image region across the orbit
 *   (b) a second, never-seeded face has zero capture dots
 *   (c) a raster of the overlay is sparse dots, not a filled rectangle
 *
 * Run:  node tests/test_lidar_overlay_orbit.js
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.resolve(__dirname, "..");

let pass = 0, fail = 0;
function check(cond, label, detail) {
  if (cond) { pass++; console.log("pass  " + label); }
  else { fail++; console.log("FAIL  " + label + (detail ? "  -> " + detail : "")); }
}

// ---------- load the shipped module in a browser-like sandbox ----------
const sandbox = {};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(
  fs.readFileSync(path.join(ROOT, "viewer", "lidar_overlay.js"), "utf8"),
  sandbox, { filename: "viewer/lidar_overlay.js" });
const LO = sandbox.window.LidarOverlay;
check(LO && typeof LO === "object", "shipped viewer/lidar_overlay.js loads (browser-like, window defined)");
check(typeof LO.createMap === "function" && typeof LO.persist === "function" &&
      typeof LO.project === "function" && typeof LO.coverMapping === "function",
      "shipped persist + reproject API present (createMap/persist/project/coverMapping)");

// ---------- synthetic cube ----------
const HALF = 0.5;                        // cube centered at the origin, side 1 m
const FOCAL = LO.DEFAULTS.focal;         // same pinhole model as capture.html
const map = LO.createMap();

// seed face A (z = +HALF) from a camera at (0,0,2.4) looking straight at it —
// through the SHIPPED persist: give normalized feature coords + depth and let
// the shipped unproject/voxel path build the world map.
const SEED_CAM = { pos: { x: 0, y: 0, z: 2.4 }, quat: { x: 0, y: 0, z: 0, w: 1 } };
const N = 9;
let seeded = 0;
for (let i = 0; i < N; i++) {
  for (let j = 0; j < N; j++) {
    const x = -0.42 + 0.84 * (i / (N - 1));
    const y = -0.42 + 0.84 * (j / (N - 1));
    const depth = SEED_CAM.pos.z - HALF;
    if (LO.persist(map, SEED_CAM, x / (FOCAL * depth), -y / (FOCAL * depth), depth) >= 0) seeded++;
  }
}
check(seeded >= 70, "shipped persist seeded one cube face (" + seeded + " world points)");

let onA = 0, onB = 0;
for (const p of map.points) {
  if (Math.abs(p.z - HALF) < 0.06 && Math.abs(p.x) <= HALF + 0.02 && Math.abs(p.y) <= HALF + 0.02) onA++;
  if (Math.abs(p.z + HALF) < 0.06) onB++;
}
check(onA === map.points.length,
      "every world point lies on the seeded face A (" + onA + "/" + map.points.length + ")");
check(onB === 0, "persist leaked nothing onto the never-seeded face B");

// face image region via the shipped path: seed a face's 4 corners through
// persist into a tiny map, then project them with the same pose.
const cornerA = LO.createMap();
{
  const d = SEED_CAM.pos.z - HALF;
  for (const sx of [-HALF, HALF]) for (const sy of [-HALF, HALF])
    LO.persist(cornerA, SEED_CAM, sx / (FOCAL * d), -sy / (FOCAL * d), d);
}
const BACK_CAM = { pos: { x: 0, y: 0, z: -2.4 }, quat: { x: 0, y: 1, z: 0, w: 0 } };
const cornerB = LO.createMap();
{
  const d = Math.abs(BACK_CAM.pos.z + HALF);
  // camera looks along +Z: world (x,y,-HALF) -> cam frame (-x, y, -d)
  for (const sx of [-HALF, HALF]) for (const sy of [-HALF, HALF])
    LO.persist(cornerB, BACK_CAM, -sx / (FOCAL * d), -sy / (FOCAL * d), d);
}
function bbox(dots) {
  if (dots.length === 0) return null;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const d of dots) {
    x0 = Math.min(x0, d.px); x1 = Math.max(x1, d.px);
    y0 = Math.min(y0, d.py); y1 = Math.max(y1, d.py);
  }
  return { x0, x1, y0, y1 };
}
function orbitPose(thetaDeg) {
  const th = thetaDeg * Math.PI / 180;
  return {
    pos: { x: 2.4 * Math.sin(th), y: 0, z: 2.4 * Math.cos(th) },
    quat: { x: 0, y: Math.sin(th / 2), z: 0, w: Math.cos(th / 2) }
  };
}

// snapshot world points: projection must not mutate the persisted map
const before = map.points.map(p => [p.x.toFixed(6), p.y.toFixed(6), p.z.toFixed(6)].join(","));

// ---------- (a) orbit the front hemisphere: dots stay on face A ----------
let orbitChecked = 0;
for (let th = -80; th <= 80; th += 20) {
  const cam = orbitPose(th);
  const dots = LO.project(map, cam);
  const region = bbox(LO.project(cornerA, cam));
  if (!region) { check(false, `orbit ${th}: face A corner region visible`); continue; }
  const M = 2; // px tolerance
  const inside = dots.every(d => d.px >= region.x0 - M && d.px <= region.x1 + M &&
                                  d.py >= region.y0 - M && d.py <= region.y1 + M);
  check(inside, `orbit ${String(th).padStart(4)}deg: all ${dots.length} dots stay inside face A image region`);
  orbitChecked++;
}
check(orbitChecked === 9, "full front orbit exercised (" + orbitChecked + "/9 poses)");
{
  const dots0 = LO.project(map, orbitPose(0));
  check(dots0.length >= 0.6 * map.points.length,
        "head-on view shows most of the captured face (" + dots0.length + "/" + map.points.length + ")");
}

// ---------- (b) look at the never-seeded face B: zero capture dots ----------
{
  const dotsB = LO.project(map, BACK_CAM);
  const regionB = bbox(LO.project(cornerB, BACK_CAM));
  check(regionB !== null, "face B corner region visible from the back pose");
  const inB = regionB ? dotsB.filter(d => d.px >= regionB.x0 && d.px <= regionB.x1 &&
                                          d.py >= regionB.y0 && d.py <= regionB.y1).length : dotsB.length;
  check(inB === 0, "never-seeded face B has zero capture dots (" + inB + ")");
  check(dotsB.length === 0,
        "backface culling hides face A dots when viewing the opposite side (" + dotsB.length + " leaked)");
}

// ---------- (c) raster of the overlay: sparse dots, not a filled wash ------
{
  const CW = 640, CH = 480;
  const cam = orbitPose(0);
  const dots = LO.project(map, cam);
  const mp = LO.coverMapping(160, 120, CW, CH, 640, 480); // shipped proc->canvas mapping
  const grid = new Uint8Array(CW * CH);
  let filled = 0;
  const R = 2; // dot radius in canvas px (same order as drawOverlay)
  for (const d of dots) {
    const cx = mp.offX + d.px * mp.sx, cy = mp.offY + d.py * mp.sy;
    for (let yy = Math.max(0, Math.floor(cy - R)); yy <= Math.min(CH - 1, Math.ceil(cy + R)); yy++)
      for (let xx = Math.max(0, Math.floor(cx - R)); xx <= Math.min(CW - 1, Math.ceil(cx + R)); xx++) {
        const dx = xx - cx, dy = yy - cy;
        if (dx * dx + dy * dy <= R * R && !grid[yy * CW + xx]) { grid[yy * CW + xx] = 1; filled++; }
      }
  }
  const coverage = filled / (CW * CH);
  console.log("      raster: " + dots.length + " dots, " + filled + " px filled, coverage " +
              (coverage * 100).toFixed(2) + "%");
  check(dots.length >= 20, "overlay holds a lidar-like set of discrete dots (" + dots.length + ")");
  check(coverage > 0.0005, "overlay is not empty where the face was scanned");
  check(coverage < 0.25, "overlay is SPARSE dots, not a filled rectangle over the canvas");
}

// ---------- shipped coverage-gap report ("did I forget the bed?") --------
{
  const Y = 72, P = 18;
  const full = new Float32Array(Y * P).fill(1);
  check(LO.uncoveredArcs(full, Y, P, 0.03, 4).length === 0,
        "uncoveredArcs: full coverage -> no missed arcs");

  const gapMid = new Float32Array(Y * P).fill(1);
  for (let c = 40; c <= 50; c++) for (let r = 7; r < 11; r++) gapMid[r * Y + c] = 0;
  const mid = LO.uncoveredArcs(gapMid, Y, P, 0.03, 4).find(a => a.band === "mid");
  check(!!mid && mid.cells === 11 && mid.from === 200,
        "uncoveredArcs: mid-band gap detected (" + (mid ? mid.cells + " cells @ " + mid.from + "deg" : "none") + ")");

  const gapWrap = new Float32Array(Y * P).fill(1);
  for (let c = 69; c < 72; c++) for (let r = 0; r < 7; r++) gapWrap[r * Y + c] = 0;
  for (let c = 0; c <= 1; c++) for (let r = 0; r < 7; r++) gapWrap[r * Y + c] = 0;
  const low = LO.uncoveredArcs(gapWrap, Y, P, 0.03, 4).find(a => a.band === "low");
  check(!!low && low.cells === 5,
        "uncoveredArcs: wrap-around gap merged into one arc (" + (low ? low.cells : "none") + " cells)");
}

// ---------- world-anchored: projection never mutates the map ----------
{
  const after = map.points.map(p => [p.x.toFixed(6), p.y.toFixed(6), p.z.toFixed(6)].join(","));
  check(after.every((v, i) => v === before[i]) && after.length === before.length,
        "reprojection left all world points untouched (dots are world-anchored)");
}

console.log("");
console.log(fail === 0 ? "ALL " + pass + " LIDAR OVERLAY CHECKS PASSED"
                       : pass + " passed, " + fail + " FAILED");
process.exit(fail === 0 ? 0 : 1);
