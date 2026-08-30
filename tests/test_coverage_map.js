#!/usr/bin/env node
// =====================================================================
// test_coverage_map.js — proves the coverage engine answers the question the
// old capture page could not: "I am looking at this cube face again; do you
// know I already covered it?"
//
// Runs the shipped viewer/coverage_map.js, no mocks.
//
// Two conventions this file leans on, both easy to get backwards:
//   * WebXR cameras look down LOCAL -Z. So a camera at z=+3 with the identity
//     quaternion is looking at a surface at z=+1 (head on), and yaw-180 there
//     is looking away from it.
//   * Coverage needs >45 deg of azimuth change to earn a new bin (8 bins), but
//     the incidence gate stops you at 68 deg off-normal. So a flat wall cannot
//     be covered by strafing alone -- you also have to raise/lower the phone
//     and pick up elevation bins. The station sets below reflect that.
// =====================================================================
"use strict";
const path = require("path");
const CM = require(path.join(__dirname, "..", "viewer", "coverage_map.js"));

let pass = 0, fail = 0;
function ok(cond, label, extra) {
  if (cond) { pass++; console.log("pass  " + label); }
  else { fail++; console.log("FAIL  " + label + (extra ? "\n      " + extra : "")); }
}
function near(a, b, tol, label) { ok(Math.abs(a - b) <= tol, label, `got ${a}, want ${b}+-${tol}`); }

// ---------------------------------------------------------------------
// A 2 m cube centred at the origin, floor at y=0. Its +Z face is our target.
// ---------------------------------------------------------------------
const FACE_Z = 1.0;
const FACE_Z_NORMAL = { x: 0, y: 0, z: 1 };
function faceZPoints() {
  const pts = [];
  for (let x = -0.9; x <= 0.9001; x += 0.2)
    for (let y = 0.3; y <= 1.9001; y += 0.2) pts.push({ x, y, z: FACE_Z });
  return pts;
}
const FACE_PTS = faceZPoints();

function observeFace(map, cam) {
  let accepted = 0;
  for (const p of FACE_PTS) if (CM.observe(map, cam, p, FACE_Z_NORMAL) >= 0) accepted++;
  return accepted;
}

// One spot, dead ahead. This is the "stand still and stare" scan.
const STARE = { x: 0, y: 1.1, z: 2.6 };
// A real scan of a flat face: strafe left/right, then high/low for elevation.
const ORBIT = [
  { x:  0.0, y: 1.1, z: 2.6 },
  { x: -1.8, y: 1.1, z: 2.2 },
  { x:  1.8, y: 1.1, z: 2.2 },
  { x:  0.0, y: 2.6, z: 2.2 },
  { x:  0.0, y: 0.4, z: 2.2 }
];
function scanFace(map) { for (const c of ORBIT) observeFace(map, c); }

// =====================================================================
console.log("\n--- direction binning ---");
{
  const o = CM.createMap().opts;
  ok(CM.dirBin(o, 0, 0, 1) === CM.dirBin(o, 0, 0, 1), "same direction always lands in the same bin");
  ok(CM.dirBin(o, 0, 0, 1) !== CM.dirBin(o, 1, 0, 0), "orthogonal directions land in different bins");
  ok(CM.dirBin(o, 0, 0, 1) !== CM.dirBin(o, 0, 1, 0), "straight-on and overhead differ (elevation bins work)");
  const seen = new Set();
  for (let i = 0; i < 64; i++) {
    const a = i / 64 * Math.PI * 2;
    for (const el of [-0.8, 0, 0.8]) seen.add(CM.dirBin(o, Math.sin(a), el, Math.cos(a)));
  }
  ok(seen.size === o.azBins * o.elBins, `every one of the ${o.azBins * o.elBins} bins is reachable (${seen.size})`);
  ok([...seen].every(b => b >= 0 && b < 32), "all bin indices fit inside a Uint32 mask");
  ok(CM.popcount32(0) === 0 && CM.popcount32(7) === 3 && CM.popcount32(1 << 23) === 1,
     "popcount32 is correct over the 24-bit mask range");
}

console.log("\n--- THE ASK: revisiting a covered face is recognised ---");
{
  const map = CM.createMap();

  // Pass 1: one station, straight on. Plenty of observations, ZERO baseline.
  observeFace(map, STARE);
  const afterOne = CM.stats(map);
  ok(afterOne.total === FACE_PTS.length, `the whole face registered surfels (${afterOne.total})`);
  ok(afterOne.covered === 0, "one long stare from a single spot is NOT covered (no baseline)");
  ok(afterOne.thin === afterOne.total, "...every surfel reads as thin");

  // Repeat the identical stare 5 more times: no new directions, no new surfels.
  const before = map.surfels.length;
  for (let i = 0; i < 5; i++) observeFace(map, STARE);
  ok(map.surfels.length === before,
     `re-observing the same surface creates NO new surfels (${before} -> ${map.surfels.length})`);
  ok(CM.stats(map).covered === 0, "...and staring longer still does not make it covered");
  ok(map.surfels[0].seen === 6, "the surfel counted all 6 visits, it just refused to trust them");

  // Now actually scan it: strafe, then raise and lower the phone.
  CM.clear(map);
  scanFace(map);
  const st = CM.stats(map);
  ok(st.covered > 0, `a real 5-station scan covers the face (${st.covered} surfels)`);
  ok(st.coveredPct > 90, `nearly the whole face is covered (${st.coveredPct.toFixed(0)}%)`);
  ok(map.surfels.length === FACE_PTS.length,
     `all 5 passes fold into the same ${FACE_PTS.length} surfels — coverage accumulated, not duplicated`);

  // The user-facing question, asked of the map:
  ok(CM.stateOf(map, map.surfels[0]) === CM.STATE_COVERED,
     "querying a revisited surfel returns COVERED — the phone can now say 'done this'");

  // And the intermediate state must exist, otherwise the HUD has nothing to show.
  const m2 = CM.createMap();
  observeFace(m2, ORBIT[0]);
  observeFace(m2, ORBIT[1]);
  const st2 = CM.stats(m2);
  ok(st2.partial > 0 && st2.covered === 0,
     `two stations reads PARTIAL, not covered (${st2.partial} partial, ${st2.covered} covered)`);
}

console.log("\n--- a face never looked at stays empty ---");
{
  const map = CM.createMap();
  scanFace(map);
  ok(map.surfels.every(s => Math.abs(s.z - FACE_Z) < 0.2), "only the scanned face produced surfels");
  ok(!map.surfels.some(s => s.z < -0.5), "the opposite (-Z) face has no surfels at all");
  ok(CM.stats(map).thinArea === 0, "nothing is left flagged as needing work");
}

console.log("\n--- range and incidence gating ---");
{
  const p = { x: 0, y: 1, z: 0 }, n = { x: 0, y: 0, z: 1 };
  const map = CM.createMap();
  ok(CM.observe(map, { x: 0, y: 1, z: 0.05 }, p, n) === -1, "a sample nearer than `near` is rejected");
  ok(CM.observe(map, { x: 0, y: 1, z: 99 }, p, n) === -1, "a sample beyond `far` is rejected");
  ok(CM.observe(map, { x: 0, y: 1, z: 2 }, p, n) >= 0, "a sample in range is accepted");
  ok(CM.observe(map, { x: 3, y: 1, z: 0.02 }, p, n) === -1,
     "a grazing observation is rejected (would not reconstruct)");
  ok(CM.observe(map, { x: 0, y: 1, z: NaN }, p, n) === -1, "a non-finite sample is rejected");
  ok(map.rejected === 4, `every rejection was counted (${map.rejected})`);

  // Distance gating: in range for mapping, too far to count toward covered.
  const map3 = CM.createMap({ goodDist: 1.5, far: 10 });
  for (const c of [{ x: -3, y: 1, z: 3 }, { x: 0, y: 3.5, z: 3.5 }, { x: 3, y: 1, z: 3 }]) {
    CM.observe(map3, c, p, n);
  }
  const s3 = CM.stats(map3);
  ok(s3.total === 1 && s3.covered === 0,
     "3 good directions but all beyond goodDist -> mapped, still not covered");
  ok(CM.popcount32(map3.surfels[0].dirs) === 3, "...the directions were recorded, just not credited");
}

console.log("\n--- projection: dots stay pinned to the surface ---");
{
  const map = CM.createMap();
  scanFace(map);

  const fx = 1.2, fy = 1.6, W = 320, H = 240;
  const identity = { x: 0, y: 0, z: 0, w: 1 };   // WebXR: looking down -Z
  const yaw180 = { x: 0, y: 1, z: 0, w: 0 };     // turned around, looking down +Z
  const eye = { x: 0, y: 1.1, z: 3.0 };          // in front of the face
  const headOn = { pos: eye, quat: identity };

  const front = CM.project(map, headOn, W, H, fx, fy);
  ok(front.length > FACE_PTS.length * 0.9, `head-on view sees the scanned face (${front.length}/${FACE_PTS.length} dots)`);
  ok(front.every(d => d.state === CM.STATE_COVERED), "head-on dots all read COVERED");
  // margin lets dots sit slightly outside the frame on purpose; nothing wilder.
  ok(front.every(d => d.px >= -0.06 * W && d.px <= 1.06 * W && d.py >= -0.06 * H && d.py <= 1.06 * H),
     "dots land inside the frame plus the configured margin");
  ok(front.every(d => d.dist > 1.9 && d.dist < 2.1), "reported depth matches the 2 m standoff");

  // Y must not be flipped: a surfel above the camera draws near the top.
  const high = front.reduce((a, b) => (map.surfels[a.idx].y > map.surfels[b.idx].y ? a : b));
  const low = front.reduce((a, b) => (map.surfels[a.idx].y < map.surfels[b.idx].y ? a : b));
  ok(high.py < low.py, `a high surfel draws above a low one (py ${high.py.toFixed(0)} < ${low.py.toFixed(0)})`);
  const left = front.reduce((a, b) => (map.surfels[a.idx].x < map.surfels[b.idx].x ? a : b));
  ok(left.px < W / 2, `a -X surfel draws left of centre (px ${left.px.toFixed(0)})`);

  // Two nearby camera positions must project the SAME surfels (world-anchored).
  const b = CM.project(map, { pos: { x: 0.05, y: 1.1, z: 3.0 }, quat: identity }, W, H, fx, fy);
  const setA = new Set(front.map(d => d.idx));
  const shared = b.filter(d => setA.has(d.idx)).length;
  ok(shared > b.length * 0.9,
     `a 5 cm camera move keeps the same dots (${shared}/${b.length} shared) — they are world-anchored`);
  const moved = b.map(d => { const m = front.find(x => x.idx === d.idx); return m ? Math.abs(m.px - d.px) : 0; });
  ok(Math.max(...moved) < W * 0.05,
     `and they shift only ${Math.max(...moved).toFixed(1)} px on screen — they do not swim`);

  // Backface: standing behind the cube and looking at the far side of the face.
  const behind = CM.project(map, { pos: { x: 0, y: 1.1, z: -3.0 }, quat: yaw180 }, W, H, fx, fy);
  ok(behind.length === 0, `the face is culled when seen from behind (${behind.length} leaked)`);
  // Turning away from a surface in front of you also shows nothing.
  const away = CM.project(map, { pos: eye, quat: yaw180 }, W, H, fx, fy);
  ok(away.length === 0, `facing away shows nothing (${away.length} leaked)`);

  // World points must not be mutated by projection.
  const snapshot = map.surfels.map(s => `${s.x},${s.y},${s.z}`).join("|");
  CM.project(map, { pos: { x: 1, y: 2, z: 4 }, quat: identity }, W, H, fx, fy);
  ok(snapshot === map.surfels.map(s => `${s.x},${s.y},${s.z}`).join("|"),
     "projection left every world point untouched");
}

console.log("\n--- gap report: WHERE do I go back to? ---");
{
  const map = CM.createMap();
  scanFace(map);                       // +Z face: properly scanned

  // The -X wall gets one glance in passing -- the wall you "completely forgot".
  const WALL_N = { x: -1, y: 0, z: 0 };
  const wallPts = [];
  for (let z = -0.9; z <= 0.9001; z += 0.2)
    for (let y = 0.3; y <= 1.9001; y += 0.2) wallPts.push({ x: -1.0, y, z });
  const glanceWall = cam => { for (const p of wallPts) CM.observe(map, cam, p, WALL_N); };
  glanceWall({ x: -3.0, y: 1.1, z: 0 });

  const st = CM.stats(map);
  ok(st.thin > 0 && st.covered > 0, `map holds both covered and thin surfels (${st.covered}/${st.thin})`);

  const gaps = CM.gapClusters(map, 0.5, 6);
  ok(gaps.length >= 1, `gap clustering found the under-scanned wall (${gaps.length} cluster(s))`);
  const biggest = gaps[0];
  near(biggest.cx, -1.0, 0.35, "the reported gap sits on the -X wall");
  ok(biggest.area > 0.1, `the gap carries a usable area estimate (${biggest.area.toFixed(2)} m2)`);
  ok(biggest.maxY > biggest.minY, `and a height range to describe (${biggest.minY.toFixed(1)}-${biggest.maxY.toFixed(1)} m)`);
  ok(gaps.every(g => g.count >= 6), "clusters below minCount are filtered out");
  ok(!gaps.some(g => Math.abs(g.cz - 1.0) < 0.35 && Math.abs(g.cx) < 0.5),
     "the properly-scanned +Z face is NOT reported as a gap");

  // Go back and scan it for real: strafe both ways, then raise the phone.
  glanceWall({ x: -2.6, y: 1.1, z: 1.6 });
  glanceWall({ x: -2.6, y: 1.1, z: -1.6 });
  glanceWall({ x: -2.8, y: 2.5, z: 0.0 });
  const gapsAfter = CM.gapClusters(map, 0.5, 6);
  const stillThere = gapsAfter.filter(g => Math.abs(g.cx + 1.0) < 0.35).length;
  ok(stillThere === 0, `after three more angles the wall drops off the gap list (${gapsAfter.length} left)`);

  // Direction to walk, for the report text.
  const bt = CM.bearingTo(0, 0, -3, 0);
  ok(bt.name === "W", `bearingTo names -X as West (got ${bt.name})`);
  near(bt.dist, 3, 0.01, "bearingTo reports the ground distance");
  ok(CM.bearingTo(0, 0, 0, -3).name === "N" && CM.bearingTo(0, 0, 0, 3).name === "S",
     "bearingTo treats -Z as North (WebXR local-floor convention)");
}

console.log("\n--- unprojectDepth <-> project round trip (the WebXR maths) ---");
{
  // A realistic off-centre phone AR frustum: skewed principal point, so a
  // dropped cxOff/cyOff term shows up immediately.
  const fx = 1.42, fy = 0.81, cxOff = 0.07, cyOff = -0.04;
  const W = 320, H = 240;

  // Camera-to-world, column-major, as XRView.transform.matrix gives it.
  // Yawed 30 deg about Y and standing at (1.2, 1.4, 2.5).
  const a = Math.PI / 6, ca = Math.cos(a), sa = Math.sin(a);
  const m = new Float64Array([
     ca, 0, -sa, 0,
      0, 1,   0, 0,
     sa, 0,  ca, 0,
    1.2, 1.4, 2.5, 1
  ]);
  const camPos = { x: m[12], y: m[13], z: m[14] };
  // Same rotation as a quaternion (about +Y by a).
  const quat = { x: 0, y: Math.sin(a / 2), z: 0, w: Math.cos(a / 2) };

  ok(CM.unprojectDepth(m, 0.5, 0.5, 0, fx, fy, cxOff, cyOff) === null,
     "a depth of exactly 0 is rejected (the spec's invalid-pixel value)");
  ok(CM.unprojectDepth(m, 0.5, 0.5, NaN, fx, fy, cxOff, cyOff) === null, "NaN depth is rejected");

  // Straight ahead: the point must be `depth` metres along the camera's -Z.
  const centre = CM.unprojectDepth(m, 0.5, 0.5, 3.0, fx, fy, 0, 0);
  near(centre.x, camPos.x - 3.0 * sa, 1e-9, "centre sample lies down the camera's -Z axis (x)");
  near(centre.y, camPos.y, 1e-9, "...level with the camera (y)");
  near(centre.z, camPos.z - 3.0 * ca, 1e-9, "...and `depth` metres away (z)");
  near(Math.hypot(centre.x - camPos.x, centre.y - camPos.y, centre.z - camPos.z), 3.0, 1e-9,
     "centre sample is exactly `depth` from the camera");

  // A depth sample is a distance from the camera PLANE, not a ray length: an
  // off-axis sample must therefore end up FURTHER than `depth` from the camera.
  const corner = CM.unprojectDepth(m, 0.05, 0.05, 3.0, fx, fy, cxOff, cyOff);
  const cornerRange = Math.hypot(corner.x - camPos.x, corner.y - camPos.y, corner.z - camPos.z);
  ok(cornerRange > 3.05,
     `an off-axis sample is further away than its z-depth (${cornerRange.toFixed(3)} m > 3 m) — plane distance, not ray length`);

  // +Y in view coords points DOWN, so a sample near the top of the buffer must
  // land ABOVE the camera. This is the flip MDN documents backwards.
  const top = CM.unprojectDepth(m, 0.5, 0.1, 3.0, fx, fy, 0, 0);
  const bot = CM.unprojectDepth(m, 0.5, 0.9, 3.0, fx, fy, 0, 0);
  ok(top.y > camPos.y && bot.y < camPos.y,
     `nyv=0.1 is above the camera and nyv=0.9 below it (${top.y.toFixed(2)} / ${bot.y.toFixed(2)}) — top-left origin, +Y down`);

  // The round trip: unproject a grid of samples, then project them back. Every
  // sample must return to the normalized view coords it came from.
  const map = CM.createMap({ voxel: 1e-6, maxIncidenceDeg: 89.9 });
  const probes = [];
  for (const nxv of [0.15, 0.35, 0.5, 0.72, 0.9])
    for (const nyv of [0.12, 0.4, 0.5, 0.66, 0.88]) {
      const depth = 1.4 + nxv * 1.7 + nyv * 0.9;      // a slanted surface
      const p = CM.unprojectDepth(m, nxv, nyv, depth, fx, fy, cxOff, cyOff);
      // normal irrelevant here; face it at the camera so nothing is gated out
      const n = { x: camPos.x - p.x, y: camPos.y - p.y, z: camPos.z - p.z };
      const nl = Math.hypot(n.x, n.y, n.z);
      n.x /= nl; n.y /= nl; n.z /= nl;
      const idx = CM.observe(map, camPos, p, n);
      if (idx >= 0) probes.push({ nxv, nyv, depth, idx });
    }
  ok(probes.length === 25, `all 25 probe samples were accepted (${probes.length})`);

  const back = CM.project(map, { pos: camPos, quat }, W, H, fx, fy, cxOff, cyOff);
  const byIdx = new Map(back.map(d => [d.idx, d]));
  let worstXY = 0, worstDepth = 0;
  for (const pr of probes) {
    const d = byIdx.get(pr.idx);
    if (!d) { worstXY = Infinity; continue; }
    worstXY = Math.max(worstXY, Math.abs(d.nx - pr.nxv), Math.abs(d.ny - pr.nyv));
    worstDepth = Math.max(worstDepth, Math.abs(d.dist - pr.depth));
  }
  ok(back.length === probes.length, `every probe projected back into view (${back.length}/${probes.length})`);
  ok(worstXY < 1e-6, `round trip returns the exact view coords (worst error ${worstXY.toExponential(1)})`);
  ok(worstDepth < 1e-6, `and the exact z-depth (worst error ${worstDepth.toExponential(1)})`);

  // Now prove the skew terms are actually load-bearing: projecting with the
  // skew dropped must visibly move the dots.
  const noSkew = CM.project(map, { pos: camPos, quat }, W, H, fx, fy);
  const byIdx2 = new Map(noSkew.map(d => [d.idx, d]));
  let skewShift = 0;
  for (const pr of probes) {
    const d = byIdx2.get(pr.idx);
    if (d) skewShift = Math.max(skewShift, Math.abs(d.nx - pr.nxv));
  }
  near(skewShift, Math.abs(cxOff) / 2, 1e-6,
     "dropping cxOff shifts every dot by exactly half the skew — the term is load-bearing");
}

console.log("\n--- insert cap is bounded and honest ---");
{
  // Small voxels over a patch that stays well inside [near, far], so the cap is
  // what stops us -- not the range gate.
  const map = CM.createMap({ maxSurfels: 50, voxel: 0.05 });
  const eye = { x: 0, y: 1, z: 0 };
  let attempted = 0;
  for (let x = -0.5; x <= 0.5001; x += 0.05)
    for (let y = 0.6; y <= 1.6001; y += 0.05) { attempted++; CM.observe(map, eye, { x, y, z: 2.0 }, null); }
  ok(attempted > 300, `attempted plenty of distinct voxels (${attempted})`);
  ok(map.surfels.length === 50, `surfel count is capped at maxSurfels (${map.surfels.length})`);
  ok(map.full === true, "the map reports that it hit the cap");
  ok(map.voxels.size === 50, "the voxel index did not grow past the cap either");

  // Existing surfels must still accumulate coverage past the cap.
  const first = map.surfels[0];
  const seenBefore = first.seen, dirsBefore = CM.popcount32(first.goodDirs);
  CM.observe(map, { x: first.x - 1.4, y: 1.9, z: first.z - 1.6 }, { x: first.x, y: first.y, z: first.z }, null);
  ok(first.seen > seenBefore, "surfels already in the map keep updating after the cap");
  ok(CM.popcount32(first.goodDirs) > dirsBefore, "...and keep earning new direction bins");
}

console.log("\n--- clear() resets everything ---");
{
  const map = CM.createMap();
  scanFace(map);
  ok(map.surfels.length > 0, "map populated before clear");
  CM.clear(map);
  const st = CM.stats(map);
  ok(map.surfels.length === 0 && map.voxels.size === 0 && st.total === 0 && !map.full && st.coveredPct === 0,
     "clear() empties surfels, voxel index, flags and stats");
  observeFace(map, STARE);
  ok(map.surfels.length === FACE_PTS.length, "and the map is reusable straight after");
}

console.log(`\n${fail === 0 ? "ALL " : ""}${pass} COVERAGE-MAP CHECKS PASSED${fail ? `, ${fail} FAILED` : ""}\n`);
process.exit(fail ? 1 : 0);
