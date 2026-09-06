#!/usr/bin/env node
/*
 * test_combat_nav.js — the walking and blind-spot half of combat mode, no browser.
 *
 * Bots only read as "waiting in a blind spot" if three geometry promises hold:
 * they path around obstructions instead of through them, they pick hold nodes
 * the player cannot see into but can still be shot from, and a missing navmesh
 * degrades loudly instead of freezing every bot in place. All of that is pure
 * math in viewer/pc/scripts/nav.js, so it is checked here rather than by eye.
 *
 * The fixture is a 20x20 m floor with a full-height wall on x=0 that has one
 * 1 m doorway at z=0. Like Recast, the wall is modelled as *absent* walkable
 * surface rather than as geometry the pathfinder has to remember.
 *
 * Run: node tests/test_combat_nav.js
 */
"use strict";
const path = require("path");
const { pathToFileURL } = require("url");

const ROOT = path.resolve(__dirname, "..");
let pass = 0, fail = 0;
function ok(cond, label, detail) {
  if (cond) { pass++; console.log("  ok   " + label); }
  else { fail++; console.log("  FAIL " + label + (detail ? "  -> " + detail : "")); }
}

const CELL = 1, HALF = 10;
const DOOR_Z = 1;                            // opening spans |z| <= 1
const wallAt = (z) => Math.abs(z) > DOOR_Z;  // blocked everywhere but the doorway

/** Cells covered by the wall (minus the doorway) are simply not walkable. */
function buildFloor({ sealed = false } = {}) {
  const verts = [], tris = [];
  let n = 0;
  for (let gz = 0; gz < HALF * 2; gz++) {
    for (let gx = 0; gx < HALF * 2; gx++) {
      const cx = -HALF + gx * CELL + CELL / 2, cz = -HALF + gz * CELL + CELL / 2;
      const isDoorCell = Math.abs(cz) <= 0.5 + 1e-6;
      const inWall = Math.abs(cx) <= 0.5 && (sealed || !isDoorCell);
      if (inWall) continue;
      const x = -HALF + gx * CELL, z = -HALF + gz * CELL;
      verts.push(x, 0, z, x + CELL, 0, z, x, 0, z + CELL, x + CELL, 0, z + CELL);
      tris.push(n, n + 1, n + 3, n, n + 3, n + 2);
      n += 4;
    }
  }
  return new Nav0(new Float32Array(verts), new Int32Array(tris));
}

let Nav0 = null;

/** Sight test against the same wall: a segment is blocked if it crosses it. */
function los(a, b) {
  const dx = b.x - a.x, dz = b.z - a.z;
  if (dx === 0 || (a.x < 0) === (b.x < 0)) return true;
  const t = -a.x / dx;
  if (t <= 0 || t >= 1) return true;
  const z = a.z + dz * t;
  return !wallAt(z);
}

/** What Combat.walkable does: stay on walkable surface and out of solid geometry. */
function makeCanWalk(nav) {
  return (a, b) => {
    if (!los({ x: a.x, z: a.z }, { x: b.x, z: b.z })) return false;
    const len = Math.hypot(b.x - a.x, b.z - a.z);
    const steps = Math.max(2, Math.ceil(len / 0.4));
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      if (nav.triAt(a.x + (b.x - a.x) * t, a.z + (b.z - a.z) * t, 0.2) < 0) return false;
    }
    return true;
  };
}

(async () => {
  ({ Nav: Nav0 } = await import(pathToFileURL(path.join(ROOT, "viewer", "pc", "scripts", "nav.js")).href));

  const nav = buildFloor();
  const canWalk = makeCanWalk(nav);

  console.log("\ngrid");
  ok(nav.triCount > 700 && nav.triCount < 800, "the wall footprint is carved out of the walkable surface", `${nav.triCount} tris`);
  ok(nav.triAt(0, -4, 0.2) < 0, "no floor inside the wall");
  ok(nav.triAt(0.5, -0.5, 0.2) >= 0, "floor remains in the doorway");
  ok(nav.heightAt(3.5, -2.5) === 0, "reads floor height anywhere on the grid");
  ok(nav.triAt(1e6, 1e6, 2) === -1, "finds nothing a million metres out");

  console.log("\npathing around occluders");
  const p = nav.findPath(-3, -4, 3, -4, { canWalk });
  ok(Array.isArray(p) && p.length >= 2, "a severed region is still reachable through the doorway");
  ok(p.some((w) => Math.abs(w.x) < 1 && Math.abs(w.z) < 1.5), "the route passes the doorway", JSON.stringify(p));
  ok(p.every((w, i) => i === 0 || canWalk(p[i - 1], w)), "every emitted hop stays on walkable ground");
  let walked = 0, prev = { x: -3, z: -4 };
  for (const w of [...p, { x: 3, z: -4 }]) { walked += Math.hypot(w.x - prev.x, w.z - prev.z); prev = w; }
  ok(walked >= 8.4 && walked < 12, "the pulled route stays near the 8.5 m geodesic through the door", `${walked.toFixed(2)} m`);
  ok(nav.findPath(-5, 3, -5, -3, { canWalk }) !== null, "same-side path needs no detour");
  const sealed = buildFloor({ sealed: true });
  ok(sealed.findPath(-3, -4, 3, -4, { canWalk: makeCanWalk(sealed) }) === null,
     "a sealed wall reports no path rather than an illegal one");
  ok(nav.findPath(-3, -4, 3, -4, { canWalk: () => false }) === null,
     "a route the mover cannot legally walk is refused, not faked");

  console.log("\nblind-spot scoring");
  const approach = [
    { x: -6, z: -2 }, { x: -4, z: -1 }, { x: -2, z: 0 },   // west corridor, heading for the door
    { x: 2, z: -4 }, { x: 5, z: -2 },                       // east side, past the wall
    { x: -6, z: 6 }, { x: 4, z: 6 },                        // the open north end
  ];
  const spots = nav.scoreSpots({ approach, los, minSpawnDist: 2, maxRange: 30, unseenRadius: 12 });
  ok(spots.length > 0, "scores candidate nodes", `${spots.length}`);
  const ambush = spots.filter((s) => s.ambush);
  ok(ambush.length > 0, "finds nodes hidden from the approach yet covering one", `${ambush.length}`);
  ok(ambush.every((s) => s.unseenNear >= 2 && s.seenFrom >= 1),
     "an ambush node is unseen from >=2 approaches and can still shoot at >=1");
  const corner = ambush.find((s) => s.x > 0.5 && s.z < -2);
  ok(!!corner, "the node behind the wall beside the doorway ranks as ambush",
     JSON.stringify(ambush.slice(0, 4).map((s) => ({ x: +s.x.toFixed(1), z: +s.z.toFixed(1) }))));
  const seenEverywhere = spots.find((s) => s.x < -5 && s.z < -5);
  ok(!!seenEverywhere && !seenEverywhere.ambush, "a spot out in the open west corridor is not an ambush node",
     JSON.stringify(seenEverywhere && { s: seenEverywhere.seenFrom, u: seenEverywhere.unseenNear }));
  ok(spots.some((s) => s.overwatch), "some nodes are rated overwatch");
  ok(spots.every((s) => s.nearestApproach >= 2), "nothing within the spawn buffer is offered");
  ok(spots.findIndex((s) => !s.ambush) === ambush.length, "every ambush node outranks every exposed node");
  ok(ambush.every((s, i) => i === 0 || ambush[i - 1].unseenNear >= s.unseenNear),
     "within ambush nodes the deepest blind spots come first");

  console.log("\nunreachable pockets");
  const pocket = nav.scoreSpots({ approach: [{ x: -6, z: -2 }], los, minSpawnDist: 2, maxRange: 30, unseenRadius: 12 });
  ok(pocket.every((s) => !s.ambush || s.seenFrom >= 1),
     "a node that cannot see any approach is never offered as an ambush post");

  console.log("\nfiltering and failure modes");
  const clipped = buildFloor().filter((x, y, z) => x < 0);
  ok(clipped.triCount < nav.triCount && clipped.triCount > 0, "filter() keeps only the accepted half", `${clipped.triCount}/${nav.triCount}`);
  ok(clipped.triAt(4, 0, 0.4) === -1, "filtered ground is no longer walkable");
  ok(clipped.triAt(-4, -4, 0.4) >= 0, "kept ground still is");
  ok(clipped.findPath(-5, -6, -5, 6, { canWalk: makeCanWalk(clipped) }) !== null,
     "adjacency is rebuilt, so paths still work after filtering");
  ok(clipped.removed > 300, "filter() reports how much it dropped", `${clipped.removed}`);

  const sloped = { nx: 12, nz: 12, cell: 1, ox: 0, oz: 0, data: new Float32Array(144) };
  for (let i = 0; i < 144; i++) sloped.data[i] = (i % 12) * 0.1;
  const hf = Nav0.fromHeightfield(sloped);
  ok(hf.triCount === 11 * 11 * 2, "the unbaked fallback triangulates the heightfield", `${hf.triCount}`);
  ok(hf.findPath(0.5, 0.5, 10.5, 10.5, { canWalk: makeCanWalk(hf) }) !== null,
     "the fallback graph is connected corner to corner");
  ok(Math.abs(hf.heightAt(5.5, 5.5) - 0.5) < 0.11, "the fallback carries the scan's floor slope", `${hf.heightAt(5.5, 5.5)}`);

  const empty = new Nav0(new Float32Array(0), new Int32Array(0));
  ok(empty.triCount === 0 && empty.triAt(0, 0, 5) === -1, "an empty navmesh reports no surface");
  ok(empty.heightAt(0, 0) === null, "and no height, rather than NaN");
  ok(empty.findPath(0, 0, 1, 1) === null, "and no path, rather than throwing");
  ok(Nav0.fromJSON({ verts: [0, 0, 0, 1, 0, 0, 0, 0, 1], tris: [0, 1, 2], build: {} }).triCount === 1,
     "parses a baked nav.json triangle list");

  console.log("");
  console.log(fail === 0 ? `ALL ${pass} COMBAT-NAV CHECKS PASSED` : `${pass} passed, ${fail} FAILED`);
  process.exit(fail ? 1 : 0);
})().catch((e) => {
  console.log("HARNESS ERROR: " + (e && e.stack ? e.stack : e));
  process.exit(1);
});
