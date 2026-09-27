#!/usr/bin/env node
/*
 * test_vr_input.js — VR locomotion maths (RH-7), no headset: the left stick walks where
 * the head looks, the dead zone holds a resting stick still, and snap turn fires once
 * per push. Run: node tests/test_vr_input.js
 */
"use strict";
const path = require("path");
const { pathToFileURL } = require("url");
let pass = 0, fail = 0;
function ok(cond, label, detail) {
  if (cond) { pass++; console.log("  ok   " + label); }
  else { fail++; console.log("  FAIL " + label + (detail ? "  -> " + detail : "")); }
}
const near = (a, b, e = 1e-6) => Math.abs(a - b) <= e;

(async () => {
  const vr = await import(pathToFileURL(path.resolve(__dirname, "..", "viewer/pc/scripts/vr_input.js")).href);
  // Head looking down -z (yaw 0): stick forward (y = -1) moves -z.
  let m = vr.stickToMove(0, -1, 0);
  ok(near(m.dx, 0) && near(m.dz, -1) && near(m.mag, 1), "forward stick, head north -> -z", JSON.stringify(m));
  // Head turned to look down -x (yaw +90 deg): forward moves -x.
  m = vr.stickToMove(0, -1, Math.PI / 2);
  ok(near(m.dx, -1) && near(m.dz, 0), "forward stick follows the head, not the body", JSON.stringify(m));
  // Strafe right with head north: +x.
  m = vr.stickToMove(1, 0, 0);
  ok(near(m.dx, 1) && near(m.dz, 0), "right stick strafes +x", JSON.stringify(m));
  ok(vr.stickToMove(0.1, 0.05, 0).mag === 0, "a resting stick inside the dead zone does not drift");
  ok(vr.stickToMove(0.3, 0, 0).mag < 0.3, "the dead zone is subtracted, so movement starts from zero");
  // Snap turn: one push, one turn; holding does not repeat; release re-arms.
  const st = { armed: true };
  const a = vr.snapTurn(st, 0.9), b = vr.snapTurn(st, 0.95), c = vr.snapTurn(st, 0.1), d = vr.snapTurn(st, -0.9);
  ok(near(a, -Math.PI / 6), "push right turns 30 deg right", a);
  ok(b === 0, "holding the stick does not keep turning", b);
  ok(c === 0 && near(d, Math.PI / 6), "released and pushed left turns 30 deg left", `${c} ${d}`);
  ok(near(vr.yawOf({ x: 0, z: -1 }), 0) && near(vr.yawOf({ x: -1, z: 0 }), Math.PI / 2), "head yaw from forward vector");
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
