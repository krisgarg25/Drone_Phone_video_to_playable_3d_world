#!/usr/bin/env node
"use strict";
const assert = require("node:assert/strict");
const { test, before } = require("node:test");
const { readFileSync } = require("node:fs");
const path = require("node:path");
let W = {};
before(async () => {
  // Import the unmodified browser ES module without changing repository package configuration.
  const source = readFileSync(path.join(__dirname, "../viewer/workspace_core.js"), "utf8");
  W = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
});
const parent = {}, origin = "http://localhost:3000";
const packet = (command, extra = {}) => ({ namespace: "groundcontrol", type: "command", command, ...extra });
const event = (data) => ({ source: parent, origin, data });
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} != ${expected}`);

// Removing either trust check would allow a sibling frame or foreign site to control the viewer.
test("commands require both the parent window and same origin", () => {
  assert.equal(typeof W.readCommand, "function", "workspace command validator is not implemented");
  assert.equal(W.readCommand({ ...event(packet("fit")), source: {} }, parent, origin), null);
  assert.equal(W.readCommand({ ...event(packet("fit")), origin: "https://evil.test" }, parent, origin), null);
  assert.equal(W.readCommand(event({ type: "command", command: "fit" }), parent, origin), null);
  assert.equal(W.readCommand(event({ namespace: "groundcontrol", type: "state" }), parent, origin), null);
  assert.equal(W.readCommand(event(null), parent, origin), null);
});

test("every contracted command is accepted with a normalized payload", () => {
  assert.equal(typeof W.readCommand, "function");
  const cases = [
    ["mode", { value: "orbit" }], ["mode", { value: "fly" }], ["mode", { value: "walk" }],
    ["fit", {}], ["frame", { index: 0 }], ["pick", { value: true }], ["pick", { value: false }],
    ["snapshot", {}], ["get-state", {}], ["clear-picks", {}],
    ...["splats", "cameras", "points", "coverage", "collider"].map(layer => ["layer", { layer, value: false }]),
  ];
  for (const [command, extra] of cases) {
    assert.deepEqual(W.readCommand(event(packet(command, extra)), parent, origin), { command, ...extra });
  }
});

test("pick tools carry a bounded renderer-side selection limit", () => {
  assert.deepEqual(W.readCommand(event(packet("pick", { value: true, limit: 2 })), parent, origin), { command: "pick", value: true, limit: 2 });
  for (const limit of [0, -1, 129, 1.5, "2", Infinity]) {
    assert.throws(() => W.readCommand(event(packet("pick", { value: true, limit })), parent, origin), /limit/i);
  }
});

test("malformed commands cannot coerce values, inject paths, or select fractional frames", () => {
  assert.equal(typeof W.readCommand, "function");
  const bad = [packet("eval", { value: "alert(1)" }), packet("mode", { value: "combat" }),
    packet("pick", { value: 1 }), packet("layer", { layer: "__proto__", value: true }),
    packet("layer", { layer: "splats", value: "false" }), packet("fit", { asset: "/other" }),
    packet("snapshot", { value: "anything" }), ...[-1, 0.5, Infinity, NaN, "2"].map(index => packet("frame", { index }))];
  for (const data of bad) assert.throws(() => W.readCommand(event(data), parent, origin), /command|frame|value|layer|field/i);
});

test("protocol gates commands while loading and replays ready after iframe load races", () => {
  assert.equal(typeof W.WorkspaceProtocol, "function");
  const sent = [], executed = [];
  const state = { mode: "orbit", layers: { splats: true, cameras: false, points: false, coverage: false, collider: false }, selectedFrame: -1 };
  const details = { capabilities: { cameras: false, points: false, coverage: false, collider: true }, cameraCount: 0 };
  const bridge = new W.WorkspaceProtocol({ parent, origin, send: data => sent.push(data), getState: () => state, getDetails: () => details, execute: cmd => executed.push(cmd) });
  bridge.receive(event(packet("pick", { value: true })));
  assert.equal(executed.length, 0);
  assert.equal(sent.at(-1).type, "error");
  bridge.receive(event(packet("get-state")));
  assert.equal(sent.at(-1).type, "state");
  assert.equal(sent.some(x => x.type === "ready"), false);
  bridge.markReady();
  bridge.receive(event(packet("get-state")));
  assert.deepEqual(sent.filter(x => x.type === "ready").at(-1), { namespace: "groundcontrol", type: "ready", mode: state.mode, layers: state.layers, ...details });
  assert.equal(sent.at(-1).selectedFrame, -1);
  bridge.receive(event(packet("mode", { value: "fly" })));
  assert.deepEqual(executed, [{ command: "mode", value: "fly" }]);
  assert.equal(sent.at(-1).type, "state");
});

test("protocol ignores untrusted input and reports command/load failures without escaping", () => {
  assert.equal(typeof W.WorkspaceProtocol, "function");
  const sent = [];
  const bridge = new W.WorkspaceProtocol({ parent, origin, send: data => sent.push(data), getState: () => ({ mode: "orbit", layers: {} }), getDetails: () => ({}), execute: () => { throw new Error("Frame unavailable"); } });
  bridge.receive({ ...event(packet("fit")), source: {} });
  assert.equal(sent.length, 0);
  bridge.markReady();
  bridge.receive(event(packet("frame", { index: 10 })));
  assert.equal(sent.at(-1).message, "Frame unavailable");
  bridge.fail("Scene failed");
  bridge.receive(event(packet("get-state")));
  assert.equal(sent.at(-1).message, "Scene failed");
  assert.equal(bridge.ready, false);
});

test("measurement render commands are accepted and bounded", () => {
  assert.deepEqual(W.readCommand(event(packet("measurements", { value: [{ id: "a", kind: "distance", points: [[0, 0, 0], [1, 1, 1]] }] })), parent, origin).value.length, 1);
  assert.equal(W.readCommand(event(packet("select", { id: "a" })), parent, origin).id, "a");
  assert.equal(W.readCommand(event(packet("select", { id: null })), parent, origin).id, null);
  assert.equal(W.readCommand(event(packet("labels", { value: false })), parent, origin).value, false);
  assert.throws(() => W.readCommand(event(packet("measurements", { value: "nope" })), parent, origin), /array|measure/i);
  assert.throws(() => W.readCommand(event(packet("measurements", { value: new Array(501).fill({}) })), parent, origin), /bounded|array/i);
  assert.throws(() => W.readCommand(event(packet("labels", { value: 1 })), parent, origin), /boolean/i);
});

test("embedded asset selection is confined to same-origin scene viewer assets", () => {
  assert.equal(typeof W.workspaceAssetPath, "function");
  assert.equal(W.workspaceAssetPath("/runtime/work/rocks/viewer_assets"), "/runtime/work/rocks/viewer_assets");
  assert.equal(W.workspaceAssetPath("/work/room_w_jsonl/viewer_assets/"), "/work/room_w_jsonl/viewer_assets");
  for (const path of ["//evil.test/a", "https://evil.test/a", "/runtime/work/../viewer_assets", "/runtime/work/a/../../secret", "/runtime/work/a%2f..%2fsecret/viewer_assets", "/other/file", "/runtime/work/a/viewer_assets?url=elsewhere"]) {
    assert.throws(() => W.workspaceAssetPath(path), /asset|path/i);
  }
});

test("layer commands are idempotent and cannot claim an unavailable diagnostic", () => {
  assert.equal(typeof W.layerState, "function");
  const state = { splats: true, cameras: false, points: false, coverage: false, collider: false };
  const caps = { cameras: true, points: false, coverage: false, collider: true };
  const once = W.layerState(state, "cameras", true, caps);
  assert.deepEqual(W.layerState(once, "cameras", true, caps), once);
  assert.equal(state.cameras, false);
  assert.throws(() => W.layerState(state, "points", true, caps), /unavailable/i);
  assert.equal(W.layerState(state, "points", false, caps).points, false);
});

test("bounds use finite observed points, without inventing geometry for missing data", () => {
  assert.equal(typeof W.boundsFromPoints, "function");
  assert.deepEqual(W.boundsFromPoints([[5, -2, 10], [-3, 4, 8], [NaN, 0, 0]]), { min: [-3, -2, 8], max: [5, 4, 10] });
  assert.equal(W.boundsFromPoints([]), null);
});

test("fit orbit encloses the scene sphere even in a narrow portrait frame", () => {
  assert.equal(typeof W.fitOrbit, "function");
  const bounds = { min: [-2, -1, -2], max: [2, 1, 2] }; // radius = 3
  const landscape = W.fitOrbit(bounds, 60, 2);
  const portrait = W.fitOrbit(bounds, 60, 0.5);
  assert.deepEqual(landscape.target, [0, 0, 0]);
  assert.ok(landscape.distance >= 6);
  assert.ok(portrait.distance > 10.8);
  assert.ok(portrait.distance > landscape.distance);
  const eye = W.orbitEye(landscape);
  near(Math.hypot(...eye), landscape.distance);
  assert.ok(eye[1] > 0, "overview looks down at actual bounds");
});

test("orbit rotates, pans in camera space and clamps zoom without crossing its target", () => {
  assert.equal(typeof W.rotateOrbit, "function");
  const state = { target: [0, 0, 0], distance: 10, yaw: 0, pitch: 0, minDistance: 0.1, maxDistance: 100 };
  assert.deepEqual(W.orbitEye(state), [0, 0, 10]);
  const rotated = W.rotateOrbit(state, Math.PI / 2, 0);
  near(W.orbitEye(rotated)[0], 10);
  near(W.orbitEye(rotated)[2], 0);
  const panned = W.panOrbit(state, 100, 50, 1000, 90);
  near(panned.target[0], -2);
  near(panned.target[1], 1);
  assert.equal(panned.distance, 10);
  assert.equal(W.zoomOrbit(state, -1e6).distance, 0.1);
  assert.equal(W.zoomOrbit(state, 1e6).distance, 100);
  assert.ok(Math.abs(W.rotateOrbit(state, 0, 100).pitch) < Math.PI / 2);
  assert.deepEqual(state.target, [0, 0, 0]);
});

test("adopting a selected source camera preserves its exact eye and forward direction", () => {
  assert.equal(typeof W.orbitFromPose, "function");
  const orbit = W.orbitFromPose([2, 3, 4], [0, 0, -5], 10);
  assert.deepEqual(orbit.target, [2, 3, -6]);
  W.orbitEye(orbit).forEach((v, i) => near(v, [2, 3, 4][i]));
  const vertical = W.orbitFromPose([1, 2, 3], [0, 1, 0], 4);
  W.orbitEye(vertical).forEach((v, i) => near(v, [1, 2, 3][i]));
  assert.throws(() => W.orbitFromPose([0, 0, 0], [0, 0, 0], 5), /pose|direction/i);
});

test("ray screen coordinates use canvas CSS pixels, offset and bounds (not device pixels)", () => {
  assert.equal(typeof W.canvasPoint, "function");
  const rect = { left: 50, top: 20, width: 400, height: 200 };
  assert.deepEqual(W.canvasPoint(250, 120, rect), [200, 100]);
  assert.equal(W.canvasPoint(49, 120, rect), null);
  assert.equal(W.canvasPoint(250, 221, rect), null);
  assert.equal(W.canvasPoint(250, 120, { ...rect, width: 0 }), null);
});

test("measurements accept only finite points on the static collision mesh, never splats or boxes", () => {
  assert.equal(typeof W.collisionSurfacePoint, "function");
  const mesh = { rigidbody: { type: "static" }, collision: { type: "mesh" } };
  const hit = { entity: mesh, point: { x: 1, y: 2, z: 3 } };
  assert.deepEqual(W.collisionSurfacePoint(hit, new Set([mesh])), [1, 2, 3]);
  assert.equal(W.collisionSurfacePoint(hit, new Set()), null);
  assert.equal(W.collisionSurfacePoint(null, new Set([mesh])), null);
  assert.equal(W.collisionSurfacePoint({ ...hit, point: { x: NaN, y: 2, z: 3 } }, new Set([mesh])), null);
  for (const entity of [{ gsplat: {} }, { rigidbody: { type: "static" }, collision: { type: "box" } }, { rigidbody: { type: "dynamic" }, collision: { type: "mesh" } }]) {
    assert.equal(W.collisionSurfacePoint({ ...hit, entity }, new Set([entity])), null);
  }
});

test("measurement history is bounded, copied, and rejects overflow rather than leaking memory", () => {
  assert.equal(typeof W.appendPick, "function");
  const original = [], point = [1, 2, 3];
  let points = W.appendPick(original, point, 2);
  point[0] = 99;
  assert.deepEqual(points, [[1, 2, 3]]);
  assert.deepEqual(original, []);
  points = W.appendPick(points, [4, 5, 6], 2);
  assert.throws(() => W.appendPick(points, [7, 8, 9], 2), /limit|clear/i);
  assert.throws(() => W.appendPick([], [0, NaN, 0]), /point/i);
});

test("snapshot waits for a rendered frame and bounds pending work", () => {
  assert.equal(typeof W.SnapshotCapture, "function");
  const sent = [];
  const shot = new W.SnapshotCapture((type, data) => sent.push({ type, ...data }));
  shot.request();
  assert.equal(sent.length, 0);
  assert.throws(() => shot.request(), /pending/i);
  shot.postrender(() => "data:image/png;base64,YWN0dWFsLWNhbnZhcy1ieXRlcw==");
  assert.deepEqual(sent, [{ type: "snapshot", dataUrl: "data:image/png;base64,YWN0dWFsLWNhbnZhcy1ieXRlcw==" }]);
  shot.postrender(() => { throw new Error("No request should capture again"); });
  assert.equal(sent.length, 1);
});

test("snapshot failures report errors and allow subsequent requests", () => {  assert.equal(typeof W.SnapshotCapture, "function");
  const sent = [];
  const shot = new W.SnapshotCapture((type, data) => sent.push({ type, ...data }));
  shot.request();
  shot.postrender(() => { throw new Error("Canvas unavailable"); });
  assert.equal(sent[0].type, "error");
  assert.match(sent[0].message, /Canvas unavailable/);
  shot.request();
  shot.postrender(() => "data:,");
  assert.equal(sent[1].type, "error");
});

test("walk distance counts only grounded travel, not falls, teleports or jitter", () => {  assert.equal(typeof W.walkDistance, "function");
  // Grounded, a real horizontal step counts its length.
  assert.ok(Math.abs(W.walkDistance([0, 0], [0.5, 0], true, 1.0) - 0.5) < 1e-9);
  // Airborne (falling / jumping) adds nothing.
  assert.equal(W.walkDistance([0, 0], [5, 0], false, 1.0), 0);
  // A frame that moved a whole body-length was a respawn teleport, not a step.
  assert.equal(W.walkDistance([0, 0], [2, 0], true, 1.0), 0);
  // Sub-millimetre physics jitter is noise, not distance.
  assert.equal(W.walkDistance([0, 0], [1e-6, 1e-6], true, 1.0), 0);
  // Vertical position is not part of horizontal walk distance.
  assert.ok(Math.abs(W.walkDistance([0, 0, 0], [0, 9, 0], true, 1.0)) < 1e-9);
});

test("measure renderable classifies point, line and closed polygon", () => {  assert.equal(W.measureRenderable({ kind: "point", points: [[1, 2, 3]] }).type, "point");
  assert.equal(W.measureRenderable({ kind: "distance", points: [[0, 0, 0], [1, 1, 1]] }).type, "line");
  const poly = W.measureRenderable({ kind: "area", points: [[0, 0, 0], [1, 0, 0], [1, 0, 1]] });
  assert.equal(poly.type, "polygon");
  assert.deepEqual(poly.points[0], poly.points[poly.points.length - 1]); // ring closed
});

test("measure anchor is the point, line midpoint or polygon centroid", () => {
  assert.deepEqual(W.measureAnchor({ kind: "point", points: [[1, 2, 3]] }), [1, 2, 3]);
  assert.deepEqual(W.measureAnchor({ kind: "distance", points: [[0, 0, 0], [2, 4, 6]] }), [1, 2, 3]);
  const c = W.measureAnchor({ kind: "area", points: [[0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2]] });
  assert.deepEqual(c, [1, 0, 1]);
});

test("measure label shows value, unit and uncertainty or honest no-support", () => {
  assert.match(W.formatMeasureLabel({ label: "Wall", kind: "distance", value: 4.02, unit: "m", uncertainty: { m: 0.05 }, valid: true }), /Wall · 4\.02m ±0\.05/);
  assert.match(W.formatMeasureLabel({ label: "X", kind: "distance", value: null, valid: false }), /no measured support/);
});

test("measure color is defined per kind and defaults safely", () => {
  assert.equal(W.measureColor("area").length, 3);
  assert.deepEqual(W.measureColor("area"), W.measureColor("area"));
  assert.equal(W.measureColor("bogus"), W.measureColor("point"));
});

test("placements command is accepted with a bounded array and rejects junk", () => {
  assert.equal(typeof W.readCommand, "function");
  const ok = W.readCommand(event(packet("placements", { value: [{ id: "a" }] })), parent, origin);
  assert.deepEqual(ok, { command: "placements", value: [{ id: "a" }] });
  assert.deepEqual(W.readCommand(event(packet("placements", { value: [] })), parent, origin), { command: "placements", value: [] });
  assert.throws(() => W.readCommand(event(packet("placements", { value: "nope" })), parent, origin));
  assert.throws(() => W.readCommand(event(packet("placements", { value: new Array(501).fill({}) })), parent, origin));
  assert.throws(() => W.readCommand(event(packet("placements", { value: [], extra: 1 })), parent, origin));
});

test("placementBox emits 8 corners matching the outline convention", () => {
  const p = { center_xz: [1, 2], center_y: 0.5, size: [2, 1, 1], yaw_deg: 0 };
  const box = W.placementBox(p);
  assert.ok(box, "a valid box must produce geometry");
  assert.equal(box.corners.length, 8);
  assert.deepEqual(box.halfExtents, [1, 0.5, 0.5]);
  // yaw 0: major axis along +x, so x spans center +/- half-length.
  const xs = box.corners.map(c => c[0]);
  near(Math.min(...xs), 0); near(Math.max(...xs), 2);
  // corners alternate bottom (y=0) / top (y=1).
  assert.equal(box.corners[0][1], 0); assert.equal(box.corners[1][1], 1);
});

test("placementBox rejects malformed records instead of throwing", () => {
  assert.equal(W.placementBox(null), null);
  assert.equal(W.placementBox({ center_xz: [0, 0] }), null);            // no size/center_y
  assert.equal(W.placementBox({ center_xz: [0, 0], center_y: 0, size: [0, 1, 1] }), null); // zero size
  assert.equal(W.placementBox({ center_xz: [NaN, 0], center_y: 0, size: [1, 1, 1] }), null);
});

test("placement color and label reflect the fit verdict honestly", () => {
  assert.deepEqual(W.placementColor({ fit: { valid: true, tight: false } }), [0.36, 0.82, 0.52]);
  // A wall-adjacent item still fits, so it stays green — not a warning.
  assert.deepEqual(W.placementColor({ fit: { valid: true, tight: true } }), [0.36, 0.82, 0.52]);
  assert.deepEqual(W.placementColor({ fit: { valid: false } }), [0.94, 0.36, 0.32]);
  assert.match(W.placementLabel({ label: "Sofa", size: [2, 0.85, 0.9], fit: { valid: true, tight: false } }), /Sofa · 2×0\.9 m · fits/);
  assert.match(W.placementLabel({ label: "Sofa", size: [2, 0.85, 0.9], fit: { valid: true, tight: true, floor_clearance_m: 0.08 } }), /fits · 0\.08 m from wall/);
  assert.match(W.placementLabel({ label: "Sofa", size: [2, 0.85, 0.9], fit: { valid: false, reason: "on a wall" } }), /on a wall/);
});

test("view and set-picks commands validate their payloads", () => {
  assert.deepEqual(W.readCommand(event(packet("view", { value: "top" })), parent, origin), { command: "view", value: "top" });
  assert.throws(() => W.readCommand(event(packet("view", { value: "sideways" })), parent, origin));
  assert.deepEqual(W.readCommand(event(packet("set-picks", { value: [[0, 0, 0], [1, 2, 3]] })), parent, origin), { command: "set-picks", value: [[0, 0, 0], [1, 2, 3]] });
  assert.throws(() => W.readCommand(event(packet("set-picks", { value: [[0, 0]] })), parent, origin));      // not a 3-point
  assert.throws(() => W.readCommand(event(packet("set-picks", { value: [[0, 0, NaN]] })), parent, origin)); // not finite
});
