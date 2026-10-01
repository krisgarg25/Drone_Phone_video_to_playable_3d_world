#!/usr/bin/env node
"use strict";
const assert = require("node:assert/strict");
const { test, before } = require("node:test");
const { readFileSync, existsSync } = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
let W = {}, M = {}, F = {}, PC = {};
// Resolve local ES modules without changing repository package configuration.
function moduleUrl(file) {
  const source = readFileSync(file, "utf8").replace(/from ["'](\.\/[^"']+)["']/g,
    (_, specifier) => `from "${moduleUrl(path.resolve(path.dirname(file), specifier))}"`);
  return `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
}
before(async () => {
  W = await import(moduleUrl(path.join(__dirname, "../viewer/workspace_core.js")));
  PC = await import(pathToFileURL(path.join(__dirname, "../viewer/pc/playcanvas.mjs")).href);
  for (const name of ["measurement_math", "furniture_geometry"]) {
    const file = path.join(__dirname, `../viewer/${name}.js`);
    if (existsSync(file)) {
      const mod = await import(moduleUrl(file));
      if (name === "measurement_math") M = mod; else F = mod;
    }
  }
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

test("measure labels preserve supported values and mark unsupported geometry as estimates", () => {
  assert.match(W.formatMeasureLabel({ label: "Wall", kind: "distance", value: 4.02, unit: "m", uncertainty: { m: 0.05 }, valid: true }), /Wall · 4\.02m ±0\.05/);
  const record = { label: "X", kind: "distance", value: null, valid: false, points: [[0, 0, 0], [3, 4, 0]], uncertainty: { m: null } };
  const original = JSON.stringify(record);
  assert.match(W.formatMeasureLabel(record), /~ 5 m · surface estimate/);
  assert.equal(JSON.stringify(record), original, "display must not promote validity or replace uncertainty");
  assert.equal(W.formatMeasureLabel({ kind: "point", label: "Door", value: null, valid: false }), "Door");
  assert.equal(W.formatMeasureLabel({ kind: "annotation", label: "Door note" }), "Door note");
  assert.doesNotMatch(W.formatMeasureLabel({ kind: "volume", valid: false, points: [[0, 0, 0], [2, 0, 0], [2, 0, 3]] }), /\d m³/);
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
  assert.match(W.placementLabel({ label: "Sofa", size: [2, 0.85, 0.9], fit: { valid: true, tight: true, floor_clearance_m: 0.08 } }), /fits · 0\.08 m to unscanned space/);
  assert.match(W.placementLabel({ label: "Sofa", size: [2, 0.85, 0.9], fit: { valid: false, reason: "on a wall" } }), /on a wall/);
  // Standing caveats belong in the panel; the on-model label carries only the verdict.
  assert.equal(W.placementLabel({ label: "Sofa", size: [2, 0.85, 0.9],
    fit: { valid: true, reason: "sampled floor support is suitable; ceiling clearance unknown; obstacle and furniture collisions are not checked" } }),
  "Sofa · 2×0.9 m · fits");
  assert.match(W.placementLabel({ label: "Sofa", size: [2, 0.85, 0.9],
    fit: { valid: false, reason: "only 28% of the footprint samples land on measured floor; ceiling clearance unknown" } }), /^Sofa · 2×0\.9 m · only 28% of the footprint samples land on measured floor$/);
});

test("view and set-picks commands validate their payloads", () => {
  assert.deepEqual(W.readCommand(event(packet("view", { value: "top" })), parent, origin), { command: "view", value: "top" });
  assert.throws(() => W.readCommand(event(packet("view", { value: "sideways" })), parent, origin));
  assert.deepEqual(W.readCommand(event(packet("set-picks", { value: [[0, 0, 0], [1, 2, 3]] })), parent, origin), { command: "set-picks", value: [[0, 0, 0], [1, 2, 3]] });
  assert.throws(() => W.readCommand(event(packet("set-picks", { value: [[0, 0]] })), parent, origin));      // not a 3-point
  assert.throws(() => W.readCommand(event(packet("set-picks", { value: [[0, 0, NaN]] })), parent, origin)); // not finite
});

test("preview math uses Y-up 3D polyline lengths, not footprint length", () => {
  assert.equal(typeof M.previewMeasurement, "function");
  const points = [[0, 0, 0], [3, 4, 0], [3, 4, 12]];
  const result = M.previewMeasurement("distance", points);
  assert.equal(result.value, 17);
  assert.equal(result.unit, "m");
  assert.deepEqual(result.segments, [5, 12]);
  assert.equal(result.horizontal, 15);
  assert.equal(result.vertical, 4);
  assert.equal(M.previewMeasurement("height", [[0, 7, 0], [90, 2, 40], [0, 20, 0]]).value, 5);
  assert.deepEqual(points, [[0, 0, 0], [3, 4, 0], [3, 4, 12]]);
});

test("preview area is XZ shoelace, closes once and preserves unscaled units", () => {
  assert.equal(typeof M.previewMeasurement, "function");
  const ring = [[0, 2, 0], [2, 2, 0], [2, 2, 3], [0, 2, 3]];
  const result = M.previewMeasurement("area", ring, "units");
  assert.equal(result.value, 6);
  assert.equal(result.unit, "units²");
  assert.equal(result.perimeter, 10);
  assert.deepEqual(result.segments, [2, 3, 2, 3]);
  assert.equal(M.previewMeasurement("area", [...ring].reverse()).value, 6);
  assert.deepEqual(M.previewMeasurement("area", [...ring, ring[0]]).segments, [2, 3, 2, 3]);
  assert.equal(M.previewMeasurement("area", [[0, 9, 0], [2, -6, 0], [2, 14, 3], [0, 1, 3]]).value, 6);
});

test("preview never invents a volume, annotation value, or incomplete result", () => {
  assert.equal(typeof M.previewMeasurement, "function");
  for (const kind of ["distance", "height", "area", "volume", "annotation", "point", null]) {
    assert.equal(M.previewMeasurement(kind, []).value, null);
    assert.equal(M.previewMeasurement(kind, [[0, 0, 0]]).value, null);
  }
  const volume = M.previewMeasurement("volume", [[0, 0, 0], [2, 4, 0], [2, 8, 3], [0, 2, 3]]);
  assert.equal(volume.value, null);
  assert.equal(volume.perimeter, 10);
  assert.equal(M.previewMeasurement("annotation", [[0, 0, 0], [2, 3, 4]]).value, null);
  assert.throws(() => M.previewMeasurement("distance", [[NaN, 0, 0]]), /point/i);
  assert.throws(() => M.previewMeasurement("distance", [], "cm"), /unit/i);
});

test("measurement-tool whitelist rejects coercion and extra fields", () => {
  for (const kind of ["point", "distance", "height", "area", "volume", null]) {
    for (const unit of ["m", "units"]) {
      assert.deepEqual(W.readCommand(event(packet("measurement-tool", { kind, unit })), parent, origin), { command: "measurement-tool", kind, unit });
    }
  }
  for (const payload of [{ kind: "annotation", unit: "m" }, { kind: "distance" }, { kind: 1, unit: "m" }, { kind: "area", unit: "cm" }, { kind: null, unit: "m", value: true }]) {
    assert.throws(() => W.readCommand(event(packet("measurement-tool", payload)), parent, origin));
  }
});

test("live measurement retains final totals when picking stops and closes area edges", () => {
  assert.equal(typeof W.liveMeasurement, "function");
  const fixed = [[0, 0, 0], [3, 4, 0]];
  const finished = W.liveMeasurement("distance", fixed, [99, 99, 99], false, "m");
  assert.deepEqual(finished.points, fixed);
  assert.equal(finished.preview.value, 5);
  assert.ok(finished.labels.some(l => /Total · 5 m/.test(l.text)));
  const moving = W.liveMeasurement("distance", [[0, 0, 0]], [3, 4, 0], true, "m");
  assert.equal(moving.preview.value, 5);
  // One segment would repeat the total beside it, so only the total is labelled.
  assert.deepEqual(moving.labels.map(l => l.role), ["total"]);
  const area = W.liveMeasurement("area", [[0, 0, 0], [2, 0, 0]], [2, 0, 3], true, "m");
  const geometry = W.measureRenderable(area);
  assert.deepEqual(geometry.points.at(-1), [0, 0, 0]);
  assert.equal(area.labels.filter(l => l.role === "segment").length, 3);
  assert.equal(W.liveMeasurement(null, fixed, null, false, "m"), null);
});

test("preview stream coalesces pointer motion to 15Hz and sends clears instead of stale points", () => {
  assert.equal(typeof W.PreviewStream, "function");
  const sent = [], stream = new W.PreviewStream((type, data) => sent.push({ type, ...data }));
  stream.update([0, 0, 0]); stream.flush(0);
  stream.update([1, 0, 0]); stream.flush(20);
  stream.update([2, 0, 0]); stream.flush(40);
  assert.equal(sent.length, 1);
  stream.flush(67);
  assert.deepEqual(sent[1], { type: "preview", point: [2, 0, 0] });
  stream.update([3, 0, 0]); stream.update(null); stream.flush(80);
  stream.flush(134);
  assert.deepEqual(sent[2], { type: "preview", point: null });
  stream.flush(200);
  assert.equal(sent.length, 3);
});

test("projected labels use actual independent drawing-buffer ratios and reject offscreen points", () => {
  assert.equal(typeof W.projectedCssPoint, "function");
  const rect = { width: 600, height: 300 };
  assert.deepEqual(W.projectedCssPoint({ x: 450, y: 300, z: 4 }, rect, 900, 600), [300, 150]);
  for (const s of [{ x: -1, y: 0, z: 4 }, { x: 901, y: 0, z: 4 }, { x: 0, y: 601, z: 4 }, { x: 0, y: 0, z: -1 }, { x: NaN, y: 0, z: 1 }]) {
    assert.equal(W.projectedCssPoint(s, rect, 900, 600), null);
  }
});

test("label decluttering gives live/selected labels priority and keeps rectangles inside canvas", () => {
  assert.equal(typeof W.layoutLabels, "function");
  const labels = [
    { id: "saved", x: 70, y: 60, width: 100, height: 22, priority: 1 },
    { id: "live", x: 70, y: 60, width: 100, height: 22, priority: 100 },
    { id: "offscreen", x: -100, y: 20, width: 100, height: 22, priority: 100 },
  ];
  const arranged = W.layoutLabels(labels, 140, 110);
  assert.equal(arranged[0].id, "live");
  assert.ok(!arranged.some(l => l.id === "offscreen"));
  for (const a of arranged) {
    assert.ok(a.left >= 0 && a.top >= 0 && a.left + a.width <= 140 && a.top + a.height <= 110);
    for (const b of arranged) if (a !== b) assert.ok(a.left + a.width <= b.left || b.left + b.width <= a.left || a.top + a.height <= b.top || b.top + b.height <= a.top);
  }
});

const placed = (item = "coffee_table", yaw_deg = 0) => ({ id: "table", item, center_xz: [4, 7], center_y: 0.225, size: [1.1, 0.45, 0.6], yaw_deg });
test("coffee table is a tabletop and four separate legs, not a solid bounding box", () => {
  assert.equal(typeof F.furnitureParts, "function");
  const parts = F.furnitureParts("coffee_table");
  assert.equal(parts.filter(p => p.name === "top").length, 1);
  assert.equal(parts.filter(p => p.name.startsWith("leg")).length, 4);
  assert.ok(!parts.some(p => p.min[0] < 0 && p.max[0] > 0 && p.min[2] < 0 && p.max[2] > 0 && p.min[1] < 0.4), "space underneath the table stays open");
  assert.notDeepEqual(parts, F.furnitureParts("dining_chair"));
});

test("all twelve furniture kinds have recognizable multi-part geometry within normalized bounds", () => {
  assert.equal(typeof F.furnitureParts, "function");
  const kinds = ["sofa", "armchair", "coffee_table", "dining_table", "dining_chair", "bed_double", "desk", "wardrobe", "bookshelf", "side_table", "tv_stand", "stool"];
  for (const kind of kinds) {
    const parts = F.furnitureParts(kind);
    assert.ok(parts.length >= 4, `${kind} needs components`);
    for (const p of parts) for (let axis = 0; axis < 3; axis++) {
      assert.ok(p.min[axis] >= (axis === 1 ? 0 : -0.5));
      assert.ok(p.max[axis] <= (axis === 1 ? 1 : 0.5));
      assert.ok(p.min[axis] < p.max[axis]);
    }
  }
  for (const [kind, names] of [["sofa", ["arm", "back", "cushion"]], ["bed_double", ["mattress", "pillow"]], ["dining_chair", ["seat", "back", "leg"]], ["bookshelf", ["shelf"]]]) {
    for (const name of names) assert.ok(F.furnitureParts(kind).some(p => p.name.startsWith(name)), `${kind}: ${name}`);
  }
});

test("furniture mesh respects placement bounds, yaw and opaque byte colors with directional shading", () => {
  assert.equal(typeof F.furnitureGeometry, "function");
  for (const yaw of [0, 37, 90, -45]) {
    const p = placed("coffee_table", yaw), geometry = F.furnitureGeometry(p);
    assert.ok(geometry.positions.length > 24 * 3);
    const angle = yaw * Math.PI / 180;
    for (let i = 0; i < geometry.positions.length; i += 3) {
      const [x, y, z] = geometry.positions.slice(i, i + 3);
      const dx = x - 4, dz = z - 7;
      assert.ok(Math.abs(dx * Math.cos(angle) + dz * Math.sin(angle)) <= 0.55 + 1e-8);
      assert.ok(Math.abs(-dx * Math.sin(angle) + dz * Math.cos(angle)) <= 0.3 + 1e-8);
      assert.ok(y >= -1e-8 && y <= 0.45 + 1e-8);
    }
    assert.equal(geometry.colors.length, geometry.positions.length / 3 * 4);
    assert.ok(geometry.colors instanceof Uint8Array);
    geometry.colors.forEach((v, i) => { assert.ok(Number.isFinite(v)); if (i % 4 === 3) assert.equal(v, 255); });
    assert.ok(new Set(Array.from(geometry.colors).filter((_, i) => i % 4 !== 3)).size > 6, "faces and components have distinct shaded colors");
    assert.equal(geometry.normals.length, geometry.positions.length);
    assert.ok(geometry.indices.every(i => i >= 0 && i < geometry.positions.length / 3));
  }
  const zero = F.furnitureGeometry(placed()), turned = F.furnitureGeometry(placed("coffee_table", 90));
  for (let i = 0; i < zero.positions.length; i += 3) {
    near(turned.positions[i] - 4, -(zero.positions[i + 2] - 7));
    near(turned.positions[i + 2] - 7, zero.positions[i] - 4);
  }
  assert.equal(F.furnitureGeometry({ item: "coffee_table" }), null);
});

test("placement signature includes item, fit and stale without depending on selection", () => {
  assert.equal(typeof W.placementSignature, "function");
  const p = placed(), base = W.placementSignature([p]);
  for (const change of [{ item: "dining_table" }, { fit: { valid: false } }, { stale: true }, { yaw_deg: 90 }, { center_y: 0.5 }]) {
    assert.notEqual(W.placementSignature([{ ...p, ...change }]), base);
  }
  assert.equal(W.placementSignature([{ ...p, selected: true }]), base);
});

test("the edit command is strict about its payload", () => {
  assert.deepEqual(W.readCommand(event(packet("edit", { value: true })), parent, origin), { command: "edit", value: true });
  assert.deepEqual(W.readCommand(event(packet("edit", { value: false })), parent, origin), { command: "edit", value: false });
  for (const payload of [{}, { value: "true" }, { value: 1 }, { value: true, id: "table" }]) {
    assert.throws(() => W.readCommand(event(packet("edit", payload)), parent, origin));
  }
});

test("a pixel ray selects the nearest placed box, not the shell behind it", () => {
  const table = placed();                                   // centre [4, 0.225, 7], 1.1 x 0.45 x 0.6
  const far = { ...table, id: "far", center_xz: [4, 40] };
  // Straight down the -Z axis from behind both boxes: the near one wins.
  assert.equal(W.placementRay([table, far], [4, 0.225, -5], [4, 0.225, 50]), "table");
  assert.equal(W.placementRay([far, table], [4, 0.225, -5], [4, 0.225, 50]), "table", "order must not matter");
  // A ray that passes beside the box hits nothing, and one that stops short either.
  assert.equal(W.placementRay([table], [12, 0.225, -5], [12, 0.225, 50]), null);
  assert.equal(W.placementRay([table], [4, 0.225, 20], [4, 0.225, 50]), null, "the segment must reach the box");
  assert.equal(W.placementRay([table], [4, 3, -5], [4, 3, 50]), null, "above the top face");
  // Yaw is undone before the slab test, so a turned table is hit on its long side.
  const turned = placed("coffee_table", 90);
  assert.equal(W.placementRay([turned], [4, 0.225, -5], [4, 0.225, 50]), "table");
  assert.equal(W.placementRay([turned], [4.45, 0.225, -5], [4.45, 0.225, 50]), null);
  assert.equal(W.placementRay([table], [4, 0.225, -5], [4, 0.225, -4]), null, "a degenerate segment");
  assert.equal(W.placementRay([table], [NaN, 0, 0], [0, 0, 1]), null);
});

test("importedModelFit maps a model's natural AABB onto its placement box exactly", () => {
  assert.equal(typeof W.importedModelFit, "function");
  const box = { center: [4, 0.5, 7], halfExtents: [1.2, 0.45, 0.55], yaw: 30 };
  // A unit cube centred at the origin fills the box with no offset and scale = 2*half.
  const centred = W.importedModelFit({ min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] }, box);
  centred.scale.forEach((v, i) => near(v, box.halfExtents[i] * 2));
  centred.offset.forEach(v => near(v, 0));
  // An off-centre model is translated so its centre lands on the box centre (offset = -c*s).
  const off = { min: [1, 0, 2], max: [2, 1, 3] };
  const fit = W.importedModelFit(off, box);
  const center = [1.5, 0.5, 2.5];
  fit.offset.forEach((v, i) => near(v, -center[i] * fit.scale[i]));
  // Resizing the box re-scales the model per axis; the model's own extent is unchanged.
  const resized = W.importedModelFit(off, { ...box, halfExtents: [2, 2, 2] });
  near(resized.scale[0], 4); near(resized.scale[1], 4); near(resized.scale[2], 4);
});

test("importedModelFit refuses degenerate or malformed inputs instead of dividing by zero", () => {
  assert.equal(typeof W.importedModelFit, "function");
  const box = { center: [0, 0, 0], halfExtents: [1, 1, 1], yaw: 0 };
  assert.equal(W.importedModelFit(null, box), null);
  assert.equal(W.importedModelFit({ min: [0, 0, 0], max: [1, 1, 1] }, null), null);
  assert.equal(W.importedModelFit({ min: [0, 0, 0], max: [1, 1, 1] }, { halfExtents: [NaN, 1, 1] }), null);
  assert.equal(W.importedModelFit({ min: [0, 0, 0], max: [1, 0, 1] }, box), null,   // zero-size axis
    "a flat model would need an infinite scale on that axis");
  assert.equal(W.importedModelFit({ min: [NaN, 0, 0], max: [1, 1, 1] }, box), null);
});

test("focus-placement validates IDs and fits only the requested rotated item", () => {
  assert.equal(typeof W.focusPlacement, "function");
  assert.deepEqual(W.readCommand(event(packet("focus-placement", { id: "table" })), parent, origin), { command: "focus-placement", id: "table" });
  for (const payload of [{ id: null }, { id: 2 }, { id: "" }, { id: "table", mode: "walk" }]) {
    assert.throws(() => W.readCommand(event(packet("focus-placement", payload)), parent, origin));
  }
  const p = placed("coffee_table", 90);
  const orbit = W.focusPlacement([p, { ...placed(), id: "far", center_xz: [300, 700] }], "table", 70, 0.5);
  assert.deepEqual(orbit.target, [4, 0.225, 7]);
  assert.ok(orbit.distance > 1 && orbit.distance < 5);
  assert.throws(() => W.focusPlacement([p], "missing", 70, 1), /placement|unavailable/i);
});

// Execute the actual pc.js workspace/DOM wiring without loading scene assets or
// Ammo. Meshes, vertex buffers, materials and camera projection use the shipped
// PlayCanvas engine; only DOM dimensions, physics hits and entity components are
// stand-ins. This catches integrations that helper-only tests cannot exercise.
function viewerHarness(t) {
  const source = readFileSync(path.join(__dirname, "../viewer/pc.js"), "utf8");
  const section = (start, end) => {
    const from = source.indexOf(start), to = source.indexOf(end, from);
    assert.ok(from >= 0 && to > from, `Missing viewer integration section: ${start}`);
    return source.slice(from, to);
  };
  const eventTarget = () => ({
    listeners: new Map(),
    addEventListener(type, fn) { this.listeners.set(type, [...(this.listeners.get(type) || []), fn]); },
    fire(type, data = {}) { for (const fn of this.listeners.get(type) || []) fn({ preventDefault() {}, ...data }); },
  });
  const domElement = () => ({
    style: {}, dataset: {}, textContent: "", children: [],
    appendChild(el) { el.parent = this; this.children.push(el); },
    remove() { if (this.parent) this.parent.children = this.parent.children.filter(el => el !== this); },
    get offsetWidth() { return Math.min(this.textContent.length * 7 + 16, parseFloat(this.style.maxWidth) || Infinity); },
    get offsetHeight() { return 24; },
  });
  const canvas = Object.assign(eventTarget(), {
    width: 900, height: 600, clientWidth: 600, clientHeight: 300,
    getBoundingClientRect: () => ({ left: 50, top: 20, width: 600, height: 300 }),
    focus() {}, setPointerCapture(id) { this.captured = id; },
    hasPointerCapture(id) { return this.captured === id; }, releasePointerCapture() { this.captured = null; },
  });
  const document = Object.assign(eventTarget(), { body: domElement(), createElement: domElement });
  const window = Object.assign(eventTarget(), { __ready: true });
  class Entity extends PC.GraphNode {
    addComponent(type, options) {
      this[type] = { ...options };
      if (type === "render") {
        for (const mi of this.render.meshInstances) mi.node = this;
        this.render.destroyMeshInstances = () => {
          for (const mi of this.render.meshInstances) mi.destroy();
          this.render.meshInstances = [];
        };
      }
    }
    destroy() {
      this.render?.destroyMeshInstances();
      for (const child of [...this.children]) child.destroy();
      this.parent?.removeChild(this);
      this.destroyed = true;
    }
  }
  const device = new PC.NullGraphicsDevice(canvas), cameraEnt = new Entity("camera");
  const engineCamera = new PC.Camera(); engineCamera.node = cameraEnt;
  engineCamera.aspectRatioMode = PC.ASPECT_MANUAL;
  engineCamera.aspectRatio = 2; engineCamera.fov = 70;
  cameraEnt.setPosition(0, 0, 12); cameraEnt.lookAt(new PC.Vec3());
  cameraEnt.camera = {
    camera: engineCamera,
    screenToWorld: (x, y, z) => engineCamera.screenToWorld(x, y, z, canvas.clientWidth, canvas.clientHeight),
  };
  for (const key of ["fov", "nearClip", "farClip"]) Object.defineProperty(cameraEnt.camera, key, {
    get: () => engineCamera[key], set: value => { engineCamera[key] = value; },
  });
  const surface = { rigidbody: { type: "static" }, collision: { type: "mesh" } }, sent = [], draws = [], callbacks = {};
  let hit = null, clock = 0, api;
  const application = {
    graphicsDevice: device, root: new Entity("root"), on: (type, fn) => { callbacks[type] = fn; },
    drawLines: points => draws.push(points.map(p => [p.x, p.y, p.z])),
    systems: { rigidbody: { raycastFirst(start, end, options) {
      assert.equal(options.filterCollisionMask, PC.BODYMASK_STATIC);
      if (hit instanceof Error) throw hit;
      return hit && options.filterCallback(hit.entity) ? hit : null;
    } } },
  };
  const workspace = new W.WorkspaceProtocol({ parent, origin, send: data => sent.push(data),
    getState: () => ({ mode: api?.state().mode, layers: {} }), getDetails: () => ({}), execute: cmd => api.command(cmd) });
  const env = { ...W, ...F, Entity, Vec3: PC.Vec3, Color: PC.Color, Mesh: PC.Mesh, MeshInstance: PC.MeshInstance,
    StandardMaterial: PC.StandardMaterial, BLEND_NONE: PC.BLEND_NONE, CULLFACE_BACK: PC.CULLFACE_BACK,
    PRIMITIVE_TRIANGLES: PC.PRIMITIVE_TRIANGLES, BODYMASK_STATIC: PC.BODYMASK_STATIC,
    EMBED: true, document, window, canvas, device, application, cameraEnt, workspace,
    performance: { now: () => clock }, console: { log() {} },
    workspaceMode: "orbit", workspaceOrbit: W.fitOrbit({ min: [-2, -2, -2], max: [2, 2, 2] }, 70, 2),
    workspaceCommand: null, workspacePick: false, workspacePickLimit: 128, editDrag: null, workspaceLookDirty: false,
    workspaceCapabilities: () => ({ collider: true }), collisionMeshEntities: new Set([surface]),
    selectedCamIdx: 3, selectedCamEntity: { enabled: true }, clearActiveKeys() {},
    CHAR_SCALE: 1, CHAR_H: 1.75, dronePos: new PC.Vec3(), P: {}, autopilot: {}, isDrone: true,
    playerRb: { type: "kinematic" }, playerEnt: { enabled: false }, frames: 3, renderedSplat: true,
  };
  const code = section("  // ---------------- embedded workspace", "  // Camera Inspector events") +
    section('  application.on("postrender"', '  application.systems.rigidbody.gravity') +
    section("  // ---------------- input handlers", "    return; // No legacy pointer lock") + "\n  }\n" + `
    return { command: cmd => workspaceCommand(cmd), draw: drawMeasurementGeometry,
      state: () => ({ mode: workspaceMode, orbit: workspaceOrbit, picking: workspacePick, picks: pickPoints,
        preview: previewPoint, live: currentLiveMeasurement(), selected: selectedMeasureId, editing: workspaceEdit,
        placements: renderedPlacements, root: placedRoot, items: placedItems, labels: labelEls }),
      dispose: () => { setRenderedPlacements([]); } };
  `;
  api = new Function(...Object.keys(env), code)(...Object.values(env));
  workspace.markReady(); sent.length = 0;
  t.after(() => { api.dispose(); device.destroy(); });
  return { ...api, sent, draws, canvas, cameraEnt, device, document, window,
    command: (cmd, extra) => workspace.receive(event(packet(cmd, extra))),
    hit(point, entity = surface) { hit = point instanceof Error ? point : point && { entity, point: new PC.Vec3(...point) }; },
    screenOf(point) {
      const cam = cameraEnt.camera.camera, dev = device, rect = canvas.getBoundingClientRect();
      const s = cam.worldToScreen(new PC.Vec3(...point), dev.width, dev.height, new PC.Vec3());
      return { clientX: rect.left + s.x * rect.width / dev.width, clientY: rect.top + s.y * rect.height / dev.height, z: s.z };
    },
    pointer(type, extra = {}) { canvas.fire(type, { pointerId: 1, clientX: 350, clientY: 170, button: 0, ...extra }); },
    click(point) { this.hit(point); this.pointer("pointerdown"); this.pointer("pointerup"); },
    frame(now = clock + 67) { clock = now; callbacks.postrender(); },
    visibleLabels() { return [...api.state().labels.values()].filter(el => el.style.visibility === "visible"); },
  };
}

test("pc.js command and pointer wiring renders live totals before save and retains the final pick", t => {
  const h = viewerHarness(t);
  h.command("measurement-tool", { kind: "height", unit: "units" });
  h.command("pick", { value: true, limit: 2 });
  h.click([0, 0, 0]);
  h.hit([3, 4, 0]); h.pointer("pointermove"); h.frame(0);
  assert.equal(h.state().live.preview.value, 4);
  assert.ok(h.visibleLabels().some(el => /Height · 4 units/.test(el.textContent)));
  assert.ok(h.visibleLabels().some(el => el.textContent === "5 units"));
  assert.deepEqual(h.sent.filter(e => e.type === "preview").at(-1).point, [3, 4, 0]);
  h.click([3, 4, 0]); h.frame();
  assert.equal(h.state().picking, false);
  assert.equal(h.state().preview, null);
  h.hit([99, 99, 99]); h.pointer("pointermove"); h.frame();
  assert.deepEqual(h.state().live.points, [[0, 0, 0], [3, 4, 0]]);
  assert.ok(h.visibleLabels().some(el => /Height · 4 units/.test(el.textContent)));
  h.command("pick", { value: false }); h.frame();
  assert.equal(h.state().live.preview.value, 4, "stop must not discard the unsaved total");
  h.command("measurement-tool", { kind: null, unit: "m" }); h.frame();
  assert.equal(h.visibleLabels().length, 0);
});

test("pc.js previews coalesce motion and flush miss/leave/error/stop clears without another move", t => {
  const h = viewerHarness(t);
  h.command("measurement-tool", { kind: "distance", unit: "m" });
  h.command("pick", { value: true }); h.click([0, 0, 0]);
  h.hit([1, 0, 0]); h.pointer("pointermove"); h.frame(0);
  h.hit([2, 0, 0]); h.pointer("pointermove"); h.frame(10);
  h.hit([3, 0, 0]); h.pointer("pointermove"); h.frame(20);
  assert.equal(h.sent.filter(e => e.type === "preview").length, 1);
  h.frame(67);
  assert.deepEqual(h.sent.filter(e => e.type === "preview").at(-1).point, [3, 0, 0]);
  for (const clear of [() => { h.hit(null); h.pointer("pointermove"); },
    () => h.pointer("pointerleave"), () => { h.hit(new Error("ray failure")); h.pointer("pointermove"); },
    () => h.pointer("pointercancel"), () => h.window.fire("blur"), () => h.command("pick", { value: false })]) {
    h.hit([3, 4, 0]); h.pointer("pointermove"); h.frame();
    clear(); h.frame();
    assert.equal(h.state().preview, null);
    assert.equal(h.sent.filter(e => e.type === "preview").at(-1).point, null);
    assert.equal(h.state().live.points.length, 1);
  }
});

test("pc.js undo reopens a limited tool and live area draws its closing edge", t => {
  const h = viewerHarness(t);
  h.command("measurement-tool", { kind: "height", unit: "m" });
  h.command("pick", { value: true, limit: 2 }); h.click([0, 0, 0]); h.click([0, 2, 0]);
  h.command("set-picks", { value: [[0, 0, 0]] });
  assert.equal(h.state().picking, true);
  h.click([0, 4, 0]); assert.equal(h.state().live.preview.value, 4);
  h.command("measurement-tool", { kind: "area", unit: "m" });
  h.command("clear-picks"); h.command("pick", { value: true });
  h.click([0, 0, 0]); h.click([2, 0, 0]); h.hit([2, 0, 3]); h.pointer("pointermove");
  h.draw();
  assert.equal(h.state().live.preview.value, 3);
  assert.deepEqual(h.draws.at(-1).slice(-2), [[2, 0, 3], [0, 0, 0]]);
  h.hit([1, 2, 3], { rigidbody: { type: "static" }, collision: { type: "box" } });
  h.pointer("pointermove"); assert.equal(h.state().preview, null, "furniture is not a measured surface");
});

test("pc.js labels use engine projection, prioritize live totals, and show only selected furniture", t => {
  const h = viewerHarness(t);
  h.command("placements", { value: [placed(), { ...placed(), id: "other" }] }); h.frame();
  assert.equal(h.state().labels.size, 0);
  h.command("select", { id: "table" }); h.frame();
  assert.deepEqual([...h.state().labels.keys()], ["placement:table"]);
  h.command("select", { id: null });
  const measurements = Array.from({ length: 45 }, (_, i) => ({ id: String(i), kind: "distance", label: "Saved", points: [[0, 0, 0], [2, 0, 0]], valid: false }));
  h.command("measurements", { value: measurements });
  h.command("measurement-tool", { kind: "distance", unit: "m" });
  h.command("set-picks", { value: [[0, 0, 0], [2, 0, 0]] }); h.frame();
  const labels = h.visibleLabels(), total = h.state().labels.get("live:total");
  assert.equal(total.style.visibility, "visible");
  assert.ok(labels.length <= 32);
  const s = h.cameraEnt.camera.camera.worldToScreen(new PC.Vec3(1, 0, 0), 900, 600);
  near(parseFloat(total.style.left) + total.offsetWidth / 2, s.x * 600 / 900);
  near(parseFloat(total.style.top) + total.offsetHeight + 12, s.y * 300 / 600);
  for (const a of labels) for (const b of labels) if (a !== b) {
    const ax = parseFloat(a.style.left), ay = parseFloat(a.style.top), bx = parseFloat(b.style.left), by = parseFloat(b.style.top);
    assert.ok(ax + a.offsetWidth <= bx || bx + b.offsetWidth <= ax || ay + a.offsetHeight <= by || by + b.offsetHeight <= ay);
  }
  assert.ok(labels.some(el => /~ 2 m · surface estimate/.test(el.textContent)));
  h.command("labels", { value: false }); h.frame(); assert.equal(h.state().labels.size, 0);
  h.command("labels", { value: true });
  h.command("clear-picks");
  h.command("measurements", { value: [{ kind: "point", points: [[0, 0, 20]] }, { kind: "point", points: [[999, 0, 0]] }] });
  h.frame(); assert.equal(h.state().labels.size, 0, "behind-camera/offscreen labels are removed");
});

test("pc.js builds opaque byte-colored furniture per item and releases a replaced mesh", t => {
  const h = viewerHarness(t), p = placed("coffee_table", 37);
  h.command("placements", { value: [p] });
  const item = h.state().items.get("table"), mi = item.holder.render.meshInstances[0], oldMesh = mi.mesh;
  const positions = [], colors = [], normals = [];
  oldMesh.getPositions(positions); oldMesh.getColors(colors); oldMesh.getNormals(normals);
  assert.ok(positions.length > 24 * 3 && positions.every(Number.isFinite));
  assert.equal(normals.length, positions.length);
  assert.equal(colors.length, positions.length / 3 * 4);
  colors.forEach((v, i) => { if (i % 4 === 3) assert.equal(v, 255); });
  assert.ok(new Set(colors).size > 6);
  const colorFormat = oldMesh.vertexBuffer.format.elements.find(e => e.name === PC.SEMANTIC_COLOR);
  assert.equal(colorFormat.dataType, PC.TYPE_UINT8); assert.equal(colorFormat.numComponents, 4); assert.equal(colorFormat.normalize, true);
  assert.equal(mi.material.blendType, PC.BLEND_NONE); assert.equal(mi.material.opacity, 1);
  assert.equal(mi.material.depthWrite, true); assert.equal(mi.material.depthTest, true);
  assert.equal(mi.material.emissiveMapVertexColor, true); assert.ok(mi.material.emissive.equals(PC.Color.WHITE));
  const root = h.state().root;
  assert.equal(root.rigidbody.type, "static"); assert.equal(root.collision.type, "compound");
  const collider = item.collider;
  assert.ok(collider.collision.halfExtents.equals(new PC.Vec3(0.55, 0.225, 0.3)));
  near(collider.getLocalEulerAngles().y, -37);
  // A poll that changes nothing the model shows must not rebuild the mesh.
  h.command("placements", { value: [{ ...p, label: "Renamed" }] }); h.command("select", { id: p.id });
  assert.equal(h.state().items.get("table").mesh, oldMesh, "labels and selection reuse the mesh");
  // A different piece replaces only its own geometry, and the old buffers go free.
  h.command("placements", { value: [{ ...p, item: "sofa", fit: { valid: false } }] });
  assert.equal(oldMesh.vertexBuffer, null, "a replaced mesh releases its buffers");
  assert.equal(h.state().items.get("table").holder, item.holder, "the item entity is reused");
  const next = h.state().items.get("table").mesh;
  // Removing one item leaves the others standing.
  h.command("placements", { value: [{ ...p, id: "other" }] });
  assert.equal(next.vertexBuffer, null);
  assert.deepEqual([...h.state().items.keys()], ["other"], "only the removed item is destroyed");
  h.command("placements", { value: [] });
  assert.equal(h.state().items.size, 0);
  h.command("placements", { value: [{ ...p, center_y: NaN }, { ...p, size: [1e308, 1e308, 1e308] }] });
  assert.equal(h.state().items.size, 0, "unrepresentable GPU/physics bounds must not be installed");
});

test("pc.js edit mode grabs a piece, drags it live and hands the host one final point", t => {
  const h = viewerHarness(t), p = placed();
  h.command("placements", { value: [p] });
  // Arming the editor releases any measurement tool; arming a tool releases the editor.
  h.command("pick", { value: true, limit: 2 });
  h.command("edit", { value: true });
  assert.equal(h.state().picking, false);
  assert.equal(h.state().editing, true);
  h.command("pick", { value: true, limit: 2 });
  assert.equal(h.state().editing, false, "one tool at a time");
  h.command("edit", { value: true });
  // A press and release without movement selects the piece under the cursor.
  const at = h.screenOf([4, 0.225, 7]);
  assert.ok(at.z > 0, "the piece is in front of the camera");
  h.hit([4, 0.225, 7]);
  h.pointer("pointerdown", at); h.pointer("pointerup", at);
  assert.deepEqual(h.sent.filter(e => e.type === "placement-pick").at(-1), { namespace: "groundcontrol", type: "placement-pick", id: "table" });
  assert.equal(h.sent.filter(e => e.type === "placement-move").length, 0, "a select is not a move");
  // Dragging moves the model immediately and coalesces host events to 15 Hz.
  h.frame(100);
  h.pointer("pointerdown", at);
  h.hit([6, 0.225, 9]);
  h.pointer("pointermove", { clientX: at.clientX + 12, clientY: at.clientY + 8 });
  assert.equal(h.sent.filter(e => e.type === "placement-move").length, 1, "motion is throttled, not per frame");
  h.pointer("pointermove", { clientX: at.clientX + 20, clientY: at.clientY + 14 });
  assert.equal(h.sent.filter(e => e.type === "placement-move").length, 1, "a second sample in the same window is dropped");
  const moved = h.sent.filter(e => e.type === "placement-move").at(-1);
  assert.deepEqual(moved.point, [6, 0.225, 9], "the grab offset follows the cursor's floor point");
  assert.equal(moved.final, false);
  assert.deepEqual(h.state().items.get("table").collider.getLocalPosition().toArray(), [6, 0.225, 9]);
  // Over unscanned space there is no surface: the piece holds, it does not jump.
  h.hit(null);
  h.frame(200);
  h.pointer("pointermove", { clientX: at.clientX + 60, clientY: at.clientY + 30 });
  assert.deepEqual(h.state().items.get("table").collider.getLocalPosition().toArray(), [6, 0.225, 9]);
  h.hit([6, 0.225, 9]);
  h.frame(300);
  h.pointer("pointerup");
  const last = h.sent.filter(e => e.type === "placement-move").at(-1);
  assert.equal(last.final, true, "the host persists once, on release");
  // A cancelled drag puts the piece back where the grab started.
  h.frame(400);
  h.pointer("pointerdown", at);
  h.hit([2, 0.225, 3]);
  h.pointer("pointermove", { clientX: at.clientX + 30, clientY: at.clientY + 20 });
  h.pointer("pointercancel");
  assert.deepEqual(h.state().items.get("table").collider.getLocalPosition().toArray(), [6, 0.225, 9]);
  assert.equal(h.sent.filter(e => e.type === "placement-move").at(-1).final, true);
});

test("pc.js focus-placement selects the item and fits an orbit without moving to other placements", t => {
  const h = viewerHarness(t), p = placed("coffee_table", 90);
  h.command("placements", { value: [p, { ...p, id: "far", center_xz: [300, 700] }] });
  h.command("mode", { value: "fly" }); h.command("focus-placement", { id: "table" });
  assert.equal(h.state().mode, "orbit"); assert.equal(h.state().selected, "table");
  assert.deepEqual(h.state().orbit.target, [4, 0.225, 7]); assert.ok(h.state().orbit.distance < 5);
  const eye = h.cameraEnt.getPosition();
  W.orbitEye(h.state().orbit).forEach((v, i) => assert.ok(Math.abs(v - [eye.x, eye.y, eye.z][i]) < 1e-6, "engine transforms use float32 matrices"));
  h.command("focus-placement", { id: "missing" });
  assert.equal(h.sent.at(-1).type, "error"); assert.match(h.sent.at(-1).message, /unavailable/);
  assert.deepEqual(h.state().orbit.target, [4, 0.225, 7]);
});

// A "view from the photo" narrows the lens; a resumed orbit may carry the lens back.
test("camera-set takes an optional field of view and rejects a bad one", () => {
  const orbit = { target: [1, 2, 3], distance: 10, yaw: 0.1, pitch: 0.2 };
  assert.deepEqual(W.readCommand(event(packet("camera-set", { value: orbit })), parent, origin).value, orbit);
  assert.equal(W.readCommand(event(packet("camera-set", { value: { ...orbit, fov: 55 } })), parent, origin).value.fov, 55);
  for (const fov of [0, 180, NaN, "70"]) {
    assert.throws(() => W.readCommand(event(packet("camera-set", { value: { ...orbit, fov } })), parent, origin));
  }
  assert.throws(() => W.readCommand(event(packet("camera-set", { value: { ...orbit, zoom: 2 } })), parent, origin));
});
