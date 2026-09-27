#!/usr/bin/env node
"use strict";
// Engine-free planning layer: payload validation, flat-shaded geometry, ray picking
// and the demolish clip (the JS test and the GPU shader must agree on every point).
const assert = require("node:assert/strict");
const { test, before } = require("node:test");
const { readFileSync } = require("node:fs");
const path = require("node:path");

let P = {};
function moduleUrl(file) {
  const source = readFileSync(file, "utf8");
  return `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
}
before(async () => { P = await import(moduleUrl(path.join(__dirname, "../viewer/plan_core.js"))); });

// A 10 x 10 x 12 box building centred at (0, 0) on y = 0, and a flat road strip at z = 20.
function box(id, cx, cz, h, color = [233, 196, 140]) {
  const x0 = cx - 5, x1 = cx + 5, z0 = cz - 5, z1 = cz + 5;
  const positions = [x0, 0, z0, x1, 0, z0, x1, 0, z1, x0, 0, z1, x0, h, z0, x1, h, z0, x1, h, z1, x0, h, z1];
  const indices = [4, 6, 5, 4, 7, 6, 0, 5, 1, 0, 4, 5, 1, 6, 2, 1, 5, 6, 2, 7, 3, 2, 6, 7, 3, 4, 0, 3, 7, 4];
  return { id, kind: "building", meshes: [{ positions, indices, color, opacity: 1 }] };
}
const road = { id: "road-1", kind: "road", meshes: [{ positions: [-30, 0.06, 17, 30, 0.06, 17, 30, 0.06, 23, -30, 0.06, 23], indices: [0, 2, 1, 0, 3, 2], color: [60, 64, 72] }] };

test("valid payloads pass and malformed ones are refused", () => {
  const plan = P.validatePlan({ features: [box("b-1", 0, 0, 12), road], clips: [], view: "proposal", selected: "b-1" });
  assert.equal(plan.features.length, 2);
  assert.equal(plan.selected, "b-1");
  const bad = [
    { features: [{ id: "x", meshes: [{ positions: [0, 0, 0], indices: [0, 1, 2], color: [1, 2, 3] }] }] },
    { features: [{ id: "x", meshes: [{ positions: [0, 0, NaN], indices: [], color: [1, 2, 3] }] }] },
    { features: [{ id: "x", meshes: [{ positions: [0, 0, 0], indices: [], color: [1, 2, 300] }] }] },
    { features: [], clips: Array(9).fill({ cx: 0, cz: 0, hx: 1, hz: 1, angle: 0, y_min: 0, y_max: 1 }) },
    { features: [], view: "sideways" },
  ];
  for (const value of bad) assert.throws(() => P.validatePlan(value));
});

test("flat-shaded geometry expands triangles and tints violations and selection", () => {
  const mesh = box("b-1", 0, 0, 12).meshes[0];
  const plain = P.planGeometry(mesh);
  assert.equal(plain.positions.length, 10 * 9);
  assert.equal(plain.colors.length, 10 * 12);
  assert.equal(plain.transparent, false);
  const red = P.planGeometry({ ...mesh, violation: true });
  assert.ok(red.colors[0] > plain.colors[0] && red.colors[1] < plain.colors[1], "violation reads red");
  const glass = P.planGeometry({ ...mesh, opacity: 0.3 });
  assert.equal(glass.transparent, true);
  assert.equal(glass.colors[3], Math.round(255 * 0.3));
});

test("ray picking returns the nearest feature and misses cleanly", () => {
  const features = [box("b-1", 0, 0, 12), box("b-2", 0, -30, 20), road];
  assert.equal(P.rayPick(features, [0, 50, 0], [0, -1, 0]).id, "b-1");
  const hit = P.rayPick(features, [0, 6, 60], [0, 0, -1]);
  assert.equal(hit.id, "b-1");                       // b-1 is in front of b-2 along -z
  assert.ok(Math.abs(hit.t - 55) < 1e-9);
  assert.equal(P.rayPick(features, [0, 50, 20], [0, -1, 0]).id, "road-1");
  assert.equal(P.rayPick(features, [100, 50, 100], [0, -1, 0]), null);
  assert.equal(P.rayPick(features, [0, 50, 20], [0, -1, 0], { kinds: ["building"] }), null);
});

test("the demolish test in JS and the shader's uniform math agree", () => {
  const clips = [{ cx: 15, cz: 0, hx: 6, hz: 4, angle: 0.6, y_min: -1, y_max: 40 },
                 { cx: -20, cz: 10, hx: 3, hz: 3, angle: 0, y_min: 0, y_max: 5 }];
  const uniforms = P.clipUniforms(clips);
  assert.equal(uniforms.length, P.MAX_CLIPS);
  const shaderInside = ([x, y, z]) => uniforms.some(([a, b]) => {
    const dx = x - a[0], dz = z - a[1];
    const u = dx * b[0] + dz * b[1], v = -dx * b[1] + dz * b[0];
    return Math.abs(u) <= a[2] && Math.abs(v) <= a[3] && y >= b[2] && y <= b[3];
  });
  let inside = 0;
  for (let i = 0; i < 20000; i++) {
    const p = [Math.random() * 80 - 40, Math.random() * 50 - 5, Math.random() * 40 - 20];
    assert.equal(P.insideClip(clips, p), shaderInside(p), JSON.stringify(p));
    if (P.insideClip(clips, p)) inside++;
  }
  assert.ok(inside > 100, "the sample actually exercised the inside branch");
  assert.ok(P.insideClip(clips, [15, 10, 0]));
  assert.ok(!P.insideClip(clips, [15, 50, 0]), "above the demolished volume stays");
  assert.ok(!P.insideClip([], [0, 0, 0]), "no clips hide nothing");
});

test("clip shaders declare every slot and implement the modify hooks in both languages", () => {
  for (const language of ["glsl", "wgsl"]) {
    const code = P.clipShader(language);
    for (let i = 0; i < P.MAX_CLIPS; i++) assert.ok(code.includes(`uPlanClipA${i}`) && code.includes(`uPlanClipB${i}`));
    for (const hook of ["modifySplatCenter", "modifySplatRotationScale", "modifySplatColor"]) assert.ok(code.includes(hook), `${language} ${hook}`);
  }
  assert.ok(P.clipShader("wgsl").includes("uniform.uPlanClipA0"));
});

test("the redraw signature changes when any vertex moves, not only the ends", () => {
  const b = box("b-1", 0, 0, 12);
  const taller = box("b-1", 0, 0, 19.2);
  assert.notEqual(P.planSignature(b, false), P.planSignature(taller, false));
  assert.equal(P.planSignature(b, false), P.planSignature(box("b-1", 0, 0, 12), false));
  assert.notEqual(P.planSignature(b, false), P.planSignature(b, true));
  const tinted = { ...b, meshes: [{ ...b.meshes[0], violation: true }] };
  assert.notEqual(P.planSignature(b, false), P.planSignature(tinted, false));
});

// ---- 3D editing: handles, drags, walk colliders and demolished collider triangles.
const square = { id: "b-1", points: [[0, 0], [10, 0], [10, 10], [0, 10]], y: 2, closed: true, vertices: true, rotate: true };

test("edit payloads are validated and the handle set matches the shape", () => {
  const edit = P.validateEdit(square);
  assert.equal(edit.min, 3);
  const handles = P.planHandles(edit);
  const count = kind => handles.filter(h => h.kind === kind).length;
  assert.deepEqual([count("vertex"), count("mid"), count("move"), count("rotate")], [4, 4, 1, 1]);
  assert.deepEqual(handles.find(h => h.kind === "move").xz, [5, 5]);
  const road = P.validateEdit({ id: "r-1", points: [[0, 0], [20, 0], [20, 20]], y: 0, closed: false, vertices: true, rotate: false });
  const rh = P.planHandles(road);
  assert.equal(rh.filter(h => h.kind === "mid").length, 2, "an open line has n-1 edges");
  assert.equal(rh.filter(h => h.kind === "rotate").length, 0);
  const object = P.validateEdit({ id: "o-1", points: [[3, 4]], y: 0, closed: false, vertices: false, rotate: true, heading_deg: 90 });
  const rot = P.planHandles(object).find(h => h.kind === "rotate");
  assert.ok(Math.abs(rot.xz[0] - 3) < 1e-9 && rot.xz[1] > 4, "the rotate grip points along the heading (+z at 90°)");
  for (const bad of [{ ...square, points: [[0, NaN]] }, { ...square, y: undefined }, { ...square, id: "" }]) {
    assert.throws(() => P.validateEdit(bad));
  }
  assert.equal(P.validatePlan({ features: [], clips: [], edit: square }).edit.id, "b-1");
});

test("drags move a corner, insert a corner, translate and rotate exactly", () => {
  const edit = P.validateEdit(square);
  const handles = P.planHandles(edit);
  const corner = handles.find(h => h.kind === "vertex" && h.index === 2);
  assert.deepEqual(P.dragShape(edit, corner, [10, 10], [12.3, 11.6]).points[2], [12.3, 11.6]);
  assert.deepEqual(P.dragShape(edit, corner, [10, 10], [12.3, 11.6], { grid: 1 }).points[2], [12, 12]);
  const mid = handles.find(h => h.kind === "mid" && h.index === 1);
  const inserted = P.dragShape(edit, mid, [10, 5], [14, 5]).points;
  assert.equal(inserted.length, 5);
  assert.deepEqual(inserted[2], [14, 5]);
  const moved = P.dragShape(edit, handles.find(h => h.kind === "move"), [5, 5], [8, 1]).points;
  assert.deepEqual(moved[0], [3, -4]);
  const grip = handles.find(h => h.kind === "rotate");
  // Drag the grip a quarter turn around the centroid (5, 5): +x toward +z is +90°.
  const turned = P.dragShape(edit, grip, [15, 5], [5, 15]);
  assert.ok(Math.abs(turned.rotation_deg - 90) < 1e-9);
  assert.ok(Math.hypot(turned.points[0][0] - 10, turned.points[0][1] - 0) < 1e-9, "(0,0) turns to (10,0)");
  const snapped = P.dragShape(edit, grip, [15, 5], [5 + 10 * Math.cos(0.3), 5 + 10 * Math.sin(0.3)], { snapDeg: 15 });
  assert.equal(Math.round(snapped.rotation_deg * 1e9) / 1e9, 15);
});

test("corners can be removed down to the minimum only", () => {
  const edit = P.validateEdit(square);
  assert.deepEqual(P.removeVertex(edit, 1), [[0, 0], [10, 10], [0, 10]]);
  assert.equal(P.removeVertex(P.validateEdit({ ...square, points: [[0, 0], [5, 0], [0, 5]] }), 0), null);
});

test("the plane hit follows the ray and refuses rays that never reach it", () => {
  assert.deepEqual(P.planeHit([0, 10, 0], [1, -1, 2], 2), [8, 16]);
  assert.equal(P.planeHit([0, 10, 0], [1, 0, 0], 2), null);
  assert.equal(P.planeHit([0, 10, 0], [0, 1, 0], 2), null, "the plane is behind the camera");
  assert.equal(P.planeHit([0, 10, 0], [1000, -1, 0], 0), null, "a grazing ray would fling the drag across the scene");
});

test("walk colliders: one wall box per footprint edge, trunks and poles but not crowns", () => {
  // Walls as the backend writes them: bottom ring then top ring, 4 corners of a 10 x 6 box.
  const ring = [[0, 0], [10, 0], [10, 6], [0, 6]];
  const positions = [...ring.flatMap(([x, z]) => [x, 0, z]), ...ring.flatMap(([x, z]) => [x, 9, z])];
  const building = { id: "b", kind: "building", meshes: [{ part: "walls", positions, indices: [] }, { part: "roof", positions, indices: [] }] };
  const tree = { id: "t", kind: "object", meshes: [
    { part: "trunk", positions: [-0.2, 0, -0.2, 0.2, 4, 0.2], indices: [] },
    { part: "crown", positions: [-3, 4, -3, 3, 12, 3], indices: [] }] };
  const boxes = P.planColliders([building, tree, { id: "r", kind: "road", meshes: [] }]);
  assert.equal(boxes.length, 5);
  const south = boxes[0];
  assert.deepEqual(south.center, [5, 4.5, 0]);
  assert.deepEqual(south.halfExtents, [5, 4.5, 0.15]);
  assert.ok(Math.abs(boxes[1].yaw + 90) < 1e-9, "the east wall runs along +z: physics yaw -90°");
  const trunk = boxes[4];
  assert.ok(trunk.halfExtents[1] === 2 && trunk.center[1] === 2, "only the trunk, 4 m tall, blocks the walker");
});

test("demolished collider triangles go, the ground under them stays", () => {
  const clips = [{ cx: 5, cz: 5, hx: 5, hz: 5, angle: 0, y_min: -1, y_max: 20 }];
  // Tri 0: roof at 9 m inside; tri 1: ground inside; tri 2: roof outside the box.
  const positions = [4, 9, 4, 6, 9, 4, 5, 9, 6, 4, 0, 4, 6, 0, 4, 5, 0, 6, 30, 9, 30, 32, 9, 30, 31, 9, 32];
  const keep = P.clipTriangles(positions, [0, 1, 2, 3, 4, 5, 6, 7, 8], clips, () => 0);
  assert.deepEqual(keep, [3, 4, 5, 6, 7, 8]);
});

test("shadow overlays are drawn but never picked", () => {
  const overlay = { id: "shadow", kind: "overlay", meshes: [{ positions: [-50, 0.1, -50, 50, 0.1, -50, 0, 0.1, 50], indices: [0, 1, 2], color: [0, 0, 0] }] };
  assert.equal(P.rayPick([overlay], [0, 10, 0], [0, -1, 0]), null);
});
