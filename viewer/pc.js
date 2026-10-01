/* Splat Walk MVP — PlayCanvas engine viewer (walk mode).
 *
 * Real physics trimesh collider + 3D Gaussian Splatting + Drone Flight +
 * Camera Frustums (COLMAP pyramids) + Sparse Tie Points + 3D Coverage Heatmap.
 *
 * URL params:
 *   ?asset=../work/room_w_jsonl/viewer_assets   asset dir
 *   &auto=1                                   run autopilot
 *   &cams=1                                   show camera frustums by default
 *   &drone=1                                  start in drone fly mode
 *   &embed=1                                  parent-controlled orbit workspace
 *   &combat=1                                 combat mode: bots, shooting, HUD
 *                                             (needs work/<scene>/pc/nav.json,
 *                                              baked by tools/navbake/bake.mjs)
 */
import {
  AppBase, AppOptions, Asset, AssetListLoader, BLEND_NONE, BLEND_NORMAL, BODYMASK_STATIC, BoundingBox,
  AnimComponentSystem,
  CameraComponentSystem, CollisionComponentSystem, Color, ContainerHandler, CULLFACE_BACK,
  Entity, FILLMODE_FILL_WINDOW, Layer, SORTMODE_BACK2FRONT,
  FOG_LINEAR, GSplatComponentSystem, GSplatHandler, KEY_A, KEY_C, KEY_D, KEY_R,
  KEY_S, KEY_SHIFT, KEY_SPACE, KEY_T, KEY_W, Keyboard, Mat4, Mesh, MeshInstance, ModelComponentSystem, Mouse,
  PRIMITIVE_LINES, PRIMITIVE_POINTS, PRIMITIVE_TRIANGLES, RenderComponentSystem, RESOLUTION_AUTO,
  RigidBodyComponentSystem, StandardMaterial, TONEMAP_LINEAR, TextureHandler,
  Vec3, WasmModule, createGraphicsDevice,
} from "playcanvas";
import { TARGET_HEIGHT, makeCharacter } from "./pc/scripts/character.js";
import {
  WorkspaceProtocol, SnapshotCapture, finitePoint, boundsFromPoints, fitOrbit,
  orbitEye, orbitFromPose, rotateOrbit, panOrbit, zoomOrbit, layerState,
  canvasPoint, collisionSurfacePoint, appendPick, workspaceAssetPath, walkDistance,
  measureRenderable, measureAnchor, formatMeasureLabel, measureColor,
  liveMeasurement, PreviewStream, projectedCssPoint, layoutLabels,
  placementBox, placementColor, placementLabel, placementRay, focusPlacement, MAX_PICKS,
  importedModelFit,
} from "./workspace_core.js";
import { furnitureGeometry } from "./furniture_geometry.js";
import {
  planGeometry, planSignature, rayPick, clipShader, clipUniforms, planHandles, dragShape, removeVertex, planeHit,
  planColliders, clipTriangles,
} from "./plan_core.js";

// Unlit material helper
function unlitMat(r = 1, g = 1, b = 1, transparent = false, opacity = 1.0) {
  const m = new StandardMaterial();
  m.useLighting = false;
  m.diffuse.set(r, g, b);
  if (transparent) {
    m.blendType = 2; // BLEND_NORMAL
    m.opacity = opacity;
  }
  m.update();
  return m;
}

const q = new URLSearchParams(location.search);
const EMBED = q.get("embed") === "1";
let workspaceMode = "orbit";
let workspaceOrbit = null;
let workspaceCommand = null;
let workspacePick = false;
let workspacePickLimit = 128;
let editDrag = null;   // { id, pointerId, offset, moved, last } while the editor drags a piece
let workspaceLookDirty = false;
const collisionMeshEntities = new Set();
const workspace = EMBED ? new WorkspaceProtocol({
  parent: window.parent, origin: location.origin,
  send: data => window.parent.postMessage(data, location.origin),
  getState: () => ({ mode: workspaceMode, layers: workspaceLayers(), selectedFrame: selectedCamIdx }),
  getDetails: () => ({ capabilities: workspaceCapabilities(), cameraCount: allCameras.length }),
  execute: cmd => workspaceCommand(cmd),
}) : null;
if (workspace) window.addEventListener("message", event => workspace.receive(event));
let rawAsset = q.get("asset") || "/work/room_w_jsonl/viewer_assets";
if (!rawAsset.includes("/")) {
  rawAsset = `/work/${rawAsset}/viewer_assets`;
}
let ASSET = rawAsset;
if (ASSET.startsWith("../")) {
  ASSET = "/" + ASSET.slice(3);
} else if (!ASSET.startsWith("/")) {
  ASSET = "/" + ASSET;
}

// Extract base scene work dir, e.g. "/work/room_w_jsonl"
const WORK_DIR = ASSET.replace(/\/viewer_assets\/?$/, "");

const AUTO = !EMBED && q.get("auto") === "1";
const SHOOT = !EMBED && q.get("shoot") === "1";
const COMBAT = !EMBED && q.get("combat") === "1";
/** Viewer shortcuts combat replaces: R reloads, F/C would break the aiming camera. */
const COMBAT_KEYS = ["KeyR", "KeyF", "KeyC"];
const UNDERLAY = !EMBED && q.get("underlay") === "1";
const SINK = Number(q.get("sink") ?? 0.7);
let combat = null;
/** WebXR state (viewer/pc/scripts/vr.js) once the arena offers VR; null elsewhere. */
let vrState = null;

// Character height: 1.75 m by default (a real person). A room preset ships
// `character_height: 0.15` (hamster) because a 2 m-tall ceiling and a 3 m
// walkable footprint cannot host a human without the capsule clipping through
// the walls. Every human-scaled offset below (eye line, camera rise, spawn
// hover) is derived from CHAR_H so one knob moves the whole rig consistently.
let CHAR_H = 1.75;
let CHAR_SCALE = 1.0;     // CHAR_H / 1.75, filled in after collision.json loads
const charCfg = q.get("char");
if (charCfg) { CHAR_H = Math.max(0.05, Number(charCfg)); }
function applyCharScale() { CHAR_SCALE = CHAR_H / 1.75; }
applyCharScale();

const loadEl = document.getElementById("load");
const hudEl = document.getElementById("hud");
const setLoad = (msg, err = false) => {
  if (loadEl) {
    if (msg && !loadEl.isConnected) document.body.appendChild(loadEl);
    loadEl.textContent = msg;
    loadEl.classList.toggle("err", err);
    if (msg) { loadEl.style.opacity = "1"; loadEl.style.pointerEvents = "auto"; }
  }
};
const fail = (msg) => {
  window.__loadError = String(msg);
  window.__ready = false;
  workspacePick = false;
  workspace?.fail(msg);
  setLoad("ERROR: " + msg, true);
  throw new Error(String(msg));
};

// ---------------- drone / splat / camera state ----------------
let isDrone = EMBED || q.get("drone") === "1";
let splatVisible = true;
let currentPly = q.get("full") === "1" ? "scene.full.ply" : ((!EMBED && q.get("ply")) || "scene.ply");
const dronePos = new Vec3(0, 0, 0);
// Fly-mode speed scales with character size so a hamster does not cross a room
// at human flight speed and blur past every wall.
let droneSpeed = 7.0 * CHAR_SCALE;
let splatEnt = null;
let applicationRef = null;
const activeKeys = {};

// Visualizer layer states
// Cameras default OFF — the debug frustums are a diagnostic, not the scene.
// Turn on with ?cams=1 (or from the toolbar).
let showCameras = q.get("cams") === "1";
let showPoints = false;
let showCoverage = false;
let showCollider = q.get("showcol") === "1";
let colliderEnt = null;
let colliderTris = 0;
let objectBoxes = [];
let objectsEnt = null;
let allCameras = [];
let allSparsePoints = null;
let allSemantics = null;
let allCoverageGrid = null;
let selectedCamIdx = -1;

let camerasEntity = null;
let trajectoryEntity = null;
let selectedCamEntity = null;
let sparsePointsEntity = null;
let semanticsEntity = null;
let showSemantics = false;
let coverageGridEntity = null;

function workspaceLayers() {
  return { splats: splatVisible, cameras: showCameras, points: showPoints,
    coverage: showCoverage, collider: showCollider, semantics: showSemantics };
}
function workspaceCapabilities() {
  return { cameras: !!camerasEntity, points: !!sparsePointsEntity,
    coverage: !!coverageGridEntity, collider: collisionMeshEntities.size > 0,
    semantics: !!semanticsEntity };
}
function clearActiveKeys() {
  for (const key of Object.keys(activeKeys)) delete activeKeys[key];
}
window.addEventListener("blur", clearActiveKeys);
document.addEventListener("visibilitychange", () => { if (document.hidden) clearActiveKeys(); });
if (EMBED) {
  const stopOnError = event => {
    clearActiveKeys();
    workspacePick = false;
    workspace.fail(event.message || event.reason?.message || event.reason || "Viewer failed");
  };
  window.addEventListener("error", stopOnError);
  window.addEventListener("unhandledrejection", stopOnError);
}

window.addEventListener("keydown", (e) => {
  activeKeys[e.code] = true;
  activeKeys[e.key.toLowerCase()] = true;
});
window.addEventListener("keyup", (e) => {
  activeKeys[e.code] = false;
  activeKeys[e.key.toLowerCase()] = false;
});

// ---------------- scene data ----------------
const SKY = new Color(0.043, 0.055, 0.078); // sleek dark void
// Human gait is ~2.6 m/s walk / ~5.2 m/s run. When the character is hamster-
// sized those numbers would move it 17x faster than its own body length per
// second and the room would look like a bullet-time effect. Scale by CHAR_H so
// the stride-to-height ratio stays realistic.
let WALK_SPEED = 2.6, RUN_SPEED = 5.2;
let HF = null;

async function loadSceneData() {
  const resp = await fetch(`${ASSET}/collision.json`);
  if (!resp.ok) {
    fail(`Could not load ${ASSET}/collision.json (HTTP ${resp.status}). Please verify scene assets exist.`);
  }
  const col = await resp.json();
  // Prefer a scene-authored character height over the 1.75 m default; the URL
  // ?char=X still wins so an operator can override on the fly.
  if (col.character_height && !q.get("char")) {
    CHAR_H = Number(col.character_height);
    applyCharScale();
    WALK_SPEED = 2.6 * CHAR_SCALE;
    RUN_SPEED = 5.2 * CHAR_SCALE;
    console.log(`[char] scene asks for ${CHAR_H.toFixed(2)} m character ` +
                `(scale ${CHAR_SCALE.toFixed(2)}x), walk ${WALK_SPEED.toFixed(2)} m/s`);
  }
  const { nx, nz, cell, origin_xz } = col;
  let data = null, src = "ground.f32";
  for (const name of ["ground.f32", "heights.f32"]) {
    try {
      const r = await fetch(`${ASSET}/${name}`);
      if (r.ok) {
        const b = await r.arrayBuffer();
        if (b.byteLength === nx * nz * 4) { data = new Float32Array(b); src = name; break; }
      }
    } catch { /* optional */ }
  }
  if (!data) fail(`no usable heightfield for ${nx}x${nz}`);
  console.log(`[hf] ${src} (${nx}x${nz})`);

  let colors = null;
  try {
    const cr = await fetch(`${ASSET}/ground_colors.rgb`);
    if (cr.ok) {
      const cbuf = await cr.arrayBuffer();
      if (cbuf.byteLength === nx * nz * 3) colors = new Uint8Array(cbuf);
    }
  } catch { /* optional */ }

  let cov = null;
  try {
    const vr = await fetch(`${ASSET}/coverage.u8`);
    if (vr.ok) {
      const vbuf = await vr.arrayBuffer();
      if (vbuf.byteLength === nx * nz) cov = new Uint8Array(vbuf);
    }
  } catch { /* optional */ }

  // Furniture, and only the furniture the router agreed to route around. It
  // wrote object_colliders itself after checking a walkable floor survives the
  // block; boxes it rejected came off a mis-measured floor, and giving those
  // colliders would wall the player into geometry no path accounts for. So this
  // request is skipped entirely on every scene that has not earned it.
  objectBoxes = [];
  if (col.object_colliders > 0) {
    try {
      const or_ = await fetch(`${ASSET}/objects.json`);
      if (or_.ok) objectBoxes = (await or_.json()).boxes || [];
    } catch { /* optional */ }
  }
  console.log(`[objects] router accepted ${col.object_colliders || 0}, ` +
              `loaded ${objectBoxes.length}`);

  HF = {
    nx, nz, cell, data, colors, cov,
    ox: origin_xz[0], oz: origin_xz[1],
    minX: origin_xz[0] - 2, maxX: origin_xz[0] + nx * cell + 2,
    minZ: origin_xz[1] - 2, maxZ: origin_xz[1] + nz * cell + 2,
  };

  // Load cameras.json, sparse_points.json, coverage_grid.json in parallel
  try {
    const [camsResp, ptsResp, covResp, semResp] = await Promise.allSettled([
      fetch(`${ASSET}/cameras.json`),
      fetch(`${ASSET}/sparse_points.json`),
      fetch(`${ASSET}/coverage_grid.json`),
      fetch(`${ASSET}/semantics.json`)
    ]);
    if (camsResp.status === "fulfilled" && camsResp.value.ok) {
      allCameras = await camsResp.value.json();
      console.log(`[viewer] Loaded ${allCameras.length} camera poses`);
    }
    if (ptsResp.status === "fulfilled" && ptsResp.value.ok) {
      allSparsePoints = await ptsResp.value.json();
      console.log(`[viewer] Loaded ${allSparsePoints.count || 0} sparse tie points`);
    }
    if (semResp.status === "fulfilled" && semResp.value.ok) {
      allSemantics = await semResp.value.json();
      console.log(`[viewer] Loaded semantics (${allSemantics.counts ? Object.values(allSemantics.counts).reduce((a, b) => a + b, 0) : 0} labelled points)`);
    }
    if (covResp.status === "fulfilled" && covResp.value.ok) {
      allCoverageGrid = await covResp.value.json();
      console.log(`[viewer] Loaded coverage grid (${allCoverageGrid.total_voxels} voxels,`
                  + ` status=${allCoverageGrid.status || "measured"})`);
    }
  } catch (e) {
    console.warn("[viewer] Optional coverage data could not be fetched:", e);
  }

  return col;
}

function groundHF(x, z) {
  if (!HF || !HF.data) return 0.0;
  const { nx, nz, cell, data, ox, oz } = HF;
  const gx = Math.min(Math.max((x - ox) / cell - 0.5, 0), nx - 1.001);
  const gz = Math.min(Math.max((z - oz) / cell - 0.5, 0), nz - 1.001);
  const x0 = Math.floor(gx), z0 = Math.floor(gz), fx = gx - x0, fz = gz - z0;
  const h00 = data[z0 * nx + x0], h10 = data[z0 * nx + x0 + 1];
  const h01 = data[(z0 + 1) * nx + x0], h11 = data[(z0 + 1) * nx + x0 + 1];
  return (h00 * (1 - fx) + h10 * fx) * (1 - fz) + (h01 * (1 - fx) + h11 * fx) * fz;
}

const app = {};
let cameraEnt, playerEnt, playerRb, playerChar;

const PROBE_UP = 6, PROBE_DOWN = 10;
function groundProbe(x, z) {
  const h = groundHF(x, z);
  if (!app.systems || !app.systems.rigidbody) return null;
  return app.systems.rigidbody.raycastFirst(
    new Vec3(x, h + PROBE_UP, z), new Vec3(x, h - PROBE_DOWN, z),
    { filterCollisionMask: BODYMASK_STATIC });
}

const P = {
  yaw: 0, pitch: -0.12, walked: 0, grounded: false, violations: 0,
  firstPerson: false, lastGood: null, prev: null, extControl: false,
};
const autopilot = { phase: "idle", t: 0 };
// The walk test's evidence: `walked: 16` is only meaningful next to the route
// that produced it, so the log carries where the body was and whether the floor
// was under it.
const WALK_SAMPLE_DT = 0.5;      // s between samples
const WALK_SAMPLE_MAX = 600;     // 300 s of walk; drive_viewer stops at 240 s
let walkSampleClock = 0;
// autopilot.t restarts on every phase change, so it cannot say how long a walk
// took; the sample stamp has to be monotonic or the logged route lies about its
// own duration.
let walkElapsed = 0;
window.__walk = { phase: "idle", walked: 0, pos: [0, 0, 0], yaw: 0, grounded: false, violations: 0, samples: [] };

function spawnFrom(col) {
  const s = window.__chosenSpawn || col.spawn || { x: 0, z: 0, face_xz: [0, 1] };
  const hit = groundProbe(s.x, s.z);
  const y = hit ? hit.point.y + CHAR_H * 0.86 : groundHF(s.x, s.z) + CHAR_H;
  if (playerEnt && playerRb) {
    playerEnt.rigidbody.teleport(s.x, y, s.z);
    playerRb.linearVelocity = new Vec3(0, 0, 0);
  }
  P.yaw = Math.atan2(-(s.face_xz[0] - s.x), -(s.face_xz[1] - s.z));
  P.walked = 0; P.violations = 0; P.lastGood = null; P.prev = null;
  window.__walk.samples.length = 0;
  walkSampleClock = 0;
  walkElapsed = 0;
  autopilot.phase = AUTO ? "settle" : "idle";
  autopilot.t = 0;
  dronePos.set(s.x, y + 0.5 * CHAR_SCALE, s.z);
}

async function chooseSpawn(col) {
  const s = col.spawn || { x: 0, z: 0, face_xz: [0, 1] };
  const path = col.walk_path || [];
  const cands = [[s.x, s.z]];
  for (let e = 0; e < path.length; e++) {
    const a = path[e], b = path[(e + 1) % path.length];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    for (let d = 2; d < len; d += 4.5) {
      cands.push([a[0] + (b[0] - a[0]) * d / len, a[1] + (b[1] - a[1]) * d / len]);
    }
  }
  for (let r = 1; r <= 2; r++) {
    for (let a = 0; a < 8; a++) {
      cands.push([s.x + r * Math.cos(a * Math.PI / 4), s.z + r * Math.sin(a * Math.PI / 4)]);
    }
  }
  let best = null;
  for (const [x, z] of cands) {
    const hit = groundProbe(x, z);
    if (!hit) continue;
    playerEnt.rigidbody.teleport(x, hit.point.y + 2.0, z);
    playerRb.linearVelocity = new Vec3(0, 0, 0);
    await new Promise((r) => setTimeout(r, 400));
    const p = playerEnt.getPosition();
    if (Math.abs(p.y - hit.point.y - 0.9) > 0.3) continue;

    const tgt = path[0] || [x + 3, z];
    P.yaw = Math.atan2(-(tgt[0] - x), -(tgt[1] - z));
    P.extControl = true;
    const w0 = P.walked;
    const tEnd = performance.now() + 900;
    while (performance.now() < tEnd) {
      const v = playerRb.linearVelocity;
      playerRb.linearVelocity = new Vec3(-Math.sin(P.yaw) * WALK_SPEED, v.y, -Math.cos(P.yaw) * WALK_SPEED);
      playerRb.activate();
      await new Promise((r) => setTimeout(r, 50));
    }
    P.extControl = false;
    const moved = P.walked - w0;
    const score = moved;
    if (!best || score > best.score) {
      best = { score, x, z, surf: +hit.point.y.toFixed(2), face_xz: s.face_xz };
    }
    if (moved >= 0.8) return best;
  }
  return best;
}

// ---------------- 3D Camera Frustums, Tie Points & Coverage Builders ----------------

function buildCameraFrustumsMesh(device, cams) {
  if (!cams || !cams.length) return null;

  const positions = [];
  const colors = [];
  const indices = [];

  let vIdx = 0;
  for (let i = 0; i < cams.length; i++) {
    const c = cams[i];
    const p = c.pos;
    const [v0, v1, v2, v3] = c.corners;
    const top = c.top_mark;

    // Time-based gradient: Start (Cyan) -> Middle (Gold) -> End (Rose)
    const tNorm = i / Math.max(cams.length - 1, 1);
    const cr = Math.sin(tNorm * Math.PI * 0.8) * 0.8 + 0.2;
    const cg = Math.cos(tNorm * Math.PI * 0.5) * 0.7 + 0.3;
    const cb = (1.0 - tNorm) * 0.9 + 0.1;

    // 6 vertices per camera: apex, 4 base corners, top orientation tick
    const baseV = vIdx;
    positions.push(
      p[0], p[1], p[2],       // 0: Apex
      v0[0], v0[1], v0[2],    // 1: Top-Left
      v1[0], v1[1], v1[2],    // 2: Top-Right
      v2[0], v2[1], v2[2],    // 3: Bottom-Right
      v3[0], v3[1], v3[2],    // 4: Bottom-Left
      top[0], top[1], top[2]  // 5: Up pointer
    );

    for (let k = 0; k < 6; k++) {
      colors.push(cr, cg, cb);
    }

    // Line indices
    indices.push(
      // Pyramid sides (Apex -> corners)
      baseV + 0, baseV + 1,
      baseV + 0, baseV + 2,
      baseV + 0, baseV + 3,
      baseV + 0, baseV + 4,
      // Base rectangle
      baseV + 1, baseV + 2,
      baseV + 2, baseV + 3,
      baseV + 3, baseV + 4,
      baseV + 4, baseV + 1,
      // Up orientation triangle
      baseV + 1, baseV + 5,
      baseV + 5, baseV + 2
    );

    vIdx += 6;
  }

  const mesh = new Mesh(device);
  mesh.setPositions(positions);
  mesh.setColors(colors, 3);
  mesh.setIndices(indices);
  mesh.update(PRIMITIVE_LINES);
  return mesh;
}

function buildObjectBoxesMesh(device, boxes) {
  // Edge outline of every furniture box, with the world corners computed here
  // from the same footprint maths the router rasterizes with:
  //   u = along the major axis at world angle yaw, v = across it.
  // Deliberately NOT a rotated unit box: the physics shape has to trust a yaw
  // sign, and an independently derived outline is what proves that sign in the
  // browser instead of letting both paths share one wrong guess.
  if (!boxes || !boxes.length) return null;
  const positions = [];
  const colors = [];
  const indices = [];
  let v = 0;
  for (const b of boxes) {
    const [cx, cz] = b.center_xz;
    const a = b.yaw_deg * Math.PI / 180;
    const ca = Math.cos(a), sa = Math.sin(a);
    const hu = b.size[0] / 2, hw = b.size[2] / 2;
    const y0 = b.center_y - b.size[1] / 2, y1 = b.center_y + b.size[1] / 2;
    const corners = [];
    for (const [qu, qv] of [[-hu, -hw], [hu, -hw], [hu, hw], [-hu, hw]]) {
      const x = cx + qu * ca - qv * sa;
      const z = cz + qu * sa + qv * ca;
      corners.push([x, y0, z], [x, y1, z]);
    }
    for (const c of corners) {
      positions.push(c[0], c[1], c[2]);
      colors.push(1.0, 0.42, 0.12);
    }
    // bottom loop, top loop, then the four verticals
    const loop = [0, 2, 4, 6];
    for (const k of [0, 1, 2, 3]) {
      indices.push(v + loop[k], v + loop[(k + 1) % 4]);
      indices.push(v + loop[k] + 1, v + loop[(k + 1) % 4] + 1);
      indices.push(v + loop[k], v + loop[k] + 1);
    }
    v += 8;
  }
  const mesh = new Mesh(device);
  mesh.setPositions(positions);
  mesh.setColors(colors, 3);
  mesh.setIndices(indices);
  mesh.update(PRIMITIVE_LINES);
  return mesh;
}

function buildTrajectoryMesh(device, cams) {
  if (!cams || cams.length < 2) return null;
  const positions = [];
  const colors = [];
  const indices = [];

  for (let i = 0; i < cams.length; i++) {
    const p = cams[i].pos;
    positions.push(p[0], p[1], p[2]);
    const tNorm = i / Math.max(cams.length - 1, 1);
    colors.push(
      (1.0 - tNorm) * 0.2 + tNorm * 1.0,
      0.8,
      (1.0 - tNorm) * 1.0 + tNorm * 0.1
    );
    if (i < cams.length - 1) {
      indices.push(i, i + 1);
    }
  }

  const mesh = new Mesh(device);
  mesh.setPositions(positions);
  mesh.setColors(colors, 3);
  mesh.setIndices(indices);
  mesh.update(PRIMITIVE_LINES);
  return mesh;
}

function buildSparsePointsMesh(device, data) {
  if (!data || !data.points || !data.points.length) return null;
  const pts = data.points;
  const rgbs = data.colors;
  const n = pts.length;

  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);

  for (let i = 0; i < n; i++) {
    positions[i * 3] = pts[i][0];
    positions[i * 3 + 1] = pts[i][1];
    positions[i * 3 + 2] = pts[i][2];

    const c = rgbs[i] || [200, 200, 200];
    // Linear gamma conversion
    colors[i * 3] = Math.pow(c[0] / 255, 2.0);
    colors[i * 3 + 1] = Math.pow(c[1] / 255, 2.0);
    colors[i * 3 + 2] = Math.pow(c[2] / 255, 2.0);
  }

  const mesh = new Mesh(device);
  mesh.setPositions(positions);
  mesh.setColors(colors, 3);
  mesh.update(PRIMITIVE_POINTS);
  return mesh;
}

// Class-coloured semantic cloud (ground/road/building/vegetation/obstacle).
// Same shape as the sparse points but sourced from semantics.json's coords/rgb.
function buildSemanticsMesh(device, data) {
  if (!data || !data.coords || !data.coords.length) return null;
  const pts = data.coords, rgbs = data.rgb || [], n = pts.length;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    positions[i * 3] = pts[i][0]; positions[i * 3 + 1] = pts[i][1]; positions[i * 3 + 2] = pts[i][2];
    const c = rgbs[i] || [200, 200, 200];
    colors[i * 3] = Math.pow(c[0] / 255, 2.0);
    colors[i * 3 + 1] = Math.pow(c[1] / 255, 2.0);
    colors[i * 3 + 2] = Math.pow(c[2] / 255, 2.0);
  }
  const mesh = new Mesh(device);
  mesh.setPositions(positions);
  mesh.setColors(colors, 3);
  mesh.update(PRIMITIVE_POINTS);
  return mesh;
}

function buildCoverageGridMesh(device, covData) {
  if (!covData || !covData.voxels || !covData.voxels.length) return null;
  const voxels = covData.voxels;
  const positions = [];
  const colors = [];
  const indices = [];

  let idx = 0;
  const sz = 0.08; // small cube crosshair per voxel

  for (const v of voxels) {
    const x = v[0], y = v[1], z = v[2];
    const status = v[4];

    // Color by coverage status:
    // Green = Good (>= 3 views)
    // Orange = Weak (1-2 views)
    // Red / Crimson = Missing / Unobserved (0 views)
    let cr = 0.1, cg = 0.95, cb = 0.3;
    if (status === "weak") {
      cr = 1.0; cg = 0.65; cb = 0.1;
    } else if (status === "missing") {
      cr = 0.95; cg = 0.2; cb = 0.25;
    }

    const b = idx;
    // 3D Crosshair (6 vertices, 3 lines)
    positions.push(
      x - sz, y, z,   x + sz, y, z,
      x, y - sz, z,   x, y + sz, z,
      x, y, z - sz,   x, y, z + sz
    );
    for (let k = 0; k < 6; k++) {
      colors.push(cr, cg, cb);
    }
    indices.push(b, b + 1, b + 2, b + 3, b + 4, b + 5);
    idx += 6;
  }

  const mesh = new Mesh(device);
  mesh.setPositions(positions);
  mesh.setColors(colors, 3);
  mesh.setIndices(indices);
  mesh.update(PRIMITIVE_LINES);
  return mesh;
}

function updateSelectedCameraMesh(device, cam) {
  if (!selectedCamEntity) {
    selectedCamEntity = new Entity("selectedCam");
    app.root.addChild(selectedCamEntity);
  }
  // Clear previous components
  if (selectedCamEntity.render) {
    selectedCamEntity.removeComponent("render");
  }
  if (!cam) return;

  const p = cam.pos;
  const [v0, v1, v2, v3] = cam.corners;
  const top = cam.top_mark;

  // Far FOV projection rays (4m out)
  const fw = cam.forward;
  const fovD = 3.5;
  const f0 = [p[0] + (v0[0] - p[0]) * fovD / 0.22, p[1] + (v0[1] - p[1]) * fovD / 0.22, p[2] + (v0[2] - p[2]) * fovD / 0.22];
  const f1 = [p[0] + (v1[0] - p[0]) * fovD / 0.22, p[1] + (v1[1] - p[1]) * fovD / 0.22, p[2] + (v1[2] - p[2]) * fovD / 0.22];
  const f2 = [p[0] + (v2[0] - p[0]) * fovD / 0.22, p[1] + (v2[1] - p[1]) * fovD / 0.22, p[2] + (v2[2] - p[2]) * fovD / 0.22];
  const f3 = [p[0] + (v3[0] - p[0]) * fovD / 0.22, p[1] + (v3[1] - p[1]) * fovD / 0.22, p[2] + (v3[2] - p[2]) * fovD / 0.22];

  const positions = [
    p[0], p[1], p[2],    v0[0], v0[1], v0[2],
    v1[0], v1[1], v1[2], v2[0], v2[1], v2[2],
    v3[0], v3[1], v3[2], top[0], top[1], top[2],
    // Far frustum box
    f0[0], f0[1], f0[2], f1[0], f1[1], f1[2],
    f2[0], f2[1], f2[2], f3[0], f3[1], f3[2]
  ];

  const indices = [
    0, 1, 0, 2, 0, 3, 0, 4,
    1, 2, 2, 3, 3, 4, 4, 1,
    1, 5, 5, 2,
    // Far projection lines
    0, 6, 0, 7, 0, 8, 0, 9,
    6, 7, 7, 8, 8, 9, 9, 6
  ];

  const mesh = new Mesh(device);
  mesh.setPositions(positions);
  mesh.setIndices(indices);
  mesh.update(PRIMITIVE_LINES);

  const mat = unlitMat(1.0, 0.88, 0.15); // Vibrant Gold
  selectedCamEntity.addComponent("render", {
    meshInstances: [new MeshInstance(mesh, mat)]
  });
}

// ---------------- boot ----------------
const canvas = document.getElementById("app");
async function boot() {
  if (EMBED) ASSET = workspaceAssetPath(ASSET);
  window.__stage = "data";
  setLoad("Loading terrain, cameras & collision metadata…");
  const col = await loadSceneData();
  window.__spawn = col.spawn || null;
  window.__walkPath = col.walk_path || null;

  if (col.spawn) {
    dronePos.set(col.spawn.x, groundHF(col.spawn.x, col.spawn.z) + CHAR_H, col.spawn.z);
    if (col.spawn.face_xz) {
      P.yaw = Math.atan2(-(col.spawn.face_xz[0] - col.spawn.x), -(col.spawn.face_xz[1] - col.spawn.z));
    }
  }

  window.__stage = "wasm";
  setLoad("Initializing physics engine (ammo.wasm)…");
  WasmModule.setConfig("Ammo", {
    glueUrl: "./pc/ammo/ammo.wasm.js",
    wasmUrl: "./pc/ammo/ammo.wasm.wasm",
    fallbackUrl: "./pc/ammo/ammo.js",
  });
  await new Promise((resolve, reject) => {
    try { WasmModule.getInstance("Ammo", () => resolve(true)); }
    catch (e) { reject(e); }
  });

  window.__stage = "device";
  setLoad("Creating WebGL2 renderer…");

  const device = await createGraphicsDevice(canvas, {
    deviceTypes: ["webgl2"], antialias: false, preserveDrawingBuffer: EMBED,
  });
  const opts = new AppOptions();
  opts.graphicsDevice = device;
  opts.componentSystems = [
    RenderComponentSystem, CameraComponentSystem, CollisionComponentSystem,
    RigidBodyComponentSystem, GSplatComponentSystem, AnimComponentSystem,
  ];
  opts.resourceHandlers = [TextureHandler, ContainerHandler, GSplatHandler];
  const application = new AppBase(canvas);
  application.init(opts);
  application.setCanvasFillMode(FILLMODE_FILL_WINDOW);
  application.setCanvasResolution(RESOLUTION_AUTO);
  application.keyboard = new Keyboard(window);
  application.mouse = new Mouse(canvas);
  Object.assign(app, application);
  applicationRef = application;
  if (EMBED) {
    // The iframe changes size when the host opens drawers; no window event is required.
    const resize = () => application.resizeCanvas();
    const observer = new ResizeObserver(resize);
    observer.observe(document.documentElement);
    window.addEventListener("resize", resize);
    application.once("destroy", () => {
      observer.disconnect();
      window.removeEventListener("resize", resize);
    });
  }

  const assets = {
    splat: new Asset("splat", "gsplat", { url: `${ASSET}/${currentPly}` }),
    collision: new Asset("collision", "container", { url: `${ASSET}/../pc/collision.collision.glb` }),
    rig: new Asset("character", "container", { url: "assets/models/cesium_man.glb" }),
  };
  window.__stage = "assets";
  setLoad(`Downloading 3D splat & collision model…`);
  await new Promise((resolve, reject) => {
    new AssetListLoader(Object.values(assets), application.assets).load(
      (err) => (err ? reject(err) : resolve()));
  });
  if (EMBED && !assets.splat.resource?.numSplats) throw new Error("Scene contains no Gaussian splats");
  window.__stage = "start";
  setLoad("Rendering 3D scene & camera frustums…");

  // A scene built from uniformly tiny, flat splats (a synthetic textured ground) loses
  // its far half to the engine's small-splat cull; such a scene lowers the thresholds.
  const cull = col.splat_cull;
  if (cull && application.scene.gsplat) {
    if (Number.isFinite(cull.min_pixel_size)) application.scene.gsplat.minPixelSize = cull.min_pixel_size;
    if (Number.isFinite(cull.min_contribution)) application.scene.gsplat.minContribution = cull.min_contribution;
  }
  application.scene.fog.type = EMBED ? "none" : FOG_LINEAR;
  application.scene.fogColor = SKY.clone();
  application.scene.fogStart = 35;
  application.scene.fogEnd = 120;
  application.scene.ambientLight = new Color(1, 1, 1);

  application.start();

  // 1. Splat Entity
  splatEnt = new Entity("splat");
  splatEnt.addComponent("gsplat", { asset: assets.splat });
  application.root.addChild(splatEnt);

  // 2. Collision Trimesh — physics always; the wireframe layer is the toggle.
  const colRoot = assets.collision.resource.instantiateRenderEntity();
  application.root.addChild(colRoot);
  colliderEnt = colRoot;
  // X-ray on purpose: the defect this exists to catch is a collider surface the
  // splat does not show (a floater shelf a metre above the visible floor), and
  // depth-testing it against the splat hides exactly that.
  const colMat = unlitMat(0.25, 0.95, 0.75);
  colMat.depthTest = false;
  colMat.update();
  colRoot.findComponents("render").forEach((render) => {
    render.entity.addComponent("rigidbody", { type: "static", friction: 0.6, restitution: 0 });
    render.entity.addComponent("collision", { type: "mesh", renderAsset: render.asset });
    if (render.meshInstances.some(mi => mi.mesh?.primitive?.[0]?.count >= 3)) {
      collisionMeshEntities.add(render.entity);
    }
    render.enabled = showCollider;
    for (const mi of render.meshInstances) {
      // The component setter is not enough: a GLB's mesh instances carry their own
      // material, and that is the one that reaches the draw call.
      mi.material = colMat;
      mi.renderStyle = 1;                    // RENDERSTYLE_WIREFRAME
      const prim = mi.mesh?.primitive?.[0];
      if (prim?.count) colliderTris += Math.floor(prim.count / 3);
    }
  });

  // 2b. Furniture as its own solids. The trimesh above is a 2.5D skin — one
  // height per column — so it can only ever draw a seat bank as a bump in the
  // floor, and the capsule cheerfully walks across the upholstery. Each box the
  // router accepted becomes a real obstacle on the same static body.
  if (objectBoxes.length) {
    const objRoot = new Entity("objects");
    objRoot.addComponent("rigidbody", { type: "static", friction: 0.6, restitution: 0 });
    // The compound is what makes "on the same static body" true rather than
    // aspirational. A child collision with no compound ancestor builds its own
    // body, and this engine registers those in the trigger group: measured with a
    // ray confined inside a seat, `filterCollisionMask: BODYMASK_STATIC` answered
    // with the scan sheet 71 cm below it and only `16` answered with the box. So
    // every sight line, ground probe and round in combat flew through furniture.
    objRoot.addComponent("collision", { type: "compound" });
    application.root.addChild(objRoot);
    for (const b of objectBoxes) {
      const e = new Entity();
      e.setLocalPosition(b.center_xz[0], b.center_y, b.center_xz[1]);
      // -yaw, derived not guessed: PlayCanvas yaw takes local +Z to +X, so local
      // +X lands on (cos t, 0, -sin t). The footprint's major axis is
      // (cos yaw, 0, sin yaw), which is t = -yaw for both axes at once.
      e.setLocalEulerAngles(0, -b.yaw_deg, 0);
      e.addComponent("collision", {
        type: "box",
        halfExtents: new Vec3(b.size[0] / 2, b.size[1] / 2, b.size[2] / 2),
      });
      objRoot.addChild(e);
    }
    const objMesh = buildObjectBoxesMesh(device, objectBoxes);
    if (objMesh) {
      const oMat = unlitMat(1, 1, 1);
      oMat.diffuseVertexColor = true;
      oMat.depthTest = false;
      oMat.update();
      objectsEnt = new Entity("objectOutlines");
      objectsEnt.addComponent("render", { meshInstances: [new MeshInstance(objMesh, oMat)] });
      objectsEnt.enabled = showCollider;
      application.root.addChild(objectsEnt);
    }
    console.log(`[objects] ${objectBoxes.length} furniture colliders on a static body`);
  }

  // 3. Build Camera Frustums & Trajectory 3D Entities
  if (allCameras.length) {
    const frustumMesh = buildCameraFrustumsMesh(device, allCameras);
    if (frustumMesh) {
      const mat = unlitMat(1, 1, 1);
      mat.diffuseVertexColor = true;
      mat.update();
      camerasEntity = new Entity("camerasEntity");
      camerasEntity.addComponent("render", {
        meshInstances: [new MeshInstance(frustumMesh, mat)]
      });
      camerasEntity.enabled = showCameras;
      application.root.addChild(camerasEntity);
    }

    const trajMesh = buildTrajectoryMesh(device, allCameras);
    if (trajMesh) {
      const matT = unlitMat(1, 1, 1);
      matT.diffuseVertexColor = true;
      matT.update();
      trajectoryEntity = new Entity("trajectoryEntity");
      trajectoryEntity.addComponent("render", {
        meshInstances: [new MeshInstance(trajMesh, matT)]
      });
      trajectoryEntity.enabled = showCameras;
      application.root.addChild(trajectoryEntity);
    }
  }

  // 4. Build Sparse Point Cloud Entity
  if (allSparsePoints && allSparsePoints.count) {
    const ptsMesh = buildSparsePointsMesh(device, allSparsePoints);
    if (ptsMesh) {
      const matP = unlitMat(1, 1, 1);
      matP.diffuseVertexColor = true;
      matP.update();
      sparsePointsEntity = new Entity("sparsePointsEntity");
      sparsePointsEntity.addComponent("render", {
        meshInstances: [new MeshInstance(ptsMesh, matP)]
      });
      sparsePointsEntity.enabled = showPoints;
      application.root.addChild(sparsePointsEntity);
    }
  }

  // 4b. Semantic class overlay (off by default; a diagnostic, not the model).
  if (allSemantics && allSemantics.coords && allSemantics.coords.length) {
    const semMesh = buildSemanticsMesh(device, allSemantics);
    if (semMesh) {
      const matS = unlitMat(1, 1, 1);
      matS.diffuseVertexColor = true;
      matS.depthTest = false;
      matS.update();
      semanticsEntity = new Entity("semanticsEntity");
      semanticsEntity.addComponent("render", { meshInstances: [new MeshInstance(semMesh, matS)] });
      semanticsEntity.enabled = false;
      application.root.addChild(semanticsEntity);
    }
  }

  // 5. Build 3D Coverage Grid Entity
  if (allCoverageGrid && allCoverageGrid.voxels) {
    const covMesh = buildCoverageGridMesh(device, allCoverageGrid);
    if (covMesh) {
      const matC = unlitMat(1, 1, 1);
      matC.diffuseVertexColor = true;
      matC.update();
      coverageGridEntity = new Entity("coverageGridEntity");
      coverageGridEntity.addComponent("render", {
        meshInstances: [new MeshInstance(covMesh, matC)]
      });
      coverageGridEntity.enabled = showCoverage;
      application.root.addChild(coverageGridEntity);
    }
  }

  // 6. Character: a measured person, not a pair of primitives.
  //
  // The old avatar drew a capsule from 0.78 m above the entity origin and a
  // sphere at 1.48 m, while the PHYSICS capsule is centred on that same origin —
  // so the figure floated half a metre off its own floor and read far taller than
  // the person it was supposed to be. Both complaints are the same bug, and
  // heighting it from a typed constant would have kept them apart.
  //
  // For a room-scene (character_height 0.15 m in collision.json) every number
  // here scales by CHAR_SCALE so the hamster gets a hamster-sized capsule, mass,
  // and friction. Mass 80 kg × CHAR_SCALE³ keeps density constant.
  playerEnt = new Entity("player");
  playerEnt.addComponent("collision", {
    type: "capsule", radius: 0.34 * CHAR_SCALE, height: 1.8 * CHAR_SCALE,
  });
  playerRb = playerEnt.addComponent("rigidbody", {
    type: "dynamic", mass: 80 * CHAR_SCALE * CHAR_SCALE * CHAR_SCALE,
    linearDamping: 0.05, angularDamping: 1,
    friction: 0.35, restitution: 0,
  });
  playerRb.angularFactor = new Vec3(0, 0, 0);
  application.root.addChild(playerEnt);

  playerChar = await makeCharacter(application, assets.rig, {
    parent: playerEnt, height: CHAR_H,
  });
  window.__char = playerChar;
  console.log(`[character] rig authored ${(playerChar.measured.authoredHeight * 100).toFixed(1)} cm ` +
              `-> ${(100 * playerChar.measured.scale).toFixed(0)}% scale for ` +
              `${CHAR_H.toFixed(2)} m, clip "${playerChar.clip || "none"}", ` +
              `rifle ${playerChar.rifle ? "in hand" : "absent"}`);

  // 7. Viewer Camera Entity
  // nearClip scales with CHAR_SCALE so a hamster's own nose does not clip the
  // geometry it is standing on; 0.08 m (human default) is 53 % of a 0.15 m
  // hamster's total height and culls the entire wall in front of it.
  cameraEnt = new Entity("camera");
  cameraEnt.addComponent("camera", {
    fov: 70, nearClip: 0.08 * CHAR_SCALE, farClip: 3000, clearColorBuffer: true,
  });
  cameraEnt.camera.clearColor = new Color(SKY.r, SKY.g, SKY.b, 1);
  cameraEnt.camera.toneMapping = TONEMAP_LINEAR;
  application.root.addChild(cameraEnt);
  // Translucent planning geometry (shadows, plot outlines, demolition volumes) must draw
  // AFTER the splats: splats blend without writing depth, so anything translucent drawn
  // before them in the World layer is simply painted over by the ground.
  const planOverlayLayer = new Layer({ name: "PlanOverlay", transparentSortMode: SORTMODE_BACK2FRONT });
  {
    const composition = application.scene.layers;
    composition.insertTransparent(planOverlayLayer, composition.getTransparentIndex(composition.getLayerByName("World")) + 1);
    cameraEnt.camera.layers = [...cameraEnt.camera.layers, planOverlayLayer.id];
  }

  // ---------------- UI and Inspector Wiring ----------------
  function updateToolbarUI() {
    const btnMode = document.getElementById("btn-mode");
    const lblMode = document.getElementById("lbl-mode");
    const btnCams = document.getElementById("btn-cams");
    const btnPoints = document.getElementById("btn-points");
    const btnCov = document.getElementById("btn-cov");
    const btnCol = document.getElementById("btn-col");
    const lblCol = document.getElementById("lbl-col");
    const btnSplatVis = document.getElementById("btn-splat-vis");
    const lblSplatVis = document.getElementById("lbl-splat-vis");
    const btnSplat = document.getElementById("btn-splat");
    const lblSplat = document.getElementById("lbl-splat");
    const btnCam = document.getElementById("btn-cam");
    const lblCam = document.getElementById("lbl-cam");

    if (btnMode && lblMode) {
      btnMode.classList.toggle("active", isDrone);
      lblMode.textContent = isDrone ? "🚁 Mode: Drone Fly" : "🚶 Mode: Ground Walk";
    }
    if (btnCams) {
      btnCams.classList.toggle("active", showCameras);
    }
    if (btnPoints) {
      btnPoints.classList.toggle("active", showPoints);
    }
    if (btnCol) {
      btnCol.classList.toggle("active", showCollider);
      if (lblCol) {
        const obj = objectBoxes.length ? ` +${objectBoxes.length} obj` : "";
        lblCol.textContent = !colliderEnt ? "Collider: none"
          : colliderTris ? `Collider ${(colliderTris / 1000).toFixed(0)}k${obj}`
            : `Collider${obj}`;
      }
    }
    if (btnCov) {
      btnCov.classList.toggle("active", showCoverage);
    }
    if (btnSplatVis && lblSplatVis) {
      btnSplatVis.classList.toggle("active", splatVisible);
      lblSplatVis.textContent = splatVisible ? "✨ Splats: ON" : "🚫 Splats: OFF";
    }
    if (btnSplat && lblSplat) {
      btnSplat.classList.toggle("active", currentPly === "scene.full.ply");
      lblSplat.textContent = `✨ Splat: ${splatCountLabel()}`;
    }
    if (btnCam && lblCam) {
      lblCam.textContent = P.firstPerson ? "1st Person" : "3rd Person";
      btnCam.classList.toggle("active", P.firstPerson);
    }
  }

  function toggleSplatVisibility() {
    splatVisible = !splatVisible;
    if (splatEnt) splatEnt.enabled = splatVisible;
    updateToolbarUI();
  }

  function toggleDroneMode(forcedState) {
    isDrone = forcedState !== undefined ? forcedState : !isDrone;
    if (isDrone) {
      if (playerEnt) {
        const p = playerEnt.getPosition();
        if (Math.abs(p.x) < 0.01 && Math.abs(p.z) < 0.01 && window.__spawn) {
          dronePos.set(window.__spawn.x, groundHF(window.__spawn.x, window.__spawn.z) + CHAR_H, window.__spawn.z);
        } else {
          dronePos.set(p.x, p.y + 0.8 * CHAR_SCALE, p.z);
        }
        playerEnt.enabled = false;
        if (playerRb) {
          playerRb.type = "kinematic";
          playerRb.linearVelocity = new Vec3(0, 0, 0);
        }
      } else if (window.__spawn) {
        dronePos.set(window.__spawn.x, groundHF(window.__spawn.x, window.__spawn.z) + CHAR_H, window.__spawn.z);
      }
    } else {
      if (playerEnt) {
        playerEnt.enabled = true;
        if (playerRb) {
          playerRb.type = "dynamic";
          const hit = groundProbe(dronePos.x, dronePos.z);
          const targetY = hit ? hit.point.y + CHAR_H * 0.86 : (groundHF(dronePos.x, dronePos.z) + CHAR_H * 0.86);
          playerEnt.rigidbody.teleport(dronePos.x, targetY, dronePos.z);
          playerRb.linearVelocity = new Vec3(0, 0, 0);
        }
      }
    }
    updateToolbarUI();
  }

  function toggleCamerasLayer() {
    showCameras = !showCameras;
    if (camerasEntity) camerasEntity.enabled = showCameras;
    if (trajectoryEntity) trajectoryEntity.enabled = showCameras;
    if (selectedCamEntity && !showCameras) selectedCamEntity.enabled = false;
    else if (selectedCamEntity && showCameras && selectedCamIdx >= 0) selectedCamEntity.enabled = true;
    updateToolbarUI();
  }

  function togglePointsLayer() {
    showPoints = !showPoints;
    if (sparsePointsEntity) sparsePointsEntity.enabled = showPoints;
    updateToolbarUI();
  }

  function toggleCollider() {
    showCollider = !showCollider;
    colliderEnt?.findComponents("render").forEach((r) => { r.enabled = showCollider; });
    if (objectsEnt) objectsEnt.enabled = showCollider;
    updateToolbarUI();
  }

  function toggleCoverageLayer() {
    showCoverage = !showCoverage;
    if (coverageGridEntity) coverageGridEntity.enabled = showCoverage;
    const covPanel = document.getElementById("cov-panel");
    if (covPanel) {
      covPanel.style.display = showCoverage ? "flex" : "none";
      if (showCoverage && allCoverageGrid) {
        const statsEl = document.getElementById("cov-stats");
        const adviceEl = document.getElementById("cov-advice");
        // "Not measurable" and "0% observed" are different verdicts and must not
        // share a colour or a sentence. The old grid shipped `covered_pct: 0` for
        // a rocks take that was 72/72 registered, because a 4.5 m reach could not
        // span a 20 m voxel: the panel then told the operator to re-fly good
        // footage. check_coverage.py now reports `measurable: false` with a
        // reason instead of a number, so the number is only ever painted green
        // when there was a number to paint.
        const measurable = allCoverageGrid.measurable !== false;
        if (statsEl && !measurable) {
          // The reason is free text off the wire, so it goes in as a text node
          // rather than being interpolated into markup.
          statsEl.innerHTML = `
            <div class="stat-row"><span>Coverage:</span> <b style="color:#a0aec0">NOT MEASURABLE</b></div>
            <div class="stat-row"><span>Why:</span> <b id="cov-unmeasurable-reason"></b></div>
          `;
          const why = document.getElementById("cov-unmeasurable-reason");
          if (why) {
            why.textContent = allCoverageGrid.unmeasurable_reason
              || "geometry did not support the measurement";
          }
        } else if (statsEl) {
          statsEl.innerHTML = `
            <div class="stat-row"><span>Total Volume Voxels:</span> <b>${allCoverageGrid.total_voxels}</b></div>
            <div class="stat-row"><span>Observed (>=3 views):</span> <b style="color:#68d391">${allCoverageGrid.covered_pct}%</b></div>
            <div class="stat-row"><span>Marginal (1-2 views):</span> <b style="color:#f6ad55">${allCoverageGrid.marginal_pct}%</b></div>
            <div class="stat-row"><span>Unobserved (Missing):</span> <b style="color:#fc8181">${allCoverageGrid.unobserved_pct}%</b></div>
          `;
        }
        if (adviceEl && allCoverageGrid.advice) {
          // Same treatment as the reason above: one div per line, text filled in
          // through textContent so nothing off the wire is parsed as markup.
          adviceEl.innerHTML = "";
          for (const a of allCoverageGrid.advice) {
            const row = document.createElement("div");
            row.textContent = `\u2022 ${a}`;
            adviceEl.appendChild(row);
          }
        }
      }
    }
    updateToolbarUI();
  }

  async function toggleSplatModel() {
    const target = currentPly === "scene.full.ply" ? "scene.ply" : "scene.full.ply";
    await setSplatModel(target);
  }

  /** What the splat toggle/HUD calls the cloud on screen, with the count the asset really holds. */
  function splatCountLabel() {
    if (!splatVisible) return "OFF (Hidden)";
    const which = currentPly === "scene.full.ply" ? "Full" : "Clean";
    const n = splatEnt?.gsplat?.asset?.resource?.numSplats;
    return n ? `${which} (${Math.round(n / 1000)}k)` : which;
  }

  async function setSplatModel(filename) {
    if (currentPly === filename && splatEnt?.gsplat?.asset) return;
    setLoad(`Loading 3D Gaussian Splat (${filename})…`);
    if (loadEl) loadEl.style.opacity = "1";
    try {
      const newAsset = new Asset("splat_" + Date.now(), "gsplat", { url: `${ASSET}/${filename}` });
      await new Promise((resolve, reject) => {
        new AssetListLoader([newAsset], applicationRef.assets).load((err) => (err ? reject(err) : resolve()));
      });
      splatEnt.gsplat.asset = newAsset;
      currentPly = filename;
    } catch (e) {
      console.error("Failed to load splat:", e);
    } finally {
      setLoad("");
      if (loadEl) { loadEl.style.opacity = "0"; loadEl.style.pointerEvents = "none"; }
      updateToolbarUI();
    }
  }

  let isPlayingSeq = false;
  let seqTimer = null;

  function toggleSequencePlayback() {
    isPlayingSeq = !isPlayingSeq;
    const btn = document.getElementById("insp-play-seq");
    if (btn) {
      btn.textContent = isPlayingSeq ? "⏸ Pause Flight Sequence" : "🎬 Play Drone Flight Sequence";
      btn.style.background = isPlayingSeq ? "rgba(229, 62, 62, 0.35)" : "rgba(72,187,120,0.25)";
      btn.style.color = isPlayingSeq ? "#feb2b2" : "#68d391";
    }
    if (isPlayingSeq) {
      if (selectedCamIdx < 0) selectedCamIdx = 0;
      seqTimer = setInterval(() => {
        let nextIdx = (selectedCamIdx + 1) % allCameras.length;
        inspectCamera(nextIdx);
        snapToSelectedCamera();
      }, 150);
    } else {
      clearInterval(seqTimer);
    }
  }

  function inspectCamera(idx) {
    if (!allCameras.length) return;
    if (idx < 0) idx = allCameras.length - 1;
    if (idx >= allCameras.length) idx = 0;
    selectedCamIdx = idx;

    const cam = allCameras[idx];
    updateSelectedCameraMesh(device, cam);
    if (EMBED) {
      selectedCamEntity.enabled = showCameras;
      return; // Never build legacy inspector HTML or image URLs from embedded scene metadata.
    }

    const inspModal = document.getElementById("cam-inspector");
    const inspTitle = document.getElementById("insp-title");
    const inspImg = document.getElementById("insp-img");
    const inspDetails = document.getElementById("insp-details");
    const inspSlider = document.getElementById("insp-slider");
    const inspFrameNum = document.getElementById("insp-frame-num");

    if (inspModal) {
      inspModal.style.display = "flex";
      if (inspTitle) inspTitle.textContent = `📷 Frame #${idx + 1}/${allCameras.length}: ${cam.name}`;
      if (inspFrameNum) inspFrameNum.textContent = `${idx + 1} / ${allCameras.length}`;
      if (inspSlider) {
        inspSlider.max = allCameras.length - 1;
        inspSlider.value = idx;
      }
      if (inspImg) {
        inspImg.src = `${WORK_DIR}/frames_train/${cam.name}`;
        inspImg.onerror = () => { inspImg.src = `${WORK_DIR}/frames_undist/${cam.name.replace("/", "__")}`; };
      }
      if (inspDetails) {
        inspDetails.innerHTML = `
          <div><b>Position:</b> (${cam.pos[0].toFixed(2)}, ${cam.pos[1].toFixed(2)}, ${cam.pos[2].toFixed(2)}) m</div>
          <div><b>Timestamp:</b> ${cam.t_sec !== null ? cam.t_sec.toFixed(2) + " s" : "N/A"}</div>
          <div><b>FOV:</b> ${cam.fov_x_deg}° x ${cam.fov_y_deg}° (${cam.width}x${cam.height})</div>
          <div><b>Intrinsics:</b> fx=${cam.fx}, fy=${cam.fy}</div>
        `;
      }
    }
  }

  function snapToSelectedCamera() {
    if (selectedCamIdx < 0 || selectedCamIdx >= allCameras.length) return;
    const cam = allCameras[selectedCamIdx];

    if (!isDrone) toggleDroneMode(true);
    dronePos.set(cam.pos[0], cam.pos[1], cam.pos[2]);

    const fw = cam.forward;
    P.yaw = Math.atan2(-fw[0], -fw[2]);
    P.pitch = Math.asin(Math.max(-0.999, Math.min(0.999, fw[1])));

    cameraEnt.setPosition(cam.pos[0], cam.pos[1], cam.pos[2]);
    cameraEnt.lookAt(new Vec3(cam.pos[0] + fw[0] * 5, cam.pos[1] + fw[1] * 5, cam.pos[2] + fw[2] * 5));
  }

  // ---------------- embedded workspace ----------------
  const snapshotCapture = new SnapshotCapture((type, data) => workspace?.emit(type, data));
  let pickPoints = [], pickLines = [];
  const pickColor = new Color(1, 0.78, 0.24);
  let markerSize = 0.02;

  // Persistent measurement geometry + floating labels, driven by the host.
  let renderedMeasurements = [];
  let selectedMeasureId = null;
  let showMeasureLabels = true;
  let measurementKind = null, measurementUnit = "m";
  let pickingRequested = false;
  let previewPoint = null;   // live cursor world point while picking
  const previewStream = new PreviewStream((type, data) => workspace?.emit(type, data));
  function setPreviewPoint(point) {
    previewPoint = finitePoint(point) ? [...point] : null;
    previewStream.update(previewPoint);
  }
  function currentLiveMeasurement() {
    return liveMeasurement(measurementKind, pickPoints, previewPoint, workspacePick, measurementUnit);
  }
  const labelLayer = EMBED ? Object.assign(document.createElement("div"), { id: "measure-labels" }) : null;
  const labelEls = new Map();
  if (labelLayer) {
    Object.assign(labelLayer.style, { position: "fixed", overflow: "hidden", pointerEvents: "none", zIndex: "5", font: "500 12px system-ui, sans-serif" });
    document.body.appendChild(labelLayer);
  }
  function segmentsFor(m) {
    const r = measureRenderable(m), out = [];
    for (let i = 0; i + 1 < r.points.length; i++) out.push(r.points[i], r.points[i + 1]);
    return { line: out.map(p => new Vec3(p[0], p[1], p[2])), anchor: measureAnchor(m) };
  }
  function drawMeasurementGeometry() {
    for (const m of renderedMeasurements) {
      const { line } = segmentsFor(m);
      if (line.length < 2) continue;
      const c = measureColor(m.kind);
      const strong = m.id === selectedMeasureId;
      application.drawLines(line, new Color(c[0], c[1], c[2], strong ? 1 : 0.9), false);
    }
    // Live rubber-band from the last pick to the cursor while placing.
    const live = currentLiveMeasurement();
    if (live) {
      const { line } = segmentsFor(live);
      if (line.length) application.drawLines(line, new Color(...measureColor(live.kind)), false);
    }
    // The piece the editor has grabbed, outlined so it is obvious what moves.
    if (workspaceEdit && selectedMeasureId) {
      const box = placementBox(placementById(selectedMeasureId));
      if (box) {
        const c = box.corners, edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [2, 4], [4, 6], [6, 0], [1, 3], [3, 5], [5, 7], [7, 1]];
        const outline = [];
        for (const [a, b] of edges) outline.push(new Vec3(...c[a]), new Vec3(...c[b]));
        application.drawLines(outline, new Color(1, 0.78, 0.24, 1), false);
      }
    }
  }
  function updateMeasureLabels() {
    if (!labelLayer) return;
    const cam = cameraEnt.camera, rect = canvas.getBoundingClientRect(), device = application.graphicsDevice;
    Object.assign(labelLayer.style, { left: rect.left + "px", top: rect.top + "px", width: rect.width + "px", height: rect.height + "px" });
    const labels = [], seen = new Set(), candidates = [];
    if (showMeasureLabels) {
      for (const [index, m] of renderedMeasurements.entries()) {
        const selected = m.id === selectedMeasureId;
        labels.push({ id: `measure:${m.id ?? index}`, text: formatMeasureLabel(m), anchor: measureAnchor(m),
          priority: selected ? 90 : 10, border: selected ? "rgb(250,170,60)" : "rgba(120,150,180,.35)" });
      }
      for (const p of renderedPlacements) {
        if (p.id !== selectedMeasureId) continue;
        const box = placementBox(p), c = placementColor(p);
        labels.push({ id: `placement:${p.id}`, text: placementLabel(p),
          anchor: [box.center[0], box.center[1] + box.halfExtents[1], box.center[2]], priority: 95,
          border: `rgb(${c[0] * 255 | 0},${c[1] * 255 | 0},${c[2] * 255 | 0})` });
      }
      for (const label of currentLiveMeasurement()?.labels || []) {
        labels.push({ ...label, border: "rgb(250,170,60)" });
      }
    }
    const eye = cameraEnt.getPosition(), forward = cameraEnt.forward;
    for (const label of labels) {
      if (!finitePoint(label.anchor)) continue;
      const world = new Vec3(...label.anchor);
      // The component wrapper uses CSS clientRect sizes in this engine version.
      // Project explicitly into the drawing buffer, then scale each axis to CSS.
      const s = cam.camera.worldToScreen(world, device.width, device.height, new Vec3());
      s.z = (world.x - eye.x) * forward.x + (world.y - eye.y) * forward.y + (world.z - eye.z) * forward.z;
      const xy = projectedCssPoint(s, rect, device.width, device.height);
      if (!xy) continue;
      let el = labelEls.get(label.id);
      if (!el) {
        el = document.createElement("div");
        el.style.cssText = "position:absolute;box-sizing:border-box;padding:2px 7px;border-radius:5px;background:rgba(12,16,22,.9);color:#eaf1f8;border:1px solid;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;backdrop-filter:blur(4px)";
        el.dataset.labelId = label.id;
        labelLayer.appendChild(el); labelEls.set(label.id, el);
      }
      el.textContent = label.text;
      el.style.borderColor = label.border;
      el.style.maxWidth = Math.max(0, rect.width - 8) + "px";
      el.style.display = "block"; el.style.visibility = "hidden";
      candidates.push({ id: label.id, x: xy[0], y: xy[1], width: el.offsetWidth, height: el.offsetHeight, priority: label.priority });
      seen.add(label.id);
    }
    for (const label of layoutLabels(candidates, rect.width, rect.height)) {
      const el = labelEls.get(label.id);
      el.style.left = label.left + "px"; el.style.top = label.top + "px";
      el.style.visibility = "visible";
    }
    for (const [id, el] of labelEls) if (!seen.has(id)) { el.remove(); labelEls.delete(id); }
  }
  function setRenderedMeasurements(list) {
    renderedMeasurements = Array.isArray(list) ? list.filter(m => m && Array.isArray(m.points)) : [];
  }

  // Recognizable, opaque component meshes; the walk character still collides
  // with conservative enclosing boxes on one compound static body. Each item is
  // its own entity so the editor can move/resize one piece without rebuilding
  // the whole layout, and so a click can identify which piece was grabbed.
  let renderedPlacements = [];
  let placedRoot = null, placedMat = null, workspaceEdit = false;
  // Planning layer (proposal roads, buildings, zones, objects, demolitions): the backend
  // computes every mesh; this only draws them, picks them and hides demolished splats.
  let planState = { features: [], clips: [], view: "proposal", selected: null };
  let planRoot = null, planSelect = false, planClipKey = "", planClipMaterial = null;
  const planEntities = new Map();   // feature id -> { entity, meshes, sig }
  const planMaterials = {};
  function planMaterial(transparent) {
    const key = transparent ? "glass" : "solid";
    if (planMaterials[key]) return planMaterials[key];
    const m = new StandardMaterial();
    m.useLighting = false;
    m.useSkybox = false;
    m.diffuse.set(0, 0, 0);
    m.emissive.set(1, 1, 1);
    m.emissiveVertexColor = true;
    m.emissiveMapVertexColor = true;
    m.opacityVertexColor = transparent;
    m.blendType = transparent ? BLEND_NORMAL : BLEND_NONE;
    m.depthWrite = !transparent;
    m.depthTest = true;
    m.cull = 0;                     // proposals are seen from above and below: draw both sides
    m.update();
    planMaterials[key] = m;
    return m;
  }
  function destroyPlanEntity(id) {
    const item = planEntities.get(id);
    if (!item) return;
    for (const mesh of item.meshes) mesh.destroy();
    item.entity.destroy();
    planEntities.delete(id);
  }
  function rebuildPlan() {
    if (!planRoot) { planRoot = new Entity("plan"); application.root.addChild(planRoot); }
    planRoot.enabled = planState.view === "proposal";
    const live = new Set(planState.features.map(f => f.id));
    for (const id of [...planEntities.keys()]) if (!live.has(id)) destroyPlanEntity(id);
    for (const f of planState.features) {
      const selected = f.id === planState.selected;
      const sig = planSignature(f, selected);
      const existing = planEntities.get(f.id);
      if (existing && existing.sig === sig) continue;
      if (existing) destroyPlanEntity(f.id);
      const entity = new Entity(`plan:${f.id}`);
      entity.addComponent("render", { meshInstances: [], castShadows: false, receiveShadows: false });
      const meshes = [], instances = [], glass = [];
      for (const part of f.meshes) {
        const geometry = planGeometry(part, { selected });
        const mesh = new Mesh(application.graphicsDevice);
        mesh.setPositions(geometry.positions);
        mesh.setNormals(geometry.normals);
        mesh.setColors32(geometry.colors);
        mesh.setIndices(geometry.indices);
        mesh.update(PRIMITIVE_TRIANGLES);
        meshes.push(mesh);
        (geometry.transparent ? glass : instances).push(new MeshInstance(mesh, planMaterial(geometry.transparent)));
      }
      entity.render.meshInstances = instances;
      if (glass.length) {
        const overlay = new Entity(`plan-glass:${f.id}`);
        overlay.addComponent("render", { meshInstances: glass, layers: [planOverlayLayer.id], castShadows: false, receiveShadows: false });
        entity.addChild(overlay);
      }
      planRoot.addChild(entity);
      planEntities.set(f.id, { entity, meshes, sig });
    }
    applyPlanClips(planState.view === "proposal" ? planState.clips : []);
    applyColliderClips(planState.view === "proposal" ? planState.clips : []);
    rebuildPlanColliders();
  }
  /**
   * Hide splats inside demolished volumes. The engine's unified splat renderer (the
   * default) applies a per-splat "work-buffer modifier" when it copies splats into its
   * work buffer; its modifySplatColor gets the world-space centre and alpha 0 drops the
   * splat. A non-unified component takes the same code as the material's gsplatModifyVS.
   */
  function applyPlanClips(clips) {
    const g = splatEnt?.gsplat;
    if (!g) return;
    const key = JSON.stringify(clips);
    const uniforms = clipUniforms(clips);
    if (g.unified !== false && typeof g.setWorkBufferModifier === "function") {
      if (key === planClipKey && planClipMaterial === g) return;
      g.setWorkBufferModifier(clips.length ? { glsl: clipShader("glsl"), wgsl: clipShader("wgsl") } : null);
      uniforms.forEach(([a, b], i) => {
        g.setParameter(`uPlanClipA${i}`, new Float32Array(a));
        g.setParameter(`uPlanClipB${i}`, new Float32Array(b));
      });
      planClipKey = key;
      planClipMaterial = g;
      return;
    }
    const material = g.material;
    if (!material) return;
    if (key !== planClipKey || material !== planClipMaterial) {
      const wgsl = application.graphicsDevice.isWebGPU;
      const chunks = material.getShaderChunks(wgsl ? "wgsl" : "glsl");
      if (clips.length) chunks.set("gsplatModifyVS", clipShader(wgsl ? "wgsl" : "glsl"));
      else chunks.delete("gsplatModifyVS");
      material.update();
      planClipKey = key;
      planClipMaterial = material;
    }
    uniforms.forEach(([a, b], i) => {
      material.setParameter(`uPlanClipA${i}`, a);
      material.setParameter(`uPlanClipB${i}`, b);
    });
  }
  function planPickAt(clientX, clientY) {
    const xy = canvasPoint(clientX, clientY, canvas.getBoundingClientRect());
    if (!xy || planState.view !== "proposal") { workspace.emit("plan-pick", { id: null }); return; }
    const camera = cameraEnt.camera;
    const start = camera.screenToWorld(xy[0], xy[1], camera.nearClip);
    const end = camera.screenToWorld(xy[0], xy[1], camera.farClip);
    const hit = rayPick(planState.features, [start.x, start.y, start.z], [end.x - start.x, end.y - start.y, end.z - start.z]);
    workspace.emit("plan-pick", { id: hit ? hit.id : null });
  }

  // ---- Editing the selected feature in 3D. Handles are DOM dots over the canvas (always
  // on top, easy to grab); a drag runs on the feature's base plane, previews locally
  // (outline, plus a live move/turn of the drawn mesh) and hands the new outline to the
  // host on release. The backend then re-derives the geometry, as for every other edit.
  const handleLayer = EMBED ? Object.assign(document.createElement("div"), { id: "plan-handles" }) : null;
  if (handleLayer) {
    Object.assign(handleLayer.style, { position: "fixed", overflow: "hidden", pointerEvents: "none", zIndex: "6" });
    document.body.appendChild(handleLayer);
  }
  const handleEls = new Map();
  let planDrag = null, planPreview = null, planPreviewTimer = null;
  const HANDLE_CSS = {
    vertex: "width:13px;height:13px;border-radius:50%;background:#fff;border:2px solid #f5a524;cursor:move",
    mid: "width:10px;height:10px;border-radius:50%;background:rgba(245,165,36,.4);border:1.5px solid #f5a524;cursor:copy",
    move: "width:22px;height:22px;border-radius:6px;background:#f5a524;border:2px solid #fff;cursor:grab;box-shadow:0 1px 5px rgba(0,0,0,.55)",
    rotate: "width:17px;height:17px;border-radius:50%;background:#fff;border:3px solid #4aa3ff;cursor:alias;box-shadow:0 1px 5px rgba(0,0,0,.55)",
  };
  const HANDLE_TITLE = { vertex: "Drag to move this corner · double-click to remove it", mid: "Drag to add a corner here",
    move: "Drag to move · Shift snaps to 1 m", rotate: "Drag to turn · Shift snaps to 15°" };
  function editPoint(clientX, clientY) {
    const xy = canvasPoint(clientX, clientY, canvas.getBoundingClientRect());
    const edit = planState.edit;
    if (!xy || !edit) return null;
    const camera = cameraEnt.camera;
    const a = camera.screenToWorld(xy[0], xy[1], camera.nearClip), b = camera.screenToWorld(xy[0], xy[1], camera.farClip);
    return planeHit([a.x, a.y, a.z], [b.x - a.x, b.y - a.y, b.z - a.z], edit.y);
  }
  function resetPlanTransforms() {
    for (const { entity } of planEntities.values()) { entity.setLocalPosition(0, 0, 0); entity.setLocalEulerAngles(0, 0, 0); }
  }
  function clearPlanPreview() {
    planPreview = null;
    if (planPreviewTimer) { clearTimeout(planPreviewTimer); planPreviewTimer = null; }
    resetPlanTransforms();
  }
  function beginPlanDrag(e, handle) {
    const edit = planState.edit;
    if (!edit || e.button !== 0 || planDrag) return false;
    const from = editPoint(e.clientX, e.clientY) || [...handle.xz];
    clearPlanPreview();
    planDrag = { handle, from, pointerId: e.pointerId, moved: false, startX: e.clientX, startY: e.clientY, result: null, target: e.currentTarget || canvas };
    planDrag.target.setPointerCapture?.(e.pointerId);
    return true;
  }
  function movePlanDrag(e) {
    if (!planDrag || planDrag.pointerId !== e.pointerId) return false;
    if (Math.hypot(e.clientX - planDrag.startX, e.clientY - planDrag.startY) > 3) planDrag.moved = true;
    if (!planDrag.moved) return true;
    const to = editPoint(e.clientX, e.clientY);
    if (!to) return true;
    const edit = planState.edit;
    planDrag.result = dragShape(edit, planDrag.handle, planDrag.from, to, e.shiftKey ? { grid: 1, snapDeg: 15 } : {});
    planPreview = planDrag.result.points;
    // Move and turn are rigid, so the drawn mesh can follow exactly without a round-trip.
    const item = planEntities.get(edit.id);
    if (item && planDrag.handle.kind === "move") {
      const [x0, z0] = edit.points[0], [x1, z1] = planPreview[0];
      item.entity.setLocalPosition(x1 - x0, 0, z1 - z0);
    } else if (item && planDrag.handle.kind === "rotate") {
      const a = planDrag.result.rotation_deg * Math.PI / 180;
      const cx = edit.points.reduce((s, p) => s + p[0], 0) / edit.points.length;
      const cz = edit.points.reduce((s, p) => s + p[1], 0) / edit.points.length;
      item.entity.setLocalEulerAngles(0, -planDrag.result.rotation_deg, 0);
      item.entity.setLocalPosition(cx - (cx * Math.cos(a) - cz * Math.sin(a)), 0, cz - (cx * Math.sin(a) + cz * Math.cos(a)));
    }
    return true;
  }
  function endPlanDrag(e, cancel = false) {
    if (!planDrag || (e && planDrag.pointerId !== e.pointerId)) return false;
    const { result, moved, target, pointerId, handle } = planDrag;
    if (target.hasPointerCapture?.(pointerId)) target.releasePointerCapture(pointerId);
    planDrag = null;
    if (cancel || !moved || !result) { clearPlanPreview(); return true; }
    workspace.emit("plan-edit", { id: planState.edit.id, points: result.points, rotation_deg: result.rotation_deg, handle: handle.kind });
    // The host answers with a new plan (or an error); either way stop previewing soon.
    planPreviewTimer = setTimeout(clearPlanPreview, 8000);
    return true;
  }
  function handleElement(key) {
    let el = handleEls.get(key);
    if (el) return el;
    el = document.createElement("div");
    el.dataset.handle = key;
    el.addEventListener("pointerdown", e => { if (beginPlanDrag(e, el._handle)) { e.preventDefault(); e.stopPropagation(); } });
    el.addEventListener("pointermove", e => { if (movePlanDrag(e)) e.preventDefault(); });
    el.addEventListener("pointerup", e => endPlanDrag(e));
    el.addEventListener("pointercancel", e => endPlanDrag(e, true));
    el.addEventListener("dblclick", e => {
      const edit = planState.edit, h = el._handle;
      if (!edit || h.kind !== "vertex") return;
      e.preventDefault();
      const points = removeVertex(edit, h.index);
      if (points) workspace.emit("plan-edit", { id: edit.id, points, rotation_deg: 0, handle: "remove" });
      else workspace.emit("error", { message: `An outline needs at least ${edit.min} corners.` });
    });
    el.addEventListener("wheel", e => { e.preventDefault(); canvas.dispatchEvent(new WheelEvent("wheel", e)); }, { passive: false });
    handleLayer.appendChild(el);
    handleEls.set(key, el);
    return el;
  }
  function updatePlanHandles() {
    if (!handleLayer) return;
    const edit = planState.edit;
    const live = new Set();
    if (edit && planSelect && planState.view === "proposal") {
      const rect = canvas.getBoundingClientRect(), device = application.graphicsDevice, cam = cameraEnt.camera;
      const eye = cameraEnt.getPosition(), forward = cameraEnt.forward;
      Object.assign(handleLayer.style, { left: rect.left + "px", top: rect.top + "px", width: rect.width + "px", height: rect.height + "px" });
      const shown = planPreview ? { ...edit, points: planPreview } : edit;
      for (const h of planHandles(shown)) {
        if (planDrag && h.kind === "mid" && planDrag.handle.kind !== "mid") continue;
        const key = `${h.kind}:${h.index}`;
        const world = new Vec3(h.xz[0], edit.y + 0.15, h.xz[1]);
        const s = cam.camera.worldToScreen(world, device.width, device.height, new Vec3());
        s.z = (world.x - eye.x) * forward.x + (world.y - eye.y) * forward.y + (world.z - eye.z) * forward.z;
        const xy = projectedCssPoint(s, rect, device.width, device.height);
        if (!xy) continue;
        const el = handleElement(key);
        if (el._kind !== h.kind) { el.style.cssText = `position:absolute;box-sizing:border-box;transform:translate(-50%,-50%);pointer-events:auto;touch-action:none;${HANDLE_CSS[h.kind]}`; el.title = HANDLE_TITLE[h.kind]; el._kind = h.kind; }
        if (!planDrag || planDrag.target !== el) el._handle = h;
        el.style.left = xy[0] + "px"; el.style.top = xy[1] + "px";
        el.style.display = "block";
        live.add(key);
      }
    }
    for (const [key, el] of handleEls) if (!live.has(key)) {
      if (planDrag && planDrag.target === el) continue;
      el.remove(); handleEls.delete(key);
    }
  }
  function drawPlanEdit() {
    const edit = planState.edit;
    if (!edit || !planSelect || planState.view !== "proposal") return;
    const pts = planPreview || edit.points;
    if (pts.length < 2) return;
    const y = edit.y + 0.12, line = [];
    for (let i = 0; i < (edit.closed ? pts.length : pts.length - 1); i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      line.push(new Vec3(a[0], y, a[1]), new Vec3(b[0], y, b[1]));
    }
    application.drawLines(line, planPreview ? new Color(1, 0.72, 0.15, 1) : new Color(1, 0.8, 0.35, 0.8), false);
  }

  // ---- Walk physics for the scheme: proposed walls and objects block the walker, and
  // demolished structures are cut out of the scanned collider (the ground they stood on
  // stays). Only in the Proposed view; Existing restores the scan's own collider.
  let planColliderRoot = null, planColliderKey = "", colliderClipKey = "[]";
  const colliderOriginal = new Map();
  function rebuildPlanColliders() {
    const boxes = planState.view === "proposal" ? planColliders(planState.features) : [];
    const key = JSON.stringify(boxes);
    if (key === planColliderKey) return;
    planColliderKey = key;
    planColliderRoot?.destroy();
    planColliderRoot = null;
    if (!boxes.length) return;
    // Assemble the whole compound off-scene, then insert it once: a static compound that
    // enters the world with one child and grows afterwards keeps that first child's bounds
    // in the broadphase, and rays (and the walker) pass straight through every later box.
    const root = new Entity("planColliders");
    for (const b of boxes) {
      const e = new Entity();
      e.setLocalPosition(b.center[0], b.center[1], b.center[2]);
      e.setLocalEulerAngles(0, b.yaw, 0);
      e.addComponent("collision", { type: "box", halfExtents: new Vec3(b.halfExtents[0], b.halfExtents[1], b.halfExtents[2]) });
      root.addChild(e);
    }
    root.addComponent("collision", { type: "compound" });
    root.addComponent("rigidbody", { type: "static", friction: 0.6, restitution: 0 });
    application.root.addChild(root);
    planColliderRoot = root;
    window.__planColliders = boxes.length;
  }
  function groundPatch(clip, step = 1) {
    // A walkable floor over the demolished box, at the measured ground height.
    const cs = Math.cos(clip.angle), sn = Math.sin(clip.angle);
    const nu = Math.max(1, Math.ceil(2 * clip.hx / step)), nv = Math.max(1, Math.ceil(2 * clip.hz / step));
    const pos = [], idx = [];
    for (let j = 0; j <= nv; j++) for (let i = 0; i <= nu; i++) {
      const u = -clip.hx + 2 * clip.hx * i / nu, v = -clip.hz + 2 * clip.hz * j / nv;
      const x = clip.cx + u * cs - v * sn, z = clip.cz + u * sn + v * cs;
      pos.push(x, groundHF(x, z), z);
    }
    for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
      const a = j * (nu + 1) + i, b = a + 1, c = a + nu + 1, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
    return { pos, idx };
  }
  function applyColliderClips(clips) {
    const key = JSON.stringify(clips);
    if (key === colliderClipKey) return;
    colliderClipKey = key;
    let removed = 0;
    for (const entity of collisionMeshEntities) {
      if (!entity.collision) continue;
      let orig = colliderOriginal.get(entity);
      if (!orig) {
        const meshes = entity.render.meshInstances.map(mi => mi.mesh);
        const world = entity.getWorldTransform();
        orig = { meshes, data: meshes.map(mesh => {
          const pos = [], idx = [];
          mesh.getPositions(pos); mesh.getIndices(idx);
          const wpos = new Float32Array(pos.length), v = new Vec3();
          for (let i = 0; i < pos.length; i += 3) { world.transformPoint(v.set(pos[i], pos[i + 1], pos[i + 2]), v); wpos[i] = v.x; wpos[i + 1] = v.y; wpos[i + 2] = v.z; }
          return { wpos, idx };
        }) };
        colliderOriginal.set(entity, orig);
      }
      let meshes = orig.meshes;
      if (clips.length) {
        // Rebuilt in world coordinates, so the collider entity must carry no transform of its own.
        const inverse = entity.getWorldTransform().clone().invert();
        const toLocal = (arr) => { const v = new Vec3(), out = new Float32Array(arr.length); for (let i = 0; i < arr.length; i += 3) { inverse.transformPoint(v.set(arr[i], arr[i + 1], arr[i + 2]), v); out[i] = v.x; out[i + 1] = v.y; out[i + 2] = v.z; } return out; };
        meshes = orig.data.map(({ wpos, idx }) => {
          const keep = clipTriangles(wpos, idx, clips, groundHF);
          removed += (idx.length - keep.length) / 3;
          const mesh = new Mesh(application.graphicsDevice);
          mesh.setPositions(toLocal(wpos)); mesh.setIndices(keep.length ? keep : [0, 0, 0]); mesh.update(PRIMITIVE_TRIANGLES);
          return mesh;
        });
        for (const clip of clips) {
          const { pos, idx } = groundPatch(clip);
          const mesh = new Mesh(application.graphicsDevice);
          mesh.setPositions(toLocal(pos)); mesh.setIndices(idx); mesh.update(PRIMITIVE_TRIANGLES);
          meshes = [...meshes, mesh];
        }
      }
      entity.collision.renderAsset = null;
      entity.collision.render = { meshes };
    }
    window.__colliderClipped = removed;
  }

  // ---- Camera sync for side-by-side / swipe compare: the host relays one viewer's
  // orbit to the other. A remote orbit is applied without echoing it back.
  let cameraFollow = false, cameraEmitAt = 0, cameraTimer = null, remoteOrbit = false;
  let remotePose = false, remotePoseUntil = 0, poseEmitAt = 0;
  /** Fly/walk compare: broadcast the eye (position + forward) instead of an orbit. */
  function broadcastPose() {
    if (!cameraFollow || workspaceMode === "orbit" || !workspace.ready) return;
    const now = performance.now();
    if (remotePose && now < remotePoseUntil) return;      // being driven: do not echo back
    remotePose = false;
    if (now - poseEmitAt < 50) return;
    poseEmitAt = now;
    const e = cameraEnt.getPosition(), f = cameraEnt.forward;
    workspace.emit("camera", { pose: { eye: [e.x, e.y, e.z], forward: [f.x, f.y, f.z], mode: workspaceMode } });
  }
  function broadcastCamera() {
    if (!cameraFollow || remoteOrbit || !workspaceOrbit || !workspace.ready) return;
    const now = performance.now();
    if (now - cameraEmitAt < 30) {
      if (!cameraTimer) cameraTimer = setTimeout(() => { cameraTimer = null; broadcastCamera(); }, 35);
      return;
    }
    cameraEmitAt = now;
    const o = workspaceOrbit;
    workspace.emit("camera", { orbit: { target: [...o.target], distance: o.distance, yaw: o.yaw, pitch: o.pitch } });
  }
  const placedItems = new Map();          // id -> { holder, collider, mesh, sig }
  const placedByCollider = new Map();     // collider entity -> id
  function placementById(id) { return renderedPlacements.find(p => p.id === id) || null; }
  function placementMaterial() {
    if (placedMat) return placedMat;
    const m = new StandardMaterial();
    m.useLighting = false;
    m.useSkybox = false;
    m.diffuse.set(0, 0, 0);
    m.emissive.set(1, 1, 1);
    m.emissiveMapVertexColor = true; // use the helper's baked, directional face shading
    m.blendType = BLEND_NONE;
    m.opacity = 1;
    m.depthTest = true;
    m.depthWrite = true;
    m.cull = CULLFACE_BACK;
    m.update();
    placedMat = m;
    return m;
  }
  function placementGeometryMesh(p) {
    const geometry = furnitureGeometry(p);
    if (!geometry || !geometry.positions.every(v => Number.isFinite(Math.fround(v)))) return null;
    const mesh = new Mesh(application.graphicsDevice);
    mesh.setPositions(geometry.positions);
    mesh.setNormals(geometry.normals);
    mesh.setColors32(new Uint8Array(geometry.colors)); // four byte RGBA components per vertex
    mesh.setIndices(geometry.indices);
    mesh.update(PRIMITIVE_TRIANGLES);
    return mesh;
  }
  // Imported glTF/GLB items render their real geometry, aligned to the same
  // conservative box the collider uses, so what you see is what the character
  // bumps into. The plain box mesh stays as a fallback while a model loads and
  // if the load fails, so a placed item is never blank.
  const modelAssets = new Map();          // file -> Promise<Asset|null>, cached per scene
  function modelAsset(file) {
    let pending = modelAssets.get(file);
    if (!pending) {
      pending = new Promise(resolve => {
        const asset = new Asset(file, "container", { url: `${WORK_DIR}/models/${file}` });
        application.assets.add(asset);
        asset.on("load", () => resolve(asset));
        asset.on("error", () => resolve(null));
        application.assets.load(asset);
      });
      modelAssets.set(file, pending);
    }
    return pending;
  }
  function modelTint(p) {
    const c = placementColor(p);
    const m = new StandardMaterial();
    m.useLighting = false;
    m.diffuse.set(0, 0, 0);
    m.emissive.set(c[0], c[1], c[2]);      // fit colour: green fits, red does not, blue unknown
    m.opacity = 1;
    m.blendType = BLEND_NONE;
    m.depthTest = true;
    m.depthWrite = true;
    m.cull = CULLFACE_BACK;
    m.update();
    return m;
  }
  /** The model's natural PlayCanvas-space world AABB, from each mesh's local box. */
  function naturalModelAabb(ent) {
    const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    const v = new Vec3(), out = new Vec3();
    ent.findComponents("render").forEach(rc => (rc.meshInstances || []).forEach(mi => {
      const box = mi.mesh && mi.mesh.aabb;        // model-local BoundingBox
      if (!box || !box.halfExtents || !mi.node) return;
      const cx = box.center.x, cy = box.center.y, cz = box.center.z;
      const hx = box.halfExtents.x, hy = box.halfExtents.y, hz = box.halfExtents.z;
      const world = mi.node.getWorldTransform();
      for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
        v.set(cx + sx * hx, cy + sy * hy, cz + sz * hz);
        world.transformPoint(v, out);
        mn[0] = Math.min(mn[0], out.x); mx[0] = Math.max(mx[0], out.x);
        mn[1] = Math.min(mn[1], out.y); mx[1] = Math.max(mx[1], out.y);
        mn[2] = Math.min(mn[2], out.z); mx[2] = Math.max(mx[2], out.z);
      }
    }));
    return [mn, mx].every(a => a.every(Number.isFinite)) ? { min: mn, max: mx } : null;
  }
  function clearModel(item) {
    const m = item.model;
    if (!m) return;
    m.token++;                             // invalidate any in-flight load for this slot
    if (m.ent) m.ent.destroy();
    if (m.wrapper) m.wrapper.destroy();
    if (m.mat) m.mat.destroy();
    item.model = null;
    if (item.holder.render) item.holder.render.enabled = true;   // the box mesh renders again
  }
  function paintModel(m, p) {
    if (m.mat) m.mat.destroy();
    m.mat = modelTint(p);
    m.ent.findComponents("render").forEach(rc => rc.meshInstances.forEach(mi => { mi.material = m.mat; }));
  }
  function fitModel(item, p, box) {
    const m = item.model;
    if (!m || !m.ent || !m.aabb || !box) return;
    m.wrapper.setLocalPosition(box.center[0], box.center[1], box.center[2]);
    m.wrapper.setLocalEulerAngles(0, -box.yaw, 0);
    const fit = importedModelFit(m.aabb, box);
    // The fit goes on its own node, never on the instantiated model: a glTF root
    // node may carry a real transform (the up-axis flip and unit scale every
    // exporter emits), and writing over it would draw the model at its raw size
    // while the collider stayed at the size the file declared.
    if (fit && m.fitter) {
      m.fitter.setLocalScale(fit.scale[0], fit.scale[1], fit.scale[2]);
      m.fitter.setLocalPosition(fit.offset[0], fit.offset[1], fit.offset[2]);
    }
  }
  /** Attach (or re-fit) the real geometry for an imported item; a no-op for primitives. */
  function bindModel(item, p, box) {
    const file = p.model && p.model.file;
    if (!file) { clearModel(item); return; }
    let m = item.model;
    if (!m || m.file !== file) {
      clearModel(item);
      const wrapper = new Entity(`model:${p.id}`);
      item.holder.addChild(wrapper);
      m = item.model = { wrapper, fitter: null, file, aabb: null, ent: null, mat: null, loading: false, token: 0 };
    }
    if (m.ent && m.aabb) { fitModel(item, p, box); paintModel(m, p); return; }
    m.wrapper.setLocalPosition(box.center[0], box.center[1], box.center[2]);
    m.wrapper.setLocalEulerAngles(0, -box.yaw, 0);
    if (m.loading) return;
    m.loading = true;
    const token = m.token;
    modelAsset(file).then(asset => {
      m.loading = false;
      if (!item.model || item.model !== m || m.token !== token) return;   // superseded or removed
      if (!asset) return;                                                  // keep the box fallback
      let ent = null;
      try {
        ent = asset.resource.instantiateRenderEntity({ castShadows: false, receiveShadows: false });
        const aabb = naturalModelAabb(ent);                               // measure before parenting
        if (!aabb) { ent.destroy(); return; }
        m.ent = ent; m.aabb = aabb;
        m.fitter = new Entity(`fit:${p.id}`);    // the box fit lives here, not on the model's own root
        m.wrapper.addChild(m.fitter);
        m.fitter.addChild(ent);
        const current = placementById(p.id) || p;
        fitModel(item, current, placementBox(current) || box);
        paintModel(m, current);
        if (item.holder.render) item.holder.render.enabled = false;        // the model replaced the box
      } catch (e) {
        console.log(`[place] could not render imported model ${file}: ${e && e.message ? e.message : e}`);
        if (ent) ent.destroy();
      }
    });
  }
  function ensurePlacedRoot() {
    if (placedRoot) return placedRoot;
    placedRoot = new Entity("placed");
    placedRoot.addComponent("rigidbody", { type: "static", friction: 0.6, restitution: 0 });
    placedRoot.addComponent("collision", { type: "compound" });
    application.root.addChild(placedRoot);
    return placedRoot;
  }
  function destroyPlacedItem(id) {
    const item = placedItems.get(id);
    if (!item) return;
    clearModel(item);
    placedByCollider.delete(item.collider);
    item.collider.destroy();
    if (item.mesh) item.mesh.destroy();
    item.holder.destroy();
    placedItems.delete(id);
  }
  function upsertPlacement(p) {
    const box = placementBox(p);
    if (!box) return;
    const sig = `${p.center_xz.join(",")}|${p.center_y}|${p.size.join(",")}|${p.yaw_deg}|${p.item}|${p.stale ? 1 : 0}|${p.fit ? JSON.stringify(p.fit) : ""}|${p.model ? p.model.file : ""}`;
    const existing = placedItems.get(p.id);
    if (existing && existing.sig === sig) return;
    if (existing) {
      // Only the shape/verdict changed: swap the mesh in place, keep the collider node.
      existing.collider.setLocalPosition(box.center[0], box.center[1], box.center[2]);
      existing.collider.setLocalEulerAngles(0, -box.yaw, 0);
      existing.collider.collision.halfExtents = new Vec3(box.halfExtents[0], box.halfExtents[1], box.halfExtents[2]);
      const mesh = placementGeometryMesh(p);
      if (mesh) {
        if (existing.mesh) existing.mesh.destroy();
        existing.holder.render.meshInstances = [new MeshInstance(mesh, placementMaterial())];
        existing.mesh = mesh;
      }
      existing.sig = sig;
      bindModel(existing, p, box);
      return;
    }
    const holder = new Entity(`place:${p.id}`);
    holder.addComponent("render", { meshInstances: [] });
    ensurePlacedRoot().addChild(holder);
    const collider = new Entity();
    collider.setLocalPosition(box.center[0], box.center[1], box.center[2]);
    collider.setLocalEulerAngles(0, -box.yaw, 0);   // physics yaw sign, as detected furniture uses
    collider.addComponent("collision", { type: "box", halfExtents: new Vec3(box.halfExtents[0], box.halfExtents[1], box.halfExtents[2]) });
    ensurePlacedRoot().addChild(collider);
    const mesh = placementGeometryMesh(p);
    if (mesh) holder.render.meshInstances = [new MeshInstance(mesh, placementMaterial())];
    const item = { holder, collider, mesh, sig };
    placedItems.set(p.id, item);
    placedByCollider.set(collider, p.id);
    bindModel(item, p, box);
  }
  function rebuildPlacements() {
    const live = new Set(renderedPlacements.filter(p => placementBox(p)).map(p => p.id));
    for (const id of [...placedItems.keys()]) if (!live.has(id)) destroyPlacedItem(id);
    for (const p of renderedPlacements) upsertPlacement(p);
  }
  /** Which placed item, if any, is under this canvas point. */
  function placementAt(clientX, clientY) {
    const xy = canvasPoint(clientX, clientY, canvas.getBoundingClientRect());
    if (!xy) return null;
    const camera = cameraEnt.camera;
    const start = camera.screenToWorld(xy[0], xy[1], camera.nearClip);
    const end = camera.screenToWorld(xy[0], xy[1], camera.farClip);
    return placementRay(renderedPlacements, [start.x, start.y, start.z], [end.x, end.y, end.z]);
  }
  /** Move one placed item where the editor is dragging it, without a host round-trip. */
  function dragPlacement(id, center) {
    const p = placementById(id);
    const box = p && placementBox(p);
    if (!box) return null;
    p.center_xz = [center[0], center[2]];
    p.center_y = center[1];
    upsertPlacement(p);
    return [p.center_xz[0], p.center_y, p.center_xz[1]];
  }
  function setRenderedPlacements(list) {
    renderedPlacements = Array.isArray(list) ? list.filter(p => {
      const box = placementBox(p);
      return box && box.corners.every(point => point.every(v => Number.isFinite(Math.fround(v)))) &&
        box.halfExtents.every(v => Number.isFinite(Math.fround(v)) && Math.fround(v) > 0);
    }) : [];
    rebuildPlacements();
  }

  function sceneBounds() {
    // Frame on the SUBJECT, not the cameras: the collider/ground shell is what a
    // user measures and walks, and it is tighter than the full splat AABB (which
    // carries distant floaters and the ground skirt that push the fit far out).
    // Camera positions are only a last-resort fallback for a scene with no ground.
    const corners = [];
    colliderEnt?.findComponents("render").forEach(render => {
      for (const mi of render.meshInstances) {
        const lo = mi.aabb.getMin(), hi = mi.aabb.getMax();
        corners.push([lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z]);
      }
    });
    const modelBounds = boundsFromPoints(corners);
    if (modelBounds) return modelBounds;
    const localBox = splatEnt?.gsplat?.customAabb;
    if (localBox) {
      const box = new BoundingBox();
      box.setFromTransformedAabb(localBox, splatEnt.getWorldTransform());
      const bounds = boundsFromPoints([
        [box.getMin().x, box.getMin().y, box.getMin().z],
        [box.getMax().x, box.getMax().y, box.getMax().z],
      ]);
      if (bounds) return bounds;
    }
    const camFallback = boundsFromPoints(allCameras.map(cam => cam.pos));
    if (camFallback) return camFallback;
    if (HF) {
      let low = Infinity, high = -Infinity;
      for (const h of HF.data) if (Number.isFinite(h)) { low = Math.min(low, h); high = Math.max(high, h); }
      return boundsFromPoints([[HF.ox, low, HF.oz], [HF.ox + HF.nx * HF.cell, high, HF.oz + HF.nz * HF.cell]]);
    }
    return null;
  }

  function syncWorkspaceFly() {
    const eye = cameraEnt.getPosition(), f = cameraEnt.forward;
    dronePos.copy(eye);
    P.yaw = Math.atan2(-f.x, -f.z);
    P.pitch = Math.asin(Math.max(-1, Math.min(1, f.y)));
    workspaceLookDirty = false;
  }
  function applyWorkspaceOrbit() {
    const eye = orbitEye(workspaceOrbit);
    cameraEnt.setPosition(...eye);
    cameraEnt.lookAt(new Vec3(...workspaceOrbit.target));
    syncWorkspaceFly();
    broadcastCamera();
  }
  function setWorkspaceMode(mode) {
    if (mode === workspaceMode) return;
    clearActiveKeys();
    if (document.pointerLockElement === canvas) document.exitPointerLock();
    if (mode === "walk") {
      if (!workspaceCapabilities().collider || !HF) throw new Error("Walk requires a collision surface and heightfield");
      const eye = cameraEnt.getPosition();
      let x = eye.x, z = eye.z, hit = groundProbe(x, z);
      if (!hit && col.spawn) { x = col.spawn.x; z = col.spawn.z; hit = groundProbe(x, z); }
      if (!hit) throw new Error("No walkable surface near this view; fit or select another camera first");
      syncWorkspaceFly();
      isDrone = false;
      P.firstPerson = true;
      P.prev = null;
      P.lastGood = null;
      playerEnt.enabled = true;
      playerRb.type = "dynamic";
      playerRb.teleport(x, hit.point.y + CHAR_H * 0.86, z);
      playerRb.linearVelocity = new Vec3();
    } else {
      syncWorkspaceFly();
      if (mode === "orbit") {
        const eye = cameraEnt.getPosition(), f = cameraEnt.forward;
        workspaceOrbit = orbitFromPose([eye.x, eye.y, eye.z], [f.x, f.y, f.z], workspaceOrbit?.distance || CHAR_H * 5);
      }
      isDrone = true;
      playerRb.linearVelocity = new Vec3();
      playerRb.type = "kinematic";
      playerEnt.enabled = false;
    }
    workspaceMode = mode;
    autopilot.phase = "idle";
  }
  function fitWorkspace(top = false) {
    clearActiveKeys();
    setWorkspaceMode("orbit");
    cameraEnt.camera.fov = 70;
    workspaceOrbit = fitOrbit(sceneBounds(), 70, canvas.clientWidth / Math.max(1, canvas.clientHeight));
    // A top-down "plan" view looks almost straight down at the floor — the angle a
    // person expects when laying out furniture — instead of the outside of the room.
    if (top) { workspaceOrbit.yaw = 0; workspaceOrbit.pitch = 1.32; }
    markerSize = Math.max(0.001, workspaceOrbit.distance * 0.003);
    cameraEnt.camera.farClip = Math.max(3000, workspaceOrbit.maxDistance * 2);
    cameraEnt.camera.nearClip = Math.min(0.08 * CHAR_SCALE, workspaceOrbit.minDistance * 0.5);
    applyWorkspaceOrbit();
    selectedCamIdx = -1;
    if (selectedCamEntity) selectedCamEntity.enabled = false;
    setPreviewPoint(null);
  }
  function focusWorkspacePlacement(id) {
    // Validate before changing modes, so a missing item does not move the view.
    const orbit = focusPlacement(renderedPlacements, id, 70, canvas.clientWidth / Math.max(1, canvas.clientHeight));
    clearActiveKeys();
    setWorkspaceMode("orbit");
    workspaceOrbit = orbit;
    cameraEnt.camera.fov = 70;
    cameraEnt.camera.nearClip = Math.min(0.08 * CHAR_SCALE, orbit.minDistance * 0.5);
    cameraEnt.camera.farClip = Math.max(3000, orbit.maxDistance * 2);
    markerSize = Math.max(0.001, orbit.distance * 0.003);
    rebuildPickLines();
    selectedMeasureId = id;
    selectedCamIdx = -1;
    if (selectedCamEntity) selectedCamEntity.enabled = false;
    setPreviewPoint(null);
    applyWorkspaceOrbit();
  }
  function setWorkspaceLayers(layer, value) {
    const layers = layerState(workspaceLayers(), layer, value, workspaceCapabilities());
    splatVisible = layers.splats; showCameras = layers.cameras; showPoints = layers.points;
    showCoverage = layers.coverage; showCollider = layers.collider; showSemantics = layers.semantics;
    splatEnt.enabled = splatVisible;
    if (camerasEntity) camerasEntity.enabled = showCameras;
    if (trajectoryEntity) trajectoryEntity.enabled = showCameras;
    if (selectedCamEntity) selectedCamEntity.enabled = showCameras && selectedCamIdx >= 0;
    if (sparsePointsEntity) sparsePointsEntity.enabled = showPoints;
    if (semanticsEntity) semanticsEntity.enabled = showSemantics;
    if (coverageGridEntity) coverageGridEntity.enabled = showCoverage;
    colliderEnt?.findComponents("render").forEach(render => { render.enabled = showCollider; });
    if (objectsEnt) objectsEnt.enabled = showCollider;
  }
  function frameWorkspace(index) {
    const cam = allCameras[index];
    if (!cam || !finitePoint(cam.pos) || !finitePoint(cam.forward) || Math.hypot(...cam.forward) === 0) {
      throw new Error("Selected source camera is unavailable or has an invalid pose");
    }
    clearActiveKeys();
    if (workspaceMode === "walk") setWorkspaceMode("fly");
    workspaceOrbit = orbitFromPose(cam.pos, cam.forward, workspaceOrbit?.distance || CHAR_H * 5);
    selectedCamIdx = index;
    if (Array.isArray(cam.corners) && cam.corners.length === 4 && cam.corners.every(finitePoint) && finitePoint(cam.top_mark)) {
      updateSelectedCameraMesh(device, cam);
      selectedCamEntity.enabled = showCameras;
    }
    cameraEnt.setPosition(...cam.pos);
    const target = new Vec3(...cam.pos).add(new Vec3(...cam.forward));
    const up = finitePoint(cam.up) && Math.hypot(...cam.up) > 0 ? new Vec3(...cam.up) : Vec3.UP;
    cameraEnt.lookAt(target, up);
    if (Number.isFinite(cam.fov_y_deg) && cam.fov_y_deg > 1 && cam.fov_y_deg < 179) cameraEnt.camera.fov = cam.fov_y_deg;
    syncWorkspaceFly();
  }
  function rebuildPickLines() {
    pickLines = [];
    for (let i = 0; i < pickPoints.length; i++) {
      const p = pickPoints[i];
      for (let axis = 0; axis < 3; axis++) {
        const a = [...p], b = [...p];
        a[axis] -= markerSize; b[axis] += markerSize;
        pickLines.push(new Vec3(...a), new Vec3(...b));
      }
      if (i && measurementKind === null) pickLines.push(new Vec3(...pickPoints[i - 1]), new Vec3(...p));
    }
  }
  function surfacePointAt(clientX, clientY) {
    const xy = canvasPoint(clientX, clientY, canvas.getBoundingClientRect());
    if (!xy) return null;
    const camera = cameraEnt.camera;
    const start = camera.screenToWorld(xy[0], xy[1], camera.nearClip);
    const end = camera.screenToWorld(xy[0], xy[1], camera.farClip);
    // Only the loaded collision mesh is measurable; splat opacity and synthesized
    // furniture boxes are not a measured surface.
    const hit = application.systems.rigidbody.raycastFirst(start, end, {
      filterCollisionMask: BODYMASK_STATIC,
      filterCallback: entity => collisionMeshEntities.has(entity),
    });
    return collisionSurfacePoint(hit, collisionMeshEntities);
  }
  function pickWorkspace(clientX, clientY) {
    if (!workspace.ready || !workspacePick) return;
    try {
      const point = surfacePointAt(clientX, clientY);
      setPreviewPoint(null);
      if (!point) {
        workspace.emit("pick-miss", { message: "No collision surface at this pixel. Splats are not a measured surface." });
        return;
      }
      pickPoints = appendPick(pickPoints, point, workspacePickLimit);
      if (pickPoints.length >= workspacePickLimit) workspacePick = false;
      rebuildPickLines();
      workspace.emit("pick", { point, geometry: "collision_surface" });
    } catch (e) { setPreviewPoint(null); workspace.emit("error", { message: e.message || String(e) }); }
  }
  if (EMBED) workspaceCommand = cmd => {
    switch (cmd.command) {
      case "mode": setWorkspaceMode(cmd.value); setPreviewPoint(null); break;
      case "fit": fitWorkspace(); break;
      case "layer": setWorkspaceLayers(cmd.layer, cmd.value); break;
      case "frame": frameWorkspace(cmd.index); setPreviewPoint(null); break;
      case "measurement-tool":
        measurementKind = cmd.kind; measurementUnit = cmd.unit;
        setPreviewPoint(null); rebuildPickLines(); break;
      case "focus-placement": focusWorkspacePlacement(cmd.id); break;
      case "edit":
        workspaceEdit = cmd.value;
        // One tool at a time: arming the editor stops measurement picking, and
        // releasing it drops any selection highlight.
        if (cmd.value) { pickingRequested = false; workspacePick = false; setPreviewPoint(null); }
        else if (!cmd.value) { selectedMeasureId = null; editDrag = null; }
        break;
      case "pick":
        if (cmd.value && !workspaceCapabilities().collider) throw new Error("Measurement requires a collision surface");
        if (cmd.value) { workspaceEdit = false; selectedMeasureId = null; }
        workspacePickLimit = cmd.limit ?? MAX_PICKS;
        pickingRequested = cmd.value;
        workspacePick = pickingRequested && pickPoints.length < workspacePickLimit;
        setPreviewPoint(null); break;
      case "clear-picks":
        pickPoints = []; pickLines = []; workspacePick = pickingRequested;
        setPreviewPoint(null); break;
      case "set-picks":
        pickPoints = cmd.value.map(p => [...p]);
        workspacePick = pickingRequested && pickPoints.length < workspacePickLimit;
        setPreviewPoint(null); rebuildPickLines(); break;
      case "measurements": setRenderedMeasurements(cmd.value); break;
      case "placements": setRenderedPlacements(cmd.value); break;
      case "select": selectedMeasureId = cmd.id; break;
      case "labels": showMeasureLabels = cmd.value; break;
      case "view": fitWorkspace(cmd.value === "top"); break;
      case "snapshot": snapshotCapture.request(); break;
      case "plan":
        planState = cmd.value;
        if (!planDrag) clearPlanPreview();
        rebuildPlan(); break;
      case "plan-select":
        planSelect = cmd.value;
        if (cmd.value) { workspaceEdit = false; editDrag = null; }
        else if (planDrag) endPlanDrag(null, true);
        break;
      case "camera-follow":
        cameraFollow = cmd.value;
        if (cameraFollow) { cameraEmitAt = 0; broadcastCamera(); }
        break;
      case "camera-set":
        if (cmd.value?.pose) {
          // Fly / walk compare: the other viewer's eye, taken as a fly camera (no physics here).
          const { eye, forward } = cmd.value.pose;
          if (!Array.isArray(eye) || !Array.isArray(forward) || eye.length !== 3 || forward.length !== 3 || ![...eye, ...forward].every(Number.isFinite)) break;
          if (workspaceMode !== "fly") setWorkspaceMode("fly");
          remotePose = true;
          dronePos.set(eye[0], eye[1], eye[2]);
          P.yaw = Math.atan2(-forward[0], -forward[2]);
          P.pitch = Math.asin(Math.max(-1, Math.min(1, forward[1])));
          workspaceLookDirty = true;
          remotePoseUntil = performance.now() + 250;
          break;
        }
        if (workspaceMode !== "orbit") setWorkspaceMode("orbit");
        // Optional lens: a "view from the photo" leaves the photo's narrow field of view
        // behind, and an orbit resumed from there can ease it back.
        if (Number.isFinite(cmd.value.fov) && cmd.value.fov > 1 && cmd.value.fov < 179) cameraEnt.camera.fov = cmd.value.fov;
        remoteOrbit = true;
        try { const { fov: _fov, ...orbit } = cmd.value; workspaceOrbit = { ...workspaceOrbit, ...orbit, target: [...orbit.target] }; applyWorkspaceOrbit(); }
        finally { remoteOrbit = false; }
        break;
    }
  };

  // Camera Inspector events
  document.getElementById("btn-inspect")?.addEventListener("click", () => {
    const inspModal = document.getElementById("cam-inspector");
    if (inspModal.style.display === "flex") {
      inspModal.style.display = "none";
    } else {
      inspectCamera(selectedCamIdx >= 0 ? selectedCamIdx : 0);
    }
  });
  document.getElementById("insp-slider")?.addEventListener("input", (e) => {
    inspectCamera(parseInt(e.target.value, 10));
    snapToSelectedCamera();
  });
  document.getElementById("insp-play-seq")?.addEventListener("click", toggleSequencePlayback);
  document.getElementById("btn-cams")?.addEventListener("click", toggleCamerasLayer);
  document.getElementById("btn-points")?.addEventListener("click", togglePointsLayer);
  document.getElementById("btn-col")?.addEventListener("click", toggleCollider);
  document.getElementById("btn-cov")?.addEventListener("click", toggleCoverageLayer);
  document.getElementById("btn-splat-vis")?.addEventListener("click", toggleSplatVisibility);
  document.getElementById("btn-splat")?.addEventListener("click", () => toggleSplatModel());
  document.getElementById("btn-mode")?.addEventListener("click", () => toggleDroneMode());
  document.getElementById("btn-cam")?.addEventListener("click", () => { P.firstPerson = !P.firstPerson; updateToolbarUI(); });
  document.getElementById("btn-auto")?.addEventListener("click", () => {
    if (isDrone) toggleDroneMode(false);
    autopilot.phase = "settle"; autopilot.t = 0;
  });
  document.getElementById("btn-reset")?.addEventListener("click", () => {
    if (isDrone) toggleDroneMode(false);
    spawnFrom(col);
  });

  document.getElementById("insp-close")?.addEventListener("click", () => {
    document.getElementById("cam-inspector").style.display = "none";
    if (isPlayingSeq) toggleSequencePlayback();
    if (selectedCamEntity) selectedCamEntity.enabled = false;
    selectedCamIdx = -1;
  });
  document.getElementById("insp-prev")?.addEventListener("click", () => {
    inspectCamera(selectedCamIdx - 1);
    snapToSelectedCamera();
  });
  document.getElementById("insp-next")?.addEventListener("click", () => {
    inspectCamera(selectedCamIdx + 1);
    snapToSelectedCamera();
  });
  document.getElementById("insp-jump")?.addEventListener("click", snapToSelectedCamera);
  document.getElementById("cov-close")?.addEventListener("click", () => {
    document.getElementById("cov-panel").style.display = "none";
  });

  const btnCombat = document.getElementById("btn-combat");
  if (btnCombat) {
    const next = new URLSearchParams(q);
    next.set("combat", COMBAT ? "0" : "1");
    btnCombat.href = `${location.pathname}?${next.toString()}`;
    btnCombat.classList.toggle("active", COMBAT);
    const lblCombat = document.getElementById("lbl-combat");
    if (lblCombat) lblCombat.textContent = COMBAT ? "Combat: ON" : "Combat: OFF";
  }

  // Pick nearest camera frustum on click
  function pickCameraFromRay(clientX, clientY) {
    if (!allCameras.length || !showCameras) return;
    const curEye = isDrone ? dronePos : cameraEnt.getPosition();
    const curLook = isDrone
      ? new Vec3(dronePos.x - Math.sin(P.yaw) * Math.cos(P.pitch), dronePos.y + Math.sin(P.pitch), dronePos.z - Math.cos(P.yaw) * Math.cos(P.pitch))
      : playerEnt.getPosition();

    // Find camera closest to ray or within proximity
    let bestIdx = -1, bestDist = 0.65;
    for (let i = 0; i < allCameras.length; i++) {
      const c = allCameras[i];
      const d = Math.hypot(c.pos[0] - curEye.x, c.pos[1] - curEye.y, c.pos[2] - curEye.z);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = i;
      }
    }
    if (bestIdx >= 0) {
      inspectCamera(bestIdx);
    }
  }

  // ---------------- main loop ----------------
  let frames = 0;
  let renderedSplat = !EMBED;
  if (EMBED) application.systems.gsplat.on("frame:ready", (camera, layer, ready, loadingCount) => {
    // The sort worker can finish later than frame three. Do not announce a blank overview.
    if (camera === cameraEnt.camera && ready && !loadingCount) renderedSplat = true;
  });
  application.on("update", (dtRaw) => {
    const dt = Math.min(dtRaw, 0.05);
    step(dt);
    if (EMBED) broadcastPose();
    combat?.update(dt);

    const w = window.__walk;
    w.walked = +P.walked.toFixed(2);
    w.phase = autopilot.phase;
    w.grounded = P.grounded;
    w.violations = P.violations;
    w.yaw = +P.yaw.toFixed(3);
    if (w.phase !== "idle") {
      walkElapsed += dt;
      walkSampleClock += dt;
      if (walkSampleClock >= WALK_SAMPLE_DT && w.samples.length < WALK_SAMPLE_MAX) {
        walkSampleClock = 0;
        w.samples.push([+walkElapsed.toFixed(1), w.walked, w.pos[0], w.pos[1],
                        w.pos[2], P.grounded ? 1 : 0, P.violations]);
      }
    }

    if (EMBED) {
      if (pickLines.length) application.drawLines(pickLines, pickColor, false);
      drawMeasurementGeometry();
      drawPlanEdit();
      return;
    }
    const splatLabel = splatCountLabel();
    const camStatus = showCameras ? `ON (${allCameras.length} cams)` : "OFF";
    // Null-safe and verdict-safe: a refused grid reports covered_pct === null,
    // which must read "not measurable" on the HUD, not "null% cov".
    const covStatus = !allCoverageGrid ? "N/A"
      : (allCoverageGrid.measurable === false ? "cov NOT MEASURABLE"
                                              : `${allCoverageGrid.covered_pct}% cov`);

    if (isDrone) {
      const isFast = !!(activeKeys["ShiftLeft"] || activeKeys["ShiftRight"] || activeKeys["shift"]);
      const isSlow = !!(activeKeys["AltLeft"] || activeKeys["AltRight"] || activeKeys["ControlLeft"] || activeKeys["ControlRight"]);
      const curSpeed = isFast ? droneSpeed * 2.5 : (isSlow ? droneSpeed * 0.3 : droneSpeed);
      hudEl.textContent =
        `[🚁 DRONE FLY]   Speed: ${curSpeed.toFixed(1)} m/s   Splat: ${splatLabel}   Cams: ${camStatus}   Coverage: ${covStatus}\n` +
        `pos: (${dronePos.x.toFixed(2)}, ${dronePos.y.toFixed(2)}, ${dronePos.z.toFixed(2)})\n` +
        `WASD 3D Fly | G Splats ON/OFF | P Cams | O Points | V Coverage | X Collider | J Snap | U Model`;
    } else {
      hudEl.textContent =
        `[🚶 GROUND WALK]   ${P.firstPerson ? "1st Person" : "3rd Person"}   walked ${P.walked.toFixed(1)} m   falls ${P.violations}\n` +
        `pos: (${w.pos.join(", ")})   ${P.grounded ? "grounded" : "air"}   Splat: ${splatLabel}   Cams: ${camStatus}\n` +
        `WASD Walk | Shift Run | G Splats ON/OFF | P Cams | O Points | V Coverage | X Collider | F Drone | C View`;
    }
  });

  application.on("postrender", () => {
    frames++;
    if (frames >= 3 && renderedSplat && !window.__ready && !window.__loadError && (!EMBED || workspaceOrbit)) {
      window.__ready = true;
      setLoad("");
      if (loadEl) {
        loadEl.style.opacity = "0";
        loadEl.style.pointerEvents = "none";
        if (!EMBED) setTimeout(() => { if (!loadEl.textContent) loadEl.remove(); }, 500);
      }
      workspace?.markReady();
    }
    if (workspace?.ready) {
      // Flush even when the pointer stops: the last sample may be a pending clear.
      previewStream.flush(performance.now());
      updateMeasureLabels();
      updatePlanHandles();
      snapshotCapture.postrender(() => canvas.toDataURL("image/png"));
    }
  });

  application.systems.rigidbody.gravity.set(0, -18, 0);
  window.__app = application;

  // Automation hooks
  window.__setCam = (eye, target) => {
    cameraEnt.setPosition(eye[0], eye[1], eye[2]);
    cameraEnt.lookAt(new Vec3(target[0], target[1], target[2]));
    if (EMBED) {
      workspaceOrbit = orbitFromPose(eye, target.map((v, i) => v - eye[i]), Math.hypot(...target.map((v, i) => v - eye[i])));
      syncWorkspaceFly();
    }
  };
  window.__playerPos = () => {
    const p = playerEnt.getPosition();
    return [p.x, p.y - 0.9, p.z];
  };

  // Pick a spawn
  window.__stage = "spawn-pick";
  // Workspace overview must not spend seconds walking a hidden player to score spawns.
  window.__chosenSpawn = EMBED ? null : await chooseSpawn(col);
  window.__stage = "live";
  spawnFrom(col);
  setTimeout(() => buildUnderlay(), 500);

  if (isDrone) toggleDroneMode(true);
  if (EMBED) {
    P.firstPerson = true;
    playerEnt.findComponents("render").forEach(render => { render.enabled = false; });
    showCameras = showCameras && !!camerasEntity;
    showCollider = showCollider && collisionMeshEntities.size > 0;
    fitWorkspace();
  }
  if (COMBAT) {
    P.firstPerson = true;
    // Mission rehearsal: ?mission=<id>&scene=<scene>[&light=day|dusk|night][&nvg=1][&fog=<m>]
    const missionId = q.get("mission"), missionScene = q.get("scene");
    const rehearsal = missionId && /^p-[0-9a-f]{10}$/.test(missionId) && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(missionScene || "")
      ? (async () => {
        const path = `mission/scenario?scene=${encodeURIComponent(missionScene)}&id=${missionId}`;
        // Through the studio's proxy when embedded there, else straight to the local service.
        for (const base of ["/api/backend/api/workspace/", "/api/workspace/"]) {
          const r = await fetch(base + path, { cache: "no-store" }).catch(() => null);
          if (r && r.ok) {
            const light = ["day", "dusk", "night"].includes(q.get("light")) ? q.get("light") : "day";
            const fog = Number(q.get("fog"));
            return { scenario: await r.json(), saveUrl: { url: `${base}mission/run`, scene: missionScene },
              conditions: { light, nvg: q.get("nvg") === "1", fog_m: Number.isFinite(fog) && fog >= 10 ? Math.min(fog, 2000) : undefined },
              // RH-9: a join link from the instructor carries the session and its token.
              session: /^s-[0-9a-f]{8}$/.test(q.get("session") || "") && q.get("token")
                ? { id: q.get("session"), token: q.get("token"), name: (q.get("name") || "Player").slice(0, 24) } : null,
              parent: window.parent !== window ? window.parent : null, origin: location.origin };
          }
        }
        throw new Error("The mission scenario could not be loaded");
      })()
      : Promise.resolve(null);
    rehearsal.then((mission) => import("./pc/scripts/combat.js").then((module) => ({ module, mission })))
      .then(({ module: { installCombat }, mission }) => installCombat(app, {
        mission,
        // The real engine: ``app`` above is a shallow copy without its methods (drawLines).
        engine: application,
        spawnPlayer: () => spawnFrom(col),
        app,
        canvas,
        camera: cameraEnt,
        player: playerEnt,
        rig: assets.rig,
        asset: ASSET,
        HF,
        spawn: col.spawn,
        walkPath: col.walk_path,
        cameras: allCameras,
        endless: !missionId && q.get("endless") !== "0",
        config: { bots: Math.max(1, Number(q.get("bots")) || 5) },
        getYaw: () => P.yaw,
        grounded: () => P.grounded,
        /** World point the player's barrel is pointing out of, once the rig has one. */
        muzzle: () => playerChar?.muzzleWorld(),
        /** Metres from the player entity's origin DOWN to its collider's floor. */
        playerFeet: () => FEET,
        nudgePitch: (d) => { P.pitch = Math.min(1.45, Math.max(-1.45, P.pitch + d)); },
        /** VR: the right controller's pointer ray while a session runs, else null (camera aim). */
        aimRay: () => vrState?.aimRay?.() ?? null,
        respawn: () => spawnFrom(col),
      }))
      .then((c) => {
        combat = c;
        window.__combat = c;
        // VR rehearsal (RH-7): offered only where the browser reports immersive-vr.
        return import("./pc/scripts/vr.js").then(({ installVr }) => {
          vrState = installVr({
            app: application, camera: cameraEnt, canvas,
            onTrigger: (on) => { if (combat) combat.firing = on; },
            onStatus: (status) => { window.__vr = status; },
          });
        });
      })
      .catch((e) => {
        window.__combatError = String(e && e.message ? e.message : e);
        console.error("[combat] failed to start", e);
        hudEl.textContent = `[COMBAT FAILED] ${window.__combatError}`;
      });
  }
  updateToolbarUI();

  // ---------------- input handlers ----------------
  if (EMBED) {
    let gesture = null;
    canvas.addEventListener("contextmenu", e => e.preventDefault());
    canvas.addEventListener("pointerdown", e => {
      if (!workspace.ready || (e.button !== 0 && e.button !== 2) || gesture || editDrag || planDrag) return;
      e.preventDefault();
      canvas.focus({ preventScroll: true });
      setPreviewPoint(null);
      // In the planner, the selected feature's own body is a move grip.
      if (planSelect && !workspacePick && planState.edit && planState.view === "proposal" && e.button === 0 && !e.shiftKey) {
        const xy = canvasPoint(e.clientX, e.clientY, canvas.getBoundingClientRect());
        if (xy) {
          const camera = cameraEnt.camera;
          const a = camera.screenToWorld(xy[0], xy[1], camera.nearClip), b = camera.screenToWorld(xy[0], xy[1], camera.farClip);
          const hit = rayPick(planState.features, [a.x, a.y, a.z], [b.x - a.x, b.y - a.y, b.z - a.z]);
          if (hit && hit.id === planState.edit.id) {
            const grip = planHandles(planState.edit).find(h => h.kind === "move");
            if (beginPlanDrag(e, grip)) { canvas.style.cursor = "grabbing"; return; }
          }
        }
      }
      // In the editor a piece under the cursor is grabbed, not orbited around.
      if (workspaceEdit && e.button === 0) {
        const id = placementAt(e.clientX, e.clientY);
        const box = id && placementBox(placementById(id));
        const floor = id && surfacePointAt(e.clientX, e.clientY);
        if (box) {
          editDrag = { id, pointerId: e.pointerId, moved: false, startX: e.clientX, startY: e.clientY,
            offset: floor ? [box.center[0] - floor[0], 0, box.center[2] - floor[2]] : [0, 0, 0],
            from: [box.center[0], box.center[1], box.center[2]], last: 0 };
          canvas.setPointerCapture(e.pointerId);
          return;
        }
      }
      gesture = { id: e.pointerId, x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY,
        pan: e.button === 2 || e.shiftKey, button: e.button, moved: false };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", e => { if (planDrag && planDrag.target === canvas) movePlanDrag(e); });
    canvas.addEventListener("pointermove", e => {
      if (!editDrag || editDrag.pointerId !== e.pointerId) return;
      if (Math.hypot(e.clientX - editDrag.startX, e.clientY - editDrag.startY) > 4) editDrag.moved = true;
      if (!editDrag.moved) return;
      const floor = surfacePointAt(e.clientX, e.clientY);
      // Over unscanned space there is no surface to sit on: hold the last real
      // position instead of snapping the piece to a guessed point.
      if (!floor) return;
      const moved = dragPlacement(editDrag.id, [floor[0] + editDrag.offset[0], editDrag.from[1], floor[2] + editDrag.offset[2]]);
      if (!moved) return;
      const now = performance.now();
      if (now - editDrag.last < 1000 / 15) return;
      editDrag.last = now;
      workspace.emit("placement-move", { id: editDrag.id, point: moved, final: false });
    });
    canvas.addEventListener("pointermove", e => {
      if (!gesture || gesture.id !== e.pointerId) return;
      if (Math.hypot(e.clientX - gesture.startX, e.clientY - gesture.startY) > 4) gesture.moved = true;
      if (!gesture.moved) return;
      const dx = e.clientX - gesture.x, dy = e.clientY - gesture.y;
      gesture.x = e.clientX; gesture.y = e.clientY;
      if (workspaceMode === "orbit") {
        workspaceOrbit = gesture.pan || e.shiftKey
          ? panOrbit(workspaceOrbit, dx, dy, canvas.clientHeight, cameraEnt.camera.fov)
          : rotateOrbit(workspaceOrbit, -dx * 0.005, dy * 0.005);
        applyWorkspaceOrbit();
      } else {
        P.yaw -= dx * 0.0025;
        P.pitch = Math.max(-1.45, Math.min(1.45, P.pitch - dy * 0.0022));
        workspaceLookDirty = true;
      }
    });
    // Local geometry follows the pointer immediately; host events are coalesced.
    canvas.addEventListener("pointermove", e => {
      if (!workspace.ready || !workspacePick || gesture) return;
      try { setPreviewPoint(surfacePointAt(e.clientX, e.clientY)); }
      catch { setPreviewPoint(null); }
    });
    canvas.addEventListener("pointerleave", () => setPreviewPoint(null));
    canvas.addEventListener("pointerup", e => {
      if (planDrag && planDrag.target === canvas && endPlanDrag(e)) {
        canvas.style.cursor = "";
        // A press on the selected feature that never moved is still a click: keep it selected.
        return;
      }
      if (editDrag && editDrag.pointerId === e.pointerId) {
        const box = placementBox(placementById(editDrag.id));
        const point = box ? [box.center[0], box.center[1], box.center[2]] : null;
        if (editDrag.moved && point) workspace.emit("placement-move", { id: editDrag.id, point, final: true });
        else if (!editDrag.moved) workspace.emit("placement-pick", { id: editDrag.id });
        if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
        editDrag = null;
        return;
      }
      if (!gesture || gesture.id !== e.pointerId) return;
      const click = !gesture.moved && !gesture.pan && gesture.button === 0 &&
        Math.hypot(e.clientX - gesture.startX, e.clientY - gesture.startY) <= 4;
      gesture = null;
      if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
      if (click) {
        if (planSelect && !workspacePick) planPickAt(e.clientX, e.clientY);
        else pickWorkspace(e.clientX, e.clientY);
      }
    });
    const cancelGesture = () => {
      if (planDrag) { endPlanDrag(null, true); canvas.style.cursor = ""; }
      if (editDrag) {
        // A cancelled drag puts the piece back where the grab started.
        const back = dragPlacement(editDrag.id, editDrag.from);
        if (back) workspace.emit("placement-move", { id: editDrag.id, point: back, final: true });
        editDrag = null;
      }
      if (gesture && canvas.hasPointerCapture(gesture.id)) canvas.releasePointerCapture(gesture.id);
      gesture = null;
      setPreviewPoint(null);
    };
    canvas.addEventListener("pointercancel", cancelGesture);
    canvas.addEventListener("lostpointercapture", () => { gesture = null; setPreviewPoint(null); });
    window.addEventListener("blur", cancelGesture);
    document.addEventListener("visibilitychange", () => { if (document.hidden) cancelGesture(); });
    canvas.addEventListener("wheel", e => {
      e.preventDefault();
      if (!workspace.ready || workspaceMode !== "orbit") return;
      setPreviewPoint(null);
      const delta = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? canvas.clientHeight : 1);
      workspaceOrbit = zoomOrbit(workspaceOrbit, delta);
      applyWorkspaceOrbit();
    }, { passive: false });
    window.addEventListener("keydown", e => {
      if (e.code === "Escape" && planDrag) { endPlanDrag(null, true); canvas.style.cursor = ""; e.stopPropagation(); return; }
      if (["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.code)) e.preventDefault();
      if (e.code === "KeyR" && workspace.ready && !e.repeat) {
        try { fitWorkspace(); workspace.state(); }
        catch (error) { workspace.emit("error", { message: error.message }); }
      }
    });
    canvas.addEventListener("webglcontextlost", e => {
      e.preventDefault();
      clearActiveKeys();
      workspacePick = false;
      window.__ready = false;
      window.__loadError = "WebGL context lost; reload the viewer to continue";
      workspace.fail(window.__loadError);
      setLoad(window.__loadError, true);
    });
    return; // No legacy pointer lock, combat, tour, or inspector shortcuts in the workspace.
  }
  let isDragging = false;
  let lastMouseX = 0, lastMouseY = 0;

  canvas.addEventListener("mousedown", (e) => {
    if (e.button === 0) {
      isDragging = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      canvas.requestPointerLock();
      if (!window.__combatActive) pickCameraFromRay(e.clientX, e.clientY);
    }
  });
  window.addEventListener("mouseup", () => { isDragging = false; });
  window.addEventListener("mousemove", (e) => {
    let dx = 0, dy = 0;
    if (document.pointerLockElement === canvas) {
      dx = e.movementX;
      dy = e.movementY;
    } else if (isDragging) {
      dx = e.clientX - lastMouseX;
      dy = e.clientY - lastMouseY;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    }
    if (dx || dy) {
      P.yaw -= dx * 0.0025;
      P.pitch = Math.min(Math.max(P.pitch - dy * 0.0022, -1.45), 1.45);
    }
  });

  window.addEventListener("keydown", (e) => {
    if (COMBAT && window.__combatActive && COMBAT_KEYS.includes(e.code)) return;
    if (e.code === "KeyF" || e.key === "f" || e.key === "F") toggleDroneMode();
    if (e.code === "KeyG" || e.key === "g" || e.key === "G") toggleSplatVisibility();
    if (e.code === "KeyP" || e.key === "p" || e.key === "P") toggleCamerasLayer();
    if (e.code === "KeyO" || e.key === "o" || e.key === "O") togglePointsLayer();
    if (e.code === "KeyV" || e.key === "v" || e.key === "V") toggleCoverageLayer();
    if (e.code === "KeyX" || e.key === "x" || e.key === "X") toggleCollider();
    if (e.code === "KeyU" || e.key === "u" || e.key === "U") toggleSplatModel();
    if (e.code === "KeyC" || e.key === "c" || e.key === "C") { P.firstPerson = !P.firstPerson; updateToolbarUI(); }
    if (e.code === "KeyJ" || e.key === "j" || e.key === "J") snapToSelectedCamera();
    if (e.code === "BracketLeft") inspectCamera(selectedCamIdx >= 0 ? selectedCamIdx - 1 : 0);
    if (e.code === "BracketRight") inspectCamera(selectedCamIdx >= 0 ? selectedCamIdx + 1 : 0);
    if ((e.code === "KeyT" || e.key === "t" || e.key === "T") && autopilot.phase === "idle") {
      if (isDrone) toggleDroneMode(false);
      autopilot.phase = "settle"; autopilot.t = 0;
    }
    if (e.code === "KeyR" || e.key === "r" || e.key === "R") {
      if (isDrone) toggleDroneMode(false);
      spawnFrom(col);
    }
  });

}

// ---------------- per-frame physics & fly glue ----------------
const fwdOf = (yaw) => [-Math.sin(yaw), -Math.cos(yaw)];
/**
 * Metres from the player entity's origin DOWN to the floor its capsule stands on.
 * Provisional: PlayCanvas hands Bullet `height - 2*radius` as a HALF height, so a
 * `height: 1.8` capsule is not obviously 1.8 m overall, and the grounding test,
 * the step probe and the character model's feet all read this one number.
 * measureFeet() replaces it with what the collider actually rests at.
 */
let FEET = 0.9 * CHAR_SCALE;
let feetMeasured = false, feetSamples = [];
function measureFeet(pos, hitY, vy) {
  // One frame is not a measurement: at the top of the spawn drop the capsule is
  // momentarily calm and 2 m off the floor, which would seat the model in mid-air.
  if (feetMeasured || Math.abs(vy) > 0.15 || !isFinite(hitY) || hitY < -1e8 ||
      Math.abs(pos.y - hitY - FEET) > 1.0) { feetSamples = []; return; }
  feetSamples.push(pos.y - hitY);
  if (feetSamples.length < 12) return;
  const sorted = feetSamples.slice().sort((a, b) => a - b);
  const spread = sorted[sorted.length - 1] - sorted[0];
  if (spread > 0.06) { feetSamples.shift(); return; }   // still ringing
  FEET = sorted[6];
  feetMeasured = true;
  feetSamples = [];
  playerChar?.seat(-FEET);
  console.log(`[player] the capsule stands ${FEET.toFixed(2)} m below its origin ` +
              `(provisional 0.90), i.e. a ${(2 * FEET).toFixed(2)} m tall collider; ` +
              `the character is seated on this`);
}
// The probes that decide whether the body is standing on anything. Every distance
// below is written for a 1.75 m human and multiplied by CHAR_SCALE, like the
// follow-cam offsets: in absolute metres a room-scale mover (FEET ~0.03 m) has its
// ground ray starting 0.8 m under its own feet, i.e. below the floor, so it reads
// airborne forever and every branch gated on grounding stops firing.
function groundRay() {
  const p = playerEnt.getPosition();
  return app.systems.rigidbody.raycastFirst(
    new Vec3(p.x, p.y - 0.80 * CHAR_SCALE, p.z),
    new Vec3(p.x, p.y - 2.6 * CHAR_SCALE, p.z),
    { filterCollisionMask: BODYMASK_STATIC });
}

/**
 * The same question as groundRay, cast from above the origin, because that one
 * starts below it and would find no floor at all if the capsule turns out to
 * stand nearer than that. Only the spawn-time measurement uses this.
 */
function feetProbe(pos) {
  return app.systems.rigidbody.raycastFirst(
    new Vec3(pos.x, pos.y + 0.3 * CHAR_SCALE, pos.z),
    new Vec3(pos.x, pos.y - 3.0 * CHAR_SCALE, pos.z),
    { filterCollisionMask: BODYMASK_STATIC });
}

const STEP_MAX = 0.5;
function stepBlocked(dx, dz, pos) {
  const l = Math.hypot(dx, dz);
  if (l < 0.1 * CHAR_SCALE) return false;
  // 0.34 is the capsule radius before it is scaled, so the whole probe - how far
  // ahead it looks and how high a riser still counts as climbable - belongs to the
  // body, not to the metre.
  const nx = dx / l, nz = dz / l,
        reach = (0.34 + 0.30) * CHAR_SCALE, feet = pos.y - FEET;
  const probe = (y) => app.systems.rigidbody.raycastFirst(
    new Vec3(pos.x, y, pos.z), new Vec3(pos.x + nx * reach, y, pos.z + nz * reach),
    { filterCollisionMask: BODYMASK_STATIC });
  return !!probe(feet + 0.06 * CHAR_SCALE) &&
         !probe(feet + (STEP_MAX + 0.10) * CHAR_SCALE);
}

function step(dt) {
  // Orbit is event-driven. Never overwrite a fitted or selected pose in the fly loop.
  if (EMBED && (!workspace.ready || workspaceMode === "orbit")) return;
  if (SHOOT) {
    const tp0 = playerEnt.getPosition();
    const w0 = window.__walk;
    w0.pos = [+tp0.x.toFixed(3), +(tp0.y - 0.9).toFixed(3), +tp0.z.toFixed(3)];
    return;
  }

  // Drone 6-DOF Fly Mode
  if (isDrone) {
    const cy = Math.cos(P.yaw), sy = Math.sin(P.yaw);
    const cp = Math.cos(P.pitch), sp = Math.sin(P.pitch);
    const fx = -sy * cp, fy = sp, fz = -cy * cp;
    const rx = cy, rz = -sy;

    const isW = !!(activeKeys["KeyW"] || activeKeys["w"] || activeKeys["ArrowUp"]);
    const isS = !!(activeKeys["KeyS"] || activeKeys["s"] || activeKeys["ArrowDown"]);
    const isA = !!(activeKeys["KeyA"] || activeKeys["a"] || activeKeys["ArrowLeft"]);
    const isD = !!(activeKeys["KeyD"] || activeKeys["d"] || activeKeys["ArrowRight"]);
    const isUp = !!(activeKeys["KeyE"] || activeKeys["e"] || activeKeys["Space"] || activeKeys[" "]);
    const isDown = !!(activeKeys["KeyQ"] || activeKeys["q"]);
    const isShift = !!(activeKeys["ShiftLeft"] || activeKeys["ShiftRight"] || activeKeys["shift"]);
    const isSlow = !!(activeKeys["AltLeft"] || activeKeys["AltRight"] || activeKeys["ControlLeft"] || activeKeys["ControlRight"]);

    let mx = 0, my = 0, mz = 0;
    if (isW) { mx += fx; my += fy; mz += fz; }
    if (isS) { mx -= fx; my -= fy; mz -= fz; }
    if (isD) { mx += rx; mz += rz; }
    if (isA) { mx -= rx; mz -= rz; }
    if (isUp) { my += 1.0; }
    if (isDown) { my -= 1.0; }

    const speed = isShift ? droneSpeed * 2.5 : (isSlow ? droneSpeed * 0.3 : droneSpeed);
    const len = Math.hypot(mx, my, mz);
    if (len > 0.0001) {
      const stepDist = (speed * dt) / len;
      dronePos.x += mx * stepDist;
      dronePos.y += my * stepDist;
      dronePos.z += mz * stepDist;
    }

    if (!EMBED || len > 0.0001 || workspaceLookDirty) {
      cameraEnt.setPosition(dronePos.x, dronePos.y, dronePos.z);
      cameraEnt.lookAt(new Vec3(dronePos.x + fx * 5, dronePos.y + fy * 5, dronePos.z + fz * 5));
      workspaceLookDirty = false;
    }

    const w = window.__walk;
    w.pos = [+dronePos.x.toFixed(3), +dronePos.y.toFixed(3), +dronePos.z.toFixed(3)];
    return;
  }

  // Ground Walk Mode
  const rb = playerRb;
  const tp = playerEnt.getPosition();
  const pos = new Vec3(tp.x, tp.y, tp.z);
  const v0 = rb.linearVelocity;
  const v = new Vec3(v0.x, v0.y, v0.z);

  let onMesh = null;
  const hit = groundRay();
  const probe = feetProbe(pos);
  measureFeet(pos, probe ? probe.point.y : -1e9, v.y);
  // 0.35 m is what a human's feet may hang above the floor it is standing on, so
  // the same fraction of the body applies at every scale.
  if (hit) onMesh = (pos.y - FEET - hit.point.y) < 0.35 * CHAR_SCALE;
  P.grounded = !!onMesh;

  let mx = 0, mz = 0;
  if (autopilot.phase !== "idle") {
    autopilotStep(dt);
    if (autopilot.moving) mz = 1;
  } else {
    if (activeKeys["KeyW"] || activeKeys["w"] || activeKeys["ArrowUp"]) mz += 1;
    if (activeKeys["KeyS"] || activeKeys["s"] || activeKeys["ArrowDown"]) mz -= 1;
    if (activeKeys["KeyA"] || activeKeys["a"] || activeKeys["ArrowLeft"]) mx -= 1;
    if (activeKeys["KeyD"] || activeKeys["d"] || activeKeys["ArrowRight"]) mx += 1;
  }
  let sp = 0, dx = 0, dz = 0;
  if (!P.extControl) {
    if (mx || mz) {
      const l = Math.hypot(mx, mz); mx /= l; mz /= l;
      sp = (activeKeys["ShiftLeft"] || activeKeys["ShiftRight"] || activeKeys["shift"]) ? RUN_SPEED : WALK_SPEED;
      const s = Math.sin(P.yaw), c = Math.cos(P.yaw);
      dx = (mz * -s + mx * c) * sp;
      dz = (mz * -c + mx * -s) * sp;
    }
    // In VR the left stick walks where the head looks; the capsule and navmesh are the same.
    if (vrState?.active && vrState.move.mag > 0) {
      sp = WALK_SPEED * vrState.move.mag;
      dx = (vrState.move.dx / vrState.move.mag) * sp;
      dz = (vrState.move.dz / vrState.move.mag) * sp;
    }
    const hs = Math.hypot(v.x, v.z);
    const keep = sp > 0 && hs > sp + 0.5 * CHAR_SCALE;
    let vy = v.y;
    // These two impulses are what a human capsule needs to clear a step or leave
    // the ground; unscaled they would throw a 3 cm mover out of the room, and
    // nothing gated on grounding could be trusted to stay on the floor.
    if (P.grounded && (dx || dz) && stepBlocked(dx, dz, pos)) vy = 4.24 * CHAR_SCALE;
    rb.linearVelocity = new Vec3(keep ? v.x : dx, vy, keep ? v.z : dz);
    rb.activate();
  }

  if (P.grounded && (activeKeys["Space"] || activeKeys[" "]) && autopilot.phase === "idle") {
    rb.linearVelocity = new Vec3(dx, 6.5 * CHAR_SCALE, dz);
  }

  // Cadence follows the body, not the keys: normalised by WALK_SPEED, one clip
  // cycle per second at walking pace, up to ~2x when sprinting. Reading the key
  // state instead would keep the legs moving while the capsule is pinned.
  playerChar?.setSpeed(Math.hypot(rb.linearVelocity.x, rb.linearVelocity.z));

  const px = playerEnt.getPosition();
  if (P.prev !== null) {
    // Count only grounded travel: a fall, a jump, the fall-recovery teleport or
    // wall-grinding jitter is not distance walked, and summing them is what made
    // the walk report more metres than the route actually covered.
    P.walked += walkDistance([P.prev.x, P.prev.z], [px.x, px.z], P.grounded, 1.0 * CHAR_SCALE);
  }
  P.prev = { x: px.x, z: px.z };

  // How far below the heightfield counts as fallen out of the world, in body
  // heights: at 4 m absolute a room-scale capsule can sink through the floor and
  // keep reporting zero falls, which makes the shipped count worth nothing.
  if (px.y < groundHF(px.x, px.z) - 4.0 * CHAR_SCALE) {
    P.violations++;
    const g = P.lastGood || { x: window.__spawn?.x ?? 0, y: groundHF(window.__spawn?.x ?? 0, window.__spawn?.z ?? 0) + 1.5 * CHAR_SCALE, z: window.__spawn?.z ?? 0 };
    playerEnt.rigidbody.teleport(g.x, g.y + 1.0 * CHAR_SCALE, g.z);
    playerRb.linearVelocity = new Vec3(0, 0, 0);
  }
  if (P.grounded && px.x > HF.minX && px.x < HF.maxX && px.z > HF.minZ && px.z < HF.maxZ) {
    P.lastGood = { x: px.x, y: px.y, z: px.z };
  }

  playerEnt.setLocalEulerAngles(0, P.yaw * 180 / Math.PI, 0);
  const [fx, fz] = fwdOf(P.yaw);
  // Every follow-cam offset is expressed for a 1.75 m human and divided out by
  // CHAR_SCALE, so a 0.15 m hamster in a room scan gets a hamster's eye line
  // and a hamster's shoulder-room behind it. Without this the camera hovers
  // 2 m over the head and looks down on the scene like a drone.
  const eye_fwd = 0.12 * CHAR_SCALE;
  const eye_up = 0.62 * CHAR_SCALE;
  const chase_back = 3.4 * CHAR_SCALE;
  const chase_up = 2.0 * CHAR_SCALE;
  const chase_min_clear = 0.4 * CHAR_SCALE;
  const look_up_fp = 0.55 * CHAR_SCALE;
  const look_up_tp = 1.15 * CHAR_SCALE;
  if (vrState?.active) {
    // The headset owns the eyes: the rig stands at the feet and turns with the body.
    vrState.update(new Vec3(px.x, px.y - FEET, px.z), P.yaw);
    if (vrState.turn) P.yaw += vrState.turn;
    const w = window.__walk;
    w.pos = [+px.x.toFixed(3), +(px.y - FEET).toFixed(3), +px.z.toFixed(3)];
    return;
  }
  if (P.firstPerson) {
    cameraEnt.setPosition(px.x + fx * eye_fwd, px.y + eye_up, px.z + fz * eye_fwd);
  } else {
    const cx = px.x - fx * chase_back, cz = px.z - fz * chase_back;
    const cy = Math.max(px.y + chase_up, groundHF(cx, cz) + chase_min_clear);
    cameraEnt.setPosition(cx, cy, cz);
  }
  const look = new Vec3(px.x + fx * 5, px.y + (P.firstPerson ? look_up_fp : look_up_tp) + Math.tan(P.pitch) * 5, px.z + fz * 5);
  cameraEnt.lookAt(look);
  const w = window.__walk;
  w.pos = [+px.x.toFixed(3), +(px.y - FEET).toFixed(3), +px.z.toFixed(3)];
}

function autopilotStep(dt) {
  const a = autopilot;
  a.t += dt;
  const path = window.__walkPath;
  const steerTo = (tx, tz) => {
    const dx = tx - playerEnt.getPosition().x, dz = tz - playerEnt.getPosition().z;
    if (Math.hypot(dx, dz) < 1.5) return true;
    const want = Math.atan2(-dx, -dz);
    let dyaw = Math.atan2(Math.sin(want - P.yaw), Math.cos(want - P.yaw));
    P.yaw += Math.max(-2.5 * dt, Math.min(2.5 * dt, dyaw));
    return false;
  };
  const legStart = () => {
    a.walkStart = P.walked; a.wpIdx = 0; a.stuckT = 0; a.unstick = 0; a.wpT = 0;
  };
  const followPath = (dt2) => {
    if (!path?.length) return;
    const t = path[a.wpIdx % path.length];
    a.wpT += dt2;
    if (steerTo(t[0], t[1]) || a.wpT > 12) { a.wpIdx++; a.wpT = 0; }
  };

  const px2 = playerEnt.getPosition();
  const instSpeed = a.prevPos
    ? Math.hypot(px2.x - a.prevPos.x, px2.z - a.prevPos.z) / Math.max(dt, 1e-4) : 9;
  a.prevPos = { x: px2.x, z: px2.z };
  const walking = a.phase === "walk" || a.phase === "walk2";
  if (walking && instSpeed < 0.35) a.stuckT += dt; else a.stuckT = 0;
  if (a.stuckT > 1.2) { a.unstick = 1.4; a.stuckT = 0; }

  a.moving = false;
  if (a.phase === "settle") {
    if (a.t > 1.0) { a.phase = "walk"; a.t = 0; legStart(); }
  } else if (a.phase === "walk") {
    a.moving = true;
    if (a.unstick > 0) {
      a.unstick -= dt;
      P.yaw += 2.2 * dt;
    } else followPath(dt);
    if ((a.walkStart !== undefined && P.walked - a.walkStart >= 50) || a.t > 120) {
      a.phase = "spin"; a.t = 0; a.spinFrom = P.yaw;
    }
  } else if (a.phase === "spin") {
    P.yaw = a.spinFrom + (a.t / 4) * Math.PI * 4;
    if (a.t >= 4) { P.yaw = a.spinFrom; a.phase = "walk2"; a.t = 0; legStart(); }
  } else if (a.phase === "walk2") {
    a.moving = true;
    if (a.unstick > 0) {
      a.unstick -= dt;
      P.yaw += 2.2 * dt;
    } else followPath(dt);
    if ((a.walkStart !== undefined && P.walked - a.walkStart >= 15) || a.t > 45) {
      a.phase = "done";
    }
  }
}

function buildUnderlay() {
  if (!UNDERLAY || !HF) return;
  const { nx, nz, cell, ox, oz } = HF;
  const stride = 3;
  const hs = new Float32Array(nx * nz).fill(NaN);
  for (let gz = 0; gz < nz; gz += stride) {
    for (let gx = 0; gx < nx; gx += stride) {
      const x = ox + gx * cell, z = oz + gz * cell;
      const hit = groundProbe(x, z);
      if (hit) hs[gz * nx + gx] = hit.point.y;
    }
  }
  for (let i = 0; i < hs.length; i++) {
    if (!isFinite(hs[i])) hs[i] = HF.data[i] - SINK - 0.06; else hs[i] -= SINK;
  }
  const sm = new Float32Array(hs.length);
  for (let gz = 0; gz < nz; gz++) {
    for (let gx = 0; gx < nx; gx++) {
      let sum = 0, n = 0;
      for (let dz2 = -1; dz2 <= 1; dz2++) {
        for (let dx2 = -1; dx2 <= 1; dx2++) {
          const j = Math.min(nz - 1, Math.max(0, gz + dz2)) * nx +
                    Math.min(nx - 1, Math.max(0, gx + dx2));
          sum += hs[j]; n++;
        }
      }
      sm[gz * nx + gx] = sum / n;
    }
  }
  const pos = new Float32Array(nx * nz * 3);
  const colr = new Float32Array(nx * nz * 3);
  for (let gz = 0; gz < nz; gz++) {
    for (let gx = 0; gx < nx; gx++) {
      const i = gz * nx + gx;
      pos[i * 3] = ox + (gx + 0.5) * cell;
      pos[i * 3 + 1] = sm[i];
      pos[i * 3 + 2] = oz + (gz + 0.5) * cell;
      const c = HF.colors ? i * 3 : 0;
      colr[i * 3] = Math.pow((HF.colors ? HF.colors[c] : 122) / 255, 2.2);
      colr[i * 3 + 1] = Math.pow((HF.colors ? HF.colors[c + 1] : 116) / 255, 2.2);
      colr[i * 3 + 2] = Math.pow((HF.colors ? HF.colors[c + 2] : 88) / 255, 2.2);
    }
  }
  const idx = [];
  const cov = HF.cov;
  for (let gz = 0; gz < nz - 1; gz++) {
    for (let gx = 0; gx < nx - 1; gx++) {
      const a = gz * nx + gx, b = a + 1, c2 = a + nx, d = c2 + 1;
      if (cov && !(cov[a] && cov[b] && cov[c2] && cov[d])) continue;
      idx.push(a, c2, b, b, c2, d);
    }
  }
  if (!idx.length) return;
  const mat = unlitMat();
  mat.diffuseVertexColor = true;
  mat.update();
  const mesh = new Mesh(app.graphicsDevice);
  mesh.setPositions(pos);
  mesh.setColors(colr, 3);
  mesh.setIndices(idx);
  mesh.update(PRIMITIVE_TRIANGLES);
  const ent = new Entity("underlay");
  ent.addComponent("render", { meshInstances: [new MeshInstance(mesh, mat)] });
  app.root.addChild(ent);
}

boot().catch((e) => {
  window.__bootStack = String((e && e.stack) || e);
  fail(e && e.message ? e.message : e);
});
