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
 */
import {
  AppBase, AppOptions, Asset, AssetListLoader, BODYMASK_STATIC, BoundingBox,
  CameraComponentSystem, CollisionComponentSystem, Color, ContainerHandler,
  Entity, FILLMODE_FILL_WINDOW,
  FOG_LINEAR, GSplatComponentSystem, GSplatHandler, KEY_A, KEY_C, KEY_D, KEY_R,
  KEY_S, KEY_SHIFT, KEY_SPACE, KEY_T, KEY_W, Keyboard, Mesh, MeshInstance, Mouse,
  PRIMITIVE_LINES, PRIMITIVE_POINTS, PRIMITIVE_TRIANGLES, RenderComponentSystem, RESOLUTION_AUTO,
  RigidBodyComponentSystem, StandardMaterial, TONEMAP_LINEAR, TextureHandler,
  Vec3, WasmModule, createGraphicsDevice,
} from "playcanvas";

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

const AUTO = q.get("auto") === "1";
const SHOOT = q.get("shoot") === "1";
const UNDERLAY = q.get("underlay") === "1";
const SINK = Number(q.get("sink") ?? 0.7);

const loadEl = document.getElementById("load");
const hudEl = document.getElementById("hud");
const setLoad = (msg, err = false) => {
  if (loadEl) {
    loadEl.textContent = msg;
    loadEl.classList.toggle("err", err);
  }
};
const fail = (msg) => {
  window.__loadError = String(msg);
  setLoad("ERROR: " + msg, true);
  throw new Error(String(msg));
};

// ---------------- drone / splat / camera state ----------------
let isDrone = q.get("drone") === "1";
let splatVisible = true;
let currentPly = q.get("full") === "1" ? "scene.full.ply" : (q.get("ply") || "scene.ply");
const dronePos = new Vec3(0, 0, 0);
let droneSpeed = 7.0;
let splatEnt = null;
let applicationRef = null;
const activeKeys = {};

// Visualizer layer states
let showCameras = q.get("cams") !== "0"; // default ON
let showPoints = false;
let showCoverage = false;
let allCameras = [];
let allSparsePoints = null;
let allCoverageGrid = null;
let selectedCamIdx = -1;

let camerasEntity = null;
let trajectoryEntity = null;
let selectedCamEntity = null;
let sparsePointsEntity = null;
let coverageGridEntity = null;

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
const WALK_SPEED = 2.6, RUN_SPEED = 5.2;
let HF = null;

async function loadSceneData() {
  const resp = await fetch(`${ASSET}/collision.json`);
  if (!resp.ok) {
    fail(`Could not load ${ASSET}/collision.json (HTTP ${resp.status}). Please verify scene assets exist.`);
  }
  const col = await resp.json();
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

  HF = {
    nx, nz, cell, data, colors, cov,
    ox: origin_xz[0], oz: origin_xz[1],
    minX: origin_xz[0] - 2, maxX: origin_xz[0] + nx * cell + 2,
    minZ: origin_xz[1] - 2, maxZ: origin_xz[1] + nz * cell + 2,
  };

  // Load cameras.json, sparse_points.json, coverage_grid.json in parallel
  try {
    const [camsResp, ptsResp, covResp] = await Promise.allSettled([
      fetch(`${ASSET}/cameras.json`),
      fetch(`${ASSET}/sparse_points.json`),
      fetch(`${ASSET}/coverage_grid.json`)
    ]);
    if (camsResp.status === "fulfilled" && camsResp.value.ok) {
      allCameras = await camsResp.value.json();
      console.log(`[viewer] Loaded ${allCameras.length} camera poses`);
    }
    if (ptsResp.status === "fulfilled" && ptsResp.value.ok) {
      allSparsePoints = await ptsResp.value.json();
      console.log(`[viewer] Loaded ${allSparsePoints.count || 0} sparse tie points`);
    }
    if (covResp.status === "fulfilled" && covResp.value.ok) {
      allCoverageGrid = await covResp.value.json();
      console.log(`[viewer] Loaded coverage grid (${allCoverageGrid.total_voxels} voxels)`);
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
let cameraEnt, playerEnt, playerRb;

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
window.__walk = { phase: "idle", walked: 0, pos: [0, 0, 0], yaw: 0, grounded: false, violations: 0, samples: [] };

function spawnFrom(col) {
  const s = window.__chosenSpawn || col.spawn || { x: 0, z: 0, face_xz: [0, 1] };
  const hit = groundProbe(s.x, s.z);
  const y = hit ? hit.point.y + 1.5 : groundHF(s.x, s.z) + 2;
  if (playerEnt && playerRb) {
    playerEnt.rigidbody.teleport(s.x, y, s.z);
    playerRb.linearVelocity = new Vec3(0, 0, 0);
  }
  P.yaw = Math.atan2(-(s.face_xz[0] - s.x), -(s.face_xz[1] - s.z));
  P.walked = 0; P.violations = 0; P.lastGood = null; P.prev = null;
  autopilot.phase = AUTO ? "settle" : "idle";
  autopilot.t = 0;
  dronePos.set(s.x, y + 0.5, s.z);
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
  window.__stage = "data";
  setLoad("Loading terrain, cameras & collision metadata…");
  const col = await loadSceneData();
  window.__spawn = col.spawn || null;
  window.__walkPath = col.walk_path || null;

  if (col.spawn) {
    dronePos.set(col.spawn.x, groundHF(col.spawn.x, col.spawn.z) + 1.8, col.spawn.z);
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

  const device = await createGraphicsDevice(canvas, { deviceTypes: ["webgl2"], antialias: false });
  const opts = new AppOptions();
  opts.graphicsDevice = device;
  opts.componentSystems = [
    RenderComponentSystem, CameraComponentSystem, CollisionComponentSystem,
    RigidBodyComponentSystem, GSplatComponentSystem,
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

  const assets = {
    splat: new Asset("splat", "gsplat", { url: `${ASSET}/${currentPly}` }),
    collision: new Asset("collision", "container", { url: `${ASSET}/../pc/collision.collision.glb` }),
  };
  window.__stage = "assets";
  setLoad(`Downloading 3D splat & collision model…`);
  await new Promise((resolve, reject) => {
    new AssetListLoader(Object.values(assets), application.assets).load(
      (err) => (err ? reject(err) : resolve()));
  });
  window.__stage = "start";
  setLoad("Rendering 3D scene & camera frustums…");

  application.scene.fog.type = FOG_LINEAR;
  application.scene.fogColor = SKY.clone();
  application.scene.fogStart = 35;
  application.scene.fogEnd = 120;
  application.scene.ambientLight = new Color(1, 1, 1);

  application.start();

  // 1. Splat Entity
  splatEnt = new Entity("splat");
  splatEnt.addComponent("gsplat", { asset: assets.splat });
  application.root.addChild(splatEnt);

  // 2. Collision Trimesh
  const colRoot = assets.collision.resource.instantiateRenderEntity();
  application.root.addChild(colRoot);
  let nCol = 0;
  colRoot.findComponents("render").forEach((render) => {
    render.entity.addComponent("rigidbody", { type: "static", friction: 0.6, restitution: 0 });
    render.entity.addComponent("collision", { type: "mesh", renderAsset: render.asset });
    render.enabled = q.get("showcol") === "1";
    nCol++;
  });

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

  // 6. Character Entity
  const orange = unlitMat(1, 0.44, 0.19);
  const skin = unlitMat(1, 0.85, 0.69);

  playerEnt = new Entity("player");
  const body = new Entity("body");
  body.addComponent("render", { type: "capsule", material: orange });
  body.setLocalScale(0.56, 0.9, 0.56); body.setLocalPosition(0, 0.78, 0);
  const head = new Entity("head");
  head.addComponent("render", { type: "sphere", material: skin });
  head.setLocalScale(0.36, 0.36, 0.36); head.setLocalPosition(0, 1.48, 0);
  playerEnt.addChild(body); playerEnt.addChild(head);
  playerEnt.addComponent("collision", { type: "capsule", radius: 0.34, height: 1.8 });
  playerRb = playerEnt.addComponent("rigidbody", {
    type: "dynamic", mass: 80, linearDamping: 0.05, angularDamping: 1,
    friction: 0.35, restitution: 0,
  });
  playerRb.angularFactor = new Vec3(0, 0, 0);
  application.root.addChild(playerEnt);

  // 7. Viewer Camera Entity
  cameraEnt = new Entity("camera");
  cameraEnt.addComponent("camera", {
    fov: 70, nearClip: 0.08, farClip: 3000, clearColorBuffer: true,
  });
  cameraEnt.camera.clearColor = new Color(SKY.r, SKY.g, SKY.b, 1);
  cameraEnt.camera.toneMapping = TONEMAP_LINEAR;
  application.root.addChild(cameraEnt);

  // ---------------- UI and Inspector Wiring ----------------
  function updateToolbarUI() {
    const btnMode = document.getElementById("btn-mode");
    const lblMode = document.getElementById("lbl-mode");
    const btnCams = document.getElementById("btn-cams");
    const btnPoints = document.getElementById("btn-points");
    const btnCov = document.getElementById("btn-cov");
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
    if (btnCov) {
      btnCov.classList.toggle("active", showCoverage);
    }
    if (btnSplatVis && lblSplatVis) {
      btnSplatVis.classList.toggle("active", splatVisible);
      lblSplatVis.textContent = splatVisible ? "✨ Splats: ON" : "🚫 Splats: OFF";
    }
    if (btnSplat && lblSplat) {
      const isFull = currentPly === "scene.full.ply";
      btnSplat.classList.toggle("active", isFull);
      lblSplat.textContent = isFull ? "📦 Splat: Full (932k)" : "✨ Splat: Clean (672k)";
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
          dronePos.set(window.__spawn.x, groundHF(window.__spawn.x, window.__spawn.z) + 1.8, window.__spawn.z);
        } else {
          dronePos.set(p.x, p.y + 0.8, p.z);
        }
        playerEnt.enabled = false;
        if (playerRb) {
          playerRb.type = "kinematic";
          playerRb.linearVelocity = new Vec3(0, 0, 0);
        }
      } else if (window.__spawn) {
        dronePos.set(window.__spawn.x, groundHF(window.__spawn.x, window.__spawn.z) + 1.8, window.__spawn.z);
      }
    } else {
      if (playerEnt) {
        playerEnt.enabled = true;
        if (playerRb) {
          playerRb.type = "dynamic";
          const hit = groundProbe(dronePos.x, dronePos.z);
          const targetY = hit ? hit.point.y + 1.5 : (groundHF(dronePos.x, dronePos.z) + 1.5);
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

  function toggleCoverageLayer() {
    showCoverage = !showCoverage;
    if (coverageGridEntity) coverageGridEntity.enabled = showCoverage;
    const covPanel = document.getElementById("cov-panel");
    if (covPanel) {
      covPanel.style.display = showCoverage ? "flex" : "none";
      if (showCoverage && allCoverageGrid) {
        const statsEl = document.getElementById("cov-stats");
        const adviceEl = document.getElementById("cov-advice");
        if (statsEl) {
          statsEl.innerHTML = `
            <div class="stat-row"><span>Total Volume Voxels:</span> <b>${allCoverageGrid.total_voxels}</b></div>
            <div class="stat-row"><span>Observed (>=3 views):</span> <b style="color:#68d391">${allCoverageGrid.covered_pct}%</b></div>
            <div class="stat-row"><span>Marginal (1-2 views):</span> <b style="color:#f6ad55">${allCoverageGrid.marginal_pct}%</b></div>
            <div class="stat-row"><span>Unobserved (Missing):</span> <b style="color:#fc8181">${allCoverageGrid.unobserved_pct}%</b></div>
          `;
        }
        if (adviceEl && allCoverageGrid.advice) {
          adviceEl.innerHTML = allCoverageGrid.advice.map(a => `<div>• ${a}</div>`).join("");
        }
      }
    }
    updateToolbarUI();
  }

  async function toggleSplatModel() {
    const target = currentPly === "scene.full.ply" ? "scene.ply" : "scene.full.ply";
    await setSplatModel(target);
  }

  async function setSplatModel(filename) {
    if (currentPly === filename && splatEnt?.gsplat?.asset) return;
    setLoad(`Loading 3D Gaussian Splat (${filename === "scene.full.ply" ? "Full 1.63M Unculled" : "Cleaned 671k"})…`);
    loadEl.style.opacity = "1";
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
      loadEl.style.opacity = "0";
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
  application.on("update", (dtRaw) => {
    const dt = Math.min(dtRaw, 0.05);
    step(dt);

    const w = window.__walk;
    w.walked = +P.walked.toFixed(2);
    w.phase = autopilot.phase;
    w.grounded = P.grounded;
    w.violations = P.violations;
    w.yaw = +P.yaw.toFixed(3);

    const splatLabel = !splatVisible
      ? "OFF (Hidden)"
      : (currentPly === "scene.full.ply" ? "Full (932k)" : "Clean (672k)");
    const camStatus = showCameras ? `ON (${allCameras.length} cams)` : "OFF";
    const covStatus = allCoverageGrid ? `${allCoverageGrid.covered_pct}% cov` : "N/A";

    if (isDrone) {
      const isFast = !!(activeKeys["ShiftLeft"] || activeKeys["ShiftRight"] || activeKeys["shift"]);
      const isSlow = !!(activeKeys["AltLeft"] || activeKeys["AltRight"] || activeKeys["ControlLeft"] || activeKeys["ControlRight"]);
      const curSpeed = isFast ? droneSpeed * 2.5 : (isSlow ? droneSpeed * 0.3 : droneSpeed);
      hudEl.textContent =
        `[🚁 DRONE FLY]   Speed: ${curSpeed.toFixed(1)} m/s   Splat: ${splatLabel}   Cams: ${camStatus}   Coverage: ${covStatus}\n` +
        `pos: (${dronePos.x.toFixed(2)}, ${dronePos.y.toFixed(2)}, ${dronePos.z.toFixed(2)})\n` +
        `WASD 3D Fly | G Splats ON/OFF | P Cams | O Points | V Coverage | J Snap | U Model`;
    } else {
      hudEl.textContent =
        `[🚶 GROUND WALK]   ${P.firstPerson ? "1st Person" : "3rd Person"}   walked ${P.walked.toFixed(1)} m   falls ${P.violations}\n` +
        `pos: (${w.pos.join(", ")})   ${P.grounded ? "grounded" : "air"}   Splat: ${splatLabel}   Cams: ${camStatus}\n` +
        `WASD Walk | Shift Run | G Splats ON/OFF | P Cams | O Points | V Coverage | F Drone | C View`;
    }
  });

  application.on("postrender", () => {
    frames++;
    if (frames === 3 && !window.__ready) {
      window.__ready = true;
      setLoad("");
      loadEl.style.opacity = "0";
      setTimeout(() => loadEl.remove(), 500);
    }
  });

  application.systems.rigidbody.gravity.set(0, -18, 0);
  window.__app = application;

  // Automation hooks
  window.__setCam = (eye, target) => {
    cameraEnt.setPosition(eye[0], eye[1], eye[2]);
    cameraEnt.lookAt(new Vec3(target[0], target[1], target[2]));
  };
  window.__playerPos = () => {
    const p = playerEnt.getPosition();
    return [p.x, p.y - 0.9, p.z];
  };

  // Pick a spawn
  window.__stage = "spawn-pick";
  window.__chosenSpawn = await chooseSpawn(col);
  window.__stage = "live";
  spawnFrom(col);
  setTimeout(() => buildUnderlay(), 500);

  if (isDrone) toggleDroneMode(true);
  updateToolbarUI();

  // ---------------- input handlers ----------------
  let isDragging = false;
  let lastMouseX = 0, lastMouseY = 0;

  canvas.addEventListener("mousedown", (e) => {
    if (e.button === 0) {
      isDragging = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      canvas.requestPointerLock();
      pickCameraFromRay(e.clientX, e.clientY);
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
    if (e.code === "KeyF" || e.key === "f" || e.key === "F") toggleDroneMode();
    if (e.code === "KeyG" || e.key === "g" || e.key === "G") toggleSplatVisibility();
    if (e.code === "KeyP" || e.key === "p" || e.key === "P") toggleCamerasLayer();
    if (e.code === "KeyO" || e.key === "o" || e.key === "O") togglePointsLayer();
    if (e.code === "KeyV" || e.key === "v" || e.key === "V") toggleCoverageLayer();
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
const FEET = 0.9;
function groundRay() {
  const p = playerEnt.getPosition();
  return app.systems.rigidbody.raycastFirst(
    new Vec3(p.x, p.y - 0.80, p.z), new Vec3(p.x, p.y - 2.6, p.z),
    { filterCollisionMask: BODYMASK_STATIC });
}

const STEP_MAX = 0.5;
function stepBlocked(dx, dz, pos) {
  const l = Math.hypot(dx, dz);
  if (l < 0.1) return false;
  const nx = dx / l, nz = dz / l, reach = 0.34 + 0.30, feet = pos.y - FEET;
  const probe = (y) => app.systems.rigidbody.raycastFirst(
    new Vec3(pos.x, y, pos.z), new Vec3(pos.x + nx * reach, y, pos.z + nz * reach),
    { filterCollisionMask: BODYMASK_STATIC });
  return !!probe(feet + 0.06) && !probe(feet + STEP_MAX + 0.10);
}

function step(dt) {
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

    cameraEnt.setPosition(dronePos.x, dronePos.y, dronePos.z);
    cameraEnt.lookAt(new Vec3(dronePos.x + fx * 5, dronePos.y + fy * 5, dronePos.z + fz * 5));

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
  if (hit) onMesh = (pos.y - FEET - hit.point.y) < 0.35;
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
    const hs = Math.hypot(v.x, v.z);
    const keep = sp > 0 && hs > sp + 0.5;
    let vy = v.y;
    if (P.grounded && (dx || dz) && stepBlocked(dx, dz, pos)) vy = 4.24;
    rb.linearVelocity = new Vec3(keep ? v.x : dx, vy, keep ? v.z : dz);
    rb.activate();
  }

  if (P.grounded && (activeKeys["Space"] || activeKeys[" "]) && autopilot.phase === "idle") {
    rb.linearVelocity = new Vec3(dx, 6.5, dz);
  }

  const px = playerEnt.getPosition();
  if (P.prev !== null) {
    const jump = Math.hypot(px.x - P.prev.x, px.z - P.prev.z);
    P.walked += (jump < 1.0 ? jump : 0);
  }
  P.prev = { x: px.x, z: px.z };

  if (px.y < groundHF(px.x, px.z) - 4.0) {
    P.violations++;
    const g = P.lastGood || { x: window.__spawn?.x ?? 0, y: groundHF(window.__spawn?.x ?? 0, window.__spawn?.z ?? 0) + 1.5, z: window.__spawn?.z ?? 0 };
    playerEnt.rigidbody.teleport(g.x, g.y + 1, g.z);
    playerRb.linearVelocity = new Vec3(0, 0, 0);
  }
  if (P.grounded && px.x > HF.minX && px.x < HF.maxX && px.z > HF.minZ && px.z < HF.maxZ) {
    P.lastGood = { x: px.x, y: px.y, z: px.z };
  }

  playerEnt.setLocalEulerAngles(0, P.yaw * 180 / Math.PI, 0);
  const [fx, fz] = fwdOf(P.yaw);
  if (P.firstPerson) {
    cameraEnt.setPosition(px.x + fx * 0.12, px.y + 0.62, px.z + fz * 0.12);
  } else {
    const cx = px.x - fx * 3.4, cz = px.z - fz * 3.4;
    const cy = Math.max(px.y + 2.0, groundHF(cx, cz) + 0.4);
    cameraEnt.setPosition(cx, cy, cz);
  }
  const look = new Vec3(px.x + fx * 5, px.y + (P.firstPerson ? 0.55 : 1.15) + Math.tan(P.pitch) * 5, px.z + fz * 5);
  cameraEnt.lookAt(look);
  const w = window.__walk;
  w.pos = [+px.x.toFixed(3), +(px.y - 0.9).toFixed(3), +px.z.toFixed(3)];
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
