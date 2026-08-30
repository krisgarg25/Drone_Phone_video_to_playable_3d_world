#!/usr/bin/env node
/*
 * test_capture_scripts.js
 *
 * Loads viewer/capture.html's browser scripts in a browser-like sandbox:
 * `window` defined, NO Node `module`/`require` visible to the page code
 * (vm.createContext isolates them). Executes, in page order:
 *   /viewer/three.min.js, /viewer/OrbitControls.js, /viewer/lidar_overlay.js,
 *   /viewer/coverage_map.js, then the inline capture-page <script> — and asserts
 *   zero errors.
 * Also checks the overlay contract:
 *   - canvas#overlay sits over video#cam-feed
 *   - the live loop does NOT re-project a world-anchored dot map in basic mode
 *     (the old pose was drifted double-integrated accelerometer data, so those
 *     dots lied — see viewer/coverage_map.js header for the full story)
 *   - per-surface coverage comes from the WebXR AR path via CoverageMap
 *   - the bearing strip still draws as a compact 72x3 HUD, never a screen flood
 *
 * Run:  node tests/test_capture_scripts.js
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.resolve(__dirname, "..");
const HTML_PATH = path.join(ROOT, "viewer", "capture.html");
const html = fs.readFileSync(HTML_PATH, "utf8");

let pass = 0, fail = 0;
function check(cond, label, detail) {
  if (cond) { pass++; console.log("pass  " + label); }
  else { fail++; console.log("FAIL  " + label + (detail ? "  -> " + detail : "")); }
}

// ---------- structural gate: overlay over video, dots not green grid ------
const videoIdx = html.indexOf('<video id="cam-feed"');
const overlayIdx = html.indexOf('<canvas id="overlay"');
check(videoIdx >= 0 && overlayIdx > videoIdx,
      "capture.html keeps canvas#overlay layered over video#cam-feed");
check(html.includes('src="/viewer/lidar_overlay.js"'),
      "capture.html loads the shipped lidar overlay module via plain <script src>");
check(!/type="module"/.test(html), "no ES-module scripts (file:// CORS safe)");
check(html.indexOf("const COLS") === -1 && html.indexOf("covAt(") === -1,
      "screen-space green cell/box coverage grid is gone from the live overlay");
check(html.includes('src="/viewer/coverage_map.js"'),
      "capture.html loads the coverage engine (viewer/coverage_map.js) via plain <script src>");
check((html.match(/class="hero-btn"/g) || []).length === 1 && html.includes('id="btn-start"'),
      "the hero offers exactly one action, not a mode choice");
check(/function startCapture\(\) \{\s*if \(AR\.arReady\) startArScan\(\); else requestCameraAndSensors\(\);/.test(html),
      "startCapture() picks AR when the phone supports it and the camera path otherwise");
check(/requiredFeatures: \["camera-access"\]/.test(html) &&
      /isSessionSupported\("immersive-ar", \{ requiredFeatures: \["camera-access"\] \}\)/.test(html),
      "AR is gated on camera-access too -- without the camera module an AR take has no video");
check(!html.includes("native-file-input") && !html.includes("handleNativeVideo"),
      "gallery video ingest is gone (existing clips are run from the laptop)");
check(/#ar-overlay\.live ~ #topbar,[\s\S]{0,60}#ar-overlay\.live ~ #dock/.test(html),
      "the top bar and basic-mode REC dock hide during an AR scan");
check(/arDrawCamera\(vp\.width, vp\.height\)/.test(html) && /arDrawCamera\(cw, ch\)/.test(html),
      "the camera background is cropped to the XR viewport, not the canvas backing store");
check(/VISION_W = 128, VISION_H = 96, VISION_INTERVAL_MS = 125/.test(html) &&
      /if \(camMode !== "cloud" \|\| AR\.session\) return;/.test(html),
      "phone guide is throttled and the hidden 3D renderer stays idle");
// THE CONTRACT after "kill the lie": the basic camera path must not present
// drifted-pose dots as captured surfaces. Per-surface coverage is the AR
// path's job, and that path runs on real ARCore pose + real depth.
check(!/LidarOverlay\.project\(/.test(html) && !/LidarOverlay\.persist\(/.test(html),
      "basic-mode overlay no longer re-projects a world dot map (drifted dots removed)");
check(/CoverageMap\.observe\(/.test(html) && /CoverageMap\.project\(/.test(html) &&
      /CoverageMap\.unprojectDepth\(/.test(html),
      "AR scan path feeds real depth through the coverage engine (observe/unproject/project)");
check(/immersive-ar/.test(html) && /depth-sensing/.test(html) &&
      /getDepthInformation/.test(html),
      "AR session requests immersive-ar with the depth-sensing module");
check(/rgba\(239, 68, 68, 0\.045\)/.test(html) === false,
      "no full-frame 'unscanned' red wash rectangles in the overlay");

// ---------- browser-like sandbox (window yes, Node globals no) ------------
function errDetail(e) {
  const m = (e.stack || "").match(/viewer[\\/](three\.min\.js|OrbitControls\.js|lidar_overlay\.js|coverage_map\.js|capture\.html[^)]*):(\d+):(\d+)/);
  if (m && m[1].endsWith(".js")) {
    const src = (fs.readFileSync(path.join(ROOT, "viewer", m[1]), "utf8").split("\n")[+m[2] - 1]) || "";
    const col = +m[3];
    return e.name + ": " + e.message + "  [" + m[1] + " col " + col + ": ..." +
           src.slice(Math.max(0, col - 90), col + 70) + "...]";
  }
  return e.name + ": " + e.message;
}

const elements = new Map();
const rectLog = [];
const pathLog = [];
const strokeLog = [];

function make2dContext() {
  return {
    drawImage() {},
    getImageData(x, y, w, h) {
      return { data: new Uint8ClampedArray(w * h * 4), width: w, height: h };
    },
    fillRect(x, y, w, h) { rectLog.push({ x, y, w, h }); },
    moveTo(x, y) { pathLog.push({ x, y }); },
    lineTo(x, y) { pathLog.push({ x, y }); },
    closePath() {}, stroke() { strokeLog.push(pathLog.splice(0).length); },
    clearRect() {}, beginPath() {}, arc() {}, fill() {},
    fillStyle: "", font: "", textAlign: "", strokeStyle: "", lineWidth: 0
  };
}

function makeGLContext(canvas) {
  // enough of a WebGL context for THREE r128's WebGLRenderer constructor
  const special = {
    canvas,
    drawingBufferWidth: 390, drawingBufferHeight: 844,
    getParameter: (pname) => {
      if (pname === 7938) return "WebGL 1.0 (script-test)";   // VERSION
      if (pname === 7936) return "script-test";               // VENDOR
      if (pname === 7937) return "script-test";               // RENDERER
      if (pname === 35724) return "WebGL GLSL ES 1.00";       // SHADING_LANGUAGE_VERSION
      return 4096;                                            // any MAX_* constant
    },
    getShaderPrecisionFormat: () => ({ rangeMin: 127, rangeMax: 127, precision: 23 }),
    getContextAttributes: () => ({ alpha: true, antialias: true, depth: true, stencil: true }),
    getExtension: () => null,
    getError: () => 0,
    isContextLost: () => false,
    // healthy-context answers for the load-time render (animateThree -> render)
    getProgramInfoLog: () => "", getShaderInfoLog: () => "",
    // pname-aware: LINK_STATUS/COMPILE_STATUS true, ACTIVE_UNIFORMS/ATTRIBUTES zero
    getProgramParameter: (pname) => ((pname === 35718 || pname === 35721) ? 0 : 1),
    getShaderParameter: () => 1,
    getActiveUniform: () => ({ name: "u", type: 35676, size: 1 }),
    getActiveAttrib: () => ({ name: "a", type: 35664, size: 1 }),
    getAttribLocation: () => 0, getUniformLocation: () => ({}),
    createShader: () => ({}), createProgram: () => ({}), createTexture: () => ({}),
    createBuffer: () => ({}), createFramebuffer: () => ({}), createRenderbuffer: () => ({})
  };
  return new Proxy(special, {
    get(t, prop) {
      if (prop in t) return t[prop];
      return () => {}; // every other gl entry is a harmless no-op at load time
    }
  });
}

function makeEl(id, tag) {
  const el = {
    id, tagName: (tag || "div").toUpperCase(),
    style: {}, dataset: {}, className: "",
    width: 0, height: 0, clientWidth: 390, clientHeight: 844,
    videoWidth: 0, videoHeight: 0, textContent: "", innerHTML: "", value: "",
    disabled: false, files: [],
    classList: {
      add() {}, remove() {}, contains() { return false; },
      toggle() {}
    },
    addEventListener() {}, removeEventListener() {},
    setAttribute() {}, getAttribute() { return null; },
    appendChild() {}, removeChild() {}, click() {}, focus() {},
    getBoundingClientRect() { return { left: 0, top: 0, width: 390, height: 844 }; },
    getContext(type) {
      if (type === "2d") return make2dContext();
      return makeGLContext(el);
    }
  };
  el.ownerDocument = sandbox.document;
  return el;
}

const sandbox = {
  console,
  innerWidth: 390, innerHeight: 844,
  devicePixelRatio: 2,
  performance: { now: () => Date.now() },
  requestAnimationFrame: () => 0,
  cancelAnimationFrame: () => {},
  setTimeout, clearTimeout, setInterval: () => 0, clearInterval: () => {},
  alert() {}, confirm() { return false; }, prompt() { return null; },
  navigator: { userAgent: "capture-script-test", clipboard: undefined, mediaDevices: undefined },
  location: { protocol: "http:", hostname: "localhost", href: "http://localhost/viewer/capture.html" },
  screen: { orientation: null },
  fetch: () => Promise.resolve({ json: () => Promise.resolve({}) }),
  addEventListener() {}, removeEventListener() {},
  dispatchEvent() { return true; }
};
sandbox.window = sandbox;
sandbox.self = sandbox;
sandbox.top = sandbox;
sandbox.globalThis = sandbox;

sandbox.document = {
  documentElement: makeEl("html", "html"),
  createElement(tag) { return makeEl("", tag); },
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeEl(id, id === "cam-feed" ? "video" : (id === "overlay" ? "canvas" : "div")));
    return elements.get(id);
  },
  addEventListener() {}, removeEventListener() {},
  fullscreenElement: null, exitFullscreen: () => Promise.resolve()
};
sandbox.document.documentElement.requestFullscreen = () => Promise.resolve();

const ctx = vm.createContext(sandbox);

// ---------- execute the page's scripts in order ---------------------------
const extRe = /<script src="\/viewer\/([^"]+)"><\/script>/g;
let m, extScripts = [];
while ((m = extRe.exec(html)) !== null) extScripts.push(m[1]);
check(extScripts.join(",") === "three.min.js,OrbitControls.js,lidar_overlay.js,coverage_map.js",
      "plain <script src> order is three -> OrbitControls -> lidar_overlay -> coverage_map (" + extScripts.join(",") + ")");

for (const name of extScripts) {
  const code = fs.readFileSync(path.join(ROOT, "viewer", name), "utf8");
  try {
    vm.runInContext(code, ctx, { filename: "viewer/" + name });
    check(true, "viewer/" + name + " executes in browser-like sandbox");
  } catch (e) {
    check(false, "viewer/" + name + " executes in browser-like sandbox", errDetail(e));
  }
}

const inline = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(x => x[1]);
check(inline.length === 1, "exactly one inline capture-page script found");
try {
  vm.runInContext(inline[0], ctx, { filename: "viewer/capture.html#inline" });
  check(true, "inline capture script executes without error (window defined, no Node globals)");
} catch (e) {
  check(false, "inline capture script executes without error", e.name + ": " + e.message);
}

// ---------- post-load assertions ------------------------------------------
check(typeof sandbox.THREE === "object" && sandbox.THREE !== null, "THREE global present after load");
check(typeof sandbox.LidarOverlay === "object" && typeof sandbox.LidarOverlay.project === "function",
      "LidarOverlay global present after load");
check(typeof sandbox.CoverageMap === "object" && typeof sandbox.CoverageMap.observe === "function" &&
      typeof sandbox.CoverageMap.project === "function" && typeof sandbox.CoverageMap.stats === "function",
      "CoverageMap global present after load (observe/project/stats exported)");
const overlayEl = elements.get("overlay");
const videoEl = elements.get("cam-feed");
check(!!overlayEl && overlayEl.tagName === "CANVAS", "canvas#overlay element was acquired by the page");
check(!!videoEl && videoEl.tagName === "VIDEO", "video#cam-feed element was acquired by the page");
check(typeof sandbox.drawOverlay === "function" || /function drawOverlay/.test(inline[0] || ""),
      "drawOverlay (live overlay draw) is part of the loaded page script");
check(typeof sandbox.requestCameraAndSensors === "function" &&
      typeof sandbox.toggleRecording === "function" &&
      typeof sandbox.uploadToLaptop === "function",
      "camera/recording/upload entry points remain defined (capture flow intact)");
check(typeof sandbox.drawBearingHUD === "function" &&
      typeof sandbox.showMissedReport === "function" &&
      typeof sandbox.closeMissedReport === "function",
      "bearing HUD + missed-report functions are part of the loaded page script");
check(typeof sandbox.startArScan === "function" && typeof sandbox.stopArScan === "function",
      "AR scan entry points are part of the loaded page script");
check(html.includes('id="missed-modal"'), "missed-coverage report modal present in markup");

// Camera-guide smoke run: even before recording, the real video view must draw
// a visible framing overlay and feature points rather than remain a blank feed.
try {
  const before = rectLog.length;
  videoEl.videoWidth = 1280; videoEl.videoHeight = 720;
  sandbox.drawOverlay([{ px: 24, py: 24 }, { px: 64, py: 48 }], Date.now());
  const guideRects = rectLog.slice(before).filter(r => r.w > 0 && r.h > 0);
  check(guideRects.length >= 4,
        "normal camera view draws its live framing guide before REC (" + guideRects.length + " guide marks)");
} catch (e) {
  check(false, "normal camera view draws its live framing guide before REC", e.name + ": " + e.message);
}

// HUD smoke run: must draw a compact bearing strip, never a full-frame flood
try {
  vm.runInContext("isRecording = true; drawBearingHUD(); isRecording = false;", ctx);
  const cells = rectLog.filter(r => r.h > 2 && r.h < 20);
  // Ignore the deliberately thin rule-of-thirds guide lines; the assertion is
  // about the bearing HUD not becoming a broad full-frame painted region.
  const tallest = rectLog.filter(r => Math.min(r.w, r.h) >= 2)
    .reduce((m, r) => Math.max(m, r.h), 0);
  check(cells.length >= 72 * 3,
        "bearing HUD drew its 72x3 bearing grid (" + cells.length + " cells)");
  check(tallest <= 80,
        "bearing HUD is a compact strip (tallest rect " + tallest.toFixed(0) +
        "px on a 1688px overlay), not a screen flood");
} catch (e) {
  check(false, "bearing HUD draws without error", e.name + ": " + e.message);
}

// ---------- behavioural: the plan view must point at the UNSCANNED wall ------
// A 4 m x 4 m room, operator at the centre facing -Z, three walls turned into
// surfels by the real coverage engine and the -X wall never observed. The live
// cue has to fire and has to point at the missing wall. The sign is the point:
// in this Y-up / -Z-forward frame +X is camera-LEFT, so a cue built from
// "azimuth minus heading" aims the arrow the wrong way -- it must come from the
// same forward/right decomposition that places the surfels.
try {
  const mapEl = sandbox.document.getElementById("ar-map");
  mapEl.width = 272; mapEl.height = 272;
  const CM = sandbox.CoverageMap;
  const map = CM.createMap();
  const cam = { x: 0, y: 1.5, z: 0 };
  const N = 45;
  const wallAtZ = (z) => { for (let i = 0; i < N; i++) { const t = -3 + 6 * i / (N - 1); add(t, 1.5, z); } };
  const wallAtX = (x) => { for (let i = 0; i < N; i++) { const t = -3 + 6 * i / (N - 1); add(x, 1.5, t); } };
  function add(x, y, z) {
    const p = { x, y, z };
    const L = Math.hypot(cam.x - x, cam.y - y, cam.z - z);
    CM.observe(map, cam, p, { x: (cam.x - x) / L, y: (cam.y - y) / L, z: (cam.z - z) / L });
  }
  wallAtZ(-3); wallAtZ(3); wallAtX(3);          // the -X wall is deliberately absent

  sandbox.__map = map;
  sandbox.__proj = new Float32Array([1.2, 0, 0, 0, 0, 1.2, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1]);
  const beforeStrokes = strokeLog.length;
  const beforeRects = rectLog.length;
  const blind = vm.runInContext(
    "AR.map = __map; AR.camWorld = { x: 0, y: 1.5, z: 0 }; AR.camFwd = { x: 0, z: -1 };" +
    " AR.lastProj = __proj; drawPlanMap(); AR.blindDeg", ctx);
  const drew = rectLog.slice(beforeRects), strokes = strokeLog.length - beforeStrokes;

  check(map.surfels.length > 120,
        "plan-view test room built (" + map.surfels.length + " surfels on 3 of 4 walls)");
  check(drew.length > 120, "plan view plotted the surfels top-down (" + drew.length + " marks)");
  check(Math.max(...drew.map(r => Math.max(r.w, r.h))) <= 8,
        "plan view draws small marks, not a filled region (largest " +
        Math.max(...drew.map(r => Math.max(r.w, r.h))) + "px on a 272px map)");
  check(typeof sandbox.drawPlanMap === "function" && /drawPlanMap\(\);/.test(html),
        "arUpdateHud drives the plan view every HUD tick");
  check(html.includes('id="ar-diag"') && /getElementById\("ar-diag"\)\.innerHTML/.test(html),
        "live diagnostics (fps / surfels / depth %) are on screen for test reports");
  check(blind !== null && blind !== undefined,
        "plan view reports a blind direction when a whole wall was never scanned");
  check(blind > 30 && blind < 160,
        "blind cue points RIGHT (+" + (blind === null ? "—" : blind.toFixed(0)) + "°) at the unobserved -X wall, not at a scanned one");
  check(strokes >= 1, "gap arrow is drawn as a stroked path, not just a number");
} catch (e) {
  check(false, "plan view draws and reports a blind direction", e.name + ": " + e.message);
}

// ---------- the view <-> image mapping behind the depth and background crop ---
// Portrait viewport (1080x2280) against a landscape sensor image (1280x720):
// the screen sees a narrow vertical strip of the sensor, so a view-normalized
// x must compress toward 0.5, and the inverse must give it back exactly.
try {
  const r = vm.runInContext(`(function(){
    AR.viewport = { w: 1080, h: 2280 }; AR.camW = 1280; AR.camH = 720;
    const f = arFit();
    const c = viewToImg(0.5, 0.5), centre = [c.x, c.y];
    const q = viewToImg(0.9, 0.5), off = [q.x, q.y];
    const e = viewToImg(1.0, 0.5), edge = e.x;
    const z = imgToView(off[0], off[1]), back = [z.x, z.y];
    return { fx: f.x, fy: f.y, centre: centre, off: off, edge: edge, back: back };
  })()`, ctx);
  check(Math.abs(r.fx - 0.266) < 0.01 && r.fy === 1,
        "portrait viewport crops a landscape sensor to x" + r.fx.toFixed(3) + " (no vertical crop)");
  check(Math.abs(r.centre[0] - 0.5) < 1e-12 && Math.abs(r.centre[1] - 0.5) < 1e-12,
        "the view centre maps to the image centre");
  check(r.off[0] > 0.5 && r.off[0] < 0.65 && r.edge < 0.7,
        "screen-right stays inside the middle of the sensor (edge maps to " + r.edge.toFixed(3) + ")");
  check(Math.abs(r.back[0] - 0.9) < 1e-12 && Math.abs(r.back[1] - 0.5) < 1e-12,
        "viewToImg and imgToView round-trip exactly (" + r.back[0] + ", " + r.back[1] + ")");
} catch (e) {
  check(false, "view <-> image mapping round-trips", e.name + ": " + e.message);
}

console.log("");
console.log(fail === 0 ? "ALL " + pass + " CAPTURE-SCRIPT CHECKS PASSED"
                       : pass + " passed, " + fail + " FAILED");
process.exit(fail === 0 ? 0 : 1);
