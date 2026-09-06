#!/usr/bin/env node
/*
 * test_capture_scripts.js
 *
 * Loads viewer/capture.html's browser scripts in a browser-like sandbox:
 * `window` defined, NO Node `module`/`require` visible to the page code
 * (vm.createContext isolates them). Executes, in page order:
 *   /viewer/coverage_map.js, then the inline capture-page <script> — and asserts
 *   zero errors.
 * Also checks the overlay contract:
 *   - canvas#overlay sits over video#cam-feed
 *   - the live loop does NOT re-project a world-anchored dot map in basic mode
 *     (the old pose was drifted double-integrated accelerometer data, so those
 *     dots lied — see viewer/coverage_map.js header for the full story)
 *   - per-surface coverage comes from the WebXR AR path via CoverageMap
 *   - the bearing strip still draws as a compact 72x3 HUD, never a screen flood
 *   - nothing ships three.js, OrbitControls or lidar_overlay.js any more: the
 *     point-cloud view they served was unreachable and the compass/IMU maths is
 *     four local functions
 *   - one way in: the start screen shows nothing but its own action, the REC
 *     dock cannot start a take with no video behind it, and the full capability
 *     check is a permanent link there — a link, never a second button
 *   - every label the operator can see is in plain words, not engine jargon
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

const meanX = (rects) => rects.reduce((t, r) => t + r.x, 0) / Math.max(1, rects.length);

// ---------- structural gate: overlay over video, dots not green grid ------
const videoIdx = html.indexOf('<video id="cam-feed"');
const overlayIdx = html.indexOf('<canvas id="overlay"');
check(videoIdx >= 0 && overlayIdx > videoIdx,
      "capture.html keeps canvas#overlay layered over video#cam-feed");
check(!/<script[^>]*src="[^"]*(three\.min|OrbitControls|lidar_overlay)\.js/.test(html),
      "capture.html loads none of the retired libraries (three, OrbitControls, lidar_overlay)");
check(!/\bTHREE\./.test(html.replace(/\/\/[^\n]*\n/g, "")),
      "no three.js call survives in the page script");
check(!/type="module"/.test(html), "no ES-module scripts (file:// CORS safe)");
check(html.indexOf("const COLS") === -1 && html.indexOf("covAt(") === -1,
      "screen-space green cell/box coverage grid is gone from the live overlay");
check(html.includes('src="/viewer/coverage_map.js"'),
      "capture.html loads the coverage engine (viewer/coverage_map.js) via plain <script src>");
check((html.match(/class="hero-btn"/g) || []).length === 1 && html.includes('id="btn-start"'),
      "the hero offers exactly one action, not a mode choice");
check(/function startCapture\(\) \{\s*if \(AR\.arReady\) startArScan\(\); else requestCameraAndSensors\(\);/.test(html),
      "startCapture() picks AR when the phone supports it and the camera path otherwise");
// The REC dock sits over the camera view, so it must not become a second way in
// that silently records a take with no video behind it.
const recBody = html.slice(html.indexOf("function toggleRecording() {"));
check(/^[\s\S]{0,240}if \(!videoEl\.srcObject\) \{ startCapture\(\); return; \}/.test(recBody) &&
      recBody.indexOf("isRecording = !isRecording;") > recBody.indexOf("startCapture(); return;"),
      "tapping REC before Start scan routes to Start scan instead of recording an empty take");
check(/function setHero\(on\) \{/.test(html) &&
      (html.match(/getElementById\("perm-hero"\)/g) || []).length === 1 &&
      /"topbar"\)\.style\.display = on \? "none" : "";/.test(html) &&
      /"dock"\)\.style\.display = on \? "none" : "";/.test(html) &&
      /setHero\(true\);/.test(html),
      "setHero() is the one writer for the start screen and hides the top bar and REC dock with it");
check(/<a class="hero-link" href="\/viewer\/xr_probe\.html"[^>]*>/.test(html) &&
      !/class="hero-link"[^>]*style="/.test(html) &&
      (html.match(/class="hero-btn"/g) || []).length === 1,
      "the full phone check is a permanent link on the start screen, still not a second action");
const readyBody = html.slice(html.indexOf("function arReady()"),
                             html.indexOf("function arStartRecording()"));
check(!/recorder\.start/.test(readyBody) && /setPanelGo\(\)/.test(readyBody) &&
      /recorder\.start\(500\)/.test(html.slice(html.indexOf("function arStartRecording()"))),
      "calibration passing only offers the take — arStartRecording is the only starter");
check(/id="missed-send"[^>]*onclick="sendFromReport\(\)"/.test(html) &&
      /function setReportSend\(canSend\)/.test(html) &&
      /setReportSend\(recordedChunks\.length > 0 \|\| poses\.length > 0\);/.test(html) &&
      /setReportSend\(recordedChunks\.length > 0\);/.test(html),
      "the end-of-take report carries Send to laptop, and only when there is a take to send");
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
      !/function animateThree|requestAnimationFrame\(animateThree\)/.test(html),
      "phone guide is throttled and no second GPU render loop runs behind the camera");
// THE CONTRACT after "kill the lie": the basic camera path must not present
// drifted-pose dots as captured surfaces. Per-surface coverage is the AR
// path's job, and that path runs on real ARCore pose + real depth.
check(!/LidarOverlay\.project\(/.test(html) && !/LidarOverlay\.persist\(/.test(html),
      "basic-mode overlay no longer re-projects a world dot map (drifted dots removed)");
check(/CoverageMap\.observe\(/.test(html) && /CoverageMap\.project/.test(html) &&
      /CoverageMap\.unprojectDepth\(/.test(html),
      "AR scan path feeds real depth through the coverage engine (observe/unproject/project)");
check(/CoverageMap\.projectPacked\(/.test(html),
      "the per-frame projection is the allocation-free packed one");
check(/immersive-ar/.test(html) && /depth-sensing/.test(html) &&
      /getDepthInformation/.test(html),
      "AR session requests immersive-ar with the depth-sensing module");
check(/rgba\(239, 68, 68, 0\.045\)/.test(html) === false,
      "no full-frame 'unscanned' red wash rectangles in the overlay");

// ---------- browser-like sandbox (window yes, Node globals no) ------------
function errDetail(e) {
  const m = (e.stack || "").match(/viewer[\\/](coverage_map\.js|capture\.html[^)]*):(\d+):(\d+)/);
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
const arcLog = [];

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
    clearRect() {}, beginPath() {}, fill() {},
    arc(x, y, r) { arcLog.push({ x, y, r }); },
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
  // One backing string for both textContent and innerHTML: the page writes
  // labels through textContent and row markup through innerHTML, and the
  // assertions read whichever is natural. A real element would too, in the
  // sense that textContent always reflects what is currently in the node.
  let text = "";
  const el = {
    id, tagName: (tag || "div").toUpperCase(),
    style: {}, dataset: {}, className: "",
    width: 0, height: 0, clientWidth: 390, clientHeight: 844,
    videoWidth: 0, videoHeight: 0, value: "",
    disabled: false, files: [],
    get textContent() { return text; },
    set textContent(v) { text = String(v); },
    get innerHTML() { return text; },
    set innerHTML(v) { text = String(v); },
    classList: {
      add() {}, remove() {}, contains() { return false; },
      toggle() {}
    },
    addEventListener() {}, removeEventListener() {},
    setAttribute() {}, getAttribute() { return null; },
    appendChild() {}, removeChild() {}, click() {}, focus() {},
    querySelector() { return null; }, querySelectorAll() { return []; },
    closest() { return null; },
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
check(extScripts.join(",") === "coverage_map.js",
      "the phone downloads exactly one library: coverage_map.js (" + extScripts.join(",") + ")");

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
check(typeof sandbox.THREE === "undefined" && typeof sandbox.LidarOverlay === "undefined",
      "no THREE or LidarOverlay global reaches the page after load");
check(typeof sandbox.CoverageMap === "object" && typeof sandbox.CoverageMap.observe === "function" &&
      typeof sandbox.CoverageMap.project === "function" && typeof sandbox.CoverageMap.stats === "function",
      "CoverageMap global present after load (observe/project/stats exported)");
check(typeof sandbox.CoverageMap.uncoveredArcs === "function",
      "CoverageMap owns uncoveredArcs (moved out of the retired lidar module)");
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
  const STORE = /<canvas id="ar-map" width="(\d+)" height="(\d+)"/.exec(html);
  const INSET_W = Number(STORE[1]), INSET_H = Number(STORE[2]);
  // The store is normally re-sized to the handset on the first draw; pin it here
  // so these numbers describe this fixture, and hand the draw a rendered box so
  // it never goes looking for one in a DOM that has no layout.
  const setInsetStore = (sc) => vm.runInContext(
    `AR.insetBox = { w: ${INSET_W * sc}, h: ${INSET_H * sc} }; AR.insetFit = null;` +
    `document.getElementById("ar-map").width = ${Math.round(INSET_W * sc)};` +
    `document.getElementById("ar-map").height = ${Math.round(INSET_H * sc)};`, ctx);
  setInsetStore(1);
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
    " AR.lastProj = __proj; AR.box = { x0:-3, x1:3, y0:0, y1:2.6, z0:-3, z1:3 };" +
    " AR.gap = { cx: 0, cy: 1.2, cz: -2, count: 40 };" +
    " drawScanInset(); AR.blindDeg", ctx);
  const drew = rectLog.slice(beforeRects), strokes = strokeLog.length - beforeStrokes;

  check(map.surfels.length > 120,
        "scan-inset test room built (" + map.surfels.length + " surfels on 3 of 4 walls)");
  check(drew.length > 120, "scan inset plotted the surfels in perspective (" + drew.length + " points)");
  const bigPoint = Math.max(...drew.map(r => Math.max(r.w, r.h)));
  check(bigPoint <= 8 * Math.min(INSET_W, INSET_H) / 272,
        "inset draws small points, not filled regions (largest " +
        bigPoint.toFixed(1) + "px on a " + INSET_W + "x" + INSET_H + " inset)");
  // A bigger box has to show a bigger cloud, not the same dots on more glass:
  // the frame is measured, so a draw at any size must land exactly where
  // proportionality predicts once the eased framing has settled.
  const drawAt = (sc) => {
    setInsetStore(sc);
    const n = rectLog.length; vm.runInContext("drawScanInset()", ctx);
    return rectLog.slice(n);
  };
  const small = drawAt(0.5), big = drawAt(1);
  // Map both into the markup frame: point centres (not fillRect corners — the
  // radius clamps move those) and dot widths.
  const toRef = (r, sc) => [(r.x + r.w / 2 - INSET_W * sc / 2) / sc + INSET_W / 2,
                            (r.y + r.h / 2 - INSET_H * sc / 2) / sc + INSET_H / 2,
                            r.w / sc];
  const worst = big.reduce((m, r, i) => {
    const a = toRef(small[i], 0.5), b = toRef(r, 1);
    return Math.max(m, Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]), Math.abs(a[2] - b[2]));
  }, 0);
  check(small.length > 100 && small.length === big.length && worst < 0.02,
        "the inset scales with its canvas — same cloud, dots and all at ×2" +
        " (worst " + worst.toFixed(3) + "px back at " + INSET_W + ")");
  setInsetStore(1);
  // The reason the frame is measured: a 6x6x2.6 m room in a square box used to
  // fill a band across the middle of it. Whatever the shape, one axis of the
  // drawing has to reach across the glass.
  const spread = (rs, ax) => {
    const v = rs.map(r => (ax === "x" ? r.x + r.w / 2 : r.y + r.h / 2));
    return Math.max(...v) - Math.min(...v);
  };
  const filled = Math.max(spread(big, "x") / INSET_W, spread(big, "y") / INSET_H);
  const tight = Math.min(spread(big, "x") / INSET_W, spread(big, "y") / INSET_H);
  check(filled > 0.7, "the scan fills its box (" + (filled * 100).toFixed(0) + "% x, " +
        (tight * 100).toFixed(0) + "% the other way, not a band floating in it)");
  check(strokes >= 2, "inset stroked the volume cage and the gap beacon (" + strokes + " paths)");
  check(typeof sandbox.drawScanInset === "function" && /drawScanInset\(\);/.test(html),
        "arUpdateHud drives the scan inset every HUD tick");
  check(html.includes('id="ar-diag"') &&
        /renderRows\(document\.getElementById\("ar-diag"\)/.test(html),
        "live diagnostics (areas / distances / size / walked) are on screen for test reports");
  check(blind !== null && blind !== undefined,
        "scan inset reports a blind direction when a whole wall was never scanned");
  check(blind > 30 && blind < 160,
        "blind cue points RIGHT (+" + (blind === null ? "—" : blind.toFixed(0)) + "°) at the unobserved -X wall");

  // The inset must NOT animate on its own: two draws at the same aim are
  // identical however much time passes, and change only when the aim changes.
  const realNow = sandbox.performance.now;
  try {
    sandbox.performance.now = () => 1000;
    vm.runInContext("for (let i = 0; i < 60; i++) drawScanInset()", ctx);  // let easings settle
    const a0 = rectLog.length; vm.runInContext("drawScanInset()", ctx);
    const runA = rectLog.slice(a0);
    sandbox.performance.now = () => 900000;
    const b0 = rectLog.length; vm.runInContext("drawScanInset()", ctx);
    const runB = rectLog.slice(b0);
    const still = runA.length === runB.length && runA.length > 50 &&
      runA.every((r, i) => r.x === runB[i].x && r.y === runB[i].y);
    check(still, "the inset holds still — no autonomous spin");

    const yawA = meanX(runA);
    vm.runInContext("AR.insetYaw += 0.6", ctx);
    const c0 = rectLog.length; vm.runInContext("drawScanInset()", ctx);
    const yawC = meanX(rectLog.slice(c0));
    check(Math.abs(yawA - yawC) > 2,
          "drag-yaw steers the view — the cloud slid " + Math.abs(yawA - yawC).toFixed(1) + "px");
    // Heading-up: turning must re-orient the view so your facing stays up, and it
    // must ease there rather than snap, so a fast spin reads as a catch-up.
    const hu = vm.runInContext(`(function(){
      AR.insetHead = null; AR.insetYaw = 0;
      AR.camFwd = { x: 0, z: -1 };                 // face -Z: heading 0
      drawScanInset();
      const first = AR.insetHead;
      AR.camFwd = { x: 1, z: 0 };                  // turn to face +X
      drawScanInset();
      const afterOne = AR.insetHead;
      for (let i = 0; i < 60; i++) drawScanInset();
      return { first: first, afterOne: afterOne, settled: AR.insetHead };
    })()`, ctx);
    const HU = -Math.PI / 2;
    check(Math.abs(hu.first) < 1e-6, "heading-up locks to the first facing");
    check(hu.afterOne !== HU && Math.abs(hu.afterOne) < Math.abs(HU),
          "a turn eases the view round instead of snapping");
    check(Math.abs(hu.settled - HU) < 0.02,
          "and settles with your facing at the top (" + (hu.settled * 180 / Math.PI).toFixed(0) + "°)");
    check(/AR\.insetYaw -= \(e\.clientX - last\[0\]\) \/ box\.w \* 1\.63/.test(html) &&
          /requestInsetRepaint\(\)/.test(html) && /pointerdown/.test(html) &&
          /touch-action: none/.test(html),
          "the inset drags as a fraction of its own size and repaints once per frame");

    // the white marker IS the operator: it has to move when the phone does
    const n0 = arcLog.length;
    vm.runInContext("AR.insetYaw = 0; drawScanInset()", ctx);
    const markA = arcLog.slice(n0)[0];
    const n1 = arcLog.length;
    vm.runInContext("AR.camWorld = { x: 2.6, y: 1.5, z: -2.4 }; AR.camFwd = { x: -0.7, z: 0.7 };" +
                    " drawScanInset()", ctx);
    const markB = arcLog.slice(n1)[0];
    const moved = markA && markB ? Math.hypot(markA.x - markB.x, markA.y - markB.y) : -1;
    check(moved > 3, "the operator marker moves with the phone (" + moved.toFixed(0) + "px)");

    // Gimbal lock: pitched at the floor or ceiling the horizontal view component
    // collapses to noise, and a heading derived from it spins — the map "loses
    // track" mid floor-scan. The heading must hold instead, and resume when the
    // phone levels out.
    const gim = vm.runInContext(`(function(){
      function mat(f) {            // column-major camera-to-world; forward = -(m[8..10])
        const m = new Array(16).fill(0);
        m[8] = -f[0]; m[9] = -f[1]; m[10] = -f[2]; m[15] = 1;
        return m;
      }
      AR.camFwd = null;
      arHoldHeading(mat([0, 0, -1]));                 // level, facing -Z
      const level = { x: AR.camFwd.x, z: AR.camFwd.z };
      arHoldHeading(mat([0, -1, 0]));                 // straight at the floor
      const down = { x: AR.camFwd.x, z: AR.camFwd.z };
      arHoldHeading(mat([0.06, -0.998, 0.03]));       // almost no horizon left
      const steep = { x: AR.camFwd.x, z: AR.camFwd.z };
      arHoldHeading(mat([0.5, -0.85, 0.16]));         // ~58° down: horizon is back
      const back = { x: AR.camFwd.x, z: AR.camFwd.z };
      return { level: level, down: down, steep: steep, back: back };
    })()`, ctx);
    check(Math.abs(gim.level.z + 1) < 1e-9 && Math.abs(gim.level.x) < 1e-9,
          "a level frame sets the heading");
    check(gim.down.z === -1 && gim.down.x === 0 && gim.steep.z === -1 && gim.steep.x === 0,
          "floor frames hold the last heading instead of spinning it");
    check(Math.abs(gim.back.x - 0.5 / Math.hypot(0.5, 0.16)) < 1e-9,
          "and a leveled-out frame picks the heading up again");
    vm.runInContext("AR.camFwd = null; AR.insetHead = null", ctx);  // before the first level frame
    const g0 = rectLog.length;
    vm.runInContext("drawScanInset()", ctx);           // must not throw, heading untouched
    check(rectLog.length > g0 && vm.runInContext("AR.insetHead", ctx) === null,
          "the inset draws fine before any heading exists");
  } finally { sandbox.performance.now = realNow; }

  // Framing ignores a lone spike, or one bad 6 m reading shrinks the whole scan.
  try {
    const pts = [];
    for (let i = 0; i < 400; i++) pts.push({ x: (i % 20) * 0.1, y: 1.0, z: Math.floor(i / 20) * 0.1 });
    pts.push({ x: 40, y: 1.0, z: 40 });                      // one wild reading
    sandbox.__pts = pts;
    const bb = vm.runInContext("arExtentBox(__pts)", ctx);
    check(bb.x1 < 3 && bb.z1 < 3,
          "extent framing rejects a lone depth spike (box " + bb.x1.toFixed(1) + "×" +
          bb.z1.toFixed(1) + " m with a spike at 40 m)");
  } catch (e) {
    check(false, "extent framing rejects a lone depth spike", e.name + ": " + e.message);
  }
} catch (e) {
  check(false, "scan inset draws and reports a blind direction", e.name + ": " + e.message);
}

// ---------- depth is read in normalized VIEW coords, never pixels -------------
// The spec throws RangeError for x/y outside 0..1, so a pixel-index call would
// silently yield nothing -- exactly how a probe once reported 0% depth on a
// phone that was fine. This pins the call, not the comment about it.
try {
  const d = vm.runInContext(`(function(){
    const n = AR.GW * AR.GH;
    AR.gOK = new Uint8Array(n); AR.gPts = new Float32Array(n*3); AR.gDepth = new Float32Array(n);
    AR.map = CoverageMap.createMap(); AR.depthOK = false; AR.depthFails = 0;
    const calls = [];
    const info = { width: 160, height: 90,
      getDepthInMeters: function (x, y) { calls.push([x, y]); return (x > 0.4 && x < 0.6) ? 2.0 : 0; } };
    const frame = { getDepthInformation: function () { return info; } };
    const m = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,1.5,0,1]);
    arReadDepth(frame, {}, m, 1.2, 1.6, 0, 0, { x:0, y:1.5, z:0 });
    let bad = 0; for (const c of calls) if (!(c[0] > 0 && c[0] < 1 && c[1] > 0 && c[1] < 1)) bad++;
    let ok = 0; for (let i = 0; i < n; i++) if (AR.gOK[i]) ok++;
    return { calls: calls.length, bad: bad, ok: ok, surfels: AR.map.surfels.length, depthOK: AR.depthOK };
  })()`, ctx);
  check(d.calls === 32 * 24, "every probe grid cell queried getDepthInMeters (" + d.calls + ")");
  check(d.bad === 0, "no call ever passed an out-of-range coordinate — " + d.bad + " would RangeError");
  check(d.ok > 30 && d.ok < 32 * 24 * 0.4, "only the valid band registered samples (" + d.ok + " of " + (32 * 24) + ")");
  check(d.surfels > 0 && d.depthOK === true, "the depth band folded into the surfel map (" + d.surfels + " surfels)");
} catch (e) {
  check(false, "depth read uses normalized view coordinates", e.name + ": " + e.message);
}

// A phone that never delivers depth must degrade, not crash. This path threw for
// a while because it wrote to an element that had been removed from the HUD.
try {
  const nd = vm.runInContext(`(function(){
    const n = AR.GW * AR.GH;
    AR.gOK = new Uint8Array(n); AR.gPts = new Float32Array(n*3); AR.gDepth = new Float32Array(n);
    AR.map = CoverageMap.createMap(); AR.depthOK = false; AR.depthFails = 0;
    const frame = { getDepthInformation: function () { return null; } };
    const m = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,1.5,0,1]);
    let threw = "";
    try { for (let i = 0; i < 95; i++) arReadDepth(frame, {}, m, 1.2, 1.6, 0, 0, { x:0, y:1.5, z:0 }); }
    catch (e) { threw = e.name + ": " + e.message; }
    return { threw: threw, fails: AR.depthFails, depthOK: AR.depthOK,
             coach: document.getElementById("ar-coach").textContent,
             diag: document.getElementById("ar-diag").innerHTML };
  })()`, ctx);
  check(nd.threw === "", "90 empty depth frames do not throw", nd.threw);
  check(nd.fails === 95 && nd.depthOK === false, "the no-depth streak is counted, not swallowed");
  check(/no distances/.test(nd.coach), "the operator is told plainly: " + nd.coach);
  check(!/unavailable/.test(nd.diag), "the rows block is not overwritten with raw status text");
} catch (e) {
  check(false, "no-depth path degrades safely", e.name + ": " + e.message);
}

// ---------- occlusion indexes the probe grid in that same view space ----------
try {
  const o = vm.runInContext(`(function(){
    const n = AR.GW * AR.GH;
    AR.gOK = new Uint8Array(n); AR.gDepth = new Float32Array(n); AR.gPts = new Float32Array(n*3);
    AR.GW = 32; AR.GH = 24;
    for (let i = 0; i < n; i++) { AR.gOK[i] = 0; AR.gDepth[i] = 0; }
    const gx = 8, gy = 6, gi = gy * AR.GW + gx;
    AR.gOK[gi] = 1; AR.gDepth[gi] = 1.0;
    const near = arOccluded((gx + 0.5) / AR.GW, (gy + 0.5) / AR.GH, 0.9);
    const far  = arOccluded((gx + 0.5) / AR.GW, (gy + 0.5) / AR.GH, 3.0);
    const elsewhere = arOccluded(0.95, 0.05, 3.0);
    return { near: near, far: far, elsewhere: elsewhere };
  })()`, ctx);
  check(o.near === false, "a surfel in front of the measured surface is drawn");
  check(o.far === true, "a surfel behind it is hidden, so 'covered' cannot be painted on an unseen wall");
  check(o.elsewhere === false, "an empty probe cell hides nothing");
} catch (e) {
  check(false, "occlusion indexes the probe grid in view space", e.name + ": " + e.message);
}

// ---------- calibration: no recording until ARCore proves it has locked -------
// WebXR exposes no tracking state and no depth confidence, so these three
// measurements are the only honest evidence. Depth and walking alone must NOT
// arm it -- geometry has to be accumulating.
try {
  const c = vm.runInContext(`(function(){
    const n = AR.GW * AR.GH;
    AR.gOK = new Uint8Array(n); AR.gPts = new Float32Array(n*3); AR.gDepth = new Float32Array(n);
    AR.map = CoverageMap.createMap(); AR.depthOK = true;
    let started = 0;
    AR.recorder = { start: function(){ started++; } };
    AR.armed = false; AR.walked = 0;
    document.getElementById("ar-hint").style.display = "block";   // startArScan opens it
    AR.cal = { t0: 0, ms: 0, lastX: null, lastZ: null, depthPct: 0, surfels: 0,
               uiT: -1e9, done: false, gaveUp: false, dismissed: false, userOpened: false };

    arCalibrate(1000, { x: 0, y: 1.5, z: 0 });
    const cold = { armed: AR.armed, msg: document.getElementById("cal-msg").textContent };

    AR.gOK.fill(1);                                  // depth perfect, and walking...
    for (let i = 0; i < 60; i++) arCalibrate(2000 + i * 33, { x: i * 0.02, y: 1.5, z: 0 });
    const noGeom = { armed: AR.armed, path: AR.walked, pct: AR.cal.depthPct,
                     msg: document.getElementById("cal-msg").textContent };

    for (let i = 0; i < 20; i++) for (let j = 0; j < 20; j++)   // ...now real surfaces
      CoverageMap.observe(AR.map, { x: 0, y: 1.5, z: 1.2 },
        { x: -1.2 + i * 0.13, y: 0.4 + j * 0.13, z: -1.0 }, { x: 0, y: 0, z: 1 });
    arCalibrate(6000, { x: 1.4, y: 1.5, z: 0 });
    const ready = { armed: AR.armed, started: started, done: AR.cal.done,
                    go: document.getElementById("cal-go").textContent,
                    panel: document.getElementById("ar-hint").style.display };
    dismissHint();                                  // the operator's tap
    return { cold: cold, noGeom: noGeom, ready: ready, armed: AR.armed, started: started,
             panel: document.getElementById("ar-hint").style.display,
             surfels: AR.map.surfels.length };
  })()`, ctx);
  check(c.cold.armed === false, "cold start does not record — nothing has been proven yet");
  check(c.noGeom.armed === false && c.noGeom.pct === 1 && c.noGeom.path > 0.6,
        "depth + walking alone still do not arm it without accumulating geometry");
  check(c.noGeom.msg.indexOf("Sweep slowly") === 0,
        "it asks for the one missing thing (\"" + c.noGeom.msg + "\")");
  check(c.ready.done === true && c.ready.armed === false && c.ready.started === 0 && c.surfels > 120,
        "all three checks holding does NOT start the take (" + c.surfels + " surfels, " +
        c.ready.started + " recorder starts)");
  check(c.ready.go === "Start recording" && c.ready.panel === "block",
        "the panel stays up and offers Start recording (\"" + c.ready.go + "\")");
  check(c.armed === true && c.started === 1 && c.panel === "none",
        "the tap is what starts the recorder and hands the screen back to the scan");

  const f = vm.runInContext(`(function(){
    let s2 = 0;
    AR.armed = false;
    AR.cal = { t0: 0, ms: 0, path: 0, lastX: null, lastZ: null, depthPct: 0, surfels: 0,
               uiT: -1e9, done: false, gaveUp: false, dismissed: false, userOpened: false };
    AR.recorder = { start: function(){ s2++; } };
    AR.gOK.fill(0);
    document.getElementById("ar-hint").style.display = "block";
    setPanelGo();                            // startArScan opens the panel with this
    arCalibrate(1000, { x: 0, y: 1.5, z: 0 });
    const label = document.getElementById("cal-go").textContent;
    dismissHint();                       // "start without waiting"
    return { armed: AR.armed, started: s2, label: label,
             coach: document.getElementById("ar-coach").textContent };
  })()`, ctx);
  check(f.armed === true && f.started === 1 && f.label === "Getting ready…",
        "tapping while it waits forces a start — it never hard-blocks (\"" + f.label + "\")");
  check(/had not measured distances yet/.test(f.coach),
        "and an early start says so out loud (\"" + f.coach + "\")");
} catch (e) {
  check(false, "calibration gates recording on proven tracking", e.name + ": " + e.message);
}

// ---------- the panel is the ONLY ui reachable mid-scan ---------------------
// dom-overlay composites one element over an immersive-ar view, so anything
// outside #ar-overlay — including the sheet that used to hold the quality
// readouts — cannot be opened while a scan is running. Everything an operator
// checks during a take therefore has to live inside it.
try {
  const ovStart = html.indexOf('<div id="ar-overlay">');
  const ovBlock = ovStart < 0 ? "" : html.slice(ovStart, html.indexOf("TOP BAR", ovStart));
  check(ovStart > 0 && ovBlock.includes('id="ar-hint"'),
        "the panel sits inside #ar-overlay, the dom-overlay root");
  for (const id of ["ar-diag", "ph-detail", "cal-msg", "cal-go"]) {
    check(ovBlock.includes('id="' + id + '"'), "#" + id + " is reachable during an AR scan");
  }
  check(ovBlock.includes('class="tabs"'), "the in-scan panel is tabbed, not a stack of panels");
  check(/function switchTab\(btn\)/.test(html) &&
        /p\.classList\.toggle\("on", p\.dataset\.body === btn\.dataset\.tab\)/.test(html),
        "switchTab shows exactly the tab body the button names");
  check((html.match(/data-tab="scan"/g) || []).length === 2 &&
        (html.match(/data-tab="phone"/g) || []).length === 2,
        "the three tabs exist in both boxes: in-scan panel and sheet");

  const shown = vm.runInContext(`(function(){
    AR.arReady = true; AR.depthOK = false; AR.depthWH = null; AR.camW = 1280; AR.camH = 720;
    AR.fps = 29; AR.walked = 1.4; AR.armed = false; AR.mime = "video/webm;codecs=vp9";
    AR.cal = { t0: 0, ms: 0, lastX: null, lastZ: null, depthPct: 0, surfels: 0, uiT: -1e9,
               done: false, gaveUp: false, dismissed: false, userOpened: false };
    showHint('help');
    return { display: document.getElementById("ar-hint").style.display,
             opened: AR.cal.userOpened,
             title: document.getElementById("cal-title").textContent,
             go: document.getElementById("cal-go").textContent,
             phone: document.getElementById("ph-detail").innerHTML };
  })()`, ctx);
  check(shown.display === "block" && shown.opened === true,
        "Help opens the panel and tells the coach to stop talking over it");
  check(shown.title === "Getting ready" && shown.go === "Getting ready…",
        "before arming it calls the wait 'Getting ready', not 'Calibrating' (\"" +
        shown.title + "\" / \"" + shown.go + "\")");
  check(/Distance sensing[\s\S]*not arriving/.test(shown.phone) &&
        /1280×720/.test(shown.phone) && /webm vp9/.test(shown.phone),
        "the Phone tab reports what the handset said, in reading order");
} catch (e) {
  check(false, "panel is reachable and populated during a scan", e.name + ": " + e.message);
}

// ---------- operator copy must not be engine jargon -------------------------
// The engine talks about surfels, patches, baselines and bearings. The operator
// has to be told what to DO with their feet, so the words on screen are checked
// independently of the words in the source.
try {
  const noComments = inline[0].replace(/\/\*[\s\S]*?\*\//g, "")
                              .replace(/(^|[^:])\/\/[^\n]*/g, "$1");
  const idLiterals = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map(x => x[1]));
  const copyLines = noComments.split("\n")
    .filter(l => /textContent|innerHTML|arCoach\(|alert\(|confirm\(/.test(l)).join("\n");
  const strings = [...copyLines.matchAll(/(["'`])(?:\\.|(?!\1)[^\\])*\1/g)].map(x => x[1])
    .filter(s => !idLiterals.has(s));
  const markupText = html.replace(/<script[\s\S]*?<\/script>/g, "")
                         .replace(/<style>[\s\S]*?<\/style>/g, "")
                         .replace(/<!--[\s\S]*?-->/g, "");
  const textNodes = [...markupText.matchAll(/>([^<>]+)</g)].map(x => x[1]);
  const hay = strings.concat(textNodes).join(" \n ");
  const banned = ["surfel", "patch", "6-DoF", "6DoF", "bearing", "quadrant", "calibrat",
                  "azimuth", "voxel", "lidar", "uncalibrated", "MAP FULL", "ARCore",
                  "immersive-ar", "dom-overlay", "depth-sensing"];
  const hits = banned.filter(w => new RegExp(w, "i").test(hay));
  check(hits.length === 0,
        "no engine jargon in the " + (strings.length + textNodes.length) + " operator-visible strings",
        hits.join(", "));
} catch (e) {
  check(false, "operator copy stays free of engine jargon", e.name + ": " + e.message);
}

// ---------- the camera background crop still spans viewport vs sensor ---------
// Portrait viewport (1080x2280) against a landscape sensor image (1280x720): the
// screen sees a narrow vertical strip of the sensor, so the sampled region must
// compress horizontally and not at all vertically.
try {
  const r = vm.runInContext(`(function(){
    AR.viewport = { w: 1080, h: 2280 }; AR.camW = 1280; AR.camH = 720;
    const f = arFit();
    const vp = arFit(1280, 720);           // destination matches the sensor: no crop
    return { fx: f.x, fy: f.y, sx: vp.x, sy: vp.y, ia: arImgAspect() };
  })()`, ctx);
  check(Math.abs(r.fx - 0.266) < 0.01 && r.fy === 1,
        "portrait viewport crops a landscape sensor to x" + r.fx.toFixed(3) + " (no vertical crop)");
  check(Math.abs(r.sx - 1) < 1e-9 && r.sy === 1,
        "a destination already matching the sensor is not cropped at all");
  check(Math.abs(r.ia - 1280 / 720) < 1e-9, "sensor aspect resolves from the camera image");
} catch (e) {
  check(false, "camera-background crop spans viewport against sensor", e.name + ": " + e.message);
}

// ---------- the raw depth image, with the phone's own doubts ------------------
// getDepthInMeters() is the convenience accessor and it throws away the
// confidence channel, which is the difference between a wall and the edge of a
// wall. Reading the buffer is also what lets one grid cell take the NEAREST
// surface inside it instead of tapping the middle.
try {
  const r = vm.runInContext(`(function(){
    const n = AR.GW * AR.GH;
    AR.gOK = new Uint8Array(n); AR.gPts = new Float32Array(n*3); AR.gDepth = new Float32Array(n);
    AR.map = CoverageMap.createMap(); AR.depthOK = false; AR.depthFails = 0;
    AR.confUse = false; AR.confMax = 0; AR.confTot = 0; AR.confPass = 0;
    const W = 64, H = 48;
    const data = new Uint16Array(W * H * 2);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const at = (y * W + x) * 2;
        data[at] = (x < 3 && y < 3) ? 5000 : 2000;      // a 5 m block in one corner
        data[at + 1] = (y >= 20 && y < 24) ? 10 : 1000; // the phone doubts two rows
      }
    }
    let calls = 0;
    const info = { width: W, height: H, rawValueToMeters: 0.001, data: data,
      getDepthInMeters: function () { calls++; return 2.0; } };
    const frame = { getDepthInformation: function () { return info; } };
    const m = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,1.5,0,1]);
    const cam = { x: 0, y: 1.5, z: 0 };
    arReadDepth(frame, {}, m, 1.2, 1.6, 0, 0, cam);   // first pass learns the scale
    arReadDepth(frame, {}, m, 1.2, 1.6, 0, 0, cam);   // second pass applies it
    let ok = 0; for (let i = 0; i < n; i++) if (AR.gOK[i]) ok++;
    return { calls: calls, ok: ok, n: n, cell0: AR.gDepth[0], cell1: AR.gDepth[1],
             confUse: AR.confUse, trusted: AR.confPass / Math.max(1, AR.confTot) };
  })()`, ctx);
  check(r.calls === 0, "a raw depth image is read from its buffer, not tapped per cell (" + r.calls + " accessor calls)");
  check(r.confUse === true, "the confidence channel is used when the image carries one");
  check(Math.abs(r.cell0 - 5) < 0.01 && Math.abs(r.cell1 - 2) < 0.01,
        "the nearest surface wins a cell that straddles an edge (" + r.cell0.toFixed(2) + " and " + r.cell1.toFixed(2) + " m)");
  check(r.ok === r.n - 64, "two rows the phone doubted are dropped (" + r.ok + " of " + r.n + " cells kept)");
  check(r.trusted > 0.9, "what survives is almost all trusted (" + Math.round(r.trusted * 100) + "%)");
} catch (e) {
  check(false, "raw depth image with the confidence channel", e.name + ": " + e.message);
}

// ---------- the phone's planes, ingested and named ----------------------------
// A horizontal plane below the eyes is the floor and one above them is the
// ceiling; a label the phone offers outranks that guess. Neither may be needed
// for the other to work, because some builds give one and not the other.
try {
  const p = vm.runInContext(`(function(){
    AR.map = CoverageMap.createMap(); AR.room = CoverageMap.createRoom();
    AR.planeIdx = new Map(); AR.refSpace = {}; AR.roomCapture = false;
    const flat = (ty) => new Float32Array([1,0,0,0, 0,0,1,0, 0,-1,0,0, 0,ty,0,1]);
    const square = (s) => [{x:-s,y:-s,z:0},{x:s,y:-s,z:0},{x:s,y:s,z:0},{x:-s,y:s,z:0}];
    const mkPlane = (label, orient, matrix, pts) => ({
      semanticLabel: label, orientation: orient, lastChangedTime: 11,
      polygon: pts, planeSpace: { matrix: matrix } });
    const floorP = mkPlane("floor", "horizontal", flat(-0.02), square(1));
    // deliberately unlabelled: this one has to be named from geometry alone
    const ceilP = mkPlane("", "horizontal", flat(2.4), square(1));
    const wallM = new Float32Array([0,0,1,0, 0,1,0,0, 1,0,0,0, -3,1.2,0,1]);
    const wallP = mkPlane("", "vertical", wallM,
      [{x:-1.5,y:-1,z:0},{x:1.5,y:-1,z:0},{x:1.5,y:1,z:0},{x:-1.5,y:1,z:0}]);
    const frame = {
      detectedPlanes: new Set([floorP, ceilP, wallP]),
      getPose: function (space) { return { transform: { matrix: space.matrix } }; }
    };
    arReadPlanes(frame, { x: 0, y: 1.5, z: 0 });
    const s = CoverageMap.roomSummary(AR.room, AR.map);
    const areas = [];
    AR.planeIdx.forEach((rec) => areas.push(rec.kind + ":" + (rec.area || 0).toFixed(2)));
    // a plane the phone stops tracking must not stay in the room
    frame.detectedPlanes.delete(ceilP);
    arReadPlanes(frame, { x: 0, y: 1.5, z: 0 });
    const gone = AR.room.phone.ceiling === null;
    return { api: AR.planeAPI, floor: s.floorY, ceiling: s.ceilingY, height: s.height,
             walls: s.walls.length, wallD: s.walls.length ? Math.abs(s.walls[0].d) : 0,
             src: s.source, areas: areas.join(" "), gone: gone,
             floorArea: AR.room.phone.floor.area };
  })()`, ctx);
  check(p.api === true, "the handset's plane set is picked up when it exists");
  check(Math.abs(p.floor + 0.02) < 0.02, "a labelled floor plane lands at its height (" + p.floor.toFixed(3) + " m)");
  check(Math.abs(p.ceiling - 2.4) < 0.03, "an unlabelled horizontal above the eyes becomes the ceiling (" + p.ceiling.toFixed(2) + " m)");
  check(Math.abs(p.height - 2.42) < 0.05, "and the room's height is the gap between them (" + p.height.toFixed(2) + " m)");
  check(p.walls === 1 && Math.abs(p.wallD - 3) < 0.05, "one vertical plane is one wall 3 m away (" + p.walls + " at " + p.wallD.toFixed(2) + " m)");
  check(Math.abs(p.floorArea - 4) < 0.01, "a 2x2 m polygon measures as 4 m^2 (" + p.floorArea.toFixed(2) + ")");
  check(p.src === "phone", "the answer is attributed to the phone, not to a guess");
  check(p.gone === true, "a plane the phone stops tracking leaves the room with it");
} catch (e) {
  check(false, "phone planes are ingested and classified", e.name + ": " + e.message);
}

// ---------- his take, replayed through the page's own plane ingest -----------
// videos/test1/data_room.json: the handset reported two level surfaces under the
// feet (a step down at -0.54 and a raised area at 0.20) and four ceiling patches.
// The biggest of each paired into a 1.60 m room, which is why the ceiling looked
// un-mappable. Every plane below is the real one, at its real height and area.
try {
  const p = vm.runInContext(`(function(){
    AR.map = CoverageMap.createMap(); AR.room = CoverageMap.createRoom();
    AR.planeIdx = new Map(); AR.refSpace = {}; AR.roomCapture = true; AR.refY = null;
    const flat = (ty) => new Float32Array([1,0,0,0, 0,0,1,0, 0,-1,0,0, 0,ty,0,1]);
    const quad = (s) => [{x:-s,y:-s,z:0},{x:s,y:-s,z:0},{x:s,y:s,z:0},{x:-s,y:s,z:0}];
    const mk = (y, area) => ({ semanticLabel: "", orientation: "horizontal",
      lastChangedTime: 5, polygon: quad(Math.sqrt(area) / 2), planeSpace: { matrix: flat(y) } });
    const planes = [mk(-0.539, 3.49), mk(0.198, 4.17), mk(1.790, 1.96),
                    mk(1.795, 4.65), mk(2.054, 1.47), mk(2.125, 0.72)];
    const frame = { detectedPlanes: new Set(planes),
      getPose: function (space) { return { transform: { matrix: space.matrix } }; } };
    for (let i = 0; i < 25; i++) arNoteHeight(1.10);
    for (let i = 0; i < 5; i++) arNoteHeight(0.55);     // one crouch under a shelf
    AR.refY = arRefHeight();
    arReadPlanes(frame, { x: 0, y: 1.20, z: 0 });
    const s = CoverageMap.roomSummary(AR.room, AR.map);
    return { refY: AR.refY, floor: s.floorY, ceiling: s.ceilingY, height: s.height,
             kind: AR.room.phone.floor.kind };
  })()`, ctx);
  check(Math.abs(p.refY - 1.10) < 0.02,
        "the reference height is the median, so crouching once does not move it (" + p.refY + ")");
  check(p.kind === "level",
        "the page names a plane's orientation and leaves floor or ceiling to the engine (" + p.kind + ")");
  check(Math.abs(p.floor + 0.539) < 0.05, "the step down is the floor (" + p.floor.toFixed(3) + " m)");
  check(Math.abs(p.ceiling - 1.792) < 0.05, "the ceiling patches merge into one level (" + p.ceiling.toFixed(3) + " m)");
  check(Math.abs(p.height - 2.33) < 0.08,
        "and his room is 2.33 m tall, not the 1.60 m reported (" + p.height.toFixed(2) + " m)");
} catch (e) {
  check(false, "his take replays through the page's plane ingest", e.name + ": " + e.message);
}

// ---------- a sample on a known floor keeps a level normal -------------------
try {
  const s = vm.runInContext(`(function(){
    AR.map = CoverageMap.createMap(); AR.room = CoverageMap.createRoom();
    AR.room.phone = { floor: { y: 0.0, area: 8 }, ceiling: null, walls: [] };
    const onFloor  = arSnapNormal(0.03, { x: 0.20, y: 0.90, z: 0.10 });
    const wall     = arSnapNormal(0.02, { x: 0.98, y: 0.10, z: 0.10 });
    const elsewhere= arSnapNormal(1.40, { x: 0.00, y: 1.00, z: 0.00 });
    return { flat: Math.abs(onFloor.y) > 0.999 && onFloor.x === 0 && onFloor.z === 0,
             tilted: Math.abs(wall.y - 0.10) < 1e-9, away: Math.abs(elsewhere.x) < 1e-9 };
  })()`, ctx);
  check(s.flat, "a near-level sample sitting on a measured floor is snapped to vertical");
  check(s.tilted, "a wall crossing that same height keeps its own normal");
  check(s.away, "and a sample away from any level surface is left alone");
} catch (e) {
  check(false, "normals snap onto measured level surfaces", e.name + ": " + e.message);
}

// ---------- coaching order: what gets told to you, over what -----------------
// His take is the shape this has to get right: turning on the spot closed the
// ring, so there was no blind sector to report, and the ceiling cue still sat
// under a 1 m² cross and never came up.
try {
  const c = vm.runInContext(`(function(){
    const vis = { thin: 0, partial: 0, covered: 40 };
    const ceil = { ceilingY: 2.4, ceilingSeen: 0.16, floorY: 0, floorSeen: 0.8 };
    const gap = { count: 100 };                       // 1 m^2 of unfinished surface
    const closed = arCoachLine(vis, ceil, 0, gap, 0.01, 40);
    const wide   = arCoachLine(vis, ceil, 110, gap, 0.01, 40);
    const bare   = arCoachLine(vis, { ceilingY: null, ceilingSeen: 0, floorY: null },
                               null, null, 0.01, 40);
    const early  = arCoachLine(vis, ceil, 0, gap, 0.01, 8);
    const done   = arCoachLine(vis, { ceilingY: 2.4, ceilingSeen: 0.95, floorY: 0 },
                               null, null, 0.01, 40);
    return { closed: closed.msg, wide: wide.msg, bare: bare.msg,
             early: early.msg, done: done.msg, doneCls: done.cls };
  })()`, ctx);
  check(/Ceiling is 84% unseen/.test(c.closed),
        "a closed ring with an unstarted ceiling says so (" + c.closed + ")");
  check(/turn and look that way/.test(c.wide),
        "a whole wall never looked at still outranks the ceiling (" + c.wide + ")");
  check(/No floor worked out yet/.test(c.bare), "and a missing floor is raised on its own (" + c.bare + ")");
  check(!/Ceiling|floor/.test(c.early), "neither room cue interrupts the first 15 seconds (" + c.early + ")");
  check(/All green/.test(c.done) && c.doneCls === "good", "a finished ceiling stops the nagging");
} catch (e) {
  check(false, "the coaching cues are ranked", e.name + ": " + e.message);
}

// A geometry feature must never be a reason the scan will not start.
check(/optionalFeatures: \[[^\]]*"plane-detection"/.test(html),
      "plane detection is asked for as an option the phone may refuse");
check(/initiateRoomCapture/.test(html) && /typeof s\.initiateRoomCapture === "function"/.test(html),
      "the room update is called only where the build actually has it");

// The room has to reach the operator in words, and the phone tab has to say
// whether the numbers came from the phone or from this page.
try {
  const rows = vm.runInContext(`(function(){
    AR.refY = 1.12;
    const s = { floorY: 0.0, ceilingY: 2.41, height: 2.41, ceilingSeen: 0.12,
                floorSeen: 0.8, wallCount: 3, spanA: 4.1, spanB: 5.2, source: "phone" };
    const got = {};
    for (const r of roomRows(s)) got[r[0]] = r[1];
    AR.refY = null;                       // the first second of a take
    const early = {};
    for (const r of roomRows(s)) early[r[0]] = r[1];
    return { got, early };
  })()`, ctx);
  check(/found/.test(rows.got.Floor || ""), "the floor is reported in plain words (" + rows.got.Floor + ")");
  check(/2\.41 m above/.test(rows.got.Ceiling || ""), "the ceiling is a height above the floor, not a bare number (" + rows.got.Ceiling + ")");
  check(/12%/.test(rows.got["Ceiling seen"] || ""), "and how much of it has been looked at (" + rows.got["Ceiling seen"] + ")");
  check(/4\.1.*5\.2/.test(rows.got.Walls || ""), "the walls give a room size an operator can check (" + rows.got.Walls + ")");
  check(/1\.12 m above the floor/.test(rows.got["Scan height"] || ""),
        "the height the split is judged against is on screen (" + rows.got["Scan height"] + ")");
  check(!rows.early["Scan height"], "and before there is one, the row is absent rather than 0.00 m");
} catch (e) {
  check(false, "the room reaches the readout in plain words", e.name + ": " + e.message);
}

console.log("");
console.log(fail === 0 ? "ALL " + pass + " CAPTURE-SCRIPT CHECKS PASSED"
                       : pass + " passed, " + fail + " FAILED");
process.exit(fail === 0 ? 0 : 1);
