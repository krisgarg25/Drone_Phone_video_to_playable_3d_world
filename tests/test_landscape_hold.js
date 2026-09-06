#!/usr/bin/env node
/*
 * test_landscape_hold.js — does a scan held sideways mean the same thing as one
 * held upright, and did fixing the sideways case cost the upright case anything?
 *
 * Everything here is decided by geometry the phone reports, so a string match
 * cannot check it: which way up the handset is held changes the shape of the
 * recorded take, the shape of the XR viewport, and the relation between the
 * phone's distance image and the view those distances are read through. Headless
 * Chromium is used as the oracle for the last of those, because the transform
 * arrives as a DOMMatrix and the only trustworthy account of what its members mean
 * is a browser applying it.
 *
 * Run: node tests/test_landscape_hold.js     (SKIPs if playwright-core is absent)
 */
"use strict";
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
let chromium = null;
try {
  ({ chromium } = require(path.join(ROOT, "scratch", "pw", "node_modules", "playwright-core")));
} catch (e) {
  console.log("SKIP  playwright-core not found in scratch/pw — cannot drive a real browser");
  process.exit(0);
}

let pass = 0, fail = 0;
function check(cond, label, detail) {
  if (cond) { pass++; console.log("pass  " + label); }
  else { fail++; console.log("FAIL  " + label + (detail ? "  -> " + detail : "")); }
}

function materialise() {
  const html = fs.readFileSync(path.join(ROOT, "viewer", "capture.html"), "utf8")
    .replace(/src="\/viewer\//g, 'src="../viewer/');
  const out = path.join(ROOT, "scratch", "_landscape_hold_test.html");
  fs.writeFileSync(out, html);
  return "file:///" + out.replace(/\\/g, "/");
}

(async () => {
  const browser = await chromium.launch();
  const url = materialise();

  // ---------- the page loads in either hold, with no forced orientation ------
  for (const vp of [{ w: 390, h: 844, want: "portrait" }, { w: 844, h: 390, want: "landscape" }]) {
    const page = await browser.newPage({ viewport: { width: vp.w, height: vp.h } });
    const errs = [];
    page.on("pageerror", e => errs.push(e.message));
    await page.goto(url, { waitUntil: "load" });
    await page.waitForTimeout(250);

    const r = await page.evaluate(() => ({
      resolved: holdResolved(),
      note: document.getElementById("hold-note").textContent,
      modes: [...document.querySelectorAll("#hold-tabs .tab")].map(b => b.dataset.hold),
      canvas: arSizeCanvasTo(innerWidth, innerHeight)
    }));
    check(errs.length === 0, `page loads held ${vp.want} with no script error`, errs.join(" | "));
    check(r.modes.join(",") === "auto,portrait,landscape",
          "the hold is asked in three choices, auto first", r.modes.join(","));
    check(r.resolved === vp.want, `held ${vp.want}: auto resolves to ${vp.want}, not to a guess`, r.resolved);
    check(/[Hh]eld (vertical|horizontal)/.test(r.note), `the note says what it resolved to: "${r.note}"`);

    // What arInitGL does when a scan starts: the first frames are already the
    // right shape rather than a hardcoded one waiting to be corrected.
    check(Math.abs(r.canvas[0] / r.canvas[1] - vp.w / vp.h) < 0.02,
          `capture canvas follows the screen (${r.canvas[0]}x${r.canvas[1]} for ${vp.w}x${vp.h})`);
    await page.close();
  }

  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await page.goto(url, { waitUntil: "load" });

  // ---------- nothing else in the page forces a hold --------------------------
  const src = fs.readFileSync(path.join(ROOT, "viewer", "capture.html"), "utf8");
  const locks = src.match(/\.lock\(/g) || [];
  check(locks.length === 1, "exactly one place asks the phone to hold still", "found " + locks.length);
  check(!/lock\("landscape"\)/.test(src) && !/lock\("portrait"\)/.test(src),
        "no hardcoded hold anywhere: the request comes from the operator's choice");

  // ---------- the recorded frame, both ways up --------------------------------
  const sized = await page.evaluate(() => {
    const out = {};
    // A 2.17:1 phone, held both ways: the two aspects the old code met with a
    // hardcoded 16:9 and an odd rounded edge.
    for (const [name, ar] of [["tall", 591 / 1280], ["wide", 1280 / 591], ["square", 1],
                              ["three", 4 / 3], ["cinema", 1.9]]) {
      const wh = arSizeCanvasTo(1000 * ar, 1000);
      out[name] = wh;
    }
    return out;
  });
  const allEven = Object.entries(sized).every(([k, v]) => v && v[0] % 2 === 0 && v[1] % 2 === 0 &&
                                                    Math.max(v[0], v[1]) <= 1280 && v[0] > 0 && v[1] > 0);
  check(allEven, "every phone shape records with even edges under the cap: " +
        JSON.stringify(sized), "");
  check(sized.tall[0] === 590 && sized.tall[1] === 1280,
        "the working vertical take keeps 1280 on its long edge (590x1280, was 591x1280)",
        JSON.stringify(sized.tall));
  check(sized.wide[0] === 1280 && sized.wide[1] === 590,
        "the same phone held sideways gets the mirror image, not a portrait frame",
        JSON.stringify(sized.wide));

  // ---------- the camera crop, both ways up ----------------------------------
  const fit = await page.evaluate(() => {
    const save = [AR.viewport, AR.camW, AR.camH];
    AR.viewport = { w: 1080, h: 2280 }; AR.camW = 1280; AR.camH = 720;
    const tall = arFit();
    // The same phone turned sideways: screen and image both transpose, so the crop
    // has to swap axes and keep its value rather than change shape.
    AR.viewport = { w: 2280, h: 1080 }; AR.camW = 720; AR.camH = 1280;
    const wide = arFit();
    [AR.viewport, AR.camW, AR.camH] = save;
    return { tall, wide };
  });
  check(Math.abs(fit.tall.x - 0.266) < 0.01 && fit.tall.y === 1,
        "vertical unchanged: a tall screen crops a wide sensor horizontally only (x=" +
        fit.tall.x.toFixed(3) + ")");
  check(Math.abs(fit.wide.x - 1) < 1e-9 && Math.abs(fit.wide.y - fit.tall.x) < 0.01,
        "horizontal is the exact mirror of it: same crop, other axis (y=" +
        fit.wide.y.toFixed(3) + " against x=" + fit.tall.x.toFixed(3) + ")");

  // ---------- the distance image's mapping, with the browser as the oracle ----
  const mapped = await page.evaluate(() => {
    // A real non-identity 2D transform, the shape a phone gives when its distance
    // image does not already line up with the view.
    const m = new DOMMatrix([0.8, 0, 0, 1.25, 0.1, -0.05]);   // a,b,c,d,e,f
    const probe = { width: 160, height: 120, normDepthBufferFromNormView: m };
    const map = arDepthMapping(probe);
    const cells = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.2, 0.9]];
    return {
      six: map ? map.fwd : null,
      // what the browser itself says a view point maps to
      browserFwd: cells.map(([x, y]) => { const p = m.transformPoint({ x, y }); return [p.x, p.y]; }),
      oursFwd: cells.map(([x, y]) => arApplyXY(map.fwd, x, y).slice()),
      roundTrip: cells.map(([x, y]) => {
        const d = arApplyXY(map.fwd, x, y);
        return arApplyXY(map.inv, d[0], d[1]).slice();
      }),
      // a phone that reports nothing usable must leave the old behaviour intact
      noMapping: arDepthMapping({ width: 160, height: 120 }),
      identity: (() => {
        const i = arDepthMapping({ normDepthBufferFromNormView: new DOMMatrix([1, 0, 0, 1, 0, 0]) });
        return i ? i.inv : null;
      })(),
      // Chromium exposes no `matrix` member; a reader that needs one reports absent
      matrixMember: "matrix" in m
    };
  });
  check(mapped.six !== null, "the phone's mapping is read from a DOMMatrix at all");
  const fwdOk = mapped.oursFwd.every((p, i) =>
    Math.abs(p[0] - mapped.browserFwd[i][0]) < 1e-9 && Math.abs(p[1] - mapped.browserFwd[i][1]) < 1e-9);
  check(fwdOk, "forward mapping agrees with the browser's own transformPoint: " +
        JSON.stringify(mapped.browserFwd) + " vs " + JSON.stringify(mapped.oursFwd));
  const rtOk = mapped.roundTrip.every((p, i) =>
    Math.abs(p[0] - [[0, 0], [0.5, 0.5], [1, 1], [0.2, 0.9]][i][0]) < 1e-9 &&
    Math.abs(p[1] - [[0, 0], [0.5, 0.5], [1, 1], [0.2, 0.9]][i][1]) < 1e-9);
  check(rtOk, "a depth pixel turned back into a view ray comes home: " + JSON.stringify(mapped.roundTrip));
  check(mapped.identity && Math.abs(mapped.identity[0] - 1) < 1e-12 && Math.abs(mapped.identity[5]) < 1e-12,
        "an identity mapping stays identity, so a phone that already lines up is untouched");
  check(mapped.noMapping === null, "no mapping object means no transform applied, not a wrong one");
  check(mapped.matrixMember === false,
        "recorded: Chromium's DOMMatrix has no `matrix` member, so reading only that reports absence");

  // ---------- where the probe grid actually gets filled ----------------------
  const grid = await page.evaluate(() => {
    const wall = 2.0;
    const m = new DOMMatrix([0.8, 0, 0, 1.25, 0.1, -0.05]);
    const data = new Uint16Array(160 * 120 * 2);
    for (let i = 0; i < 160 * 120; i++) { data[i * 2] = 2000; data[i * 2 + 1] = 255; }
    const info = { width: 160, height: 120, rawValueToMeters: 0.001, data,
                   normDepthBufferFromNormView: m,
                   getDepthInMeters: () => wall };
    const frame = { getDepthInformation: () => info };
    const view = { projectionMatrix: null, transform: {} };
    // column-major identity camera-to-world; a level, upright camera at the origin
    const camToWorld = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    const P = new Float32Array(16);
    P[0] = 1.4; P[5] = 1.6; P[8] = 0.05; P[9] = -0.03; P[10] = -1; P[11] = -1; P[14] = -2; P[15] = 1;
    const map = CoverageMap.createMap();
    const savedMap = AR.map;
    AR.map = map;
    // arInitGL owns these allocations and only runs when a scan starts; the probe
    // needs them to exist.
    const n = AR.GW * AR.GH;
    AR.gPts = new Float32Array(n * 3); AR.gOK = new Uint8Array(n); AR.gDepth = new Float32Array(n);
    const observed = [];
    const realObserve = CoverageMap.observe;
    CoverageMap.observe = function (mm, cam, pt, nrm) { observed.push([pt.x, pt.y, pt.z]); return 0; };
    arReadDepth(frame, view, camToWorld, P[0], P[5], P[8], P[9], { x: 0, y: 0, z: 0 });
    CoverageMap.observe = realObserve;
    const space = AR.gridSpace, used = AR.depthToView ? true : false;
    AR.map = savedMap;
    return { n: observed.length, first: observed[0], space, used,
             // the same cell unprojected without the phone's mapping, i.e. the old
             // behaviour, to prove the two are not the same point
             without: (() => {
               const p = CoverageMap.unprojectDepth(camToWorld, 0.5 / AR.GW, 0.5 / AR.GH, wall,
                                                    P[0], P[5], P[8], P[9], {});
               return [p.x, p.y, p.z];
             })() };
  }).catch(e => ({ error: String(e) }));
  check(!grid.error, "the distance probe runs against a synthetic phone depth image", grid.error || "");
  if (!grid.error) {
    check(grid.n > 0 && grid.space === "depth" && grid.used,
          "the raw-buffer branch fills the grid in the distance image's space and says so (" +
          grid.n + " samples)");
    check(grid.first && Math.abs(grid.first[2]) > 1.5 && Math.abs(grid.first[2]) < 2.6,
          "a 2 m wall lands 2 m away along the ray, not at a made-up distance: z=" +
          (grid.first ? grid.first[2].toFixed(3) : "—"));
    const moved = grid.first && Math.abs(grid.first[0] - grid.without[0]) > 1e-6;
    check(moved, "and the point is genuinely the transformed ray, not the untransformed one (" +
          grid.first[0].toFixed(4) + " vs " + grid.without[0].toFixed(4) + ")");
  }

  // ---------- which way it aimed, held either way -----------------------------
  const bearing = await page.evaluate(() => {
    const snap = () => {
      let best = -1, bi = 0;
      for (let i = 0; i < covSphere.length; i++) if (covSphere[i] > best) { best = covSphere[i]; bi = i; }
      return { row: Math.floor(bi / COV_YAW), col: bi % COV_YAW, peak: best };
    };
    const clear = () => covSphere.fill(0);
    clear();
    // level and north: identity quaternion looks down -Z
    eulerToQuat(camQuat, 0, 0, 0);
    markBearing(1);
    const north = snap();
    clear();
    // the same view, now described the way a sideways phone's sensors do: the
    // camera axis is unchanged by rolling the handset about it
    eulerToQuat(camQuat, 0, 0, Math.PI / 2);
    markBearing(1);
    const rolled = snap();
    clear();
    // 45 deg up: rotating the camera about its own X axis by +45 lifts the view
    eulerToQuat(camQuat, Math.PI / 4, 0, 0);
    markBearing(1);
    const up = snap();
    clear();
    return { north, rolled, up, rows: COV_PITCH, cols: COV_YAW };
  });
  const CENTRE_ROW = Math.floor(90 / 180 * bearing.rows);
  check(bearing.north.row === CENTRE_ROW && bearing.north.col === 0,
        "level and north bins at the centre row and column (row " + bearing.north.row + ", col " +
        bearing.north.col + ")");
  check(bearing.rolled.row === bearing.north.row && bearing.rolled.col === bearing.north.col,
        "the bearing tracks the optical axis: rolling the camera about it moves nothing",
        JSON.stringify(bearing.rolled) + " vs " + JSON.stringify(bearing.north));
  check(bearing.up.row < bearing.north.row,
        "looking up moves the evidence up the strip (row " + bearing.up.row + " from " +
        bearing.north.row + ")");

  // ---------- the tap on Vertical/Horizontal is also the tap that grabs fullscreen
  const fsTest = await page.evaluate(async () => {
    const calls = { request: 0, exit: 0 };
    const keepRfs = Element.prototype.requestFullscreen;
    const keepXfs = Document.prototype.exitFullscreen;
    let fakeFs = null;
    Object.defineProperty(document, "fullscreenElement", { configurable: true, get: () => fakeFs });
    Element.prototype.requestFullscreen = function () {
      calls.request++; fakeFs = this; return Promise.resolve();
    };
    Document.prototype.exitFullscreen = function () {
      calls.exit++; fakeFs = null; return Promise.resolve();
    };
    await setHold("landscape");
    const afterLandscape = { request: calls.request, exit: calls.exit, mode: HOLD.mode };
    await setHold("auto");
    const afterAuto = { request: calls.request, exit: calls.exit, mode: HOLD.mode };
    Element.prototype.requestFullscreen = keepRfs;
    Document.prototype.exitFullscreen = keepXfs;
    return { afterLandscape, afterAuto };
  });
  check(fsTest.afterLandscape.request === 1 && fsTest.afterLandscape.mode === "landscape",
        "tapping Horizontal asks for fullscreen — Chrome's orientation lock needs it",
        JSON.stringify(fsTest.afterLandscape));
  check(fsTest.afterAuto.request === 1 && fsTest.afterAuto.exit === 1,
        "returning to Auto releases fullscreen so the browser chrome comes back",
        JSON.stringify(fsTest.afterAuto));

  // ---------- the surrounding chrome has a landscape shape, not just the AR HUD
  const css = await page.evaluate(() => {
    const blocks = [...document.styleSheets].flatMap(s => {
      try { return [...s.cssRules]; } catch (e) { return []; }
    });
    const landscape = blocks.filter(r =>
      r.conditionText && /orientation:\s*landscape/.test(r.conditionText));
    const text = landscape.map(r => r.cssText).join("\n");
    return { selectors: [...text.matchAll(/(^|\})\s*([^{}]+)\{/g)].map(m => m[2].trim()) };
  });
  const wants = ["#perm-hero", "#dock", "#sheet", ".tabs.hold-tabs"];
  for (const w of wants) {
    check(css.selectors.some(s => s.includes(w)),
          `landscape CSS restyles ${w}, so the whole UI follows the phone once it locks`,
          css.selectors.join(" | "));
  }

  // ---------- the canvas re-fits when the phone turns mid-scan -----------------
  const refit = await page.evaluate(() => {
    const saveW = AR.camW, saveH = AR.camH, saveCap = AR.capWH;
    AR.camW = 590; AR.camH = 1280; AR.capWH = null;
    arFitCanvas();
    const tall = AR.capWH.slice();
    // The phone has now actually rotated: ARCore hands over swapped dimensions.
    AR.camW = 1280; AR.camH = 590;
    arFitCanvas();
    const wide = AR.capWH.slice();
    AR.camW = saveW; AR.camH = saveH; AR.capWH = saveCap;
    return { tall, wide };
  });
  check(refit.tall[0] === 590 && refit.tall[1] === 1280,
        "vertical first: canvas matches the phone's initial portrait shape",
        JSON.stringify(refit.tall));
  check(refit.wide[0] === 1280 && refit.wide[1] === 590,
        "after the phone turns, the canvas follows: the AR.sized latch no longer pins it",
        JSON.stringify(refit.wide));

  await browser.close();
  console.log(fail ? `\n${fail} FAILED, ${pass} passed` : `\nALL ${pass} LANDSCAPE-HOLD CHECKS PASSED`);
  process.exit(fail ? 1 : 0);
})();
