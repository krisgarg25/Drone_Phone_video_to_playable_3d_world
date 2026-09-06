#!/usr/bin/env node
/*
 * test_ar_overlay.js — verify what the AR pass draws over the camera image.
 *
 * The world markers are drawn with matrices built by hand (a rigid world->camera
 * inverse plus the XR projection), so a sign error would put every line in the
 * wrong place on screen and look exactly like bad tracking. Headless Chromium
 * gives a real GL driver, so this checks what a string match cannot:
 *
 *   1. the marker shaders compile and link;
 *   2. a surfel pushed through the marker matrices lands where
 *      CoverageMap.project says it should, to within a pixel;
 *   3. arCountInView tallies the coverage states in frame without drawing them,
 *      so the HUD, the coach line and the report stay honest;
 *   4. the cage, the gap beacon and the plane outlines actually rasterise, in
 *      the right colour each;
 *   5. THE CAMERA VIEW STAYS CLEAR. The per-surfel coverage shell that used to
 *      live here put one filled quad on every visible surfel and buried the
 *      camera image under ~1600 of them; that variant stays gone, and this test
 *      fails if any of the four filled-overlay code paths come back. What
 *      returned in V9.2 is a thin wireframe square per on-screen surfel at
 *      alpha 0.28, colour-coded by CoverageMap state (thin red, partial amber,
 *      covered light blue) so the coach line's "Red areas" and "Amber areas"
 *      refer to something the operator can see. The lattice has to stay under
 *      5% screen coverage — that is the real "clear view" contract now.
 *
 * Run: node tests/test_ar_overlay.js      (SKIPs if playwright-core is absent)
 */
"use strict";
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
let chromium = null;
try {
  ({ chromium } = require(path.join(ROOT, "scratch", "pw", "node_modules", "playwright-core")));
} catch (e) {
  console.log("SKIP  playwright-core not found in scratch/pw — cannot drive a real GL context");
  process.exit(0);
}

let pass = 0, fail = 0;
function check(cond, label, detail) {
  if (cond) { pass++; console.log("pass  " + label); }
  else { fail++; console.log("FAIL  " + label + (detail ? "  -> " + detail : "")); }
}

// capture.html loads /viewer/*.js by absolute path, which file:// cannot resolve.
// Rewrite to relative and drop the copy in scratch/ so the page runs unchanged.
function materialise() {
  const html = fs.readFileSync(path.join(ROOT, "viewer", "capture.html"), "utf8")
    .replace(/src="\/viewer\//g, 'src="../viewer/');
  const out = path.join(ROOT, "scratch", "_ar_overlay_test.html");
  fs.writeFileSync(out, html);
  return "file:///" + out.replace(/\\/g, "/");
}

(async () => {
  const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  const errs = [];
  page.on("pageerror", e => errs.push(e.message));
  await page.goto(materialise(), { waitUntil: "load" });
  await page.waitForTimeout(400);

  const r = await page.evaluate(() => {
    const out = {};
    // A portrait phone: tall viewport, wide sensor. Square pixels mean the
    // projection carries the VIEWPORT aspect, i.e. the image is cropped.
    const VP = { w: 1080, h: 2280 }, IMG = { w: 1280, h: 720 };
    const fx = 1.35, fy = fx * (VP.w / VP.h);
    // A real XR projection: w_clip = -z_view, so the GPU draw and the CPU
    // replication below are the same arithmetic and check 2 means something.
    const near = 0.1, far = 50;
    const P = new Float32Array([fx, 0, 0, 0,
                                0, fy, 0, 0,
                                0, 0, -(far + near) / (far - near), -1,
                                0, 0, -2 * far * near / (far - near), 0]);

    arInitGL();
    // portrait canvas so the screenshot shows what the phone would show
    const c0 = document.getElementById("ar-gl");
    c0.width = 540; c0.height = 1140;
    out.gl = !!AR.gl;
    out.programOk = !!(AR.gl && AR.progMark && AR.gl.getProgramParameter(AR.progMark, AR.gl.LINK_STATUS));
    out.shaderLog = AR.gl && AR.progMark ? AR.gl.getProgramInfoLog(AR.progMark) : "no gl";

    // --- 1. nothing may paint per-surfel colour over the camera any more -----
    out.gone = {
      buildOverlay: typeof arBuildOverlay === "undefined",
      drawOverlay: typeof arDrawOverlay === "undefined",
      buildDots: typeof arBuildDots === "undefined",
      drawDots: typeof arDrawDots === "undefined",
      patProgram: !("progPat" in AR) && !("patCount" in AR),
      dotProgram: !("progDot" in AR) && !("dotCount" in AR)
    };

    // --- a wall 1.4 m ahead, with two coverage states present ---
    AR.map = CoverageMap.createMap();
    const cam = { x: 0, y: 1.4, z: 0 };
    const put = (x, y, z, cams) => {
      for (const cc of cams) {
        const L = Math.hypot(cc.x - x, cc.y - y, cc.z - z);
        CoverageMap.observe(AR.map, cc, { x, y, z }, { x: (cc.x - x) / L, y: (cc.y - y) / L, z: (cc.z - z) / L });
      }
    };
    // Azimuth bins are 45 deg wide, so "three distinct angles" needs real spread:
    // straight on, ~50 deg to the left, and ~46 deg right from below. All three
    // stay inside goodDist (3.5 m) so they count toward covered.
    const one = [{ x: 0, y: 1.3, z: 0.5 }];
    const three = [{ x: 0, y: 1.3, z: 0.5 }, { x: -2.0, y: 1.3, z: 0.5 }, { x: 2.3, y: 0.6, z: 0.5 }];
    for (let i = 0; i < 16; i++) {                      // left half: one angle -> thin
      put(-1.9 + i * 0.12, 1.2, -1.4, one);
    }
    for (let i = 0; i < 16; i++) {                      // right half: 3 angles -> covered
      put(0.1 + i * 0.12, 1.2, -1.4, three);
    }
    out.surfels = AR.map.surfels.length;

    // camera-to-world for a camera at `cam` looking down -Z (identity rotation)
    const m = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, cam.x, cam.y, cam.z, 1]);
    AR.viewport = { w: VP.w, h: VP.h }; AR.camW = IMG.w; AR.camH = IMG.h;
    AR.depthOK = false;
    arWorldToCamera(m, AR.wc);

    // --- 2. the marker matrices must agree with project(), to a pixel --------
    const dots = CoverageMap.project(AR.map, { pos: cam, quat: { x: 0, y: 0, z: 0, w: 1 } },
                                     VP.w, VP.h, fx, fy, 0, 0);
    out.visible = dots.length;
    let worst = 0;
    const s0 = AR.map.surfels[dots[0].idx];
    for (const d of dots) {
      const s = AR.map.surfels[d.idx];
      // replicate the marker vertex shader: clip = uP * (uWC * vec4(aC, 1))
      const cx = AR.wc[0] * s.x + AR.wc[4] * s.y + AR.wc[8] * s.z + AR.wc[12];
      const cy = AR.wc[1] * s.x + AR.wc[5] * s.y + AR.wc[9] * s.z + AR.wc[13];
      const cz = AR.wc[2] * s.x + AR.wc[6] * s.y + AR.wc[10] * s.z + AR.wc[14];
      const cw = -cz;
      const ndcX = (P[0] * cx) / cw, ndcY = (P[5] * cy) / cw;
      const px = (ndcX * 0.5 + 0.5) * VP.w, py = (1 - (ndcY * 0.5 + 0.5)) * VP.h;
      worst = Math.max(worst, Math.abs(px - d.px), Math.abs(py - d.py));
    }
    out.matrixErrPx = worst;
    out.sampleSurfel = [s0.x, s0.y, s0.z];

    // --- 3. the in-view tally, which draws nothing ---------------------------
    arCountInView({ pos: cam, quat: { x: 0, y: 0, z: 0, w: 1 } }, fx, fy, 0, 0);
    out.inView = AR.inView;
    out.vis = JSON.parse(JSON.stringify(AR.vis));
    out.tallySum = AR.vis.thin + AR.vis.partial + AR.vis.covered;

    // --- 4. build the world lines and rasterise for real --------------------
    AR.box = { x0: -1.2, x1: 1.2, y0: 0.2, y1: 2.2, z0: -1.7, z1: 0.4 };
    AR.gap = { cx: 0, cy: 1.2, cz: -0.9, count: 60 };
    // Two detected planes, inset from the cage so nothing z-fights with it.
    const ring = (pts) => ({ kind: pts.kind, n: 4, poly: new Float32Array(pts.xyz) });
    AR.planeIdx = new Map([
      ["floor", ring({ kind: "level", xyz: [-1.0, 0.25, -1.5, 1.0, 0.25, -1.5, 1.0, 0.25, 0.2, -1.0, 0.25, 0.2] })],
      ["wall", ring({ kind: "wall", xyz: [-1.0, 0.4, -1.65, 1.0, 0.4, -1.65, 1.0, 2.0, -1.65, -1.0, 2.0, -1.65] })]
    ]);
    arBuildMarkers();
    out.markCount = AR.markCount;
    out.markStride = MARK_STRIDE;
    out.markCapacity = AR.markData.length;
    out.markPaired = AR.markCount > 0 && AR.markCount % 2 === 0;

    const gl = AR.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    arDrawMarkers(P);
    out.glError = gl.getError();

    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    // Every marker colour is neutral or blue-leaning, so B >= R on all of them.
    // A pixel with R meaningfully above B can only have come from the red/amber
    // coverage ramp -- which is exactly what must never reach the camera view.
    let lit = 0, warm = 0, beacon = 0, cage = 0, level = 0, wall = 0;
    for (let i = 0; i < w * h; i++) {
      const R = px[i * 4], G = px[i * 4 + 1], B = px[i * 4 + 2];
      if (R + G + B < 40) continue;
      lit++;
      if (R > B + 20) { warm++; continue; }
      if (R > 200 && G > 200 && B > 200) level++;        // floor/ceiling -> white
      else if (B > 200 && R >= 110) wall++;              // wall    -> pale blue
      else if (B > 200 && R < 110 && G > 120) beacon++;  // beacon  -> blue
      else cage++;                                       // cage    -> grey
    }
    out.lit = lit; out.warm = warm;
    out.beacon = beacon; out.cage = cage; out.level = level; out.wall = wall;
    out.canvas = [w, h];
    out.coverFrac = lit / (w * h);

    // show the canvas so the screenshot is the AR view, not the hero
    document.getElementById("perm-hero").style.display = "none";
    document.getElementById("view-ar").classList.add("active");
    return out;
  });

  await page.screenshot({ path: path.join(ROOT, "scratch", "ar_markers_test.png") });
  await browser.close();

  console.log(JSON.stringify(r, null, 2));
  console.log("");
  check(r.gl, "arInitGL obtained a real WebGL context");
  check(r.programOk && (!r.shaderLog || r.shaderLog.length === 0),
        "the world-marker shaders compile and link clean", r.shaderLog);
  check(Object.values(r.gone).every(Boolean),
        "the per-surfel coverage shell and the flat dot painter are gone",
        JSON.stringify(r.gone));
  check(r.visible > 12, "CoverageMap.project returned visible surfels (" + r.visible +
        " of " + r.surfels + "; the wall is wider than the frustum)");
  check(r.matrixErrPx < 1.5,
        "marker matrices agree with project() to under 1.5 px (worst " + r.matrixErrPx.toFixed(3) + " px)");
  check(r.inView > 12 && r.inView <= r.visible,
        "arCountInView counted " + r.inView + " surfels in frame (of " + r.visible +
        " inside the projection margin)");
  check(r.tallySum === r.inView,
        "every counted surfel landed in exactly one state bucket (" + r.tallySum + " vs " + r.inView + ")");
  check(r.vis.covered > 0 && r.vis.thin > 0,
        "the tally sees both states (thin " + r.vis.thin + ", covered " + r.vis.covered + ")");
  check(r.markCount >= 24 + 6 + 16 && r.markCount * r.markStride <= r.markCapacity,
        "world markers built (cage 24 + beacon 6 + two plane rings 16 verts, got " + r.markCount + ")");
  check(r.markPaired, "marker vertices come in pairs, as gl.LINES needs");
  check(r.glError === 0, "no GL error after drawing the markers", "0x" + r.glError.toString(16));
  check(r.warm > 0,
        "the shell's thin/partial states paint a red/amber lattice the operator can see (" +
        r.warm + " warm px)");
  check(r.lit > 300, "the markers actually rasterised (" + r.lit + " lit px of " +
        (r.canvas[0] * r.canvas[1]) + ")");
  check(r.coverFrac < 0.05,
        "the camera view stays clear: markers cover " + (r.coverFrac * 100).toFixed(2) +
        "% of the screen (must stay under 5%)");
  check(r.beacon > 20, "the blue gap beacon drew (" + r.beacon + " px)");
  check(r.cage > 100, "the grey cage drew (" + r.cage + " px)");
  check(r.level > 20 && r.wall > 20,
        "both plane outlines drew in their own colour (floor " + r.level + " px white, wall " +
        r.wall + " px pale blue)");
  check(errs.length === 0, "no page errors during the overlay run", errs.join(" | "));

  console.log("\n" + (fail === 0 ? "ALL " + pass + " AR-OVERLAY CHECKS PASSED" : pass + " passed, " + fail + " FAILED"));
  process.exit(fail === 0 ? 0 : 1);
})().catch(e => { console.log("HARNESS ERROR: " + e.message); process.exit(1); });
