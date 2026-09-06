#!/usr/bin/env node
/*
 * test_lens_choice.js — does the page pick the widest lens it can actually reach,
 * and does a lens problem ever stop a scan?
 *
 * A wide lens carries more of the room per frame, which is what the solve eats, so
 * "Auto" has to mean widest in a way that survives the two things phones do: name
 * their lenses opaquely ("camera2 0, facing back") and advertise a zoom dial they
 * then refuse. Both are stubbed here rather than mocked away, because each check is
 * about what the page does with an answer it did not expect.
 *
 * The AR side is the other half. WebXR's XRCamera is width and height and the
 * session request has no lens option, so there the lens is measured from the
 * projection matrix instead of chosen — and that maths is checked against the
 * pinhole model the calibration file uses.
 *
 * Run: node tests/test_lens_choice.js     (SKIPs if playwright-core is absent)
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
  const out = path.join(ROOT, "scratch", "_lens_choice_test.html");
  fs.writeFileSync(out, html);
  return "file:///" + out.replace(/\\/g, "/");
}

// A phone's media stack, installed over the real one. `apply` decides what the
// handset does when asked for its widest zoom: honour it, ignore it, or refuse.
// The stream is stashed on window so the page's own functions can be handed it.
const INSTALL = (spec) => {
  const state = Object.assign({ zoom: 1, width: 1280, height: 720,
                                deviceId: spec.deviceId || "dev-main",
                                facingMode: "environment" }, spec.settings || {});
  window.__calls = { constraints: [], gum: [] };
  const track = {
    label: spec.label === undefined ? "camera2 0, facing back" : spec.label,
    getCapabilities: () => (spec.caps === undefined ? { zoom: { min: 0.5, max: 10 } } : spec.caps),
    getSettings: () => Object.assign({}, state),
    applyConstraints: (c) => {
      window.__calls.constraints.push(JSON.parse(JSON.stringify(c)));
      if (spec.apply === "throw") {
        const e = new Error("nope"); e.name = "OverconstrainedError";
        return Promise.reject(e);
      }
      if (spec.apply !== "ignore" && c && c.advanced && c.advanced[0] && "zoom" in c.advanced[0])
        state.zoom = c.advanced[0].zoom;
      return Promise.resolve();
    },
    stop: () => {}
  };
  // A real MediaStream, because videoEl.srcObject rejects a lookalike object; its
  // track list is ours, so every capability read still hits the stub.
  const ms = new MediaStream();
  ms.getVideoTracks = () => [track];
  ms.getAudioTracks = () => [];
  ms.getTracks = () => [track];
  window.__stream = ms;
  const fake = {
    enumerateDevices: () => Promise.resolve(spec.devices || []),
    getSupportedConstraints: () => ({ zoom: spec.dial !== false, facingMode: true }),
    getUserMedia: (c) => {
      window.__calls.gum.push(JSON.parse(JSON.stringify(c)));
      if (spec.gumThrowOnce && window.__calls.gum.length === 1) {
        const e = new Error("gone"); e.name = "OverconstrainedError";
        return Promise.reject(e);
      }
      return Promise.resolve(window.__stream);
    }
  };
  try { Object.defineProperty(navigator, "mediaDevices", { value: fake, configurable: true }); }
  catch (e) { navigator.mediaDevices = fake; }
  return true;
};

// What a four-lens Android phone looks like to enumerateDevices: the main lens
// carries no adjective at all, which is the whole reason labels cannot decide.
const LENSES = [
  { kind: "videoinput", deviceId: "dev-main", label: "camera2 0, facing back", groupId: "g0" },
  { kind: "videoinput", deviceId: "dev-uw", label: "camera2 2, facing back (ultra wide)", groupId: "g0" },
  { kind: "videoinput", deviceId: "dev-tele", label: "camera2 4, facing back (telephoto)", groupId: "g0" },
  { kind: "videoinput", deviceId: "dev-front", label: "camera2 1, facing front", groupId: "g1" },
  { kind: "audioinput", deviceId: "mic", label: "builtin microphone", groupId: "g2" }
];

(async () => {
  const browser = await chromium.launch();
  const url = materialise();

  // ---------- the row exists before any camera has been opened ---------------
  {
    const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
    const errs = [];
    page.on("pageerror", e => errs.push(e.message));
    await page.goto(url, { waitUntil: "load" });
    await page.waitForTimeout(300);
    const r = await page.evaluate(() => ({
      buttons: [...document.querySelectorAll("#lens-tabs .tab")].map(b => b.dataset.lens),
      on: [...document.querySelectorAll("#lens-tabs .tab.on")].map(b => b.dataset.lens),
      note: document.getElementById("lens-note").textContent,
      mode: LENS.mode
    }));
    check(errs.length === 0, "the page loads with the lens row and no script error", errs.join(" | "));
    check(r.buttons.length === 1 && r.buttons[0] === "auto",
          "the lens is asked, Auto first, with nothing invented before the phone answers",
          r.buttons.join(","));
    check(r.on.join(",") === "auto" && r.mode === "auto", "Auto is the choice on arrival");
    check(r.note.length > 10, `the note says something before a scan: "${r.note}"`);
    await page.close();
  }

  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  const errs = [];
  page.on("pageerror", e => errs.push(e.message));
  let dialogs = [];
  page.on("dialog", async d => { dialogs.push(d.message()); await d.dismiss(); });
  // Registered before navigation so every later callback can call INSTALL by name.
  await page.addInitScript({ content: `window.INSTALL = ${INSTALL.toString()};` });
  await page.goto(url, { waitUntil: "load" });
  await page.waitForTimeout(200);

  // ---------- labels only break ties, they never decide ----------------------
  {
    const r = await page.evaluate((lenses) => ({
      scores: lenses.map(d => [d.label, lensScore(d.label)]),
      names: lenses.map(d => lensShortName(d.label)),
      opaque: lensShortName("camera2 0, facing back"),
      blank: lensShortName("")
    }), LENSES);
    const by = Object.fromEntries(r.scores);
    check(by["camera2 1, facing front"] < 0,
          "a front camera is never a candidate for a room scan");
    check(by["camera2 2, facing back (ultra wide)"] > by["camera2 0, facing back"],
          "an ultrawide outranks the back lens that carries no adjective");
    check(by["camera2 0, facing back"] > by["camera2 4, facing back (telephoto)"],
          "a telephoto ranks below the lens that carries more of the room");
    check(r.opaque === "Camera 0",
          `an opaque label still gets a name a person can read: "${r.opaque}"`);
    check(r.names[1] === "Ultra wide" && r.names[2] === "Tele",
          "named lenses are offered under their names: " + r.names.slice(1, 3).join(" / "));
    check(r.blank === "the phone's default", "an empty label says so instead of inventing one");
  }

  // ---------- what is asked of the camera in each mode -----------------------
  {
    const r = await page.evaluate(() => {
      setLens("auto");
      const auto = lensConstraints();
      setLens("dev-uw");
      const manual = lensConstraints();
      const mode = LENS.mode, id = LENS.deviceId;
      setLens("auto");
      return { auto, manual, mode, id };
    });
    check(r.auto.video.facingMode && r.auto.video.facingMode.ideal === "environment" &&
          !("deviceId" in r.auto.video),
          "Auto asks for the back camera and lets the phone pick the lens");
    check(r.manual.video.deviceId && r.manual.video.deviceId.exact === "dev-uw" &&
          !("facingMode" in r.manual.video),
          "a chosen lens is asked for exactly, and facingMode is dropped so it cannot conflict");
    check(r.mode === "manual" && r.id === "dev-uw", "the choice is remembered as a deviceId");
  }

  // ---------- Auto means widest, and only when something wider exists --------
  {
    const r = await page.evaluate(async (lenses) => {
      INSTALL({ devices: lenses, caps: { zoom: { min: 0.5, max: 10 } }, apply: "ok" });
      setLens("auto");
      LENS.info = null;
      const info = await arPickLens(window.__stream);
      return { info, calls: window.__calls.constraints,
               note: document.getElementById("lens-note").textContent,
               buttons: [...document.querySelectorAll("#lens-tabs .tab")]
                 .map(b => b.dataset.lens + "|" + b.textContent) };
    }, LENSES);
    check(r.calls.length === 1 && r.calls[0].advanced[0].zoom === 0.5,
          "Auto asks for the bottom of the zoom dial, which on a multi-camera phone is the ultrawide",
          JSON.stringify(r.calls));
    check(r.info.zoomNow === 0.5 && r.info.zoomAsked === 0.5,
          `and reads back what the phone actually did (${r.info.zoomNow}x)`);
    check(r.info.zoomRange && r.info.zoomRange[0] === 0.5 && r.info.zoomRange[1] === 10,
          "the dial it was offered is recorded, so a narrow take can be explained later");
    check(r.info.label === "camera2 0, facing back" && r.info.deviceId === "dev-main",
          "the lens that answered is recorded by name and id");
    check(/Recording on .*0\.50/.test(r.note), `the start screen says so: "${r.note}"`);
    check(r.buttons.length === 3 && r.buttons[0].startsWith("auto|"),
          "the phone's other scan-worthy lenses become choices: " +
          r.buttons.slice(1).map(b => b.split("|")[1]).join(", "));
    check(!r.buttons.some(b => b.includes("Front")),
          "the front camera is not offered as a lens for scanning a room");
    check(!r.buttons.some(b => /undefined|camera2 0/.test(b.split("|")[1])),
          "every offered lens has a name a person can act on");
  }

  // ---------- a Samsung that names nothing, so the count has to be honest -----
  {
    // Real Samsung/Chromium emits bare "camera2 N, facing back" labels with no
    // "(ultra wide)" suffix. The lens picker cannot tell them apart from a name,
    // but it can exclude the front one and report the count that way. The
    // operator's "4 cameras seen" on a three-rear phone came from counting the
    // selfie camera as if it were a scan candidate.
    const SAMSUNG = [
      { kind: "videoinput", deviceId: "s0", label: "camera2 0, facing back", groupId: "g0" },
      { kind: "videoinput", deviceId: "s1", label: "camera2 1, facing back", groupId: "g0" },
      { kind: "videoinput", deviceId: "s2", label: "camera2 2, facing back", groupId: "g0" },
      { kind: "videoinput", deviceId: "s3", label: "camera2 3, facing front", groupId: "g1" }
    ];
    const r = await page.evaluate(async (lenses) => {
      INSTALL({ devices: lenses, caps: { zoom: { min: 1, max: 8 } }, apply: "ok" });
      setLens("auto"); LENS.info = null;
      await lensListDevices(null);
      updateLensNote();
      return {
        rear: LENS.devices.filter(d => !lensIsFront(d.label)).length,
        total: LENS.devices.length,
        note: document.getElementById("lens-note").textContent
      };
    }, SAMSUNG);
    check(r.total === 4 && r.rear === 3,
          "the phone lists four inputs but only three are backs",
          `total=${r.total}, rear=${r.rear}`);
    check(/3 rear cameras/.test(r.note),
          `the note counts rears, not raw videoinputs: "${r.note}"`);
    check(!/4 cameras seen/.test(r.note),
          "and does not claim a fourth scan-worthy lens on a three-rear phone");
  }

  // ---------- before permission, the count says what it can honestly see ------
  {
    const r = await page.evaluate(() => {
      // EnumerateDevices on Chrome returns empty labels until the user has
      // granted the camera, so lensIsFront() has nothing to filter on and the
      // front lens is in the raw list. Say so instead of pretending to know.
      INSTALL({ devices: [
        { kind: "videoinput", deviceId: "anon0", label: "", groupId: "" },
        { kind: "videoinput", deviceId: "anon1", label: "", groupId: "" },
        { kind: "videoinput", deviceId: "anon2", label: "", groupId: "" },
        { kind: "videoinput", deviceId: "anon3", label: "", groupId: "" }
      ], caps: { zoom: { min: 1, max: 8 } }, apply: "ok" });
      setLens("auto"); LENS.info = null;
      return lensListDevices(null).then(() => {
        updateLensNote();
        return { note: document.getElementById("lens-note").textContent };
      });
    });
    check(/4 video inputs listed/.test(r.note),
          `before labels, the note is honest that it cannot tell front from back: "${r.note}"`);
  }

  // ---------- nothing wider, no dial at all, and a dial that refuses ---------
  {
    const r = await page.evaluate(async () => {
      const out = {};
      INSTALL({ devices: [], caps: { zoom: { min: 1, max: 8 } }, apply: "ok" });
      setLens("auto"); LENS.info = null;
      out.min1 = { info: await arPickLens(window.__stream), calls: window.__calls.constraints.slice(),
                   note: document.getElementById("lens-note").textContent };

      INSTALL({ devices: [], caps: {}, apply: "ok" });
      setLens("auto"); LENS.info = null;
      out.nodial = { info: await arPickLens(window.__stream), calls: window.__calls.constraints.slice(),
                     note: document.getElementById("lens-note").textContent };

      INSTALL({ devices: [], caps: { zoom: { min: 0.6, max: 10 } }, apply: "throw" });
      setLens("auto"); LENS.info = null;
      out.refused = { info: await arPickLens(window.__stream), calls: window.__calls.constraints.slice(),
                      note: document.getElementById("lens-note").textContent };

      INSTALL({ devices: [], caps: { zoom: { min: 0.6, max: 10 } }, apply: "ignore" });
      setLens("dev-tele"); LENS.info = null;
      out.manual = { info: await arPickLens(window.__stream), calls: window.__calls.constraints.slice() };
      return out;
    });
    check(r.min1.calls.length === 0 && r.min1.info.zoomRange[0] === 1,
          "a dial that starts at 1.0 is left alone: there is nothing wider to ask for");
    check(r.nodial.calls.length === 0 && r.nodial.info.zoomRange === null,
          "a phone with no zoom capability is not asked for one");
    check(/no zoom dial/.test(r.nodial.note), `and says so: "${r.nodial.note}"`);
    check(r.refused.info.refused === "OverconstrainedError" && r.refused.info.zoomNow === 1,
          "a refusal is recorded as a refusal, not as success");
    check(/refused/.test(r.refused.note), `and reaches the operator: "${r.refused.note}"`);
    check(r.manual.calls.length === 1 && r.manual.calls[0].advanced[0].zoom === 0.6,
          "a lens picked by name still reaches for the wide dial: on Android the deviceId " +
          "alone often leaves the main sensor's crop in place");
  }

  // ---------- the AR side measures, because it cannot choose ----------------
  {
    const r = await page.evaluate(() => {
      // The pinhole model calibration.json uses: focal 0.72 of the width.
      const W = 1280, H = 720, fx = 0.72 * W, fy = fx;
      const P = new Array(16).fill(0);
      P[0] = 2 * fx / W; P[5] = 2 * fy / H; P[10] = -1; P[14] = -0.2;
      const wide = arFovDeg(P);
      const tele = new Array(16).fill(0); tele[0] = 3; tele[5] = 3;
      const narrow = arFovDeg(tele);
      const expect = (f, px) => 2 * Math.atan(px / (2 * f)) * 180 / Math.PI;
      const before = AR.session;
      AR.session = {};
      const arLine = lensTabLine();
      AR.session = before;
      setLens("auto"); LENS.info = null;
      const noInfo = lensTabLine();
      return { wide, narrow, expectH: expect(fx, W), expectV: expect(fy, H),
               P0: P[0], P5: P[5], arLine, noInfo,
               nulls: [arFovDeg(null), arFovDeg(new Array(16).fill(0)), arFovDeg([NaN])] };
    });
    check(Math.abs(r.wide.h - r.expectH) < 0.01 && Math.abs(r.wide.v - r.expectV) < 0.01,
          `the FOV read off the XR matrix equals the pinhole model's own: ` +
          `${r.wide.h.toFixed(1)}x${r.wide.v.toFixed(1)} deg`);
    check(Math.abs(r.narrow.h - 2 * Math.atan(1 / 3) * 180 / Math.PI) < 0.01 && r.narrow.h < 40,
          `a telephoto reads narrow (${r.narrow.h.toFixed(0)} deg), which is the whole point of reporting it`);
    check(r.nulls.every(v => v === null),
          "a missing, all-zero or NaN matrix reports nothing rather than an angle of Infinity");
    check(/no lens dial/.test(r.arLine),
          `during an AR scan the row says the phone chose: "${r.arLine}"`);
    check(r.noInfo === "—", "before any camera has answered, the row is honest about not knowing");
  }

  // ---------- the facts reach the panel and the artifact --------------------
  {
    const r = await page.evaluate(() => {
      const keep = { info: LENS.info, proj: AR.lastProj, session: AR.session };
      // The pinhole focal length calibration.json assumes: 0.72 of the width.
      const W = 1280, H = 720, fx = 0.72 * W;
      const P = new Array(16).fill(0);
      P[0] = 2 * fx / W; P[5] = 2 * fx / H;
      LENS.info = { label: "camera2 2, facing back (ultra wide)", deviceId: "dev-uw",
                    zoomNow: 0.5, zoomRange: [0.5, 10], refused: "" };
      AR.lastProj = P; AR.session = null;
      const rows = phoneRows();
      const lens = rows.find(x => x[0] === "Lens");
      const fov = rows.find(x => x[0] === "Field of view");
      const out = { labels: rows.map(x => x[0]), lens: lens && lens[1], fov: fov && fov[1] };
      LENS.info = keep.info; AR.lastProj = keep.proj; AR.session = keep.session;
      return out;
    });
    check(r.labels.includes("Lens") && r.labels.includes("Field of view"),
          "the Phone tab carries both the lens and how much it sees");
    check(/Ultra wide/.test(r.lens) && /0\.50/.test(r.lens) && /0\.50–10\.00/.test(r.lens),
          `the Lens row names the lens, the zoom and the dial: "${r.lens}"`);
    check(/70° across × 43° up/.test(r.fov),
          `the FOV row reports the angle that matrix implies: "${r.fov}"`);
    const src = fs.readFileSync(path.join(ROOT, "viewer", "capture.html"), "utf8");
    check((src.match(/holdInfo, lensInfo/g) || []).length === 2,
          "calibration.json carries the lens on both the AR and the basic path");
    check(/arFovDeg\(AR\.lastProj\)/.test(src),
          "and the measured field of view travels with it");
  }

  // ---------- a lens the phone stopped offering must not cost the scan ------
  {
    const r = await page.evaluate(async (lenses) => {
      INSTALL({ devices: lenses, caps: { zoom: { min: 0.5, max: 10 } },
                apply: "ok", gumThrowOnce: true });
      // Keep the frame loop out of the way: this test is about opening a camera.
      visionLoopStarted = true; sheetStatsTimer = 1;
      setLens("dev-tele");
      LENS.fallback = "";
      // startCapture routes here whenever AR is not available, and is a plain
      // router that returns nothing, so the awaited call is the one that opens
      // the camera.
      await requestCameraAndSensors();
      return { mode: LENS.mode, fallback: LENS.fallback, deviceId: LENS.deviceId,
               gum: window.__calls.gum.length,
               note: document.getElementById("lens-note").textContent,
               opened: !!videoEl.srcObject };
    }, LENSES);
    check(r.gum === 2 && r.mode === "auto" && r.deviceId === null,
          "a deviceId the phone no longer offers falls back to the widest and reopens");
    check(r.opened, "and the scan still starts: a lens problem never gates a take");
    check(/not offered now/.test(r.fallback) && /not offered now/.test(r.note),
          `while saying what happened: "${r.note}"`);
    check(dialogs.length === 0, "no error alert was shown to get there", dialogs.join(" | "));
  }

  check(errs.length === 0, "no page error anywhere in the run", errs.join(" | "));

  await browser.close();
  fs.unlinkSync(path.join(ROOT, "scratch", "_lens_choice_test.html"));
  console.log(fail ? `\n${fail} FAILED, ${pass} passed` : `\nALL ${pass} LENS CHECKS PASSED`);
  process.exit(fail ? 1 : 0);
})();
