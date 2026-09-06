#!/usr/bin/env node
/*
 * test_combat_play.js — combat mode, driven in a real browser against a real scan.
 *
 * The nav math is covered by test_combat_nav.js. What cannot be checked there is
 * whether the whole thing behaves once it is standing on actual captured geometry:
 *
 *   1. combat boots with no page errors and its bots land on walkable ground;
 *   2. the weapon registers hits, with a headshot worth more than a body shot, and
 *      a tracer is drawn as a round down the barrel rather than a beam through the
 *      camera (box primitives pivot at their centre, so an uncapped one is a wedge
 *      that fills the screen);
 *   3. a bot behind real scan geometry cannot be shot, and is not drawn either —
 *      gaussian splats write no depth, so a bot that is not occlusion-tested
 *      would float visibly through the wall it is hiding behind;
 *   4. a bot that suddenly gains sight of the player does NOT fire instantly:
 *      the reaction delay is what makes an ambush read as deliberate;
 *   5. a bot with no fresh sighting scores its shot at zero, so it holds fire;
 *   6. a bot routed across the room actually walks, stays on the floor and never
 *      cuts through solid geometry;
 *   7. the plain viewer path pays for none of this — without ?combat=1 the
 *      gameplay modules are never even fetched, because the video-to-3D
 *      pipeline serves that page to operators on phones.
 *
 * Run: node tests/test_combat_play.js          (SKIPs if playwright-core is absent)
 * Env: COMBAT_SCENE=room_w_jsonl|temple|rocks  COMBAT_HEADED=1
 */
"use strict";
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const PORT = 8157;
const SCENE = process.env.COMBAT_SCENE || "room_w_jsonl";
const URL_BASE = `http://127.0.0.1:${PORT}/viewer/pc.html`;

let chromium = null;
try {
  ({ chromium } = require(path.join(ROOT, "scratch", "pw", "node_modules", "playwright-core")));
} catch {
  console.log("SKIP  playwright-core not found in scratch/pw — cannot drive a real GL context");
  process.exit(0);
}

let pass = 0, fail = 0;
function ok(cond, label, detail) {
  if (cond) { pass++; console.log("  ok   " + label); }
  else { fail++; console.log("  FAIL " + label + (detail ? "  -> " + detail : "")); }
}

function pythonExe() {
  const local = process.platform === "win32"
    ? path.join(ROOT, ".venv", "Scripts", "python.exe")
    : path.join(ROOT, ".venv", "bin", "python");
  if (fs.existsSync(local)) return local;
  return process.platform === "win32" ? "python" : "python3";
}

function startServer() {
  const log = fs.openSync(path.join(ROOT, "scratch", "serve_combat.log"), "w");
  const proc = spawn(pythonExe(), [path.join(ROOT, "_serve.py"), String(PORT), ROOT], {
    cwd: ROOT, stdio: ["ignore", log, log],
  });
  proc.on("error", (e) => { throw new Error(`_serve.py failed to start: ${e.message}`); });
  return new Promise((resolve, reject) => {
    const net = require("net");
    const t0 = Date.now();
    (function poll() {
      const sock = net.connect(PORT, "127.0.0.1");
      sock.on("connect", () => { sock.destroy(); resolve(proc); });
      sock.on("error", () => {
        sock.destroy();
        if (Date.now() - t0 > 15000) reject(new Error(`_serve.py did not bind on ${PORT} — see scratch/serve_combat.log`));
        else setTimeout(poll, 200);
      });
    })();
  });
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/** Poll a page expression until it is truthy, or throw with the viewer's own error. */
async function until(page, expr, what, timeout = 90000) {
  const t0 = Date.now();
  for (;;) {
    const err = await page.evaluate("window.__loadError || window.__combatError || null");
    if (err) throw new Error(`${what}: page reported ${err}`);
    if (await page.evaluate(expr)) return;
    if (Date.now() - t0 > timeout) throw new Error(`timed out waiting for ${what}`);
    await wait(250);
  }
}

(async () => {
  const server = await startServer();
  const browser = await chromium.launch({
    headless: !process.env.COMBAT_HEADED,
    args: ["--use-angle=default", "--enable-unsafe-swiftshader", "--disable-lcd-text", "--hide-scrollbars"],
  });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const page = await ctx.newPage();
  const pageErrors = [], consoleErrors = [], fetched = [];
  page.on("pageerror", (e) => pageErrors.push(String(e.message || e)));
  page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
  page.on("request", (r) => fetched.push(r.url()));

  try {
    console.log(`\ncombat mode on the "${SCENE}" scan`);
    await page.goto(`${URL_BASE}?asset=/work/${SCENE}/viewer_assets&combat=1&bots=3&cams=0`);
    await until(page, "window.__ready === true", "viewer ready");
    await until(page, "!!window.__combat", "combat install");
    await wait(1500);

    let snap = await page.evaluate("window.__combat.snapshot()");
    ok(pageErrors.length === 0, "no uncaught page errors", pageErrors.join(" | "));
    ok(snap.errors.length === 0, "combat reports no fatal problems", snap.errors.join(" | "));
    ok(snap.bots.length === 3, "three bots deployed", `${snap.bots.length}`);
    ok(snap.nav.tris > 100, "bots have walkable surface to stand on", `${snap.nav.tris} tris from ${snap.nav.source}`);
    ok(snap.spots.approachSamples > 0, "approach routes derived from the capture path", `${snap.spots.approachSamples}`);
    ok(snap.spots.scored > 0, "tactical nodes scored from the scan", `${snap.spots.scored}`);

    const land = await page.evaluate(`(() => {
      const c = window.__combat;
      return c.bots.map(b => {
        const g = c.groundAt(b.pos.x, b.pos.z, null);
        return { y: b.pos.y, ground: g, onNav: c.nav.triAt(b.pos.x, b.pos.z, 3) >= 0,
                 spot: b.spot ? { x: b.spot.x, z: b.spot.z, role: b.spotRole } : null };
      });
    })()`);
    ok(land.every((l) => l.onNav), "every bot starts on walkable ground");
    ok(land.every((l) => l.ground === null || Math.abs(l.y - l.ground) < 0.35),
       "bots stand on the captured floor rather than floating", JSON.stringify(land.map((l) => +(l.y - l.ground).toFixed(2))));
    ok(land.some((l) => l.spot), "bots are posted at scored nodes, not dropped at random");
    const reach = await page.evaluate(`(() => {
      const c = window.__combat, p = c.center();
      return c.bots.map(b => !!c.nav.findPath(p.x, p.z, b.pos.x, b.pos.z, { canWalk: (a, z) => c.walkable(a, z) }));
    })()`);
    ok(reach.every(Boolean), "every bot sits in the walkable region the player can actually reach", JSON.stringify(reach));
    ok(snap.nav.tris <= snap.nav.rawTris, "stranded bake islands are discarded before use",
       `${snap.nav.tris} kept of ${snap.nav.rawTris} baked`);

    console.log("\nshooting");
    const hitTest = await page.evaluate(`(() => {
      const c = window.__combat, cam = c.opts.camera;
      // The scan decides what a player can see, and this spawn looks at a wall,
      // so sweep for a direction with open space before putting a target down.
      const o0 = cam.getPosition();
      let swept = false;
      for (let i = 0; i < 24 && !swept; i++) {
        cam.setEulerAngles(-3, i * 15, 0);
        const f = cam.forward;
        const probe = c.sight.cast(o0.x, o0.y, o0.z, f.x, f.y, f.z, 6);
        if (!probe || probe.distance > 5.4) swept = true;
      }
      const origin = cam.getPosition();
      const fwd = cam.forward;
      const b = c.bots[0];
      const bx = origin.x + fwd.x * 4, bz = origin.z + fwd.z * 4;
      const by = c.groundAt(bx, bz, origin.y - 1.5);
      b.pos = { x: bx, y: by, z: bz };
      b.alive = true; b.health = 100; b.visible = true;
      for (const r of b.renderers) r.enabled = true;
      b.updateVisibility(origin);
      const aimAt = (point) => {
        const d = { x: point.x - origin.x, y: point.y - origin.y, z: point.z - origin.z };
        const l = Math.hypot(d.x, d.y, d.z);
        return { dir: { x: d.x / l, y: d.y / l, z: d.z / l }, dist: l };
      };
      const chestShot = b.hitTest(origin, aimAt(b.chest).dir, aimAt(b.chest).dist + 1);
      const headShot = b.hitTest(origin, aimAt({ x: b.pos.x, y: b.pos.y + 1.6, z: b.pos.z }).dir, 8);
      // aim the live camera down the shot line, the same call pc.js makes each frame
      const chest = b.chest;
      cam.lookAt(chest.x, chest.y, chest.z);
      const wall = c.sight.cast(origin.x, origin.y, origin.z, cam.forward.x, cam.forward.y, cam.forward.z, 4.5);
      const blocking = wall && wall.distance < 3.6;
      const before = b.health;
      const fired = b.visible && !blocking ? c.weapon.fire(c.t, c.bots) : 0;
      return { chest: chestShot && chestShot.part, head: headShot && headShot.part, fired,
               before, after: b.health, ammo: c.weapon.ammo, shots: c.weapon.shots,
               swept, clear: wall ? +wall.distance.toFixed(2) : 99,
               blockedBy: blocking ? +wall.distance.toFixed(2) : null, visible: b.visible };
    })()`);
    ok(hitTest.chest === "chest", "a bot on the aim line is inside the shot cone", `${hitTest.chest}`);
    ok(hitTest.head === "head", "the head is a separate, aim-able zone", `${hitTest.head}`);
    if (hitTest.blockedBy !== null) {
      ok(false, "the aim sweep finds a direction with open space to shoot down",
         `every yaw is walled; nearest surface ${hitTest.clear} m`);
    } else {
      ok(hitTest.fired === 1, "the weapon fires");
      ok(hitTest.after < hitTest.before, "shooting a bot costs it health", `${hitTest.before} -> ${hitTest.after}`);
      ok(hitTest.ammo === 29 && hitTest.shots === 1, "the magazine and shot counter track", JSON.stringify(hitTest));
    }
    console.log("\ntracer geometry");
    const tr = await page.evaluate(`(() => {
      const c = window.__combat, cam = c.opts.camera;
      const o = cam.getPosition(), f = cam.forward;
      const far = { x: o.x + f.x * 80, y: o.y + f.y * 80, z: o.z + f.z * 80 };
      const full = c.effects.tracer(o, far);
      const slot = c.effects.live[c.effects.live.length - 1];
      const p = slot.ent.getPosition(), s = slot.ent.getLocalScale();
      return { drawn: +s.z.toFixed(2), cap: c.effects.tracerRange, full: +full.toFixed(1),
               mid: +Math.hypot(p.x - o.x, p.y - o.y, p.z - o.z).toFixed(2),
               ahead: +((p.x - o.x) * f.x + (p.y - o.y) * f.y + (p.z - o.z) * f.z).toFixed(2) };
    })()`);
    ok(tr.drawn <= tr.cap + 0.01 && tr.full > tr.cap * 3,
       "a round down a long range fades instead of painting a laser sight",
       `${tr.drawn} m drawn of ${tr.full} m fired`);
    ok(tr.mid > tr.drawn * 0.4 && tr.mid < tr.drawn * 0.6,
       "the beam box is centred on its segment, not on the muzzle", `${tr.mid} m out on a ${tr.drawn} m box`);
    ok(tr.ahead > 0, "and it sits in front of the eye rather than through the camera", `${tr.ahead} m ahead`);
    const dmg = await page.evaluate(`(() => {
      const c = window.__combat;
      const probe = (part) => {
        const b = c.bots[1];
        b.health = 100;
        const before = b.health;
        b.damage(part === 'head' ? 22 * 3.2 : 22, part);
        return before - b.health;
      };
      return { head: probe('head'), chest: probe('chest') };
    })()`);
    ok(dmg.head > dmg.chest, "a headshot hurts more than centre mass", `${dmg.head} vs ${dmg.chest}`);

    console.log("\nocclusion by the scan itself");
    const occ = await page.evaluate(`(() => {
      const c = window.__combat, cam = c.opts.camera;
      const p = c.center();
      let wall = null;
      for (let i = 0; i < 36; i++) {
        const a = (i / 36) * Math.PI * 2;
        const h = c.sight.cast(p.x, p.y + 0.2, p.z, Math.cos(a), 0, Math.sin(a), 25);
        if (h && h.distance > 1.5 && Math.abs(h.normal.y) < 0.4 && (!wall || h.distance < wall.distance)) {
          wall = { distance: h.distance, x: h.point.x, y: h.point.y, z: h.point.z };
        }
      }
      if (!wall) return { none: true };
      const b = c.bots[2];
      const dx = wall.x - p.x, dz = wall.z - p.z, l = Math.hypot(dx, dz);
      b.alive = true; b.visible = true;
      const bx = wall.x + (dx / l) * 3, bz = wall.z + (dz / l) * 3;
      b.pos = { x: bx, y: c.groundAt(bx, bz, null) ?? wall.y, z: bz };
      b.updateVisibility(c.opts.camera.getPosition());
      const o = { x: p.x, y: p.y + 0.1, z: p.z };
      const dir = { x: (b.pos.x - o.x), y: (b.pos.y + 1.2 - o.y), z: (b.pos.z - o.z) };
      const dl = Math.hypot(dir.x, dir.y, dir.z);
      const shot = b.hitTest(o, { x: dir.x / dl, y: dir.y / dl, z: dir.z / dl }, dl);
      return { none: false, dist: wall.distance, visible: b.visible, renderersOn: b.renderers.filter(r => r.enabled).length, shot };
    })()`);
    if (occ.none) {
      ok(false, "found scan geometry to hide a bot behind", "no wall within 25 m of the player");
    } else {
      ok(occ.visible === false, "a bot behind the wall is culled, so it does not show through the scan",
         `wall at ${occ.dist.toFixed(1)} m`);
      ok(occ.renderersOn === 0, "its meshes are actually switched off", `${occ.renderersOn} still on`);
      ok(occ.shot === null, "and it cannot be shot through that wall");
    }

    console.log("\nreaction and fire discipline");
    const discipline = await page.evaluate(`(() => {
      const c = window.__combat;
      const p = c.center();
      const yaw = c.opts.getYaw();
      const b = c.bots[0];
      // Squad gunfire kills this stationary test player mid-window, and a respawn
      // resets every bot to HOLD -- which would wipe the state being probed. So the
      // roster is narrowed to the one bot whose fire gate is under test.
      c.health = c.healthMax; c.dead = 0; c.pain = 0;
      window.__squad = c.bots;
      c.bots = [b];
      b.alive = true; b.health = 100; b.stats = { shots: 0, hits: 0 };
      b.state = 'ENGAGE'; b.awareness = 1; b.lastSeen = -999; b.seenMs = 0; b.nextShot = 0; b.burstGap = 0;
      b.pos = { x: p.x - Math.sin(yaw) * 5, y: p.y - 0.6, z: p.z - Math.cos(yaw) * 5 };
      b.yaw = Math.atan2(-(p.x - b.pos.x), -(p.z - b.pos.z));
      const stale = b.shotUtility(b.ctx.now(), c.playerState);
      return { staleUtility: stale, placed: true, sees: b.canSee(c.playerState, b.eye).see };
    })()`);
    ok(discipline.staleUtility === 0, "a bot that has not seen the player recently scores no shot", `${discipline.staleUtility}`);
    // Combat timers run off the simulation clock, so this window is stepped by hand.
    // It has to be: headless Chromium stops issuing frames a couple of seconds into
    // this page, so a real-time wait would measure nothing at all. The bot is staged
    // on ground with a verified clear line to the player and only that geometry is
    // held -- the reaction delay, wind-up and utility gate below are all the bot's.
    const stage = (n) => `(() => {
      const c = window.__combat, b = c.bots[0];
      const stand = () => {
        const p = c.center();
        for (const r of [4.5, 5.5, 6.5]) {
          for (let i = 0; i < 16; i++) {
            const a = (i / 16) * Math.PI * 2;
            const x = p.x + Math.cos(a) * r, z = p.z + Math.sin(a) * r;
            const y = c.groundAt(x, z, p.y - 0.6);
            if (y === null) continue;
            b.pos = { x, y, z };
            if (b.canSee(c.playerState, b.eye).see) return true;
          }
        }
        return false;
      };
      const stood = b.canSee(c.playerState, b.eye).see || stand();
      for (let i = 0; i < ${n}; i++) {
        if (!b.canSee(c.playerState, b.eye).see) stand();
        const p = c.center();
        b.yaw = Math.atan2(-(p.x - b.pos.x), -(p.z - b.pos.z));
        c.update(1 / 60);
      }
      const s = b.snapshot();
      return { shots: b.stats.shots, stood, seenMs: s.seenMs, u: s.utility, state: s.state };
    })()`;
    const first = await page.evaluate("window.__combat.bots[0].stats.shots");
    const opened = await page.evaluate(stage(4));
    const wound = await page.evaluate(stage(70));
    const later = wound.shots;
    const stood = opened.stood;
    const playerAlive = await page.evaluate("window.__combat.health > 0");
    const why = `state=${wound.state} u=${wound.u} seenMs=${wound.seenMs}`;
    ok(first === 0, "the reseeded bot has not fired yet");
    ok(stood, "the harness found ground with a clear line to the player", "no clear stand within 6.5 m");
    ok(opened.shots === 0, "it still has not fired within ~0.07 s of being seen",
       `${opened.shots} shots, u=${opened.u}, seenMs=${opened.seenMs}`);
    ok(later > 0, "it opens up once its aim has wound up",
       `${later} shots, sees: ${discipline.sees}, player standing: ${playerAlive} :: ${why}`);
    const held = await page.evaluate(`(() => {
      const c = window.__combat, b = window.__squad[1];
      b.lastSeen = b.ctx.now() - 30; b.seenMs = 0;
      return b.shotUtility(b.ctx.now(), c.playerState);
    })()`);
    ok(held === 0, "an out-of-date sighting never justifies a shot", `${held}`);
    await page.evaluate("(() => { const c = window.__combat; c.bots = window.__squad; delete window.__squad; })()");

    console.log("\nmovement across the scan");
    const move = await page.evaluate(`(() => {
      const c = window.__combat;
      const b = c.bots[1];
      // the firing block leaves the player shot up, and a respawn re-posts every bot
      c.health = c.healthMax; c.dead = 0; c.pain = 0;
      b.alive = true; b.health = 100; b.state = 'SEARCH';
      const spots = c.spots.any.filter((s) => {
        const d = Math.hypot(s.x - b.pos.x, s.z - b.pos.z);
        return d > 6 && d < 10 && s.tri !== b.spot?.tri;
      });
      const target = spots[Math.floor(spots.length / 2)] || spots[0];
      if (!target) return { skipped: true };
      b.ctx.viewRange = 0.01;
      b.lastSeen = -9999;
      b.sinceOwnShot = 0;
      b.noise = null;
      b.awareness = 0;
      b.lastKnown = null;
      window.__roster = c.bots;
      window.__probeBot = b;
      c.bots = [b];
      b.routeTo(target.x, target.z);
      b.__shifts = 0;
      if (!b.__spy) {
        const origMove = Object.getPrototypeOf(b).move;
        b.move = function (dt) {
          const n = this.path.length;
          origMove.call(this, dt);
          b.__shifts += Math.max(0, n - this.path.length);
        };
        b.__spy = true;
      }
      const end = b.path[b.path.length - 1] || null;
      return { from: { ...b.pos }, hops: b.path.length, tx: target.x, tz: target.z,
               end: end ? { x: end.x, z: end.z } : null,
               start: Math.hypot(target.x - b.pos.x, target.z - b.pos.z) };
    })()`);
    if (!move.skipped) {
      ok(move.hops > 0, "a cross-room route is planned", `${move.hops} waypoints`);
      ok(move.end && Math.hypot(move.tx - move.end.x, move.tz - move.end.z) < 1,
         "and the route ends at the node it was asked for",
         move.end ? `${Math.hypot(move.tx - move.end.x, move.tz - move.end.z).toFixed(1)} m short` : "no waypoints");
      let crossed = 0, offFloor = 0, minLeft = Infinity, shifts = 0, walked = 0;
      const samples = [];
      let prev = null;
      for (let i = 0; i < 260; i++) {
        const r = await page.evaluate(`(() => {
          const c = window.__combat, b = window.__probeBot;
          const before = { x: b.pos.x, z: b.pos.z };
          b.sinceOwnShot = 0;
          // A bot re-posts itself when its mind changes its mind, which is correct
          // behaviour but not what this probe measures: keep pointing it at the node
          // under test. Each re-issue runs the real A*, string-pull and legality gate.
          const end = b.path[b.path.length - 1];
          if (!end || Math.hypot(end.x - ${move.tx}, end.z - ${move.tz}) > 1) b.routeTo(${move.tx}, ${move.tz});
          for (let k = 0; k < 4; k++) c.update(1 / 60);
          const g = c.groundAt(b.pos.x, b.pos.z, null);
          const w = b.path[0] || null;
          return { x: b.pos.x, y: b.pos.y, z: b.pos.z, ground: g, hops: b.path.length,
                   toWp: w ? +Math.hypot(w.x - b.pos.x, w.z - b.pos.z).toFixed(1) : null,
                   shifts: b.__shifts,
                   st: b.state,
                   blocked: c.sight.blocked(before.x, before.y + 1.1, before.z, b.pos.x, b.pos.y + 1.1, b.pos.z) };
        })()`);
        if (r.blocked) crossed++;
        if (r.ground !== null && Math.abs(r.y - r.ground) > 0.4) offFloor++;
        if (prev) walked += Math.hypot(r.x - prev.x, r.z - prev.z);
        prev = r;
        r.d = +Math.hypot(move.tx - r.x, move.tz - r.z).toFixed(1);
        minLeft = Math.min(minLeft, r.d);
        shifts = Math.max(shifts, r.shifts);
        samples.push(r);
        if (shifts >= 1 && minLeft < 1) break;
      }
      const simSecs = (samples.length * 4 / 60).toFixed(1);
      const trail = samples.filter((_, i) => i % 25 === 0)
        .map((s) => `${s.st}/${s.hops}w wp:${s.toWp}m @${s.d}m`).join(" ");
      ok(walked > 4, "the bot actually walks across the scan", `${walked.toFixed(1)} m of travel`);
      ok(minLeft < 2.5 && minLeft < move.start,
         "and it arrives at the node it was routed to",
         `${minLeft.toFixed(1)} m of the target, from ${move.start.toFixed(1)} m in ${simSecs} s :: ${trail}`);
      ok(shifts >= 1, "waypoints are consumed as it reaches them",
         `${shifts} of ${move.hops} reached :: ${trail}`);
      ok(crossed === 0, "it never steps through solid geometry", `${crossed} frames inside a wall`);
      ok(offFloor < samples.length * 0.2, "it stays planted on the captured floor", `${offFloor}/${samples.length}`);
      await page.evaluate("(() => { const c = window.__combat; if (window.__roster) { c.bots = window.__roster; } delete window.__roster; delete window.__probeBot; })()");
    } else {
      ok(false, "no route target available to test movement");
    }

    await page.screenshot({ path: path.join(ROOT, "scratch", "combat_boot.png") });
    snap = await page.evaluate("window.__combat.snapshot()");
    const hudOk = await page.evaluate(`(() => {
      const hud = document.querySelector('.cg-hud');
      return { present: !!hud, cross: !!document.querySelector('.cg-cross'),
               ammo: document.querySelector('.cg-ammo')?.textContent?.trim() ?? '',
               help: document.querySelector('.cg-help')?.textContent?.trim().slice(0, 24) ?? '' };
    })()`);
    ok(hudOk.present && hudOk.cross, "the combat HUD is on screen");
    ok(/^\d+\s*\/\s*\d+$/.test(hudOk.ammo.replace(/\/\s*$/, "").trim()) || hudOk.ammo.includes("/"),
       "the HUD mirrors real weapon state", `"${hudOk.ammo}"`);
    ok(snap.weapon.shots > 0 && snap.weapon.ammo < 30, "HUD ammo tracks the shots actually fired", JSON.stringify(snap.weapon));

    console.log("\nthe diagnostics surface");
    await page.keyboard.press("KeyH");
    await page.evaluate("(() => { const c = window.__combat; for (let i = 0; i < 6; i++) c.update(1 / 60); })()");
    const dbg = await page.evaluate(`(() => {
      const c = window.__combat, d = document.querySelector(".cg-dbg");
      const rows = [...(d ? d.querySelectorAll("tbody tr") : [])].map((tr) => [...tr.children].map((td) => td.textContent));
      const snap = c.snapshot();
      return { shown: !!d && !d.hidden, rows: rows.length, live: snap.bots.length,
               row: rows[0] ? rows[0].join(" ") : "", state: snap.bots[0] ? snap.bots[0].state : "" };
    })()`);
    ok(dbg.shown, "H opens the bot-internals table");
    ok(dbg.rows > 0 && dbg.rows === dbg.live, "one row per live bot", `${dbg.rows} rows for ${dbg.live} bots`);
    ok(dbg.state && dbg.row.includes(dbg.state), "and the row says what the sim really thinks", dbg.row);
    const probe = await page.evaluate(`(() => {
      const c = window.__combat;
      const text = JSON.stringify(c.hud.probe());
      document.querySelector('.cg-dbg [data-act="copy"]').click();
      const parsed = JSON.parse(text);
      return { bots: parsed.bots.length, nav: !!parsed.nav, weapon: !!parsed.weapon };
    })()`);
    await page.evaluate("(() => { const c = window.__combat; for (let i = 0; i < 3; i++) c.update(1 / 60); })()");
    const toast = await page.evaluate("(() => { const s = document.querySelector('.cg-status'); return s ? s.textContent : ''; })()");
    ok(probe.bots > 0 && probe.nav && probe.weapon,
       "the state probe copies the whole world, not a summary", `${probe.bots} bots, nav ${probe.nav}, weapon ${probe.weapon}`);
    ok(/copied|blocked/i.test(toast), "and it says whether the clipboard allowed it", JSON.stringify(toast));
    await page.keyboard.press("KeyH");

    console.log("\nthe plain viewer stays untouched");
    const page2 = await ctx.newPage();
    const fetched2 = [];
    page2.on("request", (r) => fetched2.push(r.url()));
    const errors2 = [];
    page2.on("pageerror", (e) => errors2.push(String(e.message || e)));
    await page2.goto(`${URL_BASE}?asset=/work/${SCENE}/viewer_assets&cams=0`);
    await until(page2, "window.__ready === true", "plain viewer ready");
    await wait(1200);
    const plain = await page2.evaluate(`(() => ({
      active: !!window.__combatActive, combat: !!window.__combat,
      hud: !!document.querySelector('.cg-hud'),
      walked: window.__walk ? window.__walk.walked : null,
      pos: window.__playerPos ? window.__playerPos() : null,
    }))()`);
    ok(!plain.active && !plain.combat, "combat never installs without ?combat=1");
    ok(!plain.hud, "no combat HUD on the operator page");
    ok(errors2.length === 0, "the plain viewer still runs clean", errors2.join(" | "));
    ok(plain.pos !== null, "the walk controller still answers the automation hook", JSON.stringify(plain));
    const leaked = fetched2.filter((u) => /scripts\/(combat|bot|weapon|nav|hud|audio|world)\.js/.test(u));
    ok(leaked.length === 0, "the gameplay modules are not even fetched for the plain viewer", leaked.join(" "));

    const strayErrors = [...pageErrors, ...consoleErrors]
      .filter((e) => !/favicon|Autoplay|AudioContext|Failed to load resource/i.test(e));
    ok(strayErrors.length === 0, "no console errors across either page", strayErrors.slice(0, 3).join(" | "));
  } catch (e) {
    fail++;
    console.log("  FAIL  harness: " + (e && e.stack ? e.stack : e));
    try { await page.screenshot({ path: path.join(ROOT, "scratch", "combat_failure.png") }); } catch { /* page gone */ }
  } finally {
    await browser.close();
    server.kill();
  }

  console.log("");
  console.log(fail === 0 ? `ALL ${pass} COMBAT-PLAY CHECKS PASSED` : `${pass} passed, ${fail} FAILED`);
  process.exit(fail ? 1 : 0);
})();
