#!/usr/bin/env node
"use strict";
// Mission rehearsal logic (viewer/pc/scripts/mission.js): scenario validation, bot
// behaviour settings, waypoint tracking, exposure, recording and conditions.
const assert = require("node:assert/strict");
const { test, before } = require("node:test");
const { readFileSync } = require("node:fs");
const path = require("node:path");

let M = {};
before(async () => {
  const source = readFileSync(path.join(__dirname, "../viewer/pc/scripts/mission.js"), "utf8");
  M = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
});

const north = [0, -1];
const route = { waypoints: [{ name: "SP", at: [0, 0, 0] }, { name: "WP2", at: [0, 0, -30] }, { name: "OBJ", at: [30, 0, -30] }] };
const bot = (extra = {}) => ({ position: [0, 0, 0], facing_xz: [0, -1], behaviour: "sentry", range_m: 100, sector_deg: 90, count: 1, patrol: null, ...extra });

test("scenarios are validated", () => {
  assert.ok(M.validateScenario({ bots: [bot()], route, north_xz: north }));
  for (const bad of [{ bots: [bot({ behaviour: "berserk" })], north_xz: north }, { bots: [bot({ count: 9 })], north_xz: north },
    { bots: [], route: { waypoints: [route.waypoints[0]] }, north_xz: north }, { bots: [] }]) {
    assert.throws(() => M.validateScenario(bad));
  }
});

test("yaw and bearings follow the combat and compass conventions", () => {
  assert.equal(M.yawOf([0, -1]), -0);                     // facing north (-z) is yaw 0
  assert.ok(Math.abs(M.yawOf([-1, 0]) - Math.PI / 2) < 1e-12); // west is +90 deg in the combat yaw
  assert.equal(M.bearingDeg([0, 0, 0], [10, 0, 0], north), 90);
  assert.equal(M.bearingDeg([0, 0, 0], [0, 0, 10], north), 180);
  assert.ok(Math.abs(M.yawBearing(-Math.PI / 2, north) - 90) < 1e-9, "yaw -90 deg faces east");
});

test("behaviours: sentries hold and scan, overwatch sees far and narrow, night and fog shorten sight", () => {
  const s = M.botSettings(bot());
  assert.equal(s.stayPut, true);
  assert.ok(Math.abs(s.scanAmp - (45 - 15) * Math.PI / 180) < 1e-12);
  assert.ok(Math.abs(M.scanYaw(s, 9 / 4) - (s.baseYaw + s.scanAmp)) < 1e-12);
  const o = M.botSettings(bot({ behaviour: "overwatch", sector_deg: 120 }));
  assert.equal(o.viewRange, 125);
  assert.ok(Math.abs(o.viewCone - 70 * Math.PI / 180) < 1e-12);
  assert.equal(M.botSettings(bot(), { light: "night" }).viewRange, 50);
  const sniper = M.botSettings(bot({ role: "sniper", range_m: 300 }));
  assert.equal(sniper.engageRange, 300, "a sniper is willing to shoot at its planned range");
  assert.equal(sniper.roundRange, 300);
  assert.equal(M.botSettings(bot()).engageRange, 30, "an ordinary post fights like an arena bot");
  assert.equal(M.botSettings(bot(), { fog_m: 40 }).viewRange, 40);
  const p = M.botSettings(bot({ behaviour: "patrol", patrol: [[0, 0, 0], [10, 0, 0]] }));
  assert.equal(p.stayPut, false);
  assert.equal(p.patrol.length, 2);
  assert.equal(M.botSettings(bot({ behaviour: "patrol" })).patrol, null, "a patrol with no route holds");
  assert.deepEqual(M.postOffsets(1), [[0, 0]]);
  assert.equal(M.postOffsets(3).length, 3);
});

test("fire profiles: a sniper fires single aimed rounds, a machine gun long bursts", () => {
  const sniper = M.botSettings(bot({ role: "sniper", range_m: 300 })).fire;
  assert.equal(sniper.burstMax, 1);
  assert.ok(sniper.betweenS[0] >= 2.5 && sniper.aimScale < 1);
  const mg = M.botSettings(bot({ role: "machine_gun" })).fire;
  assert.ok(mg.burstMin >= 4 && mg.withinS < 0.11);
  assert.equal(M.botSettings(bot()).fire.betweenS, null, "a rifleman keeps the arena's burst rhythm");
});

test("session commands apply once, in order, and ignore junk", () => {
  const cmds = [{ seq: 1, type: "move_bot", bot: 2, to: [1, 2, 3] }, { seq: 2, type: "message", text: "Go" },
    { seq: 3, type: "launch" }, { seq: 4, type: "end" }];
  const first = M.commandActions(cmds, 0);
  assert.deepEqual(first.actions.map((a) => a.kind), ["move", "message", "end"]);
  assert.equal(first.seq, 4);
  assert.equal(M.commandActions(cmds, 4).actions.length, 0, "already applied commands are skipped");
});

test("fog is drawn: opaque at the stated visibility, darker at night, none without fog", () => {
  const f = M.fogFor({ fog_m: 80, light: "night" });
  assert.equal(f.end, 80);
  assert.equal(f.start, 28);
  assert.ok(f.color[0] < M.fogFor({ fog_m: 80, light: "day" }).color[0]);
  assert.equal(M.fogFor({}), null);
});

test("the waypoint tracker arrives, skips ahead, warns off-route and completes", () => {
  const tr = new M.WaypointTracker(route, { speed: 1.5, north });
  let s = tr.update([0, 0, -10], 5);
  assert.equal(s.next.name, "WP2");
  assert.equal(Math.round(s.next.distance), 20);
  assert.equal(s.next.bearing, 0);
  assert.equal(Math.round(s.remainingM), 50);
  assert.ok(Math.abs(s.etaS - 50 / 1.5) < 1e-9);
  s = tr.update([20, 0, 0], 8);
  assert.deepEqual(s.events.map(e => e.type), ["off_route"]);
  s = tr.update([1, 0, -29], 20);
  assert.deepEqual(s.events.map(e => e.type).sort(), ["on_route", "waypoint"]);
  s = tr.update([29, 0, -30], 40);
  assert.deepEqual(s.events.map(e => e.type), ["waypoint", "complete"]);
  assert.equal(s.done, true);
  const skip = new M.WaypointTracker(route);
  assert.deepEqual(skip.update([30, 0, -31], 1).events.map(e => e.text), ["WP2", "OBJ", "OBJ reached"]);
});

test("phase lines, exposure and the recorder", () => {
  assert.equal(M.crossesLine([0, 0, 0], [0, 0, -10], [[-5, 0, -5], [5, 0, -5]]), true);
  assert.equal(M.crossesLine([0, 0, 0], [0, 0, -4], [[-5, 0, -5], [5, 0, -5]]), false);
  const ex = new M.ExposureMeter();
  const first = ex.update(0.5, 1, [3], new Map([[3, "Sentry"]]));
  assert.deepEqual(first.map(e => e.text), ["Seen by Sentry"]);
  assert.deepEqual(ex.update(0.5, 1.5, [3, 4]).map(e => e.bot), [4]);
  ex.update(0.5, 2, []);
  assert.equal(ex.total, 1);
  assert.equal(ex.perBot.get(3), 1);
  const rec = new M.Recorder(10);
  for (let i = 0; i <= 100; i++) {
    const t = i * 0.05;
    rec.sample(t, { x: t, y: 0, z: 0, yaw: 0, health: 100 }, [{ id: 0, x: 1, y: 0, z: 1, yaw: 0, sees: i > 50, alive: true }]);
  }
  assert.equal(rec.player.length, 51, "5 s at 10 Hz from 20 Hz input");
  const body = rec.toRun({ route: [[0, 0]] });
  assert.equal(body.hz, 10);
  assert.equal(body.bots[0].frames[0].length, 7);
  const small = rec.toRun({}, 1000);
  assert.ok(small.hz < 10, "an oversized run is thinned before upload");
  assert.ok(JSON.stringify(small).length <= 1000);
});

test("conditions map to a canvas filter", () => {
  assert.equal(M.conditionFilter({ light: "day" }), "");
  assert.match(M.conditionFilter({ light: "night" }), /brightness\(0\.28\)/);
  assert.match(M.conditionFilter({ light: "night", nvg: true }), /hue-rotate/);
});
