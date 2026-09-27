/* Mission rehearsal on top of the combat arena (RH-1..RH-6, RH-10).
 *
 * The combat manager keeps doing what it does - bots see, hear, shoot, the player walks and
 * fires - but the enemies stand where the mission plan put them, with the plan's facing,
 * sector, range and behaviour, and the player follows the planned route with a waypoint
 * HUD, an exposure meter and a recording that becomes the after-action review.
 */
import { Color, Entity, StandardMaterial, Vec3, BLEND_NORMAL, FOG_LINEAR } from "playcanvas";
import {
  botSettings, conditionFilter, crossesLine, ExposureMeter, postOffsets, Recorder, validateScenario, WaypointTracker, yawBearing, fogFor,
} from "./mission.js";

const CSS = `
.rh-top { position: absolute; left: 50%; top: 58px; transform: translateX(-50%); display: flex; flex-direction: column; align-items: center; gap: 6px; }
.rh-tape { position: relative; width: 420px; height: 30px; overflow: hidden; background: var(--cg-panel); border: 1px solid var(--cg-line); border-radius: 8px; }
.rh-tape .ticks { position: absolute; top: 0; height: 100%; white-space: nowrap; }
.rh-tape .ticks span { display: inline-block; width: 30px; text-align: center; font: 600 10px system-ui; color: var(--cg-dim); line-height: 30px; }
.rh-tape .ticks span.card { color: var(--cg-ink); }
.rh-tape .wp { position: absolute; top: 2px; width: 0; height: 0; border-left: 7px solid transparent; border-right: 7px solid transparent; border-top: 10px solid #ffb020; transform: translateX(-7px); }
.rh-tape .mid { position: absolute; left: 50%; top: 0; bottom: 0; width: 2px; background: #fff; transform: translateX(-1px); }
.rh-wp { background: var(--cg-panel); border: 1px solid var(--cg-line); border-radius: 8px; padding: 5px 12px; font-variant-numeric: tabular-nums; }
.rh-wp b { color: #ffb020; }
.rh-off { background: rgba(200, 120, 20, .92); color: #fff; border-radius: 6px; padding: 3px 10px; font-weight: 700; }
.rh-exp { position: absolute; right: 16px; top: 14px; background: var(--cg-panel); border: 1px solid var(--cg-line); border-radius: 10px; padding: 8px 12px; min-width: 170px; }
.rh-exp .seen { font-weight: 800; font-size: 15px; }
.rh-exp .seen.hot { color: var(--cg-bad); } .rh-exp .seen.cold { color: var(--cg-good); }
.rh-label { position: absolute; left: 50%; bottom: 8px; transform: translateX(-50%); font-size: 10px; color: rgba(255,255,255,.72); background: rgba(0,0,0,.45); padding: 3px 9px; border-radius: 5px; white-space: nowrap; }
.rh-end { position: absolute; left: 50%; top: 40%; transform: translate(-50%, -50%); background: var(--cg-panel); border: 1px solid var(--cg-line); border-radius: 12px; padding: 16px 20px; text-align: center; min-width: 280px; pointer-events: auto; }
.rh-end h3 { margin: 0 0 6px; font-size: 15px; }
.rh-nvg { position: fixed; inset: 0; pointer-events: none; z-index: 19; background: radial-gradient(circle at center, transparent 48%, rgba(0,0,0,.92) 72%); }
`;

function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
}
const fmtTime = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export class Rehearsal {
  /** ``combat`` is the arena manager; ``scenario`` the validated plan; ``options`` the run conditions. */
  constructor(combat, scenario, { conditions = {}, saveUrl = null, parent = null, origin = null, session = null } = {}) {
    this.combat = combat;
    this.engine = combat.opts.engine || combat.app;
    this.s = validateScenario(scenario);
    this.conditions = conditions;
    this.saveUrl = saveUrl;
    this.sessionInfo = session;
    this.parent = parent;
    this.origin = origin;
    this.north = scenario.north_xz;
    this.recorder = new Recorder(10);
    this.exposure = new ExposureMeter();
    this.tracker = scenario.route ? new WaypointTracker(scenario.route, { north: this.north, speed: 1.4 }) : null;
    this.lastPos = null;
    this.ended = false;
    this.lastState = null;
    this.labels = new Map();
    this.beacons = [];
    this.wasDown = false;
  }

  start() {
    const c = this.combat;
    this._css();
    this._hud();
    this._conditions();
    this._beacons();
    // The player starts at the route's start point, facing its first leg.
    this.respawnPoint = this.s.route ? 0 : null;
    c.opts.respawn = () => this._respawn();
    this._respawn();
    for (const plan of this.s.bots) {
      const settings = botSettings(plan, this.conditions);
      postOffsets(plan.count).forEach(([ox, oz], k) => {
        const [x, y, z] = plan.position;
        const bot = c.spawnPlannedBot({ x: x + ox, y, z: z + oz, yaw: settings.baseYaw }, settings, `${plan.label}${plan.count > 1 ? ` ${k + 1}` : ""}`);
        bot.plan = plan;
        this.labels.set(bot.id, bot.label);
      });
    }
    this.recorder.event({ t: 0, type: "start", text: `${this.s.mission.name}: ${this.s.bots.reduce((n, b) => n + b.count, 0)} planned enemies` });
    c.hud.feed(`Rehearsal: ${this.s.mission.name}`, "warn");
    window.addEventListener("message", (e) => {
      if (e.source !== this.parent || e.origin !== this.origin) return;
      if (e.data?.namespace === "groundcontrol" && e.data.type === "command" && e.data.command === "end-rehearsal") this.end("ended by the user");
    });
  }

  _respawn() {
    const r = this.s.route;
    if (r) {
      const k = Math.max(0, (this.tracker?.index ?? 1) - 1);
      const a = r.waypoints[Math.min(k, r.waypoints.length - 1)].at;
      const b = r.waypoints[Math.min(k + 1, r.waypoints.length - 1)].at;
      window.__chosenSpawn = { x: a[0], z: a[2], face_xz: [b[0], b[2]] };
    }
    this.combat.opts.spawnPlayer?.();
  }

  _css() {
    if (document.getElementById("rh-style")) return;
    const style = el("style"); style.id = "rh-style"; style.textContent = CSS;
    document.head.appendChild(style);
  }

  _hud() {
    const root = this.combat.hud.root;
    this.top = el("div", "rh-top");
    this.tape = el("div", "rh-tape", '<div class="ticks"></div><div class="wp"></div><div class="mid"></div>');
    const ticks = [];
    for (let rep = 0; rep < 3; rep++) for (let d = 0; d < 360; d += 15) {
      const card = { 0: "N", 90: "E", 180: "S", 270: "W" }[d];
      ticks.push(`<span class="${card ? "card" : ""}">${card || d}</span>`);
    }
    this.tape.querySelector(".ticks").innerHTML = ticks.join("");
    this.wpLine = el("div", "rh-wp");
    this.offEl = el("div", "rh-off", "OFF ROUTE");
    this.offEl.hidden = true;
    this.top.append(this.tape, this.wpLine, this.offEl);
    this.exp = el("div", "rh-exp");
    const light = this.conditions.light || "day";
    this.label = el("div", "rh-label", `REHEARSAL · terrain from reconstruction · enemy positions PLANNED, not observed · ${light}${this.conditions.nvg ? " + NVG" : ""}${this.conditions.fog_m ? ` · visibility ${this.conditions.fog_m} m` : ""}`);
    this.combat.hud.help.innerHTML = "<b>WASD</b> move &nbsp; <b>Shift</b> run &nbsp; <b>LMB</b> fire &nbsp; <b>R</b> reload &nbsp; <b>H</b> AI debug<br>Follow the route to the objective. Every second you are seen is recorded.";
    root.append(this.top, this.exp, this.label);
  }

  _conditions() {
    const canvas = this.combat.opts.canvas;
    canvas.style.filter = conditionFilter(this.conditions);
    // Visibility is drawn, not only obeyed by the bots: linear scene fog (the splat shader
    // honours it) closing to opaque at the stated range, darker at dusk and night.
    const fog = fogFor(this.conditions);
    const scene = this.engine?.scene;
    if (fog && scene) {
      scene.fog.type = FOG_LINEAR;
      scene.fogColor = new Color(...fog.color);
      scene.fogStart = fog.start;
      scene.fogEnd = fog.end;
    }
    if (this.conditions.nvg && this.conditions.light !== "day") document.body.appendChild(el("div", "rh-nvg"));
  }

  _beacons() {
    const app = this.engine;
    // Translucent beacons go on the overlay layer drawn after the splats (see pc.js).
    const overlay = app.scene.layers.getLayerByName("PlanOverlay");
    const mat = (r, g, b, a) => {
      const m = new StandardMaterial();
      m.useLighting = false; m.emissive = new Color(r, g, b); m.diffuse = new Color(0, 0, 0);
      m.opacity = a; m.blendType = BLEND_NORMAL; m.depthWrite = false; m.update();
      return m;
    };
    this.matNext = mat(1, 0.69, 0.12, 0.55);
    this.matWp = mat(1, 0.85, 0.5, 0.22);
    for (const w of this.s.route?.waypoints || []) {
      const e = new Entity(`wp:${w.name}`);
      e.addComponent("render", { type: "cylinder", castShadows: false, ...(overlay ? { layers: [overlay.id] } : {}) });
      e.setLocalScale(0.5, 14, 0.5);
      e.setPosition(w.at[0], w.at[1] + 7, w.at[2]);
      e.render.material = this.matWp;
      app.root.addChild(e);
      this.beacons.push(e);
    }
    this.routeLines = [];
    const wps = this.s.route?.waypoints || [];
    for (let k = 0; k + 1 < wps.length; k++) {
      const a = wps[k].at, b = wps[k + 1].at;
      this.routeLines.push(new Vec3(a[0], a[1] + 0.25, a[2]), new Vec3(b[0], b[1] + 0.25, b[2]));
    }
    this.phaseLines = [];
    for (const pl of this.s.phase_lines || []) {
      for (let k = 0; k + 1 < pl.line.length; k++) {
        const a = pl.line[k], b = pl.line[k + 1];
        this.phaseLines.push(new Vec3(a[0], a[1] + 0.3, a[2]), new Vec3(b[0], b[1] + 0.3, b[2]));
      }
    }
  }

  update(dt) {
    if (this.ended) return;
    if (this.sessionInfo && !this.link && !this.linkStarting) this._startSession();
    this.link?.update(dt);
    const c = this.combat, t = c.t;
    const p = c.playerState;
    const pos = [p.x, p.y, p.z];
    const events = [];
    let finished = false;
    if (this.tracker) {
      const st = this.tracker.update(pos, t);
      events.push(...st.events);
      this._drawTape(p.yaw, st);
      if (st.done) finished = true;
    }
    if (this.lastPos) {
      for (const pl of this.s.phase_lines || []) {
        if (crossesLine(this.lastPos, pos, pl.line)) events.push({ t, type: "phase_line", text: `Crossed ${pl.label}`, at: pos });
      }
    }
    this.lastPos = pos;
    const seeing = c.bots.filter(b => b.alive && t - b.lastSeen < 0.25).map(b => b.id);
    events.push(...this.exposure.update(dt, t, seeing, this.labels));
    const down = c.dead > 0;
    if (down && !this.wasDown) events.push({ t, type: "casualty", text: "Player down", at: pos });
    this.wasDown = down;
    for (const e of events) {
      this.recorder.event(e);
      if (["waypoint", "phase_line", "spotted", "off_route", "complete"].includes(e.type)) c.hud.feed(e.text, e.type === "spotted" || e.type === "off_route" ? "warn" : undefined);
    }
    this.recorder.sample(t, { ...p, health: c.health }, c.bots.map(b => ({
      id: b.id, symbol: b.plan?.symbol, label: b.label, behaviour: b.plan?.behaviour,
      x: b.pos.x, y: b.pos.y, z: b.pos.z, yaw: b.yaw, sees: t - b.lastSeen < 0.25, alive: b.alive,
    })));
    const seen = this.exposure.seenBy;
    this.exp.innerHTML = `<div class="cg-label">Exposure</div><div class="seen ${seen ? "hot" : "cold"}">${seen ? `SEEN BY ${seen}` : "UNSEEN"}</div>` +
      `<div>${this.exposure.total.toFixed(1)} s seen · ${fmtTime(t)}</div>`;
    const app = this.engine;
    if (this.routeLines.length) app.drawLines(this.routeLines, new Color(1, 0.62, 0.15, 0.9), true);
    if (this.phaseLines.length) app.drawLines(this.phaseLines, new Color(0.1, 0.1, 0.1, 1), true);
    const next = this.tracker?.done ? -1 : this.tracker?.index ?? -1;
    this.beacons.forEach((b, k) => { b.render.material = k === next ? this.matNext : this.matWp; b.enabled = k >= (this.tracker?.index ?? 0) - 1; });
    // Last: the arrival events above must be in the recording before it is uploaded.
    if (finished) this.end("objective reached");
  }

  _drawTape(yaw, st) {
    const heading = yawBearing(yaw, this.north);
    const px = 30 / 15;          // pixels per degree
    const ticks = this.tape.querySelector(".ticks");
    ticks.style.left = `${210 - (360 + heading) * px}px`;
    const wp = this.tape.querySelector(".wp");
    if (st.next) {
      let rel = st.next.bearing - heading;
      rel = ((rel + 540) % 360) - 180;
      wp.style.display = Math.abs(rel) < 100 ? "block" : "none";
      wp.style.left = `${210 + rel * px}px`;
      this.wpLine.innerHTML = `NEXT <b>${st.next.name}</b> · ${st.next.bearing.toFixed(0)}° · ${st.next.distance.toFixed(0)} m · ETA ${fmtTime(st.etaS)}`;
    } else {
      wp.style.display = "none";
      this.wpLine.innerHTML = "<b>OBJECTIVE REACHED</b>";
    }
    this.offEl.hidden = !st.offRoute;
  }

  _startSession() {
    this.linkStarting = true;
    const base = (this.saveUrl?.url || "/api/workspace/mission/run").replace(/mission\/run$/, "");
    import("./session.js").then(({ SessionLink }) => {
      this.link = new SessionLink({ engine: this.engine, combat: this.combat, rehearsal: this, base,
        session: this.sessionInfo.id, token: this.sessionInfo.token, name: this.sessionInfo.name });
      return this.link.start();
    }).catch((e) => { this.linkError = String(e?.message || e); });
  }

  async end(reason) {
    if (this.ended) return;
    this.ended = true;
    const c = this.combat;
    this.recorder.event({ t: c.t, type: "end", text: reason });
    const body = this.recorder.toRun({
      mission_revision: this.s.mission.revision,
      route: (this.s.route?.waypoints || []).map(w => [w.at[0], w.at[2]]),
      conditions: { light: this.conditions.light || "day", fog_m: String(this.conditions.fog_m || ""), filter: this.conditions.nvg ? "nvg" : "" },
    });
    const panel = el("div", "rh-end", `<h3>Rehearsal ended</h3><div>${reason} · ${fmtTime(c.t)} · seen ${this.exposure.total.toFixed(1)} s</div><div class="rh-save">Saving the run…</div>`);
    c.hud.root.appendChild(panel);
    let saved = null, error = null;
    if (this.saveUrl) {
      try {
        const r = await fetch(this.saveUrl.url, { method: "POST", headers: { "content-type": "application/json" },
          body: JSON.stringify({ scene: this.saveUrl.scene, id: this.s.mission.id, run: body }) });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || `save failed (${r.status})`);
        saved = j;
      } catch (e) { error = e.message || String(e); }
    }
    panel.querySelector(".rh-save").textContent = saved ? "Run saved for after-action review." : `Not saved: ${error || "no mission store"}`;
    window.__rehearsal = { ended: true, saved, error, reason };
    if (this.parent) this.parent.postMessage({ namespace: "groundcontrol", type: "rehearsal-ended", saved, error, reason }, this.origin);
  }

  snapshot() {
    return { t: +this.combat.t.toFixed(2), exposure: +this.exposure.total.toFixed(2), seenBy: this.exposure.seenBy,
      waypoint: this.tracker?.index ?? null, done: this.tracker?.done ?? false, frames: this.recorder.player.length,
      events: this.recorder.events.map(e => `${e.t.toFixed(1)} ${e.type} ${e.text}`), ended: this.ended };
  }
}
