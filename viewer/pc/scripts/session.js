/* Multi-user rehearsal link (RH-9): share this player's pose, show the others, obey the instructor.

Talks to the local server's in-memory session (scripts/mission_session.py) with the
session token from the join link. About eight times a second it posts this player's pose
(and, for the instructor's map, where this client's enemy posts are) and receives the other
players and any instructor commands:

* move_bot {bot, to} - teleports that enemy post here (each client simulates its own bots);
* message {text}     - a banner on the HUD;
* end                - ends this rehearsal (the run is saved as usual where saving is allowed).

Remote players are drawn as blue figures with their name; a player not heard from for five
seconds is drawn grey. Engine-free logic (command handling, staleness) is exported for tests.
*/
import { Color, Entity, StandardMaterial } from "playcanvas";
import { commandActions } from "./mission.js";

export const POST_EVERY_S = 0.125;

export class SessionLink {
  constructor({ engine, combat, rehearsal, session, token, name, base }) {
    Object.assign(this, { engine, combat, rehearsal, session, token, name, base });
    this.player = null;
    this.seq = 0;
    this.acc = 0;
    this.inflight = false;
    this.remotes = new Map();
    this.status = "joining";
    this.error = null;
  }

  async _post(route, body) {
    const r = await fetch(this.base + route, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ session: this.session, token: this.token, ...body }) });
    const out = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(out.error || `session ${route} failed (${r.status})`);
    return out;
  }

  async start() {
    try {
      const j = await this._post("mission/session/join", { name: this.name, role: "player" });
      this.player = j.player; this.seq = j.seq; this.status = "live";
      this._banner(`Joined ${j.name}`);
    } catch (e) {
      this.status = "failed"; this.error = String(e.message || e);
      this._banner(`Session: ${this.error}`);
    }
  }

  update(dt) {
    if (this.status !== "live" || this.inflight) return;
    this.acc += dt;
    if (this.acc < POST_EVERY_S) return;
    this.acc = 0;
    const p = this.combat.playerState;
    const bots = this.combat.bots.map((b) => [b.pos.x, b.pos.y, b.pos.z, b.yaw ?? 0]);
    this.inflight = true;
    this._post("mission/session/state", { player: this.player, pose: [p.x, p.y, p.z, p.yaw], health: this.combat.health,
      alive: this.combat.health > 0, since: this.seq, bots })
      .then((out) => { this._others(out.players || []); this._apply(out.commands || []); })
      .catch((e) => { this.error = String(e.message || e); })
      .finally(() => { this.inflight = false; });
  }

  _apply(commands) {
    const { actions, seq } = commandActions(commands, this.seq);
    this.seq = seq;
    for (const a of actions) {
      if (a.kind === "move") {
        const bot = this.combat.bots[a.bot];
        if (!bot) continue;
        const [x, y, z] = a.to;
        bot.pos.x = x; bot.pos.y = y; bot.pos.z = z;
        if (bot.post) { bot.post.x = x; bot.post.y = y; bot.post.z = z; }
        bot.ent?.setPosition(x, y, z);
        this._banner(`Instructor moved ${bot.label || `enemy ${a.bot}`}`);
      } else if (a.kind === "message") this._banner(`Instructor: ${a.text}`);
      else if (a.kind === "end") this.rehearsal?.end("ended by the instructor");
    }
  }

  _others(players) {
    const seen = new Set();
    for (const pl of players) {
      if (!Array.isArray(pl.pose)) continue;
      seen.add(pl.id);
      let r = this.remotes.get(pl.id);
      if (!r) r = this._figure(pl);
      const [x, y, z, yaw] = pl.pose;
      r.ent.setPosition(x, y - 0.9, z);
      r.ent.setEulerAngles(0, (yaw * 180) / Math.PI, 0);
      r.mat.diffuse = pl.lost || !pl.alive ? new Color(0.45, 0.45, 0.45) : new Color(0.25, 0.55, 1);
      r.mat.update();
      r.name = pl.name;
    }
    for (const [id, r] of this.remotes) if (!seen.has(id)) { r.ent.destroy(); this.remotes.delete(id); }
  }

  _figure(pl) {
    const mat = new StandardMaterial();
    mat.diffuse = new Color(0.25, 0.55, 1);
    mat.update();
    const ent = new Entity(`remote-${pl.id}`);
    const body = new Entity("body");
    body.addComponent("render", { type: "capsule", material: mat });
    body.setLocalScale(0.5, 1.7, 0.5);
    body.setLocalPosition(0, 0.85, 0);
    const head = new Entity("head");
    head.addComponent("render", { type: "sphere", material: mat });
    head.setLocalScale(0.28, 0.28, 0.28);
    head.setLocalPosition(0, 1.62, 0);
    ent.addChild(body); ent.addChild(head);
    this.engine.root.addChild(ent);
    const r = { ent, mat, name: pl.name };
    this.remotes.set(pl.id, r);
    return r;
  }

  _banner(text) {
    try { this.combat.hud.feed(text, "warn"); } catch { /* HUD not ready */ }
  }
}
