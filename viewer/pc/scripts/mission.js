/* Mission rehearsal logic with no engine in it (shared with Node tests).
 *
 * The scenario comes from the backend (`/api/workspace/mission/scenario`): planned enemy
 * posts with facing/sector/range/behaviour, the approach route with named waypoints,
 * phase lines and markers. This module turns that into bot settings, tracks the player
 * along the route, meters exposure and records the run for after-action review.
 * Nothing here is an observation: the HUD says so (RH-10).
 */

export const BEHAVIOURS = ["sentry", "patrol", "overwatch", "reaction"];
export const LIGHT = ["day", "dusk", "night"];
const TAU = Math.PI * 2;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const finite = v => typeof v === "number" && Number.isFinite(v);
const point3 = p => Array.isArray(p) && p.length === 3 && p.every(finite);

/** Validate the scenario payload; throw on anything malformed or oversized. */
export function validateScenario(s) {
  if (!s || typeof s !== "object") throw new Error("scenario must be an object");
  if (!Array.isArray(s.bots) || s.bots.length > 64) throw new Error("scenario bots must be a bounded list");
  for (const b of s.bots) {
    if (!point3(b.position) || !Array.isArray(b.facing_xz) || !b.facing_xz.every(finite)) throw new Error("bot needs a position and facing");
    if (!BEHAVIOURS.includes(b.behaviour)) throw new Error(`unknown behaviour ${b.behaviour}`);
    if (!(b.range_m > 0) || !(b.sector_deg > 0) || !Number.isInteger(b.count) || b.count < 1 || b.count > 8) throw new Error("bot range, sector and count must be sane");
    if (b.patrol !== null && b.patrol !== undefined && (!Array.isArray(b.patrol) || !b.patrol.every(point3))) throw new Error("patrol must be 3D points");
  }
  if (s.route !== null && s.route !== undefined) {
    if (!Array.isArray(s.route.waypoints) || s.route.waypoints.length < 2 || s.route.waypoints.length > 200 ||
        !s.route.waypoints.every(w => w && point3(w.at))) throw new Error("route needs 2..200 waypoints");
  }
  if (!Array.isArray(s.north_xz) || s.north_xz.length !== 2 || !s.north_xz.every(finite)) throw new Error("scenario needs north");
  return s;
}

/** Bot yaw (combat convention: yaw 0 faces -z, yaw = atan2(-dx, -dz)) for a facing vector. */
export function yawOf([fx, fz]) { return Math.atan2(-fx, -fz); }

/**
 * Per-bot settings for a behaviour. Sight range is the planned range, capped by fog; at
 * night without NVG the planned post sees half as far. Sentries and overwatch hold their
 * post (they turn toward noise but do not leave); patrols walk their route; a reaction
 * force moves to noise and last-known positions like the arena bots.
 */
export function botSettings(bot, conditions = {}) {
  const fog = finite(conditions.fog_m) && conditions.fog_m > 0 ? conditions.fog_m : Infinity;
  const night = conditions.light === "night" ? 0.5 : conditions.light === "dusk" ? 0.75 : 1;
  const overwatch = bot.behaviour === "overwatch";
  const sector = clamp(bot.sector_deg, 10, 360) * Math.PI / 180;
  return {
    viewRange: Math.min(bot.range_m * night * (overwatch ? 1.25 : 1), fog),
    viewCone: overwatch ? Math.min(sector, 70 * Math.PI / 180) : sector,
    baseYaw: yawOf(bot.facing_xz),
    scanAmp: bot.behaviour === "sentry" ? Math.max(0, sector / 2 - 15 * Math.PI / 180) : overwatch ? Math.min(sector / 2, 10 * Math.PI / 180) : 0,
    stayPut: bot.behaviour === "sentry" || overwatch,
    patrol: bot.behaviour === "patrol" && Array.isArray(bot.patrol) && bot.patrol.length >= 2 ? bot.patrol : null,
    moveSpeed: bot.behaviour === "patrol" ? 1.4 : bot.behaviour === "reaction" ? 3.0 : 2.45,
    // Distance at which the bot is as willing to shoot as an arena bot is at ~10 m.
    engageRange: bot.role === "sniper" ? Math.max(60, bot.range_m) : bot.role === "machine_gun" ? 80 : overwatch ? 60 : 30,
    roundRange: bot.role === "sniper" ? Math.max(90, bot.range_m) : 90,
    fire: fireProfile(bot.role),
  };
}

/**
 * How a post fires: rounds per burst, the gap within a burst and between bursts, and how
 * much tighter its aim is than a rifleman's. A sniper fires single aimed rounds seconds
 * apart; a machine gun long fast bursts; everyone else the arena's short rifle bursts.
 */
export function fireProfile(role) {
  if (role === "sniper") return { burstMin: 1, burstMax: 1, withinS: 0, betweenS: [2.5, 4.0], aimScale: 0.35 };
  if (role === "machine_gun") return { burstMin: 4, burstMax: 8, withinS: 0.08, betweenS: [0.7, 1.4], aimScale: 1.3 };
  return { burstMin: 2, burstMax: 5, withinS: 0.11, betweenS: null, aimScale: 1 };
}

/** Where copies of a post stand: the post, then a ring 1.4 m out. */
export function postOffsets(count) {
  const out = [[0, 0]];
  for (let i = 1; i < count; i++) {
    const a = (i - 1) / Math.max(1, count - 1) * TAU;
    out.push([Math.cos(a) * 1.4, Math.sin(a) * 1.4]);
  }
  return out;
}

/** Sentry scan: the yaw a holding post looks toward at time t (seconds). */
export function scanYaw(settings, t, period = 9) {
  return settings.baseYaw + settings.scanAmp * Math.sin(t * TAU / period);
}

/** Bearing in degrees clockwise from north (the scene's north vector, viewer xz). */
export function bearingDeg(from, to, north) {
  const dx = to[0] - from[0], dz = to[2] - from[2];
  const east = [-north[1], north[0]];
  const b = Math.atan2(dx * east[0] + dz * east[1], dx * north[0] + dz * north[1]) * 180 / Math.PI;
  return (b + 360) % 360;
}

/** Heading of a combat yaw as a bearing (yaw 0 faces -z). */
export function yawBearing(yaw, north) {
  return bearingDeg([0, 0, 0], [-Math.sin(yaw), 0, -Math.cos(yaw)], north);
}

function segDist(p, a, b) {
  const abx = b[0] - a[0], abz = b[2] - a[2];
  const len2 = abx * abx + abz * abz || 1e-12;
  const t = clamp(((p[0] - a[0]) * abx + (p[2] - a[2]) * abz) / len2, 0, 1);
  return Math.hypot(p[0] - (a[0] + t * abx), p[2] - (a[2] + t * abz));
}

/** Follows the planned route: next waypoint, bearing, distance, ETA, off-route, arrivals. */
export class WaypointTracker {
  constructor(route, { arriveM = 3, offRouteM = 15, speed = 1.4, north = [0, -1] } = {}) {
    this.wps = route.waypoints;
    this.index = 1;                    // the start point is where the run begins
    this.arriveM = arriveM;
    this.offRouteM = offRouteM;
    this.speed = speed;
    this.north = north;
    this.done = false;
    this.off = false;
  }

  update(pos, t) {
    const events = [];
    if (!this.done) {
      // Arrive at the next waypoint, or skip ahead to a later one the player reached first.
      for (let k = this.index; k < this.wps.length; k++) {
        const w = this.wps[k].at;
        if (Math.hypot(pos[0] - w[0], pos[2] - w[2]) <= this.arriveM) {
          for (let m = this.index; m <= k; m++) events.push({ t, type: "waypoint", text: this.wps[m].name, at: this.wps[m].at });
          this.index = k + 1;
          if (this.index >= this.wps.length) { this.done = true; events.push({ t, type: "complete", text: `${this.wps.at(-1).name} reached` }); }
          break;
        }
      }
    }
    let off = Infinity;
    for (let k = 0; k + 1 < this.wps.length; k++) off = Math.min(off, segDist(pos, this.wps[k].at, this.wps[k + 1].at));
    const wasOff = this.off;
    this.off = off > this.offRouteM;
    if (this.off && !wasOff) events.push({ t, type: "off_route", text: `${off.toFixed(0)} m off the route` });
    if (!this.off && wasOff) events.push({ t, type: "on_route", text: "Back on the route" });
    const next = this.done ? null : this.wps[this.index];
    let remaining = 0;
    if (next) {
      remaining = Math.hypot(next.at[0] - pos[0], next.at[2] - pos[2]);
      for (let k = this.index; k + 1 < this.wps.length; k++) {
        const a = this.wps[k].at, b = this.wps[k + 1].at;
        remaining += Math.hypot(b[0] - a[0], b[2] - a[2]);
      }
    }
    return {
      events, done: this.done, offRoute: this.off, offRouteM: off,
      next: next ? { name: next.name, index: this.index, distance: Math.hypot(next.at[0] - pos[0], next.at[2] - pos[2]),
        bearing: bearingDeg(pos, next.at, this.north) } : null,
      remainingM: remaining, etaS: remaining / this.speed,
    };
  }
}

/** Crossing of a phase line (2D segment intersection on the xz plane) between two positions. */
export function crossesLine(a, b, line) {
  for (let k = 0; k + 1 < line.length; k++) {
    const p = line[k], q = line[k + 1];
    const r = [b[0] - a[0], b[2] - a[2]], s = [q[0] - p[0], q[2] - p[2]];
    const den = r[0] * s[1] - r[1] * s[0];
    if (Math.abs(den) < 1e-12) continue;
    const w = [p[0] - a[0], p[2] - a[2]];
    const t = (w[0] * s[1] - w[1] * s[0]) / den, u = (w[0] * r[1] - w[1] * r[0]) / den;
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) return true;
  }
  return false;
}

/** Seconds seen, in total and per bot; a bot's first sighting becomes a "spotted" event. */
export class ExposureMeter {
  constructor() { this.total = 0; this.perBot = new Map(); this.spotted = new Set(); this.seenBy = 0; }
  update(dt, t, seeing, labels = new Map()) {
    const events = [];
    this.seenBy = seeing.length;
    if (seeing.length) this.total += dt;
    for (const id of seeing) {
      this.perBot.set(id, (this.perBot.get(id) || 0) + dt);
      if (!this.spotted.has(id)) { this.spotted.add(id); events.push({ t, type: "spotted", bot: id, text: `Seen by ${labels.get(id) || `#${id}`}` }); }
    }
    return events;
  }
}

/** Fixed-rate recording of the player and every bot, plus events, for the AAR. */
export class Recorder {
  constructor(hz = 10, maxFrames = 12000) { this.hz = hz; this.dt = 1 / hz; this.next = 0; this.player = []; this.bots = new Map(); this.events = []; this.maxFrames = maxFrames; }
  sample(t, player, bots) {
    if (t + 1e-9 < this.next || this.player.length >= this.maxFrames) return false;
    this.next = t + this.dt;
    const r = v => Math.round(v * 1000) / 1000;
    this.player.push([r(t), r(player.x), r(player.y), r(player.z), r(player.yaw), Math.round(player.health)]);
    for (const b of bots) {
      if (!this.bots.has(b.id)) this.bots.set(b.id, { id: b.id, symbol: b.symbol || "", label: b.label || "", behaviour: b.behaviour || "", frames: [] });
      this.bots.get(b.id).frames.push([r(t), r(b.x), r(b.y), r(b.z), r(b.yaw), b.sees ? 1 : 0, b.alive ? 1 : 0]);
    }
    return true;
  }
  event(e) { if (this.events.length < 5000) this.events.push(e); }
  /** The upload body; halves the rate until it fits under ``maxBytes`` (the API takes 2 MiB). */
  toRun(extra = {}, maxBytes = 1900000) {
    let stride = 1, body;
    do {
      const keep = (_, i) => i % stride === 0;
      body = { hz: this.hz / stride, player: this.player.filter(keep),
        bots: [...this.bots.values()].map(b => ({ ...b, frames: b.frames.filter(keep) })), events: this.events, ...extra };
      stride *= 2;
    } while (JSON.stringify(body).length > maxBytes && stride <= 16);
    return body;
  }
}

/** CSS filter for the canvas under the chosen conditions (NVG is a green phosphor look). */
export function conditionFilter({ light = "day", nvg = false } = {}) {
  if (nvg && light !== "day") return "grayscale(1) sepia(1) hue-rotate(58deg) saturate(3.2) brightness(1.25) contrast(1.35)";
  if (light === "night") return "brightness(0.28) saturate(0.5) contrast(1.1)";
  if (light === "dusk") return "brightness(0.62) sepia(0.35) saturate(0.8)";
  return "";
}

/** Linear fog for the stated visibility: clear to 35% of it, opaque at it. Null without fog. */
export function fogFor(conditions = {}) {
  const v = Number(conditions.fog_m);
  if (!Number.isFinite(v) || v <= 0) return null;
  const tone = conditions.light === "night" ? 0.08 : conditions.light === "dusk" ? 0.35 : 0.72;
  return { start: v * 0.35, end: v, color: [tone, tone * 1.02, tone * 1.06] };
}

/** The actions a batch of commands asks for, in order, skipping ones already applied. */
export function commandActions(commands, lastSeq) {
  const out = [];
  let seq = lastSeq;
  for (const c of commands || []) {
    if (typeof c?.seq !== "number" || c.seq <= seq) continue;
    seq = c.seq;
    if (c.type === "move_bot" && Number.isInteger(c.bot) && Array.isArray(c.to) && c.to.length === 3) out.push({ kind: "move", bot: c.bot, to: c.to });
    else if (c.type === "message" && typeof c.text === "string") out.push({ kind: "message", text: c.text.slice(0, 140) });
    else if (c.type === "end") out.push({ kind: "end" });
  }
  return { actions: out, seq };
}
