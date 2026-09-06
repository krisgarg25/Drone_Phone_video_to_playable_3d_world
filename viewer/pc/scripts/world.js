/* Shared world queries + pooled combat effects. */
import { BODYMASK_STATIC, Color, Entity, StandardMaterial, Vec3 } from "playcanvas";

const WORLD_MASK = BODYMASK_STATIC;

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Line-of-sight and hitscan against the baked collision shell. */
export class Sight {
  constructor(app) {
    this.app = app;
    this.a = new Vec3();
    this.b = new Vec3();
  }

  _rigid() {
    return this.app.systems.rigidbody;
  }

  /** PlayCanvas reports hitFraction, not distance — normalise it for callers. */
  _wrap(res, len) {
    if (!res) return null;
    return {
      entity: res.entity,
      point: res.point,
      normal: res.normal,
      distance: (res.hitFraction ?? 0) * len,
    };
  }

  /** First solid hit along a ray, or null. Distance is metres from the origin. */
  cast(ox, oy, oz, dx, dy, dz, maxDist) {
    const rigid = this._rigid();
    if (!rigid) return null;
    const l = Math.hypot(dx, dy, dz) || 1;
    this.a.set(ox, oy, oz);
    this.b.set(ox + (dx / l) * maxDist, oy + (dy / l) * maxDist, oz + (dz / l) * maxDist);
    const res = rigid.raycastFirst(this.a, this.b, { filterCollisionMask: WORLD_MASK });
    return res ? this._wrap(res, maxDist) : null;
  }

  /** True when solid scan geometry sits strictly between two world points. */
  blocked(ax, ay, az, bx, by, bz, slack = 0.15) {
    const dx = bx - ax, dy = by - ay, dz = bz - az;
    const len = Math.hypot(dx, dy, dz);
    if (len < 1e-4) return false;
    const hit = this.cast(ax, ay, az, dx, dy, dz, len);
    if (!hit) return false;
    return hit.distance > slack && hit.distance < len - slack;
  }
}

/**
 * Walk a ray origin forward until it is no longer buried in solid geometry.
 *
 * Bullet reports `hitFraction 0` for a ray that starts inside a body, which reads
 * as a hit at distance 0 — so a round spawned from the muzzle of a player leaning
 * into a seat back would die on its first probe and the gun would seem jammed.
 * Returns the origin itself when no clearance is found within `max`: a barrel
 * inside a wall SHOULD stop the round, and faking it by pushing the spawn half a
 * metre forward would let a shot cross geometry that is thicker than `max`.
 */
export function escapeSolid(sight, from, dir, max = 0.45, step = 0.06) {
  const o = { x: from.x, y: from.y, z: from.z };
  for (let d = 0; d <= max; d += step) {
    const hit = sight.cast(o.x, o.y, o.z, dir.x, dir.y, dir.z, step * 0.5);
    if (!hit || hit.distance > 1e-3) return o;
    o.x += dir.x * step; o.y += dir.y * step; o.z += dir.z * step;
  }
  return o;
}

/**
 * Pooled unlit primitives for hit decals and impact puffs. Each slot owns a
 * material so opacity can fade without disturbing anything else in the scene.
 */
export class Effects {
  constructor(app, { decals = 20, puffs = 16, flashes = 6 } = {}) {
    this.app = app;
    this.root = new Entity("fx");
    app.root.addChild(this.root);
    this.decals = this._pool("decal", "box", decals);
    this.puffs = this._pool("puff", "sphere", puffs);
    // Additive on purpose: a flash is light, not dust. The same ball of colour
    // blended normally became a flat disc that hid the room, because the barrel is
    // a metre from the eye and nothing behind it survives an opaque quad there.
    // Blend 6 is ADDITIVEALPHA (SRC_ALPHA, ONE) — plain ADDITIVE is (ONE, ONE) in
    // this engine and would ignore the opacity the fade runs on.
    this.flashes = this._pool("flash", "sphere", flashes, 6);
    this.live = [];
  }

  _pool(name, type, count, blend = 2) {
    const slots = [];
    for (let i = 0; i < count; i++) {
      const mat = new StandardMaterial();
      mat.useLighting = false;
      mat.blendType = blend;
      mat.opacity = 0;
      mat.depthWrite = false;
      mat.update();
      const ent = new Entity(`${name}${i}`);
      ent.addComponent("render", { type, material: mat });
      ent.enabled = false;
      this.root.addChild(ent);
      slots.push({ ent, mat, busy: false });
    }
    return slots;
  }

  _take(slots) {
    let free = slots.find((s) => !s.busy);
    if (!free) {
      free = slots.reduce((a, b) => (a.birth < b.birth ? a : b));
      // It is still in `live` with a fade about to be nulled — leaving it there
      // makes the next update() read `fade.life` off nothing and take the frame
      // loop down with it. A decal pool that lives for 7 s reaches this far more
      // often than the 0.09 s tracers it replaced did.
      const i = this.live.indexOf(free);
      if (i >= 0) this.live.splice(i, 1);
      this._kill(free);
    }
    free.busy = true;
    free.birth = performance.now();
    return free;
  }

  _kill(slot) {
    slot.busy = false;
    slot.fade = null;
    slot.ent.enabled = false;
    slot.mat.opacity = 0;
  }

  _color(mat, rgb) {
    mat.diffuse = new Color(rgb[0], rgb[1], rgb[2]);
    mat.update();
  }

  scorch(point, normal, rgb = [0.1, 0.09, 0.085], size = 0.14, life = 7) {
    const slot = this._take(this.decals);
    const { ent, mat } = slot;
    this._color(mat, rgb);
    const n = normal || { x: 0, y: 1, z: 0 };
    ent.enabled = true;
    // Lie flat on the struck face: box pivots are their centre, so half the skin
    // depth clear of the surface, and the flattened Z axis points along the normal.
    ent.setPosition(point.x + n.x * 0.012, point.y + n.y * 0.012, point.z + n.z * 0.012);
    ent.lookAt(point.x + n.x * 2, point.y + n.y * 2, point.z + n.z * 2);
    ent.setLocalScale(size, size, 0.006);
    slot.fade = { from: 0.85, to: 0, life };
    this.live.push(slot);
  }

  flash(point, rgb = [1, 0.78, 0.35], size = 0.05, life = 0.05) {
    const slot = this._take(this.flashes);
    const { ent, mat } = slot;
    this._color(mat, rgb);
    ent.enabled = true;
    ent.setPosition(point.x, point.y, point.z);
    ent.setLocalScale(size, size, size);
    slot.base = [size, size, size];
    slot.fade = { from: 0.9, to: 0, life, grow: 2.6 };
    this.live.push(slot);
  }

  impact(point, normal, rgb = [0.85, 0.82, 0.78], size = 0.09, life = 0.28) {
    const slot = this._take(this.puffs);
    const { ent, mat } = slot;
    this._color(mat, rgb);
    ent.enabled = true;
    ent.setPosition(
      point.x + (normal?.x ?? 0) * 0.03,
      point.y + (normal?.y ?? 0) * 0.03,
      point.z + (normal?.z ?? 0) * 0.03,
    );
    ent.setLocalScale(size, size, size);
    slot.base = [size, size, size];
    slot.fade = { from: 0.75, to: 0, life, grow: 2.4 };
    this.live.push(slot);
  }

  spark(point, rgb = [1, 0.8, 0.35]) {
    for (let i = 0; i < 3; i++) {
      const slot = this._take(this.puffs);
      const { ent, mat } = slot;
      this._color(mat, rgb);
      ent.enabled = true;
      ent.setPosition(point.x + (Math.random() - 0.5) * 0.14, point.y + (Math.random() - 0.5) * 0.14, point.z + (Math.random() - 0.5) * 0.14);
      const s = 0.05 + Math.random() * 0.04;
      ent.setLocalScale(s, s, s);
      slot.base = [s, s, s];
      slot.fade = { from: 1, to: 0, life: 0.16 + Math.random() * 0.12 };
      this.live.push(slot);
    }
  }

  update(dt) {
    const now = performance.now();
    for (let i = this.live.length - 1; i >= 0; i--) {
      const slot = this.live[i];
      const f = slot.fade;
      const t = (now - slot.birth) / 1000 / f.life;
      if (t >= 1) {
        this._kill(slot);
        this.live.splice(i, 1);
        continue;
      }
      slot.mat.opacity = clamp(f.from * (1 - t), 0, 1);
      if (f.grow) {
        const s = 1 + t * f.grow;
        slot.ent.setLocalScale(slot.base[0] * s, slot.base[1] * s, slot.base[2] * s);
      }
    }
  }
}

/**
 * Travelling rounds — the shot you can watch cross the room.
 *
 * This replaces instant hitscan. A round advances `speed * dt` metres per frame
 * and the whole of that step is swept as ONE ray, which is what makes cover solid:
 * a raycast is a segment test, not a point sample, so a 1.17 m step at 60 fps
 * cannot skip a 0.2 m furniture box. Slicing the step into 0.25 m probes was the
 * first design here and it buys nothing but five times the casts — and it is
 * actively worse if the target test rejects shallow entry angles, because the
 * probes then walk straight through a sphere whose surface keeps falling inside
 * the rejected band. See hitBody in character.js, which has no such band.
 *
 * Damage resolves on impact, about 14 ms per metre of range after the trigger
 * pull. That delay is the point of the change rather than its cost: it is what
 * makes a hit land where the streak ended up, and what makes incoming fire
 * something the player can see coming and step out of.
 *
 * Trajectory stays flat on purpose. Gravity would drop the round 0.4 m over the
 * 20 m a room scan offers, which reads as the gun missing when it did not, and
 * would fight aim assist tuned in the hitscan era.
 */
export class Projectiles {
  constructor(app, {
    sight, effects, audio, count = 32, speed = 70, streak = 1.4, width = 0.024,
    listener = null, minFx = 0.7,
  } = {}) {
    this.sight = sight;
    this.effects = effects;
    this.audio = audio;
    this.speed = speed;
    this.streak = streak;
    this.width = width;
    /** Whose eyes the effects are judged from, and how close is too close to draw. */
    this.listener = listener;
    this.minFx = minFx;
    this.stats = { fired: 0, body: 0, world: 0, expired: 0 };
    this.live = [];
    this.root = new Entity("rounds");
    app.root.addChild(this.root);
    this.pool = [];
    for (let i = 0; i < count; i++) {
      const mat = new StandardMaterial();
      mat.useLighting = false;
      mat.blendType = 2;
      mat.opacity = 0.95;
      mat.depthWrite = false;
      mat.update();
      const ent = new Entity(`round${i}`);
      ent.addComponent("render", { type: "box", material: mat });
      ent.enabled = false;
      this.root.addChild(ent);
      this.pool.push({ ent, mat, round: null });
    }
  }

  /**
   * `targets` is a function, re-read every frame, so a bot that steps into a live
   * round's path is still taken. `onHit(hit, point, dmg)` runs once, on impact.
   */
  spawn({ from, dir, range, damage, targets, onHit, shooter = null, rgb = [1, 0.9, 0.55] }) {
    const l = Math.hypot(dir.x, dir.y, dir.z) || 1;
    const d = { x: dir.x / l, y: dir.y / l, z: dir.z / l };
    const origin = escapeSolid(this.sight, from, d);
    const slot = this._take();
    slot.round = {
      x: origin.x, y: origin.y, z: origin.z, dx: d.x, dy: d.y, dz: d.z,
      traveled: 0, range, damage, shooter,
      targets: targets || (() => []), onHit: onHit || (() => {}),
    };
    // The muzzle flash is additive light, not a LightComponent: every material in
    // this viewer, the splat included, sets useLighting=false, so a real light would
    // move no pixels at all. Small and brief on purpose — the barrel sits ~1 m from
    // the eye, where an opaque 0.2 m ball covers a fifth of the screen.
    this.effects.flash(origin);
    slot.mat.diffuse = new Color(rgb[0], rgb[1], rgb[2]);
    slot.mat.update();
    this.stats.fired++;
    this._draw(slot);
    return slot.round;
  }

  update(dt) {
    for (let i = this.live.length - 1; i >= 0; i--) {
      const slot = this.live[i];
      if (this._advance(slot, dt)) this._draw(slot);
      else this.live.splice(i, 1);
    }
  }

  _take() {
    let slot = this.pool.find((s) => !s.round);
    if (!slot) {
      // Steal the oldest. The pool covers a full-auto burst plus the bots' own
      // fire inside the ~1.3 s a 90 m round takes to land, so this only happens
      // once a round has already flown off the interesting part of the scene.
      slot = this.live.shift();
      this._drop(slot);
    }
    this.live.push(slot);
    return slot;
  }

  /**
   * Draw an impact only where it can be seen. A dust ball an arm's length from the
   * eye covers the room - a round hitting the player painted the whole screen
   * brown - and it adds nothing the hit marker, the crack and the flinch do not
   * already say. Only the *visuals* are gated; the hit and its damage are not.
   */
  _farEnough(point) {
    if (!this.listener) return true;
    const e = this.listener();
    return !e || Math.hypot(point.x - e.x, point.y - e.y, point.z - e.z) > this.minFx;
  }

  _drop(slot) {
    slot.round = null;
    slot.ent.enabled = false;
  }

  /** @returns {boolean} false once the round has resolved or flown past its range. */
  _advance(slot, dt) {
    const r = slot.round;
    const d = { x: r.dx, y: r.dy, z: r.dz };
    const o = { x: r.x, y: r.y, z: r.z };
    const step = Math.min(this.speed * dt, r.range - r.traveled);
    if (step <= 1e-4) {
      this.stats.expired++;
      r.result = { kind: "expired", point: { x: r.x, y: r.y, z: r.z } };
      this._drop(slot);
      return false;
    }
    const wall = this.sight.cast(o.x, o.y, o.z, d.x, d.y, d.z, step);
    let body = null;
    for (const t of r.targets()) {
      // Capped by the wall: a body behind cover is not hit by a round cover stops.
      const hit = t.hitTest(o, d, wall ? wall.distance : step);
      if (hit && (!body || hit.t < body.t)) body = hit;
    }
    if (!wall && !body) {
      this._moveTo(r, d, step);
      if (r.traveled >= r.range - 1e-4) {
        this.stats.expired++;
        r.result = { kind: "expired", point: { x: r.x, y: r.y, z: r.z } };
        this._drop(slot);
        return false;
      }
      return true;
    }
    if (body && (!wall || body.t <= wall.distance)) {
      this._moveTo(r, d, body.t);
      const point = { x: r.x, y: r.y, z: r.z };
      const falloff = Math.max(0.45, 1 - (r.traveled / r.range) * 0.6);
      const dmg = r.damage * (body.mult ?? 1) * falloff;
      this.stats.body++;
      r.result = { kind: "body", point, part: body.part, dmg, traveled: r.traveled };
      if (this._farEnough(point)) this.effects.spark(point, [1, 0.55, 0.2]);
      r.onHit(body, point, dmg);
      this._drop(slot);
      return false;
    }
    this._moveTo(r, d, wall.distance);
    const point = wall.point ? { x: wall.point.x, y: wall.point.y, z: wall.point.z }
                             : { x: r.x, y: r.y, z: r.z };
    this.stats.world++;
    r.result = { kind: "world", point, traveled: r.traveled };
    if (this._farEnough(point)) {
      this.effects.impact(point, wall.normal);
      this.effects.scorch(point, wall.normal);
    }
    // Only the player's own rounds click on landing, as they did under hitscan;
    // every bot burst scoring an impact beside the ear would be noise, not info.
    if (!r.shooter) this.audio.impact();
    this._drop(slot);
    return false;
  }

  _moveTo(r, d, t) {
    r.x += d.x * t; r.y += d.y * t; r.z += d.z * t;
    r.traveled += t;
  }

  _draw(slot) {
    const r = slot.round;
    // Shorter than the streak until the round has flown that far, so the tail
    // never sticks out through the muzzle on the first frame.
    const len = Math.min(this.streak, r.traveled);
    const ent = slot.ent;
    ent.enabled = true;
    ent.setPosition(r.x - r.dx * len * 0.5, r.y - r.dy * len * 0.5, r.z - r.dz * len * 0.5);
    ent.lookAt(r.x, r.y, r.z);
    ent.setLocalScale(this.width, this.width, Math.max(0.03, len));
  }
}
