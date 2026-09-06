/* Enemy bot: kinematic body, sight/hearing memory, ambush state machine, utility fire. */
import { Color, Entity, StandardMaterial, Vec3 } from "playcanvas";
import { AIM, EYE_Y, GUN_Y, TARGET_HEIGHT, hitBody, makeCharacter } from "./character.js";

export const STATE = {
  HOLD: "HOLD",
  ENGAGE: "ENGAGE",
  RELOCATE: "RELOCATE",
  FLANK: "FLANK",
  SEARCH: "SEARCH",
  DEAD: "DEAD",
};

/** How exposed the bot is in each state: the more visible it is, the longer it holds fire. */
const STATE_EXPOSURE = {
  [STATE.HOLD]: 0.15,
  [STATE.FLANK]: 0.35,
  [STATE.ENGAGE]: 0.5,
  [STATE.SEARCH]: 0.6,
  [STATE.RELOCATE]: 0.7,
  [STATE.DEAD]: 1,
};

export const SKINS = [
  { suit: [0.72, 0.2, 0.18], accent: [0.15, 0.12, 0.12] },
  { suit: [0.78, 0.5, 0.12], accent: [0.12, 0.11, 0.1] },
  { suit: [0.2, 0.5, 0.62], accent: [0.1, 0.12, 0.14] },
  { suit: [0.45, 0.28, 0.66], accent: [0.11, 0.1, 0.13] },
];

/** Body heights, read off the same hit model the player uses (character.js AIM). */
const HIP = AIM.hip.y, CHEST = AIM.chest.y, HEAD = AIM.head.y, EYE = EYE_Y, MUZZLE = GUN_Y;

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export class Bot {
  constructor(id, ctx, opts = {}) {
    this.id = id;
    this.ctx = ctx;
    this.skill = opts.skill ?? 1;
    this.state = STATE.HOLD;
    this.health = opts.health ?? 100;
    this.healthMax = this.health;
    this.awareness = 0;
    this.pos = { x: opts.spawn.x, y: opts.spawn.y, z: opts.spawn.z };
    this.dist = Infinity;
    this.magSize = opts.magSize ?? 30;
    this.ammoLeft = this.magSize;
    this.yaw = opts.spawn.yaw ?? 0;
    this.wantYaw = this.yaw;
    this.path = [];
    this.pathT = 0;
    this.spot = null;
    this.spotRole = null;
    this.reserved = null;
    this.lastKnown = null;
    this.lastSeen = -999;
    this.noise = null;
    this.visible = false;
    this.seenMs = 0;
    this.nextThink = 0;
    this.nextShot = 0;
    this.burst = 0;
    this.burstGap = 0;
    this.holdTimer = 0;
    this.sinceDamage = 99;
    this.sinceOwnShot = 99;
    this.hitFlash = 0;
    this.deadT = 0;
    this.alive = true;
    this.stats = { shots: 0, hits: 0 };
    this._buildMesh();
  }

  _buildMesh() {
    const { app } = this.ctx;
    const skin = SKINS[this.id % SKINS.length];
    const mk = (r, g, b) => {
      const m = new StandardMaterial();
      m.useLighting = false;
      m.diffuse = new Color(r, g, b);
      m.update();
      return { mat: m, base: [r, g, b] };
    };
    this.mats = [mk(...skin.suit), mk(...skin.accent)];
    const suit = () => this.mats[0].mat;
    const dark = () => this.mats[1].mat;

    this.ent = new Entity(`bot${this.id}`);
    const torso = new Entity("torso");
    torso.addComponent("render", { type: "capsule", material: suit() });
    torso.setLocalScale(0.44, 0.42, 0.32);
    torso.setLocalPosition(0, HIP, 0);
    const head = new Entity("head");
    head.addComponent("render", { type: "sphere", material: suit() });
    head.setLocalScale(0.26, 0.26, 0.26);
    head.setLocalPosition(0, HEAD, 0);
    const gun = new Entity("gun");
    gun.addComponent("render", { type: "box", material: dark() });
    gun.setLocalScale(0.09, 0.12, 0.7);
    gun.setLocalPosition(0.17, MUZZLE - 0.16, -0.34);
    const legs = new Entity("legs");
    legs.addComponent("render", { type: "box", material: dark() });
    legs.setLocalScale(0.3, HIP, 0.22);
    legs.setLocalPosition(0, HIP / 2, 0);
    for (const part of [torso, head, gun, legs]) this.ent.addChild(part);
    this.renderers = [torso, head, gun, legs].map((e) => e.render);
    this.muzzleNode = gun;
    this.ent.setPosition(this.pos.x, this.pos.y, this.pos.z);
    app.root.addChild(this.ent);
    this.gunTip = new Vec3();
    this.prims = [torso, head, gun, legs];
    this.char = null;
    this._attachRig();
  }

  /**
   * Wear the same measured model the player does. Asynchronously, because
   * `spawnBot` runs inside a synchronous wave loop and the rig needs a couple of
   * frames to report its own bounding box; the primitives stay until it arrives,
   * so a viewer without the rig still shows a bot.
   */
  async _attachRig() {
    const rig = this.ctx.rig;
    if (!rig || !rig.resource) return;
    try {
      this.char = await makeCharacter(this.ctx.app, rig, {
        parent: this.ent, height: TARGET_HEIGHT,
        tint: SKINS[this.id % SKINS.length].suit,
      });
    } catch (e) {
      console.warn(`[bot${this.id}] rig failed, keeping primitives:`, e);
      return;
    }
    for (const p of this.prims) p.destroy();
    this.prims = [];
    this.renderers = this.char.renderers;
    this.mats = this.char.mats;
    // The hit zones (HIP/CHEST/HEAD above the feet) still describe the body: the
    // model is 1.75 m and the zones were set for one, so hitTest keeps working
    // without a per-bone hit map nobody has measured yet.
  }

  get chest() {
    return { x: this.pos.x, y: this.pos.y + CHEST, z: this.pos.z };
  }

  get eye() {
    return { x: this.pos.x, y: this.pos.y + EYE, z: this.pos.z };
  }

  get distSq() {
    const t = this.ctx.player();
    return (t.x - this.pos.x) ** 2 + (t.z - this.pos.z) ** 2;
  }

  /** Ray-sphere against the shared aim zones. Returns {t, part, mult, bot} for the nearest. */
  hitTest(o, d, maxDist) {
    if (!this.alive || !this.visible) return null;
    const hit = hitBody(this.pos.x, this.pos.y, this.pos.z, o, d, maxDist);
    return hit ? { ...hit, bot: this } : null;
  }

  damage(amount, part) {
    if (!this.alive) return;
    this.health -= amount;
    this.hitFlash = 1;
    this.sinceDamage = 0;
    this.awareness = 1;
    const p = this.ctx.player();
    this.lastKnown = { x: p.x, z: p.z, t: this.ctx.now() };
    if (this.health <= 0) {
      this.alive = false;
      this.state = STATE.DEAD;
      this.deadT = 0;
      this.ctx.onBotDeath?.(this, part);
    } else {
      this.ctx.onBotHurt?.(this, part);
    }
  }

  hear(pos, radius, kind) {
    const d = Math.hypot(pos.x - this.pos.x, pos.z - this.pos.z);
    if (d > radius) return false;
    const score = (kind === "gunshot" ? 1.4 : kind === "step" ? 0.7 : 0.5) * (1 - d / (radius + 1));
    if (!this.noise || score >= this.noise.score) {
      this.noise = { x: pos.x, z: pos.z, score, kind, t: this.ctx.now(), loudness: score };
    }
    if (kind === "gunshot") {
      this.awareness = Math.max(this.awareness, clamp(1.1 - d / radius, 0.35, 1));
      this.lastKnown = { x: pos.x, z: pos.z, t: this.ctx.now() };
      return true;
    }
    this.awareness = Math.max(this.awareness, clamp(score, 0, 0.7));
    return true;
  }

  /** Sight: distance, then cone, then a solid-geometry block test. */
  canSee(targetPos, eyePos) {
    const dx = targetPos.x - eyePos.x, dz = targetPos.z - eyePos.z;
    const dist = Math.hypot(dx, dz);
    if (dist > this.ctx.viewRange) return { see: false, dist };
    // Inside ~2 m of the muzzle the player is noticed regardless of where the bot faces.
    if (dist > 2.2 && angleTo(this.yaw, dx, dz) > this.ctx.viewCone * 0.5) {
      return { see: false, dist };
    }
    const blocked = this.ctx.sight.blocked(
      eyePos.x, eyePos.y, eyePos.z,
      targetPos.x, targetPos.y, targetPos.z,
    );
    return { see: !blocked, dist };
  }

  think(dt, now) {
    if (now < this.nextThink) return;
    this.nextThink = now + this.ctx.thinkInterval;
    const player = this.ctx.player();
    const sight = this.canSee({ x: player.x, y: player.y + 1.15, z: player.z }, this.eye);
    this.dist = sight.dist;
    if (sight.see) {
      const grew = now - this.lastSeen > 0.5;
      this.lastSeen = now;
      this.seenMs = grew ? 0 : this.seenMs + this.ctx.thinkInterval;
      this.lastKnown = { x: player.x, z: player.z, t: now, facing: player.yaw };
      this.awareness = clamp(this.awareness + (this.dist < 6 ? 1.6 : 0.9) * this.ctx.thinkInterval, 0, 1);
    } else {
      this.seenMs = 0;
      this.awareness = clamp(this.awareness - 0.16 * this.ctx.thinkInterval, 0, 1);
    }
    if (now - (this.noise?.t ?? -99) > 6) this.noise = null;
    this.decide(now);
  }

  /** Pick the state the whole squad behaviour is built on. */
  decide(now) {
    const fresh = now - this.lastSeen;
    if (this.state === STATE.DEAD) return;
    if (this.sinceOwnShot > 6 && this.state !== STATE.HOLD && this.state !== STATE.ENGAGE) {
      this.state = STATE.HOLD;
      this.assignSpot("hold");
    }
    if (this.state === STATE.RELOCATE && (!this.path.length || this.pathT > 4)) {
      this.state = this.lastKnown && fresh < 8 ? STATE.SEARCH : STATE.HOLD;
      if (this.state === STATE.HOLD) this.assignSpot("hold");
    }
    switch (this.state) {
      case STATE.HOLD:
        if (fresh < 1.2 && this.awareness > 0.35) {
          this.state = STATE.ENGAGE;
          this.nextShot = now + this.reactionFor(this.awareness);
        } else if (this.noise && this.noise.score > 0.32) {
          this.state = STATE.SEARCH;
          this.routeTo(this.noise.x, this.noise.z);
        }
        break;
      case STATE.ENGAGE: {
        if (this.dist < 2.6 && this.sinceOwnShot < 1.5) {
          this.state = STATE.RELOCATE;
          this.relocate();
          break;
        }
        if (fresh > 5.5) {
          this.state = this.awareness > 0.2 ? STATE.SEARCH : STATE.HOLD;
          if (this.state === STATE.SEARCH) this.routeTo(this.lastKnown?.x ?? this.pos.x, this.lastKnown?.z ?? this.pos.z);
        } else if (this.health < 34 && fresh > 1.5 && this.spotRole !== "hold") {
          this.state = STATE.RELOCATE;
          this.relocate();
        } else if (fresh > 0.9 && this.lastKnown && this.ctx.rng() < 0.5) {
          this.state = STATE.SEARCH;
          this.routeTo(this.lastKnown.x, this.lastKnown.z);
        }
        break;
      }
      case STATE.SEARCH:
        if (!this.path.length) {
          const cand = this.lastKnown || this.noise;
          if (!cand || now - cand.t > 12) {
            this.state = STATE.HOLD;
            this.assignSpot("hold");
          } else {
            this.routeTo(cand.x + (this.ctx.rng() - 0.5) * 5, cand.z + (this.ctx.rng() - 0.5) * 5);
            if (!this.path.length) this.state = STATE.HOLD;
          }
        } else if (fresh < 1.2 && this.awareness > 0.35) {
          this.state = STATE.ENGAGE;
          this.nextShot = now + this.reactionFor(this.awareness) * 0.6;
        }
        break;
      case STATE.FLANK:
        if (!this.path.length) {
          this.state = STATE.ENGAGE;
          this.nextShot = now + this.reactionFor(1) * 0.5;
        } else if (fresh > 8) {
          this.state = STATE.HOLD;
          this.assignSpot("hold");
        }
        break;
      default:
        break;
    }
  }

  /** Only these states mean "walk to where the player was last seen". */
  pursuesLastKnown() {
    return this.state === STATE.SEARCH || this.state === STATE.FLANK || this.state === STATE.RELOCATE;
  }

  reactionFor(awareness) {
    const base = 0.42 - 0.2 * this.skill;
    return Math.max(0.09, base * (1.35 - awareness));
  }

  /** Utility gate: is now the best moment to be seen firing? Returns 0..1. */
  shotUtility(now, player) {
    if (!this.lastKnown) return 0;
    if (now - this.lastSeen > 1.4) return 0;
    const dist = this.dist ?? Math.sqrt(this.distSq);
    const aim = clamp(1 - (dist - 3) / 30, 0.25, 1);
    // Winds up while it holds aim, so the first instant of contact is not the shot.
    const wound = clamp(0.35 + this.seenMs / 0.8, 0, 1);
    // How steady they hold still, not whether they shoot at all: this used to floor
    // at 0.55 against a 0.42 gate, and five bots fired ONE round in ninety seconds
    // at a player walking a circle. Movement now costs accuracy (see the aim error
    // in fireAttempt, which scales with speed) instead of buying invisibility.
    const steady = clamp(1 - (player.speed ?? 0) / 9, 0.8, 1);
    const exposure = STATE_EXPOSURE[this.state] ?? 0.5;
    const risk = 1 - exposure * 0.7;
    const surprise = this.stats.shots === 0 && (this.spotRole === "ambush" || this.state === "HOLD") ? 1.3 : 1;
    const aggression = 0.7 + 0.5 * this.skill;
    return clamp(aim * wound * steady * risk * surprise * aggression, 0, 1);
  }

  fireAttempt(now, player) {
    if (this.ctx.playerDown?.()) return;
    if (now < this.nextShot || this.burstGap > 0 || this.ammoLeft <= 0) return;
    const u = this.shotUtility(now, player);
    if (u < this.ctx.fireThreshold) {
      this.nextShot = now + 0.18;
      return;
    }
    const muzzle = this.muzzleWorld();
    // Aim at the body, not at the old `player.y + 1.05`: that offset was measured
    // from the player's capsule CENTRE, which puts it 15 cm over the head, so most
    // bot fire missed high and the error below never got to prove itself.
    // Perception still looks at a higher point, and that is right — a bot can see
    // over a seat back that will still stop its round.
    const target = this.ctx.playerChest();
    // Error in metres AT THE TARGET. Measured after the aim point moved onto the
    // body: the old band (1.2-30 cm) put 245 rounds out of 245 into a 26 cm radius
    // chest, and five bots killed a standing player in 0.7 s. So it grows with
    // range and with how fast the target is moving, which is the difference
    // between a firefight and an execution.
    const err = (0.06 + this.dist * 0.02) * (1 + (player.speed ?? 0) * 0.12) / this.skill;
    const aim = {
      x: target.x + (this.ctx.rng() - 0.5) * err * 2,
      y: target.y + (this.ctx.rng() - 0.5) * err * 1.4,
      z: target.z + (this.ctx.rng() - 0.5) * err * 2,
    };
    const dx = aim.x - muzzle.x, dy = aim.y - muzzle.y, dz = aim.z - muzzle.z;
    this.ctx.projectiles.spawn({
      from: muzzle,
      dir: { x: dx, y: dy, z: dz },
      range: this.ctx.roundRange,
      damage: 9 + 9 * this.skill,
      shooter: this,
      rgb: [0.55, 0.85, 1],
      targets: () => [this.ctx.playerTarget],
      onHit: (hit, point, dmg) => {
        this.stats.hits++;
        this.ctx.onBotHitPlayer?.(this, dmg, this.eye);
      },
    });
    this.ctx.audio.enemyShot(clamp(this.dist / 40, 0, 1));
    this.ctx.onNoise?.(this.eye, "gunshot");
    this.stats.shots++;
    this.ammoLeft--;
    this.burst++;
    this.sinceOwnShot = 0;
    this.wantYaw = angleYaw(dx, dz);
    if (this.burst >= 2 + Math.floor(this.ctx.rng() * 4)) {
      this.burst = 0;
      this.burstGap = 0.55 + this.ctx.rng() * (1.5 - this.skill);
      this.nextShot = now + this.burstGap;
    } else {
      this.nextShot = now + 0.11 + this.ctx.rng() * 0.08;
    }
    this.awareness = 1;
  }

  muzzleWorld() {
    if (this.char) {
      const p = this.char.muzzleWorld();
      this.gunTip.set(p.x, p.y, p.z);
      return { x: p.x, y: p.y, z: p.z };
    }
    const g = this.muzzleNode;
    const p = g.getPosition();
    const f = g.forward;
    this.gunTip.set(p.x + f.x * 0.42, p.y + f.y * 0.42 - 0.02, p.z + f.z * 0.42);
    return { x: this.gunTip.x, y: this.gunTip.y, z: this.gunTip.z };
  }

  assignSpot(role) {
    const pick = this.ctx.pickSpot(this, role);
    if (!pick) return;
    this.spot = pick;
    this.spotRole = role === "ambush" ? "ambush" : role;
    this.routeTo(pick.x, pick.z);
    if (this.state === STATE.HOLD) this.holdTimer = 0;
  }

  relocate() {
    this.assignSpot("relocate");
  }

  routeTo(x, z) {
    const to = this.ctx.clampToNav(x, z);
    const canWalk = (a, b) => this.ctx.walkable(a, b, this);
    const p = this.ctx.nav.findPath(this.pos.x, this.pos.z, to.x, to.z, { canWalk });
    this.path = p ? p.map((w) => ({ ...w })) : [];
    this.pathLen = this.path.length;
    this.pathT = 0;
  }

  move(dt) {
    if (!this.alive) return;
    const speed = this.ctx.moveSpeed * (this.state === STATE.SEARCH || this.state === STATE.FLANK ? 1 : this.state === STATE.RELOCATE ? 1.35 : 0.62);
    let moving = false;
    if (this.path.length) {
      const w = this.path[0];
      const dx = w.x - this.pos.x, dz = w.z - this.pos.z;
      const d = Math.hypot(dx, dz);
      if (d < 0.28) {
        this.path.shift();
        this.pathLen = this.path.length;
      } else {
        const step = Math.min(d, speed * dt);
        this.pos.x += (dx / d) * step;
        this.pos.z += (dz / d) * step;
        this.wantYaw = angleYaw(dx, dz);
        moving = true;
      }
    } else if (this.pursuesLastKnown() && this.lastKnown) {
      this.routeTo(this.lastKnown.x, this.lastKnown.z);
    }
    const want = this.ctx.groundAt(this.pos.x, this.pos.z, this.ctx.nav.heightAt(this.pos.x, this.pos.z, 1.6));
    if (want !== null && isFinite(want)) {
      this.pos.y += (want - this.pos.y) * (1 - Math.exp(-12 * dt));
    }
    this.pathT += dt;
    this.yaw = turnToward(this.yaw, this.facingTarget(moving), moving ? 6.5 : 3.2, dt);
    this.ent.setPosition(this.pos.x, this.pos.y, this.pos.z);
    this.ent.setLocalEulerAngles(0, this.yaw * 180 / Math.PI, 0);
    this.char?.setSpeed(moving ? this.ctx.moveSpeed : 0);
    this.sinceDamage += dt;
    this.sinceOwnShot += dt;
    this.burstGap = Math.max(0, this.burstGap - dt);
    if (this.ammoLeft <= 0 && this.burstGap <= 0) {
      this.ammoLeft = this.ctx.magSize;
      this.nextShot += 1.1;
    }
    if (this.state === STATE.HOLD) {
      this.holdTimer += dt;
      if (this.holdTimer > 14 && this.spotRole === "ambush") {
        this.holdTimer = 0;
        this.assignSpot("ambush");
      }
    }
  }

  facingTarget(moving) {
    if (!moving && this.lastKnown && this.ctx.now() - this.lastSeen < 9) {
      return angleYaw(this.lastKnown.x - this.pos.x, this.lastKnown.z - this.pos.z);
    }
    if (this.noise) return angleYaw(this.noise.x - this.pos.x, this.noise.z - this.pos.z);
    return this.wantYaw;
  }

  /** One world ray decides both "can the player see this bot" and its transparency. */
  updateVisibility(camPos) {
    const chest = { x: this.pos.x, y: this.pos.y + CHEST, z: this.pos.z };
    this.visible = !this.ctx.sight.blocked(camPos.x, camPos.y, camPos.z, chest.x, chest.y, chest.z, 0.25);
    for (const r of this.renderers) r.enabled = this.visible;
    if (this.hitFlash <= 0 && !this._flashed) return;
    this.hitFlash = Math.max(0, this.hitFlash - 0.08);
    this._flashed = this.hitFlash > 0;
    const k = 1 + this.hitFlash * 2.2;
    for (const { mat, base } of this.mats) {
      mat.diffuse.set(clamp(base[0] * k, 0, 1), clamp(base[1] * k * 0.7, 0, 1), clamp(base[2] * k * 0.7, 0, 1));
      mat.update();
    }
  }

  die(dt) {
    this.deadT += dt;
    const sink = clamp(this.deadT / 1.1, 0, 1);
    for (const r of this.renderers) r.enabled = sink < 0.95;
    this.ent.setLocalEulerAngles(sink * 82, this.yaw * 180 / Math.PI, 0);
    this.ent.setPosition(this.pos.x, this.pos.y - sink * 0.45, this.pos.z);
    if (sink >= 0.95) this.ent.enabled = false;
  }

  update(dt, now, player) {
    if (!this.alive) {
      this.die(dt);
      return;
    }
    this.think(dt, now);
    this.decideClock(dt);
    this.move(dt);
    if (this.state === STATE.ENGAGE || this.state === STATE.HOLD || this.state === STATE.FLANK) {
      this.fireAttempt(now, player);
    }
  }

  decideClock(dt) {
    if (this.state === STATE.RELOCATE && this.pathT > 5) this.path.length = 0;
  }

  snapshot() {
    return {
      id: this.id,
      state: this.state,
      awareness: this.awareness,
      dist: this.dist ?? Math.sqrt(this.distSq),
      path: this.pathLen ?? this.path.length,
      spot: this.spotRole ? `${this.spotRole}${this.spot ? ` x${this.spot.seenFrom}` : ""}` : null,
      seenMs: +this.seenMs.toFixed(2),
      lastKnownAge: this.lastKnown ? +(this.ctx.now() - this.lastKnown.t).toFixed(1) : null,
      shots: this.stats.shots,
      hits: this.stats.hits,
      visible: this.visible,
      health: Math.max(0, Math.round(this.health)),
      utility: +this.shotUtility(this.ctx.now(), this.ctx.player()).toFixed(2),
    };
  }

  destroy() {
    this.ent.destroy();
  }
}

export function angleYaw(dx, dz) {
  return Math.atan2(-dx, -dz);
}

export function angleTo(yaw, dx, dz) {
  let d = angleYaw(dx, dz) - yaw;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return Math.abs(d);
}

function turnToward(cur, want, maxStep, dt) {
  let d = want - cur;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  const step = maxStep * dt;
  return cur + clamp(d, -step, step);
}
