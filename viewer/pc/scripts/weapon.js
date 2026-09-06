/* Player weapon: spread, recoil, ammo, reload. The round itself is world.js's
   Projectiles — this class decides when a shot exists and where it is going. */

export class Weapon {
  constructor({
    projectiles, audio, camera,
    onHit, onNoise, hooks = {},
    rpm = 640, magSize = 30, reserveMax = 180, reloadTime = 1.7,
    damage = 22, range = 90, baseSpread = 0.0035, spreadPerShot = 0.0055,
    maxSpread = 0.05, spreadRecover = 0.055, recoil = 0.012,
  }) {
    this.projectiles = projectiles;
    this.audio = audio;
    this.camera = camera;
    this.onHit = onHit;
    this.onNoise = onNoise;
    this.hooks = hooks;
    this.config = { rpm, magSize, reserveMax, reloadTime, damage, range, baseSpread, spreadPerShot, maxSpread, spreadRecover, recoil };
    this.interval = 60 / rpm;
    this.ammo = magSize;
    this.reserve = reserveMax;
    this.spread = baseSpread;
    this.reloading = 0;
    this.lastShot = -1;
    this.shots = 0;
    this.kick = 0;
    /** Set by the manager: 1 = hip fire, < 1 tightens the cone while aiming. */
    this.ads = 1;
  }

  get reloadingNow() {
    return this.reloading > 0;
  }

  /** @returns {number} 1 fired, 0 blocked, -1 click on empty. */
  fire(now, targets) {
    if (this.reloadingNow) return 0;
    if (now - this.lastShot < this.interval) return 0;
    if (this.ammo <= 0) {
      this.audio.empty();
      return -1;
    }
    this.lastShot = now;
    this.ammo--;
    this.shots++;
    this.spread = Math.min(this.config.maxSpread, this.spread + this.config.spreadPerShot);

    const origin = this.camera.getPosition();
    const fwd = this.camera.forward;
    const dir = this._coneDirection(fwd);
    // From the barrel, not from the eye. A round that starts at the camera reads
    // as a beam coming out of the wall the muzzle is buried in, and it hands
    // escapeSolid a starting point inside the player's own collider.
    const muzzle = this.hooks.muzzle?.() ?? {
      x: origin.x + fwd.x * 0.5 + this.camera.right.x * 0.13,
      y: origin.y + fwd.y * 0.5 + this.camera.right.y * 0.13 - 0.14,
      z: origin.z + fwd.z * 0.5 + this.camera.right.z * 0.13,
    };
    this.projectiles.spawn({
      from: muzzle,
      dir,
      range: this.config.range,
      damage: this.config.damage,
      targets: () => targets,
      onHit: (hit, point, dmg) => this.onHit(hit.bot, dmg, hit.part, point),
    });
    // Own weapon, own ear: `near` used to scale with the distance the shot
    // travelled, which under a travelling round is not known when it is fired —
    // and was never physically true anyway.
    this.audio.playerShot(0);
    this.onNoise(origin, "gunshot");
    const kick = this.config.recoil * (1 + Math.min(1.4, this.kick) * 0.25);
    this.kick += kick;
    this.hooks.nudgePitch?.(kick);
    return 1;
  }

  _coneDirection(fwd) {
    const a = Math.random() * Math.PI * 2;
    const r = Math.sqrt(Math.random()) * this.spread * this.ads;
    const ca = Math.cos(a) * r, sa = Math.sin(a) * r;
    const len = Math.hypot(fwd.x, fwd.y, fwd.z) || 1;
    const fx = fwd.x / len, fy = fwd.y / len, fz = fwd.z / len;
    // up-ish basis that survives looking straight up/down
    let ux = 0, uy = 1, uz = 0;
    if (Math.abs(fy) > 0.99) { ux = 0; uy = 0; uz = 1; }
    let rx = uy * fz - uz * fy, ry = uz * fx - ux * fz, rz = ux * fy - uy * fx;
    const rl = Math.hypot(rx, ry, rz) || 1;
    rx /= rl; ry /= rl; rz /= rl;
    ux = fy * rz - fz * ry;
    uy = fz * rx - fx * rz;
    uz = fx * ry - fy * rx;
    const dx = fx + rx * ca + ux * sa;
    const dy = fy + ry * ca + uy * sa;
    const dz = fz + rz * ca + uz * sa;
    const dl = Math.hypot(dx, dy, dz) || 1;
    return { x: dx / dl, y: dy / dl, z: dz / dl };
  }

  reload() {
    if (this.reloadingNow || this.ammo === this.config.magSize || this.reserve <= 0) return false;
    this.reloading = this.config.reloadTime;
    this.audio.reloadStart();
    return true;
  }

  update(dt) {
    if (this.reloading > 0) {
      this.reloading -= dt;
      if (this.reloading <= 0) {
        const need = this.config.magSize - this.ammo;
        const take = Math.min(need, this.reserve);
        this.ammo += take;
        this.reserve -= take;
        this.reloading = 0;
        this.audio.reloadEnd();
      }
    }
    this.spread = Math.max(this.config.baseSpread, this.spread - this.config.spreadRecover * dt);
    if (this.kick > 1e-4) {
      const ease = this.kick * (1 - Math.exp(-7 * dt));
      this.kick -= ease;
      this.hooks.nudgePitch?.(-ease);
    }
  }

  get spreadDeg() {
    return (Math.atan(this.spread) * 180) / Math.PI;
  }
}
