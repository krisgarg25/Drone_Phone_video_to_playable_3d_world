/* One skinned glTF person, used for the player and for every bot.

Two defects this removes rather than restyles:

* The avatar used to float. The old body primitive was a capsule centred 0.78 m
  ABOVE the entity origin with a half height of 0.45 m, while the physics capsule
  is centred ON that origin and stands 0.90 m below it - so the figure's lowest
  point was 0.90 + 0.78 - 0.45 = 1.23 m above the floor it collided with, and its
  head sat at 2.13 m. Height and foot placement are measured from the model here
  and tied to the collider, so "too big" and "floating" cannot come back
  independently, and neither number is a constant anybody has to remember.
* Height comes from the model's own bounding box, never from a typed constant.
  cesium_man.glb is 1.507 m tall as authored; scaling that to a stated target is
  the only way this stays right when the asset is swapped, which is the whole
  reason for taking the Quaternius/CC0 route instead of drawing primitives.

The walk clip is the only animation the file carries, so standing still means the
clip is frozen mid-stride (speed 0). That is stiff and it is honest; an idle
animation is a separate asset, not something to fake.
*/
import { Color, Entity, StandardMaterial, Vec3 } from "playcanvas";

/** Metres of a real person this rig is scaled to. */
export const TARGET_HEIGHT = 1.75;

/**
 * Yaw to give the model so it looks where its parent is heading. Nothing in a
 * glTF names a forward axis, so this is read off a screenshot, and it belongs to
 * the FILE: the player and the bots rotate their own entity the same way, so one
 * value has to serve both. If a figure ever walks backwards or moonwalks in
 * place, this is the number to flip.
 */
export const RIG_FACING = 180;

/**
 * Ground speed, in m/s, at which the walk clip's feet stop skating on the floor.
 * A glTF animation carries no speed metadata, so this is calibrated by eye.
 * Kept here rather than at the callers so nobody has to agree on it twice.
 */
export const STRIDE_SPEED = 1.4;

/** Where the weapon sits, in metres IN THE BODY'S FRAME: out to the right, a
 *  fraction of the figure's height up from the soles, out in front (negative). */
const RIFLE_OFFSET = [0.22, 0.55, -0.30];

/**
 * The hit model: three spheres on a TARGET_HEIGHT figure, measured UP FROM ITS
 * SOLES, with what a round landing in each is worth. One table for the player and
 * the bots, because the two used to disagree about where a head actually is.
 * There is no limb entry: nothing in the viewer can hit an arm without hitting
 * the chest sphere behind it first.
 */
export const AIM = {
  head: { y: 1.6, r: 0.15, mult: 3.2 },
  chest: { y: 1.22, r: 0.26, mult: 1.0 },
  hip: { y: 0.86, r: 0.24, mult: 0.85 },
};

/** Where a figure looks out from, and where a primitive gun sits, up from the soles. */
export const EYE_Y = 1.58;
export const GUN_Y = 1.34;

/**
 * Ray-sphere against the three zones of a figure standing at (x, z) with its feet
 * at `feetY`. `d` must be unit length. Returns the nearest zone as {t, part, mult}.
 */
export function hitBody(x, feetY, z, o, d, maxDist) {
  let best = null;
  for (const [part, zone] of Object.entries(AIM)) {
    const ox = o.x - x, oy = o.y - (feetY + zone.y), oz = o.z - z;
    const b = ox * d.x + oy * d.y + oz * d.z;
    const c = ox * ox + oy * oy + oz * oz - zone.r * zone.r;
    const disc = b * b - c;
    if (disc < 0) continue;
    // The entry root, or 0 when the segment starts inside the volume — point blank
    // is a hit. The hitscan version rejected everything under 12 cm to keep an
    // eye-level ray off the shooter, and against a round that closes in steps that
    // band let a target walk through: each step's entry distance shrinks by the
    // step length, so it can fall inside the band on every single probe.
    const t = c < 0 ? 0 : -b - Math.sqrt(disc);
    if (t < 0 || t > maxDist) continue;
    if (!best || t < best.t) best = { t, part, mult: zone.mult };
  }
  return best;
}

/**
 * The player as something a round can hit.
 *
 * `read()` returns {x, y, z, feet}, where `feet` is how far the entity origin sits
 * ABOVE the floor the capsule stands on — for the player that is the physics
 * measurement (~0.90 m), not zero, because its origin is the centre of its
 * collider while a bot's is its sole. Getting that difference wrong is why bots
 * used to aim at `y + 1.05`: measured up from the capsule centre that lands 15 cm
 * ABOVE the head, so their shots missed far more often than their error model said.
 */
export function personTarget(read) {
  return {
    hitTest(o, d, maxDist) {
      const p = read();
      return hitBody(p.x, p.y - p.feet, p.z, o, d, maxDist);
    },
  };
}

/**
 * Wait for `n` animation frames, or `ms`, whichever comes first; true means the
 * frames arrived. Off the application's own event because the caller that needs
 * it most - a bot - is handed a context object that mirrors the app's properties
 * but not its prototype, so it has no `on`. The deadline matters either way:
 * this is awaited on the boot path, and a pump that never ticks would otherwise
 * leave the viewer sitting on its loading screen.
 */
function nextFrames(n, ms = 2000) {
  return new Promise((resolve) => {
    let left = n;
    const t0 = performance.now();
    const tick = () => {
      if (--left <= 0) resolve(true);
      else if (performance.now() - t0 > ms) resolve(false);
      else requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}

/** Every mesh instance under `root`, from whichever component API holds them. */
function meshListOf(root) {
  // The animation component needs a MODEL component (see makeCharacter), so this
  // instance carries `model`, not `render`. Read either, because nothing about
  // measuring a box should depend on which component API produced it.
  const out = [];
  for (const type of ["model", "render"])
    for (const c of root.findComponents(type)) out.push(...c.meshInstances);
  return out;
}

/** Union of every mesh instance's world AABB, as {min, max}. */
function worldBounds(root) {
  let lo = null, hi = null;
  for (const mi of meshListOf(root)) {
    const a = mi.aabb;
    if (!a || !isFinite(a.halfExtents.x)) continue;
    const c = a.center, h = a.halfExtents;
    lo = lo ? [Math.min(lo[0], c.x - h.x), Math.min(lo[1], c.y - h.y), Math.min(lo[2], c.z - h.z)]
            : [c.x - h.x, c.y - h.y, c.z - h.z];
    hi = hi ? [Math.max(hi[0], c.x + h.x), Math.max(hi[1], c.y + h.y), Math.max(hi[2], c.z + h.z)]
            : [c.x + h.x, c.y + h.y, c.z + h.z];
  }
  return lo ? { min: lo, max: hi } : null;
}

/** A rifle: built once, shared by the player and the bots, with a node at the muzzle. */
export function makeRifle(darkMat) {
  const g = new Entity("rifle");
  const add = (name, sx, sy, sz, px, py, pz, mat) => {
    const e = new Entity(name);
    e.addComponent("render", { type: "box", material: mat });
    e.setLocalScale(sx, sy, sz);
    e.setLocalPosition(px, py, pz);
    g.addChild(e);
    return e;
  };
  add("receiver", 0.07, 0.11, 0.42, 0, 0, -0.10, darkMat);
  add("barrel", 0.045, 0.045, 0.34, 0, 0.01, -0.44, darkMat);
  add("magazine", 0.05, 0.16, 0.09, 0, -0.12, -0.06, darkMat);
  add("stock", 0.06, 0.09, 0.24, 0, -0.01, 0.22, darkMat);
  const sight = add("sight", 0.03, 0.05, 0.10, 0, 0.09, -0.16, darkMat);
  const tip = new Entity("muzzle");
  tip.setLocalPosition(0, 0.01, -0.62);
  g.addChild(tip);
  g.sight = sight;
  g.tip = tip;
  return g;
}

/**
 * Instantiate the rig under `parent`, scaled to `height` with its feet on the
 * parent's local y=0 — which for the player is the BOTTOM of its collision
 * capsule, and for a bot is the ground it stands on.
 *
 * Returns { root, inst, rifle, muzzle, setSpeed, measured } once the model has
 * been rendered twice: its bounding box only exists after the skinning matrices
 * are built, and the scale is derived from that box.
 */
export async function makeCharacter(app, rigAsset, opts = {}) {
  const { parent, height = TARGET_HEIGHT, tint = null, rifle = true,
          facing = RIG_FACING } = opts;
  const inst = rigAsset.resource.instantiateRenderEntity();
  const root = new Entity("character");
  root.addChild(inst);

  // Measure at the world origin with an identity transform: mi.aabb is a WORLD
  // box, so that is the only pose in which it reads as the model's own bounds.
  app.root.addChild(root);
  root.syncHierarchy();
  const pumped = await nextFrames(2);
  let box = worldBounds(root);
  // A skinned mesh with no bone bounds yet reports nothing; one more frame of
  // rendering is what fills skinInstance.matrices in.
  if (!box || box.max[1] - box.min[1] < 1e-3) {
    await nextFrames(3);
    box = worldBounds(root);
  }
  if (!box) {
    // Never throw from here: this is awaited while the page boots, and a figure
    // at the wrong size is a visible, testable failure while an unbootable viewer
    // is not. The height assertion catches it.
    console.warn(`[character] no render bounds after ${(pumped ? "frames" : "a 2 s deadline")}` +
                 ` — drawing at the model's own size`);
    box = { min: [0, 0, 0], max: [0, height, 0] };
  }
  const authored = box.max[1] - box.min[1];
  const scale = height / authored;
  const feetY = box.min[1];

  // addChild detaches from the previous parent (_prepareInsertChild -> remove),
  // which is how this build reparents: GraphNode has no `parent` setter and no
  // setParent, and finding that out cost a boot failure.
  parent.addChild(root);
  root.setLocalScale(scale, scale, scale);
  root.setLocalEulerAngles(0, facing, 0);
  // feetY is in the MODEL's units, so the push-down carries the root's scale.
  // seat() is a method because the player's parent does not know where its
  // collider's floor actually is until the physics has settled under it.
  const seat = (y = 0) => root.setLocalPosition(0, y - feetY * scale, 0);
  seat();

  // The weapon. Mounted on the BODY, not on the hand bone: a joint's world matrix
  // is not resolved during the boot this runs from (the skin has not been posed
  // yet), so an offset solved against it put the muzzle at the figure's ankles,
  // and a hand mount would also have to fight the arm's own swing every frame.
  // On the body it rides the walk bob and turns with the character, which is the
  // most this viewer can claim without a pose the rig does not have.
  let rifleEnt = null, muzzle = null;
  if (rifle) {
    const dark = new StandardMaterial();
    dark.useLighting = false;
    dark.diffuse = new Color(0.13, 0.13, 0.15);
    dark.update();
    rifleEnt = makeRifle(dark);
    root.addChild(rifleEnt);
    // root's space is the MODEL's units: divide the metre offsets by the fit-to-
    // height scale, and measure height up from the soles, which sit at feetY.
    rifleEnt.setLocalScale(1 / scale, 1 / scale, 1 / scale);
    // root is turned by `facing` relative to the body's frame, so undo that turn
    // here or the weapon ends up behind the figure's back (measured: 0.92 m of it).
    const a = -facing * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
    rifleEnt.setLocalPosition(
      (RIFLE_OFFSET[0] * ca + RIFLE_OFFSET[2] * sa) / scale,
      feetY + height * RIFLE_OFFSET[1],
      (-RIFLE_OFFSET[0] * sa + RIFLE_OFFSET[2] * ca) / scale);
    rifleEnt.setLocalEulerAngles(0, -facing, 0);
    muzzle = rifleEnt.tip;
  }

  // The MODERN `anim` component, not the legacy `animation` one. Two dead ends
  // measured on the way here: `animation` on a render entity resolves its graph
  // through entity.model, finds nothing, and reports a clip as attached and
  // playing while currentTime sits at 0 forever; on a model entity it builds a
  // legacy Skeleton for a glTF AnimTrack and dies inside addTime reading
  // node._keys. Both look fine on an unposed figure and both passed every check
  // except the one that watches a bone move, so the clip is now attached through
  // the modern component with an explicitly built one-state graph - the smallest
  // thing this viewer can drive a single clip with.
  let anim = null;
  const track = (rigAsset.resource.animations || []).map((a) => a.resource).find(Boolean);
  if (track) {
    try {
      inst.addComponent("anim", { activate: false });
      // The library's own one-state setup: it builds the START -> walk graph and
      // marks the state default, which a hand-built graph got wrong (the layer
      // then had no current state and evaluated nothing at any dt).
      inst.anim.addAnimationState("walk", track, 1, true);
      inst.anim.activate = true;
      // TWO flags, both required, neither set by the paths above: the system's
      // update loop skips any component whose `playing` is false, and the layer
      // has its own controller state. With only the first one set, everything
      // reported healthy - clip attached, component playing, speed following the
      // body - while the bones never moved. The check that caught it watches a
      // knee's position relative to the hips, because a world-space bone moves
      // even when the pose is frozen.
      inst.anim.playing = true;
      const layer = inst.anim.findAnimationLayer("Base");
      if (layer && !layer.playing) layer.play();
      anim = inst.anim;
    } catch (e) {
      console.warn(`[character] the anim component rejected the clip: ${e && e.message}`);
    }
  } else {
    console.warn("[character] the rig carries no clip; it will stand in bind pose");
  }

  // The rig's one material is SHARED by every instance PlayCanvas spawns from it,
  // so tinting means cloning first. Multiplying the diffuse keeps the authored
  // texture working underneath - a flat colour threw away the only art in the
  // file, and the bots still read as teams because diffuse multiplies the map.
  const mats = [];
  for (const mi of meshListOf(inst)) {
    const m = mi.material.clone();
    m.useLighting = false;        // the viewer has no lights; the splat is self-lit
    const base = [m.diffuse.r, m.diffuse.g, m.diffuse.b];
    if (tint) m.diffuse.set(base[0] * tint[0], base[1] * tint[1], base[2] * tint[2]);
    m.update();
    mi.material = m;
    mats.push({ mat: m, base });
  }

  // The rifle goes on BEFORE the clip starts: its placement is read off the hand
  // joint's transform, and that has to be the bind pose rather than whichever
  // frame of the stride the figure gets caught in.
  return {
    root, inst, rifle: rifleEnt, muzzle, mats, seat,
    clip: track ? (track.name || "walk") : null,
    renderers: [...inst.findComponents("model"), ...inst.findComponents("render")],
    measured: { authoredHeight: authored, scale, feetY, height },
    /** Pass the body's ground speed in m/s; the clip's own pace is internal. */
    /** Ground speed in m/s, normalised so a stride lands with the foot. */
    setSpeed(ms) { if (anim) anim.speed = ms / STRIDE_SPEED; },
    animTime() { return anim ? anim.currentTime : null; },
    /**
     * World point the barrel starts from. GraphNode.getPosition() hands back a
     * shared internal Vec3 that the next call overwrites, so copy it out.
     */
    muzzleWorld(out = new Vec3()) {
      const p = muzzle ? muzzle.getPosition() : parent.getPosition();
      return out.set(p.x, p.y, p.z);
    },
  };
}
