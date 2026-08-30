# Why the temple world is bad — and what would fix it

**TL;DR:** Nothing is broken in the pipeline. The temple video was filmed from
*above a sea of clouds*, and the splat faithfully reconstructed the clouds as
solid geometry. The character now walks *inside that reconstructed cloud
volume*, so everything at eye level is white mush. More VRAM or better splat
quality cannot fix it — the fog is real content that was in the photos. What
fixes it: (a) a capture flown in clear air, or (b) an automated cloud-cull +
spawn-on-the-courtyard change, which is buildable but was not part of the
pipeline yet.

---

## 1. What the capture actually is

`videos/temple.mp4`: 18 s drone orbit, camera 22.5 m above the ground it filmed
(for comparison, rocks: 12 s, 12.2 m). The real footage
(`work/temple/frames_full/00040.jpg`) shows it plainly:

- A temple courtyard on a knife-edge hilltop, **rising out of an ocean of
  clouds**.
- The drone circles *above* the cloud layer. The hill pokes through; everything
  around and below the hill is white cloud.

## 2. Why the world is fog

3DGS reconstructs **everything that is consistently visible in the photos** —
including clouds. The clouds barely move during 18 s, so they are
multi-view-consistent content and get reconstructed as real geometry:

- **~36,000 gaussians (28% of the scene) fill the air** around the hill — small
  (median 5 cm), opaque (0.99), grey. They fill every cubic metre from the
  hillside up past the temple top, including the air over the courtyard itself
  (vertical profile is flat: ~3,000 gaussians per metre band, ground to +14 m).
- From the **training viewpoint** (drone looking down from above the clouds)
  the temple pixels win and the clouds sit behind — that's why `eval_renders/`
  and the bar-1 A/B stacks look fine.
- The **player stands inside the cloud volume** at 1.7 m eye height. Every view
  direction passes through dozens of cloud gaussians → the grey/white mush in
  `work/temple/walktest/frames/` and `work/temple/eye_*.png`.

## 3. The spawn made it worse

The route planner needs flat, connected, supported ground. On temple it found:

- median surface grade **39°** (rocks: gentle) — partly the real hillside,
  partly cloud-noise on the surface;
- only **6% of cells** under the 32° walkability cap;
- largest connected walkable region: **188 m²** (2% of the grid) — a hillside
  bench at −18 m, 30 m below the courtyard, **deep inside the cloud layer**;
- the walk loop degenerated to a **5 m circle, 2 waypoints** (rocks: 65 m,
  16 waypoints).

So the autopilot (and you) spawned into the single worst place in the scene:
inside the clouds, on a slope, 30 m from the interesting part.

## 4. "Do we need more / higher-quality splats?" — No

This is the important answer, so measured, not vibes:

| fact | number |
|---|---|
| splat count in temple scene | 128 k (cap is 350 k) |
| densification stopped at | 37% of the cap — the trainer **converged** |
| training VRAM usage | ~0.5–1 GB (your measurement) |
| near-ground gaussian size, temple vs rocks | median 0.10 m vs 0.10 m — **identical** |
| near-ground opacity, temple vs rocks | 0.93 vs 0.44 |

The splat is already a **faithful** reconstruction. The clouds are in the
photos, barely move, and are seen from thousands of rays — no trainer,
regardless of VRAM or steps, will make them vanish; a "higher quality" splat
would render **higher-fidelity fog**. VRAM headroom is not the bottleneck;
the content of the capture is.

Same pipeline, same code path, two outcomes:

| | rocks (good) | temple (bad) |
|---|---|---|
| air between camera and ground | empty | **cloud sea** |
| camera height above ground | 12.2 m | 22.5 m |
| terrain | bare, gentle slopes | knife-edge hill, 39° median grade |
| walkable area found | 65 m loop, 16 waypoints | 188 m² bench, 2 waypoints |
| eye-level view | grass and rock | **inside a cloud** |

## 5. Can the fix be automated?

**Partly — yes, and it's designed, just not built yet.** Two changes:

1. **Cloud cull** (new `strip_clouds.py`, sibling of `strip_sky.py`).
   Cloud gaussians have a measurable signature: desaturated (saturation
   p50 = 0.04), airborne (> 1 m above the local ground surface), inside the
   footprint. Measured on temple: cutting `sat < 0.20 AND height > ground+1 m`
   removes the fog fill while keeping the pavement (below the height cut),
   the trees and wooden temple (saturated), and the distant backdrop.
   One trap: the temple's own grey stone walls are also desaturated and
   airborne. Fix: auto-detect compact tall structures (gaussians > 3 m above
   the highest ground, footprint < 10% of the grid → the temple detects as a
   9×19 m box, 4.1% of grid) and protect that box from the cull. Gate the whole
   thing on population size (cloud fill is ~30% of the scene; a normal scene is
   ~5%) so it never fires on rocks.
2. **Spawn and route on the best surface, not the largest.** Prefer the
   highest-supported flat region (the courtyard at +2…+6 m) over the largest
   low one (the cloud bench at −18 m).

After both, temple becomes: walking a clear paved courtyard around a temple,
clouds visible below and around — which is what that place actually looks like.
What automation **cannot** fix: the hillside below the clouds stays mushy (it
was never photographed clearly), and the horizon stays cloudy (it really is
cloudy).

## 6. If you'd rather re-capture than re-code

The recipe that made rocks good, for the next flight:

1. **Clear air.** No cloud sea, no fog, no haze — the splat reconstructs all of it.
2. **Fly low and close** to what the player should walk on — 10–15 m, not 20+.
3. **Slow orbit or slow pass**, 60–70% forward overlap between frames.
4. **Gentle terrain** in frame; the walkable surface needs slopes under ~30°.
5. 60 s+ of footage if possible (temple's 114 keyframes is thin; rocks' was too).

## Evidence

- Real footage: `work/temple/frames_full/00040.jpg` (cloud sea, clear temple)
- Splat from training poses (looks fine): `work/temple/eval_renders/`
- Eye level at spawn (fog): `work/temple/eye.png`, `walktest/frames/`
- Eye level on the courtyard (fog too): `work/temple/eye_court_e.png`
- Route planner output (5 m loop): `work/temple/viewer_assets/collision.json`
- World gate (passes — the gate measures up-ness and standability, not fog):
  `check_world.py --asset work/temple/viewer_assets --work work/temple`
