# README-MVP — Drone video → 3D Gaussian Splat → walkable scene

**Status: both quality bars WON under blind review.**
Bar 1 (visual): fresh-context critic shown only blinded A/B stacks — WIN, 10/10
pairs read as the same scene, zero disqualifying artifacts
(`results/verdicts/critic_bar1_visual_v2.md`).
Bar 2 (walkability): fresh-context critic shown only unlabeled walk frames +
the raw log — WIN, 65 m traversed on the physics collider, 0 falls, no
teleports (`results/verdicts/critic_bar2_walkability.md`).

## V2 — multi-clip scenes, presets + smart defaults, phone AR pose priors

The pipeline is now generic over *how you captured* the scene:

```bat
python pipeline.py doctor                REM toolchain health check
python pipeline.py scan room             REM diagnose footage BEFORE burning GPU hours:
                                        REM   blur / texture / rotation-vs-translation verdicts
python pipeline.py capture room          REM print the capture checklist for a preset

REM any number of clips: drop them all in videos\room\
python pipeline.py run room              REM videos\room\*.mp4 (or legacy videos\room.mp4)
python pipeline.py run room --preset auto          REM diagnose footage -> auto-tune params
python pipeline.py run room --video a.mp4 b.mp4    REM explicit clips
```

Three layers, later layers win: **presets** (`room` / `drone` / `object` /
`auto`) set SfM + keyframe budgets per capture style; **smart rules** adjust
parameters from the scan diagnostics (blur, weak geometry); **CLI flags**
(`--target --width --steps --cap --overlap --prior-std ...`) override anything.

Multi-clip handling follows COLMAP's own multi-video recipe: frames land in
per-clip subfolders so sequential matching stays contiguous within each clip,
and cross-clip links come from whichever mechanism is available — `spatial`
matching when AR pose priors exist, vocab-tree retrieval when
`tools/vocab_tree.bin` is present (see `pipeline.py doctor`), exhaustive
matching for small sets.

### Phone AR poses (the big one)

COLMAP fails on rotation-heavy handheld video because spinning in place has no
parallax. If your phone logs ARCore/ARKit camera positions while you record,
those become **native COLMAP pose priors**: the mapper registers even
weak-parallax segments and metric scale survives into the final scene.

1. Record as usual; run any ARCore/ARKit logger app alongside
   (Record3D on iPhone works too — see `docs/PHONE_CAPTURE.md`, plus an
   experimental zero-install `tools/webxr_capture.html` recorder).
2. Drop each pose log next to its clip: `videos/room/walk1.mp4` +
   `videos/room/walk1_poses.jsonl`.
3. Run normally. The new `priors` step injects positions into the COLMAP
   database and mapping switches to `pose_prior_mapper`.

Verified end-to-end (`tests/selftest_priors.py`): a 12 m synthetic AR walk
reconstructs at scale **1.0000** with max camera-center deviation **7 mm**;
30/30 frames register. Capture rules that still matter are in
`docs/PHONE_CAPTURE.md` — arcs not spins, three heights, slow and steady.

One command, video file to walkable splat scene:

```bat
mvp.bat run rocks          REM videos\rocks.mp4 -> walkable scene + evidence
mvp.bat run rocks --quality standard    REM the old 640px/350k settings
mvp.bat status rocks       REM per-step done/stale state
mvp.bat view rocks         REM serve + open the walkable viewer in a browser
```

`mvp.bat` is a shim over `pipeline.py`, which owns the step order and cannot
skip a step silently: every step logs to `work\<name>\logs\`, completed steps
are marked in `work\<name>\.pipeline\`, and re-running a step invalidates
everything downstream of it. Resume with `--from <step>`, force with `--fresh`
or `--only <step,step>`; override any quality knob with `--steps/--cap/--width/--target`.

To browse without mvp.bat, serve the repo root over HTTP with the included
MIME-correct server and open the PlayCanvas viewer:
`.venv\Scripts\python.exe _serve.py 8000 .` → http://localhost:8000/viewer/pc.html?asset=/work/rocks/viewer_assets
(append `&auto=1` for the autopilot walk test). Stock `python -m http.server`
does NOT work — it serves `.mjs`/`.wasm` as `text/plain`.

## Hardware target

RTX 3050 Laptop, 6 GB VRAM, Windows 11, 16 GB RAM. Everything here was chosen to
train and run **on this machine** — no cloud, no compile-from-source.

## Quality presets

Measured on this GPU: the old settings peaked at **0.72 GiB** — over 5 GB idle.
Presets (`--quality`), with the `high` preset measured on temple:

| preset | keyframes | train width | steps | gaussian cap | peak VRAM | notes |
|---|---|---|---|---|---|---|
| `standard` | 350 | 640 | 12 000 | 350 k | ~0.7 GB | the original settings |
| `high` (default) | 500 | 800 | 15 000 | 1 500 k | **1.26 GB**, 14 min | 272 frames registered (was 114), 451 k gaussians (was 128 k) |

The cap is headroom, not a target: densification converges where the scene runs
out of detail (temple converged at 451 k under a 1.5 M cap). More VRAM does not
fix bad content — see PROBLEM-temple.md for the capture that no amount of
compute can improve.

## Pipeline

| Step | What | How |
|---|---|---|
| 1 | Frame extraction + sharpness-filtered keyframes | `scripts/extract_keyframes.py` — OpenCV decode, variance-of-Laplacian score, temporal-bin selection, near-duplicate rejection. Full-res copies kept for the visual bar; 640 px copies for SfM/training. |
| 2 | Camera poses (no GPS) | COLMAP 4.1.1 (official Windows CUDA build): GPU SIFT `feature_extractor` → `sequential_matcher` (overlap 20, quadratic) → `mapper` → TXT export. |
| 3 | Splat training | `scripts/train_splat.py` — custom lean trainer on **gsplat 1.5.3** (prebuilt Windows wheel, zero compilation): SH degree 3, pure-torch SSIM+L1 loss, DefaultStrategy densification with a hard 350 k splat cap, packed rendering, batch=1. |
| 4 | Visual evidence | `scripts/render_evals_offline.py` renders the trained PLY from the exact training-camera poses with gsplat (offline, no browser), composited into blind A/B stacks against the real frames (`make_pairs.py`). |
| 5 | World frame | `scripts/solve_frame.py` — gravity comes from the **drone's gimbal**, and metric scale from clip duration; both land in `frame.json`, which is now the only source of either. The old path took "up" from a RANSAC plane normal, whose sign is arbitrary: it came out inverted and the entire scene was exported upside down. |
| 6 | Heightfield + colours | `scripts/export_viewer_assets.py` reorients and rescales, then rasterizes splat means into a ground heightfield (low-percentile Y per cell, holes diffused) plus `coverage.u8` marking the ~44% of cells with real multi-view support. `export_ground_colors.py` vertex-colours that grid from the splat's own colours. |
| 7 | Canopy strip | `scripts/strip_sky.py` — drone footage contains sky and haze, and 3DGS reconstructs both as gaussians. Over the walkable footprint they form a separate layer (band density falls 2159/m at +4 m to 135/m at +12 m, then recovers to 1727/m by +19 m), so the script cuts in the gap. The test is **bimodality, not sparseness**, and only gaussians over the footprint are judged: an earlier "is this band sparse?" version fired on a smooth decline and would have deleted 86% of the painted frame, backdrop and all. |
| 7b | Cloud-sea cull | `scripts/strip_clouds.py` — for footage flown ABOVE a cloud layer the fog is multi-view-consistent content and reconstructs as ~30% of the scene (temple: 76 804 fog candidates, saturation p50 0.065, all >1 m above local ground). Cuts `sat < 0.20 AND airborne`, after auto-protecting compact tall structures (column-density clustering + area test: the temple reads as a 470-column cluster, 4.8% of the grid) so grey stone walls survive. Gated on **painted-area fraction** (fires at ≥5%; temple measures 18.6%, a clear-air capture 0.8%) so it never fires on good scenes — rocks passes through untouched. Idempotent via the same `scene.full.ply` convention as strip_sky, so both cuts compose. |
| 7c | Re-export from culled scene | `export_viewer_assets.py --from-scene` — the heightfield, coverage and collider box were computed from the pre-cull splat, and the cull changed what "ground" is (cloud tops were defining half the far-field grid). This regenerates every derived asset from the culled `scene.ply`, leaving coordinates/quats/scales untouched (they are already world-frame; the region test inverts the transform for its COLMAP-frame check only). Skipping this step is measurable: collider vs heightfield disagreed by a median of **28 m** over the temple plateau, and the world gate's floor/ceiling assertion failed at 18.3% before it and passes at 12.2% after. |
| 8 | Collision mesh | `scripts/build_collider.py` drives `@playcanvas/splat-transform` (`cluster_shell`, 0.35 m voxels, box + seed from `collision.json`), then two mandatory post-passes. `clip_collider.py` cuts airborne crusts **relative to the heightfield** — a flat box ceiling cannot, because the terrain's own relief overlaps the crust's height range. `ground_mesh.py` then converts the clipped shell's top face into a filtered heightfield mesh: the voxeliser emits axis-aligned faces, so every height change is a vertical **wall** (measured: 1.05 m walls every 0.6 m along the route), and a capsule of radius 0.34 cannot climb a 0.35 m riser at all — it meets the flat face at its equator, so the contact normal is horizontal and there is no lift. 14.5 k tris of slopes replace 121 k tris of walls. |
| 9 | Walk loop + spawn from the mesh itself | `scripts/walk_path_from_glb.py --smooth 1 --pick best` rasterizes the collider's top surface, keeps cells under a 32° grade, and picks a connected region: `--pick best` prefers the **highest** region at least 15% the size of the largest (so a low cloud bench cannot outvote the courtyard on the hill — temple's bench sat at −18 m while the courtyard was at +2…+6 m); `--pick largest` restores the old rule. It then string-pulls a corridor through the clearance transform, writing both loop and spawn into `collision.json`. `--smooth 1` is load-bearing: the collider **is** the filtered heightfield, so smoothing again would plan on a surface the physics does not have — the mismatch that had the autopilot steering into walls its map said were flat. |

Measured end-to-end on temple with the new pipeline (`high` preset): the spawn
moved from the −18 m cloud bench to the **courtyard at +8.8 m**, the world gate
passes all assertions, and the autopilot walk test reports **65.4 m, 0 falls**
(the pre-fix world managed a 5 m waypoint circle inside fog). Eye-level evidence:
`work/temple/eye.png` (before, fog mush) vs `work/temple/eye_final_court.png`
and `work/temple/walktest/frames/` (after, terrain under open sky — sparse in
places the drone never saw up close, which is a capture limit, not a pipeline
bug).
| 10 | Gate | `scripts/check_world.py` — 8 assertions that fail the build: heightfield coverage, terrain-is-a-floor-not-a-ceiling (an inverted scene drives this to 100%), cameras above the ground they filmed, spawn supported and flat, collider has no ceiling slab, collider tracks the ground, collider exists at the spawn. |
| 11 | Visual evidence | `scripts/render_evals_offline.py` renders the trained PLY from the exact training-camera poses with gsplat (offline, no browser), composited into blind A/B stacks against the real frames (`make_pairs.py`). |
| 12 | Walkable viewer | `viewer/pc.html` — PlayCanvas engine + ammo.js: the collision GLB loads as static trimesh bodies, the character is a dynamic capsule rigidbody, so gravity, sliding and blocking come from the engine. Every downward probe is **bounded to the ground and masked to static bodies** — `raycastFirst` from the sky latches onto whatever crust survives. Autopilot settles, walks 50 m, spins 720°, walks back (`drive_viewer.py walk`). |

## Exact commands (what pipeline.py runs)

Do not hand-type these — the runner owns the order and the resume markers:

```bat
mvp.bat run rocks
python pipeline.py run temple --from route      REM redo route + everything after
python pipeline.py status rocks
```

Step order (15 steps): keyframes -> colmap -> poses -> train -> frame -> export
-> sky -> clouds -> reexport -> colors -> collider -> route -> gate -> evals
-> pairs -> walktest. COLMAP is driven by `scripts/run_colmap.py` (the `.bat`
delegates to it; Python argv lists survive the space in this repo's path, cmd
quoting did not).

`build_collider.py` calls `splat-transform` itself rather than the README carrying
the command, because the flags are not in the coordinates you would expect. The
tool works in an **engine space equal to `(-x, -y, +z)` of the input PLY** — `-B`,
`--seed-pos`, the output GLB and `sceneBounds` all agree on it — so handing `-B`
the true content bounds of a +Y-up PLY rejects every gaussian, and
`--voxel-floor-fill` marches *downward* through the scene, filling the sky. The
script hands the tool a source pre-flipped 180° about Z so all three coincide with
scene coordinates. Also: `-w` is `--overwrite`, the output is the trailing
positional, and a `.voxel.json` extension is what requests voxelisation.

Diagnostic flags: `build_collider.py --compare` tabulates the voxeliser variants
(raw, pre-clip); `--no-clip` skips both post-passes. Viewer URL params:
`underlay=1` restores the ground sheet (off by default — see below), `sink=<m>`
sets how far under the surface it hangs, `auto=1` starts the autopilot,
`shoot=1` freezes the follow camera for scripted views.

Viewer controls (browser): click canvas once for mouse-look · W/A/S/D walk ·
Shift run · Space jump · C first/third person · T autopilot · R respawn.
Serve with `mvp.bat view <name>` or `.venv\Scripts\python.exe _serve.py 8000 .`
(stock `http.server` serves `.mjs`/`.wasm` as `text/plain`, which breaks the page).

## Setup (one-time)

- ffmpeg static build → `tools/ffmpeg/`
- COLMAP 4.1.1 CUDA → `tools/colmap/`
- `.venv` (Python 3.12): numpy, opencv-python, pillow, plyfile, playwright (+ `playwright install chromium`)
- `.venv310` (Python 3.10, for the gsplat wheel): torch 2.4.1+cu124, gsplat 1.5.3+pt24cu124 prebuilt wheel, plyfile
- `viewer/pc/`: PlayCanvas engine (`playcanvas.mjs`) + ammo.js wasm, vendored — no build step

## Known weak spots

- **Scale is not measured**: no GPS. `solve_frame.py` derives it from clip duration and gimbal telemetry, so "metres" are consistent and plausible rather than surveyed.
- **Grass is low-texture** — expect SfM to lean on the rock formation; far field may be soft/floaty (the bar-1 critic noted background smear as the main cosmetic flaw).
- **The collider is a heightfield**, so overhangs and caves cannot be expressed — a voxel shell can express them and is unwalkable, which is the trade `ground_mesh.py` makes deliberately. This capture has none worth keeping. 37% of cells have no measured shell surface and are dilated in from their neighbours, with the exported heightfield as the last resort.
- **The ground underlay is off by default.** It only ever emitted quads where all four corners already had gaussian support, so it never filled a gap — it just sheeted opaque colour over ground the splat renders, and at 6 cm under the surface (well inside the splat's own ±0.5 m scatter) the scene read as flat olive mud. Sunk 0.7 m the vegetation returns but the sheet's cut edge shows as slabs against the sky. `underlay=1` if a future capture needs it.
- **Dynamic objects**: none expected in this clip; no masking stage in the MVP.
- **12 s clip** → 114 keyframes is on the low side for splat quality; a longer capture would score better.

## Evidence locations

- `results/blinded/` — A/B stacks (real frame vs splat render, order blinded), `results/pair_key.json` holds the mapping
- `results/verdicts/` — blind critic verdicts for both bars (`critic_bar1_visual_v2.md`, `critic_bar2_walkability.md`)
- `work/<name>/eval_renders/` — raw eval renders from training-camera poses
- `work/<name>/walktest/` — walk-test frames, `walk_log.json`, webm video
- `work/<name>/shots_before/` vs `shots_after/` — the character-on-the-splat fix, shot from the same eight viewpoints
- `work/<name>/train_progress/` — training eval renders per checkpoint

## What the walk test measures now

`drive_viewer.py walk` drives the autopilot headless and logs position at 0.2 s.
Current numbers on `work/rocks`, against the same test before this round of fixes:

| | before | after |
|---|---|---|
| distance walked | 19.9 m in 215 s | **65.3 m in 32 s** |
| waypoints reached | 1 / 20 | **16 / 16**, all within 0.8 m |
| travel efficiency (walk leg) | 10% | **69%** on a closed loop, 100% on the return |
| feet vs surface (`footGapM`) | −1.8 m (inside a crust 20 m up) | **+0.008 m** |
| viewer heightfield vs physics surface | 0.101 m apart | **0.000 m** |
| falls | 0 | 0 |

Three defects had to go together, and each was found by measurement rather than
inspection:

1. **The character was not on the terrain.** Two disjoint collider shells existed
   at the spawn — ground at −16…−11 and a sky crust at +5…+9 — and every downward
   ray found the crust first, so the capsule stood on it and the underlay was
   draped over it. That is the "floating above a white sheet" screenshot.
2. **The collider was unwalkable.** See step 8: a 0.34 m capsule cannot climb a
   0.35 m voxel riser at all, and the route was planned on a smoothed surface the
   physics never had.
3. **The autopilot livelocked.** Its unstick timer was never cleared once armed,
   so every frame under 0.35 m/s re-armed the 1.4 s spin and the waypoint steering
   was never reached again. The log proves it arithmetically: the walk leg ended at
   288.6 rad of yaw, and 288.6 / 2.2 rad s⁻¹ = 131 of its 149 seconds spent
   spinning in a 1.18 m circle. Waypoints also now time out after 12 s, so one
   unreachable corner cannot consume a whole leg.

### Grid convention: samples are cell centres

Every rasterizer here indexes with `floor((x - ox) / cell)`, so cell `j` spans
`[ox + j*cell, ox + (j+1)*cell)` and its sample belongs at `ox + (j + 0.5)*cell`.
`ground_mesh.py` and `walk_path_from_glb.py` already did this; three sites did not,
and each was off by half a cell (0.32 m here):

- `groundHF` in `pc.js` interpolated as if `data[j]` sat at `ox + j*cell`, which
  put the viewer's idea of the ground **0.101 m** from the surface the physics
  actually had under the player on a 12° median slope — the one number that should
  be identically zero, since `ground.f32` *is* the collider. Now 0.000 m.
- `pick_spawn` in `export_viewer_assets.py` returned the corner of the flattest
  supported 5×5 neighbourhood it had just found, i.e. a point diagonally outside it.
- `check_world.py` located the spawn cell with `round()`. A spawn written at an
  exact cell centre lands on `j + 0.5`, so this was a coin flip: measured (45.5,
  18.5) rounding to cell (46, 19) when the spawn is in cell (45, 18) — the two
  spawn assertions were judging ground 0.9 m away from the spawn. Now `floor()`.

None of these caused a visible failure (the probe bounds and the −4 m fall
threshold carry metres of slack), which is exactly why they are worth writing down.


## V3 - capture console + splat sharpness pass

**Phone recorder (`viewer/capture.html`)**: Android AR coverage scan uses
WebXR/ARCore 6-DoF pose plus depth sensing to maintain a world-anchored surfel
map. A patch turns green only after three distinct, close viewing directions;
red and amber patches get an end-of-take “go back here” report. The recorded
12 Mbps camera video is clean—coverage dots are never burned into it—and its
matching AR pose JSONL plus projection-matrix calibration are transferred with
the take. Basic mode remains available everywhere, but it labels its compass
information honestly as bearings faced rather than surface coverage.

**Splat quality (`--quality high`)**: 1280 px training images (was 1024),
30k steps (was 15k), 3.0M gaussian cap (was 2.0M), gsplat antialiased
rasterization (kills the dilation/erosion softness when capture and train
resolutions differ), blur-aware frame sampling (sharpness-weighted, so
motion-blurred frames pollute the SH colors less), and a weak opacity
regularizer against floater crusts. New `--quality ultra` = 1440 px / 45k
steps / 4.0M cap for 8GB+ GPUs.
