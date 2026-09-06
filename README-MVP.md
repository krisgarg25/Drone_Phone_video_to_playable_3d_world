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
everything downstream of it. A marker also carries `code`, a digest of the source
that ran it (`pipeline.code_digest`: the `.py` files in the step's own command
plus `scripts/robust.py`), so editing a script makes that step stale instead of
shipping an artifact built by code that has since changed. Resume with
`--from <step>`, force with `--fresh`
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
| `smoke` | auto | 640 | 300 | 150 k | ~0.5 GB | every step, end to end, in minutes. This is the regression tier. |
| `standard` | 350 | 640 | 12 000 | 350 k | ~0.7 GB | the original settings |
| `high` (default) | 500 | 800 | 15 000 | 1 500 k | **1.26 GB**, 14 min | 272 frames registered (was 114), 451 k gaussians (was 128 k) |

The cap is headroom, not a target: densification converges where the scene runs
out of detail (temple converged at 451 k under a 1.5 M cap). More VRAM does not
fix bad content — see PROBLEM-temple.md for the capture that no amount of
compute can improve.

`smoke` deliberately does **not** skip training. `frame`, `export`, `colors`,
`collider`, `surface`, `gate`, `evals`, `pairs` and `walktest` all consume
`splat.ply`, so skipping train leaves most of the graph unexercised and the run
passes vacuously. 300 steps on the real solve keeps every step honest — the
splat is blurry, everything downstream is real.

## Production behaviour: nothing to hand-tune, nothing fails silently

The failure mode this repo had was "random crashes, then tweak the settings by
hand until it ran". Every knob that used to need tweaking is now derived, and
every failure that used to be a traceback or a bare exit 0 is now a named,
classified outcome that the runner repairs or reports.

| Was hand-tuned | Now derived from |
|---|---|
| `--preset` chosen by the operator | `--preset auto` is the default: `scan` diagnoses the footage (blur, texture, rotation-vs-translation) and picks one |
| `--width` / `--cap` to "fit the GPU" | free VRAM, per frame pixel count and frame count → `robust.train_budget()`. Frames are resized before they are cached, `K` follows the pixel grid, and the gaussian cap is whatever is left after the render reserve |
| `--voxel-size` per scene | the scene's own bounding box → `fit_voxel_size()` keeps the grid under the voxeliser's ~2²⁴-entry limit instead of raising `Map maximum size exceeded` |
| COLMAP flags per build | `colmap -h` is probed once and cached; a flag this vendored binary does not know is skipped with a warning, not passed and fatal |
| `--train-max-pixels` | the same VRAM budget |
| Gate thresholds | the scene's footprint and cell size, so a 4 m room is not failed by an outdoor 15 m walk-loop minimum |
| `frame`'s "enough geometry to bound a room" floor | how far the supported points spread across the camera path, not an absolute count: the same 300 steps yield 7.5k or 30k gaussians depending on what else held the GPU, and `max(200, …)` stopped a run that already had a finished splat. Too-spread-thin support now bounds the region by the flight path and the ground under it |
| `frame`'s two other 500-gaussian floors | both counted a slice of the train output in absolutes, so the *same take* got ground-refined gravity and a filtered cloud at 30k splats and camera-derived gravity and the raw cloud at 7.5k. Refine is now judged on how many ground points survive its own lowest-40% cut (floor 24, a fit-ability minimum, not a quality bar); the opacity prefilter is kept whenever it leaves ≥5% of the cloud |
| `--min-clearance`, and the body the router plans for | the `character_height` the viewer already scales its capsule by (`CHAR_SCALE = max(0.05, CHAR_H) / 1.75`), so a room preset plans a 0.029 m mover through a 0.077 m corridor instead of demanding a person-sized 0.34 m through 0.9 m; the ~10 m corridor minimum is capped by the walkable region's own cell count |

Every step is wrapped in a retry ladder whose rungs are declared repairs, not
blind re-runs: `OOM` halves the pixel budget or the gaussian cap,
`VOXEL_OVERFLOW` climbs the voxel ladder, a `CRASH`/`TIMEOUT` re-runs once with
more slack. `EMPTY_INPUT` is deliberately **not** retryable — re-running cannot
manufacture an artifact an upstream step never produced, and an hour should not
be spent finding that out. COLMAP has its own six-rung ladder (default → relaxed
init → priors off → global → global-permissive → incremental-permissive) and
keeps the best model across rungs rather than the last. A step that raised a
classified `StepError` has that class read back out of its own log line instead of
being re-diagnosed from incidental wording, so `report.json` names what actually
stopped it rather than a bare `failed`.

A run reports itself in `work\<name>\report.json`, and the same summary prints at
the end of the console run: per-step status, seconds, how many attempts it took
and which fallback rung won, plus `produced_assets`. Exit code 0 means a world
exists on disk — a degraded one that ships beats a discarded one — and the
severity of the degradation lives in `viewer_assets\world_check.json`.

A **hard** world-gate failure is the one step that fails the run without stopping
it. The runner records the gate's own named rules as the step detail
(`kind: world-gate`), prints `WORLD GATE FAILED (n hard): …`, and carries on to
the evaluation renders and the walk test — those are the evidence for *why* the
verdict came out that way, and `report.json` already says the step failed. A take
too degenerate to walk therefore still produces an output and a stated reason,
instead of a build that halts at step 12 and tells you nothing about the last
five. The gate is never downgraded to a pass, and the e2e matrix keeps the two
apart: a gate verdict reports as `gate=…` on its row, a pipeline defect reports as
a FAIL.

The two steps after the gate are **evidence, not inputs**: `evals` and `pairs`
only turn finished assets into pictures, so a failure in either is recorded and
the run still reaches the walk test. Their images go through
`robust.save_image` — write a temp file, `os.replace` it into place — because
`Image.save()` truncates its target in place, and a jpg a browser or an image
viewer has memory-mapped cannot be truncated: `OSError [Errno 22] Invalid
argument` on `results\blinded\rocks_AB_08.jpg` used to abort a run whose world was
already built, and cost the walk test with it. A name that refuses the write is
stepped over and the composite rewritten into the next free slot, so the pair still
ships and the stack count is the pair count; the key is written per take as
`pair_key_<tag>.json` (one shared key let the next take overwrite the previous one's
mapping and leave its stacks unreadable), and it is left as it was when nothing was
written, so a run cannot silently un-credit the stacks still on disk. The
e2e matrix reports such a gap as `evidence missing:` on its row rather than a FAIL —
but only when the step's own failure class is final (nothing to score, a tool that is
not installed); a crash in an evidence step is still a defect.

Three things make that trustworthy rather than optimistic: the trainer writes a
`splat.partial.ply` checkpoint before its first step, so a Windows-level kill
that no `except` can catch is still rescuable by the runner; every step marker
names the digest of the source that produced it, so editing a script invalidates
that step instead of leaving a `done` beside an artifact built by different code;
and a Python traceback in a step log this run wrote is a bug by definition, which
is what the e2e matrix asserts on.

```bat
.venv\Scripts\python.exe tests\check_all.py          REM every fast suite: budgets, capture,
                                                     REM   collider, gate, unit — minutes
.venv\Scripts\python.exe tests\check_all.py --e2e    REM + every take in videos\ at --quality smoke
.venv\Scripts\python.exe tests\test_e2e.py --list    REM which takes that covers
```

## Pipeline

| Step | What | How |
|---|---|---|
| 1 | Frame extraction + sharpness-filtered keyframes | `scripts/extract_keyframes.py` — OpenCV decode, variance-of-Laplacian score, temporal-bin selection, near-duplicate rejection. Full-res copies kept for the visual bar; 640 px copies for SfM/training. |
| 2 | Camera poses (no GPS) | COLMAP 4.1.1 (official Windows CUDA build): GPU SIFT `feature_extractor` → `sequential_matcher` (overlap 20, quadratic) → `mapper` → TXT export. |
| 3 | Splat training | `scripts/train_splat.py` — custom lean trainer on **gsplat 1.5.3** (prebuilt Windows wheel, zero compilation): SH degree 3, pure-torch SSIM+L1 loss, DefaultStrategy densification with a hard 350 k splat cap, packed rendering, batch=1. |
| 4 | Visual evidence | `scripts/render_evals_offline.py` renders the trained PLY from the exact training-camera poses with gsplat (offline, no browser), composited into blind A/B stacks against the real frames (`make_pairs.py`). |
| 5 | World frame | `scripts/solve_frame.py` — gravity comes from the **drone's gimbal**, and metric scale from whichever ruler the caller names: `--speed-anchor` m/s x clip duration, or `--height-anchor` m of camera above the filmed ground (the second one for anything not flown). Both land in `frame.json`, which is now the only source of either. The old path took "up" from a RANSAC plane normal, whose sign is arbitrary: it came out inverted and the entire scene was exported upside down. |
| 6 | Heightfield + colours | `scripts/export_viewer_assets.py` reorients and rescales, then rasterizes splat means into a ground heightfield (low-percentile Y per cell, holes diffused) plus `coverage.u8` marking the ~44% of cells with real multi-view support. `export_ground_colors.py` vertex-colours that grid from the splat's own colours. |
| 7 | Canopy strip | `scripts/strip_sky.py` — drone footage contains sky and haze, and 3DGS reconstructs both as gaussians. Over the walkable footprint they form a separate layer (band density falls 2159/m at +4 m to 135/m at +12 m, then recovers to 1727/m by +19 m), so the script cuts in the gap. The test is **bimodality, not sparseness**, and only gaussians over the footprint are judged: an earlier "is this band sparse?" version fired on a smooth decline and would have deleted 86% of the painted frame, backdrop and all. |
| 7b | Cloud-sea cull | `scripts/strip_clouds.py` — for footage flown ABOVE a cloud layer the fog is multi-view-consistent content and reconstructs as ~30% of the scene (temple: 76 804 fog candidates, saturation p50 0.065, all >1 m above local ground). Cuts `sat < 0.20 AND airborne`, after auto-protecting compact tall structures (column-density clustering + area test: the temple reads as a 470-column cluster, 4.8% of the grid) so grey stone walls survive. Gated on **painted-area fraction** (fires at ≥5%; temple measures 18.6%, a clear-air capture 0.8%) so it never fires on good scenes — rocks passes through untouched. Idempotent via the same `scene.full.ply` convention as strip_sky, so both cuts compose. |
| 7c | Re-export from culled scene | `export_viewer_assets.py --from-scene` — the heightfield, coverage and collider box were computed from the pre-cull splat, and the cull changed what "ground" is (cloud tops were defining half the far-field grid). This regenerates every derived asset from the culled `scene.ply`, leaving coordinates/quats/scales untouched (they are already world-frame; the region test inverts the transform for its COLMAP-frame check only). Skipping this step is measurable: collider vs heightfield disagreed by a median of **28 m** over the temple plateau, and the world gate's floor/ceiling assertion failed at 18.3% before it and passes at 12.2% after. |
| 8 | Collision mesh | `scripts/build_collider.py` drives `@playcanvas/splat-transform` (`cluster_shell`, 0.35 m voxels, box + seed from `collision.json`), then `clip_collider.py` cuts airborne crusts **relative to the heightfield** — a flat box ceiling cannot, because the terrain's own relief overlaps the crust's height range. The result is a shell of axis-aligned voxel faces, in which every height change is a vertical **wall** (measured: 1.05 m walls every 0.6 m along the route) that a capsule of radius 0.34 cannot climb at all — it meets the flat face at its equator, so the contact normal is horizontal and there is no lift. It is a *candidate* ground now, not the shipped one: see step 9. |
| 9 | Ground surface, chosen by measurement | `scripts/tune_collider.py` builds two candidates — the clipped shell's top face, and the exported heightfield — routes on each, and ships whichever one the autopilot walks further on, with ties (within 10%) going to the heightfield because that is also the array the browser draws as the underlay and the one `build_objects.py` measures furniture floors against. It then leaves `ground.f32` = the shipped mesh's own surface and runs the router on it, so **the physics mesh, the plan and the underlay cannot disagree** — verified in the browser, where 140 rays along the published tour stop on ground.f32 at a median of 0 cm. The plan also sizes the *body* from the same file: `walk_path_from_glb.py` takes both the capsule radius and the corridor clearance from `collision.json`'s `character_height` with the formula the browser uses (`CHAR_SCALE = max(0.05, CHAR_H) / 1.75`), and the ~10 m corridor floor is capped by the walkable region's own cell count. Before that, the router demanded a fixed 0.34 m capsule and 0.9 m corridor while the room presets walk a 0.029 m one — 11x more clearance than the mover needs, which is how test2horizontal's 14x17 grid of 1.7 cm cells got reported as "nowhere has 0.02 m of clearance" and routed nothing. Five scenes: auditorium hf 68 m vs shell 51 m, room_w_jsonl hf 60 vs 55 (tie band), rocks shell 94 vs hf 69, temple shell 12 vs 6, room_multi_video shell 8 vs 7. The heightfield is *not* universally smoother — it wins only where the grid cell is finer than the 0.25 m voxel the shell was quantised with — and no local roughness statistic predicts any of it, which is why the router is the judge instead of a threshold. Full evidence in `scripts/tune_collider.py`'s header. |

Measured end-to-end on temple with the new pipeline (`high` preset): the spawn
moved from the −18 m cloud bench to the **courtyard at +8.8 m**, the world gate
passes all assertions, and the autopilot walk test reports **65.4 m, 0 falls**
(the pre-fix world managed a 5 m waypoint circle inside fog). Eye-level evidence:
`work/temple/eye.png` (before, fog mush) vs `work/temple/eye_final_court.png`
and `work/temple/walktest/frames/` (after, terrain under open sky — sparse in
places the drone never saw up close, which is a capture limit, not a pipeline
bug).
| 10 | Gate | `scripts/check_world.py` — 11 severity-tiered assertions: heightfield coverage **split by how it was obtained** (measured vs camera-derived vs nothing — averaging them let a 30%-measured grid report 98%), terrain-is-a-floor-not-a-ceiling (an inverted scene drives this to 100%), cameras above the ground they filmed, spawn supported and flat, collider has no ceiling slab, collider **is** the array the route was planned on (median within 20% of a cell of `ground.f32`), collider sits on the measured heightfield, route long enough to be worth autopiloting (≥15 m), collider exists at the spawn. HARD ones (no measured ground, no collider, inverted floor, spawn in the air) name the rule and become the runner's `world-gate` detail; the rest are quality warnings that ship. It also prints which ground surface won and why, so the judgement is quotable instead of buried in a build log. temple (12 m) and room_multi_video (8 m) fail that route check honestly today: both are coarse or mis-scaled captures with 0.4% and 2.6% of the grid walkable. |
| 11 | Visual evidence | `scripts/render_evals_offline.py` renders the trained PLY from the exact training-camera poses with gsplat (offline, no browser), composited into blind A/B stacks against the real frames (`make_pairs.py`). |
| 12 | Walkable viewer | `viewer/pc.html` — PlayCanvas engine + ammo.js: the collision GLB loads as static trimesh bodies, the character is a dynamic capsule rigidbody, so gravity, sliding and blocking come from the engine. Every downward probe is **bounded to the ground and masked to static bodies** — `raycastFirst` from the sky latches onto whatever crust survives. Autopilot settles, walks 50 m, spins 720°, walks back (`drive_viewer.py walk`). |

## Exact commands (what pipeline.py runs)

Do not hand-type these — the runner owns the order and the resume markers:

```bat
mvp.bat run rocks
python pipeline.py run temple --from surface      REM redo ground + everything after
python pipeline.py status rocks
```

Step order (14 steps indoors with no AR poses, 15 with them, +3 under open sky):
keyframes -> [priors] -> colmap
-> poses -> train -> frame -> export -> **sky -> clouds -> reexport** -> colors ->
collider -> objects -> surface -> gate -> evals -> pairs -> walktest. The three bolded steps are
the canopy cull, and they belong to the preset, not to every run:
`strip_clouds` calls a gaussian fog when it is desaturated and more than a metre
above the local ground over the footprint, which is exactly a white painted
ceiling — so an indoor run used to delete the surface the `room` advice tells the
operator to point the phone up at. `--cull canopy` / `--cull none` overrides.
`--only a,b` limits the pass to those steps instead of forcing them, and a run that
solved too few frames now stops at `colmap` rather than twenty GPU-minutes later
(V6 below). COLMAP is driven by `scripts/run_colmap.py` (the `.bat` delegates to it;
Python argv lists survive the space in this repo's path, cmd quoting did not).

**A scene with both a portrait and a landscape clip used to lose half its
footage in silence.** COLMAP is given one shared camera by default, and it answers
an image whose dimensions disagree with that camera with a per-image
`CAMERA_SINGLE_DIM_ERROR` *warning* and a zero exit code — the frames simply never
reach the database. Measured here on a two-clip scene: 120 keyframes written, 60
images in the database, and the mapper printing its registration rate against 120,
which reads as a phone that tracked badly rather than as deleted footage.
`run_colmap.py` now reads each clip's frame geometry from the JPEG headers,
switches to `single_camera_per_folder` when they disagree, and stops the run if the
database still holds fewer frames than the folder does.

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

- **Scale is not measured**: no GPS. The operator names one metric number — camera speed (`--speed-anchor`) or camera height (`--height-anchor`) — and `solve_frame.py` fits the reconstruction to it, so "metres" are only as good as that number. Nothing here can tell a wrong one from a wrong-but-consistent one; the cross-check printed alongside it is the human's job.
- **Grass is low-texture** — expect SfM to lean on the rock formation; far field may be soft/floaty (the bar-1 critic noted background smear as the main cosmetic flaw).
- **The collider is a heightfield**, so overhangs and caves cannot be expressed — a voxel shell can express them and is unwalkable, which is the trade `ground_mesh.py` makes deliberately. This capture has none worth keeping. 37% of cells have no measured shell surface and are dilated in from their neighbours, with the exported heightfield as the last resort.
- **The ground underlay is off by default.** It only ever emitted quads where all four corners already had gaussian support, so it never filled a gap — it just sheeted opaque colour over ground the splat renders, and at 6 cm under the surface (well inside the splat's own ±0.5 m scatter) the scene read as flat olive mud. Sunk 0.7 m the vegetation returns but the sheet's cut edge shows as slabs against the sky. `underlay=1` if a future capture needs it.
- **Dynamic objects**: none expected in this clip; no masking stage in the MVP.
- **12 s clip** → 114 keyframes is on the low side for splat quality; a longer capture would score better.

## Evidence locations

- `results/blinded/` — A/B stacks (real frame vs splat render, order blinded); `results/pair_key_<tag>.json` holds each take's mapping, and `scripts/label_pairs.py` merges them all
- `results/verdicts/` — blind critic verdicts for both bars (`critic_bar1_visual_v2.md`, `critic_bar2_walkability.md`)
- `work/<name>/eval_renders/` — raw eval renders from training-camera poses
- `work/<name>/walktest/` — walk-test frames, `walk_log.json`, webm video
- `work/<name>/shots_before/` vs `shots_after/` — the character-on-the-splat fix, shot from the same eight viewpoints
- `work/<name>/train_progress/` — training eval renders per checkpoint

## What the walk test measures now

`drive_viewer.py walk` drives the autopilot headless and samples the walk every
0.5 s — elapsed time since spawn, position, distance so far, whether the floor was
under the capsule, falls — into `walk_log.json`'s `samples` (capped at 600, so a
240 s test never truncates). The totals on their own are not evidence: `walked: 16`
says nothing about whether the body travelled or ground its gears against a
collider, so the runner sums the sampled route and warns when it covers less than
half the distance the walk claims. Until this round the `samples` list existed but
nothing wrote to it, so every log shipped `[]` and the console printed `samples=0`
on healthy walks. The first version stamped samples with `autopilot.t`, which
restarts on every phase change: rocks' 60-sample route reported a ~30 s walk as
"6 s", which reads as 11 m/s and would let a stuck walk look brief, so the clock is
a dedicated accumulator that only resets at spawn.
Current numbers on `work/rocks`, against the same test before this round of fixes:

| | before | after |
|---|---|---|
| distance walked | 19.9 m in 215 s | **65.3 m in 32 s** |
| waypoints reached | 1 / 20 | **16 / 16**, all within 0.8 m |
| travel efficiency (walk leg) | 10% | **69%** on a closed loop, 100% on the return |
| feet vs surface (`footGapM`) | −1.8 m (inside a crust 20 m up) | **+0.008 m** |
| viewer heightfield vs physics surface | 0.101 m apart | **0.000 m** |
| falls | 0 | 0 |

Once the route was real it exposed something bigger than its own clock. Every
room-scale take logged **`335 of 335 airborne` with `falls=0`**: the walker's probes
are written in absolute metres (`groundRay` casts 0.80–2.60 m below the capsule
origin) while the capsule itself is scaled by `CHAR_SCALE = CHAR_H / 1.75`, so in a
room reconstructed at a thirtieth of life size the ray *started below the floor* and
could never find it. Nothing gated on grounding could fire — the step-up assist, the
jump, the `lastGood` clamp — and the out-of-world detector needed a 4 m drop, about
80 body heights, which is why `falls=0` was never the evidence it looked like. Every
distance in that block is now multiplied by `CHAR_SCALE`, so a human-scale world is
numerically identical (rocks re-ran at 65.2 m, 0 falls, 5/61 airborne) while the
scaled ones stand on something:

| take | airborne samples, before → after | sampled route, before → after |
|---|---|---|
| test1 | 335/335 → 0/332 | 21.2 m → **26.2 m** |
| test2train | 336/336 → 0/336 | 12.4 m → **26.4 m** |
| test2horizontal | 336/336 → 0/336 | 12.3 m → **28.0 m** |
| roomscan | 332/332 → 0/331 | 19.5 m → 15.5 m |
| room_w_jsonl | (not measured before) → 1/334 | **31.7 m** |

roomscan is the one take that got *shorter*, and the log does not say why: with zero
falls there were no respawn teleports for the distance filter to discard, so the
honest reading is that a body which now registers contact walks a different route,
and a different route is not automatically a better one. What the cross-check buys is
that the claimed 17.1 m and the sampled 15.5 m are the same walk — before, the two
agreed while the body was reported airborne for every one of 332 samples.

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
red and amber patches get an end-of-take “go back here” report. Those colours
are read off the inset map, not painted over the camera: the live AR view carries
only thin world lines (the scanned-volume cage, the blue gap cross, and outlines
round the planes the phone reports). One coloured quad per visible surfel had
buried the camera image under ~1600 overlapping squares, and a scan you cannot
see through is worse than no overlay. The recorded
12 Mbps camera video is clean—no overlay is ever burned into it—and its
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


## V4 - live AR coverage on the phone

**One action, AR by default.** `viewer/capture.html` had carried a complete
WebXR coverage scan that nobody could reach: `startArScan()` had no caller, the
capability probe detected AR support and then talked the user out of it, and a
test asserted that no AR button may exist. The hero is now a single **Start
scan** button that enters the AR scan when the phone supports it and falls back
to camera capture otherwise, with the reason in the Info sheet. AR is gated on
`camera-access` as well as `immersive-ar`, because ARCore owns the camera for
the whole session and a take with no video is not a take. Gallery ingest is gone
from the phone; existing clips are run from the laptop.

**Coverage is geometry now, not screen paint.** Each surfel draws as a quad
lying in its own measured surface plane, so coverage reads as a shell
shrink-wrapped on the object: red seen once, amber getting there, green done. A
grey cage marks the scanned volume and a blue cross sits on the biggest patch
still short of angles. The XR layer now requests a depth buffer so the shell
occludes itself instead of needing per-frame sorting. The recorded video still
carries none of it.

**One real cause of "the dots don't stick."** The camera background computed its
crop from the canvas backing-store aspect while being drawn into the XR viewport,
so the image and the projected overlay slid against each other on a portrait
phone. Every crossing between view space and image space now goes through
`arFit`. The second cause suspected in the same session was dismissed as wrong —
it was right about one of the two depth reads, and this is the other one.

**The two depth reads do not share a coordinate space.** `getDepthInMeters(x, y)`
is specified over *normalized view* coordinates and applies
`normDepthBufferFromNormView` itself, so a sample taken through that call needs no
transform at all — that part of the earlier conclusion was right, and the
`crop / fill` switch built on a different assumption stayed gone. But the
raw-buffer branch reads `depthInformation.data` **by pixel**, which is a position
in the depth image, and it was then unprojected through the view's projection
matrix as though the two were the same rectangle. On a handset whose depth image
and view already line up the error is invisible; where they do not, every patch is
dropped along the wrong ray. `arDepthMapping` now inverts the phone's own transform
for the pixel path and applies it forward for the occlusion lookup, and
`AR.gridSpace` records which space the grid was filled in — because the two
branches fill the *same arrays* in different spaces.

**Measured against a browser, not against the samples.** Chromium exposes no
`matrix` member on `DOMMatrix` at all, and carries a 2D transform's translation in
`m41`/`m42`, not the `m13`/`m23` the WebXR samples read. A reader written from
those samples returns zeros for a phone that answered in full — which is also why
the probe had been printing `NOT EXPOSED` for `normDepthBufferFromNormView` on
every Chromium handset it ran on. `arDepthAffine` takes the canonical `a`–`f`
letters first, and `tests/test_landscape_hold.js` checks the forward mapping
against `DOMMatrix.transformPoint` itself.

Passing pixel indices instead of 0..1 to `getDepthInMeters` throws `RangeError`,
which is exactly why a probe once reported 0% depth on a phone that was working
fine. `tests/test_capture_scripts.js` pins the call: every probe cell must be
queried inside 0..1.

**Calibrate before recording, visibly.** WebXR exposes no tracking state and no depth
confidence, so a null viewer pose is the only signal there is. Pressing Start scan
opens a centred panel that measures three things behind a progress bar: the pose must
translate ~0.6 m under a real walk, depth pixels must be arriving, and geometry must
be accumulating past 120 patches. It asks for whichever one is furthest from met,
never all three at once. Recording starts when they all hold, and the pose log
restarts on that instant so video and poses stay aligned. After 20 s without a lock it
prints the three numbers and offers **Start without waiting** — inform, never gate. The panel
doubles as the colour legend and hides the coach pill, the inset and the stats card
while it is up. This replaces recording from frame one, which opened every take with a
second or two of untracked frames that silently poison COLMAP.

ARCore's depth is motion-stereo ML depth, best at 0.5–5 m; published drift on indoor
walking sequences is ~0.1% of path length, rising to a few percent on stairs.

**One info panel, not two.** The stats card and the corner diagnostics merged: the
top-left card carries observed m², needs-angles m², fps and areas in view, while
areas measured, distance-read %, scanned volume and walked distance live in the
panel's own readout rows (relaid onto its Scan tab in V5).

**Landscape is chosen, not stumbled into.** An `@media (orientation: landscape)`
layout anchors the panel between the top edge and the control row and scrolls it
internally, shrinks the inset and re-spaces the coach, because a 390 px-tall screen
cannot hold the portrait stack. Above that, the hold is now asked on the start
screen — `Auto / Vertical / Horizontal`, auto following the window's own shape.
Three things had been assuming an answer: the "Full screen" chip hard-locked the
phone sideways underneath a scan built for a tall screen, the capture canvas opened
at a hardcoded 16:9 and then rounded to an **odd** edge (591, and 591 the other way
round), and the Phone tab reported none of it. The canvas now follows the phone's
image with both edges even, and the tab states the hold, the take's size, the
screen view, the distance image and which space the distance grid was filled in —
so a handset that behaves differently sideways can be read back as numbers.

What is *not* claimed fixed: the non-AR compass fallback still builds its attitude
on the upright convention that reads beta as tilt and gamma as roll, and those two
change places when the phone turns. The hold and the screen angle now travel in
`calibration.json` so that can be settled against a real handset instead of
guessed; AR mode is where measured coverage lives either way.

**Honest metrics.** The headline is surface *observed* in m². The covered percentage
is gone from the HUD entirely: `coveredPct` is a ratio over surfels that already
exist, so it reads ~100% after one well-scanned wall. A "no data to your left" cue
comes from bucketing surfels by world azimuth — the only signal that survives
everything in front of the operator reading green.

**The inset is 3D, not a floor plan.** A plan view answers "how far around have
I walked" and nothing else: from above you cannot see whether the top of a
cupboard or the upper third of a wall is done. The inset now draws the real
surfels in perspective from a camera orbiting the scanned volume, auto-framed to
its extent, with the volume cage, the gap beacon and the operator marker all in
the same scene. The look-at maths is hand-built and Canvas2D, so no second
WebGL context competes with the AR session.

**Coverage wants a baseline, not a lap around the room.** The old rule needed
three distinct world azimuth bins, 45° apart — unreachable for any surface with
one accessible side: a wardrobe front, a wall against a neighbour, one side of a
stairwell. Those stayed red no matter how well they were scanned, which is worse
than useless because it teaches the operator to ignore the colours. `stateOf`
now also accepts the angle between the two extreme in-range viewpoints
(`minBaselineDeg`, default 25°), tracked per surfel at no extra cost. Two stops
half a metre apart at arm's length clear it; staring from one spot never does.
The tests pin the wardrobe case explicitly: 2 azimuth bins touched, 59° of
baseline, COVERED.

**Verification.** `tests/test_ar_overlay.js` drives a real WebGL context in
headless Chromium: the marker shaders compile clean, a surfel drawn through the
hand-built world→camera and projection matrices lands where `CoverageMap.project`
puts it (0.000 px error), `arCountInView` tallies the states in frame without
drawing any of them, and the cage, gap beacon and plane outlines each rasterise in
their own colour. Two checks guard the camera view itself: no warm coverage colour
may reach the screen, and the markers may not cover more than 5% of it (they cover
0.68%). `test_capture_scripts.js`
pins the depth call inside 0..1, the occlusion lookup, the calibration gate (including
that depth plus walking must not arm it without accumulating geometry), the early
"scan anyway" escape, and the inset framing. A phone that never delivers depth is
covered too, because that path once threw at a removed element.


## V5 - one reachable panel, operator language, 638 KB less library

**The screen you could never reach.** The sharpness readouts, the quality numbers and
the recording facts all lived in the `Info` sheet — and the sheet was unreachable during
an AR scan for two independent reasons: the top bar that opens it hides while a scan
runs, and, fundamentally, `dom-overlay` composites **only** the root element over an
`immersive-ar` view. Anything outside `#ar-overlay` does not exist mid-scan. The whole
interface therefore moved inside that root: one centred panel, three tabs —
**Scan** (progress, instruction, readout rows), **How to read it** (the colour legend
and the baseline rule in words), **Phone** (what the handset itself reported: tracking,
distance sensing, the distance image, camera resolution, codec, frame rate, distance
walked). The sheet outside AR carries the same three tabs, so the camera path can
report the same facts. A test now measures the reachability claim: it slices the
markup from `<div id="ar-overlay">` and requires the panel, the rows, the legend, the
per-phone table and the look switch to be inside it.

**638 KB of library the phone never needed.** `three.min.js` (603 KB), `OrbitControls.js`
(26 KB) and `lidar_overlay.js` (8 KB) are deleted. three.js was there for
`Vector3`/`Quaternion`/`Euler`, a point-cloud renderer whose view had no entry point, and
a 30 fps `requestAnimationFrame` render loop that kept burning the phone's GPU behind a
hidden canvas. The compass/IMU pose log needs four operations, so the page now carries
about 30 lines of local maths in the same conventions (YXZ euler order, right-handed
rotation, `[x,y,z,w]` quaternions). The phone downloads one 18 KB library. Tests pin
that no `THREE.` call survives outside a comment and that the retired files are gone
from the markup and from disk.

**Words you can act on with your feet.** Every string that reaches the operator was
rewritten: no *surfels*, *patches*, *bearings*, *quadrant*, *azimuth*, *voxel*,
*calibrating*, *6-DoF*, *ARCore*, *MAP FULL*, *immersive-ar*. It says "Areas measured",
"Distances read", "Point the camera at a wall, about an arm's length away", "Blue cross —
1.4 m² there needs another look", "Start without waiting". A gate in
`test_capture_scripts.js` extracts the ~410 operator-visible strings — copy assignments
and markup text nodes, comments excluded — and fails on any banned word, so the pass
cannot quietly rot.

**Dead markup is now a test failure.** The `#ar-depth` crash was one bug class: an
element deleted for layout reasons with its writer still running. `tests/test_element_ids.js`
checks it statically over both phone pages and their libraries: every `getElementById`
resolves to real markup, every inline handler names a defined function, every id in the
markup is referenced by script, CSS or another attribute, every `src=` file exists, and
every `data-tab` has a matching `data-body`. It caught two survivors on its first run
(`#ph-detail` had no writer; `#rec-core` was a dead id).

**Two readouts that lied, fixed.** The gap report headline used `coveredPct`, which is a
ratio over geometry that already exists — one well-scanned wall made it read ~100%. It
now leads with surface observed in m² and splits finished from unfinished. The capability
probe claimed the coverage rule needed "three distinct viewing angles"; the rule accepts
a 25° baseline from two spots, so the probe now says "two spots far enough apart — walk
around, do not spin on the spot". The basic-mode note says plainly that its numbers are
directions looked from a compass, not surfaces measured.

**Landscape, measured again.** With the panel inside the overlay the short-screen
problem changed shape, so it was measured rather than assumed: readout rows go
two-column like the legend (a 640 px-wide band has the width to spend), and the panel
anchors to the top and caps above the 62 px control row instead of stretching into a
fixed band. Rendered heights per tab are 281/203/273 px on a 390 px-tall screen with no
tab scrolling; the harness fails if any tab needs a swipe to read its last line.

**One screen, one action.** The start screen offered two ways in: the hero's **Start
scan**, and a full-size red REC from the basic-mode dock floating above it. That REC had
no stream to record, so the second way in produced a silently empty take. `setHero()` is
now the only writer for that screen and it takes the top bar and the dock off with it —
nothing is being tracked or recorded before a scan exists; tapping REC anyway routes to
Start scan. The full phone check stays one tap away as a plain text link (until now it
hid in the `Info` sheet, which an AR scan cannot open) — a link, deliberately, so the
screen still has exactly one button. Both handset classes are screenshotted in a real
browser and the basic path is driven end to end against a fake camera: REC before there
is a stream asks for the camera, and only the second tap starts a take.

**The plan view fills its box.** It is wide now (236 × 168 portrait, 168 × 116 landscape)
because a room scan is wide, and the square it used to be wasted the point: the old focal
length fitted the volume's *longest side* to the canvas, so a 6 × 6 × 1.2 m room drew a band
across the middle of the glass. The frame is measured instead — corners, operator and beacon
projected at focal length 1, then the length that fills 86% of the limiting axis, eased
between draws so a 1 Hz re-measure never reads as a jump. Measured on the test room, the
box now used is 78% × 31% of the glass at the old 24° view and 73% × 60% at 49°, which is
why the default aim went up.
The canvas keeps its own pixel count (CSS box × device density, capped at 0.52 MP because
every pixel is refilled four times a second), and dragging repaints on the next animation
frame rather than waiting for the 4 Hz HUD tick. Sizes are still proportional to the store —
a test draws the same scene at two sizes and fails unless the cloud comes out exactly twice
as big in every way.

Orientation was the other half of "I lose it when I move": the view used to freeze at the
first heading, so once you turned, up on the map stopped meaning ahead. It is now
heading-up like a car nav — the way you face is always at the top, eased so a fast spin is a
smooth catch-up rather than a flicker, with the drag adding an offset on top instead of
fighting the follow — and the caption says "ahead is up". Because the higher aim made the
room read as a plan, the cage's floor face is filled and a faint drop line runs from the
operator dot to the floor, so crouching or reaching a high shelf visibly moves you up and
down the room.
The heading itself is only trusted while the phone looks somewhere with a real horizontal
component (`arHoldHeading`): pointed straight at the floor or ceiling that component
collapses to noise, and dividing it out used to spin the heading — the map "lost track" at
exactly the moment you were scanning the floor or ceiling. The last good heading is held
through the pitch and picked up again when the phone levels out, and the plan view reads the
stored surface normal to dim floor and ceiling dots to context, so the walls stay the
subject.

**The take starts when you say so.** Calibration passing used to start recording on its own
and close the panel after 1.4 s, so the take began at whatever instant the phone finished its
checks — often pointed at the wrong wall. `arReady()` now stops there: the bar fills, the
panel stays up and its one button becomes **Start recording**. The tap is what starts the
recorder and stamps the video/pose clock, so the two can never include the pre-roll. It is
still not a gate — tapping while the checks run starts the take anyway and says so on the
coach pill.

**A finished take can be sent.** After an AR scan the screen behind the report is the start
screen, and `setHero()` takes the dock — the only Send button — off it. So a completed take
had no way to the laptop. The report now carries it: **Send to laptop** (44 px, primary) plus
**Not now**, shown only when the take has video or a position track. Driven in a real browser
from both paths: the basic one records against a fake camera and taps its way to the send
sheet, the AR one opens the report with a finished take behind it.

**The page could not tell a floor from a ceiling, because it never asked.** The AR session
requested `camera-access`, `depth-sensing`, `dom-overlay` and `local-floor` — and nothing
else, so the only evidence of "this surface is flat and horizontal" was a normal estimated
from three depth taps 10 cm apart. That is what the dimming above was reading, and on a real
ceiling the estimate is mostly noise: nothing was ever learned about how tall the room is, or
whether the ceiling had been looked at at all. So the evidence is now pooled twice over and
merged. `CoverageMap.fitRoom` reads the whole surfel store once a second — heights binned
into a histogram, normals signed *to face the camera* — and `setPhonePlanes` takes whatever
the handset's own plane finder reports, which outranks the fit because it is built from every
frame ARCore ever saw. The test that sorts a wardrobe top from a ceiling is not flatness, it
is direction: both are horizontal, but the ceiling faces **down** at you and the cupboard
faces **up**, so a threshold on "how tilted" can never separate them and one signed normal
instantly does. Measured on a synthetic 5×4 m room with a cupboard stared at harder than the
floor was: floor 0.000 m, ceiling 2.412 m, four walls, room 5.00×6.00 m — and with no ceiling
in the take at all, no ceiling is claimed.

`plane-detection`, `hit-test` and `anchors` are **optional** features and
`initiateRoomCapture()` — ARCore's six-plane room update — is called only when a build has
it, because a refusing phone still gives a 6-DoF-posed video, which beats refusing to start.
The diagnostics therefore say which source spoke ("found (from the phone)" versus a bare
"found"), whether the phone reported any planes at all, and what the ceiling's seen percentage
is against the room footprint — the same honesty trap as `coveredPct` reading 100% after one
wall, one surface type at a time. Detected planes are drawn as outlines in the world, the
inset's height budget is clamped to floor-and-ceiling rather than to the percentile cloud,
and the whole room ships as `data_room.json` next to the calibration: floor, ceiling, wall
planes and the polygons, in the same frame as the poses. Which is ground truth the rebuild can
check itself against, where today the heightfield and the collider both have to infer which
way is down from the splats and have disagreed by 28 m.

Same pass, the depth and the frame cost. The depth image is now read from its raw buffer when
the phone hands one over, which recovers the per-texel **confidence** channel that
`getDepthInMeters()` discards — the only defence against a wall's edge before this was a 0.25 m
neighbour-jump test — and lets a grid cell take the *nearest* surface inside it instead of
tapping the middle. 768 accessor calls a frame became none, and the confidence scale is
learned from the largest value seen with a slow decay, so a phone reporting 0..1, 0..255 or
0..65535 all work and one spuriously confident texel cannot set the bar for a whole take. For
the handset itself: `CoverageMap.projectPacked` writes five floats into a caller-owned buffer
instead of allocating an object per visible surfel twice a frame, the three per-frame vertex
buffers reserve their GPU storage once and then `bufferSubData` into it, and the capture
surface — a second full-screen textured draw plus a 1280-wide canvas readback — does not run
at all until the recording tap, which is also where the take was always meant to start.
Exposure, focus and white balance are locked on the basic path, where there is a track to
constrain; during an AR scan the phone owns the camera and the row says so.

**The first real take, and the three things wrong with it.** `videos/test1` is a 94.9 s scan off
the test handset — 2591 poses, 15.1 m walked, a 360° turned in the middle of the room and then
the same turn pointed up at the ceiling and down at the floor — and the ceiling came back
unmapped. Only one of the three causes was the technique.

*Everything measured from depth was being thrown away.* A depth-grid normal is the cross product
of two steps along the phone's own image axes, and on a wall that winding points **away** from the
camera. `project()` back-face-culls on the normal, so in a replicated wall all 713 surfels were
culled and nothing the depth pass ever measured could be drawn or credited. `observe()` now
settles the sign on the way in, against the same viewing ray the incidence gate already uses, and
a sample sitting on a level surface the phone measured keeps that surface's vertical normal
instead of its own noisy estimate.

*The two sources were allowed to disagree about physics.* `setPhonePlanes` took the biggest floor
polygon and the biggest ceiling polygon it had ever seen and published the gap between them,
which in this take paired a step at 0.198 m with a bulkhead at 1.795 m: a 1.60 m room, with a
ceiling the page then reported 84% unseen. Level planes are now merged across the patches one
surface arrives as (this take handed back four ceiling polygons), and a floor and a ceiling are
**paired** — scored by how much of one lies above the other, required to leave 1.85 m of
headroom, with the phone's own `semanticLabel` outranking all of it. The same six levels read a
2.33 m room, and a mixed phone-floor-under-this-page's-ceiling is re-checked against the same
bounds before it reaches the screen.

*The technique cannot work, and nothing had been saying so.* Coverage needs a baseline, so every
observation of a ceiling from one standing spot is the same observation. The 26% of the take
spent pointed up covered 0.6 m of ground and measured nothing, while the floor pass — the same
spin — happened to spread 2 m. `scripts/analyze_take.py` measures exactly that off any take folder
(its default is this fixture) and names the band that was spun in place; the same questions are
pinned in `tests/run_tests.py`, so a take that walks *under* the ceiling is what moves them. On
the page the floor/ceiling split is judged against the take's **median** camera height rather than
wherever the phone happens to be pointing — the two differ by a metre in this take — that height
is a row in its own right in the room readout, and the ceiling cue now outranks a small gap
cluster once the ring of walls is closed, which is precisely the state a 360° turn leaves you in.

**Verification.** 265 JS checks pass — 131 capture-page (the last thirty driving the raw depth
buffer's confidence gate and nearest-surface pick, the phone plane ingest from labelled and
unlabelled planes, his own take's plane set replayed through that ingest end to end, the normal
snap, the coaching order and the room rows), 110 coverage-engine (the last fourteen pairing the
levels from that same take, holding a supplied-away normal through the backface cull, and refusing
to read an absent reference height as zero),
15 real-WebGL
overlay, 9 reference-guard — plus 24 assertions in the render harness
(`scratch/ar_hud_shot.js`): portrait/landscape layout, the start screen, the camera handover,
the phone-check shortcut being reachable and not a dead link, the Start recording tap, both
end-of-take reports offering the send, and the room fit plus one reported plane rendering
through the real page. `scratch/bench_hud.js` times the per-tick work on
a 11 040-area map: the gap clustering that runs in the same tick was the most expensive thing
the HUD did (string cell keys), and is now keyed by packed integer — 1970 µs → 981 µs on a
desktop core, with the clustering results unchanged. `tests/run_tests.py` carries 29 python
checks, the last six of them measuring `videos/test1` itself. To ask the same questions of a new
take: `.venv\Scripts\python.exe scripts\analyze_take.py videos\take2` (exit code 1 when a rule
fails, `--report-only` to just read it).
**None of it has run on a phone**, and `camera-access`, which gates AR recording, is
still unconfirmed on the test handset: **Run the full phone check** on the start screen
is the way to check it — the AR step now also reports whether plane detection was granted and
how many planes came back, which is the number that decides whether the room is measured or
inferred on a given handset.


## V6 - the crash was never about landscape

**Six cases, one variable at a time.** A horizontal take was reported to "just crash" while
vertical worked. Training a 640x296 dataset aborted with a bare `exit 3221225620`
(`0xC0000409`) and no traceback, which is the shape of a hardware complaint, so the geometry
was tested against the rasterizer directly (`scratch/gsplat_zero_probe.py`, one case per
process because a fail-fast takes the process with it):

| gaussians | frame | result |
|---|---|---|
| 2000 | 296x640 | renders + backward fine |
| 2000 | 640x296 | renders + backward fine |
| 2000 | 639x295 (odd) | renders + backward fine |
| 2 | 640x296 | renders + backward fine |
| **0** | 296x640 | **process dies, no output** |
| **0** | 640x296 | **process dies, no output** |

Landscape is not the variable. An **empty gaussian cloud** is, and it kills both shapes.
The log said so the whole time — `init 2 gaussians`, then `N 0` on the last line before the
abort — but a number at the end of a line does not read as a cause.

**Why the cloud empties.** Too few registered frames triangulate too few seed points; the
densifier prunes on opacity and takes the last one at its first refine (step 600, exactly
where the log stopped). Two shapes of footage reach that state: a take with no baseline, and
a mixed portrait/landscape scene, which used to lose half its frames silently because COLMAP
rejects dimension-disagreed images under one shared camera as a *warning* at **exit 0**. The
frame loss is fixed (one camera per clip, and a hard stop if anything is still dropped); the
collapse now reports itself in the three places it can be caught:

- `run_colmap.py` stops under `max(8, 10%)` registered, naming the count and the two things
  worth checking. Real takes in this repo measure 61%, 91%, 95% and 98%, so the floor sits an
  order of magnitude below anything that can build a world.
- `train_splat.py` refuses to start on a seedless reconstruction instead of letting CUDA kill
  it, and says so in the same words.
- Mid-run, the step that prunes the last gaussian exits 1 with `COLLAPSED at step N` instead
  of the native abort. Verified on the model that produced the original crash: same command,
  same data, and it now names the cause at step 600 and stops.

**`--only` is a restriction now.** `--only keyframes,colmap` ran all thirteen steps — the flag
forced the named steps instead of limiting the pass to them, which is how a solve-only check
ended up on the GPU. Steps outside the list are skipped before their outputs are even looked
at, and it still holds under `--fresh`, which otherwise marks every later step stale.

**Verification.** 297 JS checks across five suites (30 of them new, in
`tests/test_landscape_hold.js`: the three hold choices, auto resolving to the window's own
shape, even recording edges across five aspect ratios, the vertical crop pin and its mirror,
the `DOMMatrix` forward and inverse mapping against a real browser, and the bearing being
roll-invariant) plus 47 python checks, the newest three pinning the runner pass itself with
`run_step` stubbed. **Still handset-only:** whether the coverage map holds its shape through a
real landscape AR session, and the compass-fallback bearings in landscape, which need a
physical turn to judge.


## V7 - the lens is asked about, and measured where it cannot be chosen

**A wide lens carries more of the room per frame**, which is what the solve eats, so the start
screen now asks: `Lens — Auto (widest)` plus whatever lenses the phone names. The two capture
paths answer that differently, and the difference is in the spec rather than in the code.

**An AR scan cannot choose a lens.** WebXR's raw camera access defines `XRCamera` as `width`
and `height` and nothing else, and the session request has no facing or lens option — ARCore
picks and the page is handed the frames. So the AR side measures instead: WebXR normalizes the
projection matrix, so `P[0]` and `P[5]` are `1/tan(half-FOV)` per axis and the angle needs no
pixel size. The Phone tab reports it (`70° across × 43° up` for the focal length the
calibration file assumes) and the probe flags anything under 55° as narrow, because that is a
telephoto recording a room.

**The basic path asks, then verifies the answer.** Android Chrome supports only the zoom half
of PTZ, and on a multi-camera phone the *bottom of the zoom dial is the ultrawide* — so Auto
reads `getCapabilities().zoom` and applies the minimum when it is below 1.0. Labels only rank
the candidates; they never decide, because Chrome's are opaque (`camera2 0, facing back`) and
say nothing about focal length. A front camera is not offered at all. What actually happened
travels in `calibration.json` next to the focal length it explains: the label, the deviceId,
the zoom applied, the dial range offered, and any refusal.

**Nothing here can cost a take.** A deviceId the phone stopped offering (they rotate between
sessions) falls back to Auto, reopens the camera and says so on the start screen; a dial that
is advertised and then refused is reported as refused rather than as success. Inform, never
gate.

**The one question a handset has to answer** is whether its dial goes below 1.0 at all.
`xr_probe.html` now lists every `videoinput` with its deviceId and group, prints the zoom
range, asks for the minimum and reads back what the phone did — `WIDEST HONORED`, `IGNORED`,
or a thrown `OverconstrainedError`. One run settles it. 42 new checks in
`tests/test_lens_choice.js` pin the choice logic against a stubbed four-lens phone, including
the fallback and the FOV maths cross-checked against the pinhole model `calibration.json`
uses; the suite is now 339 JS checks across six files plus 47 python.


## V8 - one command, one dashboard: run, view, capture

**`mvp.bat` with no arguments is the whole product.** It runs `pipeline.py ui`, which probes
port 8137: if something already answers it only opens the browser, otherwise it starts
`_serve.py` (stdlib HTTP, nothing else) and opens `viewer/pipeline_gui.html`. Opening the
dashboard costs a directory listing - the server process sits at ~25 MiB with zero torch or
CUDA modules loaded, `/api/scenes` answers in ~2 ms, and no `.ply` is fetched until the View
lane is actually opened. torch/gsplat live only in the child process a run spawns.

**Three lanes over one always-visible monitor.** *Run* is the launch form (scene, preset,
quality, the advanced overrides) plus Start/Abort. *View* embeds the same page
`pipeline.py view` opens, in an iframe over `/viewer/pc.html?asset=/work/<name>/viewer_assets`,
with autopilot / cameras / ground-sheet checkboxes rebuilding the src; the iframe only gets
its src once the lane is on screen, because loaded while hidden the viewer sizes its canvas
0x0 and has no resize handler to recover. *Capture* prints the phone URL
(`https://<lan>:8138/viewer/capture.html`) ready to copy, lists the takes sitting in `videos/`,
and hands one straight to the Run lane.

**The monitor reads the logs, not a process handle.** Every step appends to
`work/<name>/logs/<NN>-<step>.log` with an `[exit E] Ss` footer, so `GET /api/scenes` rebuilds
the step table (status, seconds, exit) and `GET /api/tail?scene=&cursor=` merges the logs in
step order behind a `<file>:<bytes>` cursor - half-written last lines held back, an offset past
a `--fresh`-replaced log restarted, a MiB per poll so a reload mid-run catches the live edge at
once. The consequence is the point: **a run started in a plain terminal shows up in the
dashboard exactly like one the dashboard started**, and reloading the page mid-run replays the
terminal from the logs and keeps moving. The four training boxes (step/total, ETA, it/s + PSNR,
VRAM) read the newest `[train]` line the tail delivered, with the total taken from that run's
own `start steps=` line so one scene's replay can never lend another its number.

**Verified on this machine:** 58 python checks (including `scan_scenes` / `tail_run` against a
temp work dir and a live server bound to a spare port), 24 dashboard checks driving the real
page in Chromium, and 11 live checks that start `pipeline.py run test2train --only train
--steps 300` from a child process and watch the row say *running* while `[train]` lines stream
in, survive a mid-run reload, and close with their seconds.


## V9 - a hold tap that actually turns the phone

The last handset pass showed V7's assumptions breaking in three places, all
visible from the operator's side: pick *Horizontal* and the UI stayed tall, the
recording came out 590x1280 with `hold: "landscape"` written on it, the map
looked empty because it had nothing landscape to draw, and *4 cameras seen* sat
above a phone with three backs and a selfie.

**Full screen is not a nicety, it is the gate.** Android Chrome refuses
`screen.orientation.lock()` outside full screen and rejects with
`NotSupportedError`, which the old `applyHold` swallowed by discarding the
rejection's argument. So V7 asked, got refused, wrote `holdAsked: "landscape"`
next to `screenAngle: 0`, and the operator saw nothing happen. Now the tap on
Vertical or Horizontal is the same tap that grabs the screen: `setHold` awaits
`documentElement.requestFullscreen({ navigationUI: "hide" })` and only then
calls `applyHold`; returning to Auto releases full screen so the browser chrome
comes back; the rejection name is kept on `HOLD.err` and the note reads *"Chrome
only locks the phone in full screen - tap Horizontal again"* until it does.
A `fullscreenchange` listener re-asks so the system back gesture or an Escape
key cannot leave the note claiming a lock the phone has already dropped.

**The canvas latch was holding the whole take down.** `arFitCanvas` latched on
the first call, so even after the phone did turn, `#ar-out` stayed at its
initial portrait dimensions and `MediaRecorder` kept recording a portrait video
from a landscape sensor. The latch is gone: `arFitCanvas` compares the incoming
`AR.camW/camH` against `AR.capWH` and re-fits whenever they disagree, and the
orientation `change` listener clears `AR.capWH` so the very next bound frame
uses the new shape. That single change unblocks both *the map renders nothing
horizontal* and *a landscape take comes out portrait*, because the recorded
pixels and the coverage overlay both take their shape from the same canvas.

**"Cameras seen" was counting the selfie.** The start-screen note read
`LENS.devices.length + " cameras seen"` where the list was only filtered by
`kind === "videoinput"`, and until the operator opened a camera every label was
empty so `lensIsFront("")` could not exclude the front one either. On a
Samsung with three backs and one front, that is *4 cameras seen* above *three
buttons*. Two shapes now: labels on, count only `!lensIsFront(label)` and say
*3 rear cameras*; labels off, say *4 video inputs listed - start a scan to see
which is which*, because a phone that has not identified its cameras should not
pretend to have.

**A manual pick still needs the zoom dial.** Android exposes an ultrawide as a
distinct `deviceId`, but on this handset Chrome delivered the main sensor's crop
until `applyConstraints({ zoom: caps.zoom.min })` was also sent. V7's rule
"choosing it *was* the choice" was correct on paper and wrong on the phone, so
Camera 0 and Camera 2 showed the same narrow frame. `arPickLens` now reaches for
`zoom.min` whenever it is below 1.0, in either mode. The `test_lens_choice`
assertion that used to pin the old behaviour flipped to *manual picks still
reach the wide dial*.

**The rest of the UI follows the phone once it turns.** Landscape CSS covered
only the AR HUD before; `#perm-hero`, the hold and lens tab rows, `#dock`, and
the settings sheet kept their portrait shape and their 300 px paragraph in the
middle of a wide viewport. All four now have a landscape rule: the hero tightens
its gap and widens the copy, the tab rows stretch to `min(560px, 90vw)`, the
dock loses vertical padding, and the sheet docks right and slides in on the
x-axis instead of the y.

**The pipeline did not need changing.** `extract_keyframes` already reads the
frame's shape (`"portrait" if height > width else "landscape"`) and calibration
.json carries `focalLengthX/Y` and `imageWidth/Height` as one pinhole fact -
so once the handset records real landscape frames, the whole chain (COLMAP
solve, splat training, viewer export, coverage map) already follows the aspect
without a code change. The bug was never that vertical and horizontal needed
separate paths; it was that horizontal frames were not being produced.

**Verified on this machine:** 58 python checks, 38 landscape-hold (three new:
fullscreen is asked for when the hold is set, landscape CSS covers hero/dock/
sheet/tabs, and the canvas re-fits after a rotation), 46 lens-choice (three
new: manual pick reaches the dial, rear count excludes the selfie once labels
arrive, and before labels the note is honest that it cannot), plus the
existing AR-overlay / capture-scripts / coverage-map / element-id suites.



## V9.2 - cache-busting, a shell you can see through, and WebXR's lens limit named

Three things surfaced on the very next phone pass.

**Chrome on Android disk-caches HTML when there is no `Cache-Control`.** The phone
replays the same bytes for hours, so a fix that shipped looked identical to a fix
that had not. `_serve.py` now overrides `end_headers` and injects
`Cache-Control: no-store` whenever a response has not already declared its own
policy, and every reload from the handset actually hits this server. Every UI fix
also carries a build stamp on the start hero -- right now *"build V9.2 · shell
restored · AR-narrow warning · no-store cache"* -- so "the fix did not work" and
"the phone has not loaded the fix" are distinguishable without opening dev tools.

**The coverage shell is back, as a lattice not a wall.** An earlier pass pulled the
per-surfel quads because 1600 filled squares at alpha 0.92 buried the camera image.
Removing them over-corrected -- the coach line still said *"Red areas"* and *"Amber
areas"* with nothing on the screen to point at, and the operator missed the light
blue coverage signal they had learned to trust. V9.2 draws four thin line edges per
on-screen surfel oriented along the surfel's own normal, colour-coded by
`CoverageMap.stateOf` (thin red at 0.28, partial amber at 0.30, covered light blue
at 0.32). `test_ar_overlay.js`'s `coverFrac < 5%` budget is what keeps the view
readable, and the warm-pixel assertion flipped from *"the shell must paint nothing"*
to *"the shell must actually paint warm states"* -- which is the point of it.

**AR cannot pick a lens and now says so.** `lensTabLine()` reads the projection
matrix it already measures; when the horizontal FOV drops below 55 deg during an
AR session it returns *"narrow -- this phone's AR view gives only 38 deg and WebXR
exposes no lens dial; Basic mode + a manual pick reaches wider"* instead of the
earlier *"the phone's choice"*. On the operator's Samsung the ultrawide is behind
a deviceId the AR session never requests, so a manual pick under Basic mode is the
only path to a wider frame -- naming that dead end beats leaving it as an
unexplained *"still no wide angle"*.

**Verified:** 58 python + 18 ar-overlay (with the shell-flip) + 38 landscape-hold +
46 lens-choice + 130 capture-script + 110 coverage-map + 13 element-id = 413
checks all green.

## V10 - bots that wait in the blind spots of the scan itself

The viewer can now be played as a shooter. The toolbar's **Combat: ON** link (or
`?combat=1`) turns the same reconstructed room into an arena: a rifle with spread,
recoil and ADS, and up to a handful of bots that hold cover out of sight and
open up when their aim has wound up. Nothing about the operator path changed -- the
plain `viewer/pc.html` never fetches a gameplay module, which `test_combat_play.js`
asserts by watching the network log.

**Bought the mesh, hand-rolled the mind.** `recast-navigation` (MIT, WASM) runs
*offline only* -- `tools/navbake/bake.mjs <scene>` reads the collider GLB the
pipeline already exports and writes `work/<scene>/pc/nav.json` as a triangle list.
Shipping that WASM to the browser would have put a multi-megabyte download in front
of a phone on a field connection, so the runtime is pure JS: A* over a welded
triangle graph plus a greedy string-pull, and every smoothed chord is re-checked
against the mover's own legality rule, so a route that would clip a wall comes back
as `null` instead of as a walk through concrete. PlayCanvas stays: it is the only
renderer here that draws this project's splats.

**Splats occlude nothing, so sight does.** Gaussian splats write no scene depth and
the engine has no `colorWrite`, so a bot behind a wall would simply paint through
it. Instead the same line-of-sight ray used for perception (`world.js` `Sight`, one
ray per bot per frame, Ammo `BODYMASK_STATIC`) also decides whether that bot is
drawn and whether it can be shot. Tested: a bot placed behind real scan geometry is
culled, its meshes are switched off, and it cannot be hit through that wall.

**The ambush spots are measured, not authored.** Every walkable triangle is
LOS-tested against approach samples harvested from the capture itself -- the drone
flight path, `walk_path`, `cameras.json`, the viewer spawn -- and classified:
*ambush* (unseen from close by, seen from far, and never from the last third of the
approach), *overwatch* (sees most of the approach), *flank* (off the line, two ways
out). A bot's decision to fire is a product, not a timer: aim quality x how long it
has had the crosshair settled x how steady it is standing x how exposed its current
state leaves it, gated against a threshold, with a reaction delay scaled by skill.
That is what makes a bot that has just caught sight of you hold for a third of a
second instead of snapping.

**When the data is bad it says so.** `H` opens the AI-debug table (state, utility,
awareness, waypoint) and the HUD's probe button copies the whole world state as
text. If a scene has no usable floor near the player, combat reports the triangle
count it found and the exact re-bake command rather than deploying bots into the
void. `room_multi_video` is such a scene today -- its reconstruction drifts 50 m
vertically, so the walkable region around the spawn is a 67-triangle pocket; bots
stand and shoot there but cannot route across the room. `room_w_jsonl`, `temple` and
`rocks` bake clean.

**One clock, because the browser's lies.** The first live pass was flaky in a way
that pointed at the runtime, not the test: reaction and wind-up timings drifted
between runs. Combat stamped its timers with `performance.now()` while movement
integrated `dt`, and headless Chromium simply stops issuing frames a couple of
seconds into this page -- so a "wait 1.4 s" probe measured a frozen world. Every
combat timer now reads `Combat#t`, the accumulated simulation clock, which makes the
whole layer deterministic under hand-stepped frames and means a backgrounded tab no
longer makes bots forget you the instant they come back. The same lies are why the
browser tests step the world themselves: `Application.tick(timestamp)` takes dt from
the clock it is handed, so `pump()` in `scratch/test_features.py` feeds it timestamps
16.7 ms apart and a stride assertion measures simulated time that definitely passed
rather than hoping Chromium kept painting frames.

**Verified:** 58 python + 413 pre-existing + 34 combat-nav (pure Node, synthetic
wall with one doorway) + 50 combat-play per scan, driven headless in Chromium against
the real scans, 7 consecutive green runs across `room_w_jsonl`, `temple` and `rocks`.
One combat-play check remains scene-limited by data: `room_multi_video` has too
little connected floor to route a bot across.

## V10.1 - footage that never flew

`videos/Auditorium.mp4` is the first capture with no drone in it: 453 s of interior
dolly through a hall, 3840x2160 @ 29.97. Every scale in the pipeline was derived from
a drone, so it came apart in three places, each of them silently.

**`--vocab-tree` was parsed, stored, then dropped.** `indoor_large` sets
`loop_detection=True`, and `run_colmap.py` prints *"loop detection requested but no
vocab tree found - skipping"* and carries on at exit 0. The plan dict is built from a
key tuple and `vocab_tree` was not in it, so the flag never reached `plan.json` no
matter how it was passed -- no capture in this project's history has ever closed a
loop. It is in the tuple now, and `tools/vocab_tree.bin` is the default whenever that
file exists, because three presets promise loop closure and a missing default made
the promise a no-op. It is *not* what rescued the auditorium either: 900 keyframes
registered 11/897, 400 registered 3/400 on the first pass and 395/400 on the rescue
flags. Re-running this clip with retrieval actually on is still open.

**Scale had one ruler, and it was a speedometer.** `solve_frame.py` took
`--drone-speed 5` m/s x the clip duration / the reconstructed path length: 28.44
m/unit for a hall that is not 602 m across. Nothing questioned it, so it flowed into
a 363 x 140 x 328 m collider box and ~20 min of GPU time later `splat-transform` died
on `RangeError: Map maximum size exceeded` while extracting voxel faces at 0.25 m --
every splat is 15x fatter in metres than it should be, so the shell it voxelises is
~1000x the size. There are now two named rulers and they are mutually exclusive:
`--speed-anchor` in m/s, and `--height-anchor` in m -- how high the camera sat above
the ground it filmed. A dolly's speed is a guess; a camera at arm's length is a tape
measure. `--height-anchor 1.6` gives 1.92 m/unit, a 24 x 22 m hall with 5.7 m of
relief, and reports the implied camera speed back as 0.34 m/s, which is what a slow
dolly looks like. Pass neither and the drone pair stands, so the four existing scans
are bit-for-bit unchanged.

**A 5 m post buffer deletes a room.** With the geometry fixed, combat deployed three
bots and scored **zero** tactical nodes. Measured: the runtime keeps 109 of the 335
baked triangles, and every one sits within 3.2 m of an approach sample -- on an
interior walk the capture cameras already went everywhere, so the approach cloud
blankets the floor and no absolute distance counts as "away from it". The buffer is
now `min(5 m, this scan's own 40th percentile)`: unchanged on `rocks` (p40 12 m),
`temple` and `room_w_jsonl` (p40 ~6.5 m), relaxed to 1.9 m here, and it says so in a
HUD notice instead of quietly returning nothing.

**Verified on the auditorium:** the run is green from `frame` onward in 75 s (COLMAP
and training are cached), every world-gate check passes, the walk test covers 65.7 m
with 0 falls, the nav bake reports `PASS` at 177.2 m2, and 34 combat-nav + 50
combat-play checks are green in Chromium. It is a *small* arena: 44% of the grid has
multi-view support and the mesh lands in 37 islands with 89 m2 usable in the largest,
so bots hold blind spots behind the seating rather than flank across the hall. A
serpentine re-shoot with cross-ties -- which is what `indoor_large`'s advice already
says -- is the one thing that would grow this floor.

```
python pipeline.py run auditorium --video videos/Auditorium.mp4 --preset indoor_large \
  --quality high --target 400 --overlap 40 --height-anchor 1.6
cd tools/navbake && node bake.mjs auditorium --obj --cell 0.25 --radius 0.3 \
  --height 1.6 --climb 0.9 --slope 55 --region 0.6 && node verify.mjs auditorium
COMBAT_SCENE=auditorium node tests/test_combat_play.js
```


## V11 - a round you can watch, on a body that fits the room

Four complaints came back from playing the auditorium: the shot is invisible, the
character is a giant here and a hamster in my room video, and the ammo goes through
the chairs. Reading the code found a fifth thing none of them said out loud.

**There was no bullet to see.** `weapon.js` was pure instant hitscan, and the only
visual was a 1.8 cm box that faded over 0.09 s without ever moving. It now spawns a
real round: 70 m/s, flat, one pooled glowing streak each, damage resolving on impact
about 14 ms per metre after the trigger pull. That delay is the point. A round that
arrives later is a round you can step out of, and a hit finally lands where the
streak ended up. Bots fire through the same pool, so incoming fire is visible too.

The step designed itself wrong twice, and both times a measurement settled it. Slicing
each frame's 1.17 m into 0.25 m probes was justified as preventing tunneling: false,
because a raycast is a *segment* test, not a point sample, so one cast per frame
cannot skip a 20 cm box. The slicing was also what tunneling *would* have needed,
because the target test ignored entry distances under 12 cm, and successive probes
can put a sphere's surface inside that rejected band on every single step. One swept
segment, and a hit that counts from zero (a volume the round starts inside is point
blank, not a miss).

**A dust ball an arm's length away is not information.** A photograph of a round
hitting the player showed the entire screen painted brown: the impact puff sits on the
player's chest, which is where the camera is. Effects within 0.7 m of the eye are
simply not drawn now, and the muzzle flash is additive light instead of a normal-blended
sphere, which is what made it a disc you could not see through. Neither of those was
guessable from a number, and the round's own counters said everything was fine.

**The furniture was never in the collision group the combat rays ask for.** Measured
with a ray that starts above a seat and ends inside it: `filterCollisionMask: 2`
answered with the scan sheet 71 cm *below*, mask `16` answered with the box, and
`rigidbody.group` reported `2` for both bodies. Cause: 60 collision components hanging
off a rigid body that has no compound of its own each build their own body, and this
engine registers those in the trigger group. So every sight line, ground probe, step
test and round in combat flew through chairs while the plain raycast in the offline
test looked healthy. One line fixes it (`collision: {type: "compound"}` on the objects
root), and 360 probes across all 60 box tops now stop on the box under the mask combat
actually uses. The auditorium got real cover with that fix: 317 of 573 bot rounds in a
90 s test are stopped by geometry.

**The character was not too big, it was floating.** The render capsule spanned 0.78 to
2.58 m above the floor while its collider spanned 0 to 1.8 m, so the avatar stood
1.23 m above its own feet and read as half the ceiling. It is one measured skinned
model now (CesiumMan, CC0, 19 joints, one walk clip), scaled to 1.75 m from its own
bind box with its soles on the collider's floor, carrying the rifle the bots also
wear, which is what the barrel finally starts from. Two asset facts had to be measured
rather than assumed: the skinned world AABB is the union of per-bone boxes and reads up
to 10 cm taller mid-stride, so no amount of sampling converges on the height; and the
file is authored **Z-up**, so the mesh's local Y box is the figure's 1.14 m depth, not
its 1.51 m height. The first height assertion failed on exactly those two.

**Fixing the aim point turned the bots into perfect killers, which was the fifth
bug.** They aimed at `player.y + 1.05`, measured from the capsule *centre*, which lands
15 cm above the top of a 1.8 m collider: nearly all of their fire had been going over
the head, and the 9-18 damage per round had been tuned against that. With the round
hitting a real chest volume the measurement came back as 986 shots, 963 hits (98%),
first death 0.5 s from five bots, and separately one single shot in ninety seconds at a
player who was *walking*. Both were gates pointed at the wrong quantity: a target's
motion now costs accuracy, in an aim error that grows with range and with target speed,
instead of buying silence by suppressing the decision to fire at all. After: 98% hit on
a standing player, 45% on a walking one, and the standing number is what five rifles at
3.5 m should do. `?bots=2` is the knob if it is still too much.

**Verified:** 55 assertions in `scratch/test_features.py` green in Chromium on the real
scans, including the round tests (fires, advances, stops on the box top rather than the
sheet under it, stops on the sheet where there is no furniture, runs out of an explicit
range, pays damage on impact and not on the trigger, leaves no round unaccounted for)
and the isolation case that a scene with no accepted boxes never fetches `objects.json`.
The auditorium walk test covers 66.0 m with 0 falls, and the offline gate is unchanged
on all five scenes. Two of them still fail it honestly: `temple` routes a 12 m loop
(0.4% of its grid walkable, and a spawn 5.98 m above its own floor) and
`room_multi_video` an 8 m one. Both are scale, not collision, and that is the work in
flight: a scene's metres currently come from one scalar, and the room video fell back to
the *drone flight-speed* ruler, which is how a 1.8 m human ended up in a 35 m room.

```
python scratch/test_features.py          # boxes, character, rounds, in a real browser
python scratch/probe_lethality.py        # bot fire on a standing and on a walking player
python scripts/drive_viewer.py walk --asset work/auditorium/viewer_assets --out scratch/walk
```
