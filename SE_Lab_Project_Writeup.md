# SE LAB PROJECT — SUBMISSION

---

## 1. PROJECT TITLE

### **WALKABLE: Turning a Single Handheld or Drone Video into a Metric, Real-Time Walkable 3D World**

*Subtitle:* A reproducible Structure-from-Motion → 3D Gaussian Splatting → browser-physics pipeline that runs end-to-end on a 6 GB laptop GPU.

| | |
|---|---|
| **Course** | Software Engineering Laboratory |
| **Submitted by** | ______________________  (Roll No. __________) |
| **Batch / Section** | ______________________ |
| **Project guide** | ______________________ |
| **Date of submission** | 05 / 09 / 2026 |
| **Repository** | `Drone to 3d mesh` — Python 3.12 + 3.10, JavaScript (ES2022), ~18 000 lines of first-party code |

---

## 2. PROJECT WRITE-UP

### 2.1 Abstract

Most systems that rebuild a real scene into 3D stop at a *picture of a model*: you can orbit it, but you cannot stand inside it, and nothing about it knows what a metre is. This project builds the other half of the problem — a **single-command pipeline that takes one ordinary video of a place and returns a world you can walk through in a browser**, with gravity, collision, a metric ground surface, and an autonomous test character that proves the floor is real.

One command (`mvp.bat run <scene>`) drives eighteen pipeline stages: keyframe selection, camera-pose solving with COLMAP, Gaussian-splat training on **gsplat**, metric-scale recovery, ground-surface extraction, collision-mesh generation, and a headless physics walk test. Every stage writes typed artefacts to disk, is resumable, and is checked by an eleven-assertion quality gate that fails the build rather than shipping a silently broken world. The result is verified by **blind A/B evaluation** (a reviewer sees real photo next to engine render, order randomised) and by scripted traversal measured in metres walked and falls sustained.

Both self-defined acceptance bars were met under blind review: the visual bar with 10/10 blinded pairs judged to be the same scene with no disqualifying artefact, and the walkability bar with **65.3 m walked, 16/16 waypoints reached, 0 falls** — up from 19.9 m and 1/20 waypoints before the corrections described in §2.7.

### 2.2 Problem Statement and Motivation

Reconstructing 3D scenes from video is a solved problem in a laboratory and an unsolved one in practice. Commercial photogrammetry tools are accurate but slow, expensive, and produce outputs that are *viewed* rather than *used*. Research systems produce beautiful novel views but no surface you can measure, stand on, or navigate. Meanwhile, the hardware reality for a student lab is a laptop with a 6 GB GPU and no cloud budget.

The project therefore attacks four concrete failures of the naive approach:

1. **Geometry without a ground plane.** A Gaussian-splat scene is a cloud of ~500 000 translucent ellipsoids. It renders beautifully and collides with nothing — there is no surface, no floor, no "down".
2. **No metric scale.** Pure structure-from-motion recovers a scene only *up to an unknown scale factor*. Without GPS or surveyed markers, nothing distinguishes a 4 m room from a 40 m hall.
3. **Fragile solvers that fail quietly.** COLMAP rejects frames whose dimensions disagree with its shared camera model as a *warning* and exits with code **0**. A scene containing one portrait and one landscape clip loses half its footage and reports success. Training then dies four GPU-hours later with a bare Windows `0xC0000409` and no traceback.
4. **Hardware hostility.** Feed-forward reconstruction transformers that dominate the 2025–26 literature (VGGT, MapAnything, Pi3) need 16–24 GB of VRAM. They cannot be the foundation of a pipeline that must run here.

### 2.3 Objectives

| # | Objective | Status |
|---|---|---|
| O1 | Video file → registered camera poses → trained splat scene, one command, no cloud | Achieved |
| O2 | Recover a **metric** scale from a single operator-supplied physical number | Achieved |
| O3 | Derive a **walkable collision surface** from a splat cloud, with gravity and blocking handled by a real physics engine | Achieved |
| O4 | Automated, falsifiable acceptance tests for **visual quality** and **walkability** | Achieved |
| O5 | A capture assistant that prevents bad footage *before* it is filmed (phone AR coverage guidance) | Achieved |
| O6 | Run every stage on an RTX 3050 6 GB laptop | Achieved — peak 1.26 GB at the `high` preset |
| O7 | Interactive layer: first-person viewer plus an autonomous bot layer over a baked navmesh | Achieved |

### 2.4 Technology Survey and Architectural Choice

Three technology families were compared (full survey: `01_Research_and_Technology_Survey.md`):

| Family | Representatives | Strength | Why it is not sufficient alone |
|---|---|---|---|
| Classical photogrammetry (SfM + MVS) | COLMAP, OpenDroneMap | cm-accurate, measurable mesh, mature | Overnight runtimes; collapses on blur and low texture |
| Deep SLAM / visual odometry | DROID-SLAM, DPVO, ORB-SLAM3 | Real-time, long-sequence tracking | Scale-ambiguous; dense depth is expensive |
| Feed-forward transformers + neural rendering | VGGT, MapAnything, gsplat | Seconds instead of hours; photoreal sparse-view renders | VRAM-hungry; scale ambiguity; mesh extraction is a separate unsolved step |

The delivered architecture deliberately **combines the classical and the neural half, and rejects the expensive half**. Camera poses come from COLMAP 4.1.1 with GPU SIFT and sequential matching — the slow, well-understood, 6 GB-friendly option. Appearance comes from **3D Gaussian Splatting**, which is the only representation that stays photorealistic from a single sparse pass. Everything downstream of the splats is classical computational geometry: voxelisation, rasterisation, heightfields, A*.

The consequence is a design rule that shaped the whole project: *the neural stage is allowed to be only an image-synthesis stage.* It is never trusted with geometry, with "down", or with the metric scale.

### 2.5 System Architecture

```
 video (.mp4, any number of clips)   [+ optional ARCore/ARKit pose log .jsonl]
        |
        v
 [1] keyframes .... OpenCV decode, variance-of-Laplacian sharpness score,
                    temporal-bin selection, near-duplicate rejection
 [2] priors ....... phone AR positions injected as COLMAP pose priors  (optional)
 [3] colmap ....... GPU SIFT -> sequential/exhaustive/spatial matcher -> mapper
 [4] poses ........ camera centres, image points, intrinsics -> TXT export
 [5] train ........ gsplat 1.5.3, SH degree 3, SSIM+L1, MCMC-free densify, hard cap
 [6] frame ........ gravity from gimbal, metric scale from named ruler -> frame.json
 [7] export ....... reorient + rescale; splat means -> ground heightfield + coverage.u8
 [8] sky/cloud .... bimodality-tested canopy and cloud-sea culling; re-export
 [9] colors ....... vertex-colour the grid from the splat's own colours
[10] collider ..... splat-transform cluster_shell (0.25 m voxels) -> GLB trimesh
[11] objects ...... semantic box extraction (furniture) from the heightfield
[12] surface ...... two ground candidates measured, the walkable one shipped
[13] gate ......... 11 assertions; failure stops the build
[14] evals/pairs ... offline renders from training poses -> blinded A/B stacks
[15] walktest ..... headless Chromium + PlayCanvas: scripted 50 m traversal
        |
        v
 work/<scene>/viewer_assets  ->  browser: splats + ammo.js rigid-body capsule
```

**Design principles.** (i) Every stage is a separate CLI module with typed artefacts on disk, so any run is resumable (`--from <step>`), re-runnable in isolation (`--only a,b`), and inspectable. (ii) Completion markers live in `work/<name>/.pipeline/`, and re-running a stage invalidates everything downstream — a step cannot be skipped silently. (iii) Configuration resolves in three layers, later winning over earlier: capture-style **presets** (`room` / `drone` / `object` / `auto`) → **smart rules** derived from footage diagnostics → explicit **CLI flags**. (iv) Nothing is shipped that a machine has not measured.

### 2.6 Implementation

**Runtime and stack.**

| Layer | Component | Notes |
|---|---|---|
| Ingest | OpenCV, ffmpeg 9.0.1 | sharpness-ranked keyframing |
| SfM | COLMAP 4.1.1 (CUDA build) | driven by `scripts/run_colmap.py`, not by shell quoting |
| Training | PyTorch 2.4.1 + gsplat 1.5.3 | prebuilt Windows wheel: **zero compilation** |
| Geometry | numpy, plyfile, `@playcanvas/splat-transform` | heightfields, voxel shells |
| Runtime | PlayCanvas engine + ammo.js (Bullet) WASM | vendored; no build step |
| Navigation | `recast-navigation` (MIT), **offline bake only** | `tools/navbake/bake.mjs` → `nav.json` triangle list |
| Frontend | Vanilla ES2022 + WebXR | ~10 100 hand-written lines |
| Test harness | Playwright + headless Chromium | drives the real page, not a mock |

**Quality presets, measured on the target card.**

| Preset | Train width | Steps | Gaussian cap | Peak VRAM |
|---|---|---|---|---|
| `standard` | 640 px | 12 000 | 350 k | 0.72 GiB |
| `high` (default) | 1280 px | 15 000 | 3.0 M | 1.26 GiB — 14 min on a temple pass |
| `ultra` | 1440 px | 45 000 | 4.0 M | for 8 GB+ GPUs |

The cap is headroom, not a target: densification converges where the scene runs out of detail (the temple converged at 451 k splats under a 3 M cap).

**Phone AR pose priors — the highest-value addition.** COLMAP fails on rotation-heavy handheld video because spinning in place produces no parallax, so the pair geometry is degenerate and the mapper drops the segment. If a phone logs ARCore/ARKit positions during recording, those become **native COLMAP pose priors**: even a weak-parallax segment now carries a metric translation, so registration survives *and* the scale is real. Verified end-to-end by `tests/selftest_priors.py`: a synthetic 12 m AR walk reconstructs at scale **1.0000**, maximum camera-centre deviation **7 mm**, **30/30** frames registered. Priors also let several clips share one world via COLMAP `spatial` matching.

**The metric scale chain.** `solve_frame.py` accepts exactly one named ruler and refuses ambiguity: `--speed-anchor <m/s>` (drone cruise × clip duration ÷ reconstructed path length) or `--height-anchor <m>` (how high the camera sat above the ground it filmed). Both land in `frame.json`, the single source of truth for orientation and scale, which then propagates through the heightfield cell size, the collider voxel floor, and the navmesh bake radius. An earlier revision took "up" from a RANSAC plane normal whose sign is arbitrary — the whole scene exported upside down. Gravity now comes from the gimbal attitude, and the scale chain is documented test-first in `tests/test_reset_stability.py`.

**The capture console.** Reconstruction quality is decided before any code runs, so `viewer/capture.html` is a WebXR app that scans the room on the phone and shows a coverage *shell* — each measured surface patch coloured red / amber / blue by how many distinct viewing directions have hit it. A patch turns "done" only after two spots far enough apart (a real baseline). The overlay draws only thin world-space lines over the camera image; nothing is ever burnt into the recorded video, because a painted arrow is a feature COLMAP will happily match. The page also measures the lens it was handed: WebXR exposes no lens dial, so the projection matrix is read (`P[0] = 1/tan(half-FOV)`) and a horizontal FOV under 55° is reported as *narrow — a telephoto is recording your room*.

### 2.7 Engineering Challenges and Their Resolution

Five defects are worth reporting because in each case the intuitive fix was wrong and a measurement settled it.

**(a) The character was standing on the sky.** Two disjoint collider shells existed at the spawn — real ground at −16…−11 m and a crust of canopy splats at +5…+9 m. Every downward ray hit the crust first, so the capsule stood on it and the ground underlay draped over it. That is the "floating above a white sheet" screenshot. Fix: `strip_sky.py` decides whether to cut by testing the height histogram for **bimodality, not sparseness** — band density falls 2159 splats/m at +4 m to 135/m at +12 m and *recovers* to 1727/m by +19 m; an earlier "is this band sparse?" version fired on a smooth decline and would have deleted 86 % of a good frame.

**(b) The collider was geometrically unwalkable.** The voxel shell is a set of axis-aligned faces in which every height change is a vertical wall — measured at 1.05 m of riser every 0.6 m along the route. A capsule of radius 0.34 m meets a flat vertical face at its equator, so the contact normal is horizontal and there is *no lift term at all*. No amount of physics tuning fixes this; the surface must be wrong-free. Fix: `tune_collider.py` builds two candidate grounds (clipped shell top face vs. exported heightfield), **routes on both, and ships whichever the autopilot walks further on**, with ties going to the heightfield. It then re-runs the router on the shipped mesh so the physics surface, the planned route and the drawn underlay cannot disagree — verified in-browser as 140 probe rays landing on `ground.f32` at a median of 0 cm. Across five scenes the winner is not constant: auditorium 68 m (heightfield) vs 51 m; rocks 94 m (shell) vs 69 m; temple 12 m vs 6 m. No local roughness statistic predicts any of this, which is precisely why measurement replaced the threshold.

**(c) The autopilot livelocked, and the arithmetic proved it.** Its unstick timer was never cleared once armed, so every frame below 0.35 m/s re-armed a 1.4 s spin and waypoint steering was never reached. The log settles it: the walk leg ended at 288.6 rad of yaw, and 288.6 / 2.2 rad s⁻¹ = 131 of its 149 seconds spent spinning in a 1.18 m circle. Fix: timer reset on progress, plus a 12 s per-waypoint timeout so one unreachable corner cannot consume a leg.

**(d) A half-cell indexing error in three places.** Every rasteriser here indexes with `floor((x − ox) / cell)`, so a sample belongs at the cell *centre*, `ox + (j + 0.5) · cell`. Three sites used the corner. One of them made the viewer's idea of the ground sit **0.101 m** away from the surface the physics actually had under the player — the single number that should be identically zero, because `ground.f32` *is* the collider. None of these produced a visible failure, because probe bounds carry metres of slack; that is exactly why they are written down.

**(e) The "landscape video crashes" report was never about landscape.** Six cases were run one variable at a time directly against the rasteriser. 2000 gaussians render and back-propagate in both orientations, at even and odd resolutions; 2 gaussians are fine; **0 gaussians kill the process instantly with no output**, in both shapes. The root cause is that too few registered frames triangulate too few seed points, and the densifier prunes the last one at its first refine (step 600 — exactly where the log stopped). The log had said `init 2 gaussians` then `N 0` the entire time; a number at the end of a line does not read as a cause. Three guards now exist: COLMAP refuses to continue under `max(8, 10 %)` registration, `train_splat.py` refuses to start on a seedless reconstruction, and a mid-run collapse exits 1 with `COLLAPSED at step N`.

### 2.8 Software Engineering Practices

Because this is a software-engineering submission as much as a graphics one, the process is part of the deliverable.

- **A gate, not a checklist.** `check_world.py` runs eleven falsifiable assertions: heightfield coverage **split by how it was obtained** (measured / camera-derived / nothing — averaging them let a 30 %-measured grid report 98 %), terrain-is-a-floor-not-a-ceiling, cameras above the ground they filmed, spawn supported and flat, collider has no ceiling slab, collider *is* the array the route was planned on, route ≥ 15 m long, and more. Two of eight scenes still fail the route check today (`temple` 12 m, `room_multi_video` 8 m) and the build reports that honestly rather than hiding it.
- **Regression suites.** Twelve test files: `tests/run_tests.py` plus eleven JS suites driven in real Chromium (`test_ar_overlay`, `test_capture_scripts`, `test_coverage_map`, `test_combat_nav`, `test_combat_play`, `test_element_ids`, `test_landscape_hold`, `test_lens_choice`, `test_reset_stability`, `test_trainer_speed`, `selftest_priors`). The version-11 verification pass reports **58 + 413 + 34 + 50 + 55 ≈ 550 assertions green**, of which the browser suites exercise the actual scans rather than fixtures.
- **Static guards against whole bug classes.** `test_element_ids.js` fails the build when a `getElementById` has no matching markup, when a dead id survives, or when a `src=` file is missing — a category first seen as a runtime crash in the AR depth panel. A separate gate extracts the ~410 operator-visible UI strings and fails on any banned piece of jargon (*surfel, azimuth, voxel, 6-DoF, calibrating*), so clarity cannot quietly rot.
- **Blind evaluation as a protocol.** `results/blinded/` holds composite stacks of real frame vs. engine render with the order recorded separately in `results/pair_key.json`; a fresh-context critic scored both bars without ever seeing which was which. Visual quality is thus judged by a human under controlled conditions rather than by a PSNR number chosen to look good.
- **Instrumented observability.** Every stage appends to `work/<name>/logs/<NN>-<step>.log` with an `[exit E] Ss` footer. The dashboard reads those logs through a byte-offset cursor instead of holding a process handle, so **a run started in a plain terminal shows up identically in the browser dashboard**, and reloading mid-run replays the whole history and keeps streaming.
- **Version discipline.** Eleven numbered iterations (`V1 … V11`) are written up in `README-MVP.md` with the measurement that changed each decision — including two recorded instances of a stated hypothesis later being shown wrong by evidence.
- **Licensing posture.** Every third-party component is MIT / BSD / Apache; the vendored engine and physics WASM carry no build step, and the AGPL alternative (OpenDroneMap) was surveyed but excluded from shipped code paths.

### 2.9 Results

| Scene | Capture | Registered | Splats | Walk test |
|---|---|---|---|---|
| `rocks` | 12 s drone clip | 114 → 272 frames after tuning | 128 k → 451 k | **65.3 m, 16/16 waypoints, 0 falls**, foot gap +0.008 m |
| `temple` | drone, above cloud layer | — | 76 804 fog candidates culled | spawn moved −18 m (inside fog) → **+8.8 m courtyard**; 65.4 m, 0 falls |
| `room_w_jsonl` | handheld + AR poses | 95 % | — | 60 m on heightfield ground |
| `auditorium` | 453 s interior dolly, 4K | 395/400 on rescue flags | — | **65.7 m, 0 falls**; nav bake PASS at 177.2 m² |
| `room_multi_video` | mixed portrait + landscape | 61 % | — | 8 m — fails the gate honestly (50 m vertical drift) |

Walk-test improvement on `rocks`, same harness before and after the §2.7 fixes:

| Metric | Before | After |
|---|---|---|
| Distance walked | 19.9 m in 215 s | **65.3 m in 32 s** |
| Waypoints reached | 1 / 20 | **16 / 16**, all within 0.8 m |
| Travel efficiency | 10 % | **69 %** outbound, 100 % return |
| Feet vs. surface | −1.8 m (inside a crust 20 m up) | **+0.008 m** |
| Viewer ground vs. physics ground | 0.101 m apart | **0.000 m** |

**Interactive layer.** Beyond first-person traversal (WASD, run, jump, spawn, autopilot), the viewer loads an optional combat module: `recast-navigation` bakes a navmesh offline, a pure-JS A* with string-pulling routes bots at runtime, and every walkable triangle is line-of-sight tested against approach samples harvested from the capture itself — so *ambush*, *overwatch* and *flank* positions are measured from the scan rather than authored. Splats write no depth buffer, so sight-line rays against the collision mesh decide whether a bot is drawn and whether it can be hit. If a scene has too little connected floor, the layer reports the triangle count and the exact re-bake command instead of deploying bots into the void.

### 2.10 Limitations

Honest limits, each of which is a known open item rather than a hidden one: **scale is not measured, only named** — no GPS means metres are only as good as the operator's single number, and nothing can distinguish a wrong value from a wrong-but-self-consistent one. The shipped ground is a **heightfield**, so overhangs and caves cannot be expressed (a voxel shell can, and is unwalkable — the trade is deliberate). Dynamic objects have no masking stage. Low-texture grass makes far-field soft. Two of eight scenes fail the route gate because of scale drift, and the fix — per-region rather than per-scene scale — is the work in flight.

### 2.11 Conclusion

The research contribution of 3D Gaussian Splatting has been *photorealistic novel-view synthesis*. This project's contribution is the unglamorous half that makes it usable: a metric frame, a defensible ground surface, a collision mesh a capsule can actually stand on, and a build system that refuses to call any of it finished without a measurement. The engineering lesson repeated itself eleven times and is the part worth submitting — **the intuition was usually wrong, the log was usually right, and the number at the end of the line was the cause.**

### 2.12 Deliverables in the Repository

| Artefact | Location |
|---|---|
| Pipeline runner (18 stages, resumable) | `pipeline.py`, `mvp.bat` |
| Stage implementations | `scripts/` — 35 modules, ~6 800 lines |
| Offline navmesh baker | `tools/navbake/` |
| Walkable viewer + combat layer | `viewer/pc.html`, `viewer/pc.js`, `viewer/pc/scripts/` |
| Phone capture console (WebXR) | `viewer/capture.html`, `viewer/coverage_map.js` |
| Run dashboard | `viewer/pipeline_gui.html`, `_serve.py` |
| Test suites | `tests/` — 12 files, ~550 assertions |
| Blind A/B evidence and verdicts | `results/blinded/`, `results/verdicts/` |
| Research survey / architecture / roadmap | `01_…` `02_…` `03_…` `.md` |
| Per-iteration engineering log | `README-MVP.md` (V1–V11), `PROBLEM-temple.md` |
| Per-scene artefacts | `work/<scene>/` — poses, splats, heightfield, collider, logs, walk frames |

---

*Note — convert to PDF with any Markdown tool (VS Code + Markdown PDF, Typora, or pandoc). Tables are plain GitHub Markdown; the architecture diagram is ASCII so it survives conversion without a Mermaid renderer.*
