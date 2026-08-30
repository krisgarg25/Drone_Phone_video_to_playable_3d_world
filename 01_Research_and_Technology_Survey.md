# 01 — Research & Technology Survey
## Single-Pass Drone Video → Georeferenced, Walkable 3D Model

| | |
|---|---|
| **Version** | 1.0 |
| **Date** | 2026-08-24 |
| **Status** | Research verified against upstream repos & LICENSE files as of this date |
| **Companion docs** | `02_System_Architecture_and_Pipeline_Design.md`, `03_Roadmap_Evaluation_and_Business.md` |

---

## 1. Problem in one paragraph

From **one drone pass** (1080p/4K video + GPS + flight metadata; optionally IMU, barometric altitude, camera intrinsics, RTK/PPK), produce a **georeferenced, metrically accurate** textured mesh / point cloud of terrain, buildings, facades, roads, and vegetation — then make that model **first-person walkable**, with a **measurement/analytics layer** and **configurable scenario overlays** (urban-planning review mode, military mission-rehearsal mode with injectable entities). The pipeline must tolerate motion blur, compression artifacts, illumination change, dynamic objects, GPS noise, single-direction viewing geometry, and must run near-real-time on scalable hardware.

There are three technology families to choose from:

| Family | Examples | Strengths | Weaknesses |
|---|---|---|---|
| **Classical photogrammetry** (SfM+MVS) | COLMAP, OpenDroneMap, Metashape, Pix4D | Mature, cm-accurate, natively georeferenced, measurable meshes | Slow per-scene optimization; fails on blur/low texture; needs many overlapping views |
| **Deep SLAM / VO** | DROID-SLAM, DPVO/DPV-SLAM, ORB-SLAM3 | Real-time tracking, handles long sequences, robust odometry | Up-to-scale only; dense depth is heavy; drift without loop closure |
| **Feed-forward transformers + neural rendering** | VGGT family, Pi3, MapAnything, Depth Anything 3 + Gaussian Splatting (gsplat) | Seconds-not-hours geometry; splats render photorealistically from sparse/single-pass views; modern active ecosystem | Scale ambiguity (needs GPS fusion for meters); long-video chunking required; mesh extraction is its own step |

The winning architecture **combines all three**: feed-forward geometry for speed, SLAM-style pose-graph alignment for long videos, classical georeferencing math for metric accuracy, and Gaussian splatting for visual quality.

---

## 2. Deep dive: the four repos you proposed

### 2.1 VGGT — facebookresearch/vggt ⭐ 14.3k

**What it is.** *Visual Geometry Grounded Transformer* — CVPR 2025 **Best Paper**. A ~1B-parameter transformer that ingests 1-to-hundreds of images in a **single forward pass (<1 s)** and directly predicts: camera extrinsics + intrinsics, depth maps (+confidence), point maps (+confidence), world-frame point clouds, and tracked 3D point tracks. No per-scene optimization like COLMAP.

**Performance.** Handles ~100–200 frames/pass comfortably (a May 2026 memory fix raised frame capacity ~2–3×). Successor **VGGT-Omega** (CVPR 2026 Oral) benchmarks up to 500 frames — measured peak VRAM at 624×416 resolution on an A100: 1 frame ≈ 6 GB, 100 frames ≈ 13.4 GB, 200 frames ≈ 20.8 GB, 500 frames ≈ 43 GB. Requires Ampere-or-newer GPU for bf16 (your RTX 3050 qualifies architecturally but not in VRAM — see §6).

**Limitations for our use case.**
1. Output is **up-to-scale only** — no metric scale or absolute position without external GPS/baro fusion.
2. **Drift over long sequences** → must be chunked and pose-graph aligned (**VGGT-Long** pattern, below).
3. Motion blur degrades stability; assumes static scenes (moving vehicles corrupt point maps).
4. Practical VRAM ceiling conflicts with our 6 GB starting GPU.

**License — critical finding.** The repo switched in July 2025 from CC-BY-NC to the bespoke **"VGGT License v1"** (Llama-family): code and the gated **`VGGT-1B-Commercial`** checkpoint allow commercial use, **but the Acceptable Use Policy explicitly excludes military applications**, and the *original* checkpoint remains non-commercial. **If military reconnaissance stays in scope, VGGT weights are off the table** and we substitute BSD/Apache models (§4).

**Long-video extensions.**
- [**VGGT-Long**](https://github.com/DengKaiCQ/VGGT-Long) (ICRA 2026): chunked windowed inference + overlap alignment + Sim(3) pose-graph optimization + optional DBoW place-recognition loop closure → kilometer-scale sequences, no calibration needed, runs on a 24 GB RTX 4090 (~50 GB disk for a 4,500-frame sequence; sample ~1 fps). Crucially refactored to accept **Pi3 and MapAnything backends** — this is exactly the scaffolding our pipeline adopts.
- [**VGGT-Omega**](https://github.com/facebookresearch/vggt-omega): successor model; weights license unverified — treat as research-only until checked.

**Verdict:** ✔ Conceptually ideal geometry core for research/demo builds. ✘ Swap out for defense scope due to AUP military exclusion.

---

### 2.2 DROID-SLAM — princeton-vl/DROID-SLAM

**What it is.** NeurIPS 2021 Oral deep dense SLAM: learned optical flow feeding a **differentiable dense bundle-adjustment layer**, alternating flow/pose/depth updates. Outputs full-resolution depth maps + poses. Needs intrinsics as input.

**Performance.** Accurate and robust on forward-moving footage, but heavy: ≥11 GB GPU even for small benchmarks, 24 GB for large evals; slow at high resolution. Monocular scale ambiguity applies.

**License.** **BSD-3-Clause** (verified) — commercial OK, no military restriction.

**The better successor: [DPVO](https://github.com/princeton-vl/DPVO)** (NeurIPS 2023, **MIT**). Sparse-patch visual odometry: dramatically lighter and faster than DROID, real-time tracking, low memory, runs on "any video or image directory with one command", optional loop closure + DBoW2 backend for long trajectories. DPV-SLAM (ECCV 2024) extends it to full SLAM inside the same repo. This is the pragmatic choice for tracking an entire flight on modest hardware.

**Verdict:** ✔ DROID works but is superseded; **use DPVO/DPV-SLAM instead** (MIT, lighter, fits small GPUs better).

---

### 2.3 gsplat — nerfstudio-project/gsplat ⭐ 5.6k

**What it is.** Clean-room **Apache-2.0** CUDA rasterizer for 3D Gaussian Splatting (the legally-safe replacement for the original Inria rasterizer); rendering backend of Nerfstudio's `splatfacto`. Very active (pushed Aug 2026), [docs](https://docs.gsplat.studio/).

**Features we care about:** anti-aliasing (Mip-Splatting), MCMC densification, batching + extremely-large-scene support, multi-GPU distributed training, depth rendering, and — decisive for us — **first-class 2DGS surfel surface reconstruction** (`simple_trainer_2dgs.py` examples), which yields accurate, measurable surfaces while staying Apache-licensed. Library claims up to 4× lower training memory vs the original rasterizer. Typical large outdoor/drone scenes reach good quality in roughly **15–60 min on a 24 GB GPU**.

**Why gsplat matters beyond visuals:** every splat renderer ships zero physics/collision. Our extracted mesh doubles as (a) the measurable survey product and (b) the collision proxy + NavMesh source for the walkable scene. gsplat's 2DGS trainer gives us both from one Apache-licensed component.

**Verdict:** ✔✔ Keep — this is the right splatting engine, full stop.

---

### 2.4 splatwalk — EricEisaman/splatwalk

**What it is.** A young (created June 2026, **6 stars**) TypeScript/Rust→WASM project, MIT-licensed core, for making splat scenes walkable in the browser (Babylon.js / React-Three-Fiber integrations). Pipeline: ingest `.ply/.spz/.splat` → floater pruning → **floor extraction ("FastNav")** via floor-field / voxel-collision / marching-cubes algorithms → 2.5-D walkable column field → triangulated ground mesh → **Recast navmesh** for click-to-move agents. Exports `.glb` ground meshes and navmesh binaries.

**Honest assessment.** The *concept* (navmesh + ground collision derived automatically from splats) is exactly what we need, but the implementation is experimental: documented Y-axis/navmesh pitfalls, empty floors on large sparse outdoor scans, tiny community. **Use it as a reference implementation / algorithm source, not as our runtime foundation.**

**The universal pattern it confirms:** splats = visuals; a separately extracted mesh = collision + navmesh. Every production-grade alternative follows it:

| Route | License / status | Walkability mechanism |
|---|---|---|
| Web: [sparkjsdev/spark](https://github.com/sparkjsdev/spark) (three.js) | MIT, active Jul 2026 | Renderer only; add collision GLB + custom FPS controls |
| Web: mkkellogg/GaussianSplats3D | MIT, **no longer actively developed** | Orbit controls only, no collision |
| Editor: PlayCanvas SuperSplat | MIT, very active | Cleanup/compression editor, not an engine |
| Unity: aras-p/UnityGaussianSplatting | MIT, mature | Splat = render component → full CharacterController, Unity NavMesh AI available once colliders added |
| Godot: ReconWorldLab/godot-gaussian-splatting | MIT, active Aug 2026 | CharacterBody3D + NavigationRegion3D with supplied collision mesh |
| UE5: mlslabs renderer | Apache-2.0, active | Real-time 3DGS/4DGS; collision from separate meshes |
| UE5: xgrids LCC plugin | **No open license listed** — avoid unless licensed | Billion-scale streaming, nDisplay/VR/digital twins |

**Verdict:** ✔ Keep as algorithmic reference for auto-navmesh-from-splats; build walkability on Godot/Unity (engine route) or spark/three.js (web route) with our extracted mesh as collider.

---

## 3. Answering your question: "Are these four repos good enough?"

**They're a credible skeleton, but insufficient alone — and two need swapping depending on scope.**

| Repo | Keep? | Why |
|---|---|---|
| VGGT | ✔ for research/demo / ✘ for defense scope | Best-in-class feed-forward geometry, but AUP bans military use; original weights non-commercial |
| DROID-SLAM | ➖ Replace with DPVO/DPV-SLAM | Same authors' MIT-licensed successor is faster and lighter |
| gsplat | ✔✔ Core keeper | Apache-2.0, feature-complete, gives both visuals AND surface meshes |
| splatwalk | ✔ as reference only | Right idea, too immature to build a product on |

**Missing pieces you didn't list (and now have answers for):**

1. **Metric scale + georeferencing** — none of the four produce meters or WGS84 coordinates. Solved by GNSS/baro factor-graph fusion + pyproj transforms (Doc 2 §M3).
2. **A commercially-clean feed-front geometry model** — see §4: MapAnything-Apache, Pi3, Depth Anything 3.
3. **Keyframe extraction + per-frame GPS** from video (.srt/XMP parsing) — ODM proved this is viable natively.
4. **Dynamic-object removal** — segmentation (SAM2/YOLO) + inpainting (LaMa/ProPainter).
5. **Meshing legality** — popular mesh-from-splats repos (SuGaR, PGSR, GOF, official 2DGS) are **non-commercial Inria/ZJU licenses**; we use gsplat's own Apache 2DGS trainer or Open3D TSDF fusion instead.
6. **Streaming/digital-twin formats** — Cesium 3D Tiles, LAS/LAZ, GeoTIFF DSM/ortho exporters.
7. **Analytics** — measurement, viewshed/LOS, terrain derivatives (CloudCompare/GDAL-class functionality reimplemented on our mesh).

---

## 4. Newer & better alternatives (2025–2026 state of the art)

All licenses below were verified against the actual LICENSE files / model cards in August 2026:

| Model / repo | License | Commercial? | Why it matters for us |
|---|---|---|---|
| [**MapAnything**](https://github.com/facebookresearch/map-anything) (Meta, 3.7k★, active) | Code Apache-2.0; checkpoints in two flavors: `facebook/map-anything` = CC-BY-NC, **`facebook/map-anything-apache` = Apache-2.0** (identical functionality) | **YES (apache ckpt)** | Universal **feed-forward METRIC** reconstruction: accepts images ± intrinsics ± depth ± poses ± GNSS rays, outputs metric points/depth/poses/intrinsics/confidence. Up to 2000 views on big GPUs. Unified loader for VGGT/Pi3/DA3/etc. → **our default geometry backend** |
| [**Pi3**](https://github.com/yyfz/Pi3) (ICLR 2026, 2.1k★) | **BSD-3-Clause** | **YES** | Permutation-equivariant geometry transformer; strong VGGT-class drop-in; supported backend in VGGT-Long → **defense-scoped geometry core candidate #2** |
| [**Depth Anything 3**](https://github.com/ByteDance-Seed/Depth-Anything-3) (ByteDance, 6.2k★, active) | Code Apache-2.0; weights split: GIANT/LARGE/Nested = CC-BY-NC; **BASE/SMALL/METRIC-LARGE/MONO-LARGE = Apache-2.0** | YES (those variants) | Metric depth in meters + camera poses; beats VGGT on multi-view benchmarks; **DA3-Streaming does ultra-long video under 12 GB VRAM** (sliding windows) |
| DUSt3R / MASt3R / MUSt3R (NAVER) | CC BY-NC-SA / NAVER NC | NO | Foundational pointmap papers — cite, don't ship |
| Fast3R (Meta) | FAIR Noncommercial, repo archived | NO | Skip |
| CUT3R / Spann3R / TTT3R | NC / NC / unverified | NO | Streaming recon research references |
| MoGe-2 (Microsoft) | Code MIT/Apache; **weights gated, historically NC — verify per checkpoint** | Unclear | Metric monocular pointmaps; useful fallback |
| [COLMAP](https://github.com/colmap/colmap) | BSD (ETH/UNC) | YES | Gold-standard baseline & refinement; GLOMAP's fast global SfM was folded into COLMAP releases (GLOMAP repo deprecated Jan 2026) |
| [OpenDroneMap](https://github.com/OpenDroneMap/ODM) | **AGPL-3.0** (network copyleft) | Usable internally; copyleft if offered as SaaS | Since **v3.0.4 ODM natively extracts frames from drone video (.mp4/.mov/.lrv/.ts) paired with matching .srt GPS subtitle tracks** — proof the video-first workflow works; also our classical accuracy baseline (outputs GeoTIFF ortho, DSM, LAZ cloud, geo OBJ mesh) |
| Agisoft Metashape / Pix4D | Commercial ($179–$3,499 perpetual / subscriptions) | YES | Reference-quality baselines for accuracy benchmarking; neither ingests video natively |

**License landmines found during research (avoid in shipped product):**

| Component | License problem |
|---|---|
| VGGT original checkpoint / VGGT-Omega weights | Non-commercial / unverified; AUP bars military |
| SuGaR, GOF, official 2DGS repo | Inria Gaussian-Splatting license — non-commercial |
| PGSR (zju3dv) | Custom ZJU academic license (email approval needed) |
| DUSt3R/MASt3R/Fast3R/CUT3R/Spann3R | Non-commercial research licenses |
| xgrids UE5 splat plugin | No open-source license published |

**Safe replacements:** gsplat's built-in 2DGS surfel trainer (Apache), Open3D TSDF fusion (MIT), Pi3 (BSD), MapAnything-apache (Apache), DA3 metric variants (Apache), DPVO (MIT), COLMAP (BSD), Open3D (MIT), PDAL (BSD), pyproj (BSD), CesiumJS (Apache).

### 4.1 Prior art: [ch1bo/drone-reconstruction](https://github.com/ch1bo/drone-reconstruction) (studied 2026-08-24)

An independent hobbyist pipeline (created Jan 2026, 7★, ~40 commits, Python/Nix) doing almost exactly our Phase-1 scope: monocular DJI video → ffmpeg frames @2 fps → **COLMAP sequential matching** → `.SRT` telemetry parsed to ENU reference poses (`srt_to_reference_poses.py`) → `colmap model_aligner` **Sim(3)** fit preferring the drone's infrared/barometric `rel_alt` over GNSS altitude → CUDA PatchMatch MVS → optional Nerfstudio `splatfacto` (~30 min train) with browser viewer. Neighborhood scale, real-world meters, map overlay — no license file, no dynamic-object handling, viewer-only interactivity (no true FPS walking), no evaluation harness.

**Why it matters to us — four field-validated lessons that de-risk our design:**
1. `rel_alt` (IR/barometric, ~0.1 m precision) beats GNSS altitude (~10–20 m error) for vertical anchoring — confirms our M3 baro-fusion choice empirically.
2. **Sequential matching** is the right COLMAP mode for flyover footage (consecutive frames share most features); exhaustive matching wastes O(n²).
3. GPS-aligned models are already Z-up ENU → Nerfstudio needs `--assume-colmap-world-coordinate-convention False`; pin this gotcha in our M4 notes.
4. Splat training on neighborhood captures lands around **~30 min** on desktop GPUs — matches our latency budget assumptions.

**Gaps vs our architecture (why we don't just adopt it):** COLMAP-per-scene optimization instead of feed-forward chunks (slower, weaker on blur); no moving-object suppression; no RTK path; no accuracy benchmarking; no walkable/analytics/scenario layers. **Legal note:** the repo has *no license* = all rights reserved — study its techniques and scripts (especially the SRT→ENU parser logic and alignment flags) but write our own implementation; do not copy code into our product.

---

## 5. Georeferencing & metric accuracy without GCPs — what's achievable

| Sensor setup | Horizontal accuracy (typical) | Vertical accuracy (typical) | Notes |
|---|---|---|---|
| Standalone GNSS (no RTK), no GCPs | ~1–3 m | ~2–5 m (often worse) | Camera-center GPS tags are noisy; altitude worst |
| **RTK/PPK direct georeferencing, no GCPs** | **~2–5 cm** | **~3–8 cm** | Literature-consistent ranges; vertical ≈ 1.5–2× horizontal error |
| + 1 surveyed checkpoint | ~2–4 cm | ~3–6 cm | Cheap insurance; removes residual datum bias |
| Full GCP network (classical) | 1–3 cm | 2–5 cm | What Metashape/Pix4D achieve — our upper bound |

Method (detail in Doc 2 §M3): solve reconstruction in a local frame → robust **Sim(3)** alignment of camera centers to GNSS positions in a local ENU frame → transform via pyproj (WGS84 ↔ UTM/ECEF ↔ ENU) → fuse barometric altitude for relative vertical consistency → optional factor-graph smoothing (GTSAM/Ceres) when IMU present. Feed-forward models with **metric** heads (MapAnything-apache, DA3-METRIC) give a second, independent scale estimate to cross-check GNSS-derived scale.

Per-frame GPS sources for DJI-class drones: MP4 **XMP tags** (GpsLatitude/GpsLongitude/AbsoluteAltitude/RelativeAltitude/GimbalPitch…), companion **.SRT subtitle files**, EXIF of extracted frames — parsed with exiftool/ffmpeg + a small parser.

## 6. Hardware reality check (starting GPU: RTX 3050 6 GB)

- The 3050 is Ampere (bf16-capable) but **6 GB rules out large-chunk feed-forward inference and splat training**. Even VGGT-Omega's single-frame pass peaks near 6 GB at moderate resolution.
- **Design decision:** make every compute stage a swappable backend (`--backend local|cloud`). Local: ingest, keyframing, GPS parsing, masking, low-res/small-chunk geometry experiments, DPVO tracking (low memory), viewers, analytics UI. Cloud burst (RTX 4090 ≈ $0.35–0.70/hr, A100 80GB ≈ $1.30–2.50/hr on 2026 rental markets — approximate): chunked reconstruction, gsplat training, mesh/texturing.
- Scaling path documented in Doc 2 §7: 6 GB dev profile → 24 GB workstation profile → cloud batch profile → (stretch) Jetson Orin edge profile using DA3-Streaming-class models.

## 7. Final recommended stack

| Stage | Primary choice | Alternate / notes |
|---|---|---|
| Frame + GPS ingestion | ffmpeg keyframing (blur/sharpness-scored) + .srt/XMP parser | Validate against ODM's native video ingestion |
| Moving-object cleanup | YOLO/SAM2 masks → LaMa (image) / ProPainter (video) inpaint | Optional stage; off by default for speed |
| Tracking (whole flight) | DPVO / DPV-SLAM (MIT) | Real-time, low-memory, loop closure |
| Geometry (chunks) | **MapAnything-apache** (Apache) via VGGT-Long-style chunking + Sim(3) pose graph | Pi3 (BSD) for defense scope; VGGT-1B-Commercial only for civilian research demos |
| Splats / surface | **gsplat** splatfacto + 2DGS surfel trainer (Apache) | Open3D TSDF fusion from scaled depths as mesh fallback |
| Mesh + texture | Marching cubes / TSDF mesh ← scaled depth, texture bake from frames | Decimate + Draco/meshopt for runtime |
| Georeferencing | Robust Sim(3) to GNSS ENU + baro; pyproj WGS84↔UTM; GTSAM if IMU | Single checkpoint QA recommended |
| Exports | GLB, LAS/LAZ (PDAL), GeoTIFF DSM/DTM/orthophoto, Cesium 3D Tiles (py3dtiles/Cesium ion), splat PLY/SOG/SPZ | OSGB for legacy GIS consumers |
| Walkable runtime | Godot or Unity (MIT splat plugins) w/ colliders from our mesh + NavMesh; web viewer: spark + splatwalk-derived navmesh | Both routes share the same exported assets |
| Analytics | Distance/area/volume on mesh; LOS/viewshed + slope/aspect from DSM; annotation/report export | CloudCompare-class features in-app |
| Scenario layer | Config-driven presets (JSON/YAML): urban-planning mode, mission-rehearsal mode with entity templates, patrol paths (NavMesh), threat/visibility rings from DSM | Doc 2 §M9 |

**Bottom line:** your instinct was right — gsplat + the VGGT-*idea* + splatwalk's navmesh idea + a SLAM tracker is the correct skeleton. The research upgrades it to: **DPVO (tracking) → MapAnything-apache/Pi3 (chunked metric geometry) → gsplat 2DGS (splats+surface) → georeferenced exports → engine-based walkable scene with analytics and scenario presets**, with licensing clean enough to sell, including into defense.
