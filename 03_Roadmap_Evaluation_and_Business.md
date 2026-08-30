# 03 — Roadmap, Evaluation Criteria & Business Plan
## Single-Pass Drone Video → Georeferenced, Walkable 3D Model

| | |
|---|---|
| **Version** | 1.0 |
| **Date** | 2026-08-24 |
| **Companion docs** | `01_Research_and_Technology_Survey.md` (component choices), `02_System_Architecture_and_Pipeline_Design.md` (architecture) |

---

## 1. Desired Output specification (fills the problem-statement placeholder)

| # | Product | Format | Georeferenced | Acceptance threshold (v1) |
|---|---|---|---|---|
| O1 | Textured 3D mesh | GLB (+OBJ interchange) | WGS84 / UTM, EPSG stamped | Visually coherent; measurable; opens in Cesium/engine |
| O2 | Point cloud | LAS/LAZ | Yes | Density ≥ 20 pts/m² on surfaces within 80 m of flight path |
| O3 | Splat scene | PLY / SOG / SPZ | Yes (transform stored) | Photorealistic novel views at walkable eye height |
| O4 | DSM / DTM / orthophoto | GeoTIFF | Yes | 1×–2× GSD of source video at reference distance |
| O5 | Streaming tileset | Cesium 3D Tiles | Yes | Streams in browser globe without local install |
| O6 | Walkable scene package | collision GLB + navmesh + config | Yes | First-person traversal with gravity/collision; web + engine builds |
| O7 | Analytics report | HTML/PDF + GeoJSON annotations | Yes | Distance/area/volume tools validated against known baselines |
| O8 | QC & provenance report | HTML + JSON sidecars | n/a | Accuracy estimates, coverage map, failure flags per stage |

## 2. Evaluation Criteria (fills the problem-statement placeholder)

| Criterion | Metric | Definition | Target (RTK capture) | Target (standalone GPS) | Method |
|---|---|---|---|---|---|
| Georeferencing accuracy | Checkpoint RMSE (H/V) | Horizontal & vertical error at surveyed checkpoints not used in processing | H ≤ 10 cm, V ≤ 15 cm | H ≤ 3 m, V ≤ 5 m | RTK-surveyed checkpoints; compare model coords vs survey |
| Geometric fidelity | Chamfer distance / F-score@0.25 m | Model surface vs ground-truth LiDAR scan of a sub-area | F-score ≥ 85% | F-score ≥ 70% | Terrestrial/tripod LiDAR strip or photogrammetric GT sub-model; CloudCompare/M3C2 |
| Completeness | % reconstructed area in AOI polygon | Share of analysis area covered by valid geometry above min density | ≥ 90% | ≥ 80% | Coverage raster from point density |
| Visual quality | PSNR/SSIM/LPIPS on holdout frames | Rendered-vs-real comparison on frames excluded from training | LPIPS ≤ 0.15 | LPIPS ≤ 0.25 | gsplat render eval split |
| Scale correctness | Baseline agreement error | GNSS-derived camera-pair distances vs model distances | ≤ 2% | ≤ 5% | Automated check in M3 |
| Trajectory quality | ATE/RPE vs RTK log | Absolute/relative trajectory error of estimated camera path | ATE ≤ 0.5 m | ATE ≤ 5 m | evo toolkit vs RTK/PPK log |
| Speed | Wall-clock per minute of video | Coarse product / full product | coarse ≤ 2 min, full ≤ 30 min (cloud profile C) | same | Pipeline timing instrumentation |
| Walkability integrity | Traversal success suite | Scripted walks: no falls through floor, no wall clipping, navmesh reachability of all POIs | 100% pass | same | Automated headless engine test + manual checklist |
| Analytics validity | Tool error vs tape/laser measurements | Distance/area/volume tool checks on surveyed objects | ≤ 2% distance, ≤ 5% volume | ≤ 5% / ≤ 10% | Field validation kit |
| Robustness suite | Pass rate across stress inputs | Dusk pass, rain-slick road, busy traffic scene, coastal glare | ≥ 4 of 5 scenarios produce accepted products | same | Fixed benchmark reel |

**Benchmark datasets (no field deployment needed to score most rows):**
- [UrbanScene3D](https://github.com/OpenDroneMap/ODM)-class urban drone captures with LiDAR GT (large-scale city scenes).
- **TartanAir / Mid-Air** (synthetic drone video with perfect poses/depth) for tracker/geometry regression tests.
- **BlendedMVS / WHU MVS** urban aerial sets for mesh-quality comparisons.
- **Custom validation flight:** one DJI-class RTK drone pass over a site with 6–10 surveyed checkpoints + a small terrestrial-LiDAR strip → the primary acceptance harness.
- **Classical baseline:** identical footage through OpenDroneMap and COLMAP+MVS; report deltas (we must beat or match ODM's geometry while being dramatically faster and adding the interactive layer).

## 3. Build roadmap

### Phase 0 — Environment & data foundation *(week 1–2)*
Repo scaffolding matching Doc 2 module layout; Docker env (CUDA/PyTorch/ffmpeg/GDAL/PDAL); ingest M0 working end-to-end on a phone/DJI sample clip incl. .srt/XMP GPS parse; weight-fetch script with license verification.
*Exit:* `pipeline ingest sample.mp4 → keyframes.jsonl + GPS track plot`.

### Phase 1 — Geometry MVP *(week 3–6)*
Chunked feed-forward reconstruction (MapAnything-apache; Pi3 alternate) with Sim(3) chunk alignment (VGGT-Long pattern); DPVO full-flight track; naive TSDF mesh from scaled depths; georeference via robust GNSS Sim(3) + pyproj export.
*Exit:* georeferenced LAZ + rough textured OBJ from a single-pass video, on cloud GPU; ODM baseline comparison table.

### Phase 2 — Quality: splats, surfaces, texturing *(week 7–10)*
gsplat splatfacto + 2DGS surfel training; mesh extraction + cleanup; texture bake; moving-object masking (YOLO/SAM2 + LaMa); QC report v1; evaluation harness implementing §2 metrics against the custom validation flight.
*Exit:* products O1–O5 meeting thresholds on the validation site.

### Phase 3 — Walkable runtime + analytics *(week 9–13, overlapping)*
Engine build (Godot first: MIT plugins, CharacterBody3D, NavMesh from our collision GLB) + web viewer (spark + FPS controls); measurement/LOS/viewshed/terrain tools; scenario config loader with `urban_planning` mode complete.
*Exit:* O6/O7 shipped; a stakeholder can walk the site in a browser.

### Phase 4 — Mission-rehearsal mode *(week 14–17)*
Entity templates on NavMesh, patrol paths, threat rings via DSM LOS, cover map, objective zones, after-action replay; scenario editor UI; two demo scenarios (border sector, damaged urban block).
*Exit:* configurable scenario files drive behavior end-to-end; evaluation-suite walkability/analytics tests green.

### Phase 5 — Speed toward near-real-time *(weeks 18+)*
Chunk-level cloud parallelism; coarse-product fast path (keyframes→geometry→quick TSDF in minutes); incremental/streaming research spike (DA3-Streaming-style windows + online splat refinement); Jetson Orin feasibility probe.
*Exit:* coarse product < 2 min/min-of-video on Profile C; documented edge prototype status.

**Effort estimate:** 1–2 engineers + 1 part-time GIS/QA → credible demo at ~month 3–4, pilot-ready ~month 6.

## 4. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| VGGT-family license/AUP blocks defense use | Certain if ignored | High | Already designed out: MapAnything-apache/Pi3 defaults; install-time license verifier |
| Splat quality poor on long corridors / low-texture roads | Medium | Medium | DPVO backbone + chunk overlap tuning; capture guidance; semantic hole-fill flags |
| Standalone-GNSS jobs miss accuracy expectations | High | Medium | Honest per-job accuracy sidecars; upsell RTK; checkpoint QA option |
| Facade coverage weak on straight nadir-only passes | High | Medium | Capture guidance (oblique segments); expectation management in QC report; optional slow orbit around priority structures |
| gsplat training cost on big areas | Medium | Medium | Multi-GPU/chunked training, resolution tiers, compression formats |
| AGPL contamination fear (ODM) | Low | High | ODM only as external benchmark binary, never linked |
| Dynamic objects ghosting when masking off | Medium | Low-Med | Default-on detector for traffic scenes; residual cleanup pass |
| Scope creep toward game studio territory (scenario layer) | High | Medium | Scenario layer is config-driven on top of fixed asset contract; entity behaviors stay template-based until funded |

## 5. Business & productization

**Value proposition:** *"One flight, minutes of compute, and you're standing inside the mission."* Traditional drone photogrammetry demands planned multi-pass missions, hours of processing, and delivers passive models. We deliver a georeferenced, measurable, **walkable** world from a single ad-hoc pass — with analytics and rehearsal scenarios built in.

**Market segments → features that win them**

| Segment | Killer feature | Monetization |
|---|---|---|
| Disaster response / rapid mapping | Coarse product in minutes post-landing; damage-change compare | Government contracts / per-mission SaaS |
| Border & strategic-area mapping | Mission-rehearsal mode (LOS, threat rings, patrols) | Defense programs (license-clean stack mandatory — done) |
| Construction progress | Weekly single-pass walkthroughs + cut/fill volumes | Subscription per site |
| Infrastructure inspection | Facade mode + defect annotations | Per-inspection pricing |
| Urban planning / smart cities | Stakeholder web walkthroughs, shadow/viewshed studies | Civic SaaS seats |
| Archaeology / heritage documentation | Non-contact preservation-grade capture, shareable walkable tours | Project licensing |
| Digital twins | 3D Tiles streaming into Cesium-based enterprise twins | Integration/API tier |

**Competitive landscape:** DroneDeploy/Pix4D/Agisoft (mature, multi-pass-centric, no walkability), Polycam/Luma (consumer capture, no georef rigor), open-source ODM (batch CLI, no interactivity). Differentiation = single-pass video-first ingestion + speed + the walkable/scenario layer nobody ships with georef rigor.

**Open-core strategy:** pipeline core and viewers Apache/MIT/BSD open-source (community trust, procurement-friendly for gov/defense audits); commercial tiers = orchestration cloud, scenario editor, enterprise integrations, support. Note AGPL caution keeps ODM out of the product entirely.

**Near-term funding-shaped milestones:** (1) public demo reel — single-pass video → walkable browser scene in <30 min total; (2) validation-flight whitepaper with §2 numbers vs ODM/Metashape; (3) one lighthouse pilot per segment (disaster exercise or construction site are easiest to land).

## 6. Prerequisites & team skills

Python/CUDA/PyTorch engineering; ffmpeg/GDAL/PDAL GIS tooling; one person comfortable with pose-graph optimization (GTSAM/Ceres); Godot-or-Unity generalist for the runtime; basic licensed-drone access for validation flights (RTK-capable preferred). All model weights downloadable/gated-access as documented in Doc 1; your current RTX 3050 6 GB suffices for Phases 0–1 development with cloud burst for heavy stages (~$0.35–2.50/hr rentals).

## 7. Immediate next actions

1. Create repo skeleton per Doc 2 §M-layout; stand up Docker env (Phase 0).
2. Acquire 2–3 sample single-pass clips (any drone with .srt/XMP GPS; include one busy-traffic scene).
3. Wire M0 ingest + DPVO tracking; verify a full-flight trajectory plot over the GPS trace.
4. Rent one cloud GPU session; run MapAnything-apache chunked reconstruction end-to-end; eyeball first point cloud.
5. Book the validation site + RTK checkpoints for the evaluation harness (§2).

---
*Research findings referenced throughout are verified as of 2026-08-24 against upstream repositories and license files; see Doc 1 for links.*
