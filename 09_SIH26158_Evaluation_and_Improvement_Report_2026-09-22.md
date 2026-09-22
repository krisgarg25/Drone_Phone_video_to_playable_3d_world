# SIH26158: evidence-based evaluation and improvement report

**Research date:** 22 September 2026  
**Project assessed:** this repository's current working tree, not a clean release.  
**Authoritative requirements:** `C:\Users\krisg\Downloads\SIH26158.pdf`, three scanned pages corresponding to printed pages 37–39, “Problem Statement – 17.”

## Executive verdict

**We have a functioning offline video-to-Gaussian-splat/walkable-scene prototype. We do not yet have evidence of a compliant, georeferenced, metre-accurate single-pass UAV reconstruction system.** This is a meaningful starting point, not a finished surveying solution.

- **Strongest existing asset:** the reconstruction-to-viewer integration, recovery logic, blur-aware frame selection, and low-VRAM engineering.
- **Largest missing requirement:** real UAV telemetry → synchronized camera positions → defensible Earth-referenced geometry. Local scale from assumed speed/height is not georeferencing.
- **Largest evidence gap:** no independent ground-truth accuracy/completeness benchmark establishes the ≤1 m target.
- **Current speed warning:** a recorded 139.133-second indoor clip took 2,371.8 seconds, or 39.5 minutes, including validation/viewer steps. Even COLMAP plus training alone took 35.1 minutes. This is not the official 10-minute UAV test, but clearly does not demonstrate the target throughput.
- **Main deliverable distinction:** the visual PLY contains Gaussians and the GLB is a collider; neither demonstrates a validated textured survey mesh or standard georeferenced point-cloud product.
- **Best next direction:** preserve the viewer, but add a geometry-first, telemetry-aware reconstruction and evaluation path. Generate a splat as an optional presentation layer rather than making expensive splat training a prerequisite for every measurable output.

**Do not spend the next development cycle primarily polishing the viewer or generating invisible building backs. Accuracy, completeness, and speed carry 70% of the marks; the UI carries 5%.**

No score such as “70/100 ready” is assigned: weights alone cannot turn missing measurements into a defensible score. “Not demonstrated” is not a measured failure, but it is not a pass either.

## 1. What the supplied PDF actually requires

The tables below are transcribed from PDF page 2, printed page 38. They are not invented targets or weights.

### Desired Output

| Parameter | Official target |
|---|---|
| Reconstruction Type | 3D Mesh / Point Cloud |
| Processing Time | < 15 minutes for 10-minute video |
| Spatial Accuracy | ≤ 1 m |
| Coverage | Entire visible scene |
| Output Formats | OBJ, PLY, LAS, GeoTIFF, .glb/.gltf, .fbx |
| Visualization | Web-based or Desktop Viewer |

### Evaluation Criteria

| Criterion | Official weight | What we should demonstrate — proposed interpretation, not an extra official rule |
|---|---:|---|
| Reconstruction Accuracy | 30% | Independently measured geometry, scale, and absolute georeferencing error. |
| Model Completeness | 20% | Recovery of the scene actually visible during the single pass; holes and missing categories quantified. |
| Processing Speed | 20% | Complete processing of a genuine 10-minute video in less than 900 seconds on declared hardware. |
| Innovation | 15% | An improvement attributable to our method, established by ablation rather than a list of AI libraries. |
| Scalability | 10% | Bounded memory and practical throughput as duration, image resolution, and scene extent increase. |
| User Interface | 5% | A usable import → progress → inspect → measure → export workflow. |
| **Total** | **100%** | **Accuracy + completeness + speed account for 70%.** |

Mandatory inputs are drone video (1080p/4K), GPS coordinates, and flight metadata. IMU, barometric altitude, camera intrinsics, and RTK/PPK corrections are optional. A solution that works only with RTK, an IMU, or a specialized phone recording has not established support for the mandatory-only case.

The PDF asks for terrain and structures, facades and rooftops, roads and infrastructure, vegetation and obstacles, and textured meshes or point clouds. It explicitly names occluded surfaces as a challenge, but the output coverage target is **the entire visible scene**. This distinction matters: a method must not quietly substitute plausible invented surfaces for measured geometry.

Printed page 39 says the dataset **“Will be provided real time.”** The supplied PDF contains no downloadable evaluation dataset.

### Important ambiguities not resolved by the PDF

- “≤ 1 m” does not specify RMSE, maximum error, percentile error, horizontal error, vertical error, or 3D error. Report several statistics; do not claim an official pass from whichever one looks best.
- No evaluation GPU/CPU, RAM, network assumptions, or input scene extent is specified.
- The processing clock is not defined: end of capture, beginning of ingest, or start of offline processing. Report offline processing time and streaming latency separately.
- The format list does not say whether every format is mandatory or examples are acceptable. GeoTIFF is normally a geospatial raster product such as a DSM/orthomosaic, not a generic full 3D mesh container.
- “Single pass” has no precise flight geometry specification. A video provides many frames, but not necessarily useful sideways baseline or visibility of both sides of buildings.
- Scoring curves and pass/fail gates are absent. **A defensible numerical score out of 100 cannot be calculated from the weights alone.**

Ask the organizers about these ambiguities before making contractual performance claims. Until clarified, retain raw measurements and use conservative interpretations.

## 2. What a credible demonstration must measure

This section proposes our validation protocol. Its extra thresholds and procedures are engineering recommendations, not additional SIH requirements.

### 2.1 Keep three different questions separate

1. **Does it look good?** Held-out-view PSNR/SSIM/LPIPS, visual inspection, texture seams, and ghosting answer an appearance question.
2. **Is its shape and scale correct?** Surveyed lengths, surface distances, and scale drift answer a geometric question.
3. **Is it in the correct place on Earth?** Independent reference coordinates in a declared coordinate reference system answer a georeferencing question.

A strong result on one does not establish the other two. Gaussian parameters in a PLY do not automatically constitute a survey-quality point cloud, and a navigation/collision mesh is not automatically a reconstructed textured surface.

### 2.2 Record a reproducible run manifest

For every benchmark retain input identity/hash, recording duration, resolution/frame rate, camera calibration, timestamp convention, GPS source and uncertainty, coordinate and vertical datums, selected keyframe timestamps, pipeline versions, configuration, GPU/CPU/RAM/VRAM, stage timings, peak memory, warnings, and export paths. Keep rejected frames and the reason for rejection auditable.

Use an untouched single continuous flight recording for the acceptance run. Do not join multiple passes, prune difficult segments without disclosure, or count processing only after expensive poses were precomputed. A small existing indoor recording is useful for development but is not a substitute for this test.

### 2.3 Accuracy protocol

- Acquire reference geometry or surveyed **checkpoints used only for evaluation**. Evaluation checkpoints do not imply requiring extensive GCPs as reconstruction inputs. Declare zero reconstruction GCPs where that is the intended operating mode.
- Fit scale and georeferencing using only permitted input telemetry/calibration, not the hidden evaluation reference.
- Evaluate in a suitable local metric frame or projected CRS. Never compute metre errors by treating latitude/longitude degrees as Cartesian metres. State EPSG or the full CRS, local origin, axis convention, units, and whether heights are ellipsoidal or orthometric.
- For checkpoint error vectors `e_i = reconstructed_i - reference_i`, report horizontal RMSE `sqrt(mean(dx² + dy²))`, vertical RMSE `sqrt(mean(dz²))`, 3D RMSE `sqrt(mean(dx² + dy² + dz²))`, median 3D error, P95, maximum, and checkpoint count/distribution.
- The **primary absolute metric result is unaligned**: do not fit a new transform to the evaluation reference before calculating it.
- A rigid-aligned geometry score is a useful secondary diagnostic but removes global position/orientation error. A similarity-aligned score also removes scale error. Label each explicitly and never use a similarity-aligned score to prove metric scale or global accuracy.
- Report errors in measured lengths and heights across near/far regions, not just a single convenient object. Report the number of independent scenes, not only the number of correlated points.
- Keep uncertainty claims calibrated. A low residual against the same noisy GPS used for fitting is not independent accuracy evidence.

A practical first campaign is at least three legal, controlled outdoor scenes, including terrain/road, a building with roof and facade, and vegetation with limited dynamics. Use repeated runs where resources allow. For a small reference survey, spread approximately 10–20 held-out checkpoints across position and height rather than clustering them beside the takeoff point. These are proposed starting choices, not a statistical guarantee or official sample-size requirement.

### 2.4 Completeness protocol

Build a reference region consisting of surfaces visible from the recorded camera path, using independent reference geometry where available. Do not define visibility from our output alone: missing geometry would otherwise disappear from the denominator.

At several tolerances, for example 0.10 m, 0.25 m, 0.50 m and 1.00 m, measure:

- **Precision:** fraction of reconstructed surface samples close to reference surfaces.
- **Recall/completeness:** fraction of visible reference surface samples recovered by the reconstruction.
- **F-score:** harmonic mean of precision and recall.

Use surface-area-aware sampling or equivalent weighting so that a dense easy patch does not dominate. State sampling density and treatment of outliers, vegetation, moving objects, and boundaries. Report roofs, facades, ground/roads, vegetation, and obstacles separately. Include hole area and a map of low-support regions.

Where independent reference geometry is unavailable, image-space coverage and reprojection checks are useful **proxies**, not proof of whole-scene geometric completeness.

### 2.5 Speed and scalability protocol

The strict offline desired-output gate is **a 600-second source video processed in <900 seconds**, a processing-time/input-duration ratio below **1.5**. That does not by itself demonstrate live streaming: a live system must also report backlog, sustained input handling, update frequency, and latency.

Measure decode, quality selection, metadata synchronization, feature extraction/matching, pose estimation, dense geometry, meshing, texturing, georeferencing, export, and loadable output availability. Count all required stages. Report model-download/cold-start costs separately from warm local processing; do not include them selectively.

Publish time to first coarse map, time to first measurable map, and time to final required output. If processing overlaps flight, publish both total consumed compute time and the delay after the last captured frame. Distinguish a provisional output from the quality-qualified final output.

Test 1-, 5-, and 10-minute inputs at relevant source resolutions. Downsampling is allowed as an engineering strategy, but state the working resolution and show its effect on accuracy and completeness. Report peak VRAM/RAM, disk usage, registration failure rate, and quality versus runtime, not just successful runs.

### 2.6 Fundamental limits we should acknowledge

- GPS-based scale and placement can be noisy even when local visual geometry is excellent. RTK/PPK is helpful but optional, and receiver corrections do not replace camera timing/extrinsic calibration.
- A nearly straight camera trajectory can be poorly conditioned for a full 3D similarity alignment. Check trajectory geometry and conditioning; attitude/gravity priors can help where available.
- A single flight path cannot reveal genuinely unseen surfaces. Learned completion is a prior or hypothesis, not recovered evidence.
- A learned metric-depth estimate is not a substitute for validation of metre-scale accuracy on the target aerial domain.
- Trees moving in wind, transparent surfaces, low-texture roofs, and large depth discontinuities may remain difficult even with a better neural model.

The credible product response is to expose these limits and uncertainty, not hide them behind photorealism.

## 3. Where the current project stands

### 3.1 Weighted readiness assessment

These are evidence statuses, not invented judge scores.

| Criterion | Weight | Current standing | Evidence / missing proof |
|---|---:|---|---|
| Reconstruction Accuracy | 30% | **Not demonstrated; georeferencing path is missing** | Local rotation/scale are exported, but no Earth-referenced translation, CRS, vertical datum, or held-out checkpoint error is established. See `scripts/solve_frame.py:344-426`. |
| Model Completeness | 20% | **Partial heuristics; not benchmarked** | Frustum support and a walking coverage grid exist, but neither measures visible reference-surface recovery. See `scripts/solve_frame.py:113-137`, `scripts/check_coverage.py:163-217`. |
| Processing Speed | 20% | **Target not demonstrated; existing full run is a warning** | `work/room_w_jsonl/report.json:5-35`: 2,371.8 s, including 1,190.3 s COLMAP and 918.1 s training. No qualified 600-second UAV acceptance run. |
| Innovation | 15% | **Credible integration work; evaluation advantage unproven** | Robust retries, low-memory training, and walkable assets are real. No ablation yet demonstrates improved UAV accuracy/completeness/speed versus a baseline. |
| Scalability | 10% | **Some memory controls; large-scene behavior unproven** | Disk-loaded training and adaptive budgets exist; no verified bounded-memory, tiled 10-minute aerial workflow. Effective image resolution can be reduced automatically. |
| User Interface | 5% | **Implemented; strongest present area** | Web viewer/capture and navigation code exist, with small JS regression suites passing in the audit. A complete GIS measurement/export workflow was not interactively validated in this session. |

### 3.2 Desired-output compliance matrix

| Official target | Current evidence | Honest conclusion |
|---|---|---|
| 3D mesh / point cloud | Gaussian PLY plus collision GLB generated by the pipeline (`pipeline.py:519-548`). | A 3D artifact exists; a measurement-quality surface or conventional georeferenced XYZ/RGB cloud remains unestablished. |
| <15 min for 10-minute video | Full indoor run and tiny smoke runs; some recent reports are cached resumes. | Not proven. Do not extrapolate a smoke run or advertise a resumed-stage timer as full reconstruction. |
| ≤1 m spatial accuracy | Scale heuristics and phone-pose metric path; no independent reference error. | Not proven. Cannot honestly advertise one-metre accuracy. |
| Entire visible scene | Heuristic support/coverage, qualitative renders. | Not proven; no reference-based completeness denominator. |
| OBJ/PLY/LAS/GeoTIFF/GLB/GLTF/FBX | Splat PLY and collider GLB are visible in the integrated path. | No demonstrated complete geospatial export set; clarify whether the organizer requires every listed format. |
| Web/desktop viewer | Existing web viewer and rendered/walk-tested artifacts. | Implemented, but existing “walkable” validation is not surveying validation. |
| Mandatory GPS + flight metadata | Optional phone AR-pose importer, with fallback to no priors. | Drone telemetry integration and explicit handling of missing/invalid mandatory data are a major gap. |

### 3.3 What the recorded results actually say

| Local artifact | Recorded result | Interpretation |
|---|---|---|
| `work/room_w_jsonl/report.json:2-35,92-99` and `work/room_w_jsonl/frame.json:45-50` | 2,371.8 s total; input duration field 139.133 s; finished 8 September 2026. | Approximately 17.0 processing seconds per recorded video second. Historical indoor evidence, not a new benchmark or a prediction for a 10-minute UAV video. |
| Same report | COLMAP 1,190.3 s; training process 918.1 s; walktest 197.3 s. | Matching/pose solve and training dominate. Removing the walktest alone will not solve the speed problem. |
| `work/room_w_jsonl/logs/03-colmap.log:21-24`, `work/room_w_jsonl/logs/05-train.log:93-97` | 480/487 registered images; 15,000 iterations; 486,474 Gaussians; training-loop 863 s; final logged PSNR 20.05 dB. | Camera registration and training-image fit, not ground-truth metres or surface completeness. |
| `work/room_w_jsonl/logs/05-train.log:3-14` | Resolution reduced from 1210×908 to 984×738; Gaussian cap reduced; disk-loaded training. | Effective settings must be recorded. “High quality” requested settings are not proof of actual input resolution. |
| `work/roomscan/report.json:2-35,98-101` | 1,860.5 s with warnings. | Another recorded offline run, not a clean official acceptance pass. |
| `work/rocks/logs/04-train.log:2-17` | 72 images at 640×360; 300 iterations; 18,063 Gaussians; 8 s training loop / 21 s process. | A useful smoke test, not comparable to final-quality dense reconstruction. |
| `work/rocks/report.json:5-40` | 43.1 s, with keyframes/COLMAP/poses/training/frame stages skipped. | Resume/downstream-processing time only. |
| `work/rocks/frame.json:45-50` | 4.7336 m/unit from speed; height alternative 9.0382 m/unit; duration 11.887 s. | Competing assumptions differ by approximately 1.91×. This is scale sensitivity, not independently measured error or corroboration. |
| `work/rocks/viewer_assets/world_check.json:5-14,53-62,85-90` | 41% grid support; 2.184 m cells; internal collider/heightfield median difference 0.02 m, P95 7.86 m. | A usable-world check can pass without proving the official accuracy or completeness criterion. |
| `results/verdicts/critic_bar1_visual_v2.md:3-30`, `results/verdicts/critic_bar2_walkability.md:20-34` | Historical 10/10 visual verdicts and stable simulated walking. | Appearance and physics evidence from older artifacts; not independent surveying measurements and not automatically applicable after artifacts change. |

Current hardware was independently queried during this research: **NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6,144 MiB VRAM, driver 616.56**. The training environment reports **PyTorch 2.4.1+cu124**, CUDA available. This does not prove that every historical log was produced under identical hardware/software conditions.

### 3.4 Specific changes justified by the code

1. **Add a true UAV telemetry importer rather than relabeling phone poses.** `pipeline.py:455-463` conditionally invokes `scripts/import_phone_poses.py`; the importer rebases each log to its first sample (`scripts/import_phone_poses.py:128-141`) and may continue without priors (`:42-59`). Parse video timestamps, GPS time, position uncertainty, altitude reference, aircraft attitude, gimbal attitude, and camera calibration where supplied. Keep raw timestamps; estimate or validate clock offsets. Reject a claim of georeferenced output when the required information is unusable, while still permitting an explicitly local-only preview.

2. **Replace assumed metric rulers in the survey path.** `scripts/solve_frame.py:352-373` can select scale from speed × duration or assumed height; `:410-426` serializes only a local-frame solution. Use robust camera-position correspondences to GPS in a local metric frame, followed by uncertainty-weighted refinement. Detect degenerate near-collinear trajectories and gross GPS outliers. Preserve the mapping between local coordinates and Earth coordinates as an explicit export artifact. Do not just attach an EPSG label to untransformed coordinates.

3. **Correct and test camera-model interpretation before adding more AI.** In `scripts/train_splat.py:83-96`, `OPENCV` shares the `SIMPLE_RADIAL` focal/center indexing even though its parameter layout includes separate `fx, fy`. This is a static code defect; an end-to-end OPENCV failure was not reproduced here, and the default model need not exercise it. Add numeric tests with unequal `fx/fy`, off-center principal points, and distortion before accepting calibration-sensitive improvements.

4. **Separate presentation geometry from measured geometry.** `scripts/export_viewer_assets.py:382-436` thickens splats and removes higher-order appearance coefficients for the viewer. `scripts/build_collider.py:1-33` explicitly builds physics geometry. Preserve a raw, geometrically validated output branch that does not inherit visual thickening, floor filling, or collider smoothing. A collider may remain useful for navigation; do not market it as survey ground truth.

5. **Replace frustum counts with actual visibility and baseline evidence.** Current support does not establish freedom from occlusion, and the separate coverage script uses a fixed 4.5 m range (`scripts/check_coverage.py:163-217`), inappropriate as a general aerial coverage definition. Compute support from depth-consistent reprojections and useful triangulation angles, with thresholds conditioned on range and image quality.

6. **Mask dynamic observations without erasing the requested scene.** Existing sharpness selection is useful (`scripts/extract_keyframes.py:197-227`), but the static loss does not demonstrate dynamic masks (`scripts/train_splat.py:457-525`). Mask moving observations before feature matching and dense fusion; retain static vegetation where observable, and preserve excluded regions as coverage gaps. Do not remove every tree or every vehicle class and then claim full-scene coverage.

7. **Separate software survival tests from output-quality gates.** `tests/test_e2e.py:139-160` can accept a hard world-gate failure as successful pipeline survival. Keep this resilience test, but add a distinct acceptance result for accuracy, completeness, and speed. Immutable run manifests should bind evaluations to exact input/configuration/output hashes; overwritten logs and old visual verdicts cannot substitute for this.

8. **Do not call disk-based training “live reconstruction.”** The pipeline builds sequential stages (`pipeline.py:435-592`); streaming image loading/checkpoint PLYs conserve memory, while HTTP byte-range video delivery serves playback. None alone demonstrates processing incoming flight frames into a live evolving metric map.

## 4. Recommended engineering direction

### Keep what works; change which output is authoritative

A suitable candidate architecture is:

```text
Single continuous UAV video + GPS + flight metadata
  -> timestamp/calibration/coordinate validation
  -> blur-, motion-, and baseline-aware keyframes
  -> visual pose estimation + robust telemetry constraints
  -> confidence-filtered dense geometry
  -> measured XYZ/RGB cloud and optional textured surface
  -> geospatial export + independent quality report
  -> existing web viewer

Optional side branch: poses/images -> Gaussian splat for visual presentation
```

This is a proposed architecture, not implemented functionality or a promise of meeting the time budget. Crucially, producing a measurable point cloud should not require finishing a long photorealistic splat optimization. The official reconstruction type permits a mesh **or** point cloud; begin with a defensible dense colored point cloud, then add a textured mesh where performance permits. Clarify format expectations with the organizers.

### Geospatial output should be explicit and testable

- **PLY:** export an ordinary XYZ/RGB point cloud separately from Gaussian parameters, with units and a georeferencing sidecar if the chosen PLY consumer cannot preserve CRS metadata.
- **LAS/LAZ:** preserve projected coordinates, CRS, quantization scale/offset, and any supported quality fields. PDAL's writer documents these controls [S1]. Setting CRS metadata is not a substitute for transforming local coordinates into that CRS.
- **OBJ / GLB / glTF:** export an actual reconstructed surface, with textures where available. Use a documented local origin/transform for large Earth coordinates; do not let float precision or a Y-up/Z-up conversion silently change measurements.
- **GeoTIFF:** produce a georeferenced elevation raster and, if implemented, an orthomosaic. GDAL and Rasterio document the CRS plus pixel-to-world mapping [S2, S3]; PDAL provides point-to-raster aggregation [S4]. Preserve nodata and validity masks rather than bridging unobserved areas indiscriminately.
- **DSM versus DTM:** an elevation surface containing rooftops/trees is not a verified bare-earth terrain model. Ground classification and observations under canopy are additional requirements; rasterization alone does not solve them.
- **FBX:** defer a converter until the evaluation's exact format obligation is clarified; validate its scale/axes and license/tooling implications rather than letting it displace accuracy work.

Round-trip each deliverable through an independent consumer and compare known coordinates/lengths. A file extension appearing in a directory is not an interoperability test.

### A speed budget to guide experiments, not a performance claim

For a **warm, fully local, offline** 10-minute acceptance input, a provisional engineering allocation could be: 90 s ingest/selection, 210 s pose solve, 360 s dense reconstruction, 90 s export, and 90 s checks = **840 s**, leaving 60 s below the 900-second ceiling. No measurement currently establishes that these allocations are achievable on the 6 GB GPU. Instrument first, then change budgets or algorithms based on observed accuracy/runtime tradeoffs.

If the first experiment misses the budget, reduce redundant work rather than silently lowering quality below the accuracy/completeness gates: pair fewer useful frames, bound image resolution based on target ground sampling distance, process overlapping local submaps, fuse supported depth incrementally, and make high-quality splat training optional. Keep submap registration and seam error in the benchmark—tiling can reduce memory while introducing drift.

## 5. Creative improvements worth testing

These are proposed product/algorithm experiments, **not claims of new research novelty** and not demonstrated gains.

| Proposal | How it would work | Why it fits the rubric | Required proof / failure mode |
|---|---|---|---|
| **An evidence map, not just a pretty model** | For each surface region retain supporting frame IDs, visibility, baseline, reprojection/depth disagreement, and input provenance; show low-support regions in the viewer. | Links accuracy, completeness, innovation, and useful UI. | Confidence must correlate with held-out error. Raw Gaussian density is not calibrated confidence. |
| **Accuracy-aware keyframe scheduling** | Extend sharpness selection with expected triangulation angle, parallax, texture, timestamp spacing, and whether a frame contributes newly visible surface. | Fewer redundant matches can improve speed without discarding essential facade views. | Compare against the same uniform/blur-only frame budget: registration, metre error, completeness, and total time. Must retain hard areas, not only easy ones. |
| **Clock-offset estimation as a calibration step** | Compare visual motion with available GNSS velocity or IMU motion to estimate video/telemetry offset, then optimize cautiously with robust residuals. | Directly addresses GPS noise and metric placement without adding GCPs. | Validate with known offsets. Nearly constant straight-line motion may make offset indistinguishable from translation; do not force a spurious estimate. |
| **Two quality stages** | Publish a clearly provisional coarse local map, then replace/refine it with the qualified metric output; splats remain optional. | Improves situational visibility and practical latency. | Publish both latencies and both quality levels. A fast preview cannot be timed as the completed deliverable. |
| **Separate observed, constrained, and inferred surfaces** | Observed geometry is primary. Plane/roof constraints can regularize supported data. Pure learned completion is a separately labeled overlay. | A credible response to limited views and occlusion, without pretending unknown geometry is surveyed. | Never include inferred-only regions in the claimed measured completeness or one-metre accuracy result. Show their contribution separately. |
| **Visibility-aware dynamic filtering** | Track temporal inconsistency as well as object classes, discard moving observations from static fusion, and retain excluded-region masks. | Reduces ghost vehicles/people and moving foliage while preserving static obstacles. | Report both ghost reduction and completeness loss. A parked vehicle or a static tree may be a legitimate obstacle. |
| **Adaptive geometric detail** | Allocate denser depth/surface work to facades, edges, and high-relief regions; use coarser representation on well-supported near-planar ground. | A route to bounded memory and useful coverage on 6 GB VRAM. | Check thin structures, wires, poles, roof edges, and inter-tile seams; coarse sampling can erase important geometry. |

**Most promising project-specific innovation:** combine accuracy-aware keyframe selection with an evidence/provenance map and mandatory telemetry validation. This builds on existing frame-selection and viewer work, is easier to evaluate honestly than generative completion, and targets more heavily weighted criteria.

## 6. Prioritized improvement roadmap

Work in this order; advance when evidence satisfies the exit condition. The phases are not promised delivery dates.

| Priority | Work package | Existing integration point | Exit evidence |
|---|---|---|---|
| **P0 — establish truth** | Add run manifests, independent accuracy/completeness metrics, a cold/warm timing protocol, and camera-model unit cases. Collect one representative UAV clip with aligned telemetry and independent reference. | `tests/`, `pipeline.py`, `scripts/train_splat.py:72-96` | Reproducible baseline showing what fails and by how much; known calibration fixtures parse correctly. |
| **P1 — real coordinates** | Ingest actual GPS/flight metadata; validate timestamps/calibration; robustly align/refine camera poses and retain CRS/vertical datum/uncertainty. | Existing importer/pose-prior infrastructure; `scripts/solve_frame.py:344-426` | Synthetic known-transform/time-offset tests plus unaligned errors against held-out outdoor checkpoints. No assumed-speed ruler in a claimed survey output. |
| **P1 — genuine geometry** | Produce confidence-filtered dense XYZ/RGB, then a surface if needed; keep presentation thickening/floor filling out of measurement products. | Fork after pose solving, before mandatory splat training. | Measured surface precision/recall and metric distances; standard point-cloud export loads correctly in an independent consumer. |
| **P2 — speed** | Profile matching/pose solve first, then density/training; compare selective matching and frame budgets; introduce bounded local processing only where measured. | `scripts/extract_keyframes.py`, `scripts/run_colmap.py`, pipeline stage orchestration | A genuine 600-second video finishes in <900 seconds with the same required quality, declared hardware, no skipped prerequisite stages. |
| **P2 — visible completeness** | Add dynamic masks, occlusion-aware support, baseline-aware selection, and class/region-specific coverage. | Keyframe selection, dense fusion, coverage map. | Better visible-region recall without unacceptable precision loss, tested at several tolerances. |
| **P2 — interoperable outputs** | CRS-aware LAS/PLY, optional textured OBJ/GLB, GeoTIFF DSM/orthomosaic where required; local-to-global transform preserved. | Export stage and existing viewer. | Round-trip coordinates, axes, textures, nodata, and dimensions checked outside our own viewer. |
| **P3 — evidence-backed innovation** | Add the provenance/confidence map and the winning keyframe/telemetry experiments. | Existing viewer and selection code. | Ablation isolates improvement in one or more weighted criteria; inferred geometry cannot contaminate measured exports. |
| **P3 — scale and finish** | Test longer/high-resolution scenes, submap seams, recovery and export size; then polish import/progress/measurement UX. | Pipeline runner, viewer, automated evaluation. | Multi-scene duration/resolution/VRAM matrix with failures disclosed and repeatable results. |

### Minimum experiment matrix

Use the same input split, working resolution, hardware, telemetry availability, and reference for comparisons. Report changes in both quality and time; do not change several methods simultaneously and attribute the result to one.

1. Existing pipeline baseline, with its current heuristic scale explicitly labeled.
2. Same visual reconstruction plus validated telemetry/georeferencing.
3. Same frame budget: uniform/blur-only versus baseline-aware selection.
4. Same selected frames: existing matching versus a learned-matching candidate.
5. Same poses: a dense geometry baseline versus a geometry-oriented neural/splat candidate.
6. Dynamic filtering off versus on, including stationary obstacles and moving vegetation.
7. High-quality all-at-once output versus bounded submaps/progressive output, including seam error and final—not just preview—runtime.

Run the mandatory-input-only track first. Report a separate optional-sensor track when IMU/RTK/PPK is present; do not pool the stronger-sensor results into a claim that ordinary GPS alone meets the target.

### What not to prioritize yet

- More avatars, gameplay polish, or cinematic presentation at the expense of the 70% core criteria.
- Treating a foundation model's fast forward pass as end-to-end mapping time.
- Feeding every video frame into a global model simply because more frames are available.
- Running generative completion or aggressive enhancement on the only geometry evidence without testing hallucination-induced error.
- Claiming metre accuracy from a phone-scale ruler, training PSNR, a collider consistency check, or GPS fitting residuals.
- Buying optional hardware before measuring whether timing, calibration, and software changes address the actual error. Optional RTK can help, but must remain a separately characterized operating mode.

## 7. Verification performed in this research

- Rendered and visually read all three scanned pages of the supplied PDF using an already-installed PDF library after the default PDF reader/text extraction could not recover the scanned text.
- Read-only repository audit, with main-thread spot checks of the core pipeline, telemetry importer, scale serialization, camera parsing, visual export modifications, collider role, and timing artifacts.
- Fresh execution: `PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 .venv/Scripts/python.exe tests/check_all.py --verbose` → **5/5 suites passed** (`robust`, `capture`, `collider`, `gate`, `unit`), including **107 unit checks**. Tests used temporary fixtures; no full reconstruction was run.
- The audit subagent executed the existing JavaScript suites: `test_element_ids.js` (**13 checks**), `test_capture_scripts.js` (**130 checks**), and `test_coverage_map.js` (**110 checks**), all passing. These are software/fixture checks, not an interactive browser acceptance test.
- Queried the currently available GPU and training runtime; historical runtime evidence is labeled historical rather than presented as a new benchmark.
- Did **not** run a full 10-minute UAV reconstruction, measure against an external survey, acquire new drone data, run a model bake-off, or interactively revalidate the UI. Consequently no SIH accuracy, completeness, speed, or total-score pass is claimed.
- Research used three parallel available subagents. The tool interface has no model selector; **the requested “3.8 Flash” model could not be selected or verified**. No claim is made that the agents used it.
- No implementation changes, dependency installations, commits, pushes, or modifications to the user's pre-existing research/artifact changes were made. This report is the requested new deliverable.

## 8. Open-source options and evidence limits

Evaluate a small number of candidates against the same input/reference, not every popular model. **Code license, pretrained-weight license, dataset license, and dependency licenses are separate checks.** A paper or a public code link alone does not establish unrestricted deployability.

**Verification boundary:** GitHub CLI was unavailable, so repository READMEs, pinned commits, release assets and LICENSE files were not directly audited. The evidence below comes from fetched official documentation, author papers, project pages and explicitly identified model cards. These are not fourteen confirmed, unrestricted, production-ready packages. License uncertainty is a release blocker, not something to hide.

### 8.1 Foundations and export tools — highest practical priority

| Candidate | Practical role here | Integration / constraints | Verified licensing scope |
|---|---|---|---|
| **COLMAP — retain and extend** | Compare robust camera-center GPS alignment with reconstruction using uncertainty-weighted position priors. Use its geometric/dense reconstruction path as the reference before replacing it. | Lowest disruption because COLMAP already exists in the project. `model_aligner` performs post-hoc similarity alignment, with GPS→ECEF/ENU support and RANSAC; `pose_prior_mapper` constrains reconstruction using position-prior covariance. Three correspondences are a documented minimum for alignment, not proof of good conditioning. Neither automatically resolves clock offsets or all camera/GNSS extrinsics. [S7] | Official license says new BSD, with dependency caveats [S8]. Installed binary/version feature support still needs checking before implementation. |
| **OpenDroneMap — comparison baseline** | Turn timestamp-associated video keyframes plus geolocation into a conventional mapping baseline with point cloud/mesh/orthophoto/elevation outputs. | Moderate adaptation. Its `geo` file supports a CRS header, image positions, and horizontal/vertical accuracy. The documented optional camera-angle fields are currently for radiometric calibration; do not assume they are pose-orientation priors. Video synchronization remains our work. [S9, S10] | Official fetched pages establish the open-source product but not its exact pinned-release license. Verify redistribution/service obligations before adoption. |
| **GTSAM — only when custom fusion is justified** | Jointly model visual constraints, GPS, and optional IMU/barometer information; handle antenna-to-body/camera offsets. | High integration cost. `GPSFactor`, `GPSFactorArm`, `GPSFactorArmCalib`, and IMU preintegration provide building blocks—not a ready-made drone reconstructor or automatic clock synchronizer. [S11–S13] | Official site describes BSD licensing; exact release and dependencies not audited. |
| **PDAL** | Write conventional LAS/LAZ with geospatial metadata and quality attributes; convert supported points into raster products. | Moderate packaging/coordinate-integration work. Correct CRS transforms and LAS quantization are our responsibility. It does not reconstruct missing geometry. [S1, S4] | Official PDAL page provides BSD-style redistribution conditions [S5]. Optional dependency terms were not exhaustively audited. |
| **GDAL / Rasterio** | Export a georeferenced DSM/orthomosaic with CRS, pixel transform, nodata, and validity masks. | Low-to-moderate once trustworthy metric geometry exists; not a fix for inaccurate depth. [S2, S3] | GDAL core is MIT; official documentation warns that optional dependencies can carry less-permissive terms [S6]. Rasterio's own package license was not separately fetched. |

**RTKLIB is conditional, not a default fix.** PPK/RTK requires suitable raw GNSS observations, navigation/correction/reference data for the chosen mode; ordinary latitude/longitude telemetry cannot recreate missing carrier-phase measurements. The agent's official-site/manual fetches failed, so current version-specific features, maintenance and license were not verified. Evaluate it only when the actual capture hardware records the necessary data. A fixed GNSS solution still needs camera timing and antenna/camera-offset handling.

### 8.2 Learned matching, geometry and masking — a ranked experimental shortlist

All runtime numbers here are **author-reported under the stated conditions**, not results from this repository. Input resolution, frame sampling and omitted output stages materially affect comparisons. No speedup factor on the RTX 3050 is claimed.

| Candidate / priority | What it could improve | Published evidence and relevance to this machine | Integration, licensing and deciding experiment |
|---|---|---|---|
| **1. LightGlue + ALIKED detector ablation — test early** | Better correspondences while retaining geometric verification, bundle adjustment and external scale/georeferencing. LightGlue adapts matching effort; ALIKED is a detector/descriptor candidate, not a scale estimator. | The official `lightglue_disk` card reports about **44 ms/pair FP32**, but omits GPU model/keypoint count. This is not verified ALIKED-pair timing or full SfM throughput. [S20–S22] | Low–medium integration: import compatible features/matches into existing SfM. The particular DISK card is Apache-2.0-tagged; upstream code, ALIKED and ALIKED-compatible weights were **not** separately license-verified. Compare SIFT versus learned matching on identical frames/pairs: verified inliers, registration, time and held-out XYZ error. |
| **2. SAM 2 + geometric motion checks — test with a strict time budget** | Remove moving observations before matching/depth/fusion; propagate masks across selected video frames. | SAM 2 reports **43.8 FPS Hiera-B+ / 30.2 FPS Hiera-L**, **A100**, batch 1, **1024×1024**, compiled image encoder. These are not 6 GB laptop or arbitrary multi-object mapping rates. [S32, S33] | Medium integration. Meta explicitly announces **Apache-2.0 code and weights; BSD-3 evaluation code**. Verify exact release and any added detector's terms. Segmentation is not motion detection; retain parked/static obstacles where appropriate. Compare no mask, semantic-only mask and motion-validated mask, measuring ghost reduction **and lost coverage**. |
| **3. 2D Gaussian Splatting + TSDF — geometry experiment** | Surface-oriented splats and geometry regularization; median-depth rendering followed by Open3D TSDF fusion gives a more defensible surface-extraction baseline than a gameplay collider. | Paper experiments use **RTX 3090**, DTU resized to **800×600**, and **15k/30k** optimization settings. No verified end-to-end UAV mesh timing on 6 GB is available. Thin structures and oversmoothing remain concerns. [S29] | Medium integration, not a configuration toggle in the current trainer. Code/inherited rasterizer licenses unverified. Usually per-scene optimization rather than a pretrained geometry checkpoint. Compare MVS-depth TSDF versus 2DGS-depth TSDF with identical metric poses/voxel size; include texture baking and total runtime. |
| **4. VGGT + VGGT-SLAM 2.0 — bounded higher-memory pilot** | Joint camera/dense-geometry initialization plus submap organization, useful where classical initialization is weak. | January 2026 paper: about **120 ms/frame / 8.4 FPS** for its complete pipeline without optional semantics, **16-frame submaps, RTX 3090**. Adding semantic queries raises time to **158 ms/frame**. Input pixel dimensions were not established in fetched evidence. Raw VGGT is reported limited to about **60 frames on a 24 GB RTX 4090**. The authors explicitly say true scene scale is **not estimated**. [S23, S24] | High integration; do not feed a complete 10-minute recording to raw VGGT on 6 GB. Wrapper/backbone-weight licenses unverified. Compare initialization methods followed by the **same externally anchored refinement**, measuring drift, surface error, completeness and VRAM. |
| **5. MASt3R-SLAM — alternative dense initializer** | Two-view pointmaps, tracking/fusion and loop closure; supports calibrated and uncalibrated formulations. | About **15 FPS on RTX 4090 + i9-12900K**, inputs with maximum side **512 pixels**, using **every-second-frame subsampling**. Output pointmaps are not a finished textured mesh; scale ambiguity and substantial lens distortion remain issues. [S25] | High integration plus external anchoring/filtering/surface extraction. Wrapper and MASt3R checkpoint licenses separately unverified. Compare against VGGT-SLAM on the same calibrated sequence and reference, not headline FPS. |
| **6. DPV-SLAM / DPVO — pose-only efficiency candidate** | Sparse patch-based trajectory/keyframe estimation; DPV-SLAM adds loop closure to DPVO. | Paper uses an **RTX 3090**, reports approximately **5–7 GB** and **1–4× real-time** depending on dataset/configuration; a uniquely matched input-resolution/FPS condition was not established. Memory is already borderline or too large for this 6 GB device before dense mapping. [S26] | Medium integration, but still needs a separate dense geometry stage and absolute scale. Code/weights unverified. A non-revisiting flight cannot gain long-range loop closure. Compare drift and tracking survival with closure on/off; include downstream reconstruction cost. |
| **7. DROID-W — research pilot, not deployment choice** | CVPR 2026 DROID-family approach for dynamic outdoor footage, using uncertainty-aware adjustment and a monocular depth prior. | About **10 FPS on RTX 3090 with a 16-core CPU**. The paper's **1200×1600 capture resolution is not established as processing resolution**. Casual outdoor performance is not high-altitude UAV validation. [S27, S28] | High integration. Release availability and DROID-W/Metric3D/DINOv2 code/weight terms unverified. Compare with DROID-SLAM while ablating uncertainty weighting and depth priors. A learned Metric3D prior is not independent metric control. |
| **8. Gaussian Opacity Fields (GOF) — defer unless quality justifies cost** | Adaptive opacity-field surface extraction can avoid some uniform-TSDF costs. | Author-reported **24.2 minutes of optimization per Tanks and Temples scene on an A100 at 30k iterations**; mesh extraction is separate. Delaunay processing can add substantial overhead. Reflection-induced false geometry is a documented failure mode. [S30, S31] | Medium–high integration; code/dependency license unverified. Poor default fit for a <15-minute total target on 6 GB. Only proceed if measured surface improvement over 2DGS/MVS warrants the cost; include extraction and texturing. |

**Practical selection:** first establish the GPS-aware COLMAP baseline and standard metric outputs; test LightGlue/ALIKED and budgeted motion-validated masking; compare a geometry-first MVS/fusion result against 2DGS only if needed. Do not replace the whole pipeline with VGGT/MASt3R or add all eight candidates simultaneously.

### 8.3 Datasets and benchmarks we can honestly use

| Resource | Useful evidence | What it cannot prove / access limits |
|---|---|---|
| **UrbanScene3D: Polytech and ArtSci** | Paper describes **10 synthetic + 6 real** scenes, and independently acquired **Trimble X7 LiDAR** reference for these two real locations. Best reviewed aerial geometry candidates for surface accuracy/completeness. [S14, S15] | Do not treat every scene's mesh as independent LiDAR. Instrument ranging and registration figures are not blanket scene-wide accuracy guarantees. Paper mentions 4K aerial video in selected scenes, but the fetched landing page did not directly establish available video files; actual archive contents need checking. Raw GNSS/IMU and exposure-level synchronization were not established. |
| **UrbanScene3D synthetic scenes** | CAD reference enables controlled visibility/baseline/occlusion experiments along one selected synthetic trajectory. | Synthetic success is not real UAV robustness. Keep synthetic and real results separate. Same dataset-use restrictions apply. |
| **ETH3D** | Laser-scanned reference, public training/reference downloads and held-out testing; reports precision-like accuracy, completeness, F1, and runtime, with selectable centimetre tolerances. Useful for independently testing the dense-geometry stage. [S16–S19] | Not a UAV GPS/absolute-CRS or single-pass aerial benchmark. Leaderboard runtimes are not automatically hardware-normalized. Official dataset license is **CC BY-NC-SA 4.0**. |
| **Own controlled single-pass UAV recording** | Necessary integration test for the actual video+GPS+flight-metadata contract, with independent surveyed checkpoints/reference. | Must collect legally and safely, document calibration/time/CRS, and withhold reference from reconstruction. This dataset was not acquired during this research. |

**UrbanScene3D licensing matters:** official terms restrict use to non-commercial purposes, require citation, and prohibit redistribution, including altered variants; they also constrain commercial exploitation of derived models. Do not put downloaded scenes or derivatives in a public demo/repository without establishing permission [S14]. These terms may affect deployment plans even when a research prototype is allowed.

Use both a geometry benchmark and a real telemetry integration benchmark. Neither alone verifies all six weighted criteria, and no fetched source establishes a turnkey, already-validated SIH solution for the current machine.

## 9. Sources and evidence policy

The supplied PDF is the authority for the target tables. Repository paths above are direct local evidence at the current working tree, which already contained uncommitted changes. Historical reports are observations about those artifacts, not measurements repeated in this session.

External references below were fetched during this research, not merely copied from a search snippet. Author-reported benchmarks are not our benchmarks. A source may establish an algorithm, a format capability, or a license without establishing fitness on this GPU or this challenge.

- **[P0] Official supplied brief:** `C:\Users\krisg\Downloads\SIH26158.pdf`, scanned pages 1–3 / printed pages 37–39. Desired Output and Evaluation Criteria: page 2 / printed page 38. Dataset availability: page 3 / printed page 39.
- **[S1] PDAL LAS writer:** https://pdal.org/en/2.10.2/stages/writers.las.html — LAS/LAZ, spatial-reference metadata, scale/offset, and extra dimensions.
- **[S2] GDAL GeoTIFF driver:** https://gdal.org/en/stable/drivers/raster/gtiff.html — raster georeferencing, CRS, nodata/masks.
- **[S3] Rasterio georeferencing:** https://rasterio.readthedocs.io/en/stable/topics/georeferencing.html — CRS and pixel-to-world transform.
- **[S4] PDAL GDAL raster writer:** https://pdal.org/en/2.8.4/stages/writers.gdal.html — point-to-raster statistics, radius/window interpolation, nodata.
- **[S5] PDAL license:** https://pdal.org/en/2.9.3/copyright.html — core redistribution conditions.
- **[S6] GDAL license:** https://gdal.org/en/stable/license.html — MIT core and dependency caveats.
- **[S7] COLMAP FAQ:** https://colmap.github.io/faq.html — GPS alignment, ECEF/ENU, RANSAC, position-prior mapping and covariance.
- **[S8] COLMAP license:** https://colmap.github.io/license.html — new BSD core and separately licensed dependencies.
- **[S9] ODM geolocation format:** https://docs.opendronemap.org/geo/ — CRS, per-image positions, optional accuracy and radiometric-angle fields.
- **[S10] ODM product:** https://www.opendronemap.org/odm/ — mapping product types; not an accuracy/runtime guarantee.
- **[S11] GTSAM official site:** https://gtsam.org/ — factor-graph library and BSD licensing statement.
- **[S12] GTSAM GPS factors:** https://borglab.github.io/gtsam/gpsfactor/ — zero/known/estimated lever-arm variants and Cartesian-frame requirements.
- **[S13] GTSAM IMU factors:** https://borglab.github.io/gtsam/imufactor/ — preintegration, pose/velocity/bias modeling.
- **[S14] UrbanScene3D official dataset and terms:** https://vcc.tech/UrbanScene3D — dataset scope, download references and non-commercial/non-redistribution conditions.
- **[S15] UrbanScene3D paper:** https://arxiv.org/html/2107.04286v3 — synthetic/real scenes, selected aerial video, and Polytech/ArtSci LiDAR reference.
- **[S16] ETH3D overview:** https://www.eth3d.net/overview — reference acquisition and benchmark design.
- **[S17] ETH3D datasets:** https://www.eth3d.net/datasets — training/reference/test data access.
- **[S18] ETH3D high-resolution multi-view benchmark:** https://www.eth3d.net/high_res_multi_view — accuracy/completeness/F1 thresholds and runtime fields; no hardware-normalized winning method is asserted here.
- **[S19] ETH3D home/license statement:** https://www.eth3d.net/ — CC BY-NC-SA 4.0 dataset license.
- **[S20] LightGlue paper:** https://arxiv.org/abs/2306.13643 — adaptive feature matching.
- **[S21] ALIKED paper:** https://arxiv.org/abs/2304.03608 — local feature detector/descriptor.
- **[S22] ETH-CVG LightGlue DISK model card:** https://huggingface.co/ETH-CVG/lightglue_disk — specific card's license tag and insufficiently specified pair-time claim; not ALIKED checkpoint verification.
- **[S23] VGGT project:** https://vgg-t.github.io/ — model's geometry/camera prediction scope.
- **[S24] VGGT-SLAM 2.0 paper:** https://arxiv.org/html/2601.19887v1 — 2026 submap system, runtime conditions, memory constraints and explicit true-scale limitation.
- **[S25] MASt3R-SLAM paper:** https://arxiv.org/html/2412.12392v2 — tracking/fusion, benchmark hardware/input subsampling, calibration limitations.
- **[S26] Deep Patch Visual SLAM paper:** https://arxiv.org/html/2408.01654v1 — DPV-SLAM/DPVO distinction, approximate memory/speed evidence and loop closure.
- **[S27] DROID-W project:** https://moyangli00.github.io/droid-w/ — 2026 dynamic-outdoor research scope; repository/release status not verified.
- **[S28] DROID-W paper:** https://arxiv.org/html/2603.19076v1 — uncertainty/depth priors and hardware/runtime context.
- **[S29] 2D Gaussian Splatting paper:** https://arxiv.org/html/2403.17888v1 — surface-oriented representation, median-depth TSDF extraction and benchmark settings.
- **[S30] Gaussian Opacity Fields project:** https://niujinshuchong.github.io/gaussian-opacity-fields/ — adaptive opacity-field surface extraction.
- **[S31] Gaussian Opacity Fields paper:** https://arxiv.org/html/2404.10772v2 — 24.2-minute A100 optimization evidence, extraction overhead and reflection failure modes.
- **[S32] Meta SAM 2 announcement:** https://ai.meta.com/blog/segment-anything-2/ — code/weight/evaluation-code licenses and tracking limitations.
- **[S33] SAM 2 paper:** https://arxiv.org/html/2408.00714v2 — model-specific A100 timing conditions, not full mapping throughput.

**Bottom line:** we should position the current work as a robust visualization prototype being upgraded into an evidence-backed mapping system. The next convincing demonstration is a telemetry-aligned, independently measured, timed single-pass reconstruction—not simply a prettier render or a larger AI model.
