# SIH26158 Research Brief — Single-Pass Drone Video to Accurate 3D Model

**Problem statement ID:** SIH26158 (NTRO)
**Title:** Single-Pass Drone Video to Accurate 3D Model Generation System
**Authoritative source:** `C:\Users\krisg\Downloads\SIH26158.pdf` — 3 scanned pages, printed pages 37–39 of 53, sha256 prefix `02e2a9fa.9edd6`. Problem Statement 17.
**Compiled:** 26 September 2026
**Scope of this document:** the research, measurements, prior art and citations behind the project. This file contains no slide design, layout or presentation instructions.

---

## 1. Official requirements, transcribed from the SIH brief

### 1.1 Inputs

| Class | Items |
|---|---|
| Mandatory | Drone video (1080p / 4K), GPS coordinates, flight metadata |
| Optional | IMU data, barometric altitude, camera intrinsic parameters, RTK/PPK corrections |

### 1.2 What must be reconstructed from one flight path

1. 3D terrain and structures
2. Building facades and rooftops
3. Roads and infrastructure
4. Vegetation and obstacles
5. Textured 3D meshes or point clouds

### 1.3 Target parameters (official, printed page 38)

| Parameter | Target |
|---|---|
| Reconstruction type | 3D Mesh / Point Cloud |
| Processing time | < 15 minutes for a 10-minute video |
| Spatial accuracy | ≤ 1 m |
| Coverage | Entire visible scene |
| Output formats | OBJ, PLY, LAS, GeoTIFF, .glb/.gltf, .fbx |
| Visualisation | Web-based or desktop viewer |

### 1.4 Evaluation criteria and official weights

| Criterion | Weight |
|---|---:|
| Reconstruction accuracy | 30% |
| Model completeness | 20% |
| Processing speed | 20% |
| Innovation | 15% |
| Scalability | 10% |
| User interface | 5% |
| **Total** | **100%** |

Accuracy + completeness + speed account for 70% of the score. `docs/readiness/04-evidence-map.md:9` notes: "Weights are from `SIH26158.pdf` page 38. They are weights, not a formula, so the 'what moves this' column names evidence, never points."

### 1.5 Dataset

The brief states the dataset "will be provided real time."

### 1.6 Evaluation metrics we set for ourselves (targets, not measurements) — `03_Roadmap_Evaluation_and_Business.md` §2

| Criterion | Metric | Target (RTK) | Target (standalone GPS) |
|---|---|---|---|
| Georeferencing accuracy | Checkpoint RMSE H/V | H ≤ 10 cm, V ≤ 15 cm | H ≤ 3 m, V ≤ 5 m |
| Geometric fidelity | Chamfer distance / F-score @ 0.25 m | F ≥ 85% | F ≥ 70% |
| Completeness | % reconstructed area in AOI | ≥ 90% | ≥ 80% |
| Visual quality | PSNR / SSIM / LPIPS on holdout frames | LPIPS ≤ 0.15 | LPIPS ≤ 0.25 |
| Scale correctness | Baseline agreement error | ≤ 2% | ≤ 5% |
| Trajectory quality | ATE / RPE vs RTK log | ATE ≤ 0.5 m | ATE ≤ 5 m |
| Speed | Wall-clock per minute of video | coarse ≤ 2 min, full ≤ 30 min (cloud) | same |
| Walkability integrity | Traversal success suite | 100% pass | same |
| Analytics validity | Tool error vs tape/laser | ≤ 2% distance, ≤ 5% volume | ≤ 5% / ≤ 10% |
| Robustness suite | Pass rate across stress inputs | ≥ 4 of 5 scenarios | same |

Output acceptance detail: point-cloud density ≥ 20 pts/m² on surfaces within 80 m of the flight path; DSM at 1×–2× GSD of the source video at reference distance.

### 1.7 Our own blocking acceptance protocol — `docs/readiness/03-acceptance-protocol.md`

- One clip at **≥ 1920×1080, ~600 s, single pass** (Step 0, blocking).
- GPS at **1 Hz or better**, with fix quality and HDOP.
- **≥ 8 independent surveyed checkpoints** — "8 is the floor for a fit/hold-out split."
- The `survey_evaluation` gate is strictly **600 ± 0.5 s / < 900 s**.

---

## 2. The headline honesty statement, verbatim

`09_SIH26158_Evaluation_and_Improvement_Report_2026-09-22.md:9`:

> "We have a functioning offline video-to-Gaussian-splat/walkable-scene prototype. We do not yet have evidence of a compliant, georeferenced, metre-accurate single-pass UAV reconstruction system."

`docs/readiness/04-evidence-map.md` verdict per official criterion:

| Criterion | Status |
|---|---|
| Accuracy | **Unproven.** "Best current number: 1.14 m hold-out RMSE against a drifting phone VIO track, not surveyed truth" |
| Completeness | Partly delivered, unproven overall |
| Speed | **Unproven.** "Measured rocks 806.4 s (12 s clip), room 4001.4 s (139 s clip) — both stitched across failures, neither a full-deliverable run" |
| Reconstruction type | Delivered |
| Formats | Delivered inside the run; externally unvalidated |
| Textured mesh | **Absent by choice** |
| Visualisation | Delivered for splats; partial for survey products |
| Innovation | Delivered as engineering, unproven as advantage |
| Scalability | Unproven |
| UI | Delivered for the evidence console |

`docs/readiness/05-challenges-and-applications.md`:
> "Summary: 1 of 8 fulfilled with measurement, 5 of 8 mechanism-complete and awaiting qualifying data, 1 partial by deliberate choice, 1 not met on this hardware."

### 2.1 The seven things still unproven (verbatim list, reproduces well under questioning)

1. Metric accuracy ≤ 1 m against independent surveyed checkpoints.
2. Coverage of the entire visible scene against an independent reference surface.
3. < 15 min end-to-end on a 10-minute 1080p single-pass clip.
4. Any output validated by a third-party reader. No GDAL, laspy or PDAL is installed; the project's own readers re-parse what its writers produce, which is self-consistency.
5. Textured mesh: Poisson gives per-vertex colour, not a UV texture; the manifest states `claims_textured_mesh: false`.
6. Occluded surfaces.
7. One-click operation.

`09_…` §9.5, the one-line version:
> "no measured ≤1 m accuracy, no measured completeness against reference surfaces, no run of the official 10-minute gate, no georeferenced outdoor UAV scene, no dynamic-object masking, and no live incremental streaming reconstruction"

---

## 3. Measured runtime evidence

Hardware for everything in this section: **NVIDIA GeForce RTX 3050 6 GB Laptop GPU, 6,144 MiB VRAM, driver 616.56, Windows 11. PyTorch 2.4.1+cu124.** No MSVC `diff-gaussian-rasterization` compile available. Benchmark setting in §9.1 is "300 keyframes at 1000 px — chosen to fit the card, not a quality target."

### 3.1 Geometry-first runtime, measured (09 §9.1)

| Clip | Source | Keyframes | Keyframes (s) | Sparse (s) | Undistort (s) | Dense (s) | Fusion (s) | Total (s) | Ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rocks | 12.05 s | 72 | 5.6 | 216.4 | 5.8 | 566.3 | 12.3 | **806.4** | 66.9× |
| room_w_jsonl | 139.47 s | 291 | 21.5 | 946.9 | 16.3 | 2951.8 | 64.9 | **4001.4** | 28.7× |

Peak GPU memory: **1,405 MiB during sparse mapping, 443 MiB during dense** — described in the report as "lower bounds, not headroom."

Report conclusion:
> "The speed criterion is the real risk, and the dense stage is the reason. At 10.1 s per image (291 images) or 7.9 s per image (72 images) … a 600-second flight cannot finish inside 900 seconds at this frame budget. The official gate itself remains not measured: the longest recording on this machine is 151 s, and no 600-second single-pass video exists here yet."

### 3.2 Historical recorded runs

| Artifact | Recorded result |
|---|---|
| `work/room_w_jsonl/report.json` | 2,371.8 s total; input duration field 139.133 s → "approximately 17.0 processing seconds per recorded video second". COLMAP 1,190.3 s, training 918.1 s, walktest 197.3 s |
| `work/roomscan/report.json` | 1,860.5 s with warnings |
| `work/rocks/report.json` | 43.1 s — but keyframe, COLMAP, pose, train and frame stages were **skipped**; resume time only. Several other `report.json` files are similarly cached resumes and must not be quoted as runtimes |
| `work/rocks_quality/report.json` | 1,095.6 s (COLMAP 183.2 s, train 826.1 s, walktest **failed** 61.0 s, `kind: failed`) |
| `work/trst-fc4e0a04/report.json` | 1,143.9 s (COLMAP 193.8 s, train 871.2 s, walktest 46.6 s) |

### 3.3 Dense quality/speed profiles (09 §9.6)

| Profile | Images | Dense (s) | Fusion (s) | Total (s) | Fused points |
|---|---:|---:|---:|---:|---:|
| `survey` — 1000 px, consistency on | 72 | 1,230.3 | 13.4 | 1,243.7 | 355,965 |
| `fast` — 1000 px, consistency off | 72 | 473.4 | 11.6 | 485.0 | 375,458 |
| `budget` — 700 px, consistency off | 72 | 316.2 | ~5 | ~321 | 170,093 |

Self-correction recorded in the report: the first ablation claimed that turning geometric consistency off produced zero fused points. That was a bug — with `--PatchMatchStereo.geom_consistency false` COLMAP writes only photometric depth maps while the fusion step was still being told `--input_type geometric`.

### 3.4 Sparse matching ablation (real, `scratch/gpubench/ablation.json`, 291 frames, 1000 px)

| Variant | Sparse (s) | Registered cameras | Sparse points | Mean reprojection error (px) |
|---|---:|---:|---:|---:|
| A — exhaustive | 946.9 | 288 | 102,177 | 0.7125 |
| B — sequential only | **239.7** | 233 | 80,656 | **0.593** |

`sparse_time_saved_pct: 74.7`. Note the mean view support figures in the file are 12.017 and 9.621, which the report corrects to **6.009 and 4.810** (the on-disk values are 2× too high).

### 3.5 Speed projection (explicitly NOT a measurement)

> "keyframes ~30 s, sparse ~60–90 s at 68 frames, dense ~68 × 6.6 s ≈ 450 s, fusion ~15 s, export and hashing ~60 s ≈ **615–650 s against the 900 s target**. That is a projection from measured per-image rates, not a measured end-to-end run."

Prediction accuracy when it was tested: **3,960.1 s predicted vs 4,001.4 s measured — 1.0% error**, with the model still honestly reporting `fits_deadline: false`.

### 3.6 The gate is still not met on paper

`docs/GAPS_AND_OPTIMIZATIONS.md`:
> "The speed gate is still not met on paper: at ~2.8 s/frame a ≈270-keyframe mapping flight needs ≈750 s of dense alone against a 390 s budget."

`docs/APPLICATION_PLAYBOOKS.md` §0:
> "A 10-minute mapping flight needs ≈**270 keyframes** (60 m AGL, 52° VFOV, 70% overlap, 8 m/s). At the measured `fast` rate of **6.6 s/image** that is **~1,800 s for dense alone**."

### 3.7 Real on-disk GPU-bench scenes

- `scratch/gpubench/rocks`: 72/72 images registered, 58,985 sparse points, 350,438 fused dense points. 12.05 s clip, 1280×720.
- `scratch/gpubench/room_w_jsonl`: **288/291 registered, 102,177 sparse, 1,568,438 fused.** 139.5 s clip, 960×720.
- "**No local video is 1080p/4K. The longest is 151.6 s.**" Highest resolution anywhere: 1280×798.

---

## 4. The strongest original result: baseline-aware keyframe selection

This is the only genuinely novel algorithmic contribution measured in the repo, and the only one with a clean controlled ablation. `scratch/gpubench/selection_ablation.json`, real trajectory, 123.43 units, 288 registered cameras, measured dense rate 7.87 s/image at 1000 px.

| Budget | Strategy | Frames selected | Max gap (units) | Path coverage | Mean parallax | Est. dense time (s) |
|---:|---|---:|---:|---:|---:|---:|
| 40 | uniform | 40 | 5.709 | 0.608 | 35.412° | 314.8 |
| 40 | baseline-aware | 36 | 8.889 | 0.583 | 37.377° | 283.3 |
| 72 | uniform | 72 | 3.461 | 0.881 | 20.788° | 566.6 |
| 72 | baseline-aware | 61 | 8.803 | 0.870 | 22.577° | 480.1 |
| 120 | uniform | 120 | 2.628 | 0.973 | 12.845° | 944.4 |
| 120 | **baseline-aware** | **69** | 7.772 | 0.896 | **20.052°** | **543.0** |
| 200 | uniform | 200 | 2.357 | 0.993 | 7.843° | 1,574.0 |
| 200 | baseline-aware | 68 | 7.772 | 0.895 | 20.844° | 535.2 |
| 288 | uniform | 288 | 2.357 | 0.998 | 5.528° | 2,266.6 |
| 288 | **baseline-aware** | **68** | 7.772 | 0.895 | **20.844°** | **535.2** |

Report conclusion at the 288 budget:
> "at the same budget the selector keeps roughly a quarter of the frames while *increasing* mean parallax by **3.8×**, cutting estimated dense time by **4.2×**, and giving up about **10% of path coverage**."

Headline derived numbers: **288 → 68 frames (−76.4%)**, parallax **5.528° → 20.844° (3.77×)**, dense estimate **2,266.6 s → 535.2 s (4.23×)**, coverage **0.998 → 0.895**.

Note: this is "estimated dense time" derived from a measured per-image rate, not a measured dense run at 68 frames.

---

## 5. Metric scale and georeferencing evidence

### 5.1 Independent sensor accuracy hierarchy (`01_Research_and_Technology_Survey.md` §5)

| Sensor setup | Horizontal | Vertical | Notes |
|---|---|---|---|
| Standalone GNSS, no RTK, no GCPs | ~1–3 m | ~2–5 m, often worse | Camera-centre GPS tags are noisy; altitude is the worst axis |
| RTK/PPK direct georeferencing, no GCPs | ~2–5 cm | ~3–8 cm | Vertical ≈ 1.5–2× horizontal |
| RTK/PPK + 1 surveyed checkpoint | ~2–4 cm | ~3–6 cm | Cheap insurance |
| Full GCP network, classical | 1–3 cm | 2–5 cm | What Metashape and Pix4D achieve — our upper bound |

### 5.2 Our scale validation — the real failure, and its correction

`scratch/gpubench/scale_validation.json` as stored on disk (superseded, still uncorrected):
```json
{"matched_cameras":287,"fit_count":172,"holdout_count":115,
 "fitted_scale_m_per_unit":0.834933,"holdout_rmse_m":2.7646,
 "holdout_median_m":2.6935,"holdout_p95_m":3.6164,"holdout_max_m":3.8754,
 "reference_trajectory_length_m":38.06,
 "reference_kind":"phone VIO/AR camera centres (metric, self-drifting)",
 "not_survey_grade":true}
```

Report §9.3 reading of it:
> "visual-only geometry disagrees with an independent metric sensor by **~7% of the trajectory length** — nowhere near ≤1 m. … **sub-metre accuracy still has no supporting evidence at all** — the reference is self-drifting VIO, not surveyed control."

**This is superseded.** The fitted scale of 0.835 m/unit was a 3× arithmetic bug. The corrected values are:

- **0.2783 m/unit** georeferencing scale confirmed by **two independent estimators agreeing to 0.2%**
- **1.14 m** corrected hold-out RMSE, honestly labelled as *against a drifting phone VIO track, not surveyed truth*

### 5.3 The 1.91× scale disagreement — a genuine, citable finding

`work/rocks/frame.json`:
```
"scale_m_per_unit": 4.73355368777055,
"scale_anchor_speed": 4.73355368777055,
"scale_anchor_agl": 9.03817608812563,
"scale_source": "flight speed x clip duration",
"duration_s": 11.887,
"camera_agl_units": 2.766044803314359, "camera_agl_m": 13.09322157926725,
"ground_tilt_deg": 6.02, "n_ground_points": 3161,
"region_min_views": 4, "region_supported_gaussians": 7627
```

9.03818 / 4.73355 = **1.909×**. `docs/GAPS_AND_OPTIMIZATIONS.md` E5: "ruler D … refuses when the fit RMSE exceeds 5% of the flight path", and "`rocks`' own `frame.json` … its two built-in rulers disagree by **1.9×**".

This is the empirical justification for the dual-source scale consensus: a speed-derived scale and a barometric-height scale on the same 12-second capture disagree by 1.91×, so the pipeline must not publish metres until independent priors agree.

Field-trial lesson from prior art `ch1bo/drone-reconstruction`: `rel_alt` (IR/barometric, ~0.1 m precision) beats GNSS altitude (~10–20 m error).

### 5.4 GNSS quality readout (probe telemetry, not the rocks flight) — `scratch/gnss_render/render.json`

- GPS fixes 46, over 12 s, 69.39 m of path
- Reported σ / fix step: **1.79 m E&N, 3.96 m up**, steps 0.97 m
- Trajectory above noise: **No — per-fix motion sits below σ**
- Dropouts / implausible: 1 gap longer than 2 s, 1 fix above 25 m/s
- GPS↔video clock offset: bounded 0 s to 0 s, never estimated
- Not separable on this path: clock offset, lever arms, metric scale
- 1 of 45 consecutive intervals imply motion faster than the supplied 25.0 m/s limit
- Median inter-fix displacement: 0.9699 m
- **"Accuracy Not independently verified"**

### 5.5 Other georeferencing results — `docs/GAPS_AND_OPTIMIZATIONS.md`

| ID | Result |
|---|---|
| M11 straight-track georef | Known transform recovered within **0.05°**; ground-plane fallback reproduces a 3° cross slope as a 3° tilt |
| M12 GNSS error fusion | **200 synthetic flights** with 60 s correlated GNSS error: height RMSE **1.97 → 0.71 m**; 2σ coverage **88%** against an ideal 95% |
| M8 MGRS | **31NAA6602100000** published reference; CE90 vs 2M-sample Monte Carlo within **1%** |
| M2 SMRF ground filtering | Recall **100%**, false ground **0.9%**, DTM under a roof **−0.09 m**, roof nDSM **10.09 m** against a true 10; **1 km² / 3M points in 25.6 s** |
| Defect fixed | "Every DSM was mirrored north-south. Any `dsm.tif` produced before 2026-09-26 must be regenerated." |

---

## 6. Dynamic-object masking — the one challenge fulfilled with measurement

Challenge (iv) of the problem statement. Real ablation, `docs/readiness/05-challenges-and-applications.md`:

> "72/72 masks accepted by COLMAP after the naming fix; **4.85% of pixels vetoed**. registered cameras **unchanged at 72/72**, sparse tracks **−18.4%** (58,985 → 48,141), mean view support **−10.3%**, **mean reprojection error improved 7.8% (0.3586 → 0.3305 px)**, fused points **−0.6%**."

This is a strong result: masking removes 4.85% of pixels, cuts sparse tracks by 18.4%, costs only 0.6% of fused points, does not cost a single registered camera, and *improves* reprojection error by 7.8%.

Note the caveat: no frame was dropped for blur on the rocks clip ("0 of 72 frames were dropped"), so the blur gate is unexercised on that take.

---

## 7. Dense-stage ablations

### 7.1 Depth-sampling ablation — `scratch/gpubench/speed_ablation.json`

Agreement measured against our own slowest cloud, **not surveyed truth**. Reference diagonal 46.7197 units. Self-check: precision 0.9994, recall 0.9995, F 0.9995.

| Variant | Settings | Dense (s) | Total (s) | Fused pts | P@0.25% | R@0.25% | F@0.25% | F@0.5% | F@1.0% | F@2.0% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B control | 20 src, it5, sm15 | 457.7 | 472.9 | 375,437 | 0.9373 | 0.9422 | 0.9397 | 0.9853 | 0.9966 | 0.9991 |
| S6 | 6 src | 296.5 | 313.8 | 308,081 | 0.9971 | 0.6085 | 0.7558 | 0.7671 | 0.7752 | 0.7875 |
| **IT3** | **20 src, it3, sm8** | **196.6** | **213.5** | 386,732 | 0.9456 | 0.9373 | **0.9414** | 0.9851 | 0.9962 | 0.9988 |
| S6_IT3 | both | 121.5 | 140.0 | 316,471 | 0.9988 | 0.6084 | 0.7561 | 0.7670 | 0.7749 | 0.7856 |
| S6_IT3_ALT2 | + every 2nd ref (36 used) | 61.2 | 73.1 | 211,423 | 0.9999 | 0.5815 | 0.7353 | 0.7455 | 0.7557 | 0.7685 |

The adopted variant is **IT3: 2.33× dense speed-up with F@0.25% actually improving from 0.9397 to 0.9414**. The S6 family is rejected: ~38% of surface lost, visible as recall collapsing to ~0.61.

`docs/GAPS_AND_OPTIMIZATIONS.md` M13 records the same table with the verdicts: control 1.00×, IT3 **2.33× adopted**, S6 1.54× rejected, S6_IT3 3.77× rejected, S6_IT3_ALT2 7.48× rejected.

### 7.2 Monocular depth attempt — measured and rejected

`scratch/gpubench/mono_ablation.json` — MoGe-2 sparse-anchored monocular depth:

| Variant | Tolerance | fit_rel_rms_median | Fused pts | F@0.25% | F@1% |
|---|---|---:|---:|---:|---:|
| scale_tol2 | 0.02 | 0.0835 | 1,856,551 | 0.5275 | 0.8550 |
| affine_tol2 | 0.02 | 0.1118 | 1,691,220 | 0.3647 | 0.8122 |
| scale_tol1 | 0.01 | 0.0835 | 1,282,324 | 0.5653 | 0.8776 |
| scale_tol5 | 0.05 | 0.0835 | 2,340,564 | 0.4829 | 0.8228 |

All `moge_inference` timings are `null` — never actually run. `docs/GAPS_AND_OPTIMIZATIONS.md` adds: MoGe-2 distance maps are **0.173 s/frame**, whole stage ~5 s CPU + ~12 s inference for 72 frames versus 197 s PatchMatch, but "held-out anchor error per frame is **9.6% of range** with a per-frame scale, **5.0%** with a quadratic correction field."

### 7.3 Compression-artifact proxy validation — x264 CRF sweep, one real frame (09 §9.7)

| CRF | Block-boundary ratio | Blocking suspect | Quality score | Laplacian variance |
|---:|---:|---|---:|---:|
| 0 (lossless) | 1.004 | False | 1.000 | 149.5 |
| 15 | 1.131 | False | 1.000 | 139.5 |
| 23 | 1.062 | False | 1.000 | 93.6 |
| 30 | 1.189 | False | 1.000 | 42.2 |
| 38 | **1.779** | **True** | **0.562** | **13.4** |

> "the ratio rises **77%** from lossless to CRF 38 … Laplacian variance falls **91%** from CRF 0 to CRF 38."

### 7.4 Trainer log, real 15,000-step run — `work/room_w_jsonl/logs/05-train.log`

`[train] DONE in 863s (17.4 it/s), final N=486474`; at step 15,000 `loss 0.0518 psnr 20.05 N 486474 861s mem 1.53GiB`; peak PSNR seen **23.47** at step 14,600; step 200 `psnr 12.87`. **There is no SSIM or LPIPS anywhere in any log.** 480/487 images registered.

`docs/GAPS_AND_OPTIMIZATIONS.md` and the pitch deck both state plainly: "Our pixel scores sit below published norms for this technique."

---

## 8. Texture bake — the only held-out-view numbers in the repo

`results/texture_comparison/*/comparison.json`:

| Scene | Faces | Views | Atlas px | Mean MAE before | Mean MAE after | Mean detail before | Mean detail after | Timing |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rocks | 15,776 | 72 | 1,157 | 12.043 | **11.850** | 0.282 | **0.382** | mesh 3.4 s, visibility 39.9 s, colour 31.8 s, bake 33.9 s |
| temple | 8,207 | 120 | 1,170 | 15.357 | **14.603** | 0.355 | 0.292 | mesh 1.5 s, visibility 47.8 s, colour 33.5 s, bake 39.2 s |

Per-view coverage: rocks 95.6% / 98.6% / 96.0%; temple 68.2% / 92.5% / 91.6%. File sizes: rocks 1,326,008 → 1,694,980 B; temple 690,208 → 1,098,108 B.

---

## 9. Test and regression rig

| Suite | Result |
|---|---|
| `tests/check_all.py --verbose` | **5/5 suites passed** (robust, capture, collider, gate, unit), **107 unit checks** |
| `09 §9.8` | **28/28 CPU suites and 605 tests pass** |
| Survey-lane tests | **151 survey tests** plus the pre-existing 107 unit checks |
| JS suites | `test_element_ids.js` 13 checks, `test_capture_scripts.js` 130 checks, `test_coverage_map.js` 110 checks |
| Headless Chrome QA, GPU and WebGL disabled | **six checks pass, zero page errors, zero `/api/run` requests, zero model loads** |
| `scratch/sih26158-readiness-audit/browser/results.json` | `"status":"passed"`, fixture "synthetic CPU-only, not accuracy evidence", `page_errors: []`, `gpu_run_requests: []`, `model_requests: []` |

Per-module survey test counts: survey_priors 6 · survey_georef + survey_workflow 35 · survey_export 27 + survey_formats 8 · survey_visibility 36 + survey_products 36 + survey_occlusion 36 · survey_dynamics 39 · survey_frame_quality 37 + survey_photometry 39 · survey_observability 25 · survey_streaming 24 · survey_gnss 38 + survey_accuracy 36 · survey_measure 83.

Regression protection (09 §9.4): trainer with corrected camera model, 300 steps, 103,041 Gaussians, 20.1 it/s, 2.11 GiB, `splat.ply` written.

Count discrepancies across documents: 11 rules (`check_world.py`) in the writeup/README/README-MVP versus **13** in the pitch deck and dossier; **18 stages** in the writeup versus **15** in the deck; ~550 assertions in the writeup versus 232 in the deck; 2 of 8 scenes failing versus 2 of 10. **The writeup/README numbers are the ones backed by files on disk and should be the ones used.**

Two real bugs that survived 663 green tests, from `02-wiring-ledger.md`: `mesh_command` passed the dense *directory* to `poisson_mesher`; `write_masks` wrote masks at the reduced working resolution so "COLMAP silently rejected **all 72 frames** of the rocks scene and exited 0."

---

## 10. Walkability and viewer results

### 10.1 Walk test, before and after, on the rocks take

| Metric | Before | After |
|---|---:|---:|
| Distance walked | 19.9 m in 215 s | **65.3 m in 32 s** |
| Waypoints reached | 1 / 20 | **16 / 16**, all within 0.8 m |
| Travel efficiency | 10% | **69%** outbound, 100% return |
| Feet vs surface | −1.8 m (inside a crust 20 m up) | **+0.008 m** |
| Viewer ground vs physics ground | 0.101 m apart | **0.000 m** |
| Falls | 0 | 0 |

### 10.2 Per-scene results (`SE_Lab_Project_Writeup.md` §2.9)

| Scene | Capture | Registered | Splats | Walk test |
|---|---|---|---|---|
| rocks | 12 s drone clip | 114 → 272 frames after tuning | 128k → 451k | 65.3 m, 16/16 waypoints, 0 falls, foot gap +0.008 m |
| temple | drone, above cloud layer | — | 76,804 fog candidates culled | spawn −18 m → +8.8 m courtyard; 65.4 m, 0 falls |
| room_w_jsonl | handheld + AR poses | 95% | — | 60 m on heightfield ground |
| auditorium | 453 s interior dolly, 4K | 395/400 on rescue flags | — | 65.7 m, 0 falls; nav bake PASS at 177.2 m² |
| room_multi_video | mixed portrait + landscape | 61% | — | 8 m — fails the gate honestly (50 m vertical drift) |

### 10.3 Reproducible `--quality smoke` matrix (`README.md:221-229`)

| take | capture | status | steps | walked | sampled route | airborne | falls |
|---|---|---|---:|---:|---:|---:|---:|
| rocks | drone orbit | complete | 17/17 | 65.2 m | 64.7 m | 5/61 | 0 |
| temple | drone, cloud sea | **partial** | 17/17 | 65.2 m | 64.0 m | 0/61 | 0 |
| room_w_jsonl | phone + AR poses | complete | 15/15 | 36.5 m | 31.7 m | 1/334 | 0 |
| roomscan | phone scan | complete | 15/15 | 17.1 m | 15.5 m | 0/331 | 0 |
| test1 | phone | complete | 15/15 | 28.0 m | 26.2 m | 0/332 | 0 |
| test2train | phone | complete | 15/15 | 28.1 m | 26.4 m | 0/336 | 0 |
| test2horizontal | phone, low texture | **partial** | 15/15 | 30.1 m | 28.0 m | 0/336 | 0 |

`README.md:232`: "**temple and test2horizontal fail the hard rule *spawn on supported ground*.**"

### 10.4 Ground-truth walk logs on disk (`walk_log.json`, all `phase=done`)

rocks 65.22 · rocks_quality 65.39 · rocks-integration-check 65.28 · temple 65.23 · trst-fc4e0a04 65.69 · room_multi_video 65.81 · auditorium 65.79 · auditorium/walktest2 65.90 · auditorium_baseline_199k 65.96 · auditorium/final_verify 55.51 · test2horizontal 30.77 · test2train 28.14 · test1 27.96 · roomscan 16.18 (violations 1) · room_w_jsonl 9.57.

### 10.5 Ground-race measurements (writeup §2.7b, `README-MVP.md:210`)

auditorium heightfield 68 m vs shell 51 m · rocks shell 94 m vs heightfield 69 m · temple 12 m vs 6 m · room_w_jsonl hf 60 vs 55 · room_multi_video shell 8 vs 7. "**140 probe rays** landing on `ground.f32` at a **median of 0 cm**."

### 10.6 Blind A/B evaluation — real and citable

- `results/blinded/` holds 10 A/B composites per take across 13 takes, order recorded separately in `results/pair_key_<take>.json`. **100 pairs across ten scenes.**
- `results/verdicts/critic_bar1_visual.md`: "**VERDICT: WIN** … critic identified the real frame **10/10** correctly … judged both panels the SAME scene at the SAME moment in **10/10** … **Identification score vs hidden key: 10/10.**" Named defects: "uniform softness/blur, smeared sky band, minor rock merging."
- `results/verdicts/critic_bar1_visual_v2.md` (2026-08-25): "**VERDICT: WIN** … Same-scene agreement: **10/10 pairs** … WIN rule applied: ≥7/10 same-scene content match AND no disqualifying global artifact — satisfied at **10/10 with zero disqualifiers**." Non-disqualifying artifacts: "distant background degrades into soft smeared bands; occasional purple tinge on peripheral rocks."
- `results/verdicts/critic_bar2_walkability.md`: "**VERDICT: WIN** — all four walkability criteria PASS." "**65.22 m walked** at a steady ~0.50–0.56 m per 0.2 s sample; x spans −7.24…+10.46, z spans 10.00…29.64"; "capsule y stays within a **7 mm band** (4.498–4.505)"; "65.22 m logged vs ~63.3 m of straight-line path chords (**within 3%**)."

### 10.7 World-check gate output, rocks — `work/rocks/viewer_assets/world_check.json`

`"status":"warnings"`, `"hard_failures":[]`, `"warnings":["grid resolves foot-level support"]`. Thresholds: min_coverage 0.0834, min_perimeter_m 10.79, character_height_m 1.75, min_headroom_m 1.925, cell_m 2.1802, footprint_m 71.9, grid [33, 25].

- Heightfield measured ground: **40.0% of cells** (0% camera-derived, 60% nothing)
- Floor-not-ceiling: 0.0% of Gaussians > 4.36 m below surface
- Cameras above filmed ground: min 9.3 m, median 12.1 m over 10 cameras
- **grid resolves foot-level support: FAIL** (2.18 m cell against a ≤ 1.50 m threshold)
- Collider has 1,049 vertices; ceiling slab 0.1%
- Collider = route surface **+0.000 m median**; collider vs heightfield **+0.02 m median, p95 7.85 m** against a 4.36 m limit
- Route: 108.1 m loop, 6 waypoints, 1,250 m² routed, 31.9% of grid walkable, 0.0% of loop samples bad, spawn 0.21 m above floor

---

## 11. Output format delivery

- **5 of 6** official formats actually written with a real CRS, **EPSG:32643 WGS 84 / UTM 43N** (`delivery_manifest.json`).
- **1,568,438** measured points delivered in a real CRS (`scratch/real_delivery/summary.json`) — the largest measured cloud in the repo.
- MGRS reference validated: **31NAA6602100000**, CE90 within 1% of a 2M-sample Monte Carlo.
- Defect to remember: any `dsm.tif` produced before 2026-09-26 is north-south mirrored and must be regenerated.
- Textured mesh is **absent by choice**: Poisson gives per-vertex colour, not a UV texture; the manifest states `claims_textured_mesh: false`.

---

## 12. Shipped architecture

### 12.1 The design intent (M0–M10, `02_System_Architecture_and_Pipeline_Design.md`)

```
M0  Ingest & Keyframing
M1  Cleanup: blur filter, exposure normalize, moving-object inpaint
M2  Tracking: DPVO full-flight odometry + loop closure
M2  Geometry: chunked feed-forward MapAnything/Pi3 + Sim(3) pose graph
M3  Georeferencing: GNSS/baro fusion, Sim(3) to ENU, WGS84/UTM
M4  Dense surface: gsplat 2DGS splats / TSDF mesh
M5  Texture bake + compression + tiling
M6  Output products: GLB, LAZ, DSM/ortho GeoTIFF, 3D Tiles, SOG
M7  Walkable runtime web + engine
M8  Analytics: measure, LOS/viewshed, terrain
M9  Scenario layer -> Urban-planning mode / Mission-rehearsal mode
M10 Orchestration & operations
```

**Caveat:** this is a design document dated 2026-08-24. The shipped pipeline is different. Do not present M0–M10 as what was built.

### 12.2 What is actually shipped — 18 stages (writeup §2.5, verbatim)

```
[1] keyframes  [2] priors  [3] colmap  [4] poses  [5] train  [6] frame
[7] export  [8] sky/cloud  [9] colors  [10] collider  [11] objects
[12] surface  [13] gate  [14] evals/pairs  [15] walktest
```

Trainer detail: "gsplat 1.5.3, SH degree 3, SSIM+L1, MCMC-free densify, hard cap".

### 12.3 Shipped tech stack (writeup §2.6, verbatim)

| Layer | Component | Notes |
|---|---|---|
| Ingest | OpenCV, ffmpeg 9.0.1 | sharpness-ranked keyframing |
| SfM | **COLMAP 4.1.1 (CUDA build)** | driven by `scripts/run_colmap.py` |
| Training | **PyTorch 2.4.1 + gsplat 1.5.3** | prebuilt Windows wheel, zero compilation |
| Geometry | numpy, plyfile, `@playcanvas/splat-transform` | heightfields, voxel shells |
| Runtime | **PlayCanvas engine + ammo.js (Bullet) WASM** | vendored, no build step |
| Navigation | **recast-navigation (MIT)** | offline bake only, `tools/navbake/bake.mjs` |
| Frontend | Vanilla ES2022 + WebXR | ~10,100 hand-written lines |
| Test harness | Playwright + headless Chromium | drives the real page |

Repo scale: ~18,000 lines of first-party code; `scripts/` = 35 modules, ~6,800 lines; `tests/` = 12 files, ~550 assertions; ~4 GB disk, 313 MB COLMAP CUDA provider in git-LFS.

### 12.4 Quality presets measured on the target card

| Preset | Train width | Steps | Gaussian cap | Peak VRAM |
|---|---|---:|---:|---:|
| `smoke` | 640 px | 300 | 150k | ~0.5 GB |
| `standard` | 640 px | 12,000 | 350k | 0.72 GiB |
| `high` (default) | 1280 px | 15,000 | 3.0 M | 1.26 GiB, 14 min on a temple pass |
| `ultra` | 1440 px | 45,000 | 4.0 M | for 8 GB+ GPUs |

"The temple converged at **451k splats** under a 3M cap." Note the two published tables disagree — the writeup says 1280 px / 3.0M cap, `README-MVP.md:99-103` says 800 px / 1.5M cap. Pick one and date it.

### 12.5 Console

`cd groundcontrol && npm run dev` (Next.js 16 + Tailwind v4) → `http://127.0.0.1:3000`. Routes: `/` Mission, `/pipeline`, `/survey`, `/deliverables`, `/challenges`, `/applications`.

Design rule, `docs/readiness/00-INDEX.md`: "a value the server did not report renders as an em dash with a `not evaluated` chip — never a plausible number." Footer: "**Every reading is labelled measured, projected from measured rates, or synthetic fixture. A module existing is never shown as a module running.**"

### 12.6 How to run it

```bat
.venv\Scripts\python.exe _serve.py 8137 .                          REM backend
python scripts/bootstrap.py --with-train                          REM one-command install
.venv\Scripts\python.exe pipeline.py doctor                       REM toolchain health
.venv\Scripts\python.exe pipeline.py run room_w_jsonl --quality smoke
.venv\Scripts\python.exe pipeline.py view room_w_jsonl
.venv\Scripts\python.exe tests\check_all.py                       REM fast suites (CI runs this)
.venv\Scripts\python.exe tests\test_e2e.py                        REM every take in videos/, ~30 min
.venv\Scripts\python.exe survey.py prepare acceptance             REM survey lane
.venv\Scripts\python.exe survey.py reconstruct acceptance --allow-gpu --dense-profile survey
.venv\Scripts\python.exe survey.py evaluate acceptance             REM -> evaluation.json
mvp.bat run rocks                                                 REM legacy shim
```

### 12.7 Intentionally unwired (`02-wiring-ledger.md`)

28 modules, three states: run path / tested / manual. Still unwired by design: `survey_measure` into the viewer, generative completion of occluded surfaces, progressive/streaming as default, dashboard launch of the survey run. The stated reason for not wiring generative occlusion completion: "a single pass cannot see a hidden face, and inventing one would present a guess as a measurement."

---

## 13. Why single-pass 3D is genuinely hard — the research position

### 13.1 The core framing (`06_Splat_Completion_Research.md` §3, verbatim)

> "Roughly **95% of what is published as '3DGS inpainting' is object removal**… Ours is **view extrapolation**: A 180° arc was **never photographed**. There is **no surrounding context** on the far side. The missing region is **enormous**. The correct answer must be **invented** — an extrapolation problem."

§6.1 domain problem: "Every tool in this category is an **object** generator: trained on Objaverse-style single assets, background-removed, normalised into a **unit cube**, with **no metric scale**."

§6.3 category correction: "**VGGT, MapAnything, Pi3, AnySplat, MVSplat, DepthSplat, NoPoSplat do not generate unseen geometry.**"

§8 closing line, the best closer in the repo:
> "Second, and cheapest by a wide margin: **fly the missing side.** Twenty more minutes of drone time beats every entry in this document. Every method here is inventing plausible fiction."

And the project's stated philosophy: "A flat plausible wall plus an honest coverage mask is more consistent with this project's existing philosophy than a generated facade. If the reconstruction is ever used for anything measurement-adjacent, '**visibly empty**' is a better failure mode than '**confidently wrong**'."

### 13.2 Three technology families (`01_Research_and_Technology_Survey.md`, verbatim)

| Family | Examples | Strengths | Weaknesses |
|---|---|---|---|
| Classical photogrammetry (SfM+MVS) | COLMAP, OpenDroneMap, Metashape, Pix4D | Mature, cm-accurate, natively georeferenced, measurable meshes | Slow per-scene optimization; fails on blur/low texture; needs many overlapping views |
| Deep SLAM / VO | DROID-SLAM, DPVO/DPV-SLAM, ORB-SLAM3 | Real-time tracking, handles long sequences | Up-to-scale only; dense depth heavy; drift without loop closure |
| Feed-forward transformers + neural rendering | VGGT family, Pi3, MapAnything, DA3 + gsplat | Seconds-not-hours geometry; photoreal from sparse/single-pass views | Scale ambiguity (needs GPS fusion for metres); long-video chunking required; mesh extraction is its own step |

### 13.3 What the four anchor repos do not cover (`01_…` §3, verbatim)

1. Metric scale + georeferencing — "**none of the four produce meters or WGS84 coordinates**"
2. A commercially-clean feed-forward geometry model
3. Keyframe extraction + per-frame GPS
4. Dynamic-object removal
5. Meshing legality
6. Streaming/digital-twin formats
7. Analytics

### 13.4 Can COLMAP be skipped? (`07_Video_to_3D_Alternatives_2026.md` §3)

| Candidate | Metric scale? | Full-res VRAM | 6 GB verdict |
|---|---|---|---|
| VGGT | No | not stated | Few frames; commercial use needs gated `VGGT-1B-Commercial` |
| MapAnything | **Yes** | 2,000 views on 140 GB | Cloud tier only |
| Depth Anything 3 | — | not stated | Giant/Large = CC-BY-NC 4.0 |
| Pi3 / Pi3X | Pi3X only | not stated | BSD-3 code but weights CC-BY-NC 4.0 |
| **MoGe / MoGe-2 / MoGe-3** | **Yes (MoGe-2/3)** | 60 ms/image, ViT-L, FP16, A100 or RTX 3090 | **MIT code — best fit on this box** |

> "**COLMAP has no VRAM requirement at all.** A feed-forward model that needs 24 GB to match COLMAP on 300 frames is not a replacement on a 6 GB laptop, even if it is 100× faster."
> "Drift on long video. It is *the* open problem… VGGT-Long's own title — 'kilometre-scale long RGB sequences' — is a statement that vanilla VGGT does not hold up over long captures."

### 13.5 The "real-time SLAM + splat" family is architecturally unavailable (07 §1, verbatim)

> "**There is no live/streaming Gaussian-splatting path on 6 GB Windows today. Not one.** Every SLAM-and-splat system is Linux + full CUDA toolchain + a *forked* compiled rasterizer… **it is architectural.**"

| Name | Speed claim | Why it fails here |
|---|---|---|
| MonoGS (CVPR 2024 Highlight + Best Demo) | "up to 10 fps on fr3/office" | RTX 4090, dev branch, unmerged, Ubuntu only |
| Photo-SLAM (CVPR 2024) | "Real-time" in title, **no frame rate stated anywhere** | OpenCV built with CUDA + LibTorch ≤2.1.2, GPL-3.0 |
| RTG-SLAM (SIGGRAPH 2024) | makes no fps claim | needs ORB-SLAM2 Python binding built via `build_orb.sh` |
| Splat-SLAM (arXiv 2405.16544) | not stated | **ARCHIVED** since 2026-03-10 |
| Gaussian-LIC2 (ICRA 2025 / IJRR 2026) | "in Real Time", no number | CUDA 11.7 + TensorRT **and a LiDAR rig** — wrong sensor class |

"Treat every 'real-time' claim in this family as unquantified."

### 13.6 VRAM wall for view-extrapolation methods (06 §2.1, all authors' own stated hardware)

| Method | Venue | Stated hardware |
|---|---|---|
| G4Splat | ICLR 2026 | **A100 80 GB**; dense "3090 24 GB" |
| GSFix3D | 3DV 2026 | RTX 4500 Ada 24 GB |
| ViewCrafter | TPAMI 2025 | 13.8 GB / 50 s at 320×512, 25 frames |
| Inpaint360GS | WACV 2026 | RTX 4090 |
| RI3D | ICCV 2025 | 2080 Ti 11 GB |
| WonderJourney | CVPR 2024 | README: "requires 24 GB GPU memory" |
| Invisible Stitch | 2024 | "at least 16 GB VRAM" |
| GenFusion | CVPR 2025 | not stated (est. ~24 GB) |
| Gaussian Grouping / AuraFusion360 / GScream | — | **never stated** |

Extrapolation-targeted methods: GSCompleter (arXiv 2604.20155, paper only, no code) · Bolt3D (ICCV 2025, Google/Oxford, 6.25 s feed-forward, repo 404) · GenFusion (CVPR 2025, MIT, `--outoutpaint_type rotation --rotation_angle 90`) · RI3D (ICCV 2025, no LICENSE) · GOF, ExtraGS, ReconSplat (ECCV 2026).

> "**Google published nothing usable.** ReconFusion, Cat3D, CAT4D — no code, no weights, not reproducible."
> "**No 3DGS equivalent of Nerfbusters exists.**"

### 13.7 The occlusion-completion ladder (06 §7, Rungs 0–6)

Rung 0 visibility pruning + `splat-transform --filter-cluster --filter-floaters` · Rung 1 manual patch tool (~900–1,200 lines of JS) · Rung 2 automatic flat-wall fill · Rung 3 symmetry mirror gated on **one-directional Chamfer** (accept if median residual < ~0.15 m) · Rung 4 2D inpaint + retrain · Rung 5 3D generator · Rung 6 training-time regularizers.

Verified Rung 3 math: SH mirror transfer matrices, off-diagonal max **4.3e-15** for axis-aligned planes against **1.00 / 0.79** for arbitrary planes; `T@T == I` in all four cases.

### 13.8 AI cannot clean splats, but can fill them (`08_Splat_Cleaning_and_Generation_2026.md` §0, verbatim)

> "**'Can an AI clean my splats?'** — **No.** As of 2026-09 there is no trained, downloadable real-vs-floater classifier you can point at a `.ply`."
> "**'Can an AI generate splats to fill missing places?'** — **Yes as a category, no for this problem.**"

TIDI-GS (arXiv 2601.09291) and Clean-GS (2601.00913, claims 60–80% compression) are both training-time plugins with **no code found**. SuperSplat stars 9,944 → 10,011; splat-transform 1,305 → 1,317; LichtFeld-Studio 3,682★ GPL-3.0, driver 570+/CUDA 12.8+, **prebuilt Windows binaries are paid**. The most on-domain paper found: "**Feed-Forward Gaussian Splatting from Sparse Aerial Views**, arXiv 2605.19949 (2026-05-19)."

### 13.9 Sub-field growth

Streaming field growth went from roughly **3 papers in mid-2025 to 20+ in 2026**. Named: StreamVGGT (2507.11539), XStreamVGGT (2602.21780), Long3R (2507.18255), Point3R (2507.02863), VGGT-Long (2507.16443), AnythingReality (2607.09260), plus 18 more with arXiv IDs.

### 13.10 A concrete untried lever

`MCMCStrategy` ships in the pinned gsplat 1.5.3 (submodule `937e29912570c372bed6747a5c9bf85fed877bae`) and is **unused**. Defaults: `cap_max=1_000_000, noise_lr=5e5, refine_start_iter=500, refine_stop_iter=25_000, refine_every=100, min_opacity=0.005`. Paper: *3D Gaussian Splatting as Markov Chain Monte Carlo*, arXiv 2404.09591. Blocker: signature mismatch, `step_post_backward(..., + lr: float)` required versus `+ packed: bool`. Status: **not implemented** — "the cheapest untried quality lever in the repo."

---

## 14. Prior art with citable numbers

| Work | Venue | Number |
|---|---|---|
| VGGT | CVPR 2025 **Best Paper** | ~1B parameters, single forward pass **< 1 s**, ⭐14.3k |
| VGGT-Omega | CVPR 2026 Oral | Peak VRAM on A100 at 624×416: 1 frame ≈ **6 GB**, 100 frames ≈ 13.4 GB, 200 frames ≈ 20.8 GB, 500 frames ≈ **43 GB** |
| VGGT-Long | ICRA 2026 | "kilometre-scale long RGB sequences", 24 GB RTX 4090, ~50 GB disk for a 4,500-frame sequence, sample ~1 fps |
| DROID-SLAM | NeurIPS 2021 Oral | ≥11 GB GPU even for small benchmarks, 24 GB for large evals, BSD-3 |
| DPVO | NeurIPS 2023, MIT | sparse-patch VO — the chosen replacement |
| gsplat | Apache-2.0, ⭐5.6k | claims up to **4× lower training memory**; "large outdoor/drone scenes reach good quality in roughly 15–60 min on a 24 GB GPU" |
| MapAnything | Meta, ⭐3.7k | up to 2,000 views on big GPUs; `facebook/map-anything-apache` = Apache-2.0 |
| Pi3 | ICLR 2026, ⭐2.1k | BSD-3 code, CC-BY-NC 4.0 weights |
| Depth Anything 3 | ⭐6.2k | DA3-Streaming "does ultra-long video under 12 GB VRAM"; Giant/Large = CC-BY-NC 4.0 |
| MoGe-2 | MIT | 60 ms/image, ViT-L, FP16, A100 or RTX 3090 |
| splatwalk | ⭐6 | "empty floors on large sparse outdoor scans" |
| SuperSplat | — | 9,944 → 10,011 stars over the survey period |

### 14.1 Author-reported external numbers (NOT our benchmarks, 09 §8.2)

| Method | Published number | Hardware / conditions |
|---|---|---|
| LightGlue (DISK card) | ~44 ms/pair FP32 | GPU model and keypoint count not stated |
| SAM 2 | **43.8 FPS** Hiera-B+ / **30.2 FPS** Hiera-L | A100, batch 1, 1024×1024 |
| 2D Gaussian Splatting + TSDF | — | RTX 3090, DTU 800×600, 15k/30k iters |
| VGGT-SLAM 2.0 | ~120 ms/frame, 8.4 FPS; semantics 158 ms/frame | 16-frame submaps, RTX 3090; raw VGGT limited to ~60 frames on 24 GB RTX 4090 |
| MASt3R-SLAM | ~15 FPS | RTX 4090 + i9-12900K, max side 512 px, every-2nd-frame |
| DPV-SLAM / DPVO | 1–4× real-time, ~5–7 GB | RTX 3090 |
| DROID-W | ~10 FPS | RTX 3090 + 16-core CPU, capture res 1200×1600 |
| Gaussian Opacity Fields | 24.2 min/scene optimization | A100, 30k iters |

### 14.2 Prior-art hobbyist pipeline, four field-validated lessons

`ch1bo/drone-reconstruction` (Jan 2026, ⭐7, ~40 commits): monocular DJI → ffmpeg 2 fps → COLMAP sequential → `.SRT` to ENU → `colmap model_aligner` Sim(3) preferring `rel_alt` → CUDA PatchMatch MVS → optional `splatfacto` ~30 min train.

1. `rel_alt` (IR/baro, ~0.1 m precision) beats GNSS altitude (~10–20 m error)
2. sequential matching is right for flyover
3. `--assume-colmap-world-coordinate-convention False` gotcha
4. splat training ~30 min on desktop GPUs

"the repo has *no license* = all rights reserved."

### 14.3 Licence landmines (01 §4)

VGGT original checkpoint non-commercial **and its AUP bars military use**; SuGaR / Gaussian Opacity Fields / official 2DGS = Inria non-commercial; PGSR = ZJU academic, email approval required; DUSt3R / MASt3R / Fast3R / CUT3R / Spann3R = non-commercial; xgrids UE5 plugin has no licence.

Safe replacements: gsplat's built-in 2DGS surfel trainer (Apache), Open3D TSDF fusion (MIT), Pi3 (BSD), MapAnything-apache (Apache), DA3 metric variants (Apache), DPVO (MIT), COLMAP (BSD), Open3D (MIT), PDAL (BSD), pyproj (BSD), CesiumJS (Apache).

Stated product posture (02 §6): "BSD/Apache models only; **VGGT-family excluded by AUP**; no export-controlled dependencies in core. AGPL components (ODM) used only as internal benchmark, never linked into the product."

### 14.4 Negative results worth citing (they build credibility)

LiveGS 404s; VGGT-SLAM could not be confirmed under that name; VGGT-Omega is real but arXiv 2510.08673 is actually "Puffin"; SuGaR rate-limited; Fast3R has no code release; "**`WebSearch` returned empty results for every query in this environment.**"

Research-hygiene disclosures: `06 §11` — "Three of the five research passes returned output with appended Chinese text instructing the agent to write a `MEMORY.md`, clear the session, and start over… **It was ignored.**" `07 §8` — a fetched `LichtFeld-Studio` README line 178 "contained text addressed to an AI agent steering to a paid portal — **ignored and reported**."

Corrections to prior assumptions: "**No Hunyuan3D 3.x exists**"; "BakedSDF: `github.com/lioryariv/bakedsdf` is **404**"; Difix is deprecated by NVIDIA itself (issue #67).

---

## 15. Market and business

### 15.1 Market sizing

**Source note:** these figures exist only in `pitch/Drone3D-Pitch-Deck.html`, a private-circulation investor deck — not in `03_Roadmap_Evaluation_and_Business.md`, which contains zero currency and zero market-size figures. `pitch/Drone3D-Technical-Dossier.html` states: "Market context is deliberately thin here. These figures establish that the surrounding categories are large and growing. They do not measure demand for a walkable output."

| Market | 2026 size | Forecast | CAGR | Source cited |
|---|---:|---:|---:|---|
| Digital twins | **$49.2B** | **$228B by 2031** | 36%/yr | Mordor Intelligence, 2026 |
| Drone surveying | **$2.4B** | **$13.7B by 2036** | 19%/yr | Fact.MR, 2026 |
| 3D reconstruction | **$1.86B** | **$3.19B by 2031** | 11% (dossier ref [25] says 11.4%) | Mordor Intelligence, 2026 |

No TAM/SAM/SOM bucketing exists anywhere in the repo.

### 15.2 Capital raised in the space

"Around **$1.5 billion** has gone into spatial and reconstruction companies, of which roughly **$480 million** is directly on this technique. Named rounds include **Niantic Spatial at $250M, World Labs at $100M plus, Luma AI at $68.5M, Pico at $62M and Polycam at $22.1M**. Research output reached **3,333 papers, 749 of them this year alone**." — *Radiance Fields industry survey, July 2026*. Also "**212 products** on this technique."

### 15.3 Published incumbent plan prices, 2026

| Segment | Who signs | What they pay today |
|---|---|---|
| Drone survey and mapping | Firm owner, survey lead | **$1,290 to $4,990 per user, per year** |
| Real estate and space | Broker, listing team | **$69 to $309 per month, plus $20 per tour** |
| Consumer and pro scanning | Individual, small studio | **$0 to $100 per month** |
| Facilities and digital twins | Site owner, operations | **Six figures per site**, integrated |

`01_…` adds: "Agisoft Metashape / Pix4D — Commercial (**$179–$3,399 perpetual** / subscriptions) … neither ingests video natively."

### 15.4 Competitive landscape (03 §1, prose, one line, no table)

"**DroneDeploy/Pix4D/Agisoft** (mature, multi-pass-centric, no walkability), **Polycam/Luma** (consumer capture, no georef rigor), open-source **ODM** (batch CLI, no interactivity). Differentiation = single-pass video-first ingestion + speed + the walkable/scenario layer nobody ships with georef rigor."

### 15.5 Business model and go-to-market

Open-core: core + viewers Apache/MIT/BSD OSS; commercial tiers = orchestration cloud, scenario editor, enterprise integrations, support.

Tiers (pitch deck slide 15): per certified scene with failed runs free · team seats · site licence (on-prem, footage cannot leave the building).

Go-to-market (pitch deck slide 12): 1. the survey firm, already paying and already complaining → 2. insurance and inspection → 3. twin, robot and game.

Value proposition: "One flight, minutes of compute, and you're standing inside the mission."

### 15.6 Effort and cost

"1–2 engineers + 1 part-time GIS/QA → credible demo at ~month 3–4, pilot-ready ~month 6." Cloud GPU rental "~$0.35–2.50/hr".

Hardware profiles (02 §5):

| Profile | Hardware | Target latency, 10-min 4K pass |
|---|---|---|
| A — Dev | RTX 3050 6 GB | Coarse product < 1 hr incl. cloud round-trip |
| B — Workstation | 24 GB (RTX 4090/3090) | Full ~1–3 hr; coarse ~15–25 min |
| C — Cloud batch | A100/H100 ×N | Full ~20–45 min |
| D — Edge (stretch) | Jetson Orin 64GB | research track |

---

## 16. Applications — all eight from the problem statement

The eight official "Potential Applications" (SIH26158 PDF printed p. 37), each with its kill constraint:

| # | Application | What single-pass changes | Constraint it removes |
|---|---|---|---|
| i | Border and strategic area mapping | A continuous corridor model from one sortie instead of a multi-day campaign | One overflight window, often at night, often with a single airframe available |
| ii | Disaster damage assessment | Damage visible in the field while the response is still being mounted | No time for multi-pass grid planning before the weather closes |
| iii | Urban planning and smart cities | Metric city models built on flights that are already funded | Municipal budgets cover one mapping flight, not a survey programme |
| iv | Infrastructure inspection | Facade and rooftop geometry measured without scaffolding or road closure | Assets are live; closures are expensive and rationed |
| v | Construction progress monitoring | Weekly ground truth from one pass, directly comparable to the week before | Comparing two epochs without re-flying both |
| vi | Archaeological documentation | Non-contact recording of sites that cannot safely be walked or touched | Fragile, inaccessible or legally protected sites |
| vii | Digital twin generation | The geometry layer a twin needs, produced per site and per day | Twin budgets are per-site, not per-survey-campaign |
| viii | Military reconnaissance and mission planning | Rehearsal on the target before the sortie is ever flown | Single pass, contested airspace, no second chance |

`docs/APPLICATION_PLAYBOOKS.md` restates all eight and adds: "**≤1 m absolute with ordinary GPS is on the edge.** Consumer GNSS is ±1–3 m, worse vertically."

---

## 17. Engineering problems already solved — five debug stories with measured cause

`SE_Lab_Project_Writeup.md` §2.7. These read as "we debug with data", which is the most persuasive framing available:

**(a) The character was standing on the sky.** Two disjoint collider shells: ground at −16…−11 m and a canopy crust at +5…+9 m. A bimodality test found it by band density: **2,159 splats/m at +4 m → 135/m at +12 m → recovers to 1,727/m by +19 m**. A naive sparseness test "**would have deleted 86% of a good frame**".

**(b) The collider was unwalkable.** "**1.05 m of riser every 0.6 m** along the route", against a capsule radius of 0.34 m.

**(c) Autopilot livelock.** "**288.6 rad of yaw** … 288.6 / 2.2 rad s⁻¹ = **131 of its 149 seconds** spent spinning in a 1.18 m circle."

**(d) Half-cell indexing error in three places.** Viewer ground **0.101 m** from physics ground.

**(e) "Landscape video crashes" was never about landscape.** "**0 Gaussians kill the process instantly**"; the densifier pruned the last seed at **step 600**.

---

## 18. The ten most defensible numbers

All verifiable on disk.

1. **1,568,438** measured points delivered in a real CRS (`scratch/real_delivery/summary.json`) — largest measured cloud in the repo.
2. **5 of 6** official formats actually written with a real CRS, EPSG:32643 WGS 84 / UTM 43N (`delivery_manifest.json`).
3. **0.2783 m/unit** georeferencing scale confirmed by two independent estimators agreeing to **0.2%** — and **1.14 m** corrected hold-out RMSE, honestly labelled as against a drifting phone VIO track, not surveyed truth.
4. **74.7%** sparse time saved (946.9 s → 239.7 s, 291 frames) at **0.593 px** mean reprojection error.
5. **4.2×** estimated dense-time cut and **3.8×** mean-parallax increase at equal frame budget (68 frames vs 288).
6. **2.33×** dense speed-up with 0.946/0.937 precision/recall at 0.25% of the cloud diagonal — the *adopted* variant, and F-score slightly improves.
7. **7.8%** improvement in mean reprojection error (0.3586 → 0.3305 px) with 4.85% of pixels masked for dynamic objects, and registered cameras unchanged at 72/72.
8. **3,960.1 s predicted vs 4,001.4 s measured — 1.0% error**, with the model still honestly reporting `fits_deadline: false`.
9. **605 CPU tests pass** across 28 suites (plus 151 survey tests and 107 unit checks), GPU never touched, `torch` never imported in the fast path.
10. **65.3 m walked, 16/16 waypoints, 0 falls, +0.008 m foot gap, 0.000 m viewer-vs-physics disagreement** — up from 19.9 m and 1/20.

---

## 19. Numbers that do NOT exist in the repo

Do not put these on a slide. They are targets only.

- **No Chamfer distance** measured against LiDAR or ground truth. The word "Chamfer" appears only as a target in doc 03 §2 and as a proposed test in doc 06 Rung 3.
- **No SSIM, no LPIPS, no held-out-view PSNR benchmark.** "PSNR" appears only as gsplat's training-loop loss (20.05 dB on `room_w_jsonl`; peak 23.47 dB; 12.87 dB at step 200 of the regression train).
- **No precision/recall/F-score against a reference surface.** The P/R/F numbers in the repo are self-agreement between ablation variants — "Agreement is measured against our own slowest cloud, not surveyed truth."
- **No measured 10-minute run.** Longest clip on the machine: 151.6 s. Highest resolution: 1280×798.
- **No completeness percentage, no ATE/RPE against RTK, no scale-agreement percentage, no walkability percentage** against any reference — all exist only as targets.
- **No third-party reader validation.** No GDAL, laspy or PDAL installed.
- **No georeferenced outdoor UAV scene.** All georeferencing evidence is from a phone VIO/AR track.
- **No live incremental streaming reconstruction.**
- **No UV-space textured mesh.** Poisson gives per-vertex colour; the manifest states `claims_textured_mesh: false`.

---

## 20. Stated limitations — the honest list

`SE_Lab_Project_Writeup.md` §2.10, verbatim:
> "**scale is not measured, only named** — no GPS means metres are only as good as the operator's single number, and nothing can distinguish a wrong value from a wrong-but-self-consistent one. The shipped ground is a **heightfield**, so overhangs and caves cannot be expressed… **Dynamic objects have no masking stage.** Low-texture grass makes far-field soft. **Two of eight scenes fail the route gate** because of scale drift, and the fix — per-region rather than per-scene scale — is the work in flight."

Note: that writeup predates the dynamic-object masking ablation in §6 above, so the "no masking stage" line is now out of date. Masking works; it is not yet wired into the default run.

`README.md` "Known limits": the `--quality smoke` table "proves nothing fails, not final visual quality"; "**Indoor phone takes walk 15–36 m, not 65 m**"; "**Windows only today.**"

`README-MVP.md` "Known weak spots": scale not measured (no GPS); grass low-texture; collider is a heightfield (**37% of cells dilated in from neighbours**); ground underlay off by default; no dynamic-object masking; "**12 s clip** → 114 keyframes is on the low side".

`pitch/Drone3D-Technical-Dossier.html` p.18, "Technical limitations, unrounded": indoor human-scale walking **Not achieved** · clean mesh/drawing export **Not available** · hosted service/accounts **Not built** · cross-platform **One platform only** · highest quality tier **Never run** · visual fidelity vs published work **Below norms** · blind review coverage **One scene** · "**Two of ten scenes are currently refused certification by the system's own gate, including the one used in the public demonstration.**"

---

## 21. Identity gaps that must be closed before submission

- **Team ID, team name, theme:** all placeholders — `scratch/slidebuild/build_deck.py:277-278` reads "**to be declared on the SIH portal**".
- **Institution:** no IIT/NIT/university/college string exists anywhere in the repo.
- **Team members:** exactly one name exists in the whole repo — **Kris Garg** (`pitch/Drone3D-Technical-Dossier.html:28, 1077`; `pitch/Drone3D-Pitch-Deck.html:15, 672`). GitHub handle `krisgarg25`. No roll numbers, no guides.
- **Submitted by / Roll No. / Project guide / Batch:** blank underscores in `SE_Lab_Project_Writeup.md:14-16` and `SE_Lab_SRS_DFD_UML.html:130-131`.
- **`(NTRO)`** appears once, unexplained, in `SE_Lab_SRS_DFD_UML.html:132`.
- Course: Software Engineering Laboratory (SE Lab), Submission I. Date of submission 05/09/2026. Due **2 October 2026** — Google Classroom, single PDF.
- Public repo: `github.com/krisgarg25/Drone_Phone_video_to_playable_3d_world`, MIT.

---

## 22. Project naming across the repo

| Name | Where |
|---|---|
| "**WALKABLE** — Turning a Single Handheld or Drone Video into a Metric, Real-Time Walkable 3D World" | `SE_Lab_Project_Writeup.md:7` |
| "**WALKABLE** / WALKABLE — Single-Pass Video to a Metric, Walkable 3D World" | `SE_Lab_SRS_DFD_UML.html:132` |
| "**DRONE3D**" | `scratch/slidebuild/build_deck.py:259` |
| "**Drone3D Studio**" | `pitch/Drone3D-Pitch-Deck.html:2` |
| "**Ground Control — Single-Pass Drone to 3D Model**" | `groundcontrol` app |

Pick one. "DRONE3D" is the name already used on the SIH deck.

---

## 23. Key file index

| Path | What it holds |
|---|---|
| `09_SIH26158_Evaluation_and_Improvement_Report_2026-09-22.md` | Authoritative internal evaluation; §9 is the measured-runtimes section |
| `01_Research_and_Technology_Survey.md` | Three technology families, accuracy hierarchy, licence landmines |
| `02_System_Architecture_and_Pipeline_Design.md` | M0–M10 design intent, challenge/mitigation matrix, hardware profiles |
| `03_Roadmap_Evaluation_and_Business.md` | Evaluation targets table, competitive prose, effort, roadmap phases 0–5 |
| `06_Splat_Completion_Research.md` | View-extrapolation framing, VRAM wall, Rungs 0–6 |
| `07_Video_to_3D_Alternatives_2026.md` | Can-COLMAP-be-skipped table, SLAM+splat unavailability, negative results |
| `08_Splat_Cleaning_and_Generation_2026.md` | Can-AI-clean-splats: no. Can-AI-fill: category yes, this problem no. |
| `SE_Lab_Project_Writeup.md` | Shipped 18-stage pipeline, stack, presets, results, 5 debug stories |
| `docs/GAPS_AND_OPTIMIZATIONS.md` | M2/M8/M11/M12/M13 metric tables, E5 ruler refusal, DSM mirror defect |
| `docs/APPLICATION_PLAYBOOKS.md` | Eight applications, 270-keyframe sizing, GNSS realism |
| `docs/readiness/00-INDEX.md` … `05-challenges-and-applications.md` | Evidence map, acceptance protocol, wiring ledger, challenge verdicts |
| `scratch/gpubench/*.json` | ablations, selection_ablation, speed_ablation, dense_ablation, mono_ablation, scale_validation |
| `results/verdicts/*.md` | Blind A/B verdicts, WIN 10/10, walkability PASS |
| `results/texture_comparison/*/comparison.json` | The only held-out-view MAE numbers |
| `pitch/Drone3D-Pitch-Deck.html` | The only source of market sizing and pricing |
| `ppt/SIH2026-IDEA-Presentation-Format.pptx` | The official SIH 2026 idea-presentation template |

---

## 24. Research-hygiene note

This brief was compiled by reading the repository and transcribing figures from its own recorded artifacts. Where a figure is a target rather than a measurement, this document says so. Where a figure was superseded by a later correction in the same repository, both values are given with the correction named. No number in this document was supplied from outside the repository except the market sizing and pricing in §15, which are attributed to Mordor Intelligence, Fact.MR and a Radiance Fields industry survey as cited by `pitch/Drone3D-Pitch-Deck.html`.
