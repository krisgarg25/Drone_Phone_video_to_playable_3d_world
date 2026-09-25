# Evidence map - official target against the artifact that would prove it

Status vocabulary, used consistently:

- **delivered** - runs in the pipeline, produces the artifact, covered by tests.
- **unproven** - the mechanism exists; no qualifying data has ever been through it.
- **absent** - the capability does not exist yet.

Weights are from `SIH26158.pdf` page 38. They are weights, not a formula, so the
"what moves this" column names evidence, never points.

## Accuracy - weight 30, target ≤ 1 m spatial

| | |
|---|---|
| Status today | **unproven** |
| Artifact that proves it | `evaluation.json:criteria[accuracy].metrics` from ≥ 8 independent surveyed checkpoints, hold-out split |
| Mechanism | `survey_accuracy.report` via `survey_assess.accuracy_from_checkpoints`, wired into `evaluate` |
| What is delivered | fit/hold-out stratified split; unaligned, SE(3) and Sim(3) tracks reported separately; verdict against a named statistic |
| What would be a lie | quoting `georeference.json:fit_rmse_m` as accuracy - it is a residual of the GPS fit against itself |
| Best current number | 1.14 m hold-out RMSE against a **drifting phone VIO track**, not surveyed truth; see findings §2.1 |
| What moves it | the surveyed checkpoints in the acceptance dataset |

## Completeness - weight 20, target entire visible scene

| | |
|---|---|
| Status today | **partly delivered, unproven overall** |
| Delivered | region observability (observable / weak / unobservable cells), per-point support, occlusion-checked visible support, measured / weak / unobserved layering |
| Unproven | agreement with an independent reference surface - the only thing that can claim "entire visible scene" |
| Artifact | `run.json:completeness`, `evidence/evidence_summary.json`, `evaluation.json:model_completeness` |
| Honest limit | a single pass cannot see a hidden face. Nothing is inferred into a measured product; `survey_occlusion` publishes no inpainting entry point |
| Also required | `coverage_denominator.excluded_area_m2` must be published with any coverage percentage, or a denominator quietly shrinks the hole |

## Speed - weight 20, target < 15 min for a 10-min video

| | |
|---|---|
| Status today | **unproven**; the gate itself is implemented |
| Delivered | 600 ± 0.5 s / < 900 s gate with per-stage timing, decoder-verified duration, recorded host CPU/RAM, and `diagnostic_only` downgrade when either is missing |
| Measured | rocks 806.4 s (12 s clip), room 4001.4 s (139 s clip) - both stitched across failures, neither a full-deliverable run |
| Predicted | 615-650 s for a 600 s clip, labelled `predicted from measured per-image rates` |
| New this pass | `run.json:throughput` - per-stage prediction, progressive window plan, first-usable-output time, all carrying the not-measured disclaimer |
| What moves it | one fresh uncached run of the acceptance clip, with GPU model and driver recorded (the run record notes it captures CPU and RAM only today) |

## Reconstruction type - 3D mesh / point cloud

| | |
|---|---|
| Status today | **delivered** |
| Point cloud | fused `dense/fused.ply` -> georeferenced `cloud.ply` / `cloud.las` / `cloud.xyz` |
| Surface mesh | `poisson_mesher` (CPU) -> `dense/mesh.ply` -> `surface.obj` with faces and per-vertex colour |
| Textured mesh | **absent** by choice: no UV atlas exists, and the manifest records `claims_textured_mesh: false` |
| Caveat carried in the manifest | Poisson vertices are an isosurface estimate, not individually measured points |

## Output formats - OBJ, PLY, LAS, GeoTIFF, .glb/.gltf, .fbx

| | |
|---|---|
| Status today | **delivered inside the run; externally unvalidated** |
| Ledger | `products/delivery_manifest.json` - each of the six named formats as `delivered` or `not_delivered` with the exporter's own reason |
| Georeferencing | local ENU set **and** a UTM set whose EPSG is derived from the scene's own GPS origin (`survey_crs`), plus `positions_wgs84.csv` |
| GeoTIFF | real DSM raster in projected metres with a PROJCS WKT and the correct EPSG geokey (see findings §2.3) |
| Outstanding gap | every writer is re-read by this repository's own reader. No GDAL, laspy or PDAL is installed, so spec compliance against a third-party reader is unverified |

## Visualisation - web or desktop viewer

| | |
|---|---|
| Status today | **delivered for splats; partial for survey products** |
| Delivered | Gaussian-splat walkable viewer with colliders; survey console showing criteria, evidence chain, provenance and download links for every artifact |
| Partial | the viewer loads splat assets, not a selected survey run's mesh/cloud |
| Absent | interactive point-picking that calls `survey_measure`; measurements render from an operator-authored file |
| Note | UI was explicitly deprioritised in this pass |

## Innovation - weight 15

| | |
|---|---|
| Status today | **delivered as engineering, unproven as advantage** |
| The defensible claims | GPS enters bundle adjustment as per-camera priors rather than a post-hoc stamp; per-point evidence and occlusion-checked support; measured/weak/unobserved kept separate with no inpainting; a delivery ledger that reports refusals; a CRS derived from the scene instead of asserted |
| What a judge will ask | which of these measurably improved a reconstruction. Each needs an ablation on the acceptance dataset |
| Ablations that exist | sequential vs exhaustive matching (239.7 s vs 946.9 s, but 233/288 vs 288/291 registered); dense profiles; both were run on sub-1080p clips |

## Scalability - weight 10

| | |
|---|---|
| Status today | **unproven** |
| Delivered | frame-budget profiles, progressive window planning, per-image rate model that must be re-fed when hardware changes |
| Absent | a scaling campaign: the same scene at increasing frame counts and resolutions with memory and wall-clock curves |
| Hardware reality | one RTX 3050 6 GB laptop; peak samples are whole-GPU at 0.4 s intervals, so they are floors |

## User interface - weight 5

| | |
|---|---|
| Status today | **delivered for the evidence console** |
| Verified | headless Chrome: six criteria render, invalid metadata and overwrite conflicts produce actionable errors, checkpoints display and the report downloads, 390 px has no horizontal overflow, keyboard focus is visible, offline failure and recovery work, and declining the GPU confirmation sends no run request |
| Outstanding | survey reconstruction is still launched from the CLI behind `--allow-gpu`; the dashboard's Run lane starts the older splat pipeline |

## The one-line verdict

Everything that can be built and verified without the official dataset is now wired
and tested. The 70% of the score that rests on accuracy, completeness and speed needs
one qualifying 10-minute 1080p pass with surveyed checkpoints - and no amount of
additional code substitutes for it.
