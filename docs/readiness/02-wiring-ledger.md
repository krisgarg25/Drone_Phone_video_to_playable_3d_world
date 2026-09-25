# Wiring ledger

What each survey module does today, and whether it actually runs. Three states only:

- **run path** - executes during `survey.py prepare` / `reconstruct` / `evaluate`.
- **tested** - unit-tested and importable; nothing in the pipeline calls it.
- **manual** - a CLI the operator runs; not invoked by the orchestrator.

Verified by reading the call sites on 2026-09-23, not by inference from names.

## The run path, in order

```
survey.py prepare      (CPU)  ingest -> validate -> GPS gates -> capture plan
survey.py reconstruct  (GPU gate --allow-gpu)
   keyframes   extract_keyframes.py        400 frames, sharpness filter, dedupe
   frames      survey_frames.py            select by plan, score, flatten, mask   [NEW]
   priors      survey_priors.py            GPS -> COLMAP pose_priors table
   colmap      run_colmap.py               pose_prior mapper, image_dir=frames_match,
                                           mask_path=masks                        [NEW]
   progressive survey_progressive.py       opt-in (--progressive): windowed submaps
                                           merged with model_merger + image_registrator,
                                           checkpoints after every window; diagnostic,
                                           never fatal                           [NEW]
   poses       parse_colmap.py
   undistort   COLMAP image_undistorter    reads frames_match                     [NEW]
   dense       COLMAP patch_match_stereo   GPU
   fusion      COLMAP stereo_fusion
   mesh        COLMAP poisson_mesher       CPU, produces faces -> OBJ             [NEW]
   georeferenced_export
               survey_georef  -> alignment
               survey_deliver -> enu/ + georeferenced/ + delivery_manifest.json   [NEW]
               survey_products -> evidence cloud + support summary
               survey_assess  -> completeness + throughput blocks                 [NEW]
survey.py evaluate     (CPU)  accuracy split -> criteria ledger -> evaluation.json
```

## Module by module

| Module | State | What it contributes | Artifact |
|---|---|---|---|
| `survey_workflow` | run path | the orchestrator above | `preparation.json`, `run.json`, `evaluation.json` |
| `survey_georef` | run path | WGS84 -> ENU, RANSAC Sim(3) trajectory fit | `georeference.json` |
| `survey_priors` | run path | GPS into bundle adjustment as per-camera priors | `pose_priors.jsonl` |
| `survey_inputs` | **run path (new)** | GPX / DJI `.srt` / Pilot CSV / contract CSV ingestion, conservative uncertainty, typed blockers | normalised `telemetry.csv`, `telemetry_source.json` |
| `survey_gnss` | **run path (new)** | telemetry quality, vertical-datum signal, clock-offset bounds, fix-quality weights, trajectory identifiability | `preparation.json:gates` |
| `survey_accuracy` | **run path (new)** | stratified fit/hold-out split, unaligned + SE(3) + Sim(3) tracks, verdict, GCP requirement | `evaluation.json:accuracy_split` |
| `survey_capture` | **run path (new)** | baseline-aware frame plan, frame scoring, illumination flattening, dynamic masks | `capture_plan.json`, `masks/*.png` |
| `survey_frames` | **new module** | the stage that applies all of the above to written frames | `frames.json`, `frames_match/` |
| `survey_selection` | **run path (new)** | arc-length keyframe spacing, path coverage, parallax | via `survey_capture` |
| `survey_frame_quality` | **run path (new)** | blur/compression weights, drop decision | via `survey_frames` |
| `survey_photometry` | **run path (new)** | illumination flattening for the matching copy only | via `survey_frames` |
| `survey_dynamics` | **run path (new)** | epipolar-inconsistency masks for moving objects | `masks/` |
| `survey_crs` | **new module** | ENU -> geodetic -> UTM, EPSG/PROJCS WKT, refusal outside the zone | `delivery_manifest.json:crs` |
| `survey_deliver` | **new module** | mesh reading, both product sets, the six-format ledger | `products/{enu,georeferenced}/` |
| `survey_export` | run path | every container that needs no invention | `export_manifest.json` |
| `survey_formats` | run path | byte-level writers and independent re-readers | (inside the above) |
| `survey_assess` | **new module** | completeness, throughput prediction, control requirement, accuracy split wrapper | `run.json:completeness/throughput` |
| `survey_products` | run path | per-point support, occlusion-checked evidence cloud | `evidence/` |
| `survey_evidence` | run path | track support and confidence | `evidence_points.ply` |
| `survey_visibility` | run path | depth-map occlusion check on the sparse cloud | `evidence_summary.json` |
| `survey_evaluation` | run path | the criteria ledger and the official 600 s / < 900 s gate | `evaluation.json` |
| `survey_streaming` | **run path (new)** | predicted stage timing, progressive window plan, first-usable-output; its windows are now executed by `survey_progressive` under `--progressive` | `run.json:throughput` |
| `survey_progressive` | **run path (opt-in `--progressive`, new)** | executes the window plan: one shared COLMAP database (no frame re-extracted), per-window submaps via `Mapper.image_list_path`, merged with `model_merger` + `image_registrator`; measured per-window seconds kept separate from the plan's predictions; a failed window is recorded with its reason and never endangers the monolithic run | `progressive/checkpoints.json`, `run.json:progressive` |
| `survey_observability` | **run path (new)** | region observability, baselines, trajectory verdict | `run.json:completeness` |
| `survey_occlusion` | **run path (new)** | measured / weak / unobserved layer classification | `run.json:completeness:layers` |
| `survey_measure` | **tested** | distance, segment, area, ground plane, height, volume, with uncertainty | `survey/measurements.json` (operator-authored) |
| `survey_measure` CLI | **manual** | nothing invokes it; the dashboard renders the file, it does not create it | - |

## Deliberately still unwired

- **`survey_measure` into the viewer.** Picking two points in the browser and getting a
  distance with an uncertainty budget is a UI task, and the instruction was to
  prioritise function over interface. The measurement maths is tested; the click path
  is not built.
- **Generative completion of occluded surfaces.** `survey_occlusion.inference_policy`
  publishes no inpainting entry point on purpose: a single pass cannot see a hidden
  face, and inventing one would present a guess as a measurement.
- **Progressive/streaming reconstruction as the default path.** `survey_progressive`
  now executes the `survey_streaming` windows behind `survey.py reconstruct --progressive`:
  shared feature extraction, per-window submaps, `model_merger` + `image_registrator`
  merges, and `progressive/checkpoints.json` after every window, with measured
  per-window seconds kept apart from the plan's predictions. What is *not* wired is
  any consumer of those checkpoints (the dense/mesh stages still run monolithically
  afterwards, and the dashboard does not yet render the accumulated submaps), and the
  executor's argv has been verified against the vendored COLMAP 4.1.1 binary strings
  and the 4.1.1 sources but **has not been executed against a real COLMAP run** -
  see the caveat below.
- **Dashboard launch of the survey run.** GPU execution stays behind
  `--allow-gpu` on the CLI by design; the server refuses it.

## What this ledger is not

It records *reach*, not *validity*. A module in the run path still needs the
acceptance protocol to have been run against real footage before any number it emits
can be quoted as measured.

## The limit of this test style, learned twice in one day

The workflow tests patch `subprocess.run`, which proves the plan COLMAP is given and
never that COLMAP accepts it. Two bugs survived 663 green tests and only appeared when
the stages actually executed:

- `mesh_command` passed the dense **directory** to `poisson_mesher`, which wants the
  fused PLY path: `does not match file extension .ply`.
- `write_masks` wrote masks at the reduced working resolution. COLMAP silently
  rejected **all 72 frames** of the rocks scene and exited 0; the frame-count guard in
  `run_colmap.py` is what turned that into an error instead of a bad reconstruction.

Both are fixed and both now have a regression test. Anything in this ledger marked
"run path" but never executed against the real external tool should be treated as
unverified in exactly this way.

`survey_progressive` is in that unverified-executed category, and reading the
COLMAP 4.1.1 sources already paid for itself once: `model_merger` **exits 0 even
when the merge fails** (it then writes `input_path2` unchanged), and
`image_registrator` exits 0 whatever it managed to register. The executor
therefore never trusts those exit codes - it reads the output models back through
`model_converter` and fails the window when registered images of either merge
parent are missing. Those guards are unit-tested against a fake COLMAP; the day a
real GPU-approved run executes them, they are the lines to watch.
