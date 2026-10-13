# scripts/

116 Python modules. Each pipeline step is one script here, run as its own process by
[`pipeline.py`](../pipeline.py), so any of them can also be run by hand against a scene's
`work/<scene>/` directory. The descriptions below are the first lines of each module's own
docstring; the docstring is where the reasoning lives.

## Pipeline steps (splat lane)

In the order `pipeline.py run` executes them.

| Step | Script | What it does |
|---|---|---|
| `keyframes` | [`extract_keyframes.py`](extract_keyframes.py) | One or more video clips -> sharpness-filtered keyframes. |
| `frames` | [`survey_frames.py`](survey_frames.py) | Score, flatten and mask the frames a survey run actually matches on. |
| `priors` | [`import_phone_poses.py`](import_phone_poses.py) | Phone AR pose logs -> per-keyframe pose priors for COLMAP. |
| `colmap` | [`run_colmap.py`](run_colmap.py) | V2 COLMAP runner: plan-driven, multi-clip aware, optional AR pose priors. |
| `poses` | [`parse_colmap.py`](parse_colmap.py) | Parse COLMAP text model + our keyframes manifest -> keyframes_poses.jsonl. |
| `dynamics` | [`dynamics_pass.py`](dynamics_pass.py) | Challenge D1: mask moving objects, and only re-match if the masks register more frames. |
| `train` | [`train_splat.py`](train_splat.py) | Turbocharged 3D Gaussian Splatting trainer for high quality & fast convergence. |
| `frame` | [`solve_frame.py`](solve_frame.py) | Solve the world frame (gravity + metric scale) from COLMAP cameras. |
| `depth` | [`depth_prior.py`](depth_prior.py) | MoGe-2 metric depth as an INDEPENDENT SECOND RULER (challenge D2b). |
| `export` | [`export_viewer_assets.py`](export_viewer_assets.py) | Export trained-splat assets for the browser viewer — one pass, final coords. |
| `sky` | [`strip_sky.py`](strip_sky.py) | Strip the airborne canopy out of the exported splat. |
| `clouds` | [`strip_clouds.py`](strip_clouds.py) | Cull the reconstructed cloud/fog fill out of the exported splat. |
| `colors` | [`export_ground_colors.py`](export_ground_colors.py) | Vertex-color the collision heightfield using the trained splat's own colors. |
| `collider` | [`build_collider.py`](build_collider.py) | Build the physics collider from scene.ply via @playcanvas/splat-transform. |
| `objects` | [`build_objects.py`](build_objects.py) | Turn the stuff standing on the floor into box colliders the engine can use. |
| `surface` | [`tune_collider.py`](tune_collider.py) | Choose the collider's ground surface by measuring which one the route walks. |
| `semantics` | [`label_semantics.py`](label_semantics.py) | Geometric + colour semantic labelling of a reconstructed point cloud. |
| `rooms` | [`detect_rooms.py`](detect_rooms.py) | Find the room in an interior scan: floor, ceiling, walls, and the metres between them. |
| `texture` | [`survey_texture.py`](survey_texture.py) | Bake the source video's frames into a UV texture for a reconstructed mesh (G1). |
| `gate` | [`check_world.py`](check_world.py) | Is the exported world actually right way up and standable? Pass/fail checks. |
| `audit` | [`scenario_audit.py`](scenario_audit.py) | E2: audit a finished scene against the capture preset it claims it was built as. |
| `evals` | [`render_evals_offline.py`](render_evals_offline.py) | Render all eval poses offline with gsplat (the proven training path): scene.ply + poses.json -> eval_renders/eval_XX.png (at each camera's own size). |
| `pairs` | [`make_pairs.py`](make_pairs.py) | Compose blind A/B comparison images for the critic. |
| `walktest` | [`drive_viewer.py`](drive_viewer.py) | Drive the walkable PlayCanvas viewer (viewer/pc.html) with Playwright. |

`reexport` re-runs `export_viewer_assets.py` over the culled model, and `nav` runs
[`tools/navbake/bake.mjs`](../tools/navbake/) under Node.

## World building

Geometry the steps above call into: ground, collision and the walk route.

| Script | What it does |
|---|---|
| [`build_heightfield.py`](build_heightfield.py) | Point cloud -> walkable ground heightfield (library + CLI). |
| [`clip_collider.py`](clip_collider.py) | Clip a collision mesh to the GROUND layer, dropping airborne crusts. |
| [`ground_mesh.py`](ground_mesh.py) | Turn the clipped voxel shell into a walkable ground mesh. |
| [`walk_path_from_glb.py`](walk_path_from_glb.py) | Derive the walk loop + spawn candidates from the COLLISION MESH itself. |
| [`plan_glb.py`](plan_glb.py) | Minimal binary glTF 2.0 writer for triangle meshes (no dependencies beyond numpy). |
| [`glb_bounds.py`](glb_bounds.py) | Dump vertex bounds / triangle counts of a GLB (debug helper). |

## Capture and diagnostics

Reading footage and pose logs, and explaining a scene that came out wrong.

| Script | What it does |
|---|---|
| [`capture_diagnostics.py`](capture_diagnostics.py) | Capture-style diagnostics: what did this clip actually do, and can SfM live with it? |
| [`analyze_take.py`](analyze_take.py) | A captured take -> was it scanned well, measured off the phone's own pose log. |
| [`check_coverage.py`](check_coverage.py) | Analyze 3D multi-view coverage, camera trajectory, and sparse tie points. |
| [`camera_intrinsics.py`](camera_intrinsics.py) | CPU-only COLMAP intrinsics conversion for OpenCV undistortion. |
| [`poses_lib.py`](poses_lib.py) | Shared phone-pose math + log IO for the AR-prior pipeline. |
| [`diag_scene.py`](diag_scene.py) | Diagnose splat / heightfield / collider agreement for a viewer_assets dir. |
| [`diag_hf_steps.py`](diag_hf_steps.py) | Where do the heightfield's big steps come from? |
| [`check_eval_cams.py`](check_eval_cams.py) | Verify eval camera poses: center, forward, up for each pose in poses.json. |
| [`check_glb_coverage.py`](check_glb_coverage.py) | Check collision GLB triangle coverage over the walk loop (XZ rasterization). |
| [`shoot_viewer.py`](shoot_viewer.py) | Screenshot the walkable viewer from scripted camera poses (headless). |
| [`render_eyelevel.py`](render_eyelevel.py) | Render the splat from a walking-eye-height camera at the spawn, offline. |
| [`label_pairs.py`](label_pairs.py) | Unblind results/blinded/*.jpg using every results/pair_key_&lt;tag>.json -> labeled side-by-sides in results/side_by_side/. |
| [`learned_semantics.py`](learned_semantics.py) | Learned open-vocabulary segmentation of a scene's keyframes (challenge D6). |
| [`benchmark_hardware.py`](benchmark_hardware.py) | Comprehensive 3D Gaussian Splatting Hardware Benchmark & Throughput Profiler. |

## Survey lane

Georeferencing, accuracy evaluation and deliverable formats, driven by [`survey.py`](../survey.py).
Most of these are CPU-only and depend on NumPy and the standard library alone.

| Script | What it does |
|---|---|
| [`survey_accuracy.py`](survey_accuracy.py) | Metric accuracy without extensive ground control: the protocol, made executable. |
| [`survey_assess.py`](survey_assess.py) | What the run can say about itself without being handed surveyed truth. |
| [`survey_capture.py`](survey_capture.py) | Decide which frames a single pass should contribute, and what to do with them. |
| [`survey_change.py`](survey_change.py) | Two-epoch change detection on measured clouds (M3; CON-03, BOR-03, DIS-08, DIS-03). |
| [`survey_checkpoints.py`](survey_checkpoints.py) | Surveyed checkpoints in, the evaluator's exact `checkpoints.json` out (challenge F2). |
| [`survey_coords.py`](survey_coords.py) | Coordinates people read aloud: MGRS, DMS and a CE90 / LE90 error figure (M8). |
| [`survey_corridor.py`](survey_corridor.py) | Corridor products for border and linear-asset flights (BOR-02, BOR-04). |
| [`survey_crs.py`](survey_crs.py) | Put a measured cloud into a CRS it is allowed to name. |
| [`survey_damage.py`](survey_damage.py) | Building damage grade from a single post-event cloud (DIS-02). Heuristic, and says so. |
| [`survey_deliver.py`](survey_deliver.py) | Turn one reconstruction run into the deliverables the brief names, in a real CRS. |
| [`survey_design.py`](survey_design.py) | Design surfaces and cut/fill against them (CON-02). |
| [`survey_detections.py`](survey_detections.py) | People and vehicles as a georeferenced layer, never as geometry (M10, DIS-06, MIL-03). |
| [`survey_dynamics.py`](survey_dynamics.py) | Dynamic-object evidence: what moved, as opposed to what the camera did. |
| [`survey_evaluation.py`](survey_evaluation.py) | CPU-only evaluation of caller-supplied metric-coordinate observations. |
| [`survey_evidence.py`](survey_evidence.py) | Per-point multi-view evidence: triangulation support and reprojection error. |
| [`survey_export.py`](survey_export.py) | Turn a measured cloud into the deliverable formats - without inventing anything. |
| [`survey_firstmap.py`](survey_firstmap.py) | First map while the run continues (DIS-01 / GAPS H1). |
| [`survey_formats.py`](survey_formats.py) | Writers for the SIH26158 output format list, in stdlib plus NumPy only. |
| [`survey_frame_quality.py`](survey_frame_quality.py) | Challenge (ii): motion blur and video compression artifacts, no reference. |
| [`survey_geoid.py`](survey_geoid.py) | Ellipsoidal <-> mean-sea-level heights through the EGM96 geoid (challenge F3). |
| [`survey_georef.py`](survey_georef.py) | Bounded camera-center GPS georeferencing, using only NumPy and stdlib. |
| [`survey_gnss.py`](survey_gnss.py) | Honest quality, datum and observability checks for flight GNSS telemetry. |
| [`survey_gravity.py`](survey_gravity.py) | Straight-track georeferencing: fix the one rotation camera positions cannot (M11). |
| [`survey_inputs.py`](survey_inputs.py) | Normalise real drone telemetry into the survey_georef camera-position contract. |
| [`survey_klv.py`](survey_klv.py) | MISB ST 0601 KLV telemetry from STANAG 4609 video (MIL-16, BOR-07). |
| [`survey_m3c2.py`](survey_m3c2.py) | M3C2 point-cloud distance with level of detection (Lague, Brodu & Leroux 2013). |
| [`survey_measure.py`](survey_measure.py) | Measurement and analysis on metric ENU point clouds, using NumPy and stdlib. |
| [`survey_observability.py`](survey_observability.py) | Measure what a single flight path can actually triangulate - no promises. |
| [`survey_occlusion.py`](survey_occlusion.py) | What a single drone pass cannot see, kept separate from what it can. |
| [`survey_ortho.py`](survey_ortho.py) | True orthomosaic from the frames that built the model (M1). |
| [`survey_photometry.py`](survey_photometry.py) | Challenge (iii): variable illumination and shadows, CPU, no reference frame. |
| [`survey_priors.py`](survey_priors.py) | Turn survey GPS into COLMAP pose priors so positions enter bundle adjustment. |
| [`survey_products.py`](survey_products.py) | Georeferenced evidence products: a cloud plus per-point support reporting. |
| [`survey_progressive.py`](survey_progressive.py) | Progressive reconstruction executor: the near-real-time story, actually executed. |
| [`survey_response.py`](survey_response.py) | Disaster access and flooding on the measured surface (DIS-04, DIS-05). |
| [`survey_selection.py`](survey_selection.py) | Baseline-aware keyframe selection from camera positions (CPU, geometry only). |
| [`survey_streaming.py`](survey_streaming.py) | Stage-time prediction: can this pass finish before the deadline, honestly? |
| [`survey_terrain.py`](survey_terrain.py) | Bare-earth terrain from a measured cloud: DTM, nDSM and a ground class (M2). |
| [`survey_vertical.py`](survey_vertical.py) | Takeoff-anchored heights: GNSS sets the level once, the barometer draws the shape (M12). |
| [`survey_visibility.py`](survey_visibility.py) | Occlusion-aware view support: does a surface point survive depth consistency? |
| [`survey_workflow.py`](survey_workflow.py) | Survey lane orchestration: the `prepare`, `reconstruct`, `align`, `evaluate` and `status` commands behind `survey.py`. |

## Applications

Analyses for the eight application profiles, on top of a finished scan.

| Script | What it does |
|---|---|
| [`applications.py`](applications.py) | Mission profiles: one registry entry per SIH26158 application (playbook §4). |
| [`mission.py`](mission.py) | Mission planning features (Phase 3): tactical symbols, routes and phase lines. |
| [`mission_analysis.py`](mission_analysis.py) | Terrain analysis for mission planning: line of sight, viewsheds, route exposure, a covered route suggestion and helicopter landing zones. |
| [`mission_session.py`](mission_session.py) | Multi-user rehearsal sessions with an instructor (RH-9), held in memory on the local server. |
| [`aar_report.py`](aar_report.py) | After-action review as a PDF (RH-6 / MIL-12): what the on-screen replay shows, on paper. |
| [`inspection.py`](inspection.py) | Infrastructure inspection on a scan (INF-01, INF-02, INF-06, INF-08; ARC-03). |
| [`inspect_geometry.py`](inspect_geometry.py) | Measured geometry for inspection and archaeology on the scan's points (viewer frame, y up). |
| [`facades.py`](facades.py) | Per-building facade completeness (URB-17): which walls the flight actually saw. |
| [`urban_analysis.py`](urban_analysis.py) | Urban layers measured from the scan: roof planes, solar potential, canopy, road surface. |
| [`plan_exports.py`](plan_exports.py) | Proposal exchange formats: CityJSON, DXF and 3D Tiles out; cadastral parcels in. |
| [`plan_shadow.py`](plan_shadow.py) | Shadow study for the planning editor: where the sun is, and what a scheme shades. |
| [`twin.py`](twin.py) | Digital twin of a scanned site (TWN-02, TWN-04, TWN-05, TWN-07; CON-04). |
| [`ops_scene.py`](ops_scene.py) | A workspace scene seen by the operations analyses (Phase 4). |
| [`ops_report.py`](ops_report.py) | Field packs and one-page reports for the operations applications (DIS-09, CON-08). |
| [`provenance_record.py`](provenance_record.py) | Archive record for a scan (ARC-06): who, when, what device, which CRS, which processing, which files. |

## Workspace and server API

What the Ground Control Station talks to, through `_serve.py`.

| Script | What it does |
|---|---|
| [`workspace_api.py`](workspace_api.py) | Disk-backed workspace API. Reads never probe footage or start reconstruction. |
| [`workspace_bundle.py`](workspace_bundle.py) | One-click project bundle (G3): model, measurements, reports and CRS in one ZIP. |
| [`workspace_inspect_api.py`](workspace_inspect_api.py) | HTTP handlers for Phase 5: inspection, archaeology and the digital twin. |
| [`workspace_measure.py`](workspace_measure.py) | Adapter that lets the workspace measure with the real survey engine. |
| [`workspace_mission_api.py`](workspace_mission_api.py) | HTTP handlers for mission planning and rehearsal (`/api/workspace/mission/*`). |
| [`workspace_ops_api.py`](workspace_ops_api.py) | HTTP handlers for the Operations tab (`/api/workspace/ops/*`, Phase 4). |
| [`workspace_place.py`](workspace_place.py) | Place furniture on a scanned floor, without claiming unmeasured clearances. |
| [`workspace_plan_api.py`](workspace_plan_api.py) | HTTP handlers for the planning editor (proposal layer) and click coordinates. |
| [`workspace_proposals.py`](workspace_proposals.py) | The proposal layer: planned roads, buildings, zones, objects and demolitions (Phase 2). |

## Setup, shared code and fixtures

| Script | What it does |
|---|---|
| [`bootstrap.py`](bootstrap.py) | One command from a clean clone to a runnable pipeline. |
| [`robust.py`](robust.py) | Shared hardening for every pipeline step: subprocess execution, resource budgets, and failure classification. |
| [`scene_frames.py`](scene_frames.py) | One transform chain per scene: viewer <-> COLMAP <-> ENU <-> WGS84 / UTM / MGRS. |
| [`make_test_scene.py`](make_test_scene.py) | Synthetic INRIA-format splat scene to test the viewer end-to-end (spark loading, eval camera math, heightfield collision, autopilot) without a … |
| [`make_flat_scene.py`](make_flat_scene.py) | Textured FLAT ground for the demo video: planning (Plan) and placing (Place). |
| [`make_plan_scene.py`](make_plan_scene.py) | Synthetic FLAT scene for exercising the planning editor end to end. |
| [`make_ops_scene.py`](make_ops_scene.py) | Two synthetic epochs of one site for exercising the Operations tab end to end. |
