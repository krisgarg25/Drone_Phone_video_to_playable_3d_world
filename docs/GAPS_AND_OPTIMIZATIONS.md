# Gaps, Optimizations & Feature Roadmap

**Project:** Single-pass drone/handheld video → georeferenced, walkable 3D model
**Status date:** 2026-09-26 (tracks E, F and G delivered; each section says what is measured, what is only tested on synthetic data, and what is still open)
**Legend:** `wire` = capability already exists in code, just not connected · `build` = new work · `S/M/L` = effort · `⚠dep` = needs a new package or model weights

This document lists what exists today (verified against the codebase), what is missing, and what to build, organised by capability. It is deliberately honest: several requested features **already exist but are orphaned** (test-only or CLI-only), so the cheapest wins are wiring, not new algorithms.

---

## A. Measurements & analysis

The real engine `scripts/survey_measure.py` (1688 lines) implements distance, polygon/true/projected area, height-above-ground, volume & cut/fill, slope/aspect, cloud snapping, an RSS uncertainty budget, and GeoJSON/CSV export. The product reaches it through `scripts/workspace_measure.py`; picking happens on the collision surface, and the sparse cloud decides whether a reading is *supported*.

| ID | Item | Type | Effort | Status |
|----|------|------|--------|--------|
| A1 | Wire the real engine to HTTP + viewer (snapping + uncertainty) | wire | M | **verified** — polyline distance (≥2 points, per-segment support), snapping, uncertainty budget |
| A2 | Volume / cut-fill tool (stockpiles, spoil, earthworks) | wire | M | **partial** — footprint integrated above fitted ground on save; needs a point cloud and is never inferred from a flat outline |
| A3 | Per-class measurement (labelled building/road → footprint, height, façade area) | build | M | **partial** — class counts + grid-occupancy area proxy; slope/aspect and façade area not exposed |
| A4 | Persist + export measurement sets as GeoJSON/CSV with CRS + uncertainty | wire | S | **verified** — local-frame GeoJSON/CSV/JSON carrying uncertainty and validity |
| A5 | Fix trust: walktest distance-inflation | build | S | **verified** — grounded-only walk distance |
| A6 | Read the number while picking, not only after saving | build | M | **verified** — live per-segment + total labels from picked geometry; unsupported saved records show a marked estimate instead of nothing |

**Known limits:** the collision surface is an estimated proxy, not the visible splats (splats are not raycastable). Values inherit scene scale, labelled metric only on the AR pose-prior path. Independent survey accuracy is not established.

## B. Furniture / interior placement

`scripts/build_objects.py` only **detects** existing furniture boxes. Placement runs through `scripts/workspace_place.py` + `placements.json` + an opaque recognisable-component mesh in the viewer, edited directly in the live 3D view (`/projects/<scene>/place`): click a piece to grab it, drag it across the scanned floor, and use the dock to slide, raise/lower, rotate and resize each axis. There is deliberately no 2D floor plan.

| ID | Item | Type | Effort | Status |
|----|------|------|--------|--------|
| B1 | Drop a real-scale item, snap to floor, orient, move, delete | build | L | **verified** — drop, drag-move, rotate, keyboard nudge, delete; each write re-fit-checked and persisted |
| B2 | Whole-house multi-placement + saved layouts | build | L | **partial** — many items persist per scene and export as GeoJSON; no named layout variants or restore-from-file |
| B3 | Furniture library (primitives + glTF import) | build | M | **verified** — 12 nominal-dimension items drawn as component geometry (legs, tops, cushions, shelves, duvet), plus glTF/GLB import: a self-contained model's true size is read from its POSITION accessor min/max through the node graph, it renders as real geometry filling the same box its collider uses (root-node rotation and scale included), moves/rotates/resizes/persists like a primitive, and exports with its provenance. Bounds that cannot be read are refused, never guessed — and so is a file the shipped viewer could not draw |
| B4 | Collision for placed items so walk/game respects them | build | M | **verified** — compound static body with conservative box colliders |
| B5 | Room detection (walls/floor/ceiling) → per-room area + clearance | build | L | **not built** — floor *support* comes from the coverage grid; ceiling height is explicitly unknown and coverage-edge distance is not wall clearance |
| B6 | Reliable metric scale for interiors (see E1) | build | M | **unchanged** — placement inherits scene scale; `room_w_jsonl` is metric via AR pose-prior |

**Known limits:** furniture–furniture overlap, doorway clearance and stability on sloped floors are not checked. Catalogue sizes are nominal, not branded models. An imported model must be one self-contained file — a `.gltf` that names an external buffer is refused, because a model must never choose a path the server opens — and its collider is the conservative box around its real bounds, not a mesh. Import also refuses what the viewer demonstrably cannot draw: a `.glb` whose chunks are not in the order glTF mandates (JSON first, BIN second), and geometry behind `KHR_draco_mesh_compression` or `EXT_meshopt_compression`, whose decoders this build does not carry; accepting either would leave a coloured box where the customer's armchair should be. Imported models render in the fit colour, not their own textures. The metre an imported model states is its own; in a scene whose scale is anchored rather than measured, that is still the honest size of the object and only a provisional size *in the scene*.

## C. Gaming / first-person walkthrough

Ammo.js physics, a character controller, an A* route and a `combat.js` + triangle-graph `nav.js` NavMesh exist. `/projects/<scene>/walk` is now the arena: one start surface, then the game edge to edge with bots — no tour panel, no developer chrome over the play area.

| ID | Item | Type | Effort | Status |
|----|------|------|--------|--------|
| C1 | Bake navmesh inside the pipeline (auto `nav.json`) | wire | M | **partial** — pipeline bakes `pc/nav.json` after the final collider; quality is reported, not gated |
| C2 | First-person play as a first-class option | wire | S | **verified** — arena entry, WASD/jump, bots engaging, exit back to the workspace |
| C3 | Teleport/waypoint + minimap | build | M | **not built** — the tour map and viewpoint jumps were removed as clutter; no minimap |
| C4 | Game export: standalone playable package | build | L | **not built** |
| C5 | Bot session surfaced as a feature | wire | M | **verified** — bot count, live session, navigation-quality warning; standalone developer chrome hidden |

**Known limits:** room presets ship `character_height: 0.15` (hamster scale), so a human-scale interior walkthrough is not yet credible. Navigation can be sparse or fragmented; the panel reports the triangle/region counts instead of hiding them.

## D. Reconstruction quality (the six key challenges)

| ID | Challenge | Item | Type | Effort |
|----|-----------|------|------|--------|
| D1 | Dynamic objects | `survey_dynamics.py` (background model, motion mask, epipolar inconsistency) is implemented but **never executed** in the main pipeline — wire it | wire | M |
| D2 | Textureless / few-view / blur | Monocular metric-depth prior (Depth-Anything/Metric3D) to regularise COLMAP | build | L ⚠dep |
| D3 | Variable illumination/shadows | Photometric normalisation pass (`survey_photometry` exists — check wiring) — **delivered**, see the D3 note below | wire | M |
| D4 | GPS inaccuracy / sensor noise | GNSS uncertainty + clock-offset handling exists in survey lane — surface + default-on | wire | S |
| D5 | Limited viewing angles | Honest coverage/occlusion map (`survey_visibility`/`survey_occlusion`) as a viewer layer | wire | S |
| D6 | (quality) | Learned segmentation to upgrade heuristic semantics → real classes | build | M ⚠dep |

**D-block status (in progress, 2026-09-25).**

- **D4 — delivered (read-path only).** `survey_gnss` always ran and wrote `gates` into
  `survey/preparation.json`, but `scene_status` never read that key, so `/api/survey` could not
  reach it. Now projected through `survey_workflow._gnss_gates` → `workspace_api.survey_summary`
  → `lib/api.ts` → a `GnssReference` block in the existing COORDINATE/SPATIAL REFERENCE sections.
  `accuracy: unverified` and `georeference: local` are unchanged — bounding uncertainty is not
  claiming accuracy — and a scene with no telemetry renders "Not supplied" rather than zeros.
  **Structural limitation found:** `worst_case_along_track_error_m` can only ever be `0.0` or
  `null` at prepare time. `survey_assess.control_requirement:122-124` passes camera times
  `[0, video_duration_s]` while `survey_georef.normalize_telemetry:143-144` clamps every fix
  into that same span, so the bound is feasible only when the log spans the clip exactly — and
  then its width is zero. Getting a real number needs per-frame camera times, which only exist
  after reconstruction (`keyframes_poses.jsonl`), so this is a follow-up build, not a bug to
  patch in `survey_assess`. The UI labels the `0 s` window vacuous in the meantime.
- **D5 — delivered, and the diagnosis was partly wrong.** The layer itself already existed
  (`coverage` is in `workspace_core.js:4` and `project-workspace.tsx:25`), so this was a
  correctness fix: `check_coverage.py` reported **0% observed on a scene with 100% COLMAP
  registration** and still emitted "re-fly the perimeter" advice. Measured: rocks
  `covered_pct` 0.0 → **59.1**, in-reach voxel centres 0.07% → 93.7%; auditorium 31.8 → 32.0,
  correctly a near-no-op because it was never broken.
  Two premises were refuted by measurement: the frame `check_coverage.py` writes is **already
  metres** (`pos = (colmap @ Rg.T) * scale_m_per_unit`, `:77`), so `max_range = 4.5` was a real
  4.5-metre constant, not a unitless one; and the grid bounds were a 2/98 percentile **unioned**
  with the camera hull, not raw min/max. Actual trigger: an absolute `dist < 5.0` cloud filter
  survived on 2 of 18,034 points, tripping a `> 100 else pts_world` fallback that made the grid
  inherit the whole outlier cloud — 684x the camera hull volume. Reach/bounds/cell-size are now
  scene-derived (`region_max_range_units` → `camera_agl_m x 4` → `32 x camera step`, with the
  basis recorded), and a refused grid yields `covered_pct: null` + `status: "not_measurable"` +
  reason, so "nothing observed" and "not measurable" are finally distinguishable in the UI.
  `survey_occlusion.hidden_regions` could **not** be used: `:324-330` raises unless
  `depth_views` is non-empty, no COLMAP depth maps exist anywhere, and it has no production
  caller at all. Unobserved volume came from the corrected lattice instead
  (rocks: 32,769.6 m³ of 83,133.5 m³ gridded, plus exposed unobserved surface 6,371.5 m²).
  Follow-ups still open: propagate `not_measurable` into the step marker (`--strict` exists),
  and `build_steps` has **no coverage step** — a real run's grid is written by
  `export_viewer_assets` instead, which should be reconciled.
- **D3, D2 in progress.** D2's stated framing ("monocular depth prior to regularise COLMAP") is
  not implementable as written: COLMAP 4.1's mapper takes no per-pixel depth input, its only
  prior channel being `pose_priors` (camera-centre + scalar std). The regularisation belongs in
  `train_splat.py`, where gsplat's own rendered depth makes the textureless/few-view win
  downloadable-model-free; a learned metric model addresses only scale, which is the separate
  E1/B6 problem.


- **D1 — delivered.** `scripts/dynamics_pass.py` is pipeline step `dynamics` (position 04,
  between `poses` and `train`), so `survey_dynamics` now executes in the main run path.
  Masking needs camera centres, which only exist after a reconstruction, so it is a gated
  second matching pass: it runs only when registration is under the same `rescue_below`
  (0.6) bar the COLMAP rescue ladder uses, or under `--dynamics force`. It re-matches,
  compares registered counts, and **reverts** unless the masked model is strictly better,
  so a tie keeps the unmasked model. Verified on `rocks`: 72/72 registered, 72 masks, mean
  1.39% / max 5.4% of pixels masked, re-match tied at 72/72 → reverted, model intact;
  `auto` mode skips healthy captures in 0.26s and writes the reason. `work/<scene>/dynamics.json`
  records the decision either way, including `skipped`. The second pass reaches COLMAP through
  `run_colmap --set mask_path=masks --set image_dir=frames_match`; `mask_path` is still not in
  the `pipeline.py` plan whitelist, so the **first** pass is unmasked by construction.
  **Coupling found while wiring D3:** that `--set image_dir=frames_match` is unconditional, so
  the mask pass re-matches on the flattened copy even when D3 decided not to flatten — on a
  scene with no `frames_match` at all the second pass exits at run_colmap's missing-directory
  check (and restores pass 1, so nothing is lost), and where one exists the two passes are
  compared across two photometric conventions. `dynamics_pass` should read the plan's own
  `image_dir` instead of pinning it.
  Depth range for the static-parallax veto is taken from the reconstructed cloud in the
  model's own units — the veto compares a baseline against a depth, so the ratio is
  unaffected by how wrong the scene scale is.

- **D3 — delivered and verified on real footage.** `survey_photometry` runs in the main lane.
  `capture_diagnostics.probe_video` now measures the light on the frames it already decoded, and
  `pipeline.py` turns that into the `frames` step (position 02, between `keyframes` and `colmap`),
  which runs `survey_frames.py --flatten --keep-all` and points the COLMAP plan's `image_dir` at
  `frames_match` — the one key whose absence had kept the whole capability dead. The trigger is
  two ratios, not a brightness: the shadow-suspect AREA (`shadow_suspects`, a fraction of the
  frame measured against that frame's own lit level, so a whole-frame gain cannot move it) must
  pass 25% of the frame on the worst decile of sampled frames AND must change by 12% of the frame
  across the clip — a permanently dark surface is albedo, and flattening it only dulls the
  matching copy. `--photometric auto|on|off` overrides.

  Verified 2026-09-25 on `videos/rocks.mp4` (72 keyframes), `.venv`, COLMAP 4.1:
  * probe, run fresh: 42 sampled frames, shadow area median 7.4% / p90 **11.2%**, spread across
    the clip **6.0%**, whole-frame level drift **1.13x** → both bars missed → `auto` says
    **do not normalise**, and `rocks` stays on `frames_train`.
  * forced on: 21 steps, `frames` ran (12 s), COLMAP logged
    `feature_extractor … --image_path …\work\rocks\frames_match` and `mapper …` likewise, and
    registered **72/72 (100%)** — no regression against the `frames_train` run, which is also
    72/72 (counted independently in `colmap/sparse/txt/images.txt`, not just from the log).
    The copy really is flatter: low-frequency field contrast 0.76 → 0.16 on frame 00000
    (0.57 → 0.14, 1.04 → 0.17 on two others), same 640x360 pixel size.
  * **`frames_train` byte-identical** across the whole A/B: `sha256sum` over all 72 files before
    and after both runs is the same list, as is `frames_full`'s. `survey_frames` now records the
    same fact itself — `colour_copy.digest_before/after` in `frames_match.json` — and exits
    non-zero if the copy the trainer colourises from (`train_splat.py:134`) ever moves mid-pass,
    so the receipt is in every run's directory rather than only in a test.
  * genuine no-op: a pristine copy scene (`work/rocks-d3`) with no `frames_match` on disk, run
    with no override, logged 20 steps with **no `frames` step at all**, `--image_path …frames_train`,
    72/72 registered, and **never created** `frames_match/`.
  * branch executed on light that varies: `scratch/make_litvar_clip.py` takes the same real
    `rocks` frames and sweeps a 0.45x soft shadow band across them (content, motion, resolution
    and exposure untouched). Probed fresh: p90 **32.6%**, spread **26.9%** → `auto` picks the
    pass, `frames` runs, COLMAP reads `frames_match`, **57/57** registered, `frames_train` again
    byte-identical (57 files). `--photometric off` on the same scene then prints
    `-> off (operator override)`, drops the step, and COLMAP reads `frames_train` (57/57), while
    the `frames_match` left by the previous run keeps its old mtimes: the override defeats the
    measurement, it does not merely re-label it.
  * staleness: the decision is taken in `build_config`, before `build_steps`, and `image_dir` is
    in the hashed plan. `pipeline.py status rocks --photometric on` reports `colmap done`;
    the same scene with `--photometric off` reports `colmap stale (command changed)`, because the
    digest is in the step's own argv. A run cannot skip normalisation and still look current.
  What this does **not** claim: `photometric_consistency` is useless as a trigger on moving
  footage (0.326 raw → 0.310 flattened on `rocks`: it scores the scene change, and flattening
  damps it), and a whole-frame exposure drift — what auto-exposure actually does — is exactly
  what `flatten` provably cannot remove, so the probe reports it as a shooting warning instead of
  pretending to fix it. `image_dir` now reaches COLMAP through the plan whitelist; `mask_path`
  reaches it only through `dynamics_pass`'s `--set`.


## E. Scenario-specific optimizations

| ID | Item | Type | Effort | Status |
|----|------|------|--------|--------|
| E0 | Every run records the scenario it chose, why, and who set each parameter | build | S | **verified** — `work/<scene>/scenario.json`, served as `project.scenario` |
| E1 | Interior human scale (was `character_height 0.15` diorama), re-validate walk | build | M | **verified** — a 1.75 m body now routes a real loop in both real flats |
| E2a | Preset audit: the knobs every preset declared and none reached | wire | M | **verified** — SIFT detector settings now reach COLMAP, A/B measured |
| E2b | Auto-detect the capture scenario from measurement, or say it cannot | build | M | **verified (frames) / tested (GPS)** — handheld/aerial separated on 5 real clips; with a GPS log the flight path now names orbit / mapping grid / corridor (synthetic tracks only) |
| E5 | Drone scale from the mandatory GPS log instead of `speed × duration` | build | M | **tested** — ruler D fits the camera track to GPS; recovers a known scale exactly on synthetic flights, refuses a straight line; no real telemetry exists on this machine |
| E3 | Small-object / close-up preset | wire | S | **not built** — the knobs are declared, but no close-up footage exists to test them |
| E4 | Large aerial: coverage-gate robustness (`temple` was `partial`) | wire | M | **verified** — `temple` went `failed` → `warnings`, with the real defect named |
| E-UI | Surface the gate and the scenario to an operator | build | M | **verified** — Quality tab, per-check numbers, run-dialog guidance |

### What was actually wrong, and the numbers that say so

**The presets were a form of decoration.** All eight declared `sift_peak_threshold` and
`sift_edge_threshold`; `run_colmap.py` reads both keys off the plan; and the plan whitelist in
`build_steps` did not contain them, so `plan.get(key, 0.002)` returned the fallback for every
capture style. `rocks` ran a drone mission with the blank-wall-room detector. **Measured A/B on
the real clip, two copies, one at a time:**

| arm | peak / edge | features per image | pairs with geometry | registered | colmap |
|---|---|---|---|---|---|
| what every preset always got | 0.002 / 16 | 3819 | 2113 | 72/72 | 54 s |
| what the `drone` preset asks for | 0.004 / 12 | 3338 (−12.6%) | 2115 | 72/72 | 48 s |

Same registration, 12.6% fewer features, ~11% faster — and now the settings differ by scenario
at all. `tests/test_scenario.py` pins both halves: the keys land in `plan.json`, and the plan
digest moves with them, so a run cannot skip the change and still look current.

**`character_height: 0.15` was the wrong fix to a real problem.** The room preset shrank the
person because the walk router demanded `0.9 m` of lateral clearance for a 1.75 m body — **2.65
capsule radii**, two shoulder-widths of empty air per side. In a scanned room, where only ~13%
of the collider grid carries measured floor, that demand leaves almost nowhere to stand. The
reproductions, all on `room_w_jsonl` (a real flat, metric via the AR path, its own walls
measuring 2.69 m tall):

| body | clearance rule | routed floor | walk loop | waypoints |
|---|---|---|---|---|
| 0.15 m hamster (shipped) | 0.9 × height | 8.3 m² | 7.6 m | 8 |
| 1.75 m human, old rule | 0.90 m | 1.07 m² | — | — |
| **1.75 m human, new rule** | **1.35 × capsule radius = 0.46 m** | **4.2 m²** | **4.4 m** | **10** |

`roomscan` went the same way: 6.4 m² → **8.1 m²** at human scale. The clearance is now a
multiple of the body (`CLEARANCE_BODY_MULT`), so it needs no per-video tuning and no shrunken
person. Two honest caveats: (1) 4.2 m² is less floor than the hamster diorama walked, because a
person genuinely does not fit in as much of a scanned room as a 15 cm figure does — the old
number was not a better room, it was a smaller human; (2) `roomscan`'s standing quality warning
("collider sits on the measured heightfield", median +0.20 m) disappeared at 1.75 m because
that tolerance is `max(character_height, 2·cell)`. **That warning closing is a knob moving, not
the scene improving**, and it should not be counted as a gain.

**One scene, three different bodies.** `nav_params()` returned a drone-specific override and
nothing else, so `bake.mjs` used its own defaults: `--height 1.7 --radius 0.4 --climb 0.5` in a
scene whose `collision.json` said 0.15 m and whose router dilated by 0.34 m. The nav bake is
now handed the same height, capsule radius and step height the preset declared, and
`scenario_audit` flags any scene where they disagree.

**Two export knobs silently evaporated.** The `reexport` step re-ran
`export_viewer_assets.py --from-scene` forwarding *nothing*, and that path rewrites
`collision.json` from scratch. So a `room` run applied its character height, camera ground
height, grid resolution, opacity prune and backdrop decision at `export`, then replaced every
one of them with a script default two steps later — before the gate, the router, the viewer and
the walk test read them. `export_argv()` is now one helper used by both passes.

**The gate asked a cell-count question of a metric world.** "Spawn on supported ground" wanted
5×5 cells: **35 cm** at the room preset's 0.035 m resolution (too lenient to notice a missing
floor tile) and **20 m** at `temple`'s 4.12 m (a demand no aerial capture meets — which is why
`temple` hard-failed). The window is sized in metres now (`FOOT_SUPPORT_M = 0.5`), and where the
grid cannot express that length the check returns **n/a plus a named defect** rather than a
verdict the resolution could not support. `temple`: `failed` → `warnings`, with "grid resolves
foot-level support" as the stated reason and the cell measured at 3.49 m. `rocks` correspondingly
moved from a false `pass` to a `warnings`, which is the honest reading of a 2.18 m walk grid.

**Headroom is now asked for and honestly refused.** No array in a viewer asset set encodes what
is above the floor — `heights.f32` is a per-column *low* surface — so the gate reads `rooms.json`
and returns one of three answers: a measured ceiling (the *lowest* of several rooms governs), a
wall-top bound ("2.69 m, which bounds the room from below but is not a headroom measurement"), or
not captured. Both interior flats land in the middle state. `workspace_place`'s long-dead
`fits_height` / `ceiling_height_m` fields still return `None` and are now explained on screen
instead of silently missing.

### E2b: the auto classifier, and what the footage refuted

The old classifier compared rotation against translation. An essential matrix **normalises its
translation**, so a drone at 12 m/s and a person at 0.3 m/s give the same number — and every
clip on this machine classified as `orbit_mixed`, making `room` or `drone` the only reachable
answers out of eight presets. Two signals that *do* separate were added to the existing probe
(no extra decode pass, computed from frames already in memory), plus one from the pose logs:

| clip | truth | pose-log speed | frame bob (px) | sky p90 | horizon | decided by |
|---|---|---|---|---|---|---|
| `rocks` | drone orbit | — | **0.008** | 0.047 | 2.67 | mount |
| `temple` | drone | — | **0.010** | 0.233 | 0.91 | mount |
| `room_w_jsonl` | hand, room | **0.31 m/s**, 43 m, straightness 0.03 | 5.86 | 0.386 | 2.85 | pose log |
| `roomscan` | hand, room | 0.26 m/s, 39.6 m, 0.02 | 3.94 | 0.315 | 2.11 | pose log |
| `test1` | hand, room | 0.16 m/s, 15.1 m, 0.10 | 1.44 | 0.427 | 2.59 | pose log |

Five for five, and the margin is two orders of magnitude: 40 consecutive frames of
phase-correlated vertical residual is 0.008–0.010 px on the aerial clips and 1.44–5.86 px on the
hand-held ones. **Sky area and horizon strength were tried and refuted** — a white painted
ceiling reads 38.6% "sky", and a room's wall/ceiling junction scores a *stronger* horizon than
an oblique drone pass — so they are recorded and never consulted, and `pick_preset`'s docstring
says so. Altitude-from-horizon-curvature was rejected on the arithmetic (≈0.18° at 30 m).
Container telemetry was rejected on evidence: every clip in `videos/` has been through ffmpeg,
and the only `udta` box that survives carries `Lavf62.3.100`.

What auto still cannot do, stated on screen rather than hidden: tell an aerial orbit from a
nadir mapping map from from a facade circle, or size a room before it is reconstructed. `drone`
is the aerial default and the record says which candidates it cannot separate.
`corridor`, `indoor_large`, `object`, `outdoor_building`, `drone_mapping` and `sky_heavy` have
**never been exercised by any footage on this machine** — their knob values are assertions, and
`scenario_audit --all` prints that list every time it runs.

### E2b ledger: which scenarios pass their own gate

`python scripts/scenario_audit.py --all`, CPU-only, from artefacts already on disk
(`camera_agl_m` needs the gate re-run once per scene to record its metric):

| scene | built as | gate | AGL m | span m | cell m | reg % | findings |
|---|---|---|---|---|---|---|---|
| `room_w_jsonl` | room | pass | 1.03 | 6.8 | 0.035 | 98.6 | none |
| `roomscan` | room | pass | 1.51 | 7.9 | 0.036 | 84.3 | none |
| `test1` | room | pass | 1.42 | 5.7 | 0.098 | 69.1 | registration |
| `rocks` | drone | warnings | 12.12 | 71.9 | 2.18 | 100 | resolution ×2, scale |
| `temple` | drone | warnings | 20.73 | 111.8 | 3.49 | 100 | resolution ×2, scale |
| `auditorium` | unrecorded | pass | 1.52 | 26.9 | 0.122 | 98.8 | — (would be `indoor_large`) |

The auditorium is the useful row: never classified, it measures 1.5 m above the floor across
26.9 m, which is exactly the `indoor_large` band — a preset no rule could previously select.
Re-running `rocks` and `temple` through the fixed path cleared two findings outright — the lost
character height and the 1.75-vs-1.6 body disagreement between physics and nav — which is the
proof that `export_argv()` forwarding and the preset-derived `nav_params()` work. What remains on
those rows is the resolution of the cloud itself, not the settings.

### The interface, and why this shape

Modelled on what shipping products settled on — a short list of **numeric rows with a stated
band** (PIX4D's Quality Check: `Images`, `Dataset`, `Matching`, `Camera Optimization`,
`Georeferencing`, `Processing Failed`; RealityCapture's alignment table with its
"ideally under 0.5 px"; DroneDeploy's processing report with `Aligned Cameras`, `GSD`,
`Area Bounds (Coverage)`, `RMSE`). New **Quality** tab on every project:

- verdict first (`Blocked` / `Usable, with limits` / `Passed`) with what it means, not a colour;
- built-as, walk body and required headroom, grid and cell size, scale source, and the
  learned second ruler with its gap and its explicit "does not change your measurements";
- gate rows grouped **blocking → warnings → not measurable → passed (collapsed)**, each row
  carrying value, unit, threshold and the direction that is better;
- what the scene measured (working height, span, grid resolution, registration) beside the
  audit's findings, each with its remedy;
- the capture decision itself: every evidence line as `signal = value: because`, and the
  applied parameter table annotated with **who set each one** — `preset:drone`,
  `quality:high`, `cli`, `default`.

Also fixed because it was free: the preset `advice` string has shipped in the table since it was
written and had **never once been displayed** — choosing "Aerial Drone Orbit" now tells the
operator to fly 60-70% overlap. Quality tiers show their real pixels/steps/collider voxel
instead of hand-written labels that drift the first time a tier is retuned. A project card now
carries the gate verdict, and a hard failure is added to `warnings` rather than being inferable
from one red step tile.

### E5 + E2b follow-up: the GPS log decides what the frames could not

The brief lists GPS as a **mandatory** input, yet the pipeline lane scaled every drone
scene from an assumed flight speed. `rocks`' own `frame.json` shows how weak that is: its two
built-in rulers disagree by **1.9×** (speed ruler 4.73 m/unit, height ruler 9.04 m/unit),
and nothing on disk could say which was right.

- **Ruler D (`solve_frame.gps_scale`).** When `videos/<scene>/telemetry.csv` and
  `flight_metadata.json` exist, `pipeline.py` passes them to the `frame` step, which runs the
  survey lane's own similarity fit (`survey_georef.align_camera_trajectory`: time-matched,
  RANSAC, observability-checked) and takes the scale from it. Precedence is AR pose path >
  GPS > height anchor > speed. It refuses — and falls back, never fails the run — when the
  fit does (a straight line cannot fix rotation) or when the fit RMSE exceeds 5% of the flight
  path. `frame.json` keeps `scale_anchor_gps` beside the old rulers, the Quality panel labels
  the scale **metric**, and the audit's `scale` finding clears because the source is no longer
  `flight speed x clip duration`. Files are in the step's argv, so adding a log makes `frame`
  stale. Tested: a synthetic climbing arc recovers scale 4.000 from 20/20 cameras; a straight
  track is refused.
- **Flight pattern (`pipeline.flight_pattern`).** In the one branch where the frames say
  "steady aerial mount, sub-type unknown", the picker now consults the GPS track: ≥ 270° of
  heading turn at near-constant radius → `drone` (orbit); ≥ 2 heading reversals **with ≥ 70%
  of the path on one axis** → `drone_mapping`; ≥ 85% straightness → `corridor`; anything else
  stays unresolved with the reason recorded. The track is smoothed first — unsmoothed, 2 m of
  receiver jitter on 4 m steps made a straight run read as a zigzag — and the parallel-legs
  condition was added after random wandering was first misread as a grid. A GPS log never
  overrides a shaky (hand-held) mount.

**What this does not claim:** no real flight log for real footage exists here (`gnss-probe`
and `gnss-gap` carry probe telemetry, not the `rocks` flight), so ruler D and the pattern
classifier are verified on synthetic tracks only. `rocks` and `temple` still carry the
`scale` finding, correctly.

### Still open in E

1. **`temple` is walkable-ish, not fixed.** Its grid resolves at 3.49 m because only ~7k
   gaussians fall in the near-ground band over 111 m — the cloud is sparse, not the settings
   wrong. Forcing a finer `cell_meters` was tried, measured against that density, and
   **reverted**: a finer grid than the data supports manufactures holes. Culling the floater
   cloud that inflates the bounds is the real route, and `strip_clouds`'s "desaturated AND more
   than a metre above local ground" rule is unusable indoors, which is why cull stays keyed to
   the preset.
2. **E3 remains assertion.** The `object` preset now declares `voxel 0.01`, `cell_meters 0.01`,
   `max_step 0.02` and a token 0.15 m body, and `QUALITY_DEFER` is what lets a scene-scale knob
   survive the quality tier — but nothing close-up has ever been scanned here, so no number in
   that row is a result.
3. **Three presets still unreachable automatically** (`object`, `outdoor_building`,
   `sky_heavy`); `drone_mapping` and `corridor` are now reachable from a GPS log, and the run
   dialog says which decision the frames alone cannot make.
4. **Drone scale is measured only when a GPS log is supplied** (E5). Scenes without one keep
   `flight speed x clip duration` and the `scale` finding.


## F. Georeferencing & accuracy

| ID | Item | Type | Effort | Status |
|----|------|------|--------|--------|
| F1 | One-click survey path in-app (telemetry → align → CRS → checkpoint RMSE) | wire | M | **verified** — guided six-step panel + `POST /api/survey/advance`; runs every CPU step, stops at the GPU approval |
| F2 | GCP/checkpoint upload + auto accuracy report (turns "unverified" → "verified") | build | M | **verified** — CSV in degrees / UTM / ENU, ellipsoidal or MSL heights, → exact `checkpoints.json` → evaluation |
| F3 | Vertical datum options (EGM geoid) so heights are MSL, not just ellipsoidal | build | M ⚠dep | **verified** — EGM96 15′ grid; matches PROJ `vgridshift` to 5×10⁻¹³ m; compound CRS re-read by GDAL and laspy |

**F1 — the path was already in the app, in pieces.** Four loose buttons in Details
(save/prepare/align/evaluate) plus a survey engine in the Run dialog meant an operator had to
know the order and which step was stale. Now `survey_workflow.survey_steps()` derives six rows
(inputs, prepare, reconstruct, align, checkpoints, evaluate) from `scene_status`'s own
validated fields — so the panel cannot call a step done that the backend would call stale —
and `advance()` runs every CPU step that can run now, in order, with the same functions the
individual buttons call. It never starts GPU work: when reconstruction is next it says
"Needs you" and points at the Run dialog, whose own confirmation stays the only way in.
A stale evaluation is re-evaluated; changed inputs are re-prepared (evaluate refuses first,
then prepare runs). Verified in the browser on `gnss-probe`: Done / Done / Needs you /
Waiting ×3, and the button disables itself with "Nothing to run on this machine now".

**F2 — checkpoints without hand-writing JSON in a tangent frame.** `survey_checkpoints.py`
reads one row per point — the surveyed position and the same feature read off the delivered
model — in whichever frame each was recorded (`ref_lat/lon/height`, the scene's UTM zone, or
local ENU; the model side in the UTM or ENU product set), converts both into the alignment's
ENU frame, and writes the evaluator's exact schema. MSL heights on a GCP sheet are converted
through the geoid, never mixed with GPS heights (tested: the same sheet read as ellipsoidal
is off by ≈ 53 m at Delhi). Refused with the row number: duplicate ids, non-numbers, a point
more than 1 km from its model partner (two frames, not a 1 km error), EGM96 declared on ENU.
The operator's "withheld from reconstruction" declaration is a checkbox, stored in
`checkpoints_source.json` **bound to the SHA-256 of the checkpoints file**: hand-edit the
file and the declaration silently drops, so the accuracy report reverts to "independence
unverified". ≥ 8 points get the existing fit/hold-out split. Tested end to end on a synthetic
aligned scene: a planted 0.30/−0.40/0.12 m error comes back as 0.500 m horizontal and 0.120 m
vertical RMSE.

**F3 — mean-sea-level heights.** `survey_geoid.py` interpolates the NGA EGM96 15′ grid
(`data/geoid/us_nga_egm96_15.tif`, PROJ's public-domain file, read with the cv2 already in the
stack — no new runtime dependency). Checked against PROJ's own `vgridshift` on 5,000 random
points: max difference 5×10⁻¹³ m; N(0°, 0°) = 17.16 m, the published value. The Run dialog
offers "Heights in the exports": ellipsoidal (default, what GPS measures) or EGM96. EGM96
products carry a compound `UTM + EGM96 height` CRS (EPSG 5773 vertical) that GDAL and laspy
both resolve; the WGS84 position CSV always gets both height columns, each named for its
datum. Without the grid the run keeps ellipsoidal heights and records a refusal — it never
writes a height it cannot label. Not claimed: EGM96's own error (~0.5–1 m, worse in
mountains), or a national height datum (1–2 m further).

## G. Export & deliverables

| ID | Item | Type | Effort | Status |
|----|------|------|--------|--------|
| G1 | Textured-mesh UV bake (was per-vertex colour only; `claims_textured_mesh: False`) | build | L | **verified on real footage** — `rocks`: 15,776 faces, 72/72 views; held-out edge agreement +21% over per-vertex colour |
| G2 | Validate exports with real GDAL/laspy/PDAL (was self-parsed) | wire | M ⚠dep | **verified** — found and fixed 5 real defects; LAS + GeoTIFF now re-read by laspy/GDAL on every write |
| G3 | One-click project bundle (model + measurements + report + CRS) as ZIP | build | S | **verified** — `GET /api/workspace/bundle`; 19.5 MB `rocks` bundle through the app's proxy |

**G2 — the validators found what the round trip could not.** laspy 2.7, rasterio 1.5
(GDAL 3.12) and pyproj 3.8 are installed in `.venv` as validators only; the writers still need
nothing but NumPy.

| defect | effect before | fix |
|---|---|---|
| LAS header was a 239-byte layout of the module's own | **every LAS ever written was unreadable** — laspy: "Incoherent header size" | 375-byte LAS 1.4 R15 header, field-by-field from the spec |
| LAS VLR header had record-length and description swapped; GeoKey VLR under user id `LASF_proj` | CRS unreadable even once the header parsed | published order; `LASF_Projection`; WKT bit 4 set, one CRS record only |
| LAS extents written offset-relative; per-return counts absent | wrong bounds in any GIS | real-coordinate extents; return 1 of 1, counts filled |
| GeoTIFF nodata in tag **42112** (GDAL_METADATA) | GDAL ignored it: every DSM hole read as −9999 m terrain | tag 42113 (GDAL_NODATA) |
| `utm_forward` chose the false northing per point | a scene crossing the equator tore **10,000 km** apart inside one product | hemisphere fixed per product, as EPSG defines it |

Also fixed on the way: GeoTIFF used key 1026 (a citation key) for a geographic CRS instead of
2048, and `read_mesh_ply` crashed on COLMAP's Delaunay meshes (`vertex_index`, not
`vertex_indices`). Every LAS and GeoTIFF write is now re-opened by laspy / GDAL when installed
and **deleted** if they read it differently; the file's `verified` field names the reader,
`externally_validated` is true only then, and the export manifest lists
`externally_validated_formats`. pyproj agrees with the hand-written UTM series to < 1 mm in six
zones and with ENU↔ECEF to 10⁻⁵ m. **Still round-trip only:** glTF, FBX, OBJ (no Khronos
validator or FBX SDK here) — though the textured GLB loads in PlayCanvas's glTF parser.

**G1 — textures from the frames that built the mesh.** `survey_texture.py`: COLMAP cameras
(text or binary, with SIMPLE_RADIAL/RADIAL/OPENCV distortion applied), a per-view depth buffer
for occlusion, best view per face by `|cos| × on-screen area` with neighbour-majority seam
smoothing, a packed square-cell atlas sized to what the frames can actually resolve, and
bilinear sampling of the original frames. Writes OBJ+MTL+JPEG, glTF+bin and a single-file GLB.
Two run paths:

- **pipeline lane** — new advisory CPU step `texture`: COLMAP's graph-cut Delaunay mesher over
  the *sparse* model (1.4 s for `rocks`), the bake, and output in the viewer's world frame
  (`(P @ R.T) × scale`, exactly what `export_viewer_assets` uses) under `work/<scene>/textured/`,
  listed in Files and included in the bundle. Faces no camera saw — the mesher's closing hull
  walls — are dropped and counted, not shipped grey.
- **survey lane** — `survey_deliver` bakes onto the Poisson mesh using the undistorter's own
  pinhole cameras and frames; the textured OBJ/glTF lead the format ledger in both the ENU and
  georeferenced sets and the manifest finally says `claims_textured_mesh: true`. A failed bake
  leaves the mesh untextured with the reason recorded.

Measured on `rocks` (17,383 faces; 1,607 unseen hull faces dropped; 15,776 textured; 3,307 of
them from a grazing view; 1157² atlas; 4.5 min including the hold-out check). Hold-out: bake
without one view, render the mesh into it, compare with the real frame, against the same mesh
coloured per vertex from the same frames:

| held-out view | pixel MAE textured / per-vertex | edge agreement textured / per-vertex |
|---|---|---|
| 00000 | 11.68 / 11.98 | **0.447** / 0.360 |
| 00012 | 12.59 / 12.70 | **0.480** / 0.362 |
| 00024 | 13.27 / 13.13 | **0.448** / 0.356 |
| 00036 | 11.42 / 11.29 | **0.529** / 0.420 |
| 00048 | 10.99 / 10.66 | **0.545** / 0.433 |
| 00060 | 11.46 / 10.98 | 0.515 / 0.516 |
| mean | 11.90 / 11.79 | **0.494 / 0.408 (+21%)** |

Read it honestly: the texture puts **edges where the real frame has edges** 21% better, which
is the detail a textured mesh exists for; on raw pixel error it is a tie, because the sparse
Delaunay geometry is coarse and pixel error rewards blur on slightly misplaced surfaces. Two
bugs were found by the synthetic tests (known pattern on a plane, rendered exactly): a
depth-buffer splat capped at 48 subdivisions let a roof fail to hide the ground beneath it,
and a hull face far off-screen made the splat 20× slower — large faces are now rasterised over
their clipped bounding box. **Not done:** no exposure balancing between views (seams remain),
no photo-consistency test (a moving car can be painted onto the road), occlusion tested at
the face centroid. The Poisson-mesh (survey) path is covered by tests but has not been run on
a real survey reconstruction, because none exists here without a GPU run.

**G3 — the bundle.** `workspace_bundle.py` builds the ZIP from exactly what the project page
lists (so a stale survey product the page refuses can never ship), plus authored
measurements (JSON/GeoJSON/CSV with uncertainty), placements, reports (gate, scenario, audit,
diagnostics), `crs/` (CRS JSON + `.prj` when georeferenced, otherwise a plain "NOT
georeferenced" statement with the scale caveat), a README repeating every caveat the page
shows, and `manifest.json` with each file's SHA-256. Source frames and COLMAP internals are
excluded (tested). Capped at 4 GiB with a clear refusal.

## P1. Phase 1 scoring core (2026-09-26)

Plan: `docs/APPLICATION_PLAYBOOKS.md` §6. Everything below is tested on synthetic data or
the `rocks` clip; **no real flight with a GPS log exists yet**, so no accuracy number and
no 10-minute gate result is claimed.

| ID | Item | Module | Evidence |
|----|------|--------|----------|
| M11 | Straight-track georeferencing: GPS fixes 6 DoF, gravity (gimbal attitude, else ground plane) fixes the roll about the track; tilt check on ordinary fits | `survey_gravity` | 13 tests; known transform back within 0.05°; ground-plane fallback reproduces a 3° cross slope as a 3° tilt, exactly as documented |
| M12 | Takeoff-anchored heights: GNSS level + barometric shape, drift folded into uncertainty | `survey_vertical` | 200 synthetic flights (60 s correlated GNSS error): height RMSE 1.97 → 0.71 m; 2σ coverage 88% (ideal 95%), stated in the module |
| — | Side channels kept: `telemetry_aux.json` (gimbal attitude, relative altitude) from SRT/DJI CSV; prepare fuses heights (so pose priors get them too); alignment falls back to M11 only on a straight-track refusal | `survey_workflow` | SRT → sidecar → fused preparation → stale-on-edit, tested end to end |
| M8 | MGRS both ways, DMS, CE90 (exact for unequal sigmas) / LE90 | `survey_coords` | 11 tests; `31NAA6602100000` published reference; CE90 vs 2M-sample Monte Carlo within 1% |
| M2 | SMRF ground filter → `dtm.tif`, `ndsm.tif`, `dtm_observed.tif`, LAS class 2/1 | `survey_terrain`, `survey_export` | recall 100%, false ground 0.9%, DTM under a roof −0.09 m, roof nDSM 10.09 m (true 10); 1 km² / 3M points in 25.6 s; laspy + rasterio read-back |
| M1 | True orthomosaic `ortho.tif` (RGBA, alpha 0 = unseen) in the georeferenced set | `survey_ortho`, `survey_formats.write_rgba_geotiff`, `survey_deliver` | 5 tests on ray-cast frames; wall occluders added after the test caught see-through-buildings; GDAL samples the roof colour at its UTM coordinate |
| M13 | Dense speed: PatchMatch 3 iterations / 8 samples in `fast` and `budget` | `survey_workflow.DENSE_PROFILES` | see table below |

**Defect fixed: every DSM was mirrored north–south.** `dsm_grid` filled rows south-first
under a north-up transform. The GDAL read-back check compared values, not placement, so it
passed. A pillar at the north edge read 30 m at the south coordinate under rasterio; now
fixed with a regression test. **Any `dsm.tif` produced before 2026-09-26 must be regenerated.**

**M13 ablation** (`scratch/gpu_dense_speed_ablation.py`, rocks, 72 frames, identical poses;
agreement with the consistency-on cloud, not ground truth; 0.25% of the diagonal ≈ 0.12 units):

| Variant | Dense s | Speed-up | Precision / recall @0.25% | Verdict |
|---|---:|---:|---|---|
| control (`fast` before) | 457.7 | 1.00× | 0.937 / 0.942 | — |
| **3 iterations, 8 samples** | **196.6** | **2.33×** | 0.946 / 0.937 | **adopted** |
| 6 source views | 296.5 | 1.54× | 0.997 / 0.609 | rejected: ~38% of surface lost |
| both | 121.5 | 3.77× | 0.999 / 0.608 | rejected |
| both + every 2nd reference | 61.2 | 7.48× | 1.000 / 0.582 | rejected |

The six-view loss is not a fusion-threshold artefact: re-fusing with `min_num_pixels` 3 and 2
lifts recall only to 0.618 / 0.633. The benchmark frames are **1000×562**, so these rates are
at the full `fast` working size (1000 px); 1080p sources are downscaled to it.

**Sparse-anchored monocular depth — measured and rejected as a measured product.**
`scratch/mono_dense_ablation.py`: MoGe-2 distance maps (0.173 s/frame inference) fitted per
frame to that frame's COLMAP keypoints, back-projected along COLMAP rays, kept where ≥ 2 of 4
covisible frames agree. Whole stage ≈ 5 s CPU + ≈ 12 s inference for 72 frames (vs 197 s
PatchMatch), but precision @0.25% is 0.46–0.54 (PatchMatch 0.94) and @1% 0.82–0.89 (0.997).
The cause is MoGe itself: held-out anchor error per frame is 9.6% of range with a per-frame
scale, 5.0% with a quadratic correction field plus a log-distance term
(`scratch/mono_field_probe.py`) — an order of magnitude above PatchMatch. It may serve a
*labelled* coarse preview (P-R first look), never the measured cloud.

**The speed gate is still not met on paper:** at ~2.8 s/frame a ≈ 270-keyframe mapping flight
needs ≈ 750 s of dense alone against a 390 s budget. Remaining levers: fewer, better-spread
frames (the baseline-aware selector), 2.5D DSM for nadir flights, and overlapping dense with
sparse. Nadir and frame-budget levers need a real mapping flight to measure — `rocks` is an orbit.

## P2. Phase 2 — urban planning editor (2026-09-26)

Plan: `docs/APPLICATION_PLAYBOOKS.md` §2.2, §3.2, §6.2. A **Plan** tab
(`/projects/<scene>/plan`) draws proposed roads, buildings, plots, objects and demolitions
over the scan, in named schemes, without ever editing the scan.

| Piece | Module | Evidence |
|---|---|---|
| Frame registry: viewer ↔ COLMAP ↔ ENU ↔ WGS84/UTM/MGRS; `solve_frame` now keeps the full GPS similarity, not only its scale | `scene_frames.py` | round trip exact to 1e-9 m; local scenes refuse Earth coordinates |
| Proposal layer: schemes, features, revisions (409 on a stale write), whole-list replace for undo/redo | `workspace_proposals.py`, `workspace_plan_api.py`, routes `/api/workspace/plan/*`, `/coords` | 18 tests incl. the HTTP contract and write guard |
| Roads: centreline, width/lanes/footpaths/median, drape or graded (max grade), cut/fill, buildings in the way, trees removed | same | graded road ≤ limit and cuts the test hill; drape rests only on measured ground and **bridges** coverage gaps (reported as metres bridged) |
| Buildings: any footprint (2-click rectangle or polygon), floors × floor height, flat/gable/hip, use; GFA, height, volume | same | 10×10×4 → 400 m² GFA, gable ridge height exact |
| Plots / zones with rules: height, floors, FSI, coverage, setback → violations tinted red in 3D | same | three violations flagged, all cleared by a compliant redesign |
| Demolish: existing structure hidden on the GPU (PlayCanvas unified work-buffer modifier) | `viewer/plan_core.js`, `pc.js` | shader/JS agreement on 20,000 random points; in the browser 2,276 of 15,199 rocks splats cleared, verified via engine state and screenshot |
| Objects + arrays along a line (18-item urban catalogue) | same | 3 lights at 30 m spacing, 5 m offset |
| Existing inventory from semantic labels (≥ 12 m² footprints, 2 m tree clusters) | `existing_inventory` | test building found at 9 ± 0.2 m; on `rocks` the labelled "building" is a rock pile — labels, not a survey, and the panel says so |
| Compare: Existing / Proposed / Flicker; before/after table (buildings, footprint, GFA, tallest, road area, canopy, trees, residents, cut, fill) | `plan-panel.tsx` | exercised in the browser on `rocks` |
| Export GeoJSON (WGS84 when georeferenced, clearly LOCAL otherwise) | `export_geojson` | lat/lon checked on a georeferenced fixture |

**Found while verifying in the browser:** the viewer file allowlist blocked the new module
(403); the redraw signature ignored height edits; the splat component runs in PlayCanvas's
*unified* mode, where the material hook I first used is null — each was fixed and pinned by
a test. The drape originally followed the unscanned edge of the ground grid (a 185% "grade"
on `rocks`); it now interpolates only measured ground and reports what it bridged.

### P2.1 Phase 2 remainder (2026-09-26)

Built and verified in the browser on **`work/flatplan`**, a synthetic flat test scene
(`scripts/make_plan_scene.py`: 120 × 120 m ground at y = 0, two existing buildings with
labels and solid colliders, three trees, a *synthetic* GPS fit at 30.7333 N 76.7794 E). A
flat scene isolates editor bugs from scan-data bugs; it is labelled synthetic everywhere.

| Piece | Module | Evidence |
|---|---|---|
| 3D editing: corner handles, "+" midpoints insert a corner, double-click removes one, move grip (or grab the feature itself), rotate grip; Shift snaps 1 m / 15°; drags run on the feature's base plane and refuse grazing rays (< 4°) | `plan_core.js` (`planHandles`, `dragShape`, `removeVertex`, `planeHit`), `pc.js`, `lib/plan.ts` (`editShape`, `applyEdit`) | node tests; browser: corner drag, insert → 5 corners / 241.9 m², remove → 192 m², rotate re-triggered the plot's setback rule live, undo restored a move |
| Compare: **Swipe** (draggable divider) and **Split** (side by side) | `viewer-panel.tsx` (`CompareLayer`), `camera-follow` / `camera-set` commands | a second viewer on Existing, cameras relayed through the host; orbit/zoom in either half stay locked |
| Shadow study: one moment (with a time scrubber, re-cast after every edit) or sun hours over a day window; new shadow, sunlit-again (demolitions), hours lost in bands; ≥ 2 h-of-sun area (BRE-style) | `plan_shadow.py`, route `plan/shadow` | NOAA sun: solar-noon elevation 35.83° / 82.7° at 30.73 N on the solstices; 12 m box at 30° casts 20.8 m; winter-morning shadow falls north-west; on `flatplan` the scheme costs 1,985 m² ≥ 1 h of sun on 21 Dec 9–15 h |
| Cadastral import: GeoJSON (WGS84, declared UTM EPSG, or our own LOCAL export), KML, DXF closed polylines → plots; FSI/FAR, height, coverage, setback, floors attributes → plot rules; parcels > 500 m off the scan skipped with a reason | `plan_exports.parse_parcels`, route `plan/import` | round trip through our own GeoJSON export within 1 cm; KML/UTM/DXF placed within 1 cm; browser: a khasra parcel placed and its rules applied, a 3 km-away parcel skipped |
| Export: **CityJSON 2.0** (buildings as closed, outward LOD2 solids with wall/roof/ground semantics, roads with traffic areas, trees, furniture, land use, demolitions), **DXF** R12 (layered footprints, road centrelines/edges, 3D faces, labels), **3D Tiles 1.1** (tileset + GLB, ENU→ECEF root transform) | `plan_exports.py`, `plan_glb.py` | every directed solid edge has exactly one reverse twin; DXF footprints read back to 1 cm; tile content inside its bounding box and anchored at the origin's ECEF. **Not run through cjval / 3d-tiles-validator** (not installed) |
| Object models: tree (small/large), street light, car as real low-poly meshes; the other 14 items stay boxes, and `OBJECT_MODELS` is the one place to add more | `workspace_proposals.py` | each model within its catalogue box, yaw turns it |
| Walk physics: proposed walls (one box per footprint edge, any polygon), poles, trunks and cars block the walker; tree crowns and lamp arms do not; demolished structure is cut out of the scan's collider and a ground patch keeps the site walkable; Existing view restores the scan collider | `plan_core.planColliders` / `clipTriangles`, `pc.js` | walker stopped at z = −34.49 against the tower (wall −34.15); walked straight through the demolished block in Proposed, stopped at −15.34 against it in Existing |
| Demolition "cleared site": bare-ground patch where the hidden structure stood (a scan never saw the ground under a roof) | `derive_clip` | no black hole on `flatplan` |

**Found while verifying in the browser:** translucent plan meshes (shadows, plot outlines,
demolition volumes) were drawn *before* the splats and painted over by the ground — they now
render on a `PlanOverlay` layer after World; a static compound collider that enters the world
with one child keeps only that child in the broadphase (every other wall was walk-through) —
the compound is now assembled off-scene (the scan's own furniture colliders use the old
order and are flagged separately); the Next proxy's route allowlist needed `plan/shadow` and
`plan/import`; a projected handle used the camera component's `worldToScreen` signature; the
swipe viewer's camera broadcast was enabled before it received the main camera and pushed
its default view onto the main viewer.

**Added 2026-09-27 — per-building facade completeness (URB-17).** `scripts/facades.py`, route
`plan/facades`, Plan tab → Existing buildings → *Facade completeness*: each inventory wall is cut
into 1 m cells; a cell is observed (> 2 scan points within 0.75 m of the wall), weak (1–2) or
unobserved (none); each point counts only for its nearest wall, and points on a corner count for
neither. Unseen walls are hatched red, weak ones amber. Tests: a box with no points on its north
wall reports N 100% unobserved and 27.8% of the whole wall area (10 of 36 m of perimeter).
Browser: `flatplan` 100% (full synthetic shells); `rocks`' labelled "building" (a rock pile) 2%.

**Closed 2026-09-27:** porous crowns in the shadow study (Beer-Lambert through tree outlines,
partial sun hours under trees; `plan_shadow.light`); compare camera sync in fly and walk mode
(the eye position + forward vector is relayed, the second viewer flies to it); zipped
shapefiles (.shp/.dbf/.prj) and any declared CRS for GeoJSON / DXF / shapefiles through pyproj
(`plan_exports.read_shapefile_zip`, `to_lonlat`; a UTM-43N shapefile and an EPSG:3857 GeoJSON
land within 5 cm of their WGS84->ENU positions - the test caught that raw UTM offsets are ~0.9 deg
off true north here, which is grid convergence, not a bug).

**Still not built:** translucent overlays show through scanned objects (splats write no depth -
an engine limitation); validators for CityJSON / 3D Tiles (cjval / 3d-tiles-validator are not
installed; our own structural checks run); the Split view loads the scene twice (memory).

## P3. Phase 3 — mission planning and rehearsal (2026-09-26)

Plan: `docs/APPLICATION_PLAYBOOKS.md` §2.4, §3.1, §6.2. A **Mission** tab
(`/projects/<scene>/mission`) plans an operation on the scan, analyses it, rehearses it in
first person and reviews the run. Missions are proposals of kind `mission` in the same store as
planning schemes (revisions, undo, 3D drag editing, write guard). Verified end to end in the
browser on `work/flatplan`.

| Piece | Module | Evidence |
|---|---|---|
| Tactical symbols (MIL-02): hostile / friendly / unknown / neutral frames, 11 roles, bearing + sector + range, behaviour, count, elevation; they sit on the scanned top surface (a roof counts); routes with named waypoints and pace; phase lines | `scripts/mission.py`, `mission-panel.tsx` | sniper placed on a 9 m roof by clicking; bearing 90° = +x, 0° = −z (true north through `scene_frames` when georeferenced) |
| Sight (MIL-07): point-to-point LOS tool, per-hostile viewsheds (sector + range) as an overlay, watched / dead ground areas | `scripts/mission_analysis.py` | LOS blocked by the test block at its face (±0.6 m); viewshed shadow behind it; a roof's own edge hides the ground below from its centre |
| Route analysis (MIL-06): length, Tobler ETA × pace, climb, per-hostile exposure seconds / first seen / closest, dead-ground %, waypoint and phase-line crossing times, height profile coloured by exposure | same | 50 m flat = 36.0 s (Tobler); browser: 120 m route, 1:26, exposed 79 s |
| Covered-route suggestion: A* over walkable measured ground, cost = length × slope + exposure | same, `mission/covered-route` | browser: exposure 79 s → 27 s for 226 m instead of 120 m; added as a new route, the plan's is kept |
| HLZ finder (MIL-08): clear, flat (≤ 7°), obstacle-free (≤ 0.5 m) circles on measured ground | same | the test block is excluded; 5 candidates on `flatplan` |
| Candidate enemies from person/vehicle labels (MIL-03, heuristic) | `mission.label_candidates` | added as UNKNOWN, never as hostile; `flatplan` correctly reports none |
| Rehearsal (RH-1..5, RH-10): bots at the planned posts with their sector/range; sentry scans and holds (a roof post never walks off or sinks), overwatch far + narrow, patrol walks its route, reaction force hunts noise; snipers engage at their range; waypoint compass tape, next-waypoint bearing/distance/ETA, off-route warning, phase-line alerts, live "seen by N" meter; day / dusk / night, NVG, fog cap bot sight; respawn at the last waypoint reached | `viewer/pc/scripts/mission.js`, `rehearsal.js`, hooks in `bot.js` / `combat.js` | node tests; browser: WP2 at 47.7 s, objective at 79.1 s; at night the 120 m post could not see at 96 m while the 300 m sniper could |
| Recording + AAR (RH-6): 10 Hz player and bot poses + events, saved through the API; top-down replay over a hillshade basemap with the planned route, a track coloured by exposure, live sight lines, timeline with exposure strip, clickable event log, per-enemy exposure | `mission.save_run`, `aar-replay.tsx` | browser: night + NVG run, 2:15, 5 hits, 1 casualty, 89% exposed, objective reached — opened automatically after the run |
| Mission pack (MIL-20): KMZ (styled placemarks, routes, phase lines) + GPX (routes as `rte`, symbols as `wpt`), refused for local scenes | `mission.mission_pack` | 3 `rtept`s for a 3-waypoint route; every item says "planned, not observed" |

**Found while verifying in the browser:** a suggested route has more waypoints than the viewer's
16-meshes-per-feature limit, so the whole plan update was refused — posts are now one mesh; the
rehearsal received the engine as a shallow copy without `drawLines` (every frame threw, so no
route line) — it now gets the real engine; waypoint beacons were painted over by the splats
(same layer issue as P2.1); the run ended and uploaded before the objective events were
recorded ("not completed"); arena bots only fire within ~30 m, so a planned sniper never shot;
respawn teleports counted as walked distance.

### P3.1 Phase 3 remainder (2026-09-27)

| ID | Piece | Module | Evidence |
|---|---|---|---|
| MIL-03 | Candidate enemies from **video detections** (the M10 layer), before scan labels; added as UNKNOWN with "seen N×, t–t s" | `mission.detection_candidates`, `mission/candidates` | ops site: 2 people + 1 vehicle offered with their sightings |
| MIL-04 | Threat heatmap (heuristic): per standable cell (open ground or flat elevated top), share of the route an enemy could see (reverse viewsheds from ≤ 40 route points) × height advantage × nearby cover; ranked candidates 15 m apart, none on the route | `mission_analysis.threat_heatmap`, `mission/threat` | test block's roof ranks first; ground hidden behind it scores 0; `flatplan`: roofs at 87–100% overwatch. It picks a tree crown as "elevated" too — reported as roof *or crown* |
| MIL-09 | Obstacle list: everything ≥ N m above the ground with height, area, MGRS; small tall ones called pole/mast | `mission_analysis.obstacles`, `mission/obstacles` | test: one 8 m block, 100 m²; `flatplan`: 5 above 3 m |
| MIL-16 | KLV ingest: MPEG-TS → PAT/PMT → metadata PES (PTS) → MISB ST 0601 UAS local set with CRC-16 check; sensor lat/lon, HAE (tag 75) or MSL (tag 15, flagged for geoid), platform/sensor attitude → gimbal side channel for M11. Sniffed by content like every other flight log | `survey_klv`, `survey_inputs.parse_klv` | round trip through a synthetic STANAG 4609 stream; a corrupted byte fails the checksum; CRC-16 check value 0x29B1 |
| RH-6 | AAR PDF: summary + per-enemy exposure + events; top-down track coloured by how many enemies saw the player, posts, hits; who-saw-you-when chart | `aar_report.py`, `mission/aar-pdf`, button in the review | 3-page PDF from a stored `flatplan` run |
| RH-8 | Sand table: top-down briefing miniature — APP-6-style frames (hostile diamond, friendly rectangle, unknown circle, neutral square), sector arcs, routes with arrows and waypoint names, phase lines that turn green once crossed, 50 m talk-on grid, scale, north; the approach plays back on its Tobler timings, stepping by waypoint; fullscreen for a projector | `sand-table.tsx` | browser: Op ALPHA, 1:26 playback, SP → WP2 → OBJ |
| RH-7 | VR (WebXR immersive-vr): "Enter VR" appears only where the browser offers it; a rig at the player's feet carries the headset; left stick walks where the head looks through the same capsule, right stick snap-turns 30°, right trigger fires | `viewer/pc/scripts/vr.js`, `vr_input.js`, hooks in `pc.js` | 9 node checks on the stick/snap maths; the headless arena still loads with the module (the 3 failures in `test_combat_play.js` predate it and fail without it too). **Not run on a headset: 72 fps, comfort and controller aim are unverified; the splat-LOD path RH-7 asks for is not built; the shot leaves from the head, not the controller** |

**Closed 2026-09-27:**

| ID | Piece | Module | Evidence |
|---|---|---|---|
| RH-9 | Multi-user rehearsal with an instructor: in-memory session on the local server; players (this machine or LAN headsets over HTTPS) post pose ~8 Hz and see each other as figures; the instructor watches a live map, moves enemy posts, sends messages, ends the exercise. LAN devices may write only `session/join` and `session/state`, only with the session token, only to memory | `mission_session.py`, `workspace_mission_api`, guard in `workspace_api.handle`, `viewer/pc/scripts/session.js`, `instructor-view.tsx` | tests: token, bounds, command order; curl without an Origin: wrong token refused, other writes refused, token join accepted; browser: two players live on the instructor map, a post moved by click. Bots are simulated per client (hits are not replayed across clients) and the UI says so |
| MIL-10 | Trafficability GO / SLOW-GO / NO-GO for foot, wheeled, tracked from slope, clutter, roughness, road and water labels; unobserved is NO-GO | `mission_analysis.trafficability` | block NO-GO, open ground GO; browser map on `flatplan` |
| RH-5 | Fog drawn: linear scene fog opaque at the stated visibility (splat shader honours it), darker at dusk/night | `mission.fogFor`, `rehearsal.js` | node test |
| — | Fire by role: snipers single aimed rounds 2.5-4 s apart and 3x tighter, machine guns 4-8 round bursts | `mission.fireProfile`, `bot.js` | node test |
| RH-7 | VR aim along the right controller's pointer ray (tracked-pointer), head aim as fallback | `vr.js`, `weapon.js` aimRay hook | not verified on a headset |

A module-syntax guard now parses every viewer script as an ES module (`test_viewer_syntax.py`):
an extra comma in `rehearsal.js` passed every unit suite and only the headless arena caught it.

## P4. Phase 4 — operations analyses: disaster, construction, border (2026-09-26)

Plan: `docs/APPLICATION_PLAYBOOKS.md` §3.3–3.5, §6.2. Analysis core, CLI (`ops.py`), field
packs, `/api/workspace/ops/*` and an **Operations** tab (`/projects/<scene>/ops`: Disaster ·
Construction · Border). **Tested on synthetic scenes with known answers only; no real flight
exists, so no acceptance demo has passed.** The browser check ran on `work/opsbefore` +
`work/opsafter` (`scripts/make_ops_scene.py`: one synthetic GPS origin, a collapsed and a
half-collapsed building, debris and a parked car on the road, a 154 m³ stockpile, a pad dug
1.5 m of a 2 m design, a river channel, a 12 m ridge, two cameras and detector boxes).

| ID | Item | Module | Evidence |
|---|---|---|---|
| M3 / CON-03 / BOR-03 / DIS-08 | Two-epoch change: shared grid (per-cell mean), shift co-registration (whole cells then 0.1 cell) on iteratively found stable cells, vertical median, LoD95 per cell = 1.96·√(σ_reg² + se_a² + se_b²), gain/loss regions with volume ± σ and WGS84 + MGRS | `survey_change.detect_change` | known offset (1.3, −0.7, 0.9) m recovered within 0.15 m / 0.05 m; unregistered diff shows 20× more false change; cone pile 113 m³ and 96 m³ pit found within 15%; a gap in epoch 2 is `unobserved`, not "no change" |
| DIS-03 / CON-01 | Volume change inside a drawn polygon; fine-cell sampling gaps interpolated and reported as `filled_fraction` | `survey_change.region_volume` | pile 113.1 m³ within 6 m³ at 0.5 m cells (before the gap fill it under-counted by the 20% empty cells) |
| CON-02 | Design surfaces: LandXML TIN (N E Z order handled, hole faces skipped), DXF 3DFACE, GeoTIFF DEM (edge half-cell clamped); TINs rasterised on their own triangles | `survey_design` | 100 m³ mound and 40 m³ trench within 6 % for LandXML and DXF; GeoTIFF 240 m³ within 15; missing offset refused |
| CON-02 | Cut/fill vs design; on-grade = max(30 mm, per-cell LoD95 from pooled point noise); registration σ added as correlated error | `survey_design.cut_fill` | on-grade site < 2 m³ over 3,600 m² |
| CON-05 | Zone progress % from a pre-works baseline | `survey_design.zone_progress` | half-finished zone reads 50 ± 5 % |
| DIS-02 | Damage triage per building: roof cover, 3×3 plane residual ("rough"), height vs reference → intact / partial / collapsed / unknown, with the reason | `survey_damage.assess` | flat, gable, rubble, half-collapsed graded correctly; unseen footprint `unknown`; derived footprints say flattened buildings are missed |
| BOR-04 | Profile along a line with gaps where unobserved; post coverage of a line (range, bearing, FoV, eye height) with blind stretches by chainage; unobserved ground blocks sight | `survey_corridor.profile`, `blind_spots` | ridge hides its far side; second post adds > 15 % coverage; 60° cone gives ±23 m of line at 40 m standoff |
| BOR-02 | Tiled DSM GeoTIFF + LAS per 1 km tile with `tiles.json` | `survey_corridor.tile_products` | 2.5 km strip → 3 tiles, read back by our readers in UTM |
| DIS-09 / CON-08 | Field pack: `report.pdf` (facts, tables, map panels, limits; unobserved drawn grey), KMZ (only when georeferenced), CSV, overview GeoTIFF, `pack.json`, zip | `ops_report.build_pack`, `ops.py` | CLI end to end; local scene gets no KMZ and says why |

| DIS-01 / H1 | First map while the run continues: after every merged progressive window the accumulated sparse model becomes a colour mosaic (`preview.png`) and, when the window's cameras fit the GPS track, a UTM DSM; `latest.json` is what the Operations tab shows | `survey_firstmap`, hook in `survey_progressive.execute_plan` (`on_window`), `survey_workflow._first_map` | synthetic model + GPS: UTM zone 43N DSM, block height 8 m after the GPS scale; failures recorded, never raised; `images.txt` pose lines found by shape (empty 2D-point lines kept). **Not yet run inside a real `--progressive` COLMAP run** |
| DIS-06 / M10 | People and vehicles as a layer: `detections.json` boxes (or the dynamic masks) → bottom-centre ray from the recovered camera → hit on the scanned surface → merged across frames (3 m) with sightings, time span, "moving"; WGS84 + MGRS | `survey_detections` | ray test lands within 0.1 m; ops site: 2 people + 1 vehicle within 1 m of truth |
| DIS-04 | Blocked roads (road cells with obstacles > 0.5 m, new since the pre-event flight; the road network is taken from both epochs because debris hides the road it covers) and vehicle routing (Dijkstra, vehicle-width erosion, slope limit, road vs off-road cost, endpoints snapped ≤ 15 m to drivable ground) | `survey_response` | ops site: debris + car found; 93 m route, 84 m on road, 10 m round the debris; a site beyond the 12 m ridge is refused as unreachable |
| DIS-05 | Flood (bathtub): cells below the level connected to a seed; depth bands, volume, buildings wetted; disconnected low ground reported, not flooded | `survey_response.flood` | embankment test: the hollow behind it stays dry |
| DIS-07 / CON-07 / BOR-05 | Relief, site and surveillance objects in the Plan catalogue: helipad, staging area, relief/medical tent, water bladder, generator, toilet, tower crane, site office, material yard, observation post, camera mast | `workspace_proposals.URBAN_OBJECTS` | proposals, never part of the scan |
| — | Operations tab + API: summary (epochs of the same site, designs, first map, raw detections), change, volume, damage (pre-event footprints from the earlier flight; linear features > 150 m or 8:1 excluded), detections, access, flood, design upload, cut/fill (scene or UTM design frame, progress vs a baseline flight), corridor (line + posts), tiles, field packs of the last result | `ops_scene.py`, `workspace_ops_api.py`, `ops-panel.tsx`, `lib/ops.ts` | 13 API tests; browser: every tool run on the ops site, overlays draped on the scan, packs and tiles downloaded |

**Found while verifying in the browser:** the change search picked a 2 m false shift because on
mostly-planar terrain a horizontal shift gives a *constant* dh whose NMAD is zero at every shift
— the score is now a trimmed mean absolute deviation with a tie-break to the smaller shift; the
12 m ridge was graded as a partially collapsed "building"; the Earth-coordinate lookup's
`height_m` overwrote building heights (364 m); a clicked vehicle start on a road edge was
refused instead of snapped; the tool hint sat at the top of a scrolled panel; synthetic cameras
lacked the frustum corners the viewer needs; a TIN pad without breaklines ramped its walls.

**Closed 2026-09-27:** M3C2 (P5 below); COG tiles (`dsm_cog.tif` through GDAL's COG driver
beside the verified plain GeoTIFF); epochs timeline (Twin tab, CON-04); design-model deviation
(CON-06: a GLB design in the scan's frame or ENU; signed distance along the design's normal,
proud / short / within tolerance draped on the scan - a B1 roof modelled 30 cm low reads 91%
proud, mean +0.25 m); crane jib clearance (CON-07: jib radius and clearance on a tower crane,
scanned structure inside the swing within the clearance is a plan violation - 115 m² flagged
for a short crane beside the 9 m block, none at full height).

**Still not built:** COPC-LAZ (no LAZ compressor installed; tiles stay LAS).

## P5. Phase 5 — inspection, archaeology, digital twin (2026-09-27)

Plan: `docs/APPLICATION_PLAYBOOKS.md` §3.6-3.8. Two workspace tabs, **Inspect** (Inspection /
Archaeology) and **Twin**, on `/api/workspace/inspect/*` and `/api/workspace/twin/*`. Verified
in the browser on `work/opsafter` (synthetic frames rendered from its splats) and `flatplan`.
**No real inspection or monument scan has been checked against tape, total station or a
reference model; `temple` has not been re-run through the survey lane on this hardware.**

| ID | Piece | Module | Evidence |
|---|---|---|---|
| INF-01 / M6 | Frames that saw a clicked point: projection into every recovered camera, sight line tested against the scanned top surface, ranked by range and centrality, with pixel, range and GSD | `inspection.frames_that_saw` | test: a point on a block's east wall is seen from the east camera and blocked from the west; browser: 2 of 2 frames with MGRS and thumbnails |
| INF-02 / ARC-03 | Defect / annotation register: type, severity, status, element, note, measurements; photo crop from the best frame at full resolution; revisioned (stale write refused); pins by severity on the scan | `inspection.py`, `inspect-panel.tsx` | tests round-trip, stale revision, delete; browser: added from a trace, photo shown |
| INF-06 | Crack candidates (heuristic): black-hat + Otsu + elongation on the photo, length in mm from the frame's GSD | `inspection.crack_candidates` | a 2 px line of 285 px found within 12 px, a blob ignored, a blank photo empty; browser: the roof edge is flagged, as the note warns |
| INF-08 | Inspection report PDF: register table, one page per item with photo, position (MGRS or local), frames, note | `inspection.report_pdf` | PDF from the API |
| INF-04 | Pole / tower tilt: iterative PCA axis in a cylinder at the clicked foot; tilt, lean bearing, mm per m, top offset | `inspect_geometry.tilt` | 3.0 deg at 90 deg recovered within 0.3 deg / 6 deg |
| INF-05 | Conductor sag and clearance: catenary fitted to the top points per metre in a slab, sag vs chord, lowest clearance to anything scanned below, against a limit | `inspect_geometry.wire` | sag within 8 cm of the true catenary; clearance to a tree within 0.6 m; failing limit flagged |
| INF-07 / ARC-07 / M3 | M3C2 change along local normals with LoD95 per core point; unobserved is not "no change"; defaults follow the cloud's point spacing | `survey_m3c2.py`, `inspect/m3c2` | a 4 cm bulge in a wall: > 90% significant in the patch, < 6% outside, median 4 cm; stockpile on the ops site reads "moved out" |
| INF-09 / CON-07 | Access objects (scaffold bay, scissor lift, boom lift) in the Plan catalogue | `workspace_proposals` | — |
| ARC-01 | Top-down ortho mosaic from the splats, ground and surface hillshade (multi-directional), local relief model (DTM - 15 m mean), slope; LRM draped on the scan | `inspect_geometry.terrain_rasters`, `inspect/terrain` | a 0.8 m ditch stands out in the LRM, the regional slope is removed; browser: pad edge, bank and ridge in the drape |
| ARC-02 | True-scale section along a line: points, upper outline, ground (dashed where unobserved), SVG and DXF R12 (POINTS / OUTLINE / GROUND / UNOBSERVED) | `inspect_geometry.section`, `section_svg`, `section_dxf` | box roof at 10.0 m within 5 cm; browser section through B1 |
| ARC-04 | Observed / unobserved in exports: section ground dashed, rasters transparent where never observed, M3C2 unobserved reported | as above | — |
| ARC-05 | Hypothesis reconstruction: a Plan scheme marked inferred (with its basis) is tinted and translucent in 3D and says "inferred hypothesis, not measured" in GeoJSON, CityJSON and 3D Tiles | `workspace_proposals.set_inferred`, `plan_exports`, `lib/plan.ts` | test: exports carry the status; browser badge |
| ARC-06 | Archive record: Dublin Core fields, capture, spatial reference, processing chain, SHA-256 fixity of every delivered file; fixity check | `provenance_record.py` | a tampered hash is reported as changed |
| TWN-01 | Visual / Measured / Evidence views in one frame (splats, collision surface, coverage) | `twin-panel.tsx` | browser: Measured shows the collision surface; Evidence disabled with its reason where there is no coverage layer |
| TWN-02 | Asset inventory (buildings, trees, poles) with measured facts, open defects linked, and owner attributes kept apart; orphaned attributes listed | `twin.py` | browser: "Block A / BLD-001" saved, 1 open defect linked |
| TWN-04 / CON-04 | Epochs of a site by capture date with the change summary against the previous one | `twin.epochs` | browser: before / after with +270 / -1978 m³ |
| TWN-05 / TWN-07 | Engine package: vertex-coloured surface GLB, FBX (geometry), 3D Tiles 1.1 quadtree (coarse root + 4 REPLACE children, ECEF-placed), splats in 50 m chunks with an index, README saying what is measured | `twin.package` | GLB carries COLOR_0; tileset has 4 children and a transform; browser download 6.4 MB / 25 files |

**Phase 0 closed:** the application registry (`scripts/applications.py`, playbook §4) - eight
profiles with lanes, dense profile, presets, required products, analyses, tabs and acceptance
checks; `--application` on `pipeline.py` (can override an auto preset, recorded in
`scenario.json` with who set it; GPS-pattern suggestions otherwise) and on `survey.py
reconstruct` (picks the dense profile when none is given; the delivery record marks formats the
profile did not request as "not requested", never "missing"); the project's application and site
id are saved from Details, and the workspace shows that application's tabs first (All tools for
the rest). A PR template carries the §5 production checklist.

**Not feasible here:** anything that needs the real flights (Phase 1 exit gate, every acceptance
demo), timing work on this GPU (deliberately left, as asked), COPC-LAZ, external CityJSON / 3D
Tiles validators, and headset verification of VR.

## H. Performance / real-time

| ID | Item | Type | Effort |
|----|------|------|--------|
| H1 | Progressive reconstruction (`survey_progressive`) is manual-flag only → stream a partial model as it builds | wire | M |
| H2 | The <15-min gate is documented unreachable at current budget → profile + a fast path | build | L |

## I. Product / platform polish

| ID | Item | Type | Effort |
|----|------|------|--------|
| I1 | Scene comparison / change-detection (two scans of the same place) | build | L |
| I2 | Share links / read-only viewer for a scan | build | M |
| I3 | Annotations with photos + report PDF export (inspection) | build | M |
| I4 | Batch processing queue (many scenes) | build | M |

---

## Application coverage (can a customer use it today?)

| Application | Usable now? | Blocking gaps |
|-------------|-------------|---------------|
| Rapid mapping / situational awareness | Mostly | D1 dynamics, H1 progressive |
| Infrastructure inspection | Partial | A1 real measurements, A5 trust, I3 report |
| Survey / construction (volumes, as-built) | Mostly (with GPS + checkpoints) | A2 volume from a real cloud, a real telemetry flight to prove F end to end |
| Interior design / furniture placement | Yes (primitives + imported real-size models) | B5 full room segmentation, B6 human-scale interior geometry |
| Gaming / FVP walkthrough | Partial (needs manual nav bake) | C1–C4 |
| Disaster response | Partial | D1, D5 coverage, A2 volume |
| Heritage / virtual tour | Mostly | C2 walk, I2 share |

---

## Recommended build order

1. **A1–A5** — measurement (huge, already built, just orphaned)
2. **C1** — auto-navmesh → unlocks walk + FVP games for every scene
3. **D1** — dynamic-object masking (already built, never run)
4. **E1 / B6** — fix interior scale → then **B1–B2** furniture placement
5. **D2 / D6** — learned depth + segmentation (needs weights; confirm before any download)

This turns existing-but-dead code into shipped capability first, then builds the genuinely new features.
