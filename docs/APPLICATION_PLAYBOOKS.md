# Application Playbooks: one engine, eight mission products

**Status date:** 2026-09-26 (revision 2: mission rehearsal kept and expanded, urban planning
editor added, detailed use cases per application, phased build plan)
**Scope:** the eight "Potential Applications" in SIH26158 (PDF, printed p. 37). For each one:
who uses it, what they do with it, every feature they need (Must / Should / Could), which
pipeline to run, and what to build.
**Companion docs:** `09_SIH26158_Evaluation_and_Improvement_Report_2026-09-22.md` (rubric and
evidence), `docs/GAPS_AND_OPTIMIZATIONS.md` (what is built, wired and verified).

**Contents**
0. Strategy: how the 100 points are won
1. Pipeline catalogue
2. Shared platform: reconstruction/GIS modules, the Scene Editing Engine, the Rehearsal Engine
3. The eight application playbooks (use cases + feature tables)
4. Mission-profile layer
5. The production-grade bar
6. Phased build plan (start here)
7. Data we must collect
8. What we will not claim

---

## 0. Strategy

### 0.1 How the 100 points are awarded

| Criterion | Weight | What wins it | Where the evidence comes from |
|---|---:|---|---|
| Reconstruction accuracy | **30** | Unaligned checkpoint RMSE (H / V / 3D, median, P95) on a **real** GPS-only flight, ≤ 1 m | Core pipeline + flights A–C (§7) |
| Model completeness | **20** | Per-class visible-surface recall (roof, facade, ground/road, vegetation, obstacle), holes mapped | Core pipeline, Urban + Twin reports |
| Processing speed | **20** | A real 600 s video → final products in **< 900 s**, all stages counted, hardware declared | Core pipeline (P-S), Disaster/Military progressive map |
| Innovation | **15** | Ablation-backed technical wins **and** capabilities nobody else shows | Mission rehearsal on a reconstructed battlefield, proposal editing on a measured city, provenance, KLV, 4-DoF alignment |
| Scalability | **10** | 1 / 5 / 10 min × 1080p / 4K matrix, bounded VRAM, tiled output | Border corridor, Twin LOD |
| User interface | **5** | Import → progress → inspect → measure → edit → export, production polish | Every application |

The judges score one engine on one dataset ("will be provided real time"). The applications are
how we prove the model is "suitable for visualization, measurement and analysis", where most of
the innovation marks come from, and what makes the demo memorable. **Both matter: the core
pipeline wins the 70%; the applications win innovation and UI and make the jury remember us.**

### 0.2 What stands between us and 100 (honest)

1. **No real drone flight with its GPS log exists on this machine.** Accuracy (30) and speed
   (20) stay "not demonstrated" until flights A–C are collected (§7). Nothing substitutes.
2. **Dense reconstruction must get ~3× faster.** A 10-minute mapping flight needs ≈ 270
   keyframes (`frames ≈ path / (footprint_along × (1 − overlap))`; 60 m AGL, 52° VFOV, 70%
   overlap, 8 m/s). At the measured `fast` rate of 6.6 s/image that is ~1,800 s for dense alone.
   The earlier 615–650 s projection assumed 68 frames from a room clip that revisits itself.
3. **≤ 1 m absolute with ordinary GPS is on the edge.** Consumer GNSS is ±1–3 m, worse vertically.
   Takeoff-anchored heights (M12) and clock-offset estimation are the cheapest fixes. RTK is a
   separate, declared track.
4. **Editing and rehearsal must never contaminate measured data.** Everything a user draws,
   places or scripts lives in a separate *proposal layer* (§2.2), visually and in every export.
   This is also a talking point: "measured vs proposed" is enforced by the data model.

---

## 1. Pipeline catalogue

Two lanes: `survey.py` (geometry-first, georeferenced — the lane that scores) and
`pipeline.py` (splat + collider + navmesh — the presentation / rehearsal lane).

| ID | Name | Command | Key settings | Produces | Time target (10-min 1080p) | Used by |
|---|---|---|---|---|---|---|
| **P-R** | Rapid | `survey.py reconstruct <scene> --allow-gpu --dense-profile budget --progressive` | baseline-aware keyframes, sequential + GPS-spatial pairs, pose priors, 700 px dense | progressive submaps, LAS/PLY, DSM, ortho (M1) | first measurable map ≤ 5 min; final ≤ 12 min | Disaster, Military, Border first look |
| **P-S** | Survey | `survey.py reconstruct <scene> --allow-gpu --dense-profile fast --vertical-datum egm96` | 1000 px dense, Poisson mesh, texture bake, full formats, UTM + EGM96 | OBJ/PLY/LAS/GeoTIFF/glTF/GLB/FBX, textured mesh, evidence map, accuracy report | **≤ 15 min (the official gate)** | Every scored run |
| **P-D** | Detail | `survey.py reconstruct <scene> --allow-gpu --dense-profile survey` | 1600 px, geometric consistency on, full-res texture | as P-S, cleaner surfaces | offline, 30–60 min, declared | Inspection, Archaeology, hero Twin |
| **P-T** | Terrain | P-S + ground classification (M2) | `drone_mapping` preset, DSM → DTM → nDSM, contours, hillshade | DSM/DTM/nDSM GeoTIFF, contours, slope | ≤ 15 min | Border, Construction, Archaeology, Military |
| **P-V** | Visual / Rehearsal | `pipeline.py run <scene> --quality high` **on P-S poses** | splat, collider, navmesh, human-scale body | splat PLY, collider GLB, `nav.json`, walk/arena | +10–15 min, declared as presentation layer | Mission rehearsal, Urban before/after, Twin |

**Rules for every profile**

- The P-S product is the only thing submitted as *measured*. P-V is always "presentation
  layer" and must render in the same georeferenced frame (§2.3 frame registry).
- The timed gate is run with P-R or P-S. P-D and P-V times are reported separately.
- `--progressive` is diagnostic today (nothing consumes the submaps). P-R needs it promoted:
  each merged window writes a georeferenced DSM/LAS the viewer loads while the run continues
  (GAPS H1).
- Every run records its mission profile in `work/<scene>/scenario.json` (GAPS E0) with the
  reason and who set each knob.

**Flight pattern → preset** (auto-detected from GPS, GAPS E2b): orbit → `drone` /
`outdoor_building`; lawnmower → `drone_mapping`; straight → `corridor` (needs M11 below —
`survey_georef._noncollinear` correctly refuses straight tracks today); mixed → `drone`.

---

## 2. Shared platform

Three shared layers. Build once; every application is a configuration of them.

### 2.1 Reconstruction and GIS modules (M)

| ID | Module | Status / builds on | Effort | Used by |
|---|---|---|:-:|---|
| **M1** | True orthomosaic GeoTIFF (COG): best-view per cell (`survey_texture` rule), DSM occlusion test, nodata where unseen | **NEW**; DSM exists (`survey_export.dsm_grid`) | M | BOR DIS URB CON ARC MIL |
| **M2** | Ground classification (CSF / progressive morphological) → DTM, nDSM, LAS class 2; "interpolated under canopy" mask | **NEW**; heuristic ground in `label_semantics` | M | BOR DIS URB CON ARC MIL TWN |
| **M3** | Two-epoch co-registration (ICP on stable classes) + DSM diff + M3C2-style distance with LoD95 | partial: `survey_measure.volume_between` | M | CON DIS BOR INF MIL |
| **M4** | Terrain analytics: contours, slope/aspect/roughness rasters, hillshade + LRM, profile along a line, LOS, viewshed with range + cone | point-wise slope/aspect exist | M | BOR MIL ARC CON URB |
| **M5** | Object extraction: building footprints + height + roof planes, tree crowns (CHM), road surface/centerline, poles | classes in `label_semantics`, `learned_semantics` | L | URB DIS MIL TWN INF |
| **M6** | Provenance: click a point → source frames, pixel, timestamp | `survey_evidence` per-point views | S | INF ARC MIL DIS all |
| **M7** | Telemetry ingest: MISB ST 0601 KLV (STANAG 4609 TS), MAVLink tlog, PX4 ULog, ArduPilot BIN | SRT/DJI CSV/GPX/CSV/IMU/baro exist in `survey_inputs` | M | MIL BOR |
| **M8** | Coordinates: MGRS, DMS, UTM, click → coordinate with CE90 / LE90 | `survey_measure.uncertainty` | S | MIL BOR DIS all |
| **M9** | Exports: KML/KMZ, GPX routes, CityJSON LOD1/2, 3D Tiles, COPC-LAZ, COG, PDF report | GeoJSON/CSV exist | M | all |
| **M10** | Georeferenced detections layer (people, vehicles) with frame + time, kept out of geometry | GroundingDINO step + `survey_dynamics` masks | S | MIL DIS BOR |
| **M11** | 4-DoF alignment (scale + translation + yaw) with gravity from gimbal/IMU or ground plane, for straight tracks | **NEW** | M | BOR MIL INF |
| **M12** | Takeoff-anchored vertical (relative baro altitude + ground height at t0) | **NEW** | S | all |
| **M13** | Fast dense path (≥ 3× over `fast`), §2.5 | **NEW** | L | all (speed gate) |

### 2.2 Scene Editing Engine (ED) — the generalised placement system

The furniture placement system (`scripts/workspace_place.py`, `placement-panel.tsx`,
`viewer/furniture_geometry.js`) already does the hard parts: snap to measured ground, fit
check against the coverage grid, glTF import with real bounds, persistence, collision boxes the
walker respects. **Generalise it into one engine that every application uses**; furniture
stays as one catalogue category (the indoor add-on).

**Data model — the proposal layer.** Nothing a user creates is ever written into measured data.

```
work/<scene>/proposals/
  index.json                      # proposals list, active proposal, base scene hash
  <proposal_id>.json              # one "scenario": Existing + a set of features
    { id, name, kind: "plan"|"mission"|"annotation",
      base: { scene_hash, frame: <frame registry id> },
      features: [ { id, type, geometry (ENU metres, z included), params, style,
                    layer, locked, created_by, created_at, updated_at } ],
      history: [ commands for undo/redo, capped ] }
```

Feature types: `road`, `building`, `object` (catalogue or glTF), `zone` (polygon with rules),
`clip` (hide existing geometry inside a prism — demolish/replace), `route` (waypoints),
`symbol` (tactical or annotation marker), `annotation` (note + photo + frames), `measurement`
(existing ones migrate here).

| ID | Capability | Details | Effort |
|---|---|---|:-:|
| ED-1 | Selection + transform gizmo | click/box select, move/rotate/scale gizmo (bundled PlayCanvas has `TranslateGizmo` etc.), numeric inputs in a properties panel | M |
| ED-2 | Snapping | to measured ground (DSM/heightfield), to surfaces (collider raycast), to grid, to existing features' vertices/edges, align to surface normal, right-angle drawing | M |
| ED-3 | Draw tools | point, polyline/spline, polygon, rectangle, circle; drape on terrain; vertex editing | M |
| ED-4 | Command history | undo/redo for every edit (command pattern), autosave debounced to the API, conflict-safe (base hash check) | M |
| ED-5 | Layers + visibility | per-type layers, lock, hide, isolate; measured vs proposed styling toggle | S |
| ED-6 | Object library | catalogue (street light, pole, transformer, tree species, bench, bus stop, barrier, bollard, water tank, cell tower, solar array, vehicles, containers, tents, sandbags, checkpoint, furniture) + glTF/GLB import (exists); duplicate, array along a line at spacing | M |
| ED-7 | Clip volumes | hide reconstructed splats/mesh inside a polygon prism (shader discard) for demolish/replace and before/after | M |
| ED-8 | Conflict checks | proposed object vs reconstructed geometry (collider), vs other proposals, vs zones/rules | M |
| ED-9 | Proposal management | create / duplicate / rename / compare proposals A-B-C; lock a proposal for review | S |
| ED-10 | Export | GeoJSON, KMZ, CityJSON (buildings), DXF (roads/footprints), glTF of the proposal, all tagged "proposed, not measured" | M |

Backend: `scripts/workspace_proposals.py` (validate, CRUD, rules, metrics, export) + routes
`/api/workspace/proposals*`, migrated from `workspace_place.py`. Viewer: `viewer/editor/`
(selection, gizmo, snap, draw, generators, clip). App: new workspace tabs **Plan** and
**Mission** next to Explore / Measure / Walkthrough.

### 2.3 Frame registry (prerequisite for ED and RH)

Today the splat lane lives in a Y-up viewer frame scaled by `frame.json`, and the survey lane in
ENU/UTM via the alignment. Editing, routes, tactical symbols and exports all need **one
transform chain**: `viewer ↔ ENU ↔ UTM ↔ WGS84 (+ MGRS)`, stored once per scene, tested for
round-trip error (< 1 mm), with the scale source (`gps`, `ar_pose`, `anchored`) carried along so
every feature knows whether its metres are measured. **Build this first** (`scripts/scene_frames.py`).

### 2.4 Rehearsal Engine (RH) — the bot game becomes tactical mission rehearsal

What exists (`viewer/pc/scripts/`): `combat.js` (mode manager), `bot.js` (state machine,
sight cones), `nav.js` (triangle-graph navmesh + A*), `world.js` (effects, projectiles, line of
sight), `character.js` (capsule controller), `hud.js`, `weapon.js`, `audio.js`; human-scale body
verified (GAPS E1); navmesh baked in the pipeline (C1). `viewer/xr_probe.html` checks WebXR
support. It is a working arena with random bot posts. **What rehearsal adds is that the scenario
comes from the mission plan**, the terrain is the real reconstructed objective, and every run is
recorded for review.

| ID | Capability | Details | Effort |
|---|---|---|:-:|
| RH-1 | Scenario from plan | bots spawn at the `symbol` features marked hostile in the active mission proposal, with the facing, sector of fire, behaviour and alert radius set on the marker | M |
| RH-2 | Bot behaviours | static sentry (sector scan), patrol along a drawn route, overwatch (long range, narrow cone), reaction force (moves to noise), vehicle (on roads); difficulty presets | M |
| RH-3 | Waypoint navigation | player follows the planned route: HUD compass tape, next-waypoint bearing/distance/ETA, phase-line alerts; free manual navigation always allowed; off-route warning | M |
| RH-4 | Exposure metering | live "seen by" indicator per enemy, cumulative exposure seconds, dead-ground highlight | S |
| RH-5 | Conditions | time of day + sun position, night with NVG/thermal-style filter, fog/haze visibility range (bots' view range follows) | M |
| RH-6 | Recording + After-Action Review | record every entity's pose at 10 Hz + events (spotted, fired, hit, waypoint reached); AAR view: top-down replay with timeline scrubber, paths, sight lines at each moment, exposure chart, per-phase timings; export AAR PDF | L |
| RH-7 | VR (WebXR) | Quest browser or PC VR over LAN HTTPS (`_cert.pem` exists); teleport + smooth locomotion, snap turn, controller aim, wrist map with waypoints; comfort vignette; **rendered from decimated textured mesh + ≤ 500k-splat LOD** to hold 72 fps | L |
| RH-8 | Sand-table mode | top-down miniature of the objective for briefings; projector/fullscreen; symbols and routes animated by phase; (Could) AR tabletop | M |
| RH-9 | Multi-user session | several players in one rehearsal (WebSocket state sync on the local server), instructor view with god camera, instructor can move enemies live | L |
| RH-10 | Safety and realism labels | HUD always shows "terrain from reconstruction dated <capture time>, unobserved areas hatched"; unobserved regions are visibly different in VR, never presented as ground truth | S |

### 2.5 M13: making dense fast enough (the speed criterion lives here)

Ablate on the same frames and poses; adopt only what keeps completeness/accuracy within gates.

| Lever | Expected | How | Gate |
|---|---|---|---|
| Cap source images per reference view | ~2× | patch-match.cfg `__auto__, 20` → `__auto__, 6–8`, chosen by baseline angle | recall drop < 3 pts |
| Fewer PatchMatch iterations / samples | 1.3–1.6× | `num_iterations 5→3`, `num_samples 15→8` | P95 error unchanged within 10% |
| Depth on alternate references, fused to all | ~2× | reference every 2nd keyframe, neighbours as sources | seam error |
| Sparse-anchored monocular depth (MoGe-2 in `depth_prior.py`) | 10×+ | per-frame scale+shift fit to COLMAP points, reject bad fits, TSDF fuse | must match PatchMatch on held-out checkpoints |
| 2.5D DSM for nadir | ~3× | `drone_mapping`: DSM straight from depth maps | nadir only |
| Pipeline dense with sparse | −20–30% wall-clock | submaps start dense when their window closes | 6 GB VRAM |

**Target stage budget, P-S, 10-min flight, ~270 keyframes** (to be measured): decode + selection
60 s · telemetry sync 10 · features + matching 120 · mapping 90 · dense 390 · fusion + mesh 60 ·
georef + exports + ortho + DSM 90 · evaluation 30 = **850 s**.

---

## 3. The eight application playbooks

Every playbook: **users → use cases → feature table → pipeline → acceptance demo**.
Feature priority: **Must** (in the demo), **Should** (in the product), **Could** (if time).
IDs are referenced by the phase plan in §6.

---

### 3.1 Military reconnaissance and mission planning (hero application)

**Users:** section/platoon commanders, intelligence cells, special forces planners, instructors.
**Mission reality:** one pass over an objective (often an ISR camera with KLV metadata), air-gapped
laptop, hours not days to plan, soldiers who have never seen the ground.

**Use cases**

1. **Target coordinate:** an analyst clicks a building corner → MGRS + height + CE90/LE90 in
   seconds, copies it into a fire-support request.
2. **Where might the enemy be:** the planner marks observed enemy (from video detections, M10)
   and asks the system for *likely* positions: high ground, rooftops and windows that dominate
   the approach routes with concealment nearby. A threat heatmap appears, labelled "heuristic".
3. **Plan the approach:** the commander draws routes with waypoints, phase lines, rally points,
   objective, support-by-fire position, and HLZ; the system computes each route's length, ETA
   (slope-aware), exposure seconds to each enemy position, and dead ground; suggests the most
   covered route.
4. **Rehearse it:** soldiers walk the actual route in first person (desktop or VR), following the
   waypoints; enemy bots sit exactly where the plan marked them with the marked sectors. Night
   rehearsal with NVG filter.
5. **Review it:** after-action replay from above: who was seen, when, from where; exposure per
   phase; plan adjusted; rehearse again.
6. **Brief it:** sand-table view for the orders group; export mission pack (KMZ, GPX routes for
   handheld GPS, briefing PDF with ortho, overlays, timings).

**Features**

| ID | Feature | Pri | Builds on |
|---|---|:-:|---|
| MIL-01 | Click → MGRS/lat-lon/UTM + height + CE90/LE90, copy button | Must | M8, uncertainty budget |
| MIL-02 | Tactical symbology (MIL-STD-2525 / APP-6 style: hostile, friendly, unknown, objective, obstacle) as `symbol` features with facing + sector arcs | Must | ED, `milsymbol` JS library (MIT — verify licence) |
| MIL-03 | Detections from video as candidate hostile markers (time + frame thumbnail) | Must | M10, M6 |
| MIL-04 | Threat prediction heatmap: per cell, visibility of friendly approach × concealment × elevation advantage | Should | M4 viewshed, M2 nDSM, M5 |
| MIL-05 | Routes with waypoints, phase lines, rally points, objective, HLZ, support-by-fire position | Must | ED-3 `route` |
| MIL-06 | Route analysis: length, slope-aware ETA (Tobler's hiking function), per-enemy exposure seconds, dead ground, covered-route suggestion (A* with exposure + slope cost) | Must | M4, `nav.js`, `walk_path_from_glb` |
| MIL-07 | LOS tool (point-to-point) and viewshed from any point with range + cone | Must | M4 |
| MIL-08 | HLZ finder: diameter ≥ N m, slope < 7°, no obstacle > 0.5 m, clear approach sector | Should | M2, M4 |
| MIL-09 | Obstacle list: structures/trees/poles above a height threshold, with coordinates | Should | M2 nDSM, M5 |
| MIL-10 | Vehicle trafficability raster (slope + roughness + road class) | Could | M4, M5 |
| MIL-11 | Rehearsal: scenario from plan, bot behaviours, waypoint HUD, exposure meter, conditions | Must | RH-1..RH-5 |
| MIL-12 | After-action review with replay + exposure chart + AAR PDF | Must | RH-6 |
| MIL-13 | VR rehearsal | Should | RH-7 |
| MIL-14 | Sand-table briefing mode | Should | RH-8 |
| MIL-15 | Multi-user rehearsal with instructor | Could | RH-9 |
| MIL-16 | KLV (STANAG 4609) ingest: the video carries its own GPS + attitude | Should | M7 |
| MIL-17 | Straight-pass alignment | Must | M11 |
| MIL-18 | GPS-denied fallback: visual-only + one known point or baro scale; labelled "local, not georeferenced" | Could | `solve_frame` rulers |
| MIL-19 | Change vs previous pass (new vehicles, earthworks) | Should | M3 |
| MIL-20 | Mission pack export: KMZ overlays, GPX routes, briefing PDF | Must | M9 |
| MIL-21 | Security: offline only, no CDN, encrypted bundle option, audit log of edits/measurements | Should | G3 bundle |

**Pipeline:** P-R first (≤ 5 min to first coordinates), P-S final, P-V for rehearsal.
**Acceptance demo:** on the 10-min video — time to first target coordinate; target coordinates vs
checkpoints (CE90 computed); a planted "enemy" (person/vehicle) appears as a detection; route
exposure computed; a full rehearsal run on the route with AAR replay; the same run in VR.

---

### 3.2 Urban planning and smart cities (hero application)

**Users:** town planners, development authorities, municipal engineers, smart-city SPVs,
architects presenting to committees.
**Mission reality:** orbit/oblique flights over blocks; decisions are about *proposed* change:
new roads, new buildings, redevelopment; committees need to *see* before/after and see rule
compliance.

**Use cases**

1. **Existing city inventory:** building footprints, heights, storeys, roof area and slope
   (solar), tree canopy and heights, road widths — on a measured, textured 3D model.
2. **Propose a road:** draw a centreline on the terrain, set width/lanes/footpath/median; the
   system drapes or grades it (max gradient), computes cut/fill, flags every building and tree
   the corridor cuts (land acquisition + tree count), and exports the alignment.
3. **Propose a building:** draw a plot and footprint (or pick a template), set floors and floor
   height, drag the roof up/down, move/rotate it; the system checks height limit, FSI/FAR,
   ground coverage, setbacks from plot and road against a configurable rule set, and the
   shadow it casts on neighbours.
4. **Redevelop:** select an existing reconstructed building → demolish (clip volume hides it) →
   place the replacement.
5. **Before/after:** swipe slider, toggle, side-by-side synced cameras; a metrics table
   (built-up area, FSI, green cover, impervious area, estimated population, parking demand,
   shadow hours) for Existing vs Proposal A vs B; record a flythrough video for the committee.
6. **Place infrastructure:** street lights every 30 m along the new road, a transformer, bus
   stops, trees — directly in the reconstructed scene, with clearance checks.

**Features**

| ID | Feature | Pri | Builds on |
|---|---|:-:|---|
| URB-01 | Building footprints + height + storeys + roof planes (area/slope/aspect) | Must | M5, M2 |
| URB-02 | Canopy layer: cover %, tree heights (CHM), tree count | Should | M2, M5 |
| URB-03 | Road width at a section, road surface layer | Should | M5, measure tools |
| URB-04 | Solar potential per roof (area × slope/aspect × irradiance table) | Could | URB-01 |
| URB-05 | **Road placement**: centreline spline, width, lanes, footpath, median, material; drape vs graded with max gradient; ribbon mesh generator | Must | ED-3, generator |
| URB-06 | Road impact: cut/fill volume, intersected buildings (area), trees removed, alignment export (GeoJSON/DXF) | Must | `volume_between`, M5 |
| URB-07 | **Building placement/editing**: draw footprint or template, floors × floor height, roof type (flat/gable/hip), drag-to-height, move/rotate/scale gizmo, procedural facade material | Must | ED-1..3, extrusion generator |
| URB-08 | Rule checks: height limit, FSI/FAR, ground coverage, setbacks — configurable rule set per city, violations highlighted on the model | Must | `zone` features, ED-8 |
| URB-09 | Shadow study: sun position for date/time, shadow cast on terrain and neighbours, shadow-hours on a chosen day | Should | viewer lighting + shadow maps |
| URB-10 | Demolish/replace existing building | Must | ED-7 clip volumes |
| URB-11 | **Before/after**: toggle, swipe, side-by-side synced, flicker; proposal A/B/C | Must | ED-9, ED-7 |
| URB-12 | Metrics diff table: built-up area, FSI, green cover, impervious area, population estimate, parking demand, shadow hours | Must | URB-01/02/08/09 |
| URB-13 | **General 3D object editing**: library + glTF, snap to ground, align to normal, array along line, clearance check | Must | ED-2, ED-6, ED-8 |
| URB-14 | Cadastral / master-plan overlay (import GeoJSON/KML/DXF plot boundaries); encroachment check | Should | ED import |
| URB-15 | Presentation: saved viewpoints, flythrough camera path, video export | Should | viewer camera |
| URB-16 | Exports: CityJSON LOD1/2 (existing + proposed, flagged), 3D Tiles, glTF, DXF | Should | M9, ED-10 |
| URB-17 | Per-building facade completeness (observed / weak / unobserved), unseen facades hatched | Must | `survey_occlusion` |

**Pipeline:** P-S (`outdoor_building` / `drone`), P-V for the photoreal before/after backdrop.
**Acceptance demo:** 5 building heights vs rangefinder within 0.5 m; draw a road through the
block → impact table; place a 12-storey building → FSI and setback violation shown, fixed by
dragging; swipe before/after; per-class completeness table.

---

### 3.3 Disaster damage assessment

**Users:** NDRF/SDRF teams, district disaster management authorities, insurers, relief NGOs.
**Mission reality:** minutes matter; smoke, dust, shadows, moving rescuers; field laptop, poor
connectivity; often no pre-event 3D data.

**Use cases**

1. **First map in minutes:** progressive map while the drone is still being recovered; ortho first.
2. **Where is the worst damage:** buildings graded intact / partial / collapsed; debris volume.
3. **Can we get in:** blocked roads highlighted, vehicle route computed from staging area to site.
4. **Flood:** water extent and approximate depth from DTM; affected buildings list.
5. **Plan relief:** place relief camp tents, water tanks, helipad, staging area as objects; mark
   hazard zones; share a KMZ with field teams.
6. **Report:** one-page situation report PDF with ortho, counts, coordinates, capture time.

| ID | Feature | Pri | Builds on |
|---|---|:-:|---|
| DIS-01 | Progressive map (first ortho/DSM ≤ 5 min), "time since capture" on every layer | Must | P-R, H1 |
| DIS-02 | Building damage grade from roof planarity + nDSM height loss (+ pre-event DSM / OSM heights when given) | Must | M2, M5 |
| DIS-03 | Debris / landslide volume | Must | `volume_between` |
| DIS-04 | Blocked roads (road class with obstacles > 0.5 m) + vehicle route with a vehicle body | Should | M5, router |
| DIS-05 | Flood extent + depth estimate | Could | M2, water class |
| DIS-06 | People/vehicle detections as a separate layer (rescue value; never in geometry) | Must | M10 |
| DIS-07 | Relief planning with objects (tents, tanks, helipad) + hazard zones | Should | ED-6, `zone` |
| DIS-08 | Pre/post change map when earlier data exists | Should | M3 |
| DIS-09 | Field pack: KMZ + small COG overview + situation-report PDF | Must | M9 |

**Pipeline:** P-R with `--progressive`, `--dynamics force`, `--photometric auto`; P-S in background.
**Acceptance demo:** time-to-first-map and time-to-final on the 10-min video; a debris pile volume
within 10% of a tape-measured reference; people absent from the mesh, present as points.

---

### 3.4 Construction progress monitoring

**Users:** project managers, site engineers, quantity surveyors, lenders' engineers.
**Mission reality:** the same site flown weekly; volumes and progress vs plan matter more than
photorealism; relative accuracy between epochs matters more than absolute.

**Use cases**

1. **Stockpile volumes** every week with uncertainty.
2. **Cut/fill vs design** surface (earthworks).
3. **What changed this week:** change heatmap above level of detection.
4. **Progress %** per zone vs design volume / planned structure.
5. **As-built vs design:** place the design model (glTF/IFC-derived) and see deviations.
6. **Logistics:** place crane, site office, material yards as objects; check crane radius clearance.

| ID | Feature | Pri | Builds on |
|---|---|:-:|---|
| CON-01 | Stockpile volume: draw outline → base fit → volume ± uncertainty | Must | `survey_measure` (A2) |
| CON-02 | Design surface import (GeoTIFF DEM, LandXML TIN, DXF 3DFACE) + cut/fill map | Must | `volume_between` |
| CON-03 | Epoch co-registration + change heatmap with LoD95 | Must | M3 |
| CON-04 | Epoch timeline slider | Should | proposals / scenes list |
| CON-05 | Zone progress % | Should | `zone` features |
| CON-06 | Design model overlay (glTF) + deviation colour map | Should | ED-6, M3 distance |
| CON-07 | Site logistics objects (crane with radius, office, yard) + clearance | Could | ED-6, ED-8 |
| CON-08 | Volume/progress report PDF + CSV | Must | M9 |

**Pipeline:** P-T with `drone_mapping`; P-S for orbit captures.
**Acceptance demo:** two flights of the same site with a moved pile/boxes: change detected above
LoD, unchanged ground below; pile volume vs geometric reference within 5%.

---

### 3.5 Border and strategic area mapping

**Users:** border guarding forces, strategic planners, infrastructure agencies on borders.
**Mission reality:** long straight flights along a fence/LoC, 80–150 m AGL, sometimes oblique;
questions are *what changed, where exactly, what can be seen from where*.

**Use cases**

1. **Map a 5 km stretch** into tiled ortho + DSM/DTM + LAS.
2. **Detect change** since the last pass: new structures, trenches, tracks, fence breaches,
   vegetation clearing.
3. **Terrain profile** along the border line; **viewshed** from each post to find blind spots.
4. **Plan surveillance:** place observation posts, cameras (with FoV cones), sensors, patrol
   routes; see covered vs uncovered stretch.
5. **Report** changes with coordinates to HQ as KMZ.

| ID | Feature | Pri | Builds on |
|---|---|:-:|---|
| BOR-01 | Straight-pass alignment | Must | M11 |
| BOR-02 | Tiled products every ~1 km (COG + COPC-LAZ), bounded memory | Must | `survey_progressive`, M9 |
| BOR-03 | Change detection with LoD, change polygons with MGRS | Must | M3, M8 |
| BOR-04 | Profile along a line; viewshed from posts; blind-spot map along the border | Must | M4 |
| BOR-05 | Surveillance planning: posts, cameras with FoV cone, sensors; coverage % of the line | Should | ED-6, M4 |
| BOR-06 | Patrol route planning with ETA | Could | MIL-05/06 |
| BOR-07 | KLV / MAVLink ingest | Should | M7 |
| BOR-08 | Offline basemap (no internet tiles) | Should | viewer |

**Pipeline:** P-T with `corridor`, P-R for first look.
**Acceptance demo:** ≥ 1 km corridor aligned by M11 where 7-DoF refuses; CE90 vs checkpoints;
a planted change found, unchanged ground clean; blind-spot map from two posts.

---

### 3.6 Infrastructure inspection

**Users:** bridge/tower/powerline/pipeline/dam inspectors, utilities, railways, NHAI.
**Mission reality:** close range (10–30 m), relative accuracy in cm, and the inspector must get
back to the photo of a defect.

**Use cases**

1. **Walk around the asset** at full texture; spot a crack.
2. **Click the defect → see the best 3 source frames** at full resolution.
3. **Annotate** it (type, severity, photo, note); build a defect register.
4. **Measure**: crack length, spalling area, tower tilt, clearance of wires to trees.
5. **Compare** with last inspection (deformation, new defects).
6. **Plan repairs:** place scaffolding/access platform objects; export the inspection report.

| ID | Feature | Pri | Builds on |
|---|---|:-:|---|
| INF-01 | "Frames that saw this" strip for any clicked point | Must | M6 |
| INF-02 | Defect annotation (type, severity, photo crop, note) + register | Must | ED `annotation` |
| INF-03 | Distance, area, point-to-plane (exist) with relative uncertainty | Must | `survey_measure` |
| INF-04 | Verticality/tilt of towers and poles (axis fit) | Should | M5 poles |
| INF-05 | Wire extraction (catenary fit, keep thin low-support points) + clearance to vegetation | Should | M5 |
| INF-06 | Crack/corrosion detection on frames, projected to 3D | Could | ⚠dep small segmenter |
| INF-07 | Deformation vs baseline epoch | Should | M3 |
| INF-08 | Inspection report PDF | Must | M9 |
| INF-09 | Access planning objects (scaffold, lift) | Could | ED-6 |

**Pipeline:** P-D delivered, P-S for the timed run; `corridor` for linear assets, orbit for towers.
**Acceptance demo:** 5 tape-measured lengths, relative error < 2%; a defect traced to its frames;
report generated.

---

### 3.7 Archaeological documentation

**Users:** ASI circles, state archaeology departments, universities, conservation architects.
**Mission reality:** archive-grade record; faithful texture, microtopography, sections; **no
invented geometry**. The repo's `temple` scene is a real-footage demo already on disk.

**Use cases**

1. **Record the monument** as a textured mesh with provenance and metadata.
2. **Reveal subtle earthworks** with hillshade and local relief model.
3. **Cut a section** anywhere, export it as a drawing.
4. **Annotate** features (inscriptions, damage) with photos.
5. **Hypothesis reconstruction:** place a proposed restoration (e.g., missing shikhara) as a
   clearly labelled *inferred* proposal, toggle it on/off for public presentation.
6. **Monitor** erosion/damage between seasons.

| ID | Feature | Pri | Builds on |
|---|---|:-:|---|
| ARC-01 | Textured mesh + ortho + DTM hillshade + LRM | Must | M1, M2, M4 |
| ARC-02 | Section tool along a line, export SVG/DXF | Must | M4 profile |
| ARC-03 | Annotations with photos and frames | Should | ED `annotation`, M6 |
| ARC-04 | Observed / weak / unobserved layers in every export | Must | `survey_occlusion` |
| ARC-05 | Hypothesis reconstruction as a labelled proposal | Could | ED, ED-9 |
| ARC-06 | Metadata/provenance sidecar (who, when, device, CRS, chain, hashes) | Should | `manifest.json` |
| ARC-07 | Seasonal comparison | Could | M3 |

**Pipeline:** P-D for the monument, P-T for the site; P-V for a public virtual visit.
**Acceptance demo:** `temple`: textured mesh, hillshade, a section, unobserved regions marked,
hold-out texture check (+21% edge agreement already measured, GAPS G1).

---

### 3.8 Digital twin generation

**Users:** campus/estate/facility managers, smart-city command centres, simulation teams.
**Mission reality:** a living model with looks *and* truth, objects with attributes, and an
update path. It is the integration of everything above.

**Use cases**

1. **Navigate** the site photoreal (splat) and switch to the measured mesh to measure.
2. **Query assets:** click a building/pole/tree → attribute card (height, area, ID, notes).
3. **Plan changes** with the editing engine (proposals), compare before/after.
4. **Update** the twin by re-flying: merge the new flight, highlight changes.
5. **Hand off** to Unity/Unreal/Cesium (FBX/glTF/3D Tiles).

| ID | Feature | Pri | Builds on |
|---|---|:-:|---|
| TWN-01 | Visual (splat) ↔ measured (mesh) ↔ evidence (confidence) toggle in one frame | Must | §2.3 frame registry |
| TWN-02 | Object inventory with IDs + attribute cards | Should | M5 |
| TWN-03 | Proposal editing and before/after (shared with URB) | Must | ED |
| TWN-04 | Update merge from a new flight + change highlight | Should | M3 |
| TWN-05 | LOD / streaming: 3D Tiles for mesh, chunked splats | Should | M9 |
| TWN-06 | Walk/fly at human scale | Must | GAPS E1, C2 |
| TWN-07 | Engine exports (FBX/glTF/3D Tiles) | Must | survey formats, M9 |
| TWN-08 | Indoor add-on: furniture placement for interior scans | Could | existing B1–B4 |

---

## 4. Mission-profile layer

- **`scripts/applications.py` (NEW, S):** registry, one entry per application:
  `{label, lanes, dense_profile, preset_candidates, required_products, analyses,
  workspace_tabs, viewer_tools, acceptance_checks, not_needed}`. It only selects existing knobs.
- **`--application <id>`** on `survey.py` and `pipeline.py`, recorded in `scenario.json` as
  `application:<id>` in the "who set it" column the Quality tab already shows.
- **Auto-suggest** from the flight pattern (orbit → URB/INF/TWN/ARC; grid → CON/DIS; corridor →
  BOR/MIL/INF-linear). A suggestion, never silent.
- **Product gating:** `survey_deliver` runs only `required_products`; the format ledger marks
  skipped formats "not requested by profile", never "missing".
- **Workspace:** the application decides which tabs appear (e.g. MIL: Explore · Measure ·
  Mission · Rehearse · Exports; URB: Explore · Measure · Plan · Compare · Exports).

---

## 5. The production-grade bar

"Production grade" is a checklist, applied to every feature before it is called done.

**Experience**
- One design system (tokens, type scale, icon set, spacing) across `groundcontrol` and the viewer
  HUDs; no developer chrome, no raw JSON, no console-style text in the product.
- Every panel has loading, empty, error and "not measurable" states, each with a next action.
- Undo/redo and autosave on every edit; nothing is lost on refresh.
- Keyboard shortcuts for every tool (shown in tooltips); a `?` shortcut sheet.
- Units and coordinate system always visible; measured vs proposed always visually distinct.
- Works at 1366×768 (field laptops) and on a 4K display; light and dark themes; focus rings and
  contrast AA.
- Onboarding: first-run guided tour per application; sample project bundled.

**Performance budgets** (measured in CI on the RTX 3050 laptop)
- Workspace interactive < 3 s after load for a 3M-splat scene; tool response < 100 ms.
- ≥ 45 fps desktop viewer; ≥ 72 fps in VR (decimated mesh + splat LOD).
- API responses < 300 ms except declared long jobs, which stream progress.

**Reliability**
- Every pipeline run resumable; every failure names the stage, the cause and the remedy.
- No silent fallbacks: a refused step is shown as refused with its reason.
- Proposal files validated on load; a corrupt file is quarantined, not crashing the workspace.

**Security and deployment**
- Fully offline: no CDN, no internet tiles, no telemetry leaving the machine.
- Local users + roles (viewer / analyst / planner / instructor); audit log of edits and
  measurements; optional encrypted bundle export.
- One-command install (`setup` script or Docker) and a `doctor` check: GPU, driver, CUDA,
  COLMAP, disk space, certificates.

**Quality engineering**
- Unit tests for every new module (the repo standard); browser end-to-end test for each
  application's golden path (`tests/browser_workspace.py` pattern); screenshot regression for
  key views; the full CPU suite green before merge.
- Every demo number regenerated by a script from raw data, never typed by hand.

---

## 6. Phased build plan (start here)

### 6.1 Tracks

Work runs in parallel tracks so the scoring core is never starved by feature work.

| Track | Owns | Skills |
|---|---|---|
| **T1 Core** | pipeline speed, georeferencing, M11–M13, gate runs | Python, COLMAP, GPU |
| **T2 GIS** | M1–M5, M8–M10, analytics, exports, reports | Python, NumPy, GDAL/laspy |
| **T3 Editor** | frame registry client, ED engine, Plan/Compare UI, application picker | TypeScript, PlayCanvas, Next.js |
| **T4 Rehearsal** | RH engine, Mission tab, AAR, VR | JavaScript, PlayCanvas, WebXR |
| **T5 Data & QA** | flights, checkpoints, evaluation runs, e2e tests, demo script | field work, Python |

### 6.2 Phases

Week numbers are relative to kickoff; re-plan once the finale date is confirmed. Each phase has
an **exit gate** — advance on evidence, not on features merged.

#### Phase 0 — Foundations (weeks 1–2)

| Task | Track | Output |
|---|---|---|
| Fly flights A, B, C with SRT/logs; survey checkpoints | T5 | raw data hashed in `videos/`, checkpoint CSVs withheld |
| `scripts/scene_frames.py`: viewer ↔ ENU ↔ UTM ↔ WGS84 ↔ MGRS registry + tests | T2 | one transform per scene, < 1 mm round trip |
| `scripts/workspace_proposals.py` + `/api/workspace/proposals*`: schema, CRUD, validation, migration of placements | T3 | proposals persist; old placements load as `object` features |
| Viewer editor core: selection, gizmo, snapping to ground, command history (ED-1, ED-2, ED-4) | T3 | move/rotate/scale any feature with undo |
| `scripts/applications.py` registry + `--application` flag | T1 | scenario.json records the application |
| Design-system pass + production checklist in PR template | T3 | tokens, empty/error states pattern |
| Rehearsal refactor: combat reads a scenario object instead of random posts (RH-1 skeleton) | T4 | bots spawn from a JSON list |

**Exit gate:** real flight data on disk; a feature created in the viewer round-trips through the
API to WGS84 and back; bots spawn from a file.

#### Phase 1 — Scoring core (weeks 2–5)

| Task | Track | Output |
|---|---|---|
| M13 dense ablation (source-view cap, iterations, alternate refs, sparse-anchored MoGe) | T1 | ablation table, chosen setting |
| M12 takeoff vertical + clock offset on real data | T1 | vertical RMSE before/after |
| Full 10-min P-S run on flight A, cold and warm | T1/T5 | timed report |
| Unaligned checkpoint evaluation on A, B, C | T5 | H/V/3D RMSE, P95, CE90 |
| M1 ortho, M2 DTM, M8 coordinates | T2 | ortho/DTM open in QGIS |
| M11 4-DoF alignment on flight C | T1 | corridor aligned where 7-DoF refuses |

**Exit gate:** 600 s video < 900 s end-to-end on declared hardware; unaligned accuracy reported on
three real flights (whatever the number is).

#### Phase 2 — Urban planning editor (weeks 3–7, parallel with Phase 1)

| Task | Track | Output |
|---|---|---|
| Draw tools (ED-3), layers (ED-5), object library + arrays (ED-6) | T3 | URB-13 |
| Road generator: spline, width/lanes, drape/graded, ribbon mesh | T3 | URB-05 |
| Road impact: cut/fill, intersected buildings, trees | T2 | URB-06 |
| Building generator: footprint/template extrusion, floors, roof types, drag-height | T3 | URB-07 |
| Rule engine: height, FSI, coverage, setbacks; configurable rule set | T2 | URB-08 |
| Clip volumes for splat and mesh (demolish/replace) | T3 | URB-10 |
| Compare view: toggle, swipe, side-by-side, A/B/C + metrics diff | T3/T2 | URB-11, URB-12 |
| M5 building footprints + heights | T2 | URB-01 |

**Exit gate:** on flight B: draw a road → impact table; place and edit a building → rule
violations update live; demolish + replace; swipe before/after; metrics diff for two proposals.

#### Phase 3 — Mission planning and rehearsal (weeks 4–9, parallel)

| Task | Track | Output |
|---|---|---|
| Mission tab: tactical symbols with facing/sectors, routes, phase lines, HLZ (MIL-02, MIL-05) | T4/T3 | plan persisted as a `mission` proposal |
| M4 LOS + viewshed; route analysis with exposure + Tobler ETA + covered-route suggestion (MIL-06/07) | T2 | route report |
| Detections → candidate hostile markers (MIL-03) | T2 | M10 layer |
| Bot behaviours: sentry, patrol, overwatch, reaction (RH-2) | T4 | behaviours selectable per marker |
| Waypoint HUD, exposure meter, conditions/NVG (RH-3..5) | T4 | rehearsal playable on the plan |
| Recording + AAR replay + AAR PDF (RH-6) | T4 | MIL-12 |
| VR (RH-7) with mesh/splat LOD | T4 | 72 fps on target headset |
| Threat heatmap, HLZ finder, sand table (MIL-04/08/14) | T2/T4 | Should items |
| Mission pack export (MIL-20) | T2 | KMZ + GPX + PDF |

**Exit gate:** on flight B or C: plan with ≥ 3 hostile markers and a route → exposure report →
rehearsal run following waypoints → AAR replay; same scenario in VR at ≥ 72 fps.

#### Phase 4 — Operations applications (weeks 6–10)

Disaster (DIS-01..09 Musts: progressive map, damage grade, debris volume, detections, field pack),
Construction (CON-01/02/03/08: volumes, design cut/fill, epoch change, report), Border
(BOR-01..04: tiled corridor, change, viewshed/blind spots).
**Exit gate:** each application's acceptance demo passes on real flights.

#### Phase 5 — Inspection, Archaeology, Twin (weeks 8–11)

INF-01/02/03/08, ARC-01/02/04 on `temple`, TWN-01/03/06/07.
**Exit gate:** each acceptance demo passes; `temple` archaeology demo complete.

#### Phase 6 — Production hardening and the jury demo (weeks 10–12)

| Task | Output |
|---|---|
| Scalability matrix: 1/5/10 min × 1080p/4K on P-R and P-S; peak VRAM/RAM; failures disclosed | table for Scalability (10) |
| Performance budgets enforced; VR frame-time profiling | budgets green |
| Installer + `doctor`; offline audit (no external requests) | one-command setup |
| Security: roles, audit log, encrypted bundle | MIL-21 |
| Onboarding tours, sample projects, operator one-pagers per application | UI (5) |
| **Unknown-data drill:** run a never-seen video end-to-end in front of a timer, twice | proof the live evaluation will work |
| Demo script: 10-minute jury walkthrough (§6.4), backup recordings | rehearsed demo |

### 6.3 Priority if time runs short

Cut in this order (last first): Phase 5 Could/Should items → Phase 4 Should items → RH-9
multi-user → VR → Phase 2 Should items. **Never cut Phase 0, Phase 1 or the Must rows of
URB/MIL** — they are the scoring core and the two hero applications.

### 6.4 Jury demo storyline (10 minutes)

1. **(1 min)** Drop the 10-minute video + SRT in; application auto-suggested; timer starts.
2. **(2 min)** Progressive map appears; accuracy report vs checkpoints; format ledger (OBJ, PLY,
   LAS, GeoTIFF, GLB/glTF, FBX) all present; completeness per class with unseen areas hatched.
3. **(2 min)** Urban: draw a road through the block, place a building, rule violation, fix it,
   swipe before/after.
4. **(3 min)** Military: mark enemy positions, draw route, exposure report, rehearse in first
   person (or VR headset for a judge), AAR replay.
5. **(1 min)** Disaster or Construction: volume + change in two clicks, field pack export.
6. **(1 min)** Final timer vs 900 s; scalability table; "measured vs proposed" separation.

---

## 7. Data we must collect (the unblocker)

Legal flights only: DGCA Digital Sky green zone, applicable drone class, landowner permission.
Checkpoints withheld from reconstruction (the F2 checkbox binds this).

| Flight | Where | Pattern / settings | Feeds | Reference data |
|---|---|---|---|---|
| **A — Site grid, 10 min uninterrupted** | campus ground with a building, an earth/sand pile, parked vehicles, people walking | nadir lawnmower, 60 m AGL, 1080p + 4K repeat, SRT on | Construction, Disaster, Terrain, **speed gate** | 12–20 checkpoints spread in XY and Z (DGPS/RTK or total station from the civil dept.), pile taped |
| **B — Building orbit + facades** | a campus block with roads and trees | 2 orbits (60° and 30° gimbal), 40–70 m radius | Urban, Military rehearsal, Inspection, Twin | 10 checkpoints on roof corners/ground, 5 tape lengths, rangefinder heights |
| **C — Straight corridor ≥ 1 km, flown twice** | boundary wall or road | constant heading, 80 m AGL, forward + oblique; change placed between passes | Border, Military route, linear Inspection | 8–10 checkpoints along the line |

Record per flight: drone model, camera, resolution/fps, SRT/log file, takeoff coordinates, time,
weather. Enable video subtitles (SRT) on DJI; otherwise GPS comes from the flight log with clock
sync.

---

## 8. What we will not claim

- No accuracy from similarity-aligned fits, PSNR, collider checks or GPS residuals. Primary
  accuracy is unaligned against withheld checkpoints.
- No completeness with our own output as the denominator.
- No speed from a resumed run, a preview, or an extrapolated short clip.
- No proposed, inferred or rehearsal content presented as measured. Every export and every
  rehearsal HUD carries the measured / proposed / unobserved distinction.
- No total score. We report each criterion's evidence; the judges score it.
