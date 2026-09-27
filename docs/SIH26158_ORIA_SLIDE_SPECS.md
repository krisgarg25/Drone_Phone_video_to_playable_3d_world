# Oria Build Specs — SIH26158 Deck (6 slides)

Reference set: **SIH** (`b3fd9ead-ef19-4a27-99ed-ec115ad56e79`), template.pptx attached, cover supported.
Paste each block into `oria_build_slides` as one string per array element.

## Rendered (do NOT rebuild these)

| # | Slide | Job ID | Slide ID |
|---|-------|--------|----------|
| 1 | Title page | `d26759f9-534b-4202-91d0-50cf1a8cd3ba` | `a86851f4-8c8f-4951-b271-14b0c1f35568` |
| 2 | Idea | `9295a1a0-280a-43a6-8314-e260c5e67944` | `62aad33b-8fc4-437a-93df-f31b69853cbb` |

Deck order for preview/export:
`["a86851f4-8c8f-4951-b271-14b0c1f35568", "62aad33b-8fc4-437a-93df-f31b69853cbb", <s3>, <s4>, <s5>, <s6>]`

Blocked on: `PAYMENT_REQUIRED` — balance exhausted at slide 3.

---

## SLIDE 3 — TECHNICAL APPROACH

### Page Description (Oria Layout)
Minimal corporate SIH content slide. Plain white background, high whitespace, no decorative shapes. Bold serif ALL-CAPS black action title centered at top with a thin crimson rule beneath. Reading order top to bottom: a full-width horizontal eight-stage process band across the upper third; below it a two-column split where the left 52% is an open analytical table and the right 48% is a horizontal bar chart; a thin evidence strip runs full width above the footer. Persistent solid deep-blue footer bar (#1A3B6B) with small white text. Table uses horizontal separators only, no vertical rules, no coloured header band. No cards, no icons.

### Page Content
**[Top center, bold serif ALL CAPS, black, slide title]** THREE MEASURED LEVERS, NOT A BIGGER GPU, ARE WHAT PUT THE 900-SECOND GATE IN REACH

**[Upper third, full-width, label small bold blue ALL CAPS]** Methodology and process for implementation - one runnable pipeline, measured stage times on a single RTX 3050 6 GB laptop

**[Upper third, horizontal eight-stage process band, each stage a bold stage name above one artifact line and one measured figure, stages connected left to right by thin arrows]**
INGEST - one UAV video plus GPS and flight metadata
PREPARE - quality gates and capture plan
FRAMES - score, flatten, mask - 68 of 288 frames kept
PRIORS - GPS becomes per-camera pose priors
MAP - sequential SfM, RANSAC Sim(3) - 239.7 s against 946.9 s exhaustive
DENSE and FUSE - PatchMatch stereo - 1,568,438 fused points
MESH - Poisson surface reconstruction
DELIVER - 5 of 6 official formats, EPSG:32643 derived from scene origin

**[Lower left, section label small bold blue ALL CAPS]** Technologies to be used (e.g. programming languages, frameworks, hardware)

**[Lower left, open analytical table, horizontal separators only, two columns: Layer | Stack. Seven rows]**
Language and runtime | Python 3.12, CUDA 12.4, PyTorch, Docker and conda environments
Reconstruction core | COLMAP 4.1.1 vendored build: pose-prior mapper, PatchMatchStereo, stereo_fusion, poisson_mesher
Geometry and alignment | Robust Sim(3) with RANSAC, WGS84 to local ENU, coordinate reference system derived from scene origin
Frame intelligence | ffmpeg and ffprobe decode; Laplacian, Tenengrad, anisotropy, spectral-centroid and blocking scores; epipolar-inlier dynamic masks
Telemetry parsing | GPX, DJI .srt, Pilot CSV and contract CSV readers
Output products | PLY, LAS, GeoTIFF DSM, glTF and GLB, FBX, WGS84 position table
Interface and QA | PlayCanvas WebGL viewer with collision and navmesh, Next.js evidence console, headless-Chrome walk QA

**[Bottom strip, full width, one line]** Working prototype: repository with continuous integration and a 4 minute 30 second demo; 11 of 11 CPU test suites green plus 151 survey tests.

**[Footer bar, left, small white text]** SIH 2026 · TECHNICAL APPROACH
**[Footer bar, right, small white text]** 3 / 6

### Page Charts
Chart title: Measured speed-up of each lever on one RTX 3050 6 GB laptop
Caption: Multiplier against that lever's own baseline on the same machine. Exact wall-clock seconds are shown in the pipeline band above.
Position: lower right of the body, beside the technologies table.
Type: horizontal bar chart, single series, one common scale.
X-axis: Speed-up versus its own baseline, unit times, domain 0 to 5, gridlines at 1, 2, 3, 4.
Y-axis categories top to bottom: Baseline-aware frame selection; Sequential against exhaustive matching; Dense profile fast against survey; Dense profile budget against survey.
Series name: Measured speed-up. Values: 4.2, 3.9, 2.6, 3.9. Value labels at bar ends reading 4.2x, 3.9x, 2.6x, 3.9x.
Annotation: dashed vertical reference line at 3.0x labelled "Dense must reach about 3x to fit 900 s at a 270-frame aerial budget".
Caveat printed under the chart: first bar is estimated dense time from measured per-image rates, the other three are measured stage times.

---

## SLIDE 4 — FEASIBILITY AND VIABILITY

### Page Description (Oria Layout)
Minimal corporate SIH content slide. Plain white background, no decorative shapes, high whitespace. Bold serif ALL-CAPS black action title centered at top with a thin crimson rule beneath. Body is a two-column split: left 55% is the official desired-output compliance table, right 45% is the official evaluation-criteria table carrying in-row weight bars on one common scale, with a single interpretation callout beneath it. A full-width band at the bottom pairs risks with strategies in two columns. Persistent solid deep-blue footer bar (#1A3B6B) with small white text. All tables use horizontal separators only, no vertical rules, no coloured header bands, no cards or pills.

### Page Content
**[Top center, bold serif ALL CAPS, black, slide title]** EVERY OUTPUT THE BRIEF NAMES NOW RUNS: WHAT IS LEFT IS MEASUREMENT, NOT MISSING CODE

**[Left column, section label small bold blue ALL CAPS]** Feasibility analysis - official desired output for SIH26158 against the position today

**[Left column, open analytical table, horizontal separators only, three columns: Parameter | Official target | Position today. Six rows]**
Reconstruction type | 3D mesh or point cloud | Delivered: Poisson mesh plus 1,568,438-point fused cloud
Processing time | Under 15 min per 10-min video | Gate implemented, not yet met; levers measured
Spatial accuracy | 1 m or better | Mechanism complete; zero qualifying checkpoints
Coverage | Entire visible scene | Observability layers delivered; 8 of 8 cells at 2 m
Output formats | OBJ, PLY, LAS, GeoTIFF, glb and gltf, fbx | 5 of 6 from one real run; OBJ fixed after a real-data bug
Visualisation | Web or desktop viewer | Delivered: walkable browser scene plus evidence console

**[Right column, section label small bold blue ALL CAPS]** Official evaluation criteria - weight and evidence today, in-row bars on a 0 to 30 scale

**[Right column, open analytical table, horizontal separators only, three columns: Criterion | Weight with in-row bar | Evidence today. Six rows]**
Accuracy | 30 | Mechanism complete, unproven on qualifying data
Completeness | 20 | Observability measured, 8 of 8 cells at 2 m
Speed | 20 | Levers of 2.6x to 4.2x, gate still unmet
Innovation | 15 | Delivered and distinct from survey-first pipelines
Scalability | 10 | Runs on one 6 GB laptop, peak 1,405 MiB
Usability | 5 | Walkable viewer, 65.3 m walked, 16 of 16 waypoints, 0 falls

**[Right column, single callout under the table]** Accuracy, completeness and speed carry 70 of 100 points and all three sit inside the reconstruction core. The interface is worth 5. That weighting is the reason the build order went to geometry first and the viewer second.

**[Bottom full-width band, two paired column labels small bold blue ALL CAPS]** Potential challenges and risks | Strategies for overcoming these challenges

**[Bottom band, four risk-to-strategy pairs, risk on the left strategy on the right, aligned rows]**
Dense stereo needs about 3x more speed at a true 270-frame aerial budget | Three measured levers behind one profile flag, plus progressive windowed submaps; the proving run is one 10-minute flight executed twice
Consumer GNSS is 1 to 3 m and worse vertically | GPS enters as uncertainty-weighted priors with takeoff-anchored heights and clock-offset estimation; RTK and PPK stay a separate declared track
Occluded faces cannot be imaged from a single path | The hole is measured and published with its excluded area; generative inpainting is deliberately absent, so no guess becomes a measurement
Exports never read by a third-party reader | Add GDAL, PDAL and laspy validation gates to continuous integration before the qualifying flight

**[Bottom, below the pairs, one line, italic small]** Against the brief's eight key challenges: one is fulfilled with measurement, five are mechanism-complete awaiting data, one is partial by design, one is not met on this hardware.

**[Footer bar, left, small white text]** SIH 2026 · FEASIBILITY AND VIABILITY
**[Footer bar, right, small white text]** 4 / 6

---

## SLIDE 5 — IMPACT AND BENEFITS

### Page Description (Oria Layout)
Minimal corporate SIH content slide. Plain white background, high whitespace, no decorative shapes. Bold serif ALL-CAPS black action title centered at top with a thin crimson rule beneath. Body is a two-column split: left 58% carries a horizontal four-stage application spine with two hero applications expanded beneath it and six supporting applications listed compactly; right 42% carries three stacked benefit blocks in one column, separated by thin horizontal rules. A full-width evidence strip sits above the footer. Persistent solid deep-blue footer bar (#1A3B6B) with small white text. No cards, no pills, no tiles, no icons.

### Page Content
**[Top center, bold serif ALL CAPS, black, slide title]** ONE ENGINE, EIGHT MISSION PRODUCTS, RUNS OFFLINE ON ONE 6 GB LAPTOP

**[Left column, section label small bold blue ALL CAPS]** Impact on target audience - what each mission gets, and what it must still buy

**[Left column, horizontal four-stage application spine, each stage a bold name over one detail line, connected left to right by thin arrows]**
BORDER AND PERIMETER - hostile terrain mapped without a survey crew entering it
DISASTER FIRST RESPONSE - damage model before the first team lands
FACILITIES AND TELEMETRY - existing sites digitised from footage already flown
MISSION REHEARSAL - walkable scene of the objective area before approach

**[Left column, two hero applications expanded, each a bold name then two tight lines]**
Military reconnaissance - available today: a 65.3 m walkable terrain scene with 16 of 16 waypoints reached and 0 falls, plus position, ground, obstacle and flyable data. Still one build away: enemy indicators such as helmet, vehicle, tent and gun, and full mission-rehearsal data that flying alone cannot supply.
Urban planning - available today: georeferenced buildings, roads, vegetation and utilities from real scenes, with 5 of 6 official formats produced in one run. Still one build away: semantic layers and a text legend.

**[Left column, compact single list line, small]** Supporting applications - disaster response, AI-in-the-loop flight optimisation, generation-X digital twins, historical preservation, forestry and agriculture, inspection.

**[Right column, section label small bold blue ALL CAPS]** Benefits

**[Right column, three stacked benefit blocks separated by thin horizontal rules, each a bold name then one dense line]**
Social - first damage model in minutes for search and rescue, no cell tower or cloud needed; the terrain of a landslide or a flood is walked remotely before anyone walks it for real.
Economic - one flight instead of a survey sortie and a ground-control crew; developed on a 6 GB laptop with a peak of 1,405 MiB, so there is no workstation to buy and no station to establish.
Strategic and environmental - runs fully offline for contested or air-gapped sites; licence-clean and reproducible from one command; fewer repeat sorties over the same ground.

**[Bottom strip, full width, one line]** Target audience: field engineers, disaster-response and border-mapping agencies, survey and construction firms, facility and autonomous-operations teams.

**[Footer bar, left, small white text]** SIH 2026 · IMPACT AND BENEFITS
**[Footer bar, right, small white text]** 5 / 6

---

## SLIDE 6 — RESEARCH AND REFERENCES

### Page Description (Oria Layout)
Minimal corporate SIH content slide. Plain white background, high whitespace, no decorative shapes. Bold serif ALL-CAPS black action title centered at top with a thin crimson rule beneath. Body is a four-column reference layout, each column a bold small-caps group heading over a compact left-aligned list, with thin horizontal rules between groups. A full-width evidence-accountability band sits above the footer. Persistent solid deep-blue footer bar (#1A3B6B) with small white text. Purely typographic, no charts, no icons, no cards.

### Page Content
**[Top center, bold serif ALL CAPS, black, slide title]** THE COMPONENT CHOICES, THE EVIDENCE RULE, AND WHERE EVERY FIGURE ON THIS DECK COMES FROM

**[Left column, group heading small bold blue ALL CAPS]** Reconstruction and mapping foundations
- COLMAP structure-from-motion and multi-view stereo - colmap.github.io
- OpenDroneMap end-to-end photogrammetry - github.com/OpenDroneMap/ODM
- Single-pass UAV trajectory planning for 3D reconstruction - github.com/ch1bo/drone-reconstruction

**[Second column, group heading small bold blue ALL CAPS]** Feed-forward geometry and tracking
- DPVO deep patch visual odometry - github.com/princeton-vl/DPVO
- Pi3 permutation-equivariant feed-forward reconstruction - github.com/yyfz/Pi3
- Depth-Anything-3 - github.com/ByteDance-Seed/Depth-Anything-3
- MapAnything generalisable metric 3D mapping - github.com/facebookresearch/map-anything
- VGGT-Long long-sequence online reconstruction - github.com/DengKaiCQ/VGGT-Long

**[Third column, group heading small bold blue ALL CAPS]** Rendering, physics and evaluation
- gsplat Gaussian splatting research stack - docs.gsplat.studio
- MuJoCo physics and reinforcement-learning engine - mujoco.org
- evo trajectory-evaluation tool - github.com/MichaelGrupp/evo
- RGB-D reconstruction and completion datasets - dornhelge.github.io

**[Fourth column, group heading small bold blue ALL CAPS]** This submission's evidence trail
- Problem statement and official desired output - SIH26158 problem statement, pages 38 to 40
- Evaluation and improvement report - 09_SIH26158_Evaluation_and_Improvement_Report_2026-09-22
- Evidence map, wiring ledger, readiness findings - docs/readiness/04-evidence-map, 02-wiring-ledger, 01-findings-2026-09-23
- Prototype and demo - github.com/krisgarg25/Drone_Phone_video_to_playable_3d_world, recorded 4 minute 30 second demo

**[Bottom strip, full width, section label small bold blue ALL CAPS then one line]** Evidence accountability - measured figures are stage times, frame counts, point counts and errors read from machine-generated reports; anything not yet proven is labelled a target and stated as unproven; the 1.14 m hold-out figure is reported against a drifting phone inertial reference and is presented as reference-dependent rather than as a pass; no total evaluation score is shown, because only the accuracy and completeness terms could be computed.

**[Footer bar, left, small white text]** SIH 2026 · RESEARCH AND REFERENCES
**[Footer bar, right, small white text]** 6 / 6
