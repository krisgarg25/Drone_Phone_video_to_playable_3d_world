# Key challenges and application verticals - what is fulfilled, what is not

Answering the brief's two lists directly, with the evidence behind each verdict.
Updated 2026-09-23 after the first real GPU ablation.

Verdict vocabulary: **fulfilled** (runs, and something measures it), **mechanism**
(runs, no qualifying data proves the benefit yet), **partial**, **not fulfilled**,
**absent** (does not exist).

## The eight key challenges

| # | Challenge | Verdict | Evidence and limit |
|---|---|---|---|
| i | Limited viewing angles from one path | **mechanism** | `survey_selection` spends the frame budget on flight geometry with a minimum-baseline floor; the plan reports `path_coverage` and mean parallax at prepare time. `survey_georef` refuses a collinear trajectory outright, because a straight pass cannot fix rotation. What no code can do is see a face that was never imaged. |
| ii | Motion blur and compression | **mechanism** | `survey_frame_quality` scores every kept frame (Laplacian, Tenengrad, anisotropy, spectral centroid, 8x8 blocking) and drops below 0.25 weight. On the rocks clip **0 of 72 frames were dropped** - the clip is not blurry enough to exercise the rule, so the rule is untested against genuinely bad footage. |
| iii | Variable illumination and shadows | **mechanism** | `survey_photometry.flatten` divides the low-frequency illumination field out of the *matching* copy only; the colourised frames are untouched, so a shadow never becomes a painted wall. Not separable from masking in the first ablation (see iv). |
| iv | Dynamic objects | **fulfilled - measured** | 72/72 masks accepted by COLMAP after the naming fix; 4.85% of pixels vetoed. Effect on the reconstruction: registered cameras **unchanged at 72/72**, sparse tracks **-18.4%** (58,985 → 48,141), mean view support -10.3%, **mean reprojection error improved 7.8%** (0.3586 → 0.3305 px), fused points -0.6%. Fewer features, a better-conditioned model, and the surface barely changed - which is what removing traffic and people should do. Caveat: masks and flattening ran together, so the gain is attributable to the pair, not to either alone. |
| v | GPS inaccuracy and sensor noise | **mechanism** | GPS enters bundle adjustment as per-camera priors with per-axis uncertainty, not as a post-hoc stamp; deterministic RANSAC rejects outlier cameras; fix-quality/HDOP are turned into std conservatively and the pipeline refuses to invent them. Preparation reports clock-offset bounds, vertical-datum signals and trajectory identifiability. Absolute accuracy is still unproven - that needs surveyed checkpoints. |
| vi | Near-real-time processing | **not fulfilled** | Measured dense stage alone: **554.3 s for 72 frames at 1000 px**. The rate model predicted a 291-frame run at 3960.1 s against 4001.4 s measured (**1.0% error**) and it says `fits_deadline: false`. Progressive windows are planned by `survey_streaming` and executed by nothing. On a 6 GB laptop at survey quality the < 900 s gate is not met; `fast`/`budget` profiles are the mitigation and are predictions, not measurements. |
| vii | Occluded surfaces | **partial, by design** | Measured / weak / unobserved are classified and kept separate; `hidden_regions` measures the hole and `coverage_denominator` publishes the excluded area with any coverage figure; `constrained_extension` adds a plane-derived roof continuation and labels it `constrained`, never `measured`. Generative inpainting is deliberately absent: `inference_policy()` publishes no entry point. A hidden face cannot be recovered as a measurement from one pass. |
| viii | Metric accuracy without extensive GCPs | **mechanism** | GPS-as-priors plus a robust Sim(3) fit, and `gcp_requirement` states what this scene's own telemetry can and cannot identify, with four named constraint options. Numerically unproven: the only hold-out figure available is 1.14 m against a **drifting phone VIO track**, which is the wrong reference for a ≤ 1 m claim. |

**Summary: 1 of 8 fulfilled with measurement, 5 of 8 mechanism-complete and awaiting
qualifying data, 1 partial by deliberate choice, 1 not met on this hardware.**

## The eight application verticals

What each one needs, and whether the system can do it today.

| Application | Available today | Missing | UI panel |
|---|---|---|---|
| i Border / strategic mapping | Georeferenced UTM + WGS84 position table + DSM raster; offline, laptop-class | Line-of-sight analysis, map basemap, zone handling for scenes crossing a zone edge | **Map** |
| ii Disaster damage assessment | `volume_between` cut/fill with uncertainty (tested, not wired); two-cloud diff is arithmetic away | A registration flow for pre/post surveys, damage classification | **Change** |
| iii Urban planning / smart cities | Metric mesh + cloud, per-point support | **Building/road/vegetation extraction does not exist.** Semantic layers are the whole ask here | **Objects** |
| iv Infrastructure inspection | `measure_segment`, `height_above`, `ground_plane`, clearance and facade height with a stated uncertainty budget | Point-picking in the viewer; reporting | **Measure** |
| v Construction progress | Volume diff, as-built measurement | Time-series registration, planned-vs-actual model import (IFC) | **Change** |
| vi Archaeological documentation | Metric archive, orthophoto-style DSM, per-phase volumes, non-invasive | Phase management, finds register, provenance export | **Archive** |
| vii Digital twin / **interior: place furniture in a bought flat** | The indoor path works: room scenes reconstruct, colliders and navmesh exist, walk test measured at 65.3 m with 16/16 waypoints and 0 falls | Object placement with real collision, lighting match, scale lock. The geometry and physics are there; the placement tool is not | **Interior** |
| viii Military recon / mission planning | Walkable rehearsal environment, navmesh baked from real terrain, cover from scanned geometry, offline | Route planning with sight-lines, load-bearing path analysis | **Route** |

**Honest read:** the metrology and measurement verticals (iv, vi, vii, viii, ii) are
close to real because they need tools on top of geometry the system already produces.
The semantic verticals (iii, and the "vegetation and obstacles" line of the brief)
need a segmentation model that does not exist in this repository. That is the single
biggest remaining scope decision, and it is a build, not a wiring job.

## What this document is used for

It is the content model for the Next.js console: each vertical becomes a panel whose
tools are either live, marked "needs a run", or marked "not built" - the console must
not present a planned capability as an available one.
