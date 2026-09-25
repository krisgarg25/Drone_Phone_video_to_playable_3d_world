# Acceptance protocol - what has to run, in what order

Everything below touches the RTX 3050 and is **gated behind explicit approval**.
Nothing here has been run: this is the procedure, written so that the run is one
command per step and every number it produces lands in a file a reviewer can open.

CPU-only verification is already done: 663 tests, `torch` never imported, plus the
headless dashboard checks. Do not re-run those to feel better about a score; they
prove software behaviour, not reconstruction quality.

## Step 0 - get a qualifying dataset (blocking, nothing else matters first)

The brief's mandatory input is a **single-pass 1080p/4K drone video with GPS and
flight metadata**. None exists locally: the longest clip is 151.6 s and the highest
resolution of the seven MP4s is 1280x798. The official dataset "will be provided
real time", so until it arrives every accuracy, completeness and speed figure is a
projection.

Minimum for a defensible run:

| Item | Requirement | Why |
|---|---|---|
| Video | one clip, ≥ 1920x1080, ~600 s, single pass | the official processing gate is defined on a 10-minute clip |
| GPS | 1 Hz or better, WGS84, with fix quality or HDOP per sample | uncertainty comes from evidence, not from a guess |
| Metadata | `time_offset_s`, `altitude_datum`, `position_reference`, `single_pass`, `video_duration_s` | the contract `survey_georef.validate_metadata` insists on |
| Truth | ≥ 8 independent surveyed checkpoints, plus a reference surface if completeness is claimed | 8 is the floor for a fit/hold-out split; below that no split is attempted |

The surveyed points are the whole ballgame for 30% of the score. RTK rover, total
station, or a pre-existing surveyed plan - any of them, but they must not come from
the same GPS track the model is aligned to.

## Step 1 - prepare (CPU, safe)

```bash
mkdir -p "videos/acceptance"
cp /path/to/pass.mp4 "videos/acceptance/pass.mp4"
.venv/Scripts/python.exe survey.py inputs acceptance \
    --telemetry /path/to/track.gpx --metadata /path/to/metadata.json
.venv/Scripts/python.exe survey.py prepare acceptance
```

`inputs` accepts a GPX, a DJI `.srt` subtitle, a DJI Pilot CSV or the six-field
contract CSV, and writes the normalised form plus `telemetry_source.json`. If the log
carries no evidence of precision it refuses and says what to declare; it will not
invent a standard deviation.

Read out of `work/acceptance/survey/preparation.json`:

- `gates.observability.identifiability` - can clock offset and lever arm be separated?
- `gates.vertical_reference.status` - is an ellipsoidal claim consistent with the series?
- `gates.requirement.recommended_minimum` - which constraint to obtain, if any
- `capture_plan.path_coverage` and `frame_count` - what the budget bought geometrically

## Step 2 - reconstruct (GPU, needs approval)

```bash
.venv/Scripts/python.exe survey.py reconstruct acceptance --allow-gpu --dense-profile survey
```

Sequential stages, each timed into `run.json:steps`:
`keyframes -> frames -> priors -> colmap -> poses -> undistort -> dense -> fusion -> mesh`.

Budget guidance for a 6 GB card: the measured baseline was 291 frames at 1000 px
(peaks sampled at 1405 MiB sparse / 443 MiB dense, whole-GPU so those are floors, not
process peaks). A 600 s clip at the plan's 400-frame budget is roughly that scale;
`--dense-profile fast` drops geometric consistency and 1000 px, `budget` drops to
700 px. Run one profile at a time - two concurrent jobs skew every timing, and timing
is the evidence.

## Step 3 - evaluate (CPU)

```bash
# place surveyed points first, in the scene's own ENU frame:
# work/acceptance/survey/checkpoints.json
#   {"independent": true, "alignment": "none", "coordinate_frame": <from georeference.json>,
#    "checkpoints": [{"reconstructed": [e,n,u], "reference": [e,n,u]}, ...]}
.venv/Scripts/python.exe survey.py evaluate acceptance
```

`evaluation.json` then carries:

- `criteria[accuracy].metrics` - the **unaligned** hold-out error in metres
- `accuracy_split.tracks.{unaligned,se3_aligned,sim3_aligned}` - what a rigid or
  similarity fit would hide
- `accuracy_split.verdict` - against the ≤ 1 m target, with the statistic named
- `model_completeness.regions` - observable / weak / unobservable cell fractions
- `processing_prediction` - labelled `predicted from measured per-image rates`

## Step 4 - the deliverable check

```bash
ls work/acceptance/survey/runs/*/products/georeferenced/
cat  work/acceptance/survey/runs/*/products/delivery_manifest.json
```

Six formats must read `delivered`: `surface.obj`, `cloud.ply`, `cloud.las`,
`dsm.tif`, `cloud.gltf`, `cloud.fbx`, plus `positions_wgs84.csv`. The ledger reports
each format's own status - it is not a count of files that happen to exist.

Then, once, with a real GIS toolchain installed in a throwaway environment (not the
project's): `gdalinfo dsm.tif`, `lasinfo cloud.las`, and open `surface.obj` in
MeshLab/Blender. That closes the one gap the writers cannot close themselves: they
re-parse their own output, which is self-consistency, not third-party validation.

## Step 5 - speed, honestly

The official gate is strictly under 900 s for a 600 ± 0.5 s clip. `run.json:secs`
measures from the approval gate through every uncached stage including export and
hashing; the final manifest commit is excluded and says so. To make the number
quotable:

1. fresh scene directory, no cached steps;
2. record the host: CPU model, RAM, GPU model and driver, disk type;
3. one run, one number, no stitching across failures;
4. quote `speed.official_status` from `evaluation.json`, never a hand-summed total.

## What would make each claim fail safely

Every gate below already exists in code; the point of this list is that a failure
produces a named reason rather than a quiet pass:

| Condition | What happens |
|---|---|
| Video below 1080p | reconstruction refused before any work |
| Declared duration disagrees with the container | refused |
| GPS has no bracket around a keyframe | that camera is omitted, never extrapolated |
| Trajectory is collinear | alignment refuses: a straight pass cannot fix rotation |
| Fewer than 8 checkpoints | no split attempted; paired error reported with that reason |
| Mesh absent | OBJ stays `not_delivered` with the reason, faces are never invented |
| CRS not derivable | georeferenced set refused with the reason, local set still written |
| No depth maps | occlusion reported as *not checked* |
