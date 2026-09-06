# Phone AR capture for better reconstructions

Handheld phone video plus AR pose priors. The poses are optional input that
makes COLMAP register captures plain SfM gives up on.

## 0. The built-in phone recorder (`viewer/capture.html`)

Start the included server, then open the HTTPS address it prints on the phone:

```bat
.venv\Scripts\python.exe _serve.py 8137 .
```

Tap **Start camera capture**. The normal view is the fast path: it records a
720p camera stream at 8 Mbps and draws a lightweight guide directly over the
video—framing lines, a centre reticle, visible texture features, and a
`MOVE SIDEWAYS` nudge when motion is mostly rotation. It is intentionally a
capture-quality guide, not a false claim of per-surface AR coverage.

Coverage colour lives on the inset map, never over the camera image, and no
overlay is drawn into the recorded video, so nothing can become a false COLMAP
feature. The live AR view carries only thin world lines: the scanned-volume cage,
a blue cross on the spot that needs another look, and outlines round the planes
the phone reports. At the end of the take, the page reports
the largest mapped patches that still need another angle. A surface never seen
by the depth camera cannot be reported as missing, so still follow the capture
checklist below.

WebXR depth remains experimental because phone/browser support varies widely.
When it is available, it can provide the deeper per-surface coverage workflow;
the main camera view does not depend on it, so it remains responsive and useful
on every phone with camera permission.

## 1. Why

COLMAP fails on rotation-in-place handheld video: standing still and panning
gives frames with almost no parallax, so pair geometry is degenerate and the
mapper drops the segment or welds it into a wrong pose graph. Walking arcs fix
this, but some coverage (corners, tight rooms, looking straight up a wall) is
hard to get while always translating.

ARCore (Android) and ARKit (iPhone) track the camera in metric space with
visual-inertial odometry. Feeding those per-frame camera positions to COLMAP as
pose priors breaks the degeneracy: even a weak-parallax segment has metric
translations, so registration survives. Priors also let multiple clips share
one world (each clip's AR origin is where you started recording, but the
mapper snaps them together), and they make a single systematic walk cover every
corner instead of relying on loop closures.

What this buys, concretely:

| without priors | with priors |
|---|---|
| spin-in-place segments dropped or distorted | registered via metric translation |
| each clip solved independently, scale per clip | clips linked in one reconstruction |
| coverage limited by parallax-friendly motion | full room in one walk |

## 2. Capture checklist (indoor room walkthrough)

- Move in **arcs and orbits**, not spins. Every rotation should ride on a
  translation.
- Do **three height passes**: waist height, raised (above head), low (knee
  height). This covers furniture tops, floors, and verticals.
- **Slow, steady speed**. Roughly a slow walk; if footage looks fast at 8x,
  it is too fast.
- **Lock exposure and focus** if the capture app allows it; rolling exposure
  shifts colours between frames and hurts matching.
- **Good light.** AR tracking and SfM both degrade in the dark.
- Keep **60-70%+ overlap** between successive views.
- **Corners get extra orbit shots**: an arc of half a circle around each
  corner region.
- **Never stand still and pan** without stepping sideways. This is the one
  move that actively poisons the solve.
- **Pick one hold per room and say so on the start screen** (`Auto / Vertical /
  Horizontal`). Scan a whole room the same way round; do not start vertical, then
  turn the phone sideways for a wide wall.
- **Leave the lens on `Auto (widest)`** unless the room is tiny. A wide lens puts
  more of the room in every frame, and frames that share more of the same wall are
  what the solver matches on. In an AR scan the phone picks the lens and the page
  cannot ask for another, so read the **Field of view** row on the Phone tab: under
  about 55° across means a telephoto is recording the room.

Typical good length: **1.5-4 minutes per room**. The pipeline wants roughly
**100-400 keyframes total** after thinning; `extract_keyframes.py` selects by
sharpness and temporal bins, so err on the longer side rather than rushing.

**Why one hold matters more than it looks.** A vertical clip and a horizontal one
in the same scene used to reconstruct as though half the footage had gone missing:
COLMAP is handed one shared camera and quietly refuses every frame whose dimensions
disagree with it, as a warning, with a zero exit code. It now notices the mix and
gives each clip its own camera, and it stops the run outright if frames still go
missing — but a room shot the same way round is still the cleaner solve, and the
coverage map on the phone reads more consistently for it.

## 3. Option A - Android with any ARCore logger app

Any app that logs the ARCore camera pose per frame works. The importer accepts
CSV or JSONL rows in this layout:

| field | meaning |
|---|---|
| `timestamp_seconds` | seconds; matched against the video timeline of its clip |
| `qx qy qz qw` | orientation, scalar-last quaternion |
| `tx ty tz` | position, metres |

Frame convention: **camera-to-world, right-handed, metres, Y-up** - i.e. native
ARCore output, no conversion needed. One row per frame is ideal; any steady
rate works because the importer resamples onto the clip timebase.

Open-source examples found on GitHub:

- **PyojinKim/ARCore-Data-Logger** - exports `timestamp, q_x, q_y, q_z, q_w,
  t_x, t_y, t_z`, exactly this layout.
- **Spectacular Rec** app + `sai-cli` - exports `data.jsonl` plus the video.
- **OSUPCVLab/mobile-ar-sensor-logger** - records synced video + IMU +
  timestamps CSV from one app.

Record the **normal camera video at the same time**; the logger runs alongside
it. End up with two files per take: `walk1.mp4` and its pose log, named alike
(see section 6).

## 4. Option B - iPhone with Record3D ($ app)

Record3D exports either the `.r3d` container (it is a zip) or an EXR+JPG
folder. Inside is a metadata JSON whose `poses` field holds N x 7 rows,
one per RGB frame:

    [qx, qy, qz, qw, tx, ty, tz]

camera-to-world, metres, OpenGL Y-up convention.

Workflow: record in Record3D, export, unzip the archive into a subfolder of
your scene folder - the pipeline auto-detects it. The layout it expects is
whatever the zip contains:

    videos/<scene>/rec3d1/rgbd/0.jpg, 1.jpg, ...
    videos/<scene>/rec3d1/metadata

Frames are imported directly from `rgbd/` with 1:1 poses from `metadata` -
no video decoding step at all, which also sidesteps any
compression-timestamp fuzz. No extra flags needed; just run the scene and
the `priors` step picks them up.

## 5. Option C - Standalone experimental WebXR page (Android Chrome only)

`tools/webxr_capture.html` records pose JSONL and the composited camera video
from a WebXR `immersive-ar` session, entirely in the browser.

1. Prefer the integrated recorder in `viewer/capture.html`; it has the same
   WebXR foundation plus transfer, calibration, and coverage reporting.
2. To use this older standalone page, serve the repo root from the PC:
   `.venv\Scripts\python.exe _serve.py 8137 .`
3. Open `https://<pc-ip>:8138/tools/webxr_capture.html` on the phone.
4. Grant camera permission, press START, walk the scene, STOP, download
   `poses.jsonl` + `capture.webm` into `videos/<scene>/`.
5. Run the pipeline with `--poses` (section 6).

Caveats (also printed in the page's red banner): Android Chrome only, requires
the raw camera access feature, video-pose sync is approximate. Prefer a native
logger when available.

Practical notes for step 2:

- WebXR needs a **secure context**. Plain LAN http triggers a refusal; either
  set `chrome://flags/#unsafely-treat-insecure-origin-as-secure` to the page
  URL, use `adb reverse tcp:8137 tcp:8137` and open
  `http://localhost:8137/tools/webxr_capture.html` (loopback counts as secure),
  or serve over https.
- `_serve.py` binds the LAN interface and serves both HTTP and HTTPS. Use its
  printed HTTPS URL for WebXR; the self-signed certificate must be accepted on
  the phone once before Chrome will allow camera and AR access.

## 6. How to feed into the pipeline

By convention: put clips and pose logs into `videos/<myscene>/`, logs named
similarly to their clip:

    videos/myscene/walk1.mp4
    videos/myscene/walk1_poses.jsonl
    videos/myscene/walk2.mp4
    videos/myscene/walk2_poses.csv

Or pass everything explicitly:

```bat
python pipeline.py run myscene --video clip1.mp4 clip2.mp4 --poses walk1_poses.jsonl walk2_poses.jsonl
```

Rules:

| rule | detail |
|---|---|
| one log per clip | pose logs are per-video; no single global log across clips |
| timestamps flexible | relative-to-clip-start and epoch-stamped logs both work - the importer re-bases to each clip's 0:00 automatically |
| units | metres, always |
| AR world origin/scale drift | fine - priors guide registration, COLMAP re-anchors the world |

Then just run the pipeline. It injects the priors into COLMAP automatically
(`pose_prior_mapper`); confidence is controlled with `--prior-std`
(default 0.15 m). Lower it when you trust the AR tracking, raise it when the
phone drifted.

## 7. Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| "pose count mismatch" warnings | clock offsets between recorder and logger, different fps | benign if counts are close; the importer resamples onto the video timebase |
| "no priors matched" | wrong log paired with wrong clip (e.g. `walk2_poses.jsonl` given for `clip1.mp4`) | check pairing; explicit `--video ... --poses ...` order must correspond |
| run stops at colmap: "only N of M keyframes solved" | the take gave SfM no baseline (spun in place, or one subject from one spot), or footage mixes a vertical hold with a horizontal one | re-walk it in arcs; keep one hold per room. `python scripts/analyze_take.py <take>` says which it was |
| Phone tab shows a narrow field of view (under ~55° across) or "the phone refused the wide setting" | the AR scan is recording through a telephoto, or the zoom dial is advertised and then not honoured | walk slower and closer so the narrow frames still overlap. Plain capture on `Auto (widest)` records more of the room but gives up world-locked coverage; `xr_probe.html` says whether the wide lens is reachable from a page at all |
| registered cameras sit far from the priors / scene warped | AR tracking drifted during the take | re-record, or loosen trust with a higher `--prior-std` (default 0.15 m) |
| device has no AR / browser refuses WebXR | no ARCore, old phone | fall back to plain capture - the section 2 movement rules still apply, arcs and overlap carry the solve without priors |
