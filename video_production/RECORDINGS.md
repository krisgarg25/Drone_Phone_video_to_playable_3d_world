# Recordings for the final film

Everything the edit needs, recorded 2026-09-28. All screen takes are 1920x1080 at 60 fps (`raw/*.mp4`), and each has a `.json` with its chapters on the video's own clock. The plan is [FINAL_VIDEO_PLAN.md](FINAL_VIDEO_PLAN.md).

**Do not use:** the first ~2 s of `pipeline-app.mp4` and `pipeline-viewer.mp4` show the Projects page, which lists the personal room scan. Start those clips after the page has changed.

## Main take

| File | What | Times |
|---|---|---|
| `raw/demo.mp4` | The 12-chapter feature demo (5:22) | [TIMESTAMPS.md](TIMESTAMPS.md) |

## Pipeline

| File | What | Length |
|---|---|---|
| `raw/pipeline-app.mp4` | New project → import `rocks.mp4` → Reconstruct (high) → job banner and live log (keyframes, camera solve) | 0:49.4 |
| `raw/pipeline-train.mp4` | The same run mid-training: stage chips with real times, training log streaming (PSNR, Gaussian count) | 0:18.3 |
| `raw/pipeline-viewer.mp4` | The finished Boulder field project: open it, fly to photo 41 (the quiz view), feature points + camera path, solid surface, back to photo 41 | 0:43.8 |

Chapters:

- `pipeline-app`: New project 0:01.0–0:13.0; Start the reconstruction 0:13.0–0:19.8; Processing 0:19.8–0:49.4
- `pipeline-train`: Training 0:01.1–0:18.3
- `pipeline-viewer`: Ready 0:01.2–0:14.1; Camera solve 0:14.1–0:25.1; Physics 0:25.1–0:34.2; Photo-real 0:34.2–0:43.8

| Asset | What |
|---|---|
| `film/assets/pipeline/timelapse.mp4` | 151 frames (30 fps, 5 s): frame 41's camera rendered every 100 training steps, step 1 → 15,000. Step numbers in `timelapse_steps.txt` |
| `film/assets/pipeline/features.json` | COLMAP keypoints on frames 35 and 41 (11,749 and 12,087) and a sample of their 3,280 verified matches, normalised 0..1; scene totals 72 images, 2,396 pairs, 3,151,673 matches |
| `film/assets/pipeline/report.json` | The Boulder field run: 24 stages, 1,326 s (colmap 180 s, train 914 s), gate: all checks passed |
| `film/assets/pipeline/train_log.txt` | Loss / PSNR / Gaussian count every 200 steps, for the training curve |
| `film/assets/quiz/real.png`, `render.png` | Your two photos at the same 1280x720 framing: `render.png` is `3d mode image.png` aligned (SIFT, scale 0.914, no rotation) and cropped to the real frame |

The source run is `work/boulder-field-0aa9375f` (project "Boulder field"), from `_archived_workspaces/videos/rocks.mp4`. `rocks_quality` was not touched.

## Feature minis

**The film uses `raw/minis-clean.mp4`** (rig take `take_minis_clean.py`, 9:00, recorded 2026-09-28). Every mini starts from the scene's clean state: the workspace is reset, the page reloaded and the camera set to the default view, then only that mini's own set-up runs. Chapters are in `raw/minis-clean.json` (`mini:` chapters only). The older sequential take below left each mini's results visible in the next ones.

### Older sequential take

`raw/minis.mp4` (one take). Use only the `mini:` chapters; the `prep:` chapters between them are set-up moves.

| Mini | Scene | In | Out | Length |
|---|---|---|---|---|
| Object types | Rocks | 0:03.7 | 0:11.1 | 7.4 s |
| Distance | Rocks | 0:17.9 | 0:27.3 | 9.4 s |
| Area | Rocks | 0:27.3 | 0:38.6 | 11.3 s |
| Volume | Rocks | 0:38.6 | 0:50.0 | 11.3 s |
| Note | Rocks | 0:50.0 | 0:59.0 | 9.0 s |
| Cut a section | Rocks | 1:04.2 | 1:11.9 | 7.7 s |
| Archive record | Rocks | 1:11.9 | 1:17.5 | 5.5 s |
| Road access | Rocks | 1:21.2 | 1:31.4 | 10.2 s |
| Map tiles | Rocks | 1:31.4 | 1:38.0 | 6.6 s |
| Enemy guess | Rocks | 2:02.1 | 2:08.7 | 6.6 s |
| Brief | Rocks | 2:08.7 | 2:15.0 | 6.3 s |
| Rehearse | Rocks | 2:16.7 | 2:30.2 | 13.4 s |
| Landing zones | Open ground | 2:39.5 | 2:50.1 | 10.6 s |
| Relief camp | Open ground | 2:57.3 | 3:19.2 | 21.9 s |
| Site logistics | Open ground | 3:19.2 | 3:31.4 | 12.2 s |
| Rules | Open ground | 3:51.0 | 4:00.4 | 9.4 s |
| Impact | Open ground | 4:00.4 | 4:04.6 | 4.2 s |
| Split | Open ground | 4:04.6 | 4:11.4 | 6.8 s |
| Share | Open ground | 4:13.1 | 4:22.7 | 9.6 s |

## Text cards (no honest result on these scenes)

| Feature | Why a card |
|---|---|
| Camera coverage (Explore) | Draws nothing readable on rocks |
| Pole tilt, Cable sag (Inspect) | No poles or cables in either scene |
| What moved, Seasonal change (Inspect); What changed, Debris volume, Stockpile volume (Operations) | Need an earlier flight of the same site |
| Building damage, People & vehicles, First map (Operations) | No buildings; 0 sightings in the video; no progressive run |
| Cut & fill, Built vs design (Operations) | Need a design or BIM model |
| Restoration idea (Inspect) | Opens Plan with Hypothesis ticked; shown by the Plan minis |
| Assets (Twin) | Lists boulders as "buildings" on rocks, so not shown as a highlight |

Notes: the Rehearse clip shows Chrome's pointer-lock notice at the top, and Map tiles and Share show a download toast; crop them in the mini window. Measurements on rocks are approximate (the app labels them "Approx. metres").
