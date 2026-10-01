# Final video: plan

Draft 3, 2026-09-28. **All recordings are done; see [RECORDINGS.md](RECORDINGS.md).** Your decisions: any length, with every feature included; the quiz uses your two photos; features that can't be shown become text cards; you record the voiceover. The finished film is built in HyperFrames (HTML + GSAP → MP4) from `raw/demo.mp4`, new recordings and real pipeline artefacts. Recordings come first; the edit starts once every shot on the list below exists.

## The story

| Act | Target | What the viewer sees |
|---|---|---|
| 1. Hook: real or rendered? | 0:00–0:18 | Two frames side by side, A and B. "One is a frame from the drone video. The other was rendered from the 3D model. Which is real?" A 3-second countdown, then the reveal: frame 41 of the video is real, the other is the render. "Everyone I asked got it wrong." |
| 2. Bridge | 0:18–0:24 | "Before I show how it's made, here's what you can do with it." The rendered frame pushes in and cuts to the live model in the app. |
| 3. Capabilities | 0:24–3:25 | `demo.mp4`, cut down from 5:22 to about 3:00. Each chapter gets a title card, and the features the take does not use appear as mini windows. |
| 4. How it's made | 3:25–4:20 | The real pipeline on this laptop: 12 s drone clip → 72 keyframes → camera solve → training timelapse → clean-up → collision surface → checks. It ends on frame 41 again (the callback). |
| 5. Close | 4:20–4:30 | Back on the model. Title, one line, repo link. |

**Length:** open. Nothing is cut for time. `demo.mp4` keeps every chapter, and only dead time (typing, idle cursor) is sped up. The act times above are only a guide; the voiceover sets the pace.

## Act 1: the quiz frames

Only your two photos are used:
- **Real:** `C:/Users/krisg/Downloads/00041.jpg`, 1280×720.
- **Rendered:** `C:/Users/krisg/Downloads/3d mode image.png`, 1536×782.

**Different shapes:** the two are 16:9 and 1.96:1, so shown as they are, the shape alone gives the answer away. Both go into identical frames:
- the real frame is centre-cropped to 1.96:1 (1280×652);
- both are shown at the same size, with the same corner radius and border;
- both are re-encoded the same way, so neither looks sharper from compression.

Copies go into the film's `assets/quiz/`. The originals are not changed.

**Reveal wording:** "rendered from the 3D model". The model was trained with frame 41 included, so the video does not claim the model never saw this view.

## Act 3: cutting demo.mp4 to ~3:00

Times are from `TIMESTAMPS.md`. Speed-ups use `data-playback-rate` on camera glides and cursor travel only. Result moments play at 1×. "Mini windows" lists the features the take does not use.

| # | Chapter | Source | Keep | Cut or speed up | Mini windows |
|---|---|---|---|---|---|
| 1 | Open | 0:02.2–0:13.3 | open rocks, swing round | speed up the swing | – |
| 2 | Explore | 0:13.3–0:41.5 | flight path, collision mesh, fly to photo | panel-tabs tour | Feature points, Camera coverage, Object types |
| 3 | Measure | 0:41.5–0:58.9 | Height 7.48 m (approx.) | tools tour, typing | Distance, Area, Volume (boulder), Note |
| 4 | Inspect | 0:58.9–1:32.7 | spot → photos → view from photo; relief drape | tasks tour, naming | Defect list, Cut a section, Archive record · cards: Pole tilt, Cable sag, What moved |
| 5 | Operations | 1:32.7–2:11.9 | Flood fills the valley; line coverage | tours | Relief camp, Debris volume, Map tiles, Site logistics · cards: What changed, Building damage, People & vehicles, Cut & fill, Built vs design |
| 6 | Twin | 2:11.9–2:26.9 | measured / how sure / looks | – | Assets list (fly to asset) |
| 7 | Mission | 2:26.9–3:08.5 | enemy post, route, exposure, terrain | naming | Landing zones (on Open ground), Enemy guess, Rehearse, Brief |
| 8 | Walk | 3:08.5–3:23.8 | walk and look | – | – |
| 9 | Export | 3:23.8–3:39.3 | pick, one zip | rename | – |
| 10 | Open ground | 3:39.3–3:50.1 | – | mostly cut | – |
| 11 | Plan | 3:50.1–4:59.1 | building, floors, road, swipe, sunlight | naming, second building at speed | Rules, Impact, Share |
| 12 | Place | 4:59.1–5:21.7 | furniture on the patio | – | – |

**Mini windows:**
- **When:** at the end of each chapter there is a 4–6 s feature break. The main video eases back to about 62% scale on the right, and two to four mini windows rise on the left, one every ~1.2 s.
- **Style:** they look like app windows: graphite card, 1 px border, the task's icon and label, a 4–6 s looping clip, and the app's own one-line `what` text from `TASKS` in the panel components.
- **Clips versus cards:** a feature that cannot show a real result on rocks or Open ground (poles, cables, a second flight, a BIM model) gets an icon card with its one-line description, never a faked result.

**Look:** the app's design system (`--st-*` graphite tokens, Geist / Geist Mono, white primary, amber only for a live signal). No glows, no ALL-CAPS mono labels, no gradients.

## Act 4: the pipeline, using real artefacts

`work/rocks_quality/report.json` gives the real run: 16 stages, 1,096 s. The two big ones are COLMAP (183 s) and training (826 s, 15,000 steps, 55k → 233k Gaussians, PSNR ~41.6).

| Beat | ~s | Source |
|---|---|---|
| Input | 4 | `videos/rocks.mp4` (12 s, 1280×720, 24 fps) plays in a window |
| Keyframes | 5 | 72 frames from `frames_full/rocks/` fly into a contact sheet; blur floor 67 |
| Camera solve | 10 | feature points and matches drawn on two frames (from `colmap/database.db`), then an orbit of the sparse points and camera path in the viewer |
| Training | 14 | timelapse from frame 41's camera sharpening from blur to photo, with a step counter, Gaussian count and a PSNR curve drawn from `train_log.txt`. It lands on the quiz frame. |
| Physics | 5 | collision mesh and walkable surface (reuse `demo.mp4` 0:25–0:34 or a new orbit) |
| Checks | 5 | eval renders against the real frames (`eval_renders/eval_XX.png` vs `eval_pairs.json`), wiped |
| Summary | 5 | stage-time bar from `film/assets/pipeline/report.json` (Boulder field run): "22 minutes, 24 stages, one RTX 3050 laptop" |
| In the app | 6 | `pipeline-app.mp4` + `pipeline-train.mp4`: import the video → Reconstruct → the job banner and live log |
| Ready | 5 | `pipeline-viewer.mp4`: the new project opens and flies to photo 41, the quiz view |

**Claims:** the app labels scale "Approx. metres", and no survey exit gate has passed on real flight data. Numbers on screen are the measured ones, with the "approx." kept, and there are no survey-grade accuracy claims.

## Recordings (done)

Sky and cloud removal is **not** a beat: on this scene they dropped 63 Gaussians (0.0%) and kept all clouds, so there is nothing to show. The physics beat uses `pipeline-viewer.mp4` (solid surface) and the collider numbers.

### Original shot list

| ID | What | How | Status |
|---|---|---|---|
| R1 | Quiz frames | your two photos, cropped and matched as described above | ready |
| R2 | Training timelapse from a fixed camera | `train_splat.py` saves `step_*.jpg` only from the random training view it draws each step, and only every 3,000 steps (5 images). Add `--preview-image rocks/00041.jpg --preview-every 100` (≈150 frames) and re-run the pipeline into a **new** scene, `rocks_demo`. `rocks_quality` is not touched. | code change + ~20 min run |
| R3 | App capture of starting that run | rig take: Projects → import `rocks.mp4` → Reconstruct → the banner changes stage; later the project shows Ready | rig take |
| R4 | Pipeline viewer shots | rig takes: sparse points + camera path orbit; sky and clouds before and after; collision orbit | rig takes |
| R5 | Feature-match image | script: SIFT keypoints and matches for two keyframes from the COLMAP database → PNG | script |
| R6 | Feature minis (≈20) | one short rig take each (4–6 s of result), same 1080p60 capture, workspace pre-opened. Each one is rehearsed first; if a feature fails, it becomes a card. | rig takes |
| R7 | Music | your pick, or a royalty-free track | to do |
| R8 | Voiceover | you record it from `VO_SCRIPT.md`, which I write once the rough cut is locked | after rough cut |

**Order:**
1. R2, with R3 captured while it starts. The GPU is busy for ~20 min, so no viewer recording happens during the run.
2. R4, R5
3. R6
4. Rough cut, with on-screen text and no audio
5. VO script timed to the cut; you record it
6. Final edit paced to your voice, then music, then render

## Build

- **Project:** `video_production/film/` (`npx hyperframes init`). Node 24 is installed (22+ needed).
- **ffmpeg:** HyperFrames needs it on PATH; use `tools/ffmpeg/ffmpeg-9.0.1-essentials_build/bin`.
- **Structure:** a root `index.html` (1920×1080, `data-fps="60"` to match the capture) and one nested composition per act (`hook`, `bridge`, `capabilities`, `minis`, `pipeline`, `outro`).
- **demo.mp4 segments:** `<video>` clips using `data-media-start` for the trims and `data-playback-rate` for speed-ups. Never set `currentTime` from script.
- **Agent skills:** `npx skills add heygen-com/hyperframes` gives the agent the composition, animation and CLI contracts.
- **Loop:** `lint` → `check` → `snapshot` at the key frames → `render --quality draft` to review → `render --quality high --fps 60` for the final.
