---
workflow: general-video
flow: automation
storyboard: no
message: "One drone video becomes a 3D world you can measure, plan on and walk through, on one laptop."
aspect: 1920x1080
language: en
length: open (every feature included)
audience: SIH 2026 judges and technical reviewers (problem statement SIH26158)
---

## Intent

The demo film for Ground Control. The order is the user's:
1. A quiz: two frames, one real and one rendered from the 3D model. Which is real? Frame 41 is the real one, and every friend the user asked got it wrong.
2. "Before I show how it's made, here is what you can do with it."
3. The recorded feature take, `demo.mp4`. The features it does not use appear as motion-graphic mini windows.
4. How the pipeline makes it.
5. Close.

## Assets

- `assets/quiz/real.png`, `assets/quiz/render.png`: the user's two photos (`Downloads/00041.jpg`, `Downloads/3d mode image.png`), matched to the same 1280x720 framing. Only these two images are used for the quiz.
- `assets/video/demo.mp4`: the continuous feature take; `../TIMESTAMPS.md` and `../raw/demo.json` give its chapters.
- `assets/video/minis.mp4`: the feature minis, one `mini:` chapter each (`../raw/minis.json`).
- `assets/video/pipeline-*.mp4`, `assets/video/rocks.mp4`, and `assets/pipeline/*`: the real Boulder field run (see `../RECORDINGS.md`).

## Customizations

- Features with no honest result on these scenes become text cards (the user chose this).
- The user records the voiceover after the rough cut. The rough cut therefore carries on-screen text and no audio, and music is left for later.

## Notes

- Never show the personal room scan (the "Room W Jsonl" card on the Projects page): cut every Projects-page stretch.
- Measurements are approximate ("Approx. metres"), and no survey accuracy gate has passed on field data. Make no survey-grade claims.
- Frame 41 was a training view, so the reveal says "rendered from the 3D model", never "unseen".
