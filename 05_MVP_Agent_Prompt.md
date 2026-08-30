# 05 — MVP Agent Prompt (gauntlet-loop style)

Paste-ready prompt for an agentic coding CLI (Claude Code, ZCode, Codex, etc.). If you have the [gauntlet-loop](https://github.com/robonuggets/gauntlet-loop) skill installed you can also invoke `/gauntlet-loop` with the GOAL line below; otherwise just paste the whole prompt block into a fresh session. Pattern credit: Matt Shumer's builder/critic technique, packaged by RoboNuggets (CC BY 4.0).

Put your two videos in a `./videos/` folder next to this file before starting.

---

## THE PROMPT

```text
GOAL
Build, right here, an end-to-end MVP: drone flyover video in → 3D Gaussian Splat world out → a character walking around inside it (first- or third-person, gravity + collision). One command goes from video file to walkable scene. This is an MVP: cool and working beats polished and planned.

INPUT
Two videos live in ./videos/. There is NO GPS/SRT telemetry — skip all georeferencing. Start with videos/video_01; if it proves unrecoverable, switch to videos/video_02.

HARDWARE (hard constraint)
RTX 3050 laptop GPU, 6 GB VRAM, Windows. Everything you choose must actually train and run within 6 GB: downscale frames (~540–720p), cap keyframes (a few hundred max) and splat count, prefer memory-frugal trainers and configs. If anything OOMs, cut resolution/iterations and retry — never declare success without having RUN it on this machine.

QUALITY BAR (the only judge)
1. VISUAL BAR = the source video itself. Render your splat scene from camera poses corresponding to several moments of the original video and produce side-by-side comparisons against the REAL frames at those timestamps.
2. WALKABILITY BAR = the character spawns on solid ground, walks 50 m forward through the world without falling through geometry, turns freely, and the scene stays visually stable while doing it.

MANDATORY LOOP (gauntlet pattern)
- Decompose into small pieces: frame extraction/keyframing → camera poses → splat training → collision proxy/navmesh → walkable viewer with character. Build one piece at a time.
- After EVERY piece and EVERY overall iteration: actually run/open the output, capture screenshots (or screen-recordings for walking), strip labels, and place them next to the bar (real video frames / walk-test requirements).
- A SEPARATE fresh-context subagent is the CRITIC. It sees only the two unlabeled artifacts and issues a BINARY verdict: does the splat render hold up against the real frame? Does the walk test pass? No numeric scores, no "close enough", no partial credit.
- The builder NEVER judges its own work. When the critic says LOSE, fix precisely what the critique names and loop again — new screenshots, new blind comparison. Continue until WIN on both bars or the human stops you. No fixed round limit; "done" is forbidden unless the critic said WIN.

FREEDOM
Use ANY open-source components you like (gsplat, Nerfstudio splatfacto, OpenSplat, COLMAP sequential mode, VGGT/Pi3-class models, ffmpeg, three.js/spark viewers, Godot/Unity splat plugins, etc.). License restrictions are explicitly waived for this MVP — we will sort rights after it works. Spin up as many subagents as you want, in parallel if useful: researchers, builders, critics, testers. Install anything you need.

DELIVERABLES
- mvp.bat (or make.sh): `run <video>` → extracts frames → reconstructs → trains splats → builds collision → launches the walkable viewer with the character.
- README-MVP.md: what you used, exact commands, known weak spots.
- results/: final labeled side-by-side screenshots (render vs real frame) and the critic verdicts that won.
Show evidence at every step. Working screenshots or it didn't happen.
```

---

## Usage notes

- Fresh session, working directory = this project folder, videos already in `./videos/`.
- On Claude Code with the skill installed, `/gauntlet-loop` + the GOAL line works; the full prompt above encodes the same loop manually so it works anywhere.
- Expect the visual bar to be genuinely hard to "win" on the first passes — that's the point. Typical winning configuration on 6 GB lands around: 300–600 sharpness-filtered keyframes at ~0.5–1 fps equivalent, ~540–640px training resolution, gsplat/OpenSplat trainer with capped densification, collision from TSDF/fused-depth mesh, browser or Godot viewer with a capsule character.
- Keep the two bars separate in critiques: a scene can look great and still be unwalkable (floater-covered floors are the classic failure).
