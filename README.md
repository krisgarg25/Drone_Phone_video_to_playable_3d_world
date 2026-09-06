# Video → walkable 3D world

Turn a drone or phone video into a navigable 3D Gaussian-Splatting world you can walk
through in the browser, with a collision mesh, a ground surface, a generated tour
route, and an autopilot walk test that proves the world is actually walkable.

[![fast suites](https://github.com/krisgarg25/Drone_Phone_video_to_playable_3d_world/actions/workflows/ci.yml/badge.svg)](https://github.com/krisgarg25/Drone_Phone_video_to_playable_3d_world/actions/workflows/ci.yml)

<sub>The badge reads "unknown" until `.github/workflows/ci.yml` has run once on the remote.</sub>

Feed it a clip. It runs ~15–17 steps — keyframes, COLMAP pose solve, gsplat training,
world framing, asset export, sky/cloud removal, collider, heightfield, a validity gate,
and the walk test — and ships `work/<name>/viewer_assets/` plus a report saying what
each step did and why anything is missing.

The design rule: **any video produces an output, and every failure names itself.**
Nothing needs hand-tuned settings. Where a value could be wrong for one scene and right
for another, it is measured from the scene or the machine instead of being a default —
free VRAM sizes the training budget, the scene's own bounding box sizes the voxel grid,
the capture style picks the preset.

---

## Requirements

| | |
|---|---|
| OS | **Windows 10/11** — COLMAP and ffmpeg are vendored as `.exe` binaries |
| GPU | **Required.** NVIDIA, driver new enough for CUDA 12.4. `train` is not an optional step and gsplat has no CPU path, so without an NVIDIA GPU the pipeline cannot build a world. 6 GB completes every take here because the budget follows whatever VRAM is free, and `--quality` scales down further |
| Python | **3.12** (pipeline) and **3.10** (training — `--with-train`) |
| Node | 18+ (the collision mesh is built by `@playcanvas/splat-transform`) |
| Disk | ~4 GB for the two environments, plus ~1 GB per take while it works |
| Clone | `git-lfs` — one 313 MB COLMAP binary is stored in LFS |

## Install

```bash
git clone https://github.com/krisgarg25/Drone_Phone_video_to_playable_3d_world.git
cd Drone_Phone_video_to_playable_3d_world
python scripts/bootstrap.py --with-train
```

That creates `.venv` (3.12) and `.venv310` (3.10 + torch/gsplat), installs the Node
packages, downloads Chromium for the walk test, and finishes by running
`pipeline.py doctor`. Add `--check` to see what it would do without doing it.

If you'd rather do it by hand:

```bash
git lfs install
git clone --recursive <the url above>
python -m venv .venv && .venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m playwright install chromium
py -3.10 -m venv .venv310 && .venv310\Scripts\python -m pip install -r requirements-train.txt
cd tools && npm install && cd ..
.venv\Scripts\python pipeline.py doctor
```

`doctor` is the health check worth trusting — it runs each COLMAP subcommand this repo
uses, checks that `pycolmap` matches the vendored COLMAP version, probes the GPU, and
prints a `fix:` line for anything it rejects.

## Run it

```bash
# put a clip in videos/ ...
copy MyClip.mp4 videos\rocks.mp4        # Windows
.venv\Scripts\python pipeline.py run rocks
.venv\Scripts\python pipeline.py view rocks
```

`run` defaults to `--quality high --preset auto`, i.e. full training and a preset
diagnosed from your footage. To check a scene end to end in minutes instead of an hour,
add `--quality smoke` — every step still runs, including a real 300-step train.

Other commands: `scan` (diagnose footage, no reconstruction), `status`, `coverage`,
`capture` (what to film for a given preset), `benchmark`, `ui` (browser dashboard),
`doctor`.

### What you get

| path | |
|---|---|
| `work/<name>/viewer_assets/` | the walkable world — splat, collision mesh, heightfield, route |
| `work/<name>/report.json` | per-step outcome, timing, which fallback fired and why |
| `work/<name>/walktest/` | the autopilot's frames, video, and `walk_log.json` trajectory |
| `work/<name>/logs/` | one numbered log per step, the raw evidence behind the report |
| `results/blinded/` + `results/pair_key_<name>.json` | real frame vs engine render, A/B order kept out of the image |

## Does it work on any video?

Measured on this machine, all seven takes in `videos/`, every step completing:

| take | status | steps | walked | sampled route | airborne | falls |
|---|---|---|---|---|---|---|
| rocks | complete | 17/17 | 65.2 m | 64.7 m | 5/61 | 0 |
| room_w_jsonl | complete | 15/15 | 36.5 m | 31.7 m | 1/334 | 0 |
| roomscan | complete | 15/15 | 17.1 m | 15.5 m | 0/331 | 0 |
| temple | partial | 17/17 | 65.2 m | 64.0 m | 0/61 | 0 |
| test1 | complete | 15/15 | 28.0 m | 26.2 m | 0/332 | 0 |
| test2horizontal | partial | 15/15 | 30.1 m | 28.0 m | 0/336 | 0 |
| test2train | complete | 15/15 | 28.1 m | 26.4 m | 0/336 | 0 |

`partial` means the world shipped and the gate then refused to certify it — temple and
test2horizontal both fail the hard rule *spawn on supported ground*, and the run says so
instead of quietly passing a bad world. Those two are coarse or mis-scaled captures.
This was `--quality smoke`, which is a test that nothing fails, not a claim about final
visual quality.

Reproduce it with:

```bash
.venv\Scripts\python tests\check_all.py      # fast suites, seconds
.venv\Scripts\python tests\test_e2e.py       # every take in videos/, ~30 min
```

`check_all.py` covers unit checks (107), the failure classifier, capture diagnostics,
the collider, and the gate — and it is what CI runs on every push, on Windows, from a
clean checkout with only `requirements.txt` installed. It cannot cover `train`, `evals`
or a real reconstruction, so those are the local `test_e2e.py` run's job.
`test_e2e.py` runs each take in `videos/` end to end and prints the table above; it
exits non-zero if any take fails to produce an output **or if there are no takes at
all**, so it cannot pass vacuously.

## Failure policy, in one place

- **Classified, not tracebacked.** Every step's failure gets a kind (`oom`,
  `voxel-overflow`, `unsupported-flag`, `unsupported-asset`, `empty-input`,
  `missing-tool`, `timeout`, `crash`). Retryable ones are repaired — OOM halves the
  pixel budget, voxel overflow climbs the voxel ladder — and the repair is recorded.
- **Derived, not tuned.** `--preset auto` diagnoses the footage. Train sizes itself from
  free VRAM at that moment. The collider grid is fitted to the scene's extent under the
  voxeliser's own entry limit. The router plans for the body the viewer actually builds.
- **Degrade, don't discard.** A step that can't do its best job does its measurable
  second-best and says what it gave up. If multi-view geometry is too thin to bound the
  room, the region falls back to the camera path and the ground under it, with a warning
  — the run does not stop after already paying for the training.
- **The gate stays honest.** The world gate's hard verdicts are never downgraded to make
  a run green, and the walk test logs a sampled trajectory that the runner cross-checks
  against the distance the walk claims, so a number that disagrees with the route is
  caught rather than shipped.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| COLMAP dies with `stack buffer overrun` / exit `0xC0000409` | A git-lfs file came down as a 130-byte pointer. `git lfs install && git lfs pull`. `python scripts/bootstrap.py --check` confirms it in one line. |
| `vocab_tree_matcher` fails the same way | The vendored `tools/vocab_tree.bin` is a legacy flann index this COLMAP 4.x build refuses to read. Loop closure is disabled and the solve still completes. Re-download https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin to `tools/vocab_tree.bin` to get cross-clip matching back. |
| `'C:\Users\you\Desktop\Drone' is not recognized` | The repo root has spaces. Nothing here may use `shell=True`; if you hit this, it is a bug — `scripts/robust.py:run_cmd` refuses both forms on purpose. |
| Walk test reports `connection refused` on port 8137 | A stale `_serve.py` is holding it, or none is. The runner starts its own and reuses a live one; `netstat -ano \| findstr 8137` shows who owns it. |
| `train` says no CUDA | `.venv310` is missing or torch can't see the GPU. `python scripts/bootstrap.py --with-train`, then `pipeline.py doctor`. |
| `splat-transform is not installed` | The collider step now names this instead of raising `[WinError 2]`: `cd tools && npm install`, or re-run `python scripts/bootstrap.py`. (`npx` cannot be used as a fallback — npm's `npx` is a `.cmd` shim that `subprocess` cannot start without a shell.) |
| A step says `evidence missing` | An evidence step (evals, blinded pairs) hit something final, like a locked file. The world still shipped; the report names the gap. |
| Nothing in `videos/` | The clips are gitignored at 86 MB each; the per-take `calibration.json` / `data_poses.jsonl` fixtures are deliberately not, so a clone gets pose data and no footage. `tests/test_e2e.py` names that case and exits non-zero rather than reporting an empty pass. |

## Layout

```
pipeline.py            the runner: step graph, budgets, retries, report, viewer
scripts/               one script per step + shared hardening in robust.py
viewer/                PlayCanvas walkable viewer (pc.js), phone capture page
tools/                 vendored COLMAP, ffmpeg, vocab tree, splat-transform, navbake
tests/                 check_all.py (fast suites), test_e2e.py (all takes)
docs/                  capture technique, phone AR notes
work/<name>/           per-take output (regenerable, gitignored)
README-MVP.md          the engineering log: every step, every measured number, every
                       defect found this way
```

`README-MVP.md` is where the detail lives — per-step design, the collider's walkability
analysis, and the measured before/after for each fix. This file is only the door.

## License

MIT — see [LICENSE](LICENSE). The vendored binaries keep their own terms: COLMAP is
BSD-3-Clause, ffmpeg is LGPL/GPL as built, and `@playcanvas/splat-transform` is MIT.
