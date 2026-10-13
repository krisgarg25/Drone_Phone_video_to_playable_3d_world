# Contributing

Thanks for looking. Bug reports, footage that breaks the pipeline, and pull requests are
all useful here, roughly in that order.

## Reporting a problem

The pipeline fails for reasons that depend heavily on the footage and the machine, so a
report is only actionable with both. Please include:

1. **The command you ran**, with its flags.
2. **`pipeline.py doctor` output.** It checks COLMAP, both Python environments, CUDA and
   the Node collider tool, and most install problems show up here.
3. **The failing step's log**: `work/<scene>/logs/<nn>-<step>.log`. The last 50 lines are
   usually enough.
4. **`pipeline.py scan <scene>` output** if the failure is in `keyframes` or `colmap`. It
   describes what the footage did (blur, overlap, motion) without reconstructing it.
5. **GPU model and VRAM**, and whether the clip is from a drone or a phone.

Use the bug report form; it asks for these.

### Footage that fails is a contribution

A clip that registers 40% of its frames, or produces a floor the walk test falls through,
is exactly what the capture presets are tuned against. If you can share one, say so in the
issue. Please do not attach footage of people or private property you do not have the
right to share.

## Setting up

```bash
git clone https://github.com/krisgarg25/Drone_Phone_video_to_playable_3d_world.git
cd Drone_Phone_video_to_playable_3d_world
git lfs pull
python scripts/bootstrap.py --with-train
```

`--with-train` builds the CUDA training environment and needs an NVIDIA GPU. Leave it off
if you are only working on code the fast suites cover; nothing in them imports PyTorch.

## Running the tests

```bat
.venv\Scripts\python.exe tests\check_all.py
```

This is what CI runs. It needs no GPU and finishes in a few minutes. Run a single suite
with `python tests/<file>.py`.

`python tests/check_all.py --e2e` also reconstructs every take in `videos/` at
`--quality smoke`. That needs a GPU and your own footage, since `videos/` is not in the
repository.

## Pull requests

Keep a pull request to one change. The template asks for three things, and they are the
standard the existing code is held to:

- **Evidence.** Numbers from a test or a run, with the suite and scene named. Adjectives
  are not evidence.
- **Honesty in the output.** If a value is measured, proposed, inferred or unobserved, the
  UI and every export should say which. A step that refuses to produce a result must say
  what it refused and why. No silent fallbacks.
- **Tests against known answers** for new modules, and a clean `tests/check_all.py`.

If your change alters what is built, verified or still open, update
[docs/GAPS_AND_OPTIMIZATIONS.md](docs/GAPS_AND_OPTIMIZATIONS.md) in the same pull request.

## Conventions

- **Match the code around you.** Comments in this codebase explain *why* a value or a
  branch exists, usually by naming the failure that led to it. Keep doing that.
- **A number needs a source.** Thresholds are derived from the scene (cell size, character
  height) or declared in a preset. A new constant should say what it was measured on.
- **Keep the two lanes separate.** `pipeline.py` (splat lane) and `survey.py` (survey lane)
  report different kinds of results. Do not present one lane's number as the other's.
- **Windows is the supported platform.** Paths with spaces and `.cmd` shims are real cases
  here, and CI runs on Windows for that reason.
- **No network at runtime.** The viewer and the console work offline: no CDNs, no external
  map tiles, no telemetry.

## Where things are

| You want to change | Look in |
|---|---|
| Step order, presets, budgets | `pipeline.py` |
| What a step does | `scripts/` (index in [scripts/README.md](scripts/README.md)) |
| The walkable viewer or phone capture | `viewer/` |
| The Ground Control Station | `groundcontrol/` |
| The local server and its API | `_serve.py`, `scripts/workspace_*_api.py` |

## Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
