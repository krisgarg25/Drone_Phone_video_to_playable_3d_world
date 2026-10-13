# tests/

## The fast suites

```bat
.venv\Scripts\python.exe tests\check_all.py
```

64 suites, more than 1,000 test functions, no GPU. This is exactly what CI runs on every
push, on Windows. Suites run cheapest first, so a long failure at the end cannot hide that
the first ten were clean.

Run one suite on its own:

```bat
.venv\Scripts\python.exe tests\test_gate.py
```

`mvp.bat check` is a shortcut for the same thing.

| Area | Suites |
|---|---|
| Pipeline core | `test_robust`, `test_capture`, `test_collider`, `test_gate`, `test_scenario`, `test_detect_rooms`, `run_tests` |
| Survey lane | `test_survey_*`: georeferencing, GNSS, CRS and geoid, accuracy, evidence, formats, terrain, change, M3C2 and more |
| Applications | `test_mission*`, `test_inspection`, `test_ops_pack`, `test_facades`, `test_applications`, `test_plan_extras` |
| Workspace API | `test_workspace_*`: place, measure, proposals, operations, inspection |
| Viewer | `test_viewer_syntax`: every viewer script parses as an ES module |

New suites have to be registered in the `SUITES` list in
[`check_all.py`](check_all.py). A suite that only runs by hand goes stale.

## What the fast suites cannot prove

Anything that needs an NVIDIA GPU, the vendored COLMAP binary, or a real reconstruction.
For that:

```bat
.venv\Scripts\python.exe tests\check_all.py --e2e
```

This adds [`test_e2e.py`](test_e2e.py), which runs every take in `videos/` through the
whole pipeline at `--quality smoke` and fails if any take produces no output. It needs a
GPU and your own footage, because `videos/` is not in the repository. Allow up to three
hours.

## JavaScript tests

The 13 `test_*.js` files test the viewer's engine-free modules (coverage map, planning
core, capture scripts, rehearsal logic) with Node's built-in `assert`. Each is a script:

```bash
node tests/test_coverage_map.js
```

They are run by hand and are not part of `check_all.py`.

## Browser checks

`browser_*.py` drive the real pages with Playwright against a running server: the survey
console, the workspace, and model import. They are for checking a change by eye and by
assertion together, and are not part of CI.

## Writing a test

- **Test against a known answer.** Build the input so the right result is known in
  advance: a plane at a set tilt, a pile of a set volume, a camera track with a set scale.
- **Say what the number was judged on.** A gate check records its value, its threshold
  and which direction is better, so a failure reads as "4% against a 5% bar", not "fail".
- **A refusal is a result.** Where the code declines to answer (no measured ground, too
  few views), test that it refuses and says why.
