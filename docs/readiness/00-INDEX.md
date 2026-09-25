# SIH26158 readiness documentation

Working documents for the Single-Pass Drone Video to Accurate 3D Model build.
Written 2026-09-23. Everything here is dated and sourced; nothing here is a claim
that a criterion has been met.

| File | What it is for |
|---|---|
| [01-findings-2026-09-23.md](01-findings-2026-09-23.md) | What the audit found, including two arithmetic errors in the benchmark evidence and their corrected numbers. |
| [02-wiring-ledger.md](02-wiring-ledger.md) | Every survey module, whether it runs in the pipeline or only in tests, and what artifact it produces. |
| [03-acceptance-protocol.md](03-acceptance-protocol.md) | The exact commands and data needed to turn "implemented and tested" into "measured on a real flight". GPU-gated. |
| [04-evidence-map.md](04-evidence-map.md) | Each official target against the artifact that would prove it, and its status right now. |
| [05-challenges-and-applications.md](05-challenges-and-applications.md) | The brief's eight key challenges and eight application verticals, each graded with the evidence behind the verdict. |

## The console

`groundcontrol/` is a Next.js 16 + Tailwind v4 instrument panel over the survey API.
It replaces nothing on disk; the legacy `viewer/pipeline_gui.html` still works.

```bash
# terminal 1 - the reconstruction backend (plain HTTP on 8137)
.venv/Scripts/python.exe _serve.py 8137 .

# terminal 2 - the console
cd groundcontrol && npm run dev        # http://127.0.0.1:3000
```

Design rule it obeys: every reading comes from the backend, and a value the server did
not report renders as an em dash with a `not evaluated` chip - never a plausible number.
`/challenges` and `/applications` are documentation screens and render with no backend
at all, so a reviewer can always see the honest verdict tables.

## Reading order for a reviewer

1. `04-evidence-map.md` - what is claimed versus what is proven.
2. `03-acceptance-protocol.md` - what has to run before any score is defensible.
3. `01-findings-2026-09-23.md` - the corrections, so nobody re-quotes a stale number.

## Standing rule used throughout this project

A number is labelled by its kind: **measured on real data**, **projected from
measured rates**, or **synthetic fixture**. A module existing and being tested is
never reported as it running in the pipeline. See `09_SIH26158_Evaluation_and_
Improvement_Report_2026-09-22.md` §9 for the measurements that predate this folder.
