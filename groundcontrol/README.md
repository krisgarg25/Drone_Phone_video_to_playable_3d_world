# Ground Control Station

The web console for the pipeline: a project library, a run monitor, and a nine-mode
workspace for measuring, inspecting and planning on a scanned scene.

Next.js 16 (App Router) · React 19 · Tailwind CSS v4 · TypeScript

## Run it

The console is a front end only. Start the Python backend first, from the repository root:

```bat
.venv\Scripts\python.exe _serve.py 8137 .
```

Then, in this directory:

```bash
npm install
npm run dev
```

Open <http://localhost:3000>. To point at a backend somewhere else, set `BACKEND_ORIGIN`
(default `http://127.0.0.1:8137`).

## The one rule

**Every reading comes from the backend.** A value the server did not report is drawn as an
em dash with a "not evaluated" chip. It is never filled in with a plausible number. If you
add a panel, keep that property: measured, proposed, inferred and unobserved values must
look different on screen and in every export.

## Pages

| Route | What it is |
|---|---|
| `/` | Project library: every scene on disk, with import and run dialogs. `?view=processing` shows runs in progress. |
| `/projects/[scene]` | The workspace for one scene. `?tab=` or `/projects/[scene]/[workspace]` selects the mode. |
| `/projects/[scene]/export` | Export centre: deliverable formats and the one-click project bundle. |
| `/applications` | The eight application profiles and what each one needs. Renders without a backend. |
| `/challenges` | The brief's eight challenges, each graded against evidence. Renders without a backend. |

`/pipeline`, `/survey` and `/deliverables` are kept as redirects into the library and the
workspace, so older links still land somewhere useful.

Besides the nine modes below, a workspace has four inspector tabs: Overview, Quality (the
world gate's checks with their values and thresholds), Project and Files.

## Workspace modes

Defined in [`lib/applications.ts`](lib/applications.ts) (`WORKSPACE_TABS`) and described in
plain words in [`lib/modes.ts`](lib/modes.ts), which the tooltips and help popovers read.

| Mode | Panel | Purpose |
|---|---|---|
| Explore | `explore-overview.tsx` | Look around the model and see what it is made of. |
| Measure | `measurement-panel.tsx` | Distances, heights, areas and volumes by clicking on the model. |
| Inspect | `inspect-panel.tsx` | Log defects, find the photos that saw a spot, measure tilt and cable sag. |
| Operations | `ops-panel.tsx` | First map, change, damage, debris volume, access and flood. |
| Twin | `twin-panel.tsx` | Inventory, attributes, epochs and packaging. |
| Plan | `plan-panel.tsx` | Draw proposals on the scan, check rules, run a shadow study. |
| Mission | `mission-panel.tsx`, `sand-table.tsx` | Coordinates, line of sight, routes, rehearsal and after-action review. |
| Place | `placement-panel.tsx` | Put furniture and equipment on the scanned floor. |
| Walk | `walkthrough-panel.tsx` | Walk the scene on its verified collision surface. |

Which modes a scene shows depends on its application profile. The profiles here mirror
[`scripts/applications.py`](../scripts/applications.py), which is what the backend records
against a run. Change both together.

## How it talks to the backend

Two route handlers, and nothing else, reach the Python server:

- **`app/api/backend/[...path]`** forwards API calls. It only passes an allow-list of
  routes, and it refuses any write that does not originate from this local application.
- **`app/runtime/[...path]`** streams files from `viewer/`, `work/` and `videos/`, with
  range requests, so the embedded 3D viewer loads from the same origin.

The 3D view itself is the PlayCanvas viewer in [`../viewer/`](../viewer/), embedded by
`components/viewer-panel.tsx`.

## Layout

```text
app/          Routes, the two backend route handlers, global styles
components/   One panel per workspace mode, dialogs, the project library
lib/          API client, application profiles, mode descriptions, per-mode helpers
public/       Static assets
```

## A note on Next.js 16

This version has breaking changes from earlier releases. Read the guides in
`node_modules/next/dist/docs/` before writing code against APIs you remember; see
[AGENTS.md](AGENTS.md).

```bash
npm run lint
npm run build
```
