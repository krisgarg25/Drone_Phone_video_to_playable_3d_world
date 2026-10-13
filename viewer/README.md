# viewer/

The browser side: the walkable 3D viewer, the phone capture page and the dashboard. Plain
HTML and ES modules served by [`_serve.py`](../_serve.py). There is no build step and
nothing is loaded from a CDN, so it all works offline.

## Pages

| Page | What it is |
|---|---|
| [`pc.html`](pc.html) | **The walkable viewer.** PlayCanvas renders the Gaussian splat; ammo.js runs physics against the collision mesh. Third person, first person, drone flight and VR. |
| [`capture.html`](capture.html) | **Phone capture.** Records video with AR pose priors and shows live coverage while you scan. Must be opened over HTTPS. |
| [`pipeline_gui.html`](pipeline_gui.html) | **Dashboard.** Start a run, watch it, open a scene. The original console; [`groundcontrol/`](../groundcontrol/) is the newer one. |
| [`textured_preview.html`](textured_preview.html) | Preview of the UV-textured mesh. |
| [`xr_probe.html`](xr_probe.html) | Reports what capture features a phone's browser supports. |

Open the viewer for a scene with:

```bat
.venv\Scripts\python.exe pipeline.py view myscene
```

## Viewer controls

| Key | Action | Key | Action |
|---|---|---|---|
| `W` `A` `S` `D` | Walk, or fly in drone mode | `F` | Toggle drone flight / ground walk |
| `Shift` | Run | `C` | Toggle first / third person |
| `G` | Gaussian splats on / off | `X` | Collision mesh wireframe |
| `P` | COLMAP camera frustums | `O` | Sparse tie points |
| `V` | Coverage overlay | `T` | Play the generated tour |
| `[` `]` | Step through source frames | `R` | Reset to spawn |

The HUD shows distance walked, falls, position and whether the character is grounded.
Those are the same values the headless walk test logs.

## What the viewer loads

Everything comes from `work/<scene>/viewer_assets/` and `work/<scene>/pc/`, written by the
pipeline: `scene.ply` (the splat), `collision.collision.glb`, `heights.f32` and
`ground.f32` (the ground the physics and the route share), `cameras.json`, `poses.json`,
and `world_check.json` (the gate's verdict).

## Modules

| File | Role |
|---|---|
| [`pc.js`](pc.js) | The viewer: scene loading, character controller, cameras, overlays, HUD. |
| [`coverage_map.js`](coverage_map.js) | Surfel coverage map with per-surface viewing directions, for capture. |
| [`workspace_core.js`](workspace_core.js) | Workspace contract and camera maths, with no DOM or engine in it. |
| [`plan_core.js`](plan_core.js) | Planning-layer logic for proposals drawn on a scan. |
| [`measurement_math.js`](measurement_math.js) | Geometry estimates for picked points. |
| [`furniture_geometry.js`](furniture_geometry.js) | Procedural furniture for Place mode. No downloaded assets. |
| [`survey_dashboard.js`](survey_dashboard.js) | The survey evidence console. |
| [`pc/scripts/`](pc/scripts/) | Mission rehearsal: navmesh pathing, bots, sessions with an instructor, WebXR. Inert unless enabled from the URL. |

The `*_core.js` and `*_math.js` modules are engine-free on purpose: the same files are
imported by the Node tests in [`tests/`](../tests/).

## Vendored

`pc/playcanvas.mjs` is the [PlayCanvas engine](https://github.com/playcanvas/engine) (MIT)
and `pc/ammo/` is [ammo.js](https://github.com/kripken/ammo.js), the Bullet physics port.
Both are checked in so the viewer never needs the network.
