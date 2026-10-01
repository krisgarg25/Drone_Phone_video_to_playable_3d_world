# Demo video: one continuous take

v4, 2026-09-28. Recorded as `raw/demo.mp4`: 5:22, 1920x1080 at 60 fps. [TIMESTAMPS.md](TIMESTAMPS.md) lists every chapter and every action. `take-frames.jpg` has one frame per highlight.

The app is opened once from the projects page. Every workspace is reached from the header, with no reloads, and the camera glides between views. The only page changes are the ones a person would make: opening Export, going back to Projects, and opening Open ground.

## Scenes

| Scene | What it is |
|---|---|
| `rocks_quality` | The real scan: a grassy hillside with boulders. Everything except Plan and Place. |
| `flat_ground` ("Open ground") | Generated, not scanned (`scripts/make_flat_scene.py`). 100 m of flat ground with CC0 aerial textures from Poly Haven, a paved patio, a tree line and a collision mesh. Used for Plan and Place. |

## Chapters

| # | Chapter | What happens |
|---|---|---|
| 1 | Open a scan | Projects page, open rocks, the camera swings round |
| 2 | Explore | Panel tabs; Layers: drone flight path; collision mesh on its own; photo model back; click a photo to fly to where it was taken |
| 3 | Measure | Tools tour; Height of the boulder (7.48 m); name; save |
| 4 | Inspect | Tasks tour; click a spot, every photo that saw it, view from the photo, save to the defect list; Heritage: local relief draped on the model |
| 5 | Operations | Disaster tour, then **Flood** fills the valley; Construction tour; Border **line coverage** (fence line, observation post, seen and blind stretches) |
| 6 | Twin | As measured / How sure / As it looks; Versions; Game engine |
| 7 | Mission | Name it; **Symbol**: enemy post on the boulders, turned to face the meadow; Symbol: objective; Route; **Exposure** of the route; **Terrain** map for wheeled vehicles |
| 8 | Walk | 1-bot practice; look, walk, look back; free the mouse |
| 9 | Export | Pick files; rename one; one .zip; download |
| 10 | Open ground | Back to Projects, open the flat ground |
| 11 | Plan | Name the scheme; Rules / Impact / Existing; building + 4 more floors; second building; road; **Swipe** existing against proposed; **Sunlight**: shadows scrubbed from morning to late afternoon |
| 12 | Place | Coffee table, sofa and armchair on the patio; look around |

## Rig

The rig is in `rig/`. `take.py` is the take, and `demo.py` holds the helpers. It is driven by browser-harness, while high-frequency input goes over one persistent CDP websocket.

- **GPU:** Chrome is forced onto the RTX 3050 (`--force_high_performance_gpu`). On the integrated Intel GPU the splats rendered at about 28 fps.
- **Smooth motion:** the cursor and camera moves are animated inside the page on every frame. The camera uses the viewer's own `camera-set` orbit command, which now takes an optional `fov`, so an orbit can ease out of a photo's narrow field of view.
- **Cursor:** a drawn cursor with a click ripple. CDP input moves no OS pointer.
- **Capture:** ffmpeg Desktop Duplication at 60 fps with NVENC. The timestamp log runs on the capture clock.
- **Clean state:** the take starts from a clean state. Earlier saves are moved to `_archived_workspaces/rig_reset/`, never deleted. Afterwards, rocks gets its original measurement and plan back.

### Re-record

From the repo root in PowerShell. Don't touch the mouse or keyboard during the take, and leave the demo Chrome window un-minimized:

```
.venv/Scripts/python.exe video_production/rig/launch_chrome.py --record
$env:BH_TELEMETRY="0"; $env:BU_CDP_URL="http://127.0.0.1:9333"; $env:MODE="record"
Get-Content video_production/rig/run.py | browser-harness
.venv/Scripts/python.exe video_production/rig/make_timestamps.py
```

To rehearse without recording, use `MODE=rehearse` in a normal window (no `--record`). The page is laid out at 1920x1080, and a still is saved after every step in `raw/stills/`.
