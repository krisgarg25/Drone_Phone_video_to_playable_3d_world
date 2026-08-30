# 04 — Sample Videos & Test Data Catalog

| | |
|---|---|
| **Version** | 1.0 |
| **Date** | 2026-08-24 |
| **Purpose** | Ready-to-download inputs to exercise every stage of the pipeline (`02_System_Architecture_and_Pipeline_Design.md`) |

---

## TL;DR — grab these first

1. **7 free 4K drone flyovers** from Mixkit (no account needed, commercial-friendly license) → geometry/splat/walkability testing. Table below.
2. **5 real DJI `.SRT` GPS tracks** from `JuanIrache/dji-srt-viewer` → GPS-parser development today, no drone needed.
3. **One real DJI flight of your own** (video + its `.srt`, even a 2-minute park flight) → the only way to test the *full* video+GPS→georeferenced-mesh path end-to-end before you have RTK gear.
4. **UrbanScene3D + Mid-Air** datasets → accuracy scoring against ground truth.

---

## Category A — Free stock drone footage (geometry & splatting tests)

No GPS metadata in these files — they test M0 keyframing, M2 geometry, M4 splats/mesh, M7 walkability. All from [Mixkit](https://mixkit.co/free-stock-video/drone/) (free download without account, watermark-free; check current Mixkit License terms before redistribution). Filter the site by **"4K Available"** before downloading.

| Clip | Why it's useful for us | Tests |
|---|---|---|
| [City traffic roundabout from above](https://mixkit.co/free-stock-video/city-traffic-going-around-a-roundabout-from-above-1615/) | Moving cars contaminate reconstruction | Dynamic-object masking (M1), ghost-cleanup |
| [Drone flying between city buildings](https://mixkit.co/free-stock-video/drone-in-the-sky-at-a-city-581/) | Facades seen from single direction | Facade coverage limits (challenge i), building geometry |
| [Aerial view of rocks in countryside](https://mixkit.co/free-stock-video/aerial-view-of-rocks-in-the-countryside-2789/) | Forward pass over open grass | Baseline terrain reconstruction, TSDF scaling |
| [Drone shot over hills and dock](https://mixkit.co/free-stock-video/drone-shot-over-hills-and-dock-101506/) | Mixed terrain + man-made structure | Walkable ground extraction over slopes (M7 navmesh) |
| [Drone view over trees](https://mixkit.co/free-stock-video/drone-view-over-trees-613/) | Dense vegetation (worst case for meshes) | Vegetation handling, splats-vs-mesh comparison |
| [Flight above jungle river](https://mixkit.co/free-stock-video/drone-flight-above-a-jungle-river-49334/) | Low-texture water + canopy | Robustness suite (challenge ii/iii) |
| [Wooded landscape in the morning](https://mixkit.co/free-stock-video/aerial-view-of-a-wooded-landscape-in-the-morning-2795/) | Soft low-angle light, haze | Illumination variation (challenge iii) |

**Avoid for reconstruction:** pullbacks/reveals/pans, night timelapses, hyperlapses — camera rotation without translation breaks SfM-style geometry. Our M0 blur/motion filters should reject them automatically; keep one such clip as a *negative test*.

Other free sources if you need more variety: [Pixabay drone videos](https://pixabay.com/videos/search/drone/) (Pixabay License), [Pexels aerial videos](https://www.pexels.com/search/videos/drone%20aerial/) (Pexels License; blocks scripts — browse manually), [Videvo](https://www.videvo.net/stock-video/drone/) (check per-clip license, some attribution-required).

## Category B — Real DJI GPS tracks (`.srt`) — georeferencing-path tests

The [`JuanIrache/dji-srt-viewer`](https://github.com/JuanIrache/dji-srt-viewer) repo ships five genuine DJI flight subtitle tracks (GPS lat/lon, altitude, gimbal, ISO/shutter per frame):

```
https://raw.githubusercontent.com/JuanIrache/dji-srt-viewer/master/samples/sample0.SRT   (~69 KB)
https://raw.githubusercontent.com/JuanIrache/dji-srt-viewer/master/samples/sample1.SRT   (~121 KB)
https://raw.githubusercontent.com/JuanIrache/dji-srt-viewer/master/samples/sample2.SRT   (~74 KB)
https://raw.githubusercontent.com/JuanIrache/dji-srt-viewer/master/samples/sample3.SRT   (~79 KB)
https://raw.githubusercontent.com/JuanIrache/dji-srt-viewer/master/samples/sample4.SRT   (~80 KB)
```

Use them to build and unit-test the M0 `.srt` parser, GPX export, and GNSS outlier rejection immediately — no aircraft required. Pair them later with your own DJI clips once you record flights. Companion tools worth reading: [`DJI_SRT_Parser`](https://github.com/JuanIrache/DJI_SRT_Parser) (MIT, parsing logic reference) and [`jonm3D/DJI_SRT_Tool`](https://github.com/jonm3D/DJI_SRT_Tool).

**The irreplaceable sample:** one continuous single-pass recording from a real DJI drone where the matching `<name>.SRT` sits next to `<name>.MP4` (any recent DJI does this when subtitles are enabled). A lazy 2-minute park/flyover flight is enough to validate the entire video→keyframes→DPVO→MapAnything→Sim(3)-to-GNSS→UTM chain. Borrow a friend's Mini/Air/Mavic for an afternoon if you don't own one — this is the single highest-value test asset.

## Category C — Research datasets with ground truth (accuracy scoring)

| Dataset | What you get | Use it for | Access |
|---|---|---|---|
| **UrbanScene3D** ([project page](https://vcc.tech/UrbanScene3D)) | Long-range drone **videos** + high-res oblique images + registered **LiDAR scans** of large urban areas | Our primary quantitative benchmark: Chamfer/F-score vs LiDAR (Doc 03 §2) | Project page (was slow/timing out on 2026-08-24 — retry or use mirror links inside the repos: [Linxius/UrbanScene3D](https://github.com/Linxius/UrbanScene3D), [yilinliu77/UrbanScene3D](https://github.com/yilinliu77/UrbanScene3D)) |
| **Mid-Air** ([midair.ulg.ac.be](https://midair.ulg.ac.be/)) | Synthetic drone flights: RGB + depth/normals/semantics + simulated **GPS + IMU**, selectable subsets via downloader | Regression tests for tracker & georef fusion with perfect GT; weather/season robustness suite | Automated selective download; **CC BY-NC-SA — research/testing only** |
| **TartanAir** ([theairlab.org/tartanair-dataset](http://theairlab.org/tartanair-dataset/)) | Synthetic challenging flights, exact poses + depth | Hard-case geometry regression (fog, motion, clutter) | Cloud download tooling |
| **UAV123** ([KAUST page](https://cemse.kaust.edu.sa/ivul/uav123)) | 123 real short drone video clips | Quick visual-odometry sanity checks on real footage | Direct downloads |

## Category D — Classical baseline data

Run the same scenes through OpenDroneMap (our accuracy baseline): since **v3.0.4** ODM ingests `video.mp4 + video.srt` pairs directly — drop both into the images folder and compare its georeferenced outputs against ours. Community sample imagery and help live at [community.opendronemap.org](https://community.opendronemap.org); video flags are documented at [docs.opendronemap.org](https://docs.opendronemap.org).

## Bonus — YouTube route (only Creative Commons)

For scene diversity you can pull CC-BY-licensed drone uploads:

```bash
# On YouTube: filter search results by "Creative Commons" license first!
yt-dlp -f "bv*[height<=2160]" "<video-url>"     # then re-encode to constant-fps mp4
ffmpeg -i in.webv -vsync cfr -r 30 -c:v libx264 -crf 18 sample_yt.mp4
```

CC-BY requires attribution — keep a credits file. Standard-YouTube-license videos must be left alone. These clips have no GPS tracks, so treat them like Category A.

---

## Suggested test matrix

| Pipeline capability | Primary sample(s) |
|---|---|
| Keyframe selection + blur rejection | Mixkit set incl. one pan/reveal clip as negative test |
| `.srt`/XMP GPS parsing, GPX/QC plots | dji-srt-viewer samples 0–4; own DJI flight |
| Chunked metric geometry + chunk alignment | Countryside-rocks, hills-and-dock, own DJI flight |
| Full georeferencing chain (Sim(3)→ENU→UTM) | Own DJI flight (only source with matched video+GPS) |
| Dynamic-object suppression | Roundabout traffic clip |
| Facade quality / occlusion honesty | Between-buildings clip |
| Vegetation & water robustness | Trees, jungle-river clips |
| Walkable navmesh over slopes | Hills-and-dock, countryside-rocks meshes |
| Quantitative accuracy scoring | UrbanScene3D (LiDAR GT), Mid-Air (synthetic GT) |
| Baseline comparison | Same inputs through OpenDroneMap ≥3.0.4 |

## What makes a good capture when YOU shoot samples

- Continuous forward motion at steady speed (rough rule: ≥70% frame overlap → slower or higher); no pans, no hyperlapse, no orbit unless deliberately testing orbits.
- Lock exposure/white balance; max bitrate or ALL-I GOP; 4K/30 preferred.
- Enable subtitles (.srt) on the drone; keep filename pairing intact.
- Include one slow climb segment — barometric-altitude fusion needs vertical variation.
- 20–60 seconds is plenty per test case; keep total sample library < ~20 GB so cloud jobs stay cheap.
