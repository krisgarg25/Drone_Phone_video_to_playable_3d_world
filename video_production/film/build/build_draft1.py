"""Build the film's compositions from the recordings' own logs.

    .venv/Scripts/python.exe video_production/film/build/build.py

Writes index.html, compositions/*.html, STORYBOARD.md and assets/stills/*. Every cut point
comes from raw/demo.json, raw/minis.json and raw/pipeline-*.json, and every number on screen
from the Boulder field run's own files (assets/pipeline/*). Edit this script, not the HTML.
"""
import json
import math
import re
import subprocess
from pathlib import Path

FILM = Path(__file__).resolve().parents[1]
RAW = FILM.parent / "raw"
ROOT = FILM.parents[2]
FFMPEG = str(ROOT / "tools/ffmpeg/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe")
FPS = 60
X = 0.6                       # crossfade between scenes (s)

# ------------------------------------------------------------------ shared look
FONTS = """
@font-face { font-family: "Geist"; src: url("assets/fonts/Geist-Variable.woff2") format("woff2"); font-weight: 100 900; }
@font-face { font-family: "Geist Mono"; src: url("assets/fonts/GeistMono-Variable.woff2") format("woff2"); font-weight: 100 900; }
"""
TOKENS = """
  --bg: hsl(225 12% 5%); --panel: hsl(222 12% 8%); --raised: hsl(222 11% 11%);
  --line: hsl(222 9% 22%); --ink: hsl(220 25% 96%); --dim: hsl(218 12% 76%);
  --muted: hsl(218 8% 62%); --accent: hsl(36 100% 57%); --green: hsl(156 62% 52%);
  --soft: hsl(36 55% 11%);
"""
BASE = """
#ROOT { position: absolute; inset: 0; overflow: hidden; background: var(--bg); color: var(--ink);
  font-family: "Geist", sans-serif; %TOKENS% }
#ROOT * { box-sizing: border-box; }
#ROOT .mono { font-family: "Geist Mono", monospace; }
#ROOT .eyebrow { font-family: "Geist Mono", monospace; font-size: 24px; letter-spacing: 0.02em; color: var(--accent); }
#ROOT .h1 { font-size: 84px; font-weight: 750; letter-spacing: -0.035em; line-height: 1.1; margin: 0 0 8px; }
#ROOT .h2 { font-size: 56px; font-weight: 700; letter-spacing: -0.03em; line-height: 1.12; margin: 0 0 4px; }
#ROOT .body { font-size: 32px; font-weight: 350; color: var(--dim); line-height: 1.3; margin: 0; }
#ROOT .win { position: absolute; background: var(--panel); border: 2px solid var(--line); border-radius: 14px; overflow: hidden; }
#ROOT .crop { position: absolute; left: 0; top: 0; width: 1920px; height: 1080px; transform-origin: 0 0; }
#ROOT .crop > video, #ROOT .crop > img { position: absolute; left: 0; top: 0; width: 1920px; height: 1080px; object-fit: cover; }
#ROOT .stat { font-family: "Geist Mono", monospace; font-size: 76px; font-weight: 600; color: var(--accent); letter-spacing: -0.02em; line-height: 1; }
#ROOT .stat-l { font-size: 26px; color: var(--dim); margin-top: 10px; line-height: 1.3; }
#ROOT .pill { display: inline-flex; align-items: center; gap: 12px; padding: 10px 18px; border-radius: 999px;
  font-size: 26px; font-weight: 600; background: var(--panel); border: 2px solid var(--line); }
#ROOT .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
"""


def css(cid, extra=""):
    return FONTS + (BASE + extra).replace("#ROOT", f"#{cid}").replace("%TOKENS%", TOKENS)


def subcomp(cid, dur, body, script, extra_css=""):
    """A templated sub-composition; every id inside is prefixed with the composition id."""
    return f"""<!doctype html>
<html lang="en">
  <head><meta charset="UTF-8" /></head>
  <body>
    <template>
      <style>{css(cid, extra_css)}</style>
      <div id="{cid}" data-composition-id="{cid}" data-width="1920" data-height="1080" data-duration="{dur:.3f}">
{body}
      </div>
      <script>
        (function () {{
          const tl = gsap.timeline({{ paused: true }});
{script}
          window.__timelines["{cid}"] = tl;
        }})();
      </script>
    </template>
  </body>
</html>
"""


def crop_div(src_html, x, y, w, win_w):
    """A 1920x1080 source shown through a window of width win_w, cropped to (x, y, w)."""
    s = win_w / w
    return (f'<div class="crop" style="transform: translate({-x * s:.2f}px, {-y * s:.2f}px) '
            f'scale({s:.5f})">{src_html}</div>')


def video(vid, src, start, dur, media_start, rate=1.0, cls="", style=""):
    r = f' data-playback-rate="{rate:.4f}"' if abs(rate - 1) > 1e-6 else ""
    return (f'<video id="{vid}" class="{cls}" src="{src}" data-start="{start:.3f}" data-duration="{dur:.3f}" '
            f'data-media-start="{media_start:.3f}"{r} muted playsinline style="{style}"></video>')


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def still(src_mp4, t, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(src_mp4), "-frames:v", "1",
                    "-q:v", "2", str(out)], check=True)


# ------------------------------------------------------------------ act 1: the quiz
HOOK_D = 22.0
FW, FH = 844, 475            # the two frames
AX, BX, FY = 96, 96 + FW + 40, 300
CW, CH, CX, CY = 1280, 720, 320, 300     # the combined wipe frame
S_AB = CW / FW


def hook():
    cid = "hook"
    body = f"""
        <div id="hook-top" style="position:absolute; left:96px; top:88px; width:1728px;">
          <div id="hook-eye" class="eyebrow">A quick test</div>
          <p id="hook-l1" class="h2" style="margin-top:18px">One of these is a frame from a drone video.</p>
          <p id="hook-l2" class="h2" style="color:var(--dim)">The other was rendered from a 3D model.</p>
        </div>
        <div id="hook-bridge" style="position:absolute; left:96px; top:96px; width:1728px; opacity:0">
          <p id="hook-b1" class="h1">Before I show you how it's made,</p>
          <p id="hook-b2" class="h1" style="color:var(--accent)">here's what you can do with it.</p>
        </div>
        <div id="hook-a" class="win" style="box-shadow:0 0 0 0px hsl(156 62% 52% / 0); left:{AX}px; top:{FY}px; width:{FW}px; height:{FH}px; transform-origin:0 0">
          <img id="hook-a-img" src="assets/quiz/real.png" style="position:absolute; inset:0; width:100%; height:100%; object-fit:cover" />
        </div>
        <div id="hook-b" class="win" style="left:{BX}px; top:{FY}px; width:{FW}px; height:{FH}px; transform-origin:0 0">
          <img id="hook-b-img" src="assets/quiz/render.png" style="position:absolute; inset:0; width:100%; height:100%; object-fit:cover" />
        </div>
        <div id="hook-la" class="mono" style="position:absolute; left:{AX}px; top:{FY + FH + 22}px; font-size:40px; font-weight:600">A</div>
        <div id="hook-lb" class="mono" style="position:absolute; left:{BX}px; top:{FY + FH + 22}px; font-size:40px; font-weight:600">B</div>
        <div id="hook-ta" class="pill" style="position:absolute; left:{AX + 60}px; top:{FY + FH + 16}px; opacity:0">
          <span class="dot" style="background:var(--green)"></span>Real · frame 41 of the drone video</div>
        <div id="hook-tb" class="pill" style="position:absolute; left:{BX + 60}px; top:{FY + FH + 16}px; opacity:0">
          <span class="dot" style="background:var(--accent)"></span>Rendered from the 3D model</div>
        <p id="hook-q" class="h1" style="position:absolute; left:96px; top:880px; opacity:0">Which one is real?</p>
        <p id="hook-f" class="h2" style="position:absolute; left:96px; top:890px; width:1728px; opacity:0">I showed this to my friends. <span style="color:var(--accent)">Every one of them got it wrong.</span></p>
        <div id="hook-ring" style="position:absolute; left:1690px; top:866px; width:134px; height:134px; opacity:0">
          <svg width="134" height="134" viewBox="0 0 134 134"><circle cx="67" cy="67" r="58" fill="none" stroke="hsl(222 9% 22%)" stroke-width="6"/>
          <circle id="hook-ring-arc" cx="67" cy="67" r="58" fill="none" stroke="hsl(36 100% 57%)" stroke-width="6" stroke-linecap="round"
            stroke-dasharray="364.4" stroke-dashoffset="0" transform="rotate(-90 67 67)"/></svg>
          <div id="hook-n3" class="mono" style="position:absolute; inset:0; display:grid; place-items:center; font-size:60px; font-weight:600">3</div>
          <div id="hook-n2" class="mono" style="position:absolute; inset:0; display:grid; place-items:center; font-size:60px; font-weight:600; opacity:0">2</div>
          <div id="hook-n1" class="mono" style="position:absolute; inset:0; display:grid; place-items:center; font-size:60px; font-weight:600; opacity:0">1</div>
        </div>
        <div id="hook-wl" style="position:absolute; left:{CX}px; top:{CY - 12}px; width:4px; height:{CH + 24}px; background:var(--accent); opacity:0; border-radius:2px"></div>
        <div id="hook-wlab-a" class="pill" style="position:absolute; left:{CX + 24}px; top:{CY + CH - 76}px; opacity:0"><span class="dot" style="background:var(--green)"></span>Real</div>
        <div id="hook-wlab-b" class="pill" style="position:absolute; left:{CX + CW - 220}px; top:{CY + CH - 76}px; opacity:0"><span class="dot" style="background:var(--accent)"></span>Rendered</div>
        <div id="hook-full" style="position:absolute; left:0; top:0; width:1920px; height:1080px; transform-origin:0 0; opacity:0">
          <img src="assets/quiz/render-full.png" style="position:absolute; inset:0; width:100%; height:100%; object-fit:cover" />
        </div>"""
    ax, bx = CX - AX, CX - BX
    script = f"""
          tl.fromTo("#hook-eye", {{ opacity: 0, y: 12 }}, {{ opacity: 1, y: 0, duration: 0.5, ease: "power3.out" }}, 0.3);
          tl.fromTo("#hook-l1", {{ opacity: 0, y: 28 }}, {{ opacity: 1, y: 0, duration: 0.6, ease: "power3.out" }}, 0.6);
          tl.fromTo("#hook-l2", {{ opacity: 0, y: 28 }}, {{ opacity: 1, y: 0, duration: 0.6, ease: "power3.out" }}, 1.7);
          tl.fromTo("#hook-a", {{ opacity: 0, y: 60, scale: 0.97 }}, {{ opacity: 1, y: 0, scale: 1, duration: 0.8, ease: "expo.out" }}, 2.6);
          tl.fromTo("#hook-b", {{ opacity: 0, y: 60, scale: 0.97 }}, {{ opacity: 1, y: 0, scale: 1, duration: 0.8, ease: "expo.out" }}, 2.78);
          tl.fromTo(["#hook-la", "#hook-lb"], {{ opacity: 0 }}, {{ opacity: 1, duration: 0.4, stagger: 0.15 }}, 3.2);
          tl.fromTo("#hook-q", {{ opacity: 0, x: -30 }}, {{ opacity: 1, x: 0, duration: 0.6, ease: "power3.out" }}, 4.3);
          tl.fromTo("#hook-ring", {{ opacity: 0, scale: 0.8 }}, {{ opacity: 1, scale: 1, duration: 0.4, ease: "back.out(1.6)" }}, 4.8);
          tl.fromTo("#hook-ring-arc", {{ attr: {{ "stroke-dashoffset": 0 }} }}, {{ attr: {{ "stroke-dashoffset": 364.4 }}, duration: 3.0, ease: "none" }}, 5.0);
          tl.set("#hook-n3", {{ opacity: 0 }}, 6.0); tl.set("#hook-n2", {{ opacity: 1 }}, 6.0);
          tl.set("#hook-n2", {{ opacity: 0 }}, 7.0); tl.set("#hook-n1", {{ opacity: 1 }}, 7.0);
          // the reveal
          tl.to("#hook-ring", {{ opacity: 0, scale: 0.8, duration: 0.3 }}, 8.05);
          tl.to(["#hook-la", "#hook-lb"], {{ opacity: 0, duration: 0.2 }}, 8.1);
          tl.fromTo("#hook-ta", {{ opacity: 0, y: 16 }}, {{ opacity: 1, y: 0, duration: 0.5, ease: "expo.out" }}, 8.2);
          tl.fromTo("#hook-tb", {{ opacity: 0, y: 16 }}, {{ opacity: 1, y: 0, duration: 0.5, ease: "expo.out" }}, 8.35);
          tl.to("#hook-a", {{ boxShadow: "0 0 0 4px hsl(156 62% 52% / 1)", duration: 0.4 }}, 8.2);
          tl.to("#hook-q", {{ opacity: 0, duration: 0.3 }}, 8.05);
          tl.fromTo("#hook-f", {{ opacity: 0, x: -30 }}, {{ opacity: 1, x: 0, duration: 0.6, ease: "power3.out" }}, 9.0);
          // the two frames become one; a wipe shows they are the same view
          tl.to(["#hook-top", "#hook-f", "#hook-ta", "#hook-tb"], {{ opacity: 0, duration: 0.4 }}, 12.4);
          tl.to("#hook-a", {{ x: {ax}, y: {CY - FY}, scale: {S_AB:.5f}, boxShadow: "0 0 0 0px hsl(156 62% 52% / 0)", duration: 1.0, ease: "expo.inOut" }}, 12.6);
          tl.to("#hook-b", {{ x: {bx}, y: {CY - FY}, scale: {S_AB:.5f}, duration: 1.0, ease: "expo.inOut" }}, 12.6);
          tl.fromTo("#hook-a-img", {{ clipPath: "inset(0% 0% 0% 0%)" }}, {{ clipPath: "inset(0% 100% 0% 0%)", duration: 1.6, ease: "power2.inOut" }}, 13.9);
          tl.to("#hook-a-img", {{ clipPath: "inset(0% 50% 0% 0%)", duration: 1.1, ease: "power2.inOut" }}, 15.5);
          tl.fromTo("#hook-wl", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.2 }}, 13.8);
          tl.fromTo("#hook-wl", {{ x: {CW} }}, {{ x: 0, duration: 1.6, ease: "power2.inOut", immediateRender: false }}, 13.9);
          tl.to("#hook-wl", {{ x: {CW / 2}, duration: 1.1, ease: "power2.inOut" }}, 15.5);
          tl.fromTo(["#hook-wlab-a", "#hook-wlab-b"], {{ opacity: 0, y: 10 }}, {{ opacity: 1, y: 0, duration: 0.4, stagger: 0.1 }}, 16.2);
          // bridge: into the model
          tl.to(["#hook-wl", "#hook-wlab-a", "#hook-wlab-b"], {{ opacity: 0, duration: 0.3 }}, 17.2);
          tl.to("#hook-a-img", {{ clipPath: "inset(0% 100% 0% 0%)", duration: 0.5, ease: "power2.in" }}, 17.2);
          tl.set("#hook-bridge", {{ opacity: 1 }}, 17.4);
          tl.fromTo("#hook-b1", {{ opacity: 0, y: 24 }}, {{ opacity: 1, y: 0, duration: 0.6, ease: "power3.out" }}, 17.4);
          tl.fromTo("#hook-b2", {{ opacity: 0, y: 24 }}, {{ opacity: 1, y: 0, duration: 0.6, ease: "power3.out" }}, 18.3);
          tl.fromTo("#hook-full", {{ opacity: 0, x: {CX}, y: {CY}, scale: {CW / 1920:.5f} }}, {{ opacity: 1, x: {CX}, y: {CY}, scale: {CW / 1920:.5f}, duration: 0.01 }}, 19.6);
          tl.set(["#hook-a", "#hook-b"], {{ opacity: 0 }}, 19.62);
          tl.to("#hook-bridge", {{ opacity: 0, y: -30, duration: 0.5, ease: "power2.in" }}, 19.9);
          tl.to("#hook-full", {{ x: 0, y: 0, scale: 1, duration: 1.3, ease: "expo.inOut" }}, 19.9);
          // hand-off: the render dissolves into the live model in the app
          tl.to("#hook-full", {{ opacity: 0, filter: "blur(8px)", duration: {X:.2f}, ease: "power2.inOut" }}, {HOOK_D - X:.3f});
"""
    return subcomp(cid, HOOK_D, body, script)


# ------------------------------------------------------------------ act 3: the features
DEMO = "assets/video/demo.mp4"
MINIS = "assets/video/minis.mp4"
MODE_WHAT = {
    "Explore": "Look around the 3D model and see what it is made of.",
    "Measure": "Distances, heights, areas and volumes, by clicking on the model.",
    "Inspect": "Log defects, find every photo that saw a spot, and read the ground.",
    "Operations": "Flooding, damage, debris, vehicle access and what a post can see.",
    "Twin": "The site as it looks, as it was measured, and how sure each part is.",
    "Mission": "Mark positions and routes, see what can be seen, then rehearse.",
    "Walk": "Step inside the scene in first person.",
    "Export": "Pick the files you need and take them away in one zip.",
    "Plan": "Draw buildings, roads and zones and check them against the rules.",
    "Place": "Drop real-size furniture or your own 3D models into the scene.",
}
# playback rate per demo.json event index: results at 1x, tours and typing faster
RATE = {2: 1.3, 3: 2.2, 4: 1.0, 5: 1.15, 6: 1.3, 7: 1.0,
        8: 1.6, 9: 2.2, 10: 1.0, 11: 2.2,
        12: 1.6, 13: 2.6, 14: 1.0, 15: 1.0, 16: 1.0, 17: 2.2, 18: 1.2,
        19: 1.6, 20: 2.6, 21: 1.0, 22: 1.0, 23: 2.6, 24: 1.6, 25: 1.6, 26: 1.2, 27: 1.0,
        28: 1.6, 29: 1.0, 30: 1.0, 31: 1.0, 32: 1.3,
        33: 2.2, 34: 1.2, 35: 1.6, 36: 1.6, 37: 1.7, 38: 1.0, 39: 1.0,
        40: 1.6, 41: 1.7, 42: 1.0, 43: 1.0, 44: 1.0, 45: 1.0,
        46: 1.6, 47: 1.2, 48: 2.2, 49: 1.5,
        51: 1.0,
        52: 2.2, 53: 2.2, 54: 1.6, 55: 1.0, 56: 2.2, 57: 1.7, 58: 1.2, 59: 1.0,
        60: 1.6, 61: 1.3, 62: 1.3, 63: 1.3, 64: 1.0}
SKIP = {0, 1, 50}                     # Projects page: it lists the personal room scan
CLAMP = {2: (7.27, None), 51: (227.9, None)}
CARD_AT = {3: ("01", "Explore"), 8: ("02", "Measure"), 12: ("03", "Inspect"), 19: ("04", "Operations"),
           28: ("05", "Twin"), 33: ("06", "Mission"), 40: ("07", "Walk"), 46: ("08", "Export"),
           52: ("09", "Plan"), 60: ("10", "Place")}

VIEW = (0, 70, 1510)
WIDE = (192, 0, 1728)
MINI = {  # name: (what, crop, scene)
    "Object types": ("Ground, rock, trees and obstacles, labelled.", VIEW, ""),
    "Distance": ("Straight line or a path.", VIEW, ""),
    "Area": ("Trace around a shape.", VIEW, ""),
    "Volume": ("A pile above the ground.", VIEW, ""),
    "Note": ("Mark something you saw.", VIEW, ""),
    "Cut a section": ("A true-scale profile through the site.", WIDE, ""),
    "Archive record": ("How the model was made, a checksum for every file.", (960, 60, 960), ""),
    "Road access": ("A route a vehicle can still drive.", VIEW, ""),
    "Map tiles": ("Height-map and point-cloud tiles for GIS.", WIDE, ""),
    "Relief camp": ("Tents, a medical tent and a helipad on the scan.", VIEW, "Open ground"),
    "Site logistics": ("A tower crane and its jib circle.", VIEW, "Open ground"),
    "Enemy guess": ("Where an enemy would watch the route from.", WIDE, ""),
    "Brief": ("A sand-table view for the briefing.", (0, 0, 1728), ""),
    "Rehearse": ("Walk the mission at night, goggles on.", (0, 100, 1600), ""),
    "Landing zones": ("Flat, clear circles big enough for a helicopter.", WIDE, "Open ground"),
    "Rules": ("A plot's height limit, and the building that breaks it.", WIDE, ""),
    "Impact": ("What the scheme changes against the scan.", WIDE, ""),
    "Split": ("Existing and proposed, side by side.", VIEW, ""),
    "Share": ("GeoJSON, CityJSON, DXF or 3D Tiles.", WIDE, ""),
}
CARD = {
    "Camera coverage": "How well each area was seen.",
    "Pole tilt": "Click the foot of a pole or mast to see how far it leans.",
    "Cable sag": "Click both ends of a cable: sag and ground clearance.",
    "What moved": "Compare with an older scan: surfaces that moved.",
    "Seasonal change": "Compare with an earlier season: erosion and movement.",
    "Restoration idea": "Sketch a missing wall or tower, marked as a guess.",
    "What changed": "Compare with the flight before the event.",
    "Building damage": "Grade every building: intact, partly damaged, collapsed.",
    "Debris volume": "Outline a debris pile or landslide for its volume.",
    "People & vehicles": "Put what the video saw onto the map.",
    "First map": "The quick map made while the drone is still flying.",
    "Stockpile volume": "A stockpile's volume against the earlier flight.",
    "Cut & fill": "Earth still to dig or fill to reach the design.",
    "Built vs design": "What was built, against the BIM or CAD model.",
}
BREAKS = [  # (after event, id, chapter number, name, duration, minis, cards)
    (7, "brk-explore", "01", "Explore", 7.0, ["Object types"], ["Camera coverage"]),
    (11, "brk-measure", "02", "Measure", 9.0, ["Distance", "Area", "Volume", "Note"], []),
    (18, "brk-inspect", "03", "Inspect", 9.0, ["Cut a section", "Archive record"],
     ["Pole tilt", "Cable sag", "What moved", "Seasonal change", "Restoration idea"]),
    (27, "brk-ops", "04", "Operations", 11.0, ["Road access", "Map tiles", "Relief camp", "Site logistics"],
     ["What changed", "Building damage", "Debris volume", "People & vehicles", "First map",
      "Stockpile volume", "Cut & fill", "Built vs design"]),
    (39, "brk-mission", "06", "Mission", 10.0, ["Enemy guess", "Brief", "Rehearse", "Landing zones"], []),
    (59, "brk-plan", "09", "Plan", 9.0, ["Rules", "Impact", "Split", "Share"], []),
]


def capabilities():
    """The demo take, re-timed, plus where each feature break sits (local time)."""
    ev = json.loads((RAW / "demo.json").read_text())["events"]
    minis = {c["title"][6:]: c for c in json.loads((RAW / "minis.json").read_text())["chapters"]
             if c["title"].startswith("mini:")}
    brk_after = {b[0]: b for b in BREAKS}
    t = 0.0
    clips, cards, holds, breaks, story = [], [], [], [], []
    for i, e in enumerate(ev):
        if i in SKIP:
            continue
        a, b = e["start"], e["end"]
        lo, hi = CLAMP.get(i, (None, None))
        a = lo if lo else a
        rate = RATE[i]
        d = (b - a) / rate
        if i in CARD_AT:
            cards.append((t, *CARD_AT[i]))
        if i == 51:
            cards.append((t, "", "Open ground"))
        clips.append((f"cap-v{i:02d}", t, d, a, rate))
        t += d
        if i in brk_after:
            _, bid, num, name, bd, mlist, clist = brk_after[i]
            sp = RAW.parent / "film" / "assets" / "stills" / f"{bid}.jpg"
            still(RAW / "demo.mp4", b - 1.0 / FPS, sp)
            holds.append((f"cap-h-{bid}", t, bd, f"assets/stills/{bid}.jpg"))
            breaks.append((bid, t, bd, num, name, [(m, minis[m]) for m in mlist], clist))
            t += bd
    cap_d = t

    body = ['        <div id="cap-stage" style="position:absolute; inset:0; transform-origin:50% 50%">']
    for vid, s, d, ms, r in clips:
        body.append("          " + video(vid, DEMO, s, d, ms, r, "clip",
                                        "position:absolute; left:0; top:0; width:1920px; height:1080px"))
    for hid, s, d, src in holds:
        body.append(f'          <img id="{hid}" class="clip" src="{src}" data-start="{s:.3f}" data-duration="{d:.3f}" '
                    f'style="position:absolute; left:0; top:0; width:1920px; height:1080px" />')
    body.append("        </div>")
    script = []
    spaced, last = [], 0.9
    for s0, num, name in cards:
        s0 = max(s0, last + 4.5)
        spaced.append((s0, num, name))
        last = s0
    cards = spaced
    for k, (s, num, name) in enumerate(cards):
        cid = f"cap-card{k}"
        if name == "Open ground":
            eye, title, what = "A second site", "Open ground", "Generated, not scanned: 100 m of flat ground for planning."
        else:
            eye, title, what = num, name, MODE_WHAT[name]
        body.append(f"""        <div id="{cid}" class="clip" data-start="{s:.3f}" data-duration="4.2" style="position:absolute; left:40px; top:812px; width:900px; height:220px">
          <div id="{cid}-in" style="position:absolute; left:0; bottom:0; max-width:900px; padding:22px 28px; background:hsl(222 12% 8% / .94); border:2px solid var(--line); border-radius:14px">
            <div class="eyebrow">{esc(eye)}</div>
            <div style="font-size:48px; font-weight:750; letter-spacing:-0.03em; margin-top:4px">{esc(title)}</div>
            <div style="font-size:26px; color:var(--dim); margin-top:6px">{esc(what)}</div>
          </div>
        </div>""")
        script.append(f'          tl.fromTo("#{cid}-in", {{ opacity: 0, x: -40 }}, {{ opacity: 1, x: 0, duration: 0.5, ease: "power3.out" }}, {s + 0.15:.3f});')
        script.append(f'          tl.to("#{cid}-in", {{ opacity: 0, x: -20, duration: 0.4, ease: "power2.in" }}, {s + 3.7:.3f});')
    # the intro card over the swing round
    body.append("""        <div id="cap-intro" class="clip" data-start="0" data-duration="5.4" style="position:absolute; left:40px; top:812px; width:900px; height:220px">
          <div id="cap-intro-in" style="position:absolute; left:0; bottom:0; padding:22px 28px; background:hsl(222 12% 8% / .94); border:2px solid var(--line); border-radius:14px">
            <div class="eyebrow">The scan</div>
            <div style="font-size:48px; font-weight:750; letter-spacing:-0.03em; margin-top:4px">A 12-second drone flight</div>
            <div style="font-size:26px; color:var(--dim); margin-top:6px">72 photos, one 3D model. Everything here runs on it.</div>
          </div>
        </div>""")
    script.append('          tl.fromTo("#cap-intro-in", { opacity: 0, x: -40 }, { opacity: 1, x: 0, duration: 0.5, ease: "power3.out" }, 0.9);')
    script.append('          tl.to("#cap-intro-in", { opacity: 0, x: -20, duration: 0.4, ease: "power2.in" }, 4.9);')
    for bid, s, d, *_ in breaks:
        script.append(f'          tl.to("#cap-stage", {{ scale: 1.05, filter: "blur(18px) brightness(0.45)", duration: 0.7, ease: "power2.inOut" }}, {s:.3f});')
        script.append(f'          tl.to("#cap-stage", {{ scale: 1, filter: "blur(0px) brightness(1)", duration: 0.7, ease: "power2.inOut" }}, {s + d - 0.7:.3f});')
    html = subcomp("capabilities", cap_d, "\n".join(body), "\n".join(script))
    return html, cap_d, breaks, clips


def feature_break(bid, d, num, name, minis, cards):
    n = len(minis)
    RX, RW = 760, 1064
    gap = 24
    cw = (RW - gap) // 2 if n > 1 else RW
    vh = cw / 1.6
    bar = 66
    ch = vh + bar
    rows = math.ceil(n / 2) if n > 1 else 1
    total_h = rows * ch + (rows - 1) * 20
    y0 = 1080 / 2 - total_h / 2 + 40
    P = d - 0.9
    body = [f"""        <div id="{bid}-shade" style="position:absolute; inset:0; background:hsl(225 12% 5% / .55); opacity:0"></div>
        <div id="{bid}-head" style="position:absolute; left:96px; top:110px; width:600px">
          <div class="eyebrow">{num} · {esc(name)}</div>
          <p class="h1" style="margin-top:14px; font-size:{76 if len(name) <= 8 else 60}px">More in {esc(name)}</p>
          <p class="body" style="margin-top:18px; font-size:30px">{esc({1: "One more, recorded on this scan.", 2: "Two more, recorded on this scan."}.get(n, f"{n} more, recorded live."))}</p>
        </div>"""]
    script = [f'          tl.fromTo("#{bid}-shade", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.6, ease: "power2.out" }}, 0);',
              f'          tl.fromTo("#{bid}-head", {{ opacity: 0, x: -40 }}, {{ opacity: 1, x: 0, duration: 0.6, ease: "power3.out" }}, 0.25);']
    for k, (m, ch_) in enumerate(minis):
        what, (cx, cy, cwid), scene = MINI[m]
        col, row = (k % 2, k // 2) if n > 1 else (0, 0)
        x = RX + col * (cw + gap)
        y = y0 + row * (ch + 20)
        L = ch_["end"] - ch_["start"]
        rate = max(0.6, min(3.0, L / P))
        st0 = 0.5 + 0.12 * k
        dk = d - st0
        ms = max(0.0, ch_["end"] - (dk - 0.4) * rate)
        wid = f"{bid}-w{k}"
        tag = f'<span class="mono" style="font-size:20px; color:var(--accent); margin-left:12px">{esc(scene)}</span>' if scene else ""
        vid = video(f"{wid}-v", MINIS, st0, dk, ms, rate)
        body.append(f"""        <div id="{wid}" class="win" style="left:{x:.0f}px; top:{y:.0f}px; width:{cw}px; height:{ch:.0f}px">
          <div style="position:absolute; left:0; top:0; right:0; height:{bar}px; padding:9px 18px; background:var(--raised); border-bottom:2px solid var(--line)">
            <div style="font-size:26px; font-weight:650; letter-spacing:-0.01em">{esc(m)}{tag}</div>
            <div style="font-size:19px; color:var(--dim); margin-top:1px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis">{esc(what)}</div>
          </div>
          <div style="position:absolute; left:0; top:{bar}px; width:{cw}px; height:{vh:.0f}px; overflow:hidden">
            {crop_div(vid, cx, cy, cwid, cw)}
          </div>
        </div>""")
        script.append(f'          tl.fromTo("#{wid}", {{ opacity: 0, y: 50, scale: 0.96 }}, {{ opacity: 1, y: 0, scale: 1, duration: 0.6, ease: "expo.out" }}, {0.35 + 0.12 * k:.2f});')
    if cards:
        cy = 380
        ch2 = 74 if len(cards) > 5 else 86
        body.append(f"""        <div id="{bid}-cards-h" class="mono" style="position:absolute; left:96px; top:{cy - 44}px; font-size:22px; color:var(--muted)">Also here · not shown on these scans</div>""")
        for k, c in enumerate(cards):
            y = cy + k * (ch2 + 8)
            body.append(f"""        <div id="{bid}-c{k}" style="position:absolute; left:96px; top:{y}px; width:600px; height:{ch2}px; padding:10px 18px; border:2px solid var(--line); border-radius:12px; background:hsl(222 12% 8% / .9)">
          <div style="font-size:24px; font-weight:650">{esc(c)}</div>
          <div style="font-size:19px; color:var(--dim); margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis">{esc(CARD[c])}</div>
        </div>""")
        script.append(f'          tl.fromTo("#{bid}-cards-h", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.4 }}, 0.8);')
        script.append(f'          tl.fromTo({json.dumps([f"#{bid}-c{k}" for k in range(len(cards))])}, {{ opacity: 0, x: -30 }}, {{ opacity: 1, x: 0, duration: 0.45, ease: "power3.out", stagger: {min(0.08, 0.5 / len(cards)):.3f} }}, 0.9);')
    # hand back to the footage
    script.append(f'          tl.to("#{bid}", {{ opacity: 0, duration: 0.6, ease: "power2.inOut" }}, {d - 0.65:.3f});')
    return subcomp(bid, d, "\n".join(body), "\n".join(script))


# ------------------------------------------------------------------ act 4: how it's made
RAIL = ["Video", "Keyframes", "Camera solve", "Training", "Physics", "Checks"]


def rail(cid, active):
    items = []
    x = 96
    for k, r in enumerate(RAIL):
        on = k == active
        done = k < active
        col = "var(--accent)" if on else ("var(--dim)" if done else "var(--muted)")
        dot = "var(--accent)" if on else ("var(--dim)" if done else "hsl(222 9% 30%)")
        w = 40 + len(r) * 13
        items.append(f'<div style="position:absolute; left:{x}px; top:0; display:flex; align-items:center; gap:12px; font-family:\'Geist Mono\',monospace; font-size:22px; color:{col}; white-space:nowrap">'
                     f'<span class="dot" style="background:{dot}"></span>{r}</div>')
        x += w + 44
    return f'<div id="{cid}-rail" style="position:absolute; left:0; top:56px; width:1920px; height:30px">{"".join(items)}</div>'


RUN = []          # (name, secs) of every stage in the Boulder field run, set in pipeline()


def runbar(cid, lit):
    """All stages in run order, widths in proportion to their real time; `lit` stages amber."""
    total = sum(sc for _, sc in RUN)
    x, parts = 96.0, []
    avail = 1728.0 - 2 * (len(RUN) - 1) - 4 * len(RUN)
    for nm, sc in RUN:
        w = 4.0 + sc / total * avail
        col = "var(--accent)" if nm in lit else "hsl(222 9% 26%)"
        parts.append(f'<div style="position:absolute; left:{x:.1f}px; top:0; width:{w:.1f}px; height:10px; border-radius:3px; background:{col}"></div>')
        x += w + 2
    idx = [k for k, (nm, _) in enumerate(RUN) if nm in lit]
    secs = sum(RUN[k][1] for k in idx)
    names = " + ".join(RUN[k][0] for k in idx)
    rng = f"{idx[0] + 1}-{idx[-1] + 1}" if len(idx) > 1 else (str(idx[0] + 1) if idx else "")
    label = (f"stage {rng} of {len(RUN)} · {names} · {secs:,.1f} s" if idx
             else f"{len(RUN)} stages · {total / 60:.0f} min in all")
    return (f'<div id="{cid}-run" style="position:absolute; left:0; top:966px; width:1920px; height:60px">{"".join(parts)}'
            f'<div class="mono" style="position:absolute; left:96px; top:24px; font-size:22px; color:var(--dim)">{label}</div></div>')


def pscene(cid, dur, active, left, right, script, extra_css="", lit=None, lw=500):
    body = (rail(cid, active) if active is not None else "") + (runbar(cid, lit) if lit is not None else "") + f"""
        <div id="{cid}-left" style="position:absolute; left:96px; top:176px; width:{lw}px">{left}</div>
        {right}"""
    head = f'          tl.fromTo("#{cid}", {{ opacity: 0 }}, {{ opacity: 1, duration: {X}, ease: "power2.inOut" }}, 0);\n'
    head += f'          tl.fromTo("#{cid}-left > :not(.late)", {{ opacity: 0, y: 26 }}, {{ opacity: 1, y: 0, duration: 0.55, ease: "power3.out", stagger: 0.14 }}, 0.35);\n'
    return subcomp(cid, dur, body, head + script, extra_css)


WX, WY, WW = 640, 176, 1184          # the media window on pipeline scenes
WH = WW * 9 / 16


def window(cid, inner, x=WX, y=WY, w=WW, h=WH):
    return f'<div id="{cid}-win" class="win" style="left:{x}px; top:{y}px; width:{w}px; height:{h:.0f}px">{inner}</div>'


def stat(v, label, vid=None):
    i = f' id="{vid}"' if vid else ""
    return f'<div style="margin-top:34px"><div class="stat"{i}>{v}</div><div class="stat-l">{label}</div></div>'


def pipeline():
    rep = json.loads((FILM / "assets/pipeline/report.json").read_text())
    feats = json.loads((FILM / "assets/pipeline/features.json").read_text())
    log = (FILM / "assets/pipeline/train_log.txt").read_text()
    pts = [(int(a), float(b), int(c)) for a, b, c in re.findall(r"step\s+(\d+) .*?psnr ([\d.]+)\s+N (\d+)", log)]
    steps = [int(s) for s in (FILM / "assets/pipeline/timelapse_steps.txt").read_text().strip(",\n").split(",")]
    secs = {s["name"]: s["secs"] for s in rep["steps"]}
    total = rep["secs"]
    RUN[:] = [(s["name"], s["secs"]) for s in rep["steps"]]
    scenes = []

    # -- title
    cid = "p-title"
    scenes.append((cid, 5.5, pscene(cid, 5.5, None, f"""
          <div class="eyebrow">How it's made</div>
          <p class="h1" style="margin-top:18px; width:1500px">From a 12-second drone clip</p>
          <p class="h1" style="width:1500px; color:var(--dim)">to a world you can walk through.</p>
          <p class="mono" style="margin-top:46px; font-size:28px; color:var(--dim); width:1500px">1 video · 1280×720 · 12.05 s · no GPS log · one laptop (RTX 3050, 6 GB)</p>""", "", "")))

    # -- the input
    cid = "p-input"
    inner = (f'<div class="crop" style="width:1280px; height:720px; transform: scale({WW / 1280:.5f})">'
             + video(f"{cid}-v", "assets/video/rocks.mp4", 0.2, 7.3, 0.0, 1.0, "", "width:1280px; height:720px") + "</div>")
    scenes.append((cid, 7.5, pscene(cid, 7.5, 0, lit=[], left=f"""
          <div class="eyebrow">Input</div>
          <p class="h2" style="margin-top:14px">One pass of drone video.</p>
          <p class="body" style="margin-top:18px">No ground markers, no survey log. Just the footage.</p>
          {stat("12.05 s", "1280×720 at 24 fps")}""",
        right=window(cid, inner),
        script=f'          tl.fromTo("#{cid}-win", {{ opacity: 0, scale: 0.97 }}, {{ opacity: 1, scale: 1, duration: 0.7, ease: "expo.out" }}, 0.2);\n')))

    # -- in the app
    cid = "p-app"
    app = [(4.2, 19.8, 1.5), (19.8, 40.0, 4.0)]
    t = 0.3
    vids = []
    for k, (a, b, r) in enumerate(app):
        d = (b - a) / r
        vids.append(video(f"{cid}-v{k}", "assets/video/pipeline-app.mp4", t, d, a, r))
        t += d
    ta, tb, tr = 5.0, 16.0, 2.2
    dtr = (tb - ta) / tr
    vids.append(video(f"{cid}-v2", "assets/video/pipeline-train.mp4", t, dtr, ta, tr))
    t += dtr
    dur = t + 0.4
    t_proc = 0.3 + (19.8 - 4.2) / 1.5
    scenes.append((cid, dur, pscene(cid, dur, 0, lit=[], left=f"""
          <div class="eyebrow">In the app</div>
          <p class="h2" style="margin-top:14px">Drop the video in. Press Reconstruct.</p>
          <p class="body" style="margin-top:18px">It runs on this machine. Nothing is uploaded.</p>
          <div id="{cid}-live" style="margin-top:40px"><span class="pill"><span class="dot" style="background:var(--accent)"></span>Live log, stage by stage</span></div>""",
        right=window(cid, crop_div("".join(vids), 0, 0, 1920, WW)),
        script=f'          tl.fromTo("#{cid}-win", {{ opacity: 0, scale: 0.97 }}, {{ opacity: 1, scale: 1, duration: 0.7, ease: "expo.out" }}, 0.2);\n')))

    # -- keyframes
    cid = "p-frames"
    thumbs = sorted((FILM / "assets/pipeline/keyframes").glob("*.jpg"))
    cells = []
    tw, th, g = 124, 70, 8
    gx0, gy0 = 640, 176
    for k, p in enumerate(thumbs):
        r, c = divmod(k, 9)
        cells.append(f'<img id="{cid}-t{k}" src="assets/pipeline/keyframes/{p.name}" style="position:absolute; left:{gx0 + c * (tw + g)}px; top:{gy0 + r * (th + g)}px; width:{tw}px; height:{th}px; border-radius:6px; object-fit:cover" />')
    scr = ""
    for r in range(8):
        ids = json.dumps([f"#{cid}-t{k}" for k in range(r * 9, min(72, r * 9 + 9))])
        scr += f'          tl.fromTo({ids}, {{ opacity: 0, y: 24, scale: 0.85 }}, {{ opacity: 1, y: 0, scale: 1, duration: 0.45, ease: "expo.out", stagger: 0.03 }}, {0.6 + r * 0.17:.2f});\n'
    scr += f'          tl.to("#{cid}-t41", {{ scale: 1.5, zIndex: 5, boxShadow: "0 0 0 3px hsl(36 100% 57%)", duration: 0.5, ease: "back.out(1.7)" }}, 3.4);\n'
    scr += f'          tl.fromTo("#{cid}-n", {{ textContent: 0 }}, {{ textContent: 72, duration: 1.6, ease: "power2.out", snap: {{ textContent: 1 }} }}, 0.6);\n'
    scenes.append((cid, 7.0, pscene(cid, 7.0, 1, lit=["keyframes"], left=f"""
          <div class="eyebrow">Keyframes · {secs['keyframes']:.1f} s</div>
          <p class="h2" style="margin-top:14px">The sharpest frames are kept.</p>
          <p class="body" style="margin-top:18px">Blurred or duplicate frames are dropped before anything is solved.</p>
          <div style="margin-top:34px"><div class="stat" id="{cid}-n">72</div><div class="stat-l">keyframes from 12 s of video</div></div>
          <p class="mono" style="margin-top:30px; font-size:24px; color:var(--accent)">frame 41 is the quiz frame</p>""",
        right="\n        ".join(cells), script=scr)))

    # -- camera solve
    cid = "p-solve"
    fa, fb = feats["frames"]["a"], feats["frames"]["b"]
    IW = 580
    IH = IW * 9 / 16
    ax_, bx_, iy = 640, 640 + IW + 24, 300
    dots = "".join(f'<circle cx="{p[0] * IW:.1f}" cy="{p[1] * IH:.1f}" r="{max(1.6, min(4.5, p[2] / 3)):.1f}"/>' for p in fb["points"][:900])
    lines = "".join(f'<line x1="{ax_ + m[0] * IW:.1f}" y1="{iy + m[1] * IH:.1f}" x2="{bx_ + m[2] * IW:.1f}" y2="{iy + m[3] * IH:.1f}"/>' for m in feats["matches"]["lines"][:70])
    solve_v0 = 14.2
    vd = (25.15 - solve_v0) / 1.5
    right = f"""
        <img id="{cid}-a" src="assets/pipeline/frame35.jpg" style="position:absolute; left:{ax_}px; top:{iy}px; width:{IW}px; height:{IH:.0f}px; border-radius:10px; opacity:0" />
        <div id="{cid}-b" style="position:absolute; left:{bx_}px; top:{iy}px; width:{IW}px; height:{IH:.0f}px">
          <img src="assets/pipeline/frame41.jpg" style="position:absolute; inset:0; width:100%; height:100%; border-radius:10px" />
          <svg id="{cid}-dots" width="{IW}" height="{IH:.0f}" style="position:absolute; inset:0; fill:hsl(36 100% 57%)">{dots}</svg>
        </div>
        <svg id="{cid}-lines" width="1920" height="1080" style="position:absolute; left:0; top:0; stroke:hsl(36 100% 57% / .4); stroke-width:1.6">{lines}</svg>
        <div id="{cid}-la" class="mono" style="position:absolute; left:{ax_}px; top:{iy + IH + 16:.0f}px; font-size:22px; color:var(--dim); opacity:0">frame 35 · {fa['total']:,} features</div>
        <div id="{cid}-lb" class="mono" style="position:absolute; left:{bx_}px; top:{iy + IH + 16:.0f}px; font-size:22px; color:var(--dim); opacity:0">frame 41 · {fb['total']:,} features</div>
        {window(cid, crop_div(video(f"{cid}-v", "assets/video/pipeline-viewer.mp4", 7.4, vd, solve_v0, 1.5), 0, 0, 1920, WW), y=WY)}"""
    scr = f"""          tl.fromTo("#{cid}-b", {{ opacity: 0, scale: 0.96 }}, {{ opacity: 1, scale: 1, duration: 0.6, ease: "expo.out" }}, 0.3);
          tl.fromTo("#{cid}-dots circle", {{ opacity: 0, scale: 0, transformOrigin: "50% 50%" }}, {{ opacity: 1, scale: 1, duration: 0.25, stagger: {{ each: 0.0015, from: "random" }} }}, 0.9);
          tl.fromTo("#{cid}-lb", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.4 }}, 1.4);
          tl.fromTo("#{cid}-a", {{ opacity: 0, x: -40 }}, {{ opacity: 1, x: 0, duration: 0.6, ease: "power3.out" }}, 3.0);
          tl.fromTo("#{cid}-la", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.4 }}, 3.3);
          tl.fromTo("#{cid}-lines line", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.3, stagger: {{ each: 0.018, from: "random" }} }}, 3.7);
          tl.to(["#{cid}-a", "#{cid}-b", "#{cid}-lines", "#{cid}-la", "#{cid}-lb"], {{ opacity: 0, filter: "blur(8px)", duration: 0.6, ease: "power2.inOut" }}, 6.9);
          tl.fromTo("#{cid}-win", {{ opacity: 0, scale: 0.97 }}, {{ opacity: 1, scale: 1, duration: 0.6, ease: "power2.out" }}, 7.2);
          tl.fromTo("#{cid}-s2", {{ opacity: 0, y: 20 }}, {{ opacity: 1, y: 0, duration: 0.5 }}, 7.6);
"""
    dur = 7.4 + vd + 0.3
    scenes.append((cid, dur, pscene(cid, dur, 2, lit=["colmap", "poses"], left=f"""
          <div class="eyebrow">Camera solve · COLMAP · {secs['colmap'] / 60:.0f} min</div>
          <p class="h2" style="margin-top:14px">The same points, found in many photos.</p>
          <p class="body" style="margin-top:18px">Matching them places every camera in 3D.</p>
          {stat(f"{feats['matches']['total']:,}", "verified matches between these two frames")}
          <div id="{cid}-s2" class="late">{stat("72 / 72", f"cameras placed · {feats['scene']['verified_matches'] / 1e6:.2f} M matches")}</div>""",
        right=right, script=scr)))

    # -- training timelapse
    cid = "p-train"
    rate = 0.55
    tl_d = (len(steps) / 30.0) / rate
    t0 = 0.4
    curve_w, curve_h = 460, 150
    smax, pmin, pmax = 15000, 15.0, 45.0
    poly = " ".join(f"{s / smax * curve_w:.1f},{curve_h - (p - pmin) / (pmax - pmin) * curve_h:.1f}" for s, p, n in pts)
    ns = json.dumps([[s, n] for s, _, n in pts])
    inner = f'<div class="crop" style="transform: scale({WW / 1920:.5f})">' + video(f"{cid}-v", "assets/pipeline/timelapse.mp4", t0, tl_d, 0.0, rate) + "</div>"
    right = window(cid, inner) + f"""
        <div id="{cid}-step" class="mono" style="position:absolute; left:{WX + 24}px; top:{WY + WH - 64:.0f}px; padding:8px 16px; border-radius:10px; background:hsl(225 12% 5% / .8); font-size:28px">step 1</div>"""
    scr = f"""          tl.fromTo("#{cid}-win", {{ opacity: 0, scale: 0.97 }}, {{ opacity: 1, scale: 1, duration: 0.6, ease: "expo.out" }}, 0.2);
          const steps = {json.dumps(steps)}, ns = {ns};
          const nAt = (s) => {{ let a = ns[0]; for (const p of ns) {{ if (p[0] > s) {{ const f = (s - a[0]) / (p[0] - a[0] || 1); return a[1] + f * (p[1] - a[1]); }} a = p; }} return a[1]; }};
          const st = {{ f: 0 }};
          const stepEl = document.getElementById("{cid}-step"), nEl = document.getElementById("{cid}-gn");
          tl.fromTo(st, {{ f: 0 }}, {{ f: {len(steps) - 1}, duration: {tl_d:.3f}, ease: "none",
            onUpdate: () => {{ const s = steps[Math.min(steps.length - 1, Math.floor(st.f))];
              stepEl.textContent = "step " + s.toLocaleString("en-US") + " / 15,000";
              nEl.textContent = Math.round(nAt(Math.max(200, s))).toLocaleString("en-US"); }} }}, {t0});
          tl.fromTo("#{cid}-curve", {{ strokeDashoffset: 2000 }}, {{ strokeDashoffset: 0, duration: {tl_d:.3f}, ease: "none" }}, {t0});
"""
    dur = t0 + tl_d + 1.2
    scenes.append((cid, dur, pscene(cid, dur, 3, lit=["train"], left=f"""
          <div class="eyebrow">Training · {secs['train'] / 60:.0f} min on the GPU</div>
          <p class="h2" style="margin-top:14px">The model learns to look like the photos.</p>
          <p class="body" style="margin-top:18px">Frame 41's camera, every 100 steps.</p>
          <div style="margin-top:34px"><div class="stat" id="{cid}-gn">47,240</div><div class="stat-l">Gaussians, the model's building blocks</div></div>
          <div style="margin-top:34px"><svg width="{curve_w}" height="{curve_h}" style="overflow:visible"><polyline id="{cid}-curve" points="{poly}" fill="none" stroke="hsl(36 100% 57%)" stroke-width="3" stroke-dasharray="2000" stroke-dashoffset="2000"/></svg>
            <div class="stat-l">PSNR on the training photos, per step (noisy: a different photo each step)</div></div>""",
        right=right, script=scr)))

    # -- physics
    cid = "p-physics"
    v0, v1, r = 25.15, 34.18, 1.35
    vd = (v1 - v0) / r
    dur = vd + 0.6
    scenes.append((cid, dur, pscene(cid, dur, 4, lit=["collider", "surface", "nav"], left=f"""
          <div class="eyebrow">Physics · {secs['collider'] + secs['surface'] + secs['nav']:.0f} s</div>
          <p class="h2" style="margin-top:14px">A solid surface under the picture.</p>
          <p class="body" style="margin-top:18px">What you measure on, walk on, and bump into.</p>
          {stat("360,714", "triangles in the collision mesh")}""",
        right=window(cid, crop_div(video(f"{cid}-v", "assets/video/pipeline-viewer.mp4", 0.2, vd, v0, r), 0, 0, 1920, WW)),
        script=f'          tl.fromTo("#{cid}-win", {{ opacity: 0, scale: 0.97 }}, {{ opacity: 1, scale: 1, duration: 0.6, ease: "expo.out" }}, 0.2);\n')))

    # -- checks
    cid = "p-checks"
    inner = (f'<img src="assets/pipeline/eval_real.jpg" style="position:absolute; inset:0; width:100%; height:100%; object-fit:cover" />'
             f'<img id="{cid}-r" src="assets/pipeline/eval_render.png" style="position:absolute; inset:0; width:100%; height:100%; object-fit:cover" />')
    right = window(cid, inner) + f"""
        <div id="{cid}-line" style="position:absolute; left:{WX}px; top:{WY - 10}px; width:4px; height:{WH + 20:.0f}px; background:var(--accent); border-radius:2px"></div>
        <div id="{cid}-l1" class="pill" style="position:absolute; left:{WX + 24}px; top:{WY + WH - 80:.0f}px"><span class="dot" style="background:var(--green)"></span>Photo</div>
        <div id="{cid}-l2" class="pill" style="position:absolute; left:{WX + WW - 190}px; top:{WY + WH - 80:.0f}px"><span class="dot" style="background:var(--accent)"></span>Render</div>"""
    scr = f"""          tl.fromTo("#{cid}-win", {{ opacity: 0, scale: 0.97 }}, {{ opacity: 1, scale: 1, duration: 0.6, ease: "expo.out" }}, 0.2);
          tl.fromTo("#{cid}-r", {{ clipPath: "inset(0% 0% 0% 100%)" }}, {{ clipPath: "inset(0% 0% 0% 0%)", duration: 1.8, ease: "power2.inOut" }}, 0.8);
          tl.to("#{cid}-r", {{ clipPath: "inset(0% 0% 0% 50%)", duration: 1.2, ease: "power2.inOut" }}, 2.8);
          tl.fromTo("#{cid}-line", {{ x: {WW} }}, {{ x: 0, duration: 1.8, ease: "power2.inOut" }}, 0.8);
          tl.to("#{cid}-line", {{ x: {WW / 2}, duration: 1.2, ease: "power2.inOut" }}, 2.8);
          tl.fromTo(["#{cid}-l1", "#{cid}-l2"], {{ opacity: 0 }}, {{ opacity: 1, duration: 0.4, stagger: 0.1 }}, 3.4);
"""
    ps = json.loads((ROOT / "work/boulder-field-0aa9375f/train_progress/train_report.json").read_text())["in_sample"]["psnr"]
    scenes.append((cid, 7.0, pscene(cid, 7.0, 5, lit=["gate", "audit", "evals", "pairs", "walktest"], left=f"""
          <div class="eyebrow">Checks · {secs['gate'] + secs['audit'] + secs['evals'] + secs['pairs'] + secs['walktest']:.0f} s</div>
          <p class="h2" style="margin-top:14px">Rendered against the real photo.</p>
          <p class="body" style="margin-top:18px">Same camera, frame 42. Then a bot walks the scene to prove it is walkable.</p>
          {stat(f"{ps:.1f} dB", "PSNR on the training views")}
          <div style="margin-top:34px"><span class="pill"><span class="dot" style="background:var(--green)"></span>Quality gate: all checks passed</span></div>""",
        right=right, script=scr)))

    # -- the whole run
    cid = "p-summary"
    top = sorted(rep["steps"], key=lambda s: -s["secs"])[:5]
    rest = total - sum(s["secs"] for s in top)
    rows = [(s["name"], s["secs"]) for s in top] + [(f"{len(rep['steps']) - 5} more stages", rest)]
    bw = 900
    bars = []
    for k, (nm, sc) in enumerate(rows):
        y = 300 + k * 104
        w = max(6, sc / top[0]["secs"] * bw)
        bars.append(f"""<div id="{cid}-r{k}" style="position:absolute; left:840px; top:{y}px; width:980px; height:90px">
          <div style="font-size:28px; font-weight:600">{esc(nm)}<span class="mono" style="color:var(--dim); margin-left:16px; font-size:24px">{sc:,.1f} s</span></div>
          <div style="position:absolute; left:0; top:48px; width:{bw}px; height:22px; border-radius:6px; background:var(--panel)"></div>
          <div id="{cid}-b{k}" style="position:absolute; left:0; top:48px; width:{w:.0f}px; height:22px; border-radius:6px; background:var(--accent); transform-origin:0 50%"></div>
        </div>""")
    scr = f'          tl.fromTo({json.dumps([f"#{cid}-r{k}" for k in range(len(rows))])}, {{ opacity: 0, x: 30 }}, {{ opacity: 1, x: 0, duration: 0.5, ease: "power3.out", stagger: 0.08 }}, 0.5);\n'
    scr += f'          tl.fromTo({json.dumps([f"#{cid}-b{k}" for k in range(len(rows))])}, {{ scaleX: 0 }}, {{ scaleX: 1, duration: 0.9, ease: "expo.out", stagger: 0.08 }}, 0.7);\n'
    mm, ss = divmod(round(total), 60)
    scenes.append((cid, 7.5, pscene(cid, 7.5, None, lw=700, left=f"""
          <div class="eyebrow">The whole run</div>
          <div class="stat" style="margin-top:24px; font-size:112px; white-space:nowrap">{mm} min {ss} s</div>
          <div class="stat-l" style="font-size:32px">from a video file to a 3D world</div>
          <p class="body" style="margin-top:40px">{len(rep['steps'])} stages, one RTX 3050 laptop GPU (6 GB), start to finish.</p>""",
        right="\n        ".join(bars), script=scr)))

    # -- ready, and back to frame 41
    cid = "p-ready"
    v0, v1 = 4.2, 9.89
    vd = v1 - v0
    dur = vd + 3.0
    scenes.append((cid, dur, pscene(cid, dur, None, f"""
          <div class="eyebrow">Ready</div>
          <p class="h2" style="margin-top:14px">Open it. Click photo 41.</p>
          <p id="{cid}-cb" class="h2 late" style="margin-top:30px; color:var(--accent)">The frame from the start.</p>""",
        window(cid, crop_div(video(f"{cid}-v", "assets/video/pipeline-viewer.mp4", 0.2, vd + 2.6, v0, 1.0), 0, 0, 1920, WW)),
        f'          tl.fromTo("#{cid}-win", {{ opacity: 0, scale: 0.97 }}, {{ opacity: 1, scale: 1, duration: 0.6, ease: "expo.out" }}, 0.2);\n'
        f'          tl.fromTo("#{cid}-cb", {{ opacity: 0, y: 20 }}, {{ opacity: 1, y: 0, duration: 0.6, ease: "power3.out", immediateRender: false }}, {vd + 0.3:.2f});\n')))
    return scenes


def close():
    cid = "close"
    d = 9.0
    body = """
        <div id="close-in" style="position:absolute; left:96px; top:300px; width:1728px">
          <div id="close-e" class="eyebrow">Ground Control</div>
          <p id="close-t" class="h1" style="margin-top:22px; font-size:104px">One drone video in.</p>
          <p id="close-t2" class="h1" style="font-size:104px; color:var(--dim)">A 3D world you can measure, plan on and walk out.</p>
          <p id="close-m" class="mono" style="margin-top:60px; font-size:28px; color:var(--dim)">SIH 2026 · SIH26158 · github.com/krisgarg25/Drone_Phone_video_to_playable_3d_world</p>
        </div>"""
    script = """          tl.fromTo("#close", { opacity: 0 }, { opacity: 1, duration: 0.6, ease: "power2.inOut" }, 0);
          tl.fromTo(["#close-e", "#close-t", "#close-t2", "#close-m"], { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.7, ease: "power3.out", stagger: 0.25 }, 0.4);
          tl.to("#close-in", { opacity: 0, duration: 1.0, ease: "power2.inOut" }, 7.8);
"""
    return subcomp(cid, d, body, script), d


# ------------------------------------------------------------------ assemble
def main():
    comps = FILM / "compositions"
    comps.mkdir(exist_ok=True)
    hosts = []
    story = []

    def add(cid, start, dur, html, title, scene, track):
        (comps / f"{cid}.html").write_text(html, encoding="utf-8")
        hosts.append((cid, start, dur, track))
        story.append((cid, title, start, dur, scene))

    add("hook", 0.0, HOOK_D, hook(), "Real or rendered?", "Frame 41 against its render; the reveal; the bridge into the model", 1)
    cap_html, cap_d, breaks, _ = capabilities()
    cap0 = HOOK_D - X
    add("capabilities", cap0, cap_d, cap_html, "What it can do", "The feature take, re-timed, with chapter cards", 0)
    for bid, s, d, num, name, minis, cards in breaks:
        add(bid, cap0 + s, d, feature_break(bid, d, num, name, minis, cards), f"More in {name}",
            f"{len(minis)} minis, {len(cards)} cards", 2)
    t = cap0 + cap_d - X
    for cid, d, html in pipeline():
        add(cid, t, d, html, cid, "How it's made", 3)
        t += d - X
    chtml, cd = close()
    add("close", t, cd, chtml, "Close", "Title and repo", 3)
    total = t + cd

    slots = []
    for z, (cid, s, d, track) in enumerate(hosts):
        slots.append(f'      <div id="slot-{cid}" data-composition-id="{cid}" data-composition-src="compositions/{cid}.html" '
                     f'data-start="{s:.3f}" data-duration="{d:.3f}" data-track-index="{track}" data-width="1920" data-height="1080" '
                     f'style="z-index:{z + 1}"></div>')
    index = f"""<!doctype html>
<html lang="en" data-resolution="landscape">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=1920, height=1080" />
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
    <style>
      body {{ margin: 0; background: hsl(225 12% 5%); }}
      #root {{ position: relative; width: 100%; height: 100%; overflow: hidden; background: hsl(225 12% 5%); }}
      #root > div[data-composition-src] {{ position: absolute; inset: 0; }}
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="root" data-start="0" data-width="1920" data-height="1080" data-fps="{FPS}" data-duration="{total:.3f}">
{chr(10).join(slots)}
    </div>
    <script>
      window.__timelines["root"] = gsap.timeline({{ paused: true }});
    </script>
  </body>
</html>
"""
    (FILM / "index.html").write_text(index, encoding="utf-8")

    sb = ["---", "format: 1920x1080", f"duration: {total:.0f}s", "message: \"One drone video becomes a 3D world you can measure, plan on and walk through, on one laptop.\"",
          "arc: Hook (real or rendered?) → Capabilities → How it's made → Close", "audience: SIH 2026 judges", "mode: autonomous", "---", ""]
    for k, (cid, title, s, d, scene) in enumerate(story, 1):
        sb += [f"## Frame {k} — {title}", "- status: animated", f"- src: compositions/{cid}.html",
               f"- duration: {d:.1f}s", f"- scene: {scene} (starts {s:.1f}s)", ""]
    (FILM / "STORYBOARD.md").write_text("\n".join(sb), encoding="utf-8")
    m, s = divmod(total, 60)
    print(f"total {int(m)}:{s:04.1f}  capabilities {cap_d:.1f}s  breaks {len(breaks)}  scenes {len(hosts)}")


if __name__ == "__main__":
    main()
