"""Build the film's compositions from the recordings' own logs.

    .venv/Scripts/python.exe video_production/film/build/build.py

Writes index.html, compositions/*.html, STORYBOARD.md and the cut footage in assets/cut/.
Every cut point comes from raw/demo.json, raw/minis-clean.json and raw/pipeline-*.json, and
every number on screen from the Boulder field run's own files (assets/pipeline/*). Edit this
script, not the HTML.

Footage is pre-cut with ffmpeg, frame-exact on the 60 fps grid:
* the whole feature take becomes ONE continuous file (assets/cut/cap-*.mp4), re-timed, with
  the frame held still under every feature break. One <video> element plays it, so there is
  no clip boundary where a wrong frame can flash between chapters;
* every mini is its own cropped file that ends on a held frame, so a window never plays
  into the set-up moves that follow it in the recording.
"""
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from illos import illo  # noqa: E402

FILM = Path(__file__).resolve().parents[1]
RAW = FILM.parent / "raw"
ROOT = FILM.parents[2]
FFMPEG = str(ROOT / "tools/ffmpeg/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe")
CUT = FILM / "assets" / "cut"
FPS = 60
X = 0.6                       # overlap between scenes (s)


def nf(t):
    return int(round(t * FPS))


# ------------------------------------------------------------------ footage cutting
ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "15", "-pix_fmt", "yuv420p",
       "-r", str(FPS), "-g", str(FPS), "-an"]


def _run(args):
    subprocess.run([FFMPEG, "-hide_banner", "-v", "error", "-y", *args], check=True)


def seg(src, a, rate, n_play, n_hold, tag, pre=""):
    """Source from `a`, sped up by `rate`, exactly n_play frames, then the last frame held
    for n_hold frames. Cached by its parameters."""
    key = hashlib.md5(repr((str(src), round(a, 4), round(rate, 4), n_play, n_hold, pre, ENC)).encode()).hexdigest()[:10]
    out = CUT / "seg" / f"{tag}-{key}.mp4"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    span = n_play / FPS * rate + 0.3
    vf = (f"{pre}setpts=(PTS-STARTPTS)/{rate:.5f},fps={FPS},trim=end_frame={n_play},setpts=PTS-STARTPTS,"
          f"tpad=stop_mode=clone:stop_duration={(n_hold + 6) / FPS:.4f}")
    _run(["-ss", f"{a:.4f}", "-t", f"{span:.4f}", "-i", str(src), "-vf", vf,
          "-frames:v", str(n_play + n_hold), *ENC, str(out)])
    return out


def concat(parts, name):
    key = hashlib.md5("|".join(p.name for p in parts).encode()).hexdigest()[:10]
    out = CUT / f"{name}-{key}.mp4"
    if out.exists():
        return out
    lst = CUT / f"{name}.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    _run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
    return out


def frame_jpg(src, t, out, w=None):
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = ["-vf", f"scale={w}:-2"] if w else []
    _run(["-ss", f"{t:.4f}", "-i", str(src), "-frames:v", "1", *vf, "-q:v", "2", str(out)])
    return out


def rel(p):
    return p.relative_to(FILM).as_posix()


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
#ROOT { position: absolute; inset: 0; overflow: hidden; background: %BG%; color: var(--ink);
  font-family: "Geist", sans-serif; %TOKENS% }
#ROOT * { box-sizing: border-box; }
#ROOT .mono { font-family: "Geist Mono", monospace; }
#ROOT .eyebrow { font-family: "Geist Mono", monospace; font-size: 24px; letter-spacing: 0.02em; color: var(--accent);
  display: flex; align-items: center; gap: 16px; }
#ROOT .eyebrow .rule { display: inline-block; width: 44px; height: 2px; background: var(--accent); transform-origin: 0 50%; }
#ROOT .h1 { font-size: 84px; font-weight: 750; letter-spacing: -0.035em; line-height: 1.08; margin: 0 0 8px; }
#ROOT .h2 { font-size: 56px; font-weight: 700; letter-spacing: -0.03em; line-height: 1.12; margin: 0 0 4px; }
#ROOT .body { font-size: 32px; font-weight: 350; color: var(--dim); line-height: 1.32; margin: 0; }
#ROOT .acc { color: var(--accent); }
#ROOT .grn { color: var(--green); }
#ROOT .wm { display: inline-block; overflow: hidden; vertical-align: top; padding: 0 0.05em 0.14em; margin: 0 -0.05em -0.14em; }
#ROOT .w { display: inline-block; will-change: transform; }
#ROOT .win { position: absolute; background: var(--panel); border: 2px solid var(--line); border-radius: 16px; overflow: hidden;
  box-shadow: 0 40px 90px -30px hsl(0 0% 0% / .8), 0 0 0 1px hsl(0 0% 100% / .03); }
#ROOT .bar { position: absolute; left: 0; top: 0; right: 0; height: 44px; display: flex; align-items: center; gap: 9px; padding: 0 18px;
  background: var(--raised); border-bottom: 2px solid var(--line); font-family: "Geist Mono", monospace; font-size: 19px; color: var(--dim); }
#ROOT .bar i { width: 11px; height: 11px; border-radius: 50%; background: hsl(222 9% 28%); display: inline-block; }
#ROOT .bar b { font-weight: 500; margin-left: 10px; color: var(--ink); }
#ROOT .crop { position: absolute; left: 0; top: 0; width: 1920px; height: 1080px; transform-origin: 0 0; }
#ROOT .crop > video, #ROOT .crop > img { position: absolute; left: 0; top: 0; width: 1920px; height: 1080px; object-fit: cover; }
#ROOT .stat { font-family: "Geist Mono", monospace; font-size: 76px; font-weight: 600; color: var(--accent); letter-spacing: -0.02em; line-height: 1; }
#ROOT .stat-l { font-size: 26px; color: var(--dim); margin-top: 10px; line-height: 1.3; }
#ROOT .pill { display: inline-flex; align-items: center; gap: 12px; padding: 10px 20px; border-radius: 999px; white-space: nowrap;
  font-size: 26px; font-weight: 600; background: hsl(222 12% 8% / .92); border: 2px solid var(--line); }
#ROOT .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; flex: none; }
#ROOT .amb { position: absolute; inset: 0; overflow: hidden; }
#ROOT .amb-grid { position: absolute; left: -88px; top: -88px; width: 2096px; height: 1256px;
  background-image: radial-gradient(hsl(220 25% 85% / .13) 1.3px, hsl(220 25% 85% / 0) 1.9px); background-size: 44px 44px;
  -webkit-mask-image: radial-gradient(ellipse 70% 65% at 55% 45%, #000 20%, rgba(0,0,0,0) 80%);
          mask-image: radial-gradient(ellipse 70% 65% at 55% 45%, #000 20%, rgba(0,0,0,0) 80%); }
#ROOT .amb-a { position: absolute; left: -520px; top: -700px; width: 1600px; height: 1600px; border-radius: 50%;
  background: radial-gradient(circle, hsl(36 100% 57% / .13), hsl(36 100% 57% / 0) 62%); }
#ROOT .amb-b { position: absolute; left: 1000px; top: 300px; width: 1500px; height: 1500px; border-radius: 50%;
  background: radial-gradient(circle, hsl(212 70% 55% / .10), hsl(212 70% 55% / 0) 62%); }
#ROOT .vig { position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(ellipse 85% 80% at 50% 50%, rgba(0,0,0,0) 55%, rgba(0,0,0,.45) 100%); }
"""

# JS helpers every composition gets; `R` is the composition's own root selector.
HELPERS = """
          const R = "#%CID%";
          const $ = (s) => s.split(",").map((p) => R + " " + p.trim()).join(", ");
          const WORDS = (s, at, o) => { o = o || {}; tl.fromTo($(s + " .w"), { yPercent: 118, rotate: o.r === undefined ? 5 : o.r, opacity: 0 },
            { yPercent: 0, rotate: 0, opacity: 1, duration: o.d || 0.85, ease: o.e || "expo.out", stagger: o.s === undefined ? 0.055 : o.s }, at); };
          const UP = (s, at, o) => { o = o || {}; tl.fromTo($(s), { y: o.y === undefined ? 30 : o.y, opacity: 0, filter: "blur(12px)" },
            { y: 0, opacity: 1, filter: "blur(0px)", duration: o.d || 0.75, ease: o.e || "power3.out", stagger: o.s || 0 }, at); };
          const OUT = (s, at, o) => { o = o || {}; tl.to($(s), { y: o.y === undefined ? -26 : o.y, opacity: 0, filter: "blur(12px)",
            duration: o.d || 0.5, ease: "power2.in", stagger: o.s || 0 }, at); };
          const RULE = (s, at, d) => tl.fromTo($(s), { scaleX: 0 }, { scaleX: 1, duration: d || 0.9, ease: "expo.inOut" }, at);
          const COUNT = (s, to, at, d, fmt) => { const el = document.querySelector($(s)); const o = { v: 0 };
            tl.fromTo(o, { v: 0 }, { v: to, duration: d, ease: "power2.out", onUpdate: () => { el.textContent = fmt(o.v); } }, at); };
          // closed-form spring step response, normalised to the tween: a settle with a hair of overshoot
          const SPRING = (w, z) => { const wd = w * Math.sqrt(1 - z * z);
            const f = (p) => 1 - Math.exp(-z * w * p) * (Math.cos(wd * p) + (z * w / wd) * Math.sin(wd * p)); const e = f(1);
            return (p) => (p >= 1 ? 1 : f(p) / e); };
          const SPR = SPRING(10, 0.76), SOFT = SPRING(8.5, 0.97);
          // ambient drift at a constant speed on a shared clock (t0), so scenes that cross
          // over each other show the same background at the seam
          const AMB = (dur, t0, k) => { t0 = t0 || 0; k = k || 1; const t1 = t0 + dur;
            tl.fromTo($(".amb-a"), { x: 6 * k * t0, y: 3 * k * t0 }, { x: 6 * k * t1, y: 3 * k * t1, duration: dur, ease: "none" }, 0);
            tl.fromTo($(".amb-b"), { x: -5 * k * t0, y: -2.2 * k * t0 }, { x: -5 * k * t1, y: -2.2 * k * t1, duration: dur, ease: "none" }, 0);
            tl.fromTo($(".amb-grid"), { backgroundPosition: (-8 * k * t0) + "px 0px" }, { backgroundPosition: (-8 * k * t1) + "px 0px", duration: dur, ease: "none" }, 0); };
"""
AMB_HTML = '<div class="amb" data-layout-allow-overflow><div class="amb-grid"></div><div class="amb-a"></div><div class="amb-b"></div></div>'


def css(cid, extra="", bg="var(--bg)"):
    return FONTS + (BASE + extra).replace("#ROOT", f"#{cid}").replace("%TOKENS%", TOKENS).replace("%BG%", bg)


def subcomp(cid, dur, body, script, extra_css="", bg="var(--bg)"):
    """A templated sub-composition; every id inside is prefixed with the composition id."""
    return f"""<!doctype html>
<html lang="en">
  <head><meta charset="UTF-8" /></head>
  <body>
    <template>
      <style>{css(cid, extra_css, bg)}</style>
      <div id="{cid}" data-composition-id="{cid}" data-width="1920" data-height="1080" data-duration="{dur:.3f}">
{body}
      </div>
      <script>
        (function () {{
          const tl = gsap.timeline({{ paused: true }});
{HELPERS.replace("%CID%", cid)}
{script}
          window.__timelines["{cid}"] = tl;
        }})();
      </script>
    </template>
  </body>
</html>
"""


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def kt(text):
    """Words wrapped for a masked per-word reveal. *word* is accent, _word_ is green."""
    out = []
    for w in text.split(" "):
        cls = ""
        if w.startswith("*") and w.rstrip(".,?!").endswith("*"):
            cls, w = " acc", w.replace("*", "")
        elif w.startswith("_") and w.rstrip(".,?!").endswith("_"):
            cls, w = " grn", w.replace("_", "")
        out.append(f'<span class="wm"><span class="w{cls}">{esc(w)}</span></span>')
    return " ".join(out)


def video(vid, src, start, dur, media_start=0.0, rate=1.0, cls="", style=""):
    r = f' data-playback-rate="{rate:.4f}"' if abs(rate - 1) > 1e-6 else ""
    return (f'<video id="{vid}" class="{cls}" src="{src}" data-start="{start:.3f}" data-duration="{dur:.3f}" '
            f'data-media-start="{media_start:.3f}"{r} muted playsinline style="{style}"></video>')


def crop_div(src_html, x, y, w, win_w):
    """A 1920x1080 source shown through a window of width win_w, cropped to (x, y, w)."""
    s = win_w / w
    return (f'<div class="crop" style="transform: translate({-x * s:.2f}px, {-y * s:.2f}px) '
            f'scale({s:.5f})">{src_html}</div>')


def eyebrow(text, eid=""):
    i = f' id="{eid}"' if eid else ""
    return f'<div class="eyebrow"{i}><span class="rule"></span><span class="eb-t">{esc(text)}</span></div>'


def eyebrow_in(sel, at):
    """The rule draws, then the label slides out of it."""
    return (f'          RULE("{sel} .rule", {at:.2f}, 0.6);\n'
            f'          tl.fromTo($("{sel} .eb-t"), {{ opacity: 0, x: -14 }}, {{ opacity: 1, x: 0, duration: 0.5, ease: "power3.out" }}, {at + 0.3:.2f});\n')


# ------------------------------------------------------------------ act 1: the quiz
HOOK_D = 17.8
FW, FH = 844, 475            # the two frames
AX, BX, FY = 96, 96 + FW + 40, 318
CW, CH, CX, CY = 1280, 720, 320, 220     # the combined wipe frame
S_AB = CW / FW
HX, HY, HS = 144, 30, 1632 / FW          # where the app frame sits in the next act


def hook():
    cid = "hook"
    # A is the render, B is the real frame.
    body = f"""
        {AMB_HTML}
        <div id="hook-top" style="position:absolute; left:96px; top:84px; width:1728px;">
          {eyebrow("A quick test", "hook-eye")}
          <p id="hook-l1" class="h2" style="margin-top:20px">{kt("One of these is a frame from a drone video.")}</p>
          <p id="hook-l2" class="h2" style="color:var(--dim)">{kt("The other was *rendered* from a 3D model.")}</p>
        </div>
        <div id="hook-bridge" style="position:absolute; left:96px; top:250px; width:1728px">
          <p id="hook-b1" class="h1" style="font-size:96px">{kt("Before we show how it's made,")}</p>
          <p id="hook-b2" class="h1" style="font-size:96px">{kt("here's what you can *do* with it.")}</p>
        </div>
        <div id="hook-persp" style="position:absolute; inset:0; perspective:1600px">
          <div id="hook-a" class="win" style="z-index:2; left:{AX}px; top:{FY}px; width:{FW}px; height:{FH}px; transform-origin:0 0">
            <img id="hook-a-img" src="assets/quiz/render.png" style="position:absolute; inset:0; width:100%; height:100%; object-fit:cover" />
            <div id="hook-a-ring" style="position:absolute; inset:0; border-radius:14px; box-shadow: inset 0 0 0 5px hsl(36 100% 57%); opacity:0"></div>
          </div>
          <div id="hook-b" class="win" style="z-index:1; left:{BX}px; top:{FY}px; width:{FW}px; height:{FH}px; transform-origin:0 0">
            <img id="hook-b-img" src="assets/quiz/real.png" style="position:absolute; inset:0; width:100%; height:100%; object-fit:cover" />
            <div id="hook-b-ring" style="position:absolute; inset:0; border-radius:14px; box-shadow: inset 0 0 0 5px hsl(156 62% 52%); opacity:0"></div>
          </div>
        </div>
        <div id="hook-la" class="mono" style="position:absolute; left:{AX + 22}px; top:{FY + 20}px; width:76px; height:76px; border-radius:50%;
          display:grid; place-items:center; font-size:40px; font-weight:700; background:hsl(225 12% 5% / .82); border:2px solid hsl(0 0% 100% / .2)">A</div>
        <div id="hook-lb" class="mono" style="position:absolute; left:{BX + 22}px; top:{FY + 20}px; width:76px; height:76px; border-radius:50%;
          display:grid; place-items:center; font-size:40px; font-weight:700; background:hsl(225 12% 5% / .82); border:2px solid hsl(0 0% 100% / .2)">B</div>
        <div id="hook-ta" class="pill" style="position:absolute; left:{AX}px; top:{FY + FH + 26}px">
          <span class="dot" style="background:var(--accent)"></span>A · Rendered from the 3D model</div>
        <div id="hook-tb" class="pill" style="position:absolute; left:{BX}px; top:{FY + FH + 26}px">
          <span class="dot" style="background:var(--green)"></span>B · Real: frame 41 of the drone video</div>
        <p id="hook-q" class="h1" style="position:absolute; left:96px; top:880px">{kt("Which one is real?")}</p>
        <div id="hook-f" style="position:absolute; left:96px; top:888px; width:1728px">
          <p class="h1" style="font-size:80px">{kt("If you picked A, *you* *were* *wrong.*")}</p>
        </div>
        <div id="hook-ring" style="position:absolute; left:1690px; top:866px; width:134px; height:134px">
          <svg width="134" height="134" viewBox="0 0 134 134"><circle cx="67" cy="67" r="58" fill="none" stroke="hsl(222 9% 22%)" stroke-width="6"/>
          <circle id="hook-ring-arc" cx="67" cy="67" r="58" fill="none" stroke="hsl(36 100% 57%)" stroke-width="6" stroke-linecap="round"
            stroke-dasharray="364.4" stroke-dashoffset="0" transform="rotate(-90 67 67)"/></svg>
          <div id="hook-n3" class="mono" style="position:absolute; inset:0; display:grid; place-items:center; font-size:60px; font-weight:600">3</div>
          <div id="hook-n2" class="mono" style="position:absolute; inset:0; display:grid; place-items:center; font-size:60px; font-weight:600">2</div>
          <div id="hook-n1" class="mono" style="position:absolute; inset:0; display:grid; place-items:center; font-size:60px; font-weight:600">1</div>
        </div>
        <p id="hook-same" class="mono" style="position:absolute; left:{CX}px; top:{CY - 64}px; font-size:26px; color:var(--dim)">Same camera. Same frame. Slide between them.</p>
        <div id="hook-wl" style="position:absolute; left:{CX - 2}px; top:{CY - 14}px; width:4px; height:{CH + 28}px; background:var(--accent); border-radius:2px;
          box-shadow: 0 0 24px 2px hsl(36 100% 57% / .55)">
          <div style="position:absolute; left:-24px; top:{CH / 2 - 12:.0f}px; width:52px; height:52px; border-radius:50%; background:var(--accent);
            display:grid; place-items:center; color:hsl(225 12% 5%); font-size:26px; font-weight:800">‹›</div>
        </div>
        <div id="hook-wlab-a" class="pill" style="position:absolute; left:{CX + 24}px; top:{CY + CH - 78}px"><span class="dot" style="background:var(--accent)"></span>Rendered</div>
        <div id="hook-wlab-b" class="pill" style="position:absolute; left:{CX + CW - 170}px; top:{CY + CH - 78}px"><span class="dot" style="background:var(--green)"></span>Real</div>
        <div id="hook-flash" style="position:absolute; inset:0; background:#fff; opacity:0; pointer-events:none"></div>
        <div class="vig"></div>"""
    ax, bx = CX - AX, CX - BX
    script = f"""
          AMB({HOOK_D});
{eyebrow_in("#hook-eye", 0.25)}
          WORDS("#hook-l1", 0.55);
          WORDS("#hook-l2", 1.55);
          // the two frames swing in on a shared perspective
          tl.fromTo($("#hook-a"), {{ opacity: 0, rotationY: 24, x: -120, z: -200, filter: "blur(14px) brightness(1)" }},
            {{ opacity: 1, rotationY: 0, x: 0, z: 0, filter: "blur(0px) brightness(1)", duration: 1.2, ease: "expo.out" }}, 2.5);
          tl.fromTo($("#hook-b"), {{ opacity: 0, rotationY: -24, x: 120, z: -200, filter: "blur(14px) brightness(1)" }},
            {{ opacity: 1, rotationY: 0, x: 0, z: 0, filter: "blur(0px) brightness(1)", duration: 1.2, ease: "expo.out" }}, 2.65);
          tl.fromTo([$("#hook-a-img"), $("#hook-b-img")], {{ scale: 1.0 }}, {{ scale: 1.06, duration: 9.5, ease: "none" }}, 2.5);
          tl.fromTo([$("#hook-la"), $("#hook-lb")], {{ opacity: 0, scale: 0.4 }}, {{ opacity: 1, scale: 1, duration: 0.6, ease: "back.out(2.2)", stagger: 0.14 }}, 3.2);
          WORDS("#hook-q", 4.2, {{ s: 0.07 }});
          tl.fromTo($("#hook-ring"), {{ opacity: 0, scale: 0.6, rotate: -60 }}, {{ opacity: 1, scale: 1, rotate: 0, duration: 0.6, ease: "back.out(1.8)" }}, 4.6);
          tl.fromTo($("#hook-ring-arc"), {{ attr: {{ "stroke-dashoffset": 0 }} }}, {{ attr: {{ "stroke-dashoffset": 364.4 }}, duration: 3.0, ease: "none" }}, 5.0);
          tl.fromTo($("#hook-n3"), {{ opacity: 0, scale: 1.8 }}, {{ opacity: 1, scale: 1, duration: 0.35, ease: "expo.out" }}, 5.0);
          tl.to($("#hook-n3"), {{ opacity: 0, scale: 0.5, duration: 0.2 }}, 5.85);
          tl.fromTo($("#hook-n2"), {{ opacity: 0, scale: 1.8 }}, {{ opacity: 1, scale: 1, duration: 0.35, ease: "expo.out" }}, 6.0);
          tl.to($("#hook-n2"), {{ opacity: 0, scale: 0.5, duration: 0.2 }}, 6.85);
          tl.fromTo($("#hook-n1"), {{ opacity: 0, scale: 1.8 }}, {{ opacity: 1, scale: 1, duration: 0.35, ease: "expo.out" }}, 7.0);
          tl.to($("#hook-n1"), {{ opacity: 0, scale: 0.5, duration: 0.2 }}, 7.85);
          tl.fromTo([$("#hook-a"), $("#hook-b")], {{ scale: 1 }}, {{ scale: 0.985, duration: 0.5, ease: "sine.inOut", yoyo: true, repeat: 5 }}, 5.0);
          // the reveal: a flash, B is real, A was the render
          tl.to($("#hook-ring"), {{ opacity: 0, scale: 1.4, duration: 0.35, ease: "power2.in" }}, 7.95);
          OUT("#hook-q .w", 7.95, {{ s: 0.03, d: 0.35 }});
          tl.fromTo($("#hook-flash"), {{ opacity: 0 }}, {{ opacity: 0.32, duration: 0.08, ease: "none" }}, 8.1);
          tl.to($("#hook-flash"), {{ opacity: 0, duration: 0.7, ease: "power2.out" }}, 8.18);
          tl.fromTo($("#hook-b-ring"), {{ opacity: 0 }}, {{ opacity: 1, duration: 0.3 }}, 8.15);
          tl.fromTo($("#hook-a-ring"), {{ opacity: 0 }}, {{ opacity: 1, duration: 0.3 }}, 8.15);
          tl.to($("#hook-b"), {{ scale: 1.035, duration: 0.8, ease: "expo.out" }}, 8.15);
          tl.to($("#hook-a"), {{ scale: 0.965, duration: 0.8, ease: "expo.out" }}, 8.15);
          tl.to($("#hook-lb"), {{ backgroundColor: "hsl(156 62% 52%)", color: "hsl(225 12% 5%)", borderColor: "hsl(156 62% 52%)", duration: 0.3 }}, 8.15);
          tl.to($("#hook-la"), {{ backgroundColor: "hsl(36 100% 57%)", color: "hsl(225 12% 5%)", borderColor: "hsl(36 100% 57%)", duration: 0.3 }}, 8.15);
          tl.fromTo($("#hook-tb"), {{ opacity: 0, y: 24, filter: "blur(10px)" }}, {{ opacity: 1, y: 0, filter: "blur(0px)", duration: 0.6, ease: "expo.out" }}, 8.3);
          tl.fromTo($("#hook-ta"), {{ opacity: 0, y: 24, filter: "blur(10px)" }}, {{ opacity: 1, y: 0, filter: "blur(0px)", duration: 0.6, ease: "expo.out" }}, 8.45);
          WORDS("#hook-f", 9.0, {{ s: 0.07 }});
          // bridge: B steps back, A becomes the backdrop for the line into the app
          OUT("#hook-top, #hook-f, #hook-ta, #hook-tb, #hook-la, #hook-lb", 11.9, {{ s: 0.04 }});
          tl.to([$("#hook-a-ring"), $("#hook-b-ring")], {{ opacity: 0, duration: 0.4 }}, 12.0);
          tl.to($("#hook-b"), {{ opacity: 0, scale: 0.9, x: "+=90", filter: "blur(8px) brightness(0.55)", duration: 0.8, ease: "power3.inOut" }}, 12.1);
          tl.to($("#hook-a"), {{ x: 526, y: 282, scale: 0.8, filter: "blur(6px) brightness(0.55)", duration: 1.0, ease: "expo.inOut" }}, 12.1);
          WORDS("#hook-b1", 12.4);
          WORDS("#hook-b2", 13.2);
          OUT("#hook-bridge .w", 15.2, {{ s: 0.025, d: 0.4 }});
          // hand-off: the render glides into the app's frame, and the app fades up inside it
          tl.to($("#hook-a"), {{ x: {HX - AX}, y: {HY - FY:.1f}, scale: {HS:.5f}, filter: "blur(0px) brightness(1)", borderWidth: 0.8, borderRadius: {16 / HS:.2f}, boxShadow: "0 40px 100px -30px #000, 0 0 0 0.8px hsl(0 0% 100% / .09)",
            duration: 1.3, ease: SPR }}, 15.2);
          tl.to($("#hook-a-img"), {{ scale: 1, duration: 1.3, ease: SPR }}, 15.2);
"""
    extra = f"""
#hook #hook-ta, #hook #hook-tb, #hook #hook-wlab-a, #hook #hook-wlab-b, #hook #hook-same, #hook #hook-wl, #hook #hook-ring {{ opacity: 0; }}
#hook #hook-n2, #hook #hook-n1 {{ opacity: 0; }}
"""
    return subcomp(cid, HOOK_D, body, script, extra)


# ------------------------------------------------------------------ act 3: the features
DEMO = RAW / "demo.mp4"
MODE_WHAT = {
    "Explore": "Look around the model and see what it is made of.",
    "Measure": "Distances, heights, areas and volumes, by clicking on the model.",
    "Inspect": "Log defects, find every photo that saw a spot, read the ground.",
    "Operations": "Flooding, damage, vehicle access and what a post can see.",
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
# the user's speed-ups (2026-09-28): collision mesh, construction tour, the border line,
# building up the Plan scheme, and all of Place
RATE.update({5: 2.0, 6: 2.0, 23: 4.0, 24: 3.0, 25: 3.0, 26: 2.5, 27: 1.8, 54: 2.2, 55: 2.0, 56: 3.5, 57: 3.0, 60: 2.6, 61: 2.2, 62: 2.2, 63: 2.2, 64: 1.8})
SKIP = {0, 1, 50, 59}                 # Projects page (it lists the personal room scan); 59 = sunlight, cut
CLAMP = {2: (7.27, None), 51: (227.9, None)}
CARD_AT = {3: ("01", "Explore"), 8: ("02", "Measure"), 12: ("03", "Inspect"), 19: ("04", "Operations"),
           28: ("05", "Twin"), 33: ("06", "Mission"), 40: ("07", "Walk"), 46: ("08", "Export"),
           52: ("09", "Plan"), 60: ("10", "Place")}
# What each step shows, in words for someone who has never seen the app.
CAPTION = {
    3: "Quality, project and file panels", 4: "The drone's flight path, as solved",
    5: "Collision mesh: the solid surface under the picture", 6: "Back to the photo-real model",
    7: "Click a photo: fly to where it was taken",
    9: "Distance, height, area, volume, notes", 10: "Height: click the foot, then the top", 11: "Name it and save it",
    13: "Defects, pole tilt, cable sag, what moved", 14: "Click a spot on the boulder", 15: "Every photo that saw that spot",
    16: "See it from the original photo", 17: "Saved to the defect list", 18: "Heritage: local relief draped on the model",
    20: "Disaster response tools", 21: "Flood: click where the water comes in", 22: "The valley fills, following the terrain",
    23: "Construction tools", 24: "Border: line-of-sight coverage", 25: "Draw the fence line",
    26: "Place an observation post", 27: "What the post can see",
    29: "As measured: the geometry", 30: "How sure: confidence, area by area", 31: "As it looks: photo-real",
    32: "Versions and a game-engine export",
    33: "Name a mission", 34: "Mark an enemy post on the boulders", 35: "Turn it to face the meadow",
    36: "Mark the objective", 37: "Draw the route in", 38: "Exposure: what the enemy sees of the route",
    39: "Terrain: where vehicles can move",
    41: "Walk inside the scan", 42: "Look over the meadow", 43: "The ground is solid: you walk on it",
    44: "Look back at the boulders",
    47: "Pick the files you need", 49: "One zip, one click",
    52: "Start a scheme", 53: "Rules, impact, existing", 54: "Draw a building footprint", 55: "Add floors",
    56: "A second building", 57: "A road past them", 58: "Swipe: existing against proposed",
    59: "Sunlight: shadows through the day",
    61: "Real-size furniture on the patio", 62: "A sofa behind it", 63: "An armchair to the side",
    64: "Look around the layout",
}

# ---- feature minis
MINIS_SRC = RAW / "minis-clean.mp4"
MINIS_LOG = RAW / "minis-clean.json"
VIEW = (0, 70, 1510)
WIDE = (192, 0, 1728)
CENTER = (96, 54, 1728)            # every mini: the whole app, centred, 1.11x
MINI = {  # name: (what, crop, scene)
    "Object types": ("Ground, rock, trees and obstacles, labelled.", CENTER, ""),
    "Distance": ("Straight line or a path, clicked on the model.", CENTER, ""),
    "Area": ("Trace around a shape: its area on the ground.", CENTER, ""),
    "Volume": ("A pile above the ground, in cubic metres.", CENTER, ""),
    "Note": ("Mark something you saw, right where it is.", CENTER, ""),
    "Cut a section": ("A true-scale profile through the site.", CENTER, ""),
    "Archive record": ("How the model was made, with a checksum for every file.", CENTER, ""),
    "Road access": ("A route a vehicle can still drive.", CENTER, ""),
    "Map tiles": ("Height-map and point-cloud tiles for GIS.", CENTER, ""),
    "Relief camp": ("Tents, a medical tent and a helipad, at real size.", CENTER, "Open ground"),
    "Site logistics": ("A tower crane and its jib circle.", CENTER, "Open ground"),
    "Enemy guess": ("Where an enemy would watch the route from.", CENTER, ""),
    "Brief": ("A sand-table view for the briefing.", CENTER, ""),
    "Rehearse": ("Walk the mission at night, goggles on.", CENTER, ""),
    "Landing zones": ("Flat, clear circles big enough for a helicopter.", CENTER, "Open ground"),
    "Rules": ("A plot's height limit, and the building that breaks it.", CENTER, "Open ground"),
    "Impact": ("What the scheme changes against the scan.", CENTER, "Open ground"),
    "Split": ("Existing and proposed, side by side.", CENTER, "Open ground"),
    "Share": ("GeoJSON, CityJSON, DXF or 3D Tiles.", CENTER, "Open ground"),
}
CARD = {  # name: (what, icon)
    "Camera coverage": ("How well each area was seen by the drone.", "camera"),
    "Pole tilt": ("Click the foot of a pole or mast: how far it leans.", "tilt"),
    "Cable sag": ("Click both ends of a cable: sag and ground clearance.", "cable"),
    "What moved": ("Against an older scan: the surfaces that moved.", "compare"),
    "Seasonal change": ("Against an earlier season: erosion and movement.", "clock"),
    "Restoration idea": ("Sketch a missing wall or tower, marked as a guess.", "pencil"),
    "What changed": ("Against the flight before the event.", "compare"),
    "Building damage": ("Every building graded: intact, damaged, collapsed.", "building"),
    "Debris volume": ("Outline a debris pile or landslide: its volume.", "pile"),
    "People & vehicles": ("What the video saw, put onto the map.", "people"),
    "First map": ("A quick map while the drone is still flying.", "map"),
    "Stockpile volume": ("A stockpile's volume against the last flight.", "pile"),
    "Cut & fill": ("Earth still to dig or fill to reach the design.", "layers"),
    "Built vs design": ("What was built, against the BIM or CAD model.", "building"),
}
ICON = {
    "camera": '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
    "tilt": '<path d="M6 21 L14 4"/><path d="M4 21h16"/><path d="M6 13a8 8 0 0 1 4-1" />',
    "cable": '<path d="M4 4v17M20 4v17"/><path d="M4 6 Q12 18 20 6"/>',
    "compare": '<rect x="3" y="5" width="11" height="11" rx="1.5"/><rect x="10" y="9" width="11" height="11" rx="1.5"/>',
    "clock": '<circle cx="12" cy="12" r="8.5"/><path d="M12 7v5l3.5 2"/>',
    "pencil": '<path d="M4 20l1-4L16 5l3 3L8 19z"/><path d="M14 7l3 3"/>',
    "building": '<path d="M5 21V5l7-2v18M12 8l7 2v11"/><path d="M3 21h18M8 8v1M8 12v1M8 16v1M15 13v1M15 17v1"/>',
    "pile": '<path d="M2 20 Q7 8 12 11 Q16 6 22 20z"/>',
    "people": '<circle cx="8" cy="8" r="3"/><circle cx="17" cy="9" r="2.5"/><path d="M3 20c0-4 2.5-6 5-6s5 2 5 6M14 20c0-3 1.5-5 3-5s4 2 4 5"/>',
    "map": '<path d="M3 6l6-2 6 2 6-2v14l-6 2-6-2-6 2z"/><path d="M9 4v14M15 6v14"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
}
BREAKS = [  # (after event, id, chapter number, name, minis, cards, note)
    (7, "brk-explore", "01", "Explore", ["Object types"], ["Camera coverage"],
     "Coverage draws nothing readable on this scan."),
    (11, "brk-measure", "02", "Measure", ["Distance", "Area", "Volume", "Note"], [], ""),
    (18, "brk-inspect", "03", "Inspect", ["Cut a section", "Archive record"],
     ["Pole tilt", "Cable sag", "What moved", "Seasonal change", "Restoration idea"],
     "These need poles, cables or an earlier flight of the same site."),
    (27, "brk-ops", "04", "Operations", ["Road access", "Map tiles", "Relief camp", "Site logistics"],
     ["What changed", "Building damage", "Debris volume", "People & vehicles", "First map",
      "Stockpile volume", "Cut & fill", "Built vs design"],
     "These need buildings, a second flight or a design model."),
    (39, "brk-mission", "06", "Mission", ["Enemy guess", "Brief", "Rehearse", "Landing zones"], [], ""),
    (58, "brk-plan", "09", "Plan", ["Rules", "Impact", "Split", "Share"], [], ""),
]

# break layout
BWX, BWY, BWW = 704, 196, 1120            # the big window
BBAR = 44
BVH = BWW * 9 / 16
BWH = BBAR + BVH
TW = 176                                   # tray tiles
TS = TW / BWW
TH = BWH * TS
TY = 912
EXP = 0.8                                  # expand / collapse time
HOLD = 0.35                                # a mini rests on its result this long
SWAP = 0.3                                 # the next mini expands this long after the last one starts to collapse
SLOT_PAD = EXP * 0.5 + HOLD + SWAP          # per mini, on top of its own length


MINI_SRC_START = {}
MINI_CLICKS = RAW / "minis-clicks.json"
MINI_TRACK = RAW / "minis-clicks-track.json"


def mini_marks(name, t0, dm, rate):
    """Click rings for one mini window, in its cropped source's pixels and the break's clock."""
    if not (MINI_CLICKS.is_file() and MINI_TRACK.is_file()):
        return []
    a = MINI_SRC_START[name]
    b = a + dm * rate
    track = json.loads(MINI_TRACK.read_text())
    got = [c for c in json.loads(MINI_CLICKS.read_text()) if a <= c["t"] < b] + rest_clicks(track, a, b)
    got.sort(key=lambda c: c["t"])
    uniq = []
    for c in got:
        if uniq and c["t"] - uniq[-1]["t"] < 0.3 and abs(c["x"] - uniq[-1]["x"]) + abs(c["y"] - uniq[-1]["y"]) < 40:
            continue
        uniq.append(c)
    cx, cy, cw = CENTER
    return [(t0 + (c["t"] - a) / rate, c["x"] - cx, c["y"] - cy) for c in uniq
            if cx <= c["x"] < cx + cw and cy <= c["y"] < cy + cw * 9 / 16]


def cut_minis():
    """Each mini as its own cropped, re-timed clip ending on a held frame, plus its poster."""
    log = json.loads(MINIS_LOG.read_text())
    ch = {c["title"][6:]: c for c in log["chapters"] if c["title"].startswith("mini:")}
    out = {}
    for name, (what, (cx, cy, cw), scene) in MINI.items():
        c = ch[name]
        a, b = c["start"], c["end"]
        L = b - a
        rate = min(3.2, max(1.0, L / 4.0))
        n_play = nf(L / rate)
        n_tail = nf(EXP + 1.6)
        chh = round(cw * 9 / 16)
        pre = f"crop={cw}:{chh}:{cx}:{cy},scale=1280:720:flags=lanczos,"
        slug = re.sub(r"[^a-z]+", "-", name.lower()).strip("-")
        p = seg(MINIS_SRC, a, rate, n_play, n_tail, f"mini-{slug}", pre)
        poster = frame_jpg(p, (n_play - 1) / FPS, CUT / "posters" / f"{p.stem}.jpg", w=360)
        out[name] = (p, n_play / FPS, poster, rate)
        MINI_SRC_START[name] = a
    return out


def break_len(minis, mcut, cards):
    t = 1.5
    for m in minis:
        t += mcut[m][1] + SLOT_PAD
    t += EXP - SWAP + 0.3
    if cards:
        t += 1.0 + 0.09 * len(cards) + 4.4 + 0.2 * len(cards)
    return t + 0.9


def capabilities(mcut):
    """The demo take, re-timed into one file, plus where each feature break sits (local time)."""
    ev = json.loads((RAW / "demo.json").read_text())["events"]
    brk_after = {b[0]: b for b in BREAKS}
    n = 0                                       # film frame counter
    segs = []                                   # (film start, source start, rate, film length)
    parts, cards, caps, breaks = [], [], [], []
    for i, e in enumerate(ev):
        if i in SKIP:
            continue
        a, b = e["start"], e["end"]
        lo, _ = CLAMP.get(i, (None, None))
        a = lo if lo else a
        rate = RATE[i]
        n_play = nf((b - a) / rate)
        n_hold = 0
        if i in CARD_AT:
            cards.append((n / FPS, *CARD_AT[i]))
        if i == 51:
            cards.append((n / FPS, "", "Open ground"))
        if i in CAPTION:
            caps.append((n / FPS, n_play / FPS, CAPTION[i]))
        if i in brk_after:
            _, bid, num, name, mlist, clist, note = brk_after[i]
            bd = break_len(mlist, mcut, clist)
            n_hold = nf(bd)
            breaks.append((bid, (n + n_play) / FPS, n_hold / FPS, num, name, mlist, clist, note))
        parts.append(seg(DEMO, a, rate, n_play, n_hold, f"cap{i:02d}"))
        segs.append((n / FPS, a, rate, n_play / FPS))
        n += n_play + n_hold
    cap_d = n / FPS
    cap_file = concat(parts, "cap")
    return cap_file, cap_d, cards, caps, breaks, segs


# the app footage sits in a frame with a caption band under it
CFX, CFY, CFS = 144, 30, 0.85
FWD, FHD = 1920 * CFS, 1080 * CFS
SX, SWD = 700, 1124                          # the frame's position during a chapter slate
SS = SWD / FWD
SY = (1080 - FHD * SS) / 2
BAND = CFY + FHD + (1080 - CFY - FHD) / 2      # caption band centre line


def chip_dots(k):
    return "".join(f'<span style="display:inline-block; width:{24 if j == k else 9}px; height:6px; border-radius:3px; margin-left:6px; '
                   f'background:{"var(--accent)" if j == k else ("hsl(220 20% 80% / .7)" if j < k else "hsl(220 20% 80% / .2)")}"></span>'
                   for j in range(10))


CLICKS = RAW / "demo-clicks.json"           # every press, from build/find_clicks.py
TRACK = RAW / "demo-clicks-track.json"   # the cursor's position, from build/find_clicks.py
# The steps where the cursor clicks on the model itself, and the detail is small on screen.
ZOOM_EVENTS = [10, 14, 21, 34, 36]


# The moments worth a close look: a click on the model that makes a result. Menu picks,
# tabs and drags stay wide. Source-clock times and the click spots come from the
# cursor track (build/find_clicks.py); each shot frames every spot it lists.
SHOTS = [
    ("height: foot, then top", 51.8, 55.4, [(938, 717), (886, 403)]),
    ("a spot on the boulder", 69.3, 71.5, [(922, 557)]),
    ("flood: where the water enters", 104.5, 105.9, [(1017, 592)]),
    ("enemy post on the boulders", 156.2, 158.3, [(859, 426)]),
    ("the objective", 164.9, 167.0, [(1176, 503)]),
    ("building footprint", 243.5, 247.6, [(989, 488), (1165, 516), (1154, 587), (954, 552)]),
]


def click_zooms(segs, busy, events, cap_d):
    """Steady close-ups on the moments that matter: ease in on the click spot, hold still
    while the result appears, ease back out. One smooth move each way, nothing follows
    the cursor, so the camera never jitters."""
    def film_of(src):
        for f0, a, r, ln in segs:
            if a <= src < a + ln * r:
                return f0 + (src - a) / r
        return None

    out, n = [], 0
    for name, s_in, s_out, pts in SHOTS:
        a, b = film_of(s_in), film_of(s_out)
        if a is None or b is None:
            continue
        for lo, hi in busy:                           # a slate or break is still moving the frame
            if lo - 0.4 < a < hi + 0.05:
                a = hi + 0.05
        if b - a < 1.4:
            b = a + 1.4
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        z = max(1.3, min(1.55, 1920 / (bw * 1.8 + 560), 1080 / (bh * 1.8 + 320)))
        # the spot moves toward the centre, and the view never shows past the recording
        sx, sy = cx + (960 - cx) * 0.6, cy + (540 - cy) * 0.6
        tx = min(0.0, max(1920 * (1 - z), sx - z * cx))
        ty = min(0.0, max(1080 * (1 - z), sy - z * cy))
        din = min(0.9, max(0.6, (b - a) * 0.35))
        out += [f'          tl.to($("#cap-zoom"), {{ x: {tx:.1f}, y: {ty:.1f}, scale: {z:.3f}, duration: {din:.2f}, ease: "power2.inOut" }}, {a:.3f});  // {name}',
                f'          tl.to($("#cap-zoom"), {{ x: 0, y: 0, scale: 1, duration: 0.9, ease: "power2.inOut" }}, {b:.3f});']
        n += 1
    print(f"  close-ups: {n}")
    return [f'          tl.set($("#cap-zoom"), {{ x: 0, y: 0, scale: 1, transformOrigin: "0px 0px" }}, 0);'] + out


# Steps whose clicks land on the model: the press detector loses the cursor over busy 3D
# ground, so for these the click is taken from where the cursor comes to rest (the rig's
# click() dwells 0.14 s at the target before it presses).
MODEL_CLICK_EVENTS = [10, 14, 21, 25, 26, 34, 36, 37, 54, 57, 61, 62, 63]
RIPPLE_CSS = """
#ROOT .rp { position: absolute; width: 0; height: 0; pointer-events: none; }
#ROOT .rp i { position: absolute; left: -24px; top: -24px; width: 48px; height: 48px; border-radius: 50%;
  border: 2.5px solid rgba(255,255,255,.95); box-shadow: 0 0 0 1px rgba(0,0,0,.28), 0 0 14px rgba(255,255,255,.35); opacity: 0; }
#ROOT .rp b { position: absolute; left: -11px; top: -11px; width: 22px; height: 22px; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,.75), rgba(255,255,255,0) 70%); opacity: 0; }
"""


def rest_clicks(track, a, b):
    """Press moments inside [a, b]: where the cursor comes to rest over the 3D view."""
    pts = [p for p in track if a <= p[0] <= b and 40 < p[1] < 1500 and 80 < p[2] < 1000]
    out, run = [], []
    for p in pts + [None]:
        if p is not None and run and abs(p[1] - run[-1][1]) + abs(p[2] - run[-1][2]) <= 3 and p[0] - run[-1][0] < 0.12:
            run.append(p)
            continue
        if len(run) >= 3:
            out.append({"t": run[0][0] + 0.14, "x": run[0][1], "y": run[0][2]})
        run = [p] if p is not None else []
    return out


def all_clicks(clicks, track, events):
    got = list(clicks)
    for i in MODEL_CLICK_EVENTS:
        got += rest_clicks(track, events[i]["start"], events[i]["end"])
    got.sort(key=lambda c: c["t"])
    uniq = []
    for c in got:
        if uniq and c["t"] - uniq[-1]["t"] < 0.3 and abs(c["x"] - uniq[-1]["x"]) + abs(c["y"] - uniq[-1]["y"]) < 40:
            continue
        uniq.append(c)
    return uniq


def ripple_marks(prefix, marks):
    """marks: [(film second, x, y)] in the coordinates of the layer the ripples sit in."""
    html, js = [], []
    for k, (t, x, y) in enumerate(marks):
        rid = f"{prefix}{k}"
        html.append(f'<div id="{rid}" class="rp" style="left:{x:.0f}px; top:{y:.0f}px"><b></b><i></i></div>')
        js += [f'          tl.fromTo($("#{rid} i"), {{ scale: 0.3, opacity: 0.95 }}, {{ scale: 1.3, opacity: 0, duration: 0.55, ease: "power2.out", immediateRender: false }}, {t:.3f});',
               f'          tl.fromTo($("#{rid} b"), {{ scale: 0.5, opacity: 1 }}, {{ scale: 1.4, opacity: 0, duration: 0.3, ease: "power1.out", immediateRender: false }}, {t:.3f});']
    return html, js


def cap_ripples(segs, events):
    if not (CLICKS.is_file() and TRACK.is_file()):
        return [], []

    def film_of(src):
        for f0, a, r, ln in segs:
            if a <= src < a + ln * r:
                return f0 + (src - a) / r
        return None

    got = all_clicks(json.loads(CLICKS.read_text()), json.loads(TRACK.read_text()), events)
    marks = [(film_of(c["t"]), c["x"], c["y"]) for c in got]
    marks = [m for m in marks if m[0] is not None]
    print(f"  click marks: {len(marks)}")
    return ripple_marks("cap-rp", marks)


def capabilities_comp(cap_file, cap_d, cards, caps, breaks, segs):
    rp_html, rp_js = cap_ripples(segs, json.loads((RAW / "demo.json").read_text())["events"])
    body = [f"        {AMB_HTML}",
            f'        <div id="cap-cam" data-layout-allow-overflow style="position:absolute; left:{CFX}px; top:{CFY}px; width:{FWD:.0f}px; height:{FHD:.0f}px; transform-origin:0 0">',
            f'          <div id="cap-frame" style="position:absolute; inset:0; border-radius:16px; overflow:hidden; transform-origin:50% 50%;'
            f' box-shadow:0 40px 100px -30px #000, 0 0 0 1.5px hsl(0 0% 100% / .09)">',
            f'            <div style="position:absolute; left:0; top:0; width:1920px; height:1080px; transform-origin:0 0; transform:scale({CFS})">',
            '              <div id="cap-zoom" style="position:absolute; left:0; top:0; width:1920px; height:1080px; transform-origin:0 0">',
            "                " + video("cap-v", rel(cap_file), 0, cap_d, 0, 1.0, "clip",
                                       "position:absolute; left:0; top:0; width:1920px; height:1080px"),
            '                <div id="cap-clicks" style="position:absolute; inset:0">' + "".join(rp_html) + "</div>",
            "              </div>",
            "            </div>",
            "          </div>",
            "        </div>",
            '        <div class="vig"></div>']
    script = [f"          AMB({cap_d:.2f}, 0, 0.12);",
              '          tl.fromTo(R, { opacity: 0 }, { opacity: 1, duration: 0.5, ease: "power2.out" }, 0);',
              '          tl.set($("#cap-frame"), { filter: "blur(0px) brightness(1) saturate(1)" }, 0);']

    # chapter slates: the frame glides right and shrinks, the chapter title takes the left
    slates = [(1.3, "", "The scan", "A 12-second drone flight", "72 photos, one 3D model. Everything here runs on it.")]
    last = 1.3
    for s0, num, name in cards:
        s0 = max(s0, last + 5.2)
        last = s0
        if name == "Open ground":
            slates.append((s0, "", "A second site", "Open ground", "Generated, not scanned: flat ground for planning."))
        else:
            slates.append((s0, num, f"Chapter {num} / 10", name, MODE_WHAT[name]))
    SL = 4.4
    for k, (s, num, eye, title, what) in enumerate(slates):
        # a slate that runs straight into the next one keeps the frame where it is
        chained = k + 1 < len(slates) and slates[k + 1][0] - (s + SL) < 1.6
        se = slates[k + 1][0] + 0.4 if chained else s + SL      # when this slate's text is gone
        so = se - 0.55 if chained else s + SL - 1.05            # when it starts to leave
        sid = f"cap-sl{k}"
        big = (f'<div id="{sid}-num" class="mono" data-layout-allow-overflow style="position:absolute; left:-10px; top:-250px; font-size:230px; font-weight:700; line-height:1; '
               f'color:hsl(36 70% 40% / .22); -webkit-text-stroke:2px hsl(36 100% 57% / .85); letter-spacing:-0.05em">{num}</div>') if num else ""
        fs = 100 if len(title) <= 8 else (80 if len(title) <= 12 else 64)
        body.append(f"""        <div id="{sid}" class="clip" data-start="{s:.3f}" data-duration="{se - s:.2f}" style="position:absolute; left:96px; top:470px; width:560px; height:420px">
          {big}
          <div id="{sid}-eye">{eyebrow(eye)}</div>
          <p id="{sid}-t" class="h1" style="font-size:{fs}px; margin-top:16px">{kt(title)}</p>
          <p id="{sid}-d" class="body" style="width:540px; margin-top:12px; font-size:30px">{esc(what)}</p>
        </div>""")
        script += [
            f'          tl.to($("#cap-cam"), {{ x: {SX - CFX}, y: {SY - CFY:.1f}, scale: {SS:.5f}, duration: 1.05, ease: SPR }}, {s:.3f});',
            f'          tl.to($("#cap-chips"), {{ opacity: 0, duration: 0.35 }}, {s:.3f});',
        ]
        if num:
            script.append(f'          tl.fromTo($("#{sid}-num"), {{ opacity: 0, x: -90, filter: "blur(18px)" }}, {{ opacity: 1, x: 0, filter: "blur(0px)", duration: 1.1, ease: "expo.out" }}, {s + 0.2:.3f});')
            script.append(f'          tl.to($("#{sid}-num"), {{ x: 30, duration: {SL - 1.3:.2f}, ease: "none" }}, {s + 1.3:.3f});')
        script.append(eyebrow_in(f"#{sid}-eye", s + 0.3).rstrip("\n"))
        script.append(f'          WORDS("#{sid}-t", {s + 0.4:.3f}, {{ s: 0.08, d: 0.9 }});')
        script.append(f'          UP("#{sid}-d", {s + 0.8:.3f});')
        script.append(f'          OUT("#{sid}-eye, #{sid}-t, #{sid}-d{", #" + sid + "-num" if num else ""}", {so:.3f}, {{ s: 0.05, d: 0.45 }});')
        if not chained:
            script.append(f'          tl.to($("#cap-cam"), {{ x: 0, y: 0, scale: 1, duration: 1.05, ease: SPR }}, {s + SL - 0.95:.3f});')
            script.append(f'          tl.to($("#cap-chips"), {{ opacity: 1, duration: 0.4 }}, {s + SL - 0.2:.3f});')

    # the chapter chip, right of the caption band
    chaps = [(s, num, title) for s, num, eye, title, what in slates if num]
    body.append('        <div id="cap-chips" style="position:absolute; inset:0">')
    for k, (s, num, title) in enumerate(chaps):
        s1 = s + SL - 0.5
        e1 = chaps[k + 1][0] + 0.4 if k + 1 < len(chaps) else cap_d
        cid_ = f"cap-chip{k}"
        body.append(f"""        <div id="{cid_}" class="clip" data-start="{s1:.3f}" data-duration="{e1 - s1:.3f}" style="position:absolute; left:{1920 - CFX - 700}px; top:{BAND - 30:.0f}px; width:700px; height:60px">
          <div id="{cid_}-in" style="position:absolute; right:0; top:0; height:60px; display:flex; align-items:center; gap:14px">
            <span class="mono" style="font-size:22px; color:var(--accent); font-weight:600">{num}</span>
            <span style="font-size:26px; font-weight:650; letter-spacing:-0.01em; color:var(--dim)">{esc(title)}</span>
            <span style="margin-left:8px; display:flex; align-items:center">{chip_dots(int(num) - 1)}</span>
          </div>
        </div>""")
        script.append(f'          tl.fromTo($("#{cid_}-in"), {{ opacity: 0, x: 40, filter: "blur(8px)" }}, {{ opacity: 1, x: 0, filter: "blur(0px)", duration: 0.7, ease: "expo.out" }}, {s1 + 0.1:.3f});')
    body.append("        </div>")

    # step captions, in the band under the frame
    busy = [(s, s + SL) for s, *_ in slates] + [(bs, bs + bd) for _, bs, bd, *_ in breaks]

    # zoom in on the cursor where it clicks on the model, so the detail it acts on reads
    for k, (s, d, text) in enumerate(caps):
        a, b = s + 0.1, s + d - 0.05
        for lo, hi in busy:
            if lo - 0.2 < a < hi:
                a = hi + 0.05
            if lo < b < hi + 0.1:
                b = lo - 0.05
        if b - a < 1.2:
            continue
        cid_ = f"cap-cc{k}"
        body.append(f"""        <div id="{cid_}" class="clip" data-start="{a:.3f}" data-duration="{b - a:.3f}" style="position:absolute; left:{CFX}px; top:{BAND - 40:.0f}px; width:1100px; height:80px">
          <div id="{cid_}-in" style="position:absolute; left:0; top:0; height:80px; display:flex; align-items:center; gap:20px">
            <span class="cc-bar" style="display:block; width:5px; height:44px; border-radius:3px; background:var(--accent); transform-origin:50% 50%"></span>
            <span class="cc-t" style="font-size:42px; font-weight:620; letter-spacing:-0.02em; white-space:nowrap">{kt(text)}</span>
          </div>
        </div>""")
        script.append(f'          tl.fromTo($("#{cid_} .cc-bar"), {{ scaleY: 0 }}, {{ scaleY: 1, duration: 0.5, ease: SPR }}, {a:.3f});')
        script.append(f'          WORDS("#{cid_} .cc-t", {a + 0.06:.3f}, {{ s: 0.035, d: 0.7, r: 0 }});')
        script.append(f'          tl.to($("#{cid_}-in"), {{ opacity: 0, y: -14, filter: "blur(10px)", duration: 0.28, ease: "power2.in" }}, {b - 0.28:.3f});')

    for bid, s, d, *_ in breaks:
        script.append(f'          tl.to($("#cap-chips"), {{ opacity: 0, duration: 0.4 }}, {s:.3f});')
        script.append(f'          tl.to($("#cap-chips"), {{ opacity: 1, duration: 0.5 }}, {s + d - 0.5:.3f});')
        script.append(f'          tl.to($("#cap-frame"), {{ scale: 1.05, filter: "blur(20px) brightness(0.55) saturate(0.8)", duration: 0.9, ease: "power3.inOut" }}, {s:.3f});')
        script.append(f'          tl.to($("#cap-frame"), {{ scale: 1, filter: "blur(0px) brightness(1) saturate(1)", duration: 0.9, ease: "power3.inOut" }}, {s + d - 0.9:.3f});')
    # hand-off to "How it's made": the app recedes into the dark
    script.append(f'          tl.to($("#cap-cam"), {{ x: {(1920 - FWD * 0.7) / 2 - CFX:.1f}, y: {(1080 - FHD * 0.7) / 2 - CFY:.1f}, scale: 0.7, duration: 1.3, ease: "power3.in" }}, {cap_d - 1.3:.3f});')
    script.append(f'          tl.to($("#cap-frame"), {{ filter: "blur(3px) brightness(0.75) saturate(1)", duration: 1.3, ease: "power2.in" }}, {cap_d - 1.3:.3f});')
    script += rp_js
    return subcomp("capabilities", cap_d, "\n".join(body), "\n".join(script), RIPPLE_CSS)


def cards_layout(n):
    cols = n if n <= 5 else 4
    rows = math.ceil(n / cols)
    w = min(560, (1728 - (cols - 1) * 24) / cols)
    sw = w - 32
    sh = sw * 170 / 320
    h = 16 + sh + 14 + 88 + 16
    gw = cols * w + (cols - 1) * 24
    gh = rows * h + (rows - 1) * 24
    x0 = 96 + (1728 - gw) / 2
    y0 = 318 + (700 - gh) / 2
    return cols, rows, w, h, sw, sh, x0, y0


def feature_break(bid, d, num, name, minis, cards, note, mcut):
    n = len(minis)
    body = [f"""        <div id="{bid}-glass" style="position:absolute; inset:0; background:hsl(225 12% 5% / .55)"></div>
        <div class="vig"></div>
        <div id="{bid}-head" style="position:absolute; left:96px; top:70px; width:1728px">
          {eyebrow(f"{num} · {name}")}
          <p class="h1" style="margin-top:12px; font-size:68px">{kt(f"More in {name}")}</p>
        </div>
        <div id="{bid}-count" class="mono" style="position:absolute; left:96px; top:236px; font-size:24px; color:var(--dim)">{n} more · each on a clean scene</div>"""]
    script = [f'          tl.fromTo($("#{bid}-glass"), {{ opacity: 0 }}, {{ opacity: 1, duration: 0.8, ease: "power2.out" }}, 0);',
              f'          tl.fromTo($(".vig"), {{ opacity: 0 }}, {{ opacity: 1, duration: 0.8 }}, 0);',
              eyebrow_in(f"#{bid}-head", 0.3).rstrip("\n"),
              f'          WORDS("#{bid}-head .h1", 0.45, {{ s: 0.07 }});',
              f'          UP("#{bid}-count", 0.9);']
    tx0 = 96
    for k, m in enumerate(minis):
        p, dm, poster, rate = mcut[m]
        tx = tx0 + k * (TW + 22)
        body.append(f"""        <div id="{bid}-slot{k}" style="position:absolute; left:{tx}px; top:{TY}px; width:{TW}px; height:{TH:.0f}px; border-radius:10px; border:2px dashed hsl(0 0% 100% / .22)"></div>
        <div id="{bid}-tile{k}" style="position:absolute; left:{tx}px; top:{TY}px; width:{TW}px; height:{TH:.0f}px; border-radius:10px; overflow:hidden; border:2px solid var(--line); background:var(--panel)">
          <img src="{rel(poster)}" style="position:absolute; left:0; top:{BBAR * TS:.1f}px; width:100%; height:{BVH * TS:.1f}px; object-fit:cover" />
          <div style="position:absolute; left:0; right:0; bottom:0; padding:18px 8px 5px; background:linear-gradient(0deg, hsl(225 12% 5% / .92), hsl(225 12% 5% / 0));
            font-size:15px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis">{esc(m)}</div>
          <div id="{bid}-tick{k}" style="position:absolute; right:6px; top:6px; width:24px; height:24px; border-radius:50%; background:var(--accent);
            display:grid; place-items:center; color:hsl(225 12% 5%); font-size:15px; font-weight:800; opacity:0">✓</div>
        </div>""")
    tiles = [f"#{bid}-tile{k}" for k in range(n)]
    slots = [f"#{bid}-slot{k}" for k in range(n)]
    script.append(f'          tl.fromTo({json.dumps(tiles)}.map((s) => R + " " + s), {{ opacity: 0, y: 50, scale: 0.85 }}, {{ opacity: 1, y: 0, scale: 1, duration: 0.8, ease: SPR, stagger: 0.08 }}, 0.8);')
    script.append(f'          tl.fromTo({json.dumps(slots)}.map((s) => R + " " + s), {{ opacity: 0 }}, {{ opacity: 1, duration: 0.3 }}, 1.0);')

    t = 1.5
    for k, m in enumerate(minis):
        p, dm, poster, rate = mcut[m]
        what, _, scene = MINI[m]
        tx = tx0 + k * (TW + 22)
        wid, tid = f"{bid}-w{k}", f"{bid}-t{k}"
        rp_html, rp_js = ripple_marks(f"{wid}-rp", mini_marks(m, t, dm, rate))
        script += rp_js
        vid = video(f"{wid}-v", rel(p), t, dm + EXP * 0.5 + HOLD + EXP + 0.1, 0.0, 1.0, "",
                    f"position:absolute; left:0; top:0; width:{BWW - 4}px; height:{BVH:.0f}px; object-fit:cover")
        tag = f'<span class="pill" style="font-size:20px; padding:6px 14px; margin-top:26px"><span class="dot" style="background:var(--muted)"></span>{esc(scene)}</span>' if scene else ""
        speed = f' · shown at {rate:.1f}×' if rate > 1.05 else ""
        body.append(f"""        <div id="{wid}" class="win" style="left:{BWX}px; top:{BWY}px; width:{BWW}px; height:{BWH:.0f}px; transform-origin:0 0; opacity:0">
          <div class="bar"><i></i><i></i><i></i><b>{esc(m)}</b><span style="margin-left:auto; color:var(--muted)">{k + 1:02d} / {n:02d}</span></div>
          <div style="position:absolute; left:0; top:{BBAR}px; width:{BWW - 4}px; height:{BVH:.0f}px; overflow:hidden">{vid}
            <div style="position:absolute; left:0; top:0; width:{CENTER[2]}px; height:{CENTER[2] * 9 / 16:.0f}px; transform-origin:0 0; transform:scale({(BWW - 4) / CENTER[2]:.5f})">{"".join(rp_html)}</div>
            <div id="{wid}-pg" style="position:absolute; left:0; bottom:0; width:100%; height:4px; background:var(--accent); transform-origin:0 50%"></div></div>
        </div>
        <div id="{tid}" style="position:absolute; left:96px; top:330px; width:560px">
          <div class="mono" style="font-size:22px; color:var(--accent)">{k + 1:02d} / {n:02d}<span style="color:var(--muted)">{speed}</span></div>
          <p class="h2" style="margin-top:14px; font-size:{60 if len(m) <= 12 else 50}px">{kt(m)}</p>
          <p class="body" style="margin-top:14px">{esc(what)}</p>
          {tag}
        </div>""")
        fx, fy = tx - BWX, TY - BWY
        te = t + dm + EXP * 0.5 + HOLD          # collapse starts
        script += [
            f'          tl.fromTo($("#{wid}"), {{ opacity: 0 }}, {{ opacity: 1, duration: 0.1, ease: "none" }}, {t:.3f});',
            f'          tl.fromTo($("#{wid}"), {{ x: {fx:.1f}, y: {fy:.1f}, scale: {TS:.5f} }}, {{ x: 0, y: 0, scale: 1, duration: {EXP + 0.15}, ease: SPR, immediateRender: false }}, {t:.3f});',
            f'          tl.fromTo($("#{wid}-pg"), {{ scaleX: 0 }}, {{ scaleX: 1, duration: {dm:.3f}, ease: "none" }}, {t:.3f});',
            f'          tl.to($("#{bid}-tile{k}"), {{ opacity: 0, duration: 0.12 }}, {t + 0.04:.3f});',
            f'          tl.to($("#{bid}-slot{k}"), {{ borderColor: "hsl(36 100% 57% / .8)", duration: 0.3 }}, {t:.3f});',
            f'          WORDS("#{tid} .h2", {t + 0.35:.3f}, {{ s: 0.07 }});',
            f'          UP("#{tid} .mono, #{tid} .body{", #" + tid + " .pill" if scene else ""}", {t + 0.5:.3f}, {{ s: 0.1 }});',
            f'          OUT("#{tid}", {te - 0.1:.3f}, {{ d: 0.45 }});',
            f'          tl.to($("#{wid}"), {{ x: {fx:.1f}, y: {fy:.1f}, scale: {TS:.5f}, duration: {EXP}, ease: SOFT }}, {te:.3f});',
            f'          tl.to($("#{wid}"), {{ opacity: 0, duration: 0.1, ease: "none" }}, {te + EXP - 0.12:.3f});',
            f'          tl.to($("#{bid}-tile{k}"), {{ opacity: 1, duration: 0.1 }}, {te + EXP - 0.16:.3f});',
            f'          tl.to($("#{bid}-slot{k}"), {{ borderColor: "hsl(0 0% 100% / .22)", duration: 0.3 }}, {te + EXP - 0.1:.3f});',
            f'          tl.fromTo($("#{bid}-tick{k}"), {{ opacity: 0, scale: 0.3 }}, {{ opacity: 1, scale: 1, duration: 0.5, ease: "back.out(2.5)" }}, {te + EXP - 0.05:.3f});',
        ]
        t = te + SWAP
    t += EXP - SWAP + 0.3
    if cards:
        tc = t
        nc = len(cards)
        cols, rows, cw, chh, sw, sh, x0, y0 = cards_layout(nc)
        body.append(f"""        <div id="{bid}-also" class="mono" style="position:absolute; left:96px; top:236px; font-size:24px; color:var(--dim); white-space:nowrap">
          <span style="color:var(--accent)">Also in {esc(name)}</span> · {nc} more, not shown on these scans · {esc(note)}</div>""")
        ids = []
        for k, c in enumerate(cards):
            what, icon = CARD[c]
            r_, c_ = divmod(k, cols)
            rowlen = min(cols, nc - r_ * cols)
            xr = x0 + (cols - rowlen) * (cw + 24) / 2          # centre a short last row
            x = xr + c_ * (cw + 24)
            y = y0 + r_ * (chh + 24)
            cid_ = f"{bid}-c{k}"
            ids.append(f"#{cid_}")
            st = tc + 0.9 + 0.09 * k
            svg, js = illo(c, f"{cid_}-il", st, d - 0.9 - st)
            body.append(f"""        <div id="{cid_}" style="position:absolute; left:{x:.0f}px; top:{y:.0f}px; width:{cw:.0f}px; height:{chh:.0f}px; padding:16px; border-radius:18px; overflow:hidden;
          background:hsl(222 12% 8% / .9); border:1.5px solid hsl(0 0% 100% / .1); box-shadow:0 30px 60px -28px #000">
          <div style="position:relative; width:{sw:.0f}px; height:{sh:.0f}px; border-radius:10px; background:hsl(222 12% 6%); overflow:hidden">
            <svg width="{sw:.0f}" height="{sh:.0f}" viewBox="0 0 320 170" style="position:absolute; inset:0">{svg}</svg>
          </div>
          <div style="display:flex; align-items:center; gap:10px; margin-top:14px">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="hsl(36 100% 57%)" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{ICON[icon]}</svg>
            <span style="font-size:{26 if cw > 360 else 23}px; font-weight:680; letter-spacing:-0.01em; white-space:nowrap">{esc(c)}</span>
            <span class="mono" style="margin-left:auto; font-size:16px; color:var(--muted)">{k + 1:02d}</span>
          </div>
          <div style="font-size:{19 if cw > 360 else 17}px; color:var(--dim); margin-top:6px; line-height:1.3">{esc(what)}</div>
          <div id="{cid_}-sheen" style="position:absolute; inset:0; pointer-events:none;
            background:linear-gradient(105deg, hsl(0 0% 100% / 0) 35%, hsl(0 0% 100% / .09) 50%, hsl(0 0% 100% / 0) 65%)"></div>
        </div>""")
            script.append(js.rstrip("\n"))
            script.append(f'          tl.fromTo($("#{cid_}-sheen"), {{ xPercent: -120 }}, {{ xPercent: 120, duration: 1.1, ease: "power2.inOut" }}, {tc + 0.7 + 0.09 * k:.3f});')
        script += [
            f'          tl.to({json.dumps(tiles + slots)}.map((s) => R + " " + s), {{ opacity: 0, y: 70, duration: 0.55, ease: "power3.in", stagger: 0.04 }}, {tc - 0.1:.3f});',
            f'          OUT("#{bid}-count", {tc - 0.1:.3f}, {{ d: 0.35 }});',
            f'          UP("#{bid}-also", {tc + 0.2:.3f});',
            f'          tl.fromTo({json.dumps(ids)}.map((s) => R + " " + s), {{ opacity: 0, y: 80, scale: 0.86, rotationX: 28, transformPerspective: 1400, filter: "blur(14px)" }}, '
            f'{{ opacity: 1, y: 0, scale: 1, rotationX: 0, filter: "blur(0px)", duration: 1.0, ease: SPR, stagger: {{ each: 0.09, from: "start" }} }}, {tc + 0.3:.3f});',
        ]
    # hand back to the footage
    script.append(f'          tl.to(R + " > *", {{ opacity: 0, filter: "blur(10px)", duration: 0.8, ease: "power2.inOut" }}, {d - 0.85:.3f});')
    return subcomp(bid, d, "\n".join(body), "\n".join(script), RIPPLE_CSS, bg="transparent")


# ------------------------------------------------------------------ act 4: how it's made
RAIL = ["Inputs", "Plan", "Keyframes", "Camera solve", "Training", "Physics", "Checks"]
RAIL_X = [96 + k * 272 for k in range(len(RAIL))]


def rail(cid, active):
    items = []
    for k, r in enumerate(RAIL):
        on, done = k == active, k < active
        col = "var(--accent)" if on else ("var(--ink)" if done else "var(--muted)")
        dot = "var(--accent)" if (on or done) else "hsl(222 9% 26%)"
        items.append(f'<div style="position:absolute; left:{RAIL_X[k] - 7}px; top:-7px; width:14px; height:14px; border-radius:50%; background:{dot}; '
                     f'{"box-shadow:0 0 0 6px hsl(36 100% 57% / .2);" if on else ""}"></div>'
                     f'<div class="mono" style="position:absolute; left:{RAIL_X[k] - 7}px; top:18px; font-size:21px; color:{col}; white-space:nowrap">{r}</div>')
    frac0 = (RAIL_X[max(0, active - 1)] - RAIL_X[0]) if active else 0
    frac1 = RAIL_X[active] - RAIL_X[0]
    return (f'<div id="{cid}-rail" style="position:absolute; left:0; top:62px; width:1920px; height:50px">'
            f'<div style="position:absolute; left:{RAIL_X[0]}px; top:-1px; width:{RAIL_X[-1] - RAIL_X[0]}px; height:2px; background:hsl(222 9% 20%)"></div>'
            f'<div id="{cid}-fill" style="position:absolute; left:{RAIL_X[0]}px; top:-1px; width:{max(1, frac1)}px; height:2px; background:var(--accent); transform-origin:0 50%" '
            f'data-f0="{frac0 / max(1, frac1):.4f}"></div>{"".join(items)}</div>')


# Every step pipeline.build_steps can put in a run, in run order. The video decides which
# of them a run gets (scenario.json "steps"); the rest are shown as skipped, with the reason.
ALL_STEPS = [
    ("keyframes", "Keyframes", "prepare"), ("frames", "Even-light copy", "prepare"), ("priors", "Pose priors", "prepare"),
    ("colmap", "Camera solve", "solve"), ("poses", "Camera poses", "solve"), ("dynamics", "Moving objects", "solve"),
    ("train", "Train the model", "train"), ("frame", "Scale and frame", "train"), ("depth", "Depth cross-check", "train"),
    ("export", "Export", "world"), ("sky", "Sky removal", "world"), ("clouds", "Cloud removal", "world"),
    ("reexport", "Re-export", "world"), ("colors", "Ground colours", "world"), ("collider", "Collider", "world"),
    ("objects", "Objects", "world"), ("surface", "Walk surface", "world"), ("nav", "Nav mesh", "world"),
    ("semantics", "Object types", "world"), ("rooms", "Rooms", "world"), ("texture", "Textured mesh", "world"),
    ("gate", "Quality gate", "check"), ("audit", "Plan audit", "check"), ("evals", "Eval renders", "check"),
    ("pairs", "Photo pairs", "check"), ("walktest", "Walk test", "check"),
]
STEP_NAME = {k: n for k, n, _ in ALL_STEPS}
SKIP_WHY = {"frames": "lighting measured even", "priors": "no pose log"}
GROUPS = [("prepare", "Prepare"), ("solve", "Solve"), ("train", "Train"), ("world", "Build the world"), ("check", "Check")]
GROUP_COL = {"prepare": "hsl(205 75% 62%)", "solve": "hsl(262 60% 72%)", "train": "hsl(36 100% 57%)",
             "world": "hsl(156 62% 52%)", "check": "hsl(220 15% 82%)"}
N_PRESETS = 8

RUN_SHOWN = []
RUN = []          # (name, secs) of every stage in the Boulder field run, set in pipeline()


def runbar(cid, lit):
    """The run as a step strip: all possible steps, the ones this flight got filled,
    the ones it skipped hollow, and this scene's steps lit, with names and real times."""
    chosen = {n for n, _ in RUN}
    secs = dict(RUN)
    gap, ggap = 6, 18
    w = (1728 - (len(ALL_STEPS) - 1) * gap - (len(GROUPS) - 1) * ggap) / len(ALL_STEPS)
    x, cells, labels, prev = 96.0, [], [], None
    for key, name, grp in ALL_STEPS:
        if prev and grp != prev:
            x += ggap
        if grp != prev:
            labels.append(f'<div class="mono" style="position:absolute; left:{x:.1f}px; top:40px; font-size:15px; color:{GROUP_COL[grp]}; opacity:.8; white-space:nowrap">'
                          f'{dict(GROUPS)[grp]}</div>')
        prev = grp
        if key in lit:
            style = "background:var(--accent)"
            cls = "rb rb-lit"
        elif key in chosen:
            style = "background:hsl(222 9% 30%)"
            cls = "rb"
        else:
            style = "background:transparent; border:1.5px dashed hsl(222 9% 36%)"
            cls = "rb rb-skip"
        cells.append(f'<div class="{cls}" style="position:absolute; left:{x:.1f}px; top:18px; width:{w:.1f}px; height:14px; border-radius:4px; {style}; transform-origin:50% 50%"></div>')
        x += w + gap
    if lit:
        names = " + ".join(STEP_NAME[k] for k in ALL_STEPS_KEYS if k in lit)
        t = sum(secs.get(k, 0) for k in lit)
        tstr = f"{t / 60:.0f} min" if t >= 90 else f"{t:.0f} s"
        left = f'<span style="color:var(--accent)">Now</span> · {esc(names)} · {tstr}'
    else:
        left = f'{len(ALL_STEPS)} possible steps · {N_PRESETS} capture presets'
    right = f'{len(chosen)} of {len(ALL_STEPS)} steps chosen for this flight'
    return (f'<div id="{cid}-run" style="position:absolute; left:0; top:950px; width:1920px; height:100px">{"".join(cells)}{"".join(labels)}'
            f'<div class="mono" style="position:absolute; left:96px; top:-14px; font-size:19px; color:var(--dim); white-space:nowrap">{left}</div>'
            f'<div class="mono" style="position:absolute; right:96px; top:-14px; font-size:19px; color:var(--muted); white-space:nowrap">{right}</div></div>')


ALL_STEPS_KEYS = [k for k, _, _ in ALL_STEPS]


TL_PREV = [0.0]   # where the run timeline's playhead stands when the next scene takes over
TL_SHOWN = []


def run_timeline(cid, span):
    """The footer: the real run as one timeline, drawn the same in every scene. Each
    step is a segment as long as it really took; the playhead carries on from scene to
    scene, and names the step it is in, with the run's own clock."""
    total = sum(sc for _, sc in RUN)
    x, parts, marks, prev, last_lbl = 96.0, [], [], None, -999.0
    grp_of = {k: g for k, _, g in ALL_STEPS}
    for nm, sc in RUN:
        w = sc / total * 1728
        g = grp_of.get(nm, "world")
        if g != prev:
            if x - last_lbl > 120:          # a group too short to label shares its neighbour's
                marks.append(f'<div class="mono" style="position:absolute; left:{x:.1f}px; top:72px; font-size:15px; color:{GROUP_COL[g]}; white-space:nowrap">{dict(GROUPS)[g]}</div>')
                last_lbl = x
            if prev:
                marks.append(f'<div style="position:absolute; left:{x - 1:.1f}px; top:48px; width:2px; height:22px; background:hsl(222 9% 40%)"></div>')
        prev = g
        parts.append(f'<div class="rb" style="position:absolute; left:{x:.1f}px; top:54px; width:{max(2.0, w - 2):.1f}px; height:10px; border-radius:2px; background:{GROUP_COL[g]}; opacity:.28"></div>')
        x += w
    mm, ss = divmod(round(total), 60)
    return (f'<div id="{cid}-run" style="position:absolute; left:0; top:938px; width:1920px; height:120px">'
            f'<div class="mono" style="position:absolute; left:96px; top:0; font-size:18px; color:var(--muted); white-space:nowrap">The real run · {mm} min {ss:02d} s on one laptop</div>'
            f'<div id="{cid}-clk" class="mono" style="position:absolute; right:96px; top:0; font-size:19px; color:var(--ink); white-space:nowrap">00:00 / {mm:02d}:{ss:02d}</div>'
            f'{"".join(parts)}'
            f'<div id="{cid}-prog" style="position:absolute; left:96px; top:54px; width:1728px; height:10px; border-radius:2px; background:hsl(36 100% 57% / .9); transform-origin:0 50%; transform:scaleX(0)"></div>'
            f'{"".join(marks)}'
            f'<div id="{cid}-ph" style="position:absolute; left:96px; top:46px; width:3px; height:26px; margin-left:-1.5px; border-radius:2px; background:var(--ink); box-shadow:0 0 12px hsl(36 100% 57% / .8)"></div>'
            f'<div id="{cid}-lbl" style="position:absolute; left:0; top:22px; font-size:18px; font-weight:650; color:var(--accent); white-space:nowrap"></div>'
            f'</div>')


def run_timeline_js(cid, span, dur):
    """Move the playhead from where the last scene left it to the end of `span`."""
    total = sum(sc for _, sc in RUN)
    ends, acc = [], 0.0
    for nm, sc in RUN:
        acc += sc
        ends.append([round(acc, 2), STEP_NAME.get(nm, nm)])
    t0 = TL_PREV[0]
    t1 = t0
    if span:
        acc = 0.0
        for nm, sc in RUN:
            acc += sc
            if nm in span:
                t1 = acc
    TL_PREV[0] = t1
    js = ""
    if not TL_SHOWN:
        TL_SHOWN.append(cid)
        js += (f'          tl.fromTo($("#{cid}-run .rb"), {{ scaleX: 0, transformOrigin: "0% 50%" }}, {{ scaleX: 1, duration: 0.9, ease: "expo.out", stagger: 0.02 }}, 0.9);\n'
               f'          tl.fromTo($("#{cid}-run > .mono, #{cid}-ph, #{cid}-lbl"), {{ opacity: 0 }}, {{ opacity: 1, duration: 0.6 }}, 1.3);\n')
    js += (f'          {{ const ends = {json.dumps(ends)}, total = {total:.2f};\n'
           f'            const prog = document.querySelector($("#{cid}-prog")), ph = document.querySelector($("#{cid}-ph")),\n'
           f'                  lbl = document.querySelector($("#{cid}-lbl")), clk = document.querySelector($("#{cid}-clk"));\n'
           f'            const mmss = (v) => String(Math.floor(v / 60)).padStart(2, "0") + ":" + String(Math.floor(v % 60)).padStart(2, "0");\n'
           f'            const o = {{ t: {t0:.2f} }};\n'
           f'            const draw = () => {{ const f = o.t / total; prog.style.transform = "scaleX(" + f + ")";\n'
           f'              const px = 96 + f * 1728; ph.style.transform = "translateX(" + (f * 1728) + "px)";\n'
           f'              const hit = ends.find((e) => o.t < e[0] - 0.01) || ends[ends.length - 1];\n'
           f'              lbl.textContent = o.t <= 0.01 ? "Ready to run" : hit[1];\n'
           f'              const w = lbl.offsetWidth || 160; lbl.style.transform = "translateX(" + Math.min(1824 - w, Math.max(96, px - w / 2)) + "px)";\n'
           f'              clk.textContent = mmss(Math.min(o.t, {round(total)})) + " / {round(total) // 60:02d}:{round(total) % 60:02d}"; }};\n'
           f'            tl.fromTo(o, {{ t: {t0:.2f} }}, {{ t: {t1:.2f}, duration: {dur:.2f}, ease: "power1.inOut", onUpdate: draw }}, 0); draw(); }}\n')
    return js




PT = [0.0]        # the pipeline scenes' shared clock, for a seamless background across wipes


def pscene(cid, dur, active, left, right, script, extra_css="", lit=None, lw=500, wipe=True, tl_d=None):
    t0 = PT[0]
    PT[0] += dur - X
    body = AMB_HTML + (rail(cid, active) if active is not None else "") + (run_timeline(cid, lit) if lit is not None else "") + f"""
        <div id="{cid}-left" style="position:absolute; left:96px; top:190px; width:{lw}px">{left}</div>
        {right}
        <div class="vig"></div>
        <div id="{cid}-edge" style="position:absolute; left:0; top:0; width:260px; height:1080px; pointer-events:none;
          background:linear-gradient(90deg, hsl(36 100% 57% / 0), hsl(36 100% 57% / .16) 88%, hsl(36 100% 70% / .95) 99%, hsl(36 100% 57% / 0) 100%)"></div>"""
    head = f"          AMB({dur:.2f}, {t0:.3f});\n"
    if wipe:
        # the scene wipes on over the last one; an amber edge leads it
        head += (f'          tl.fromTo(R, {{ clipPath: "inset(0% 100% 0% 0%)" }}, {{ clipPath: "inset(0% 0% 0% 0%)", duration: {X + 0.25:.2f}, ease: "expo.inOut" }}, 0);\n'
                 f'          tl.fromTo($("#{cid}-edge"), {{ x: -260, opacity: 1 }}, {{ x: 1920 - 250, duration: {X + 0.25:.2f}, ease: "expo.inOut" }}, 0);\n'
                 f'          tl.to($("#{cid}-edge"), {{ opacity: 0, duration: 0.3 }}, {X + 0.2:.2f});\n')
    else:
        head += f'          tl.fromTo(R, {{ opacity: 0 }}, {{ opacity: 1, duration: {X}, ease: "power2.inOut" }}, 0);\n'
        head += f'          tl.set($("#{cid}-edge"), {{ opacity: 0 }}, 0);\n'
    if active is not None:
        head += (f'          {{ const f = document.querySelector($("#{cid}-fill")); const f0 = +f.dataset.f0;\n'
                 f'            tl.fromTo(f, {{ scaleX: f0 }}, {{ scaleX: 1, duration: 1.1, ease: "expo.inOut" }}, 0.35); }}\n')
    if lit is not None:
        head += run_timeline_js(cid, lit, tl_d if tl_d is not None else dur - 0.2)
    head += (f'          tl.fromTo($("#{cid}-left > :not(.late)"), {{ opacity: 0, y: 34, filter: "blur(12px)" }}, '
             f'{{ opacity: 1, y: 0, filter: "blur(0px)", duration: 0.8, ease: "expo.out", stagger: 0.12 }}, 0.4);\n')
    head += f'          if (document.querySelector($("#{cid}-left .h2 .w"))) WORDS("#{cid}-left .h2", 0.45);\n'
    return subcomp(cid, dur, body, head + script, extra_css)


WX, WY, WW = 640, 190, 1184          # the media window on pipeline scenes
WH = WW * 9 / 16


def window(cid, inner, x=WX, y=WY, w=WW, h=WH):
    return f'<div id="{cid}-win" class="win" style="left:{x}px; top:{y}px; width:{w}px; height:{h:.0f}px">{inner}</div>'


def win_in(cid, at=0.3):
    return (f'          tl.fromTo($("#{cid}-win"), {{ opacity: 0, scale: 0.9, rotationX: 10, y: 60, filter: "blur(14px)", transformPerspective: 1600 }}, '
            f'{{ opacity: 1, scale: 1, rotationX: 0, y: 0, filter: "blur(0px)", duration: 1.1, ease: "expo.out" }}, {at});\n')


def stat(v, label, vid=None):
    i = f' id="{vid}"' if vid else ""
    return f'<div style="margin-top:34px"><div class="stat"{i}>{v}</div><div class="stat-l">{label}</div></div>'


def scene_sort():
    """Inputs: everything goes in with the video, and each file is recognised and filed."""
    cid = "p-sort"
    D = 10.0
    files = [("DJI_0041.MP4", "Video", "video"), ("DJI_0041.SRT", "DJI telemetry", "doc"),
             ("track.gpx", "GPX track", "map"), ("calibration.json", "Camera calibration", "camera")]
    slots = [("m", "Drone video", "1080p / 4K", "video"), ("m", "GPS coordinates", "lat, lon, height", "gps"),
             ("m", "Flight metadata", "clock, duration, datum", "meta"),
             ("o", "IMU data", "gimbal attitude", "imu"), ("o", "Barometric altitude", "height above take-off", "baro"),
             ("o", "Camera intrinsics", "focal length, centre", "intr"), ("o", "RTK / PPK corrections", "fix quality → accuracy", "rtk")]
    edges = [(0, "video"), (0, "meta"), (1, "gps"), (1, "meta"), (1, "baro"), (1, "imu"), (1, "rtk"), (2, "gps"), (3, "intr")]
    ic = {"video": '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3"/>',
          "doc": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M8 13h8M8 17h5"/>',
          "map": ICON["map"], "camera": ICON["camera"]}
    CX0, CW_, CH_, CY0, CG = 660, 340, 64, 262, 26
    SX_, SW_, SH_ = 1130, 694, 52
    sy = {}
    y = 222
    rows = []
    for gi, (grp, title) in enumerate((("m", "Mandatory"), ("o", "Optional"))):
        rows.append(f'<div id="{cid}-g{gi}" class="mono" style="position:absolute; left:{SX_}px; top:{y}px; font-size:18px; color:{"var(--accent)" if grp == "m" else "var(--muted)"}">{title}</div>')
        y += 30
        for key_grp, name, sub, key in slots:
            if key_grp != grp:
                continue
            sy[key] = y
            rows.append(f"""<div id="{cid}-s-{key}" style="position:absolute; left:{SX_}px; top:{y}px; width:{SW_}px; height:{SH_}px; border-radius:12px; padding:0 18px;
              display:flex; align-items:center; gap:14px; background:hsl(222 12% 8% / .9); border:1.5px solid hsl(0 0% 100% / .08)">
              <span id="{cid}-k-{key}" style="width:24px; height:24px; border-radius:50%; border:1.5px solid hsl(222 9% 36%); display:grid; place-items:center; font-size:14px; font-weight:800; color:hsl(225 12% 5%)">✓</span>
              <span style="font-size:23px; font-weight:640">{esc(name)}</span><span class="mono" style="font-size:16px; color:var(--muted)">{esc(sub)}</span>
              <span id="{cid}-f-{key}" class="mono" style="margin-left:auto; font-size:16px; color:var(--accent); opacity:0"></span></div>""")
            y += SH_ + 10
        y += 18
    chips = []
    for k, (fn, kind, icon) in enumerate(files):
        cy = CY0 + k * (CH_ + CG)
        chips.append(f"""<div id="{cid}-c{k}" style="position:absolute; left:{CX0}px; top:{cy}px; width:{CW_}px; height:{CH_}px; border-radius:14px; padding:0 16px;
              display:flex; align-items:center; gap:14px; background:hsl(222 12% 10%); border:1.5px solid hsl(0 0% 100% / .12); box-shadow:0 20px 40px -20px #000; overflow:hidden">
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="hsl(220 25% 96%)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{ic[icon]}</svg>
              <div><div class="mono" style="font-size:19px">{esc(fn)}</div>
              <div id="{cid}-t{k}" style="font-size:15px; color:var(--accent); margin-top:2px; opacity:0">{esc(kind)} · recognised</div></div>
              <div id="{cid}-scan{k}" style="position:absolute; top:0; bottom:0; left:0; width:60px; background:linear-gradient(90deg, hsl(36 100% 57% / 0), hsl(36 100% 57% / .35), hsl(36 100% 57% / 0)); opacity:0"></div></div>""")
    paths = []
    for e, (k, key) in enumerate(edges):
        x0, y0 = CX0 + CW_, CY0 + k * (CH_ + CG) + CH_ / 2
        x1, y1 = SX_, sy[key] + SH_ / 2
        paths.append(f'<path id="{cid}-e{e}" d="M{x0} {y0:.0f} C {x0 + 70} {y0:.0f}, {x1 - 70} {y1:.0f}, {x1} {y1:.0f}" fill="none" stroke="hsl(36 100% 57%)" '
                     f'stroke-width="2" stroke-dasharray="400" stroke-dashoffset="400" opacity=".9"/>')
    tag = {"video": ".MP4", "meta": ".MP4 + .SRT", "gps": ".SRT / .gpx", "baro": ".SRT", "imu": ".SRT", "rtk": ".SRT", "intr": ".json"}
    right = ("\n        ".join(rows) + "\n        " + "\n        ".join(chips)
             + f'\n        <svg width="1920" height="1080" style="position:absolute; left:0; top:0; overflow:visible">{"".join(paths)}</svg>'
             + f'\n        <div id="{cid}-drop" style="position:absolute; left:{CX0 - 20}px; top:{CY0 - 60}px; width:{CW_ + 40}px; height:{4 * (CH_ + CG) + 80}px; border-radius:18px; border:2px dashed hsl(0 0% 100% / .16)">'
             + f'<div class="mono" style="position:absolute; left:20px; top:16px; font-size:16px; color:var(--muted)">one drop, with the video</div></div>')
    scr = f'          tl.fromTo($("#{cid}-drop"), {{ opacity: 0, scale: 0.96 }}, {{ opacity: 1, scale: 1, duration: 0.6, ease: SPR }}, 0.3);\n'
    for k in range(len(files)):
        scr += (f'          tl.fromTo($("#{cid}-c{k}"), {{ opacity: 0, y: -160, rotation: {(-6, 5, -4, 6)[k]}, scale: 0.9 }}, '
                f'{{ opacity: 1, y: 0, rotation: 0, scale: 1, duration: 0.8, ease: SPR }}, {0.5 + 0.2 * k:.2f});\n'
                f'          tl.fromTo($("#{cid}-scan{k}"), {{ x: -60, opacity: 1 }}, {{ x: {CW_}, opacity: 1, duration: 0.55, ease: "power2.inOut" }}, {1.9 + 0.18 * k:.2f});\n'
                f'          tl.to($("#{cid}-scan{k}"), {{ opacity: 0, duration: 0.1 }}, {2.45 + 0.18 * k:.2f});\n'
                f'          tl.fromTo($("#{cid}-t{k}"), {{ opacity: 0, x: -8 }}, {{ opacity: 1, x: 0, duration: 0.35 }}, {2.2 + 0.18 * k:.2f});\n')
    scr += f'          tl.fromTo($("[id^={cid}-g], [id^={cid}-s-]"), {{ opacity: 0, x: 30 }}, {{ opacity: 1, x: 0, duration: 0.6, ease: "expo.out", stagger: 0.05 }}, 1.2);\n'
    first = {}
    for e, (k, key) in enumerate(edges):
        at = 3.1 + 0.32 * e
        first.setdefault(key, at)
        scr += f'          tl.fromTo($("#{cid}-e{e}"), {{ attr: {{ "stroke-dashoffset": 400 }} }}, {{ attr: {{ "stroke-dashoffset": 0 }}, duration: 0.55, ease: "power2.inOut" }}, {at:.2f});\n'
        scr += f'          tl.to($("#{cid}-e{e}"), {{ opacity: 0.25, duration: 0.6 }}, {at + 0.9:.2f});\n'
    for key, at in first.items():
        scr += (f'          tl.to($("#{cid}-k-{key}"), {{ backgroundColor: "hsl(156 62% 52%)", borderColor: "hsl(156 62% 52%)", duration: 0.25 }}, {at + 0.5:.2f});\n'
                f'          tl.fromTo($("#{cid}-k-{key}"), {{ scale: 0.6 }}, {{ scale: 1, duration: 0.5, ease: "back.out(3)" }}, {at + 0.5:.2f});\n'
                f'          tl.to($("#{cid}-s-{key}"), {{ borderColor: "hsl(156 62% 52% / .45)", duration: 0.3 }}, {at + 0.5:.2f});\n'
                f'          {{ const el = document.querySelector($("#{cid}-f-{key}")); el.textContent = "from {tag[key]}"; }}\n'
                f'          tl.fromTo($("#{cid}-f-{key}"), {{ opacity: 0, x: 10 }}, {{ opacity: 1, x: 0, duration: 0.35 }}, {at + 0.55:.2f});\n')
    scr += f'          UP("#{cid}-honest", 6.8);\n'
    left = f"""
          {eyebrow("Inputs · sorted on upload")}
          <p class="h2" style="margin-top:16px">{kt("Drop it all in with the *video.*")}</p>
          <p class="body" style="margin-top:18px">Each file is read, recognised and filed where it belongs. Nothing to fill in.</p>
          <p id="{cid}-honest" class="body late" style="margin-top:28px; font-size:25px; color:var(--dim)">A log with no precision in it is reported as that, never given a made-up accuracy.</p>
          <div style="margin-top:30px"><span class="pill" style="font-size:20px; padding:8px 16px"><span class="dot" style="background:var(--muted)"></span>This demo flight: video only</span></div>"""
    return cid, D, pscene(cid, D, 0, left=left, right=right, script=scr)


def scene_decide(scenario):
    """Plan: the probe measures the footage, and the run is built from what it found."""
    cid = "p-decide"
    D = 10.0
    cap = scenario["capture"]
    ap = scenario["applied"]
    chosen = set(scenario["steps"])
    ev = {e["signal"]: e["value"] for e in scenario["evidence"]}
    rows = [("Mount steadiness", f'{ev["mount_residual_px"]} px of bob', f'Aerial preset · {scenario["preset"]}'),
            ("Light across the clip", f'shadow spread {cap["shadow_area_spread"] * 100:.1f} % · drift {cap["exposure_drift_ratio"]:.2f}×', "Even-light pass: off"),
            ("Pose log", "none supplied", "Pose priors: off"),
            ("Quality", scenario["quality"], f'{ap["steps"]:,} training steps · {ap["width"]} px')]
    RX, RW, RH = 640, 1184, 58
    html = []
    for k, (sig, val, dec) in enumerate(rows):
        y = 190 + k * (RH + 10)
        html.append(f"""<div id="{cid}-r{k}" style="position:absolute; left:{RX}px; top:{y}px; width:{RW}px; height:{RH}px; border-radius:12px; padding:0 20px; overflow:hidden;
          display:flex; align-items:center; gap:16px; background:hsl(222 12% 8% / .9); border:1.5px solid hsl(0 0% 100% / .08)">
          <span class="mono" style="width:270px; font-size:18px; color:var(--muted)">{esc(sig)}</span>
          <span class="mono" style="width:360px; font-size:18px; color:var(--accent)">{esc(val)}</span>
          <svg width="30" height="14" viewBox="0 0 30 14"><path d="M0 7h26M20 1l6 6-6 6" fill="none" stroke="hsl(218 12% 62%)" stroke-width="1.8"/></svg>
          <span id="{cid}-d{k}" style="font-size:22px; font-weight:650">{esc(dec)}</span>
          <div id="{cid}-sw{k}" style="position:absolute; top:0; bottom:0; left:0; width:120px; background:linear-gradient(90deg, hsl(36 100% 57% / 0), hsl(36 100% 57% / .18), hsl(36 100% 57% / 0))"></div></div>""")
    cols, tw, th, g = 7, 160, 62, 10
    gx = RX + (RW - (cols * tw + (cols - 1) * g)) / 2
    gy = 190 + 4 * (RH + 10) + 34
    tiles = []
    for k, (key, name, grp) in enumerate(ALL_STEPS):
        r, c = divmod(k, cols)
        x, y = gx + c * (tw + g), gy + r * (th + g)
        skip = key not in chosen
        why = f'<div class="mono" style="font-size:12px; color:var(--muted); margin-top:3px">{esc(SKIP_WHY.get(key, "not needed"))}</div>' if skip else ""
        tiles.append(f"""<div id="{cid}-t{k}" class="{cid}-tile{' skip' if skip else ''}" style="position:absolute; left:{x:.0f}px; top:{y:.0f}px; width:{tw}px; height:{th}px; border-radius:10px;
          padding:10px 12px 0 16px; overflow:hidden; background:hsl(222 12% 7%); border:1.5px solid hsl(0 0% 100% / .08)">
          <div style="position:absolute; left:0; top:0; bottom:0; width:4px; background:{GROUP_COL[grp]}"></div>
          <div style="font-size:16px; font-weight:620; white-space:nowrap">{esc(name)}</div>{why}
          {'<div class="strike" style="position:absolute; left:12px; right:12px; top:21px; height:1.5px; background:hsl(218 12% 62%); transform-origin:0 50%"></div>' if skip else ''}</div>""")
    html.append(f'<div id="{cid}-gh" class="mono" style="position:absolute; left:{gx:.0f}px; top:{gy - 30:.0f}px; font-size:18px; color:var(--muted)">Every step the pipeline can run</div>')
    scr = ""
    for k in range(len(rows)):
        at = 0.6 + 0.45 * k
        scr += (f'          tl.fromTo($("#{cid}-r{k}"), {{ opacity: 0, x: 60 }}, {{ opacity: 1, x: 0, duration: 0.7, ease: "expo.out" }}, {at:.2f});\n'
                f'          tl.fromTo($("#{cid}-sw{k}"), {{ x: -120 }}, {{ x: {RW}, duration: 0.8, ease: "power2.inOut" }}, {at + 0.1:.2f});\n'
                f'          tl.fromTo($("#{cid}-d{k}"), {{ opacity: 0, x: -10, filter: "blur(6px)" }}, {{ opacity: 1, x: 0, filter: "blur(0px)", duration: 0.45 }}, {at + 0.55:.2f});\n')
    scr += (f'          UP("#{cid}-gh", 2.6);\n'
            f'          tl.fromTo($(".{cid}-tile"), {{ opacity: 0, scale: 0.8, y: 20 }}, {{ opacity: 0.55, scale: 1, y: 0, duration: 0.6, ease: SPR, stagger: {{ each: 0.03, from: "start" }} }}, 2.8);\n'
            f'          tl.to($(".{cid}-tile:not(.skip)"), {{ opacity: 1, backgroundColor: "hsl(222 12% 13%)", borderColor: "hsl(0 0% 100% / .22)", duration: 0.35, stagger: 0.09 }}, 4.4);\n'
            f'          tl.to($(".{cid}-tile.skip"), {{ opacity: 0.4, duration: 0.3 }}, 6.6);\n'
            f'          tl.fromTo($(".{cid}-tile.skip .strike"), {{ scaleX: 0 }}, {{ scaleX: 1, duration: 0.4, ease: "power2.out" }}, 6.6);\n'
            f'          COUNT("#{cid}-n", {len(chosen)}, 4.4, 2.2, (v) => Math.round(v) + " / {len(ALL_STEPS)}");\n')
    left = f"""
          {eyebrow("Plan · chosen by the video")}
          <p class="h2" style="margin-top:16px">{kt("It watches the footage, then picks its *own* *steps.*")}</p>
          <p class="body" style="margin-top:18px">{len(ALL_STEPS)} possible steps and {N_PRESETS} capture presets. Every choice comes from a measurement, and the reason is written down.</p>
          <div style="margin-top:34px"><div class="stat" id="{cid}-n">0 / {len(ALL_STEPS)}</div><div class="stat-l">steps chosen for this flight</div></div>"""
    return cid, D, pscene(cid, D, 1, lit=[], left=left, right="\n        ".join(html + tiles), script=scr)


def pipeline():
    rep = json.loads((FILM / "assets/pipeline/report.json").read_text())
    feats = json.loads((FILM / "assets/pipeline/features.json").read_text())
    log = (FILM / "assets/pipeline/train_log.txt").read_text()
    pts = [(int(a), float(b), int(c)) for a, b, c in re.findall(r"step\s+(\d+) .*?psnr ([\d.]+)\s+N (\d+)", log)]
    steps = [int(s) for s in (FILM / "assets/pipeline/timelapse_steps.txt").read_text().strip(",\n").split(",")]
    secs = {s["name"]: s["secs"] for s in rep["steps"]}
    RUN[:] = [(s["name"], s["secs"]) for s in rep["steps"]]
    scenes = []

    # -- title and input, one scene: the drone clip plays while the act opens
    cid = "p-title"
    D = 8.5
    inner = (f'<div class="crop" style="width:1280px; height:720px; transform: scale({WW / 1280:.5f})">'
             + video(f"{cid}-v", "assets/video/rocks.mp4", 0.25, D - 0.2, 0.0, 1.0, "", "width:1280px; height:720px") + "</div>")
    scenes.append((cid, D, pscene(cid, D, 0, lw=520, left=f"""
          {eyebrow("Part two")}
          <p class="h1" style="margin-top:14px; font-size:100px; letter-spacing:-0.045em">{kt("How it's *made.*")}</p>
          <p class="body" style="margin-top:18px">It starts with one pass of drone video. No ground markers, no survey team.</p>
          <div style="margin-top:34px"><div class="stat">12.05 s</div><div class="stat-l">1280×720 at 24 fps · no GPS log</div></div>""",
        right=window(cid, inner),
        script=win_in(cid, 0.25) + f'          WORDS("#{cid}-left .h1", 0.45, {{ s: 0.1, d: 1.0 }});\n')))

    scenes.append(scene_sort())

    # -- in the app
    cid = "p-app"
    app = [(4.2, 19.8, 2.6), (19.8, 40.0, 6.0)]
    t = 0.3
    vids = []
    for k, (a, b, r) in enumerate(app):
        d = (b - a) / r
        vids.append(video(f"{cid}-v{k}", "assets/video/pipeline-app.mp4", t, d, a, r))
        t += d
    ta, tb, tr = 5.0, 16.0, 3.0
    dtr = (tb - ta) / tr
    vids.append(video(f"{cid}-v2", "assets/video/pipeline-train.mp4", t, dtr, ta, tr))
    t += dtr
    dur = t + 0.4
    scenes.append((cid, dur, pscene(cid, dur, 0, lit=[], left=f"""
          {eyebrow("In the app")}
          <p class="h2" style="margin-top:16px">{kt("Drop the video in. Press *Reconstruct.*")}</p>
          <p class="body" style="margin-top:18px">It runs on this machine. Nothing is uploaded.</p>
          <div style="margin-top:40px"><span class="pill"><span class="dot" style="background:var(--accent)"></span>Live log, stage by stage</span></div>""",
        right=window(cid, crop_div("".join(vids), 0, 0, 1920, WW)), script=win_in(cid))))

    scenes.append(scene_decide(json.loads((ROOT / "work/boulder-field-0aa9375f/scenario.json").read_text())))

    # -- keyframes
    cid = "p-frames"
    thumbs = sorted((FILM / "assets/pipeline/keyframes").glob("*.jpg"))
    cells = []
    tw, th, g = 124, 70, 8
    gx0, gy0 = 640, 190
    for k, p in enumerate(thumbs):
        r, c = divmod(k, 9)
        cells.append(f'<img id="{cid}-t{k}" src="assets/pipeline/keyframes/{p.name}" style="position:absolute; left:{gx0 + c * (tw + g)}px; top:{gy0 + r * (th + g)}px; width:{tw}px; height:{th}px; border-radius:6px; object-fit:cover" />')
    scr = ""
    for r in range(8):
        ids = json.dumps([f"#{cid}-t{k}" for k in range(r * 9, min(72, r * 9 + 9))])
        scr += f'          tl.fromTo({ids}.map((s) => R + " " + s), {{ opacity: 0, x: -60, scale: 0.6, rotationY: -40, filter: "blur(8px)", transformPerspective: 800 }}, {{ opacity: 1, x: 0, scale: 1, rotationY: 0, filter: "blur(0px)", duration: 0.6, ease: "expo.out", stagger: 0.035 }}, {0.6 + r * 0.16:.2f});\n'
    scr += f'          tl.to($("#{cid}-t41"), {{ scale: 1.9, zIndex: 5, boxShadow: "0 0 0 3px hsl(36 100% 57%), 0 20px 40px -10px #000", duration: 0.6, ease: "back.out(1.7)" }}, 3.4);\n'
    scr += f'          COUNT("#{cid}-n", 72, 0.6, 1.8, (v) => Math.round(v));\n'
    scr += f'          UP("#{cid}-q", 3.6);\n'
    scenes.append((cid, 6.0, pscene(cid, 6.0, 2, lit=["keyframes"], left=f"""
          {eyebrow(f"Keyframes · {secs['keyframes']:.1f} s")}
          <p class="h2" style="margin-top:16px">{kt("The sharpest frames are kept.")}</p>
          <p class="body" style="margin-top:18px">Blurred or duplicate frames are dropped before anything is solved.</p>
          <div style="margin-top:34px"><div class="stat" id="{cid}-n">72</div><div class="stat-l">keyframes from 12 s of video</div></div>
          <p id="{cid}-q" class="mono late" style="margin-top:30px; font-size:24px; color:var(--accent)">frame 41 is the quiz frame</p>""",
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
    vd = (25.15 - solve_v0) / 2.2
    right = f"""
        <img id="{cid}-a" src="assets/pipeline/frame35.jpg" style="position:absolute; left:{ax_}px; top:{iy}px; width:{IW}px; height:{IH:.0f}px; border-radius:10px; opacity:0" />
        <div id="{cid}-b" style="position:absolute; left:{bx_}px; top:{iy}px; width:{IW}px; height:{IH:.0f}px">
          <img src="assets/pipeline/frame41.jpg" style="position:absolute; inset:0; width:100%; height:100%; border-radius:10px" />
          <svg id="{cid}-dots" width="{IW}" height="{IH:.0f}" style="position:absolute; inset:0; fill:hsl(36 100% 57%)">{dots}</svg>
        </div>
        <svg id="{cid}-lines" width="1920" height="1080" style="position:absolute; left:0; top:0; stroke:hsl(36 100% 57% / .45); stroke-width:1.6">{lines}</svg>
        <div id="{cid}-la" class="mono" style="position:absolute; left:{ax_}px; top:{iy + IH + 16:.0f}px; font-size:22px; color:var(--dim); opacity:0">frame 35 · {fa['total']:,} features</div>
        <div id="{cid}-lb" class="mono" style="position:absolute; left:{bx_}px; top:{iy + IH + 16:.0f}px; font-size:22px; color:var(--dim); opacity:0">frame 41 · {fb['total']:,} features</div>
        {window(cid, crop_div(video(f"{cid}-v", "assets/video/pipeline-viewer.mp4", 5.9, vd, solve_v0, 2.2), 0, 0, 1920, WW), y=WY)}"""
    scr = f"""          tl.fromTo($("#{cid}-b"), {{ opacity: 0, scale: 0.92, filter: "blur(12px)" }}, {{ opacity: 1, scale: 1, filter: "blur(0px)", duration: 0.8, ease: "expo.out" }}, 0.3);
          tl.fromTo($("#{cid}-dots circle"), {{ opacity: 0, scale: 0, transformOrigin: "50% 50%" }}, {{ opacity: 1, scale: 1, duration: 0.25, stagger: {{ each: 0.0015, from: "random" }} }}, 0.9);
          tl.fromTo($("#{cid}-lb"), {{ opacity: 0 }}, {{ opacity: 1, duration: 0.4 }}, 1.4);
          tl.fromTo($("#{cid}-a"), {{ opacity: 0, x: -60, filter: "blur(12px)" }}, {{ opacity: 1, x: 0, filter: "blur(0px)", duration: 0.8, ease: "expo.out" }}, 2.4);
          tl.fromTo($("#{cid}-la"), {{ opacity: 0 }}, {{ opacity: 1, duration: 0.4 }}, 2.7);
          tl.fromTo($("#{cid}-lines line"), {{ opacity: 0 }}, {{ opacity: 1, duration: 0.3, stagger: {{ each: 0.014, from: "random" }} }}, 3.0);
          COUNT("#{cid}-m", {feats['matches']['total']}, 0.9, 2.4, (v) => Math.round(v).toLocaleString("en-US"));
          tl.to([$("#{cid}-a"), $("#{cid}-b"), $("#{cid}-lines"), $("#{cid}-la"), $("#{cid}-lb")], {{ opacity: 0, scale: 1.04, filter: "blur(12px)", duration: 0.6, ease: "power2.inOut" }}, 5.4);
{win_in(cid, 5.7)}          UP("#{cid}-s2", 6.1);
"""
    dur = 5.9 + vd + 0.3
    scenes.append((cid, dur, pscene(cid, dur, 3, lit=["colmap", "poses"], left=f"""
          {eyebrow(f"Camera solve · COLMAP · {secs['colmap'] / 60:.0f} min")}
          <p class="h2" style="margin-top:16px">{kt("The same points, found in many photos.")}</p>
          <p class="body" style="margin-top:18px">Matching them places every camera in 3D.</p>
          {stat(f"{feats['matches']['total']:,}", "verified matches between these two frames", f"{cid}-m")}
          <div id="{cid}-s2" class="late">{stat("72 / 72", f"cameras placed · {feats['scene']['verified_matches'] / 1e6:.2f} M matches")}</div>""",
        right=right, script=scr)))

    # -- training timelapse
    cid = "p-train"
    rate = 0.65
    tl_d = (len(steps) / 30.0) / rate
    t0 = 0.4
    curve_w, curve_h = 460, 92
    smax, pmin, pmax = 15000, 15.0, 45.0
    poly = " ".join(f"{s / smax * curve_w:.1f},{curve_h - (p - pmin) / (pmax - pmin) * curve_h:.1f}" for s, p, n in pts)
    ns = json.dumps([[s, n] for s, _, n in pts])
    inner = f'<div class="crop" style="transform: scale({WW / 1920:.5f})">' + video(f"{cid}-v", "assets/pipeline/timelapse.mp4", t0, tl_d, 0.0, rate) + "</div>"
    right = window(cid, inner) + f"""
        <div id="{cid}-step" class="mono" style="position:absolute; left:{WX + 24}px; top:{WY + WH - 70:.0f}px; padding:8px 16px; border-radius:10px; background:hsl(225 12% 5% / .8); font-size:28px">step 1</div>"""
    scr = win_in(cid, 0.2) + f"""          const steps = {json.dumps(steps)}, ns = {ns};
          const nAt = (s) => {{ let a = ns[0]; for (const p of ns) {{ if (p[0] > s) {{ const f = (s - a[0]) / (p[0] - a[0] || 1); return a[1] + f * (p[1] - a[1]); }} a = p; }} return a[1]; }};
          const st = {{ f: 0 }};
          const stepEl = document.querySelector($("#{cid}-step")), nEl = document.querySelector($("#{cid}-gn"));
          tl.fromTo(st, {{ f: 0 }}, {{ f: {len(steps) - 1}, duration: {tl_d:.3f}, ease: "none",
            onUpdate: () => {{ const s = steps[Math.min(steps.length - 1, Math.floor(st.f))];
              stepEl.textContent = "step " + s.toLocaleString("en-US") + " / 15,000";
              nEl.textContent = Math.round(nAt(Math.max(200, s))).toLocaleString("en-US"); }} }}, {t0});
          tl.fromTo($("#{cid}-curve"), {{ strokeDashoffset: 2000 }}, {{ strokeDashoffset: 0, duration: {tl_d:.3f}, ease: "none" }}, {t0});
"""
    dur = t0 + tl_d + 1.2
    scenes.append((cid, dur, pscene(cid, dur, 4, lit=["dynamics", "train", "frame", "depth"], left=f"""
          {eyebrow(f"Training · {secs['train'] / 60:.0f} min on the GPU")}
          <p class="h2" style="margin-top:16px">{kt("The model learns to look like the photos.")}</p>
          <p class="body" style="margin-top:18px">Frame 41's camera, every 100 steps.</p>
          <div style="margin-top:34px"><div class="stat" id="{cid}-gn">47,240</div><div class="stat-l">Gaussians, the model's building blocks</div></div>
          <div style="margin-top:34px"><svg width="{curve_w}" height="{curve_h}" style="overflow:visible"><polyline id="{cid}-curve" points="{poly}" fill="none" stroke="hsl(36 100% 57%)" stroke-width="3" stroke-dasharray="2000" stroke-dashoffset="2000"/></svg>
            <div class="stat-l" style="font-size:21px">PSNR per step (noisy: a different photo each step)</div></div>""",
        right=right, script=scr)))

    # -- physics
    cid = "p-physics"
    v0, v1, r = 25.15, 34.18, 1.7
    vd = (v1 - v0) / r
    dur = vd + 0.6
    scenes.append((cid, dur, pscene(cid, dur, 5, lit=["export", "sky", "clouds", "reexport", "colors", "collider", "objects", "surface", "nav"], left=f"""
          {eyebrow(f"Physics · {secs['collider'] + secs['surface'] + secs['nav']:.0f} s")}
          <p class="h2" style="margin-top:16px">{kt("A solid surface under the picture.")}</p>
          <p class="body" style="margin-top:18px">What you measure on, walk on, and bump into.</p>
          {stat("360,714", "triangles in the collision mesh", f"{cid}-n")}""",
        right=window(cid, crop_div(video(f"{cid}-v", "assets/video/pipeline-viewer.mp4", 0.2, vd, v0, r), 0, 0, 1920, WW)),
        script=win_in(cid, 0.2) + f'          COUNT("#{cid}-n", 360714, 0.5, 1.6, (v) => Math.round(v).toLocaleString("en-US"));\n')))

    # -- checks: the quiz frame and its render, then the film ends on them
    cid = "p-checks"
    inner = (f'<img src="assets/quiz/real.png" style="position:absolute; inset:0; width:100%; height:100%; object-fit:cover" />'
             f'<img id="{cid}-r" src="assets/quiz/render.png" style="position:absolute; inset:0; width:100%; height:100%; object-fit:cover" />')
    SF = 1920 / WW
    right = window(cid, inner) + f"""
        <div id="{cid}-ov" style="position:absolute; left:{WX}px; top:{WY}px; width:{WW}px; height:{WH:.0f}px; transform-origin:0 0">
          <div id="{cid}-line" style="position:absolute; left:-2px; top:-10px; width:4px; height:{WH + 20:.0f}px; background:var(--accent); border-radius:2px;
            box-shadow:0 0 22px 2px hsl(36 100% 57% / .5)"></div>
        </div>
        <div id="{cid}-l1" class="pill" style="position:absolute; left:{WX + 24}px; top:{WY + WH - 80:.0f}px"><span class="dot" style="background:var(--accent)"></span>Rendered</div>
        <div id="{cid}-l2" class="pill" style="position:absolute; left:{WX + WW - 150}px; top:{WY + WH - 80:.0f}px"><span class="dot" style="background:var(--green)"></span>Real</div>
        <div id="{cid}-black" style="position:absolute; inset:0; background:#000; opacity:0"></div>"""
    D = 12.5
    scr = win_in(cid, 0.2) + f"""          tl.fromTo($("#{cid}-r"), {{ clipPath: "inset(0% 100% 0% 0%)" }}, {{ clipPath: "inset(0% 0% 0% 0%)", duration: 1.8, ease: "power2.inOut" }}, 0.9);
          tl.to($("#{cid}-r"), {{ clipPath: "inset(0% 50% 0% 0%)", duration: 1.2, ease: "power2.inOut" }}, 2.9);
          tl.fromTo($("#{cid}-line"), {{ x: 0, opacity: 0 }}, {{ x: {WW}, opacity: 1, duration: 1.8, ease: "power2.inOut" }}, 0.9);
          tl.to($("#{cid}-line"), {{ x: {WW / 2}, duration: 1.2, ease: "power2.inOut" }}, 2.9);
          tl.fromTo([$("#{cid}-l1"), $("#{cid}-l2")], {{ opacity: 0, y: 14 }}, {{ opacity: 1, y: 0, duration: 0.45, stagger: 0.1, ease: "power3.out" }}, 3.5);
          WORDS("#{cid}-cb", 4.4, {{ s: 0.06 }});
          // the ending: the comparison fills the frame, then black
          OUT("#{cid}-left > *, #{cid}-rail, #{cid}-run", 5.6, {{ s: 0.05 }});
          tl.set($("#{cid}-win"), {{ transformOrigin: "0% 0%" }}, 5.75);
          tl.to([$("#{cid}-win"), $("#{cid}-ov")], {{ x: {-WX}, y: {-WY - (WH * SF - 1080) / 2:.1f}, scale: {SF:.5f}, borderRadius: 0, borderWidth: 0, duration: 1.4, ease: "expo.inOut" }}, 5.8);
          tl.to($("#{cid}-line"), {{ scaleX: {1 / SF:.4f}, duration: 1.4, ease: "expo.inOut" }}, 5.8);
          tl.to($("#{cid}-l1"), {{ x: {48 - (WX + 24)}, y: {980 - (WY + WH - 80):.0f}, duration: 1.4, ease: "expo.inOut" }}, 5.8);
          tl.to($("#{cid}-l2"), {{ x: {1920 - 174 - (WX + WW - 150)}, y: {980 - (WY + WH - 80):.0f}, duration: 1.4, ease: "expo.inOut" }}, 5.8);
          tl.to($("#{cid}-r"), {{ clipPath: "inset(0% 42% 0% 0%)", duration: 2.4, ease: "sine.inOut" }}, 6.9);
          tl.to($("#{cid}-line"), {{ x: {WW * 0.58:.1f}, duration: 2.4, ease: "sine.inOut" }}, 6.9);
          tl.to($("#{cid}-black"), {{ opacity: 1, duration: 1.4, ease: "power2.inOut" }}, {D - 1.5:.2f});
"""
    ps = json.loads((ROOT / "work/boulder-field-0aa9375f/train_progress/train_report.json").read_text())["in_sample"]["psnr"]
    scenes.append((cid, D, pscene(cid, D, 6, lit=["semantics", "rooms", "texture", "gate", "audit", "evals", "pairs", "walktest"], tl_d=5.0, left=f"""
          {eyebrow(f"Checks · {secs['gate'] + secs['audit'] + secs['evals'] + secs['pairs'] + secs['walktest']:.0f} s")}
          <p class="h2" style="margin-top:16px">{kt("Rendered against the real photo.")}</p>
          <p class="body" style="margin-top:18px">Frame 41 and the model's render from the same camera.</p>
          {stat(f"{ps:.1f} dB", "PSNR on the training views")}
          <div style="margin-top:34px"><span class="pill"><span class="dot" style="background:var(--green)"></span>Quality gate: all checks passed</span></div>
          <p id="{cid}-cb" class="h2 late" style="margin-top:36px; font-size:44px">{kt("*The* *frame* *from* *the* *start.*")}</p>""",
        right=right, script=scr)))
    return scenes


# ------------------------------------------------------------------ assemble
def main():
    comps = FILM / "compositions"
    comps.mkdir(exist_ok=True)
    for old in comps.glob("*.html"):
        old.unlink()
    hosts = []
    story = []

    def add(cid, start, dur, html, title, scene, track):
        (comps / f"{cid}.html").write_text(html, encoding="utf-8")
        hosts.append((cid, start, dur, track))
        story.append((cid, title, start, dur, scene))

    mcut = cut_minis()
    add("hook", 0.0, HOOK_D, hook(), "Real or rendered?", "Render (A) against frame 41 (B); the reveal; the bridge into the model", 1)
    cap_file, cap_d, cards, caps, breaks, segs = capabilities(mcut)
    cap0 = HOOK_D - X
    add("capabilities", cap0, cap_d, capabilities_comp(cap_file, cap_d, cards, caps, breaks, segs),
        "What it can do", "The feature take as one re-timed video, with chapter slates and step captions", 0)
    for bid, s, d, num, name, minis, clist, note in breaks:
        add(bid, cap0 + s, d, feature_break(bid, d, num, name, minis, clist, note, mcut), f"More in {name}",
            f"{len(minis)} minis expanded one by one, {len(clist)} cards", 2)
    t = cap0 + cap_d - X
    for cid, d, html in pipeline():
        add(cid, t, d, html, cid, "How it's made", 3)
        t += d - X
    total = t + X

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
          "arc: Hook (real or rendered?) → Capabilities → How it's made (ends on the quiz frame)", "audience: SIH 2026 judges", "mode: autonomous", "---", ""]
    for k, (cid, title, s, d, scene) in enumerate(story, 1):
        sb += [f"## Frame {k} — {title}", "- status: animated", f"- src: compositions/{cid}.html",
               f"- duration: {d:.1f}s", f"- scene: {scene} (starts {s:.1f}s)", ""]
    (FILM / "STORYBOARD.md").write_text("\n".join(sb), encoding="utf-8")
    m, s = divmod(total, 60)
    print(f"total {int(m)}:{s:04.1f}  capabilities {cap_d:.1f}s  breaks {len(breaks)}  scenes {len(hosts)}")
    for bid, s, d, *_ in breaks:
        print(f"  {bid:12s} at {cap0 + s:6.1f}s  {d:4.1f}s")


if __name__ == "__main__":
    main()
