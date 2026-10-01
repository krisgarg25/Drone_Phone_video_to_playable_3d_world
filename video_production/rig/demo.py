"""Demo-recording helpers, run inside browser-harness (its helpers `cdp`, `js` are in scope).

    exec(open(RIG / "demo.py", encoding="utf-8").read())

Everything on screen is the real app. Motion is animated inside the page, frame by
frame, so it is smooth whatever the CDP round trip costs:
* a drawn cursor (CDP input moves no OS pointer) that glides on requestAnimationFrame
  while real CDP mouse events follow the same path underneath (hover, clicks, drags);
* camera moves are the viewer's own orbit command, sent every animation frame from
  the page (`cam_glide`, `drag_orbit`), not a stream of 20-40 ms CDP drags;
* targets by visible text / aria-label / selector, or by a 3D world point projected
  through the viewer's live camera, so a click lands on the same rock in every take;
* ffmpeg Desktop Duplication capture (1920x1080, 60 fps CFR, NVENC) whose clock the
  timestamp log shares.
"""
import json
import math
import subprocess
import threading
import time
from pathlib import Path

RIG = Path(__file__).resolve().parent if "__file__" in globals() else Path(r"C:\Users\krisg\Desktop\Drone to 3d mesh\video_production\rig")
OUT = RIG.parent
ROOT = OUT.parent
FFMPEG = str(ROOT / "tools/ffmpeg/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe")
APP = "http://localhost:3000"
W, H = 1920, 1080
FPS = 60
VIEWER = "iframe:not(.compare-frame)"

# ---------------------------------------------------------------- fast CDP channel
# browser-harness opens a fresh local connection per call; the thousands of mouse
# events in a take exhaust Windows' ephemeral ports (WinError 10048). High-frequency
# calls go over one persistent websocket to the page instead; the harness keeps the
# browser-level and set-up calls.
import urllib.request
import websockets.sync.client as _wsc
from websockets.exceptions import ConnectionClosed as _Closed


class _Fast:
    def __init__(self):
        self.ws, self.n = None, 0

    def call(self, method, **params):
        for attempt in (0, 1):
            try:
                if self.ws is None:
                    with urllib.request.urlopen("http://127.0.0.1:9333/json/list", timeout=5) as r:
                        tabs = json.load(r)
                    page = next(t for t in tabs if t.get("type") == "page" and t.get("url", "").startswith(APP))
                    self.ws = _wsc.connect(page["webSocketDebuggerUrl"], max_size=None, open_timeout=5)
                self.n += 1
                i = self.n
                self.ws.send(json.dumps({"id": i, "method": method, "params": params}))
                while True:
                    m = json.loads(self.ws.recv(timeout=15))
                    if m.get("id") == i:
                        if "error" in m:
                            raise RuntimeError(f"{method}: {m['error']}")
                        return m.get("result", {})
            except (OSError, _Closed, TimeoutError, StopIteration):
                self.ws = None
                if attempt:
                    raise


_FAST = _Fast()


def fcdp(method, **params):
    return _FAST.call(method, **params)


def js(expression):
    """Evaluate in the page over the persistent channel; returns the value."""
    r = fcdp("Runtime.evaluate", expression=expression, returnByValue=True)
    if r.get("exceptionDetails"):
        raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description") or r["exceptionDetails"].get("text"))
    return r.get("result", {}).get("value")


# ---------------------------------------------------------------- page-side helpers
CURSOR_JS = r"""
(() => {
  if (window.top !== window) {
    // Viewer iframe: forward its mouse events to the cursor drawn in the top page.
    const fwd = (e) => {
      const f = window.frameElement; const top = window.top;
      if (!f || !top.__demoCursor) return;
      const r = f.getBoundingClientRect();
      top.__demoCursor.on(e.type, e.clientX + r.left, e.clientY + r.top);
    };
    for (const t of ["mousemove", "mousedown", "mouseup"]) window.addEventListener(t, fwd, true);
    return;
  }
  const ease = (u) => u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
  const install = () => {
    if (window.__demoCursor || !document.body) return;
    const hide = document.createElement("style");     // the Next.js dev badge is not the product
    hide.textContent = "nextjs-portal, [data-nextjs-dev-tools-button], #__next-build-watcher { display: none !important; }";
    document.head.appendChild(hide);
    const root = document.createElement("div");
    root.id = "demo-cursor-root";
    root.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:2147483647;overflow:hidden";
    const cur = document.createElement("div");
    cur.style.cssText = "position:absolute;left:0;top:0;width:24px;height:24px;transform:translate(-100px,-100px);will-change:transform;filter:drop-shadow(0 1px 2px rgba(0,0,0,.55))";
    cur.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24"><path d="M3 2 L3 19 L7.6 14.8 L10.6 21.6 L13.6 20.3 L10.7 13.6 L17 13.6 Z" fill="#fff" stroke="#111" stroke-width="1.4" stroke-linejoin="round"/></svg>';
    root.appendChild(cur); document.body.appendChild(root);
    let x = -100, y = -100, down = false, hidden = false, anim = 0;
    const place = () => { cur.style.transform = `translate(${x - 3}px, ${y - 2}px) scale(${down ? 0.86 : 1})`; cur.style.opacity = hidden ? 0 : 1; };
    const ripple = () => {
      const r = document.createElement("div");
      r.style.cssText = `position:absolute;left:${x - 18}px;top:${y - 18}px;width:36px;height:36px;border-radius:50%;border:2px solid rgba(255,255,255,.95);box-shadow:0 0 0 1px rgba(0,0,0,.25);opacity:.95;transform:scale(.3);transition:transform .42s cubic-bezier(.2,.8,.2,1),opacity .42s ease-out`;
      root.appendChild(r);
      requestAnimationFrame(() => { r.style.transform = "scale(1.35)"; r.style.opacity = "0"; });
      setTimeout(() => r.remove(), 500);
    };
    window.__demoCursor = {
      // While a glide runs, real events only drive press/release: the glide owns the position.
      on(type, cx, cy) {
        if (type.endsWith("down")) { if (!anim) { x = cx; y = cy; } down = true; ripple(); }
        else if (type.endsWith("up")) down = false;
        else if (!anim) { x = cx; y = cy; }
        place();
      },
      glide(x1, y1, dur, arc) {
        const x0 = x, y0 = y, nx = -(y1 - y0), ny = x1 - x0, t0 = performance.now(), id = ++anim;
        const step = (t) => {
          if (anim !== id) return;
          const u = Math.min(1, (t - t0) / dur), e = ease(u), b = Math.sin(Math.PI * e) * arc;
          x = x0 + (x1 - x0) * e + nx * b; y = y0 + (y1 - y0) * e + ny * b; place();
          if (u < 1) requestAnimationFrame(step); else anim = 0;
        };
        requestAnimationFrame(step);
      },
      press(v) { down = v; if (v) ripple(); place(); },
      hide(v) { hidden = v; place(); },
    };
    const own = (e) => window.__demoCursor.on(e.type, e.clientX, e.clientY);
    for (const t of ["mousemove", "mousedown", "mouseup"]) window.addEventListener(t, own, true);
    // The viewer's own orbit, driven every animation frame from here.
    window.__camGlide = (a, b, dur) => {
      const f = document.querySelector("iframe:not(.compare-frame)"), t0 = performance.now();
      const step = (t) => {
        const u = Math.min(1, (t - t0) / dur), e = ease(u), v = a.map((p, i) => p + (b[i] - p) * e);
        const value = { target: v.slice(0, 3), distance: v[3], yaw: v[4], pitch: v[5] };
        if (v.length > 6) value.fov = v[6];
        f.contentWindow.postMessage({ namespace: "groundcontrol", type: "command", command: "camera-set", value }, location.origin);
        if (u < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };
    window.addEventListener("message", (e) => { if (e.data?.namespace === "groundcontrol" && e.data.type === "camera") window.__lastOrbit = e.data.orbit; });
  };
  if (document.body) install(); else document.addEventListener("DOMContentLoaded", install);
})();
"""


def install_cursor():
    """Every document (and the viewer iframe) gets the helpers, now and after any reload."""
    if not globals().get("_cursor_registered"):
        cdp("Page.enable")
        cdp("Page.addScriptToEvaluateOnNewDocument", source=CURSOR_JS)
        globals()["_cursor_registered"] = True
    js(CURSOR_JS)
    js("""(() => { for (const f of document.querySelectorAll('iframe')) { try { f.contentWindow.eval(%s); } catch (e) {} } })()""" % json.dumps(CURSOR_JS))


# ---------------------------------------------------------------- viewport
def ensure_window():
    """A minimized window stops requestAnimationFrame, and WebGL never reports ready."""
    w = cdp("Browser.getWindowForTarget")
    if w["bounds"].get("windowState") == "minimized":
        cdp("Browser.setWindowBounds", windowId=w["windowId"], bounds={"windowState": "normal"})
        time.sleep(1.0)


def emulate_1080p(on=True):
    """Rehearsal in a small window: lay the page out at exactly 1920x1080."""
    if on:
        cdp("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
    else:
        cdp("Emulation.clearDeviceMetricsOverride")


def shot(path):
    """A rehearsal still of the whole 1920x1080 page."""
    import base64
    data = cdp("Page.captureScreenshot", format="jpeg", quality=80)["data"]
    Path(path).write_bytes(base64.b64decode(data))
    return path


# ---------------------------------------------------------------- targets
_POS = [W / 2, H / 2]

FIND_JS = r"""
((q) => {
  const vis = (el) => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el); return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none"; };
  let scope = document;
  if (q.within) { scope = [...document.querySelectorAll(q.within)].find(vis) || document; }
  let els = [...scope.querySelectorAll(q.sel || "button, a, [role=button], [role=tab], [role=slider], label, summary, input, select, [data-tip], [aria-label], li, h2, h3")];
  if (q.text) {
    const norm = (s) => (s || "").replace(/\s+/g, " ").trim().toLowerCase();
    const want = norm(q.text);
    els = els.filter((el) => {
      const t = [norm(el.innerText), norm(el.getAttribute("aria-label")), norm(el.getAttribute("data-tip")), norm(el.getAttribute("title"))];
      return q.exact ? t.includes(want) : t.some((v) => v && v.includes(want));
    });
    if (!q.sel) els = els.filter((el) => !els.some((o) => o !== el && el.contains(o)));   // innermost match
  }
  els = els.filter(vis);
  const el = els[q.nth || 0];
  if (!el) return null;
  el.scrollIntoView({ block: "nearest", inline: "nearest" });
  const r = el.getBoundingClientRect();
  return [r.left + r.width * (q.fx ?? 0.5), r.top + r.height * (q.fy ?? 0.5), r.width, r.height];
})
"""


def find(text=None, sel=None, within=None, nth=0, exact=False, fx=0.5, fy=0.5, timeout=6.0):
    q = {"text": text, "sel": sel, "within": within, "nth": nth, "exact": exact, "fx": fx, "fy": fy}
    end = time.time() + timeout
    while True:
        r = js(f"({FIND_JS})({json.dumps(q)})")
        if r:
            return r[0], r[1]
        if time.time() > end:
            raise LookupError(f"not found: {q}")
        time.sleep(0.15)


def exists(**q):
    try:
        find(timeout=0.3, **q)
        return True
    except LookupError:
        return False


def viewer_js(expr):
    """Evaluate inside the main viewer iframe (same origin)."""
    return js("(() => { const w = document.querySelector(%s)?.contentWindow; return w ? w.eval(%s) : null; })()"
              % (json.dumps(VIEWER), json.dumps(expr)))


def world(x, y, z):
    """Screen position of a world point, through the viewer's live camera."""
    r = viewer_js("""(() => { const a = window.__app;
      const cam = a.root.findComponents('camera').find(c => c.enabled && c.entity.enabled);
      const V = cam.entity.getPosition().constructor;
      const s = cam.worldToScreen(new V(%f, %f, %f));          // canvas CSS px
      const c = a.graphicsDevice.canvas.getBoundingClientRect(), f = window.frameElement.getBoundingClientRect();
      return [f.left + c.left + s.x, f.top + c.top + s.y, s.z]; })()""" % (x, y, z))
    return r[0], r[1]


def post(command, **value):
    """A workspace command straight to the viewer (set-up only, never on camera)."""
    js("(() => { const f = document.querySelector(%s); f.contentWindow.postMessage(%s, location.origin); })()"
       % (json.dumps(VIEWER), json.dumps({"namespace": "groundcontrol", "type": "command", "command": command, **value})))


def set_orbit(target, distance, yaw, pitch):
    post("camera-set", value={"target": list(target), "distance": distance, "yaw": yaw, "pitch": pitch})


def get_orbit():
    """The viewer's current orbit {target, distance, yaw, pitch}."""
    js("window.__lastOrbit = null")
    post("camera-follow", value=True)
    end = time.time() + 1.5
    o = None
    while time.time() < end and not o:
        time.sleep(0.05)
        o = js("window.__lastOrbit")
    post("camera-follow", value=False)
    return o


def cam_fov():
    return viewer_js("(() => { const c = window.__app.root.findComponents('camera').find(c => c.enabled && c.entity.enabled); return c.fov; })()")


def _orbit_vec(o, fov=None):
    return [*o["target"], o["distance"], o["yaw"], o["pitch"]] + ([fov] if fov is not None else [])


def cam_glide(target, distance, yaw, pitch, dur=2.0, wait=True):
    """A smooth camera move to a pose, animated in the page every frame."""
    o = get_orbit()
    fov = cam_fov() or 70
    b = [*target, distance, yaw, pitch, 70]
    a = _orbit_vec(o, fov) if o else b
    b[4] = a[4] + ((b[4] - a[4] + math.pi) % (2 * math.pi) - math.pi)   # shortest way round
    js("window.__camGlide(%s, %s, %d)" % (json.dumps(a), json.dumps(b), int(dur * 1000)))
    if wait:
        time.sleep(dur + 0.05)


def drag_orbit(dyaw=0.0, dpitch=0.0, zoom=1.0, dur=2.2, at=(760, 560)):
    """What a hand on the mouse does - press, drag, release - with the camera following
    the viewer's own orbit every frame instead of 20-40 ms mouse steps."""
    o = get_orbit()
    if not o:
        return
    a = _orbit_vec(o)
    b = [*a[:3], a[3] * zoom, a[4] + dyaw, max(0.05, min(1.45, a[5] + dpitch))]
    move(at)
    time.sleep(0.12)
    js("window.__demoCursor.press(true)")
    x1, y1 = at[0] - dyaw * 420, at[1] + dpitch * 420
    js("window.__demoCursor.glide(%f, %f, %d, 0)" % (x1, y1, int(dur * 1000)))
    js("window.__camGlide(%s, %s, %d)" % (json.dumps(a), json.dumps(b), int(dur * 1000)))
    time.sleep(dur + 0.05)
    js("window.__demoCursor.press(false)")
    _mouse("mouseMoved", x1, y1)
    _POS[:] = [x1, y1]
    time.sleep(0.15)


def zoom_to(factor, dur=1.4):
    """Scroll-wheel zoom, as a smooth dolly of the orbit distance."""
    o = get_orbit()
    if o:
        a = _orbit_vec(o)
        b = [*a[:3], a[3] * factor, a[4], a[5]]
        js("window.__camGlide(%s, %s, %d)" % (json.dumps(a), json.dumps(b), int(dur * 1000)))
        time.sleep(dur + 0.05)


# ---------------------------------------------------------------- input
def _ease(t):
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def _mouse(kind, x, y, button="none", buttons=0, clicks=0):
    fcdp("Input.dispatchMouseEvent", type=kind, x=x, y=y, button=button, buttons=buttons, clickCount=clicks)


def _sleep_until(t):
    while True:
        left = t - time.perf_counter()
        if left <= 0:
            return
        time.sleep(min(left, 0.004))


def move(to, dur=None, button="none", buttons=0, arc=0.06):
    """Eased, slightly curved path. The drawn cursor glides in the page at display rate;
    real CDP events follow the same path underneath for hover and drag state."""
    x0, y0 = _POS
    x1, y1 = to
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1:
        _mouse("mouseMoved", x1, y1, button, buttons)
        return
    dur = dur if dur is not None else min(1.05, 0.38 + dist / 2200)
    js("window.__demoCursor && window.__demoCursor.glide(%f, %f, %d, %f)" % (x1, y1, int(dur * 1000), arc))
    nx, ny = -(y1 - y0), (x1 - x0)
    t0 = time.perf_counter()
    while True:
        u = min(1.0, (time.perf_counter() - t0) / dur)
        t = _ease(u)
        bend = math.sin(math.pi * t) * arc
        _mouse("mouseMoved", x0 + (x1 - x0) * t + nx * bend, y0 + (y1 - y0) * t + ny * bend, button, buttons)
        if u >= 1.0:
            break
    _POS[:] = [x1, y1]


def _pt(target):
    if isinstance(target, (tuple, list)):
        return float(target[0]), float(target[1])
    if isinstance(target, dict):
        return find(**target)
    return find(text=target)


def hover(target, dwell=0.5, **kw):
    move(_pt(target), **kw)
    time.sleep(dwell)


def click(target, dwell=0.14, after=0.35, **kw):
    x, y = _pt(target)
    move((x, y), **kw)
    time.sleep(dwell)
    _mouse("mousePressed", x, y, "left", 1, 1)
    time.sleep(0.07)
    _mouse("mouseReleased", x, y, "left", 0, 1)
    time.sleep(after)
    return x, y


def drag(start, end, dur=1.2, button="left"):
    """A real press-move-release (sliders, handles, items)."""
    bmask = {"left": 1, "right": 2, "middle": 4}[button]
    move(_pt(start))
    time.sleep(0.12)
    x, y = _POS
    _mouse("mousePressed", x, y, button, bmask, 1)
    move(_pt(end), dur=dur, button=button, buttons=bmask, arc=0.0)
    x, y = _POS
    _mouse("mouseReleased", x, y, button, 0, 1)
    time.sleep(0.2)


KEYS = {"w": ("KeyW", 87), "a": ("KeyA", 65), "s": ("KeyS", 83), "d": ("KeyD", 68),
        "Escape": ("Escape", 27), "Enter": ("Enter", 13), "Backspace": ("Backspace", 8), "\\": ("Backslash", 220)}


def key_down(k):
    code, vk = KEYS.get(k, (k, 0))
    fcdp("Input.dispatchKeyEvent", type="keyDown", key=k, code=code, windowsVirtualKeyCode=vk, nativeVirtualKeyCode=vk)


def key_up(k):
    code, vk = KEYS.get(k, (k, 0))
    fcdp("Input.dispatchKeyEvent", type="keyUp", key=k, code=code, windowsVirtualKeyCode=vk, nativeVirtualKeyCode=vk)


def tap(k, after=0.3):
    key_down(k)
    time.sleep(0.06)
    key_up(k)
    time.sleep(after)


def look(dx, dy=0, dur=1.0, start=(960, 540)):
    """First-person mouse look under pointer lock: a steady sweep of the hidden pointer."""
    x0, y0 = start
    t0 = time.perf_counter()
    while True:
        u = min(1.0, (time.perf_counter() - t0) / dur)
        e = _ease(u)
        fcdp("Input.dispatchMouseEvent", type="mouseMoved", x=x0 + dx * e, y=y0 + dy * e, button="none")
        if u >= 1.0:
            break
    _POS[:] = [x0 + dx, y0 + dy]


def cursor_visible(on):
    js("window.__demoCursor && window.__demoCursor.hide(%s)" % ("false" if on else "true"))


def select_all():
    fcdp("Input.dispatchKeyEvent", type="keyDown", key="a", code="KeyA", windowsVirtualKeyCode=65, modifiers=2)
    fcdp("Input.dispatchKeyEvent", type="keyUp", key="a", code="KeyA", windowsVirtualKeyCode=65, modifiers=2)


def type_slow(text, cps=14):
    for ch in text:
        fcdp("Input.insertText", text=ch)
        time.sleep(1 / cps)


# ---------------------------------------------------------------- waiting
def wait_text(text, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if js("document.body.innerText.includes(%s)" % json.dumps(text)):
            return True
        time.sleep(0.2)
    raise TimeoutError(f"text never appeared: {text}")


def wait_ready(timeout=60):
    ensure_window()
    wait_text("3D view ready", timeout)
    install_cursor()


def reset(scene):
    """Back to the clean state: move whatever earlier takes saved into the workspace
    aside (to _archived_workspaces/rig_reset/<scene>/<time>/), never delete it."""
    import shutil
    keep = set(json.loads((RIG / "clean_state.json").read_text())[scene])
    root = ROOT
    work = root / "work" / scene
    extra = [p for p in work.iterdir() if p.name not in keep]
    # Schemes and missions live in proposals/ (existing.json there is only a cache).
    extra += [p for pat in ("proposals/index.json", "proposals/p-*.json") for p in work.glob(pat)
              if p.relative_to(work).parts[0] in keep]           # else the whole folder already goes
    if not extra:
        return
    dest = root / "_archived_workspaces" / "rig_reset" / scene / time.strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for p in extra:
        target = dest / p.relative_to(work)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(target))


# ---------------------------------------------------------------- recording + timeline
class Take:
    """One continuous recording: ffmpeg capture plus a chaptered timestamp log on the
    capture's own clock."""

    def __init__(self, name, record=True, stills=None):
        self.name, self.record, self.stills = name, record, stills
        self.events, self.chapters, self.t0, self.proc = [], [], None, None
        self.path = OUT / "raw" / f"{name}.mp4"

    def __enter__(self):
        if self.record:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.proc = subprocess.Popen(
                [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", f"ddagrab=output_idx=0:framerate={FPS}:draw_mouse=0",
                 "-vf", f"hwdownload,format=bgra,fps={FPS}", "-c:v", "h264_nvenc", "-preset", "p5",
                 "-rc", "vbr", "-cq", "19", "-b:v", "0", "-maxrate", "40M", "-bufsize", "80M",
                 "-g", str(FPS * 2), "-pix_fmt", "yuv420p",
                 "-progress", "pipe:1", "-nostats", str(self.path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            got = threading.Event()

            def watch():
                for raw in self.proc.stdout:
                    line = raw.decode(errors="replace").strip()
                    if line.startswith("out_time_us=") and not got.is_set():
                        v = line.split("=")[1]
                        us = int(v) if v.lstrip("-").isdigit() else 0
                        if us > 0:
                            self.t0 = time.perf_counter() - us / 1e6
                            got.set()
            threading.Thread(target=watch, daemon=True).start()
            if not got.wait(15):
                raise RuntimeError("ffmpeg produced no frames")
        else:
            self.t0 = time.perf_counter()
        time.sleep(1.0)                                   # head handle
        return self

    def now(self):
        return time.perf_counter() - self.t0

    def chapter(self, title):
        if self.chapters:
            self.chapters[-1]["end"] = round(self.now(), 2)
        self.chapters.append({"title": title, "start": round(self.now(), 2), "end": None})

    def step(self, label):
        take = self

        class _Step:
            def __enter__(s):
                s.start = take.now()
                return s

            def __exit__(s, *exc):
                take.events.append({"chapter": take.chapters[-1]["title"] if take.chapters else "",
                                    "action": label, "start": round(s.start, 2), "end": round(take.now(), 2)})
                if take.stills:
                    shot(Path(take.stills) / f"{take.name}-{len(take.events):02d}.jpg")
        return _Step()

    def __exit__(self, *exc):
        time.sleep(1.5)                                   # tail handle
        length = self.now()
        if self.chapters:
            self.chapters[-1]["end"] = round(length, 2)
        if self.proc:
            try:
                self.proc.stdin.write(b"q")
                self.proc.stdin.flush()
            except OSError:
                pass
            self.proc.wait(30)
        log = OUT / "raw" / f"{self.name}.json"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps({"clip": self.name, "length": round(length, 2), "recorded": bool(self.record),
                                   "failed": exc[0] is not None, "chapters": self.chapters,
                                   "events": self.events}, indent=1))
        return False
