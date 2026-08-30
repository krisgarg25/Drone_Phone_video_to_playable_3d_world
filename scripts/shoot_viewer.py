"""Screenshot the walkable viewer from scripted camera poses (headless).

Unlike drive_viewer.py (which runs the autopilot), this parks the camera where
you ask and shoots — the fast loop for judging "does the player stand on the
splat, and does the scene look right".

  python shoot_viewer.py --asset work/rocks/viewer_assets --out work/rocks/shots
  python shoot_viewer.py --asset ... --out ... --no-underlay --no-splat

Views: spawn3p (third person at spawn), spawn1p, orbit_* (scene overview from
four sides), top (straight down). Each shot also records the player's position
and the ground hit under it, written to shots/report.json.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_server(port: int) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, str(ROOT / "_serve.py"), str(port), str(ROOT)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# Camera placements evaluated in the page. Each returns [eye, target].
VIEWS = {
    # third person, behind the player, looking at them
    "spawn3p": """(() => {
        const p = window.__playerPos();
        return [[p[0] + 5, p[1] + 3, p[2] + 5], [p[0], p[1] + 0.9, p[2]]];
    })()""",
    # eye level at the player, looking out across the scene
    "spawn1p": """(() => {
        const p = window.__playerPos(), c = window.__contentCenter();
        return [[p[0], p[1] + 1.6, p[2]], [c[0], p[1] + 1.2, c[2]]];
    })()""",
    # low ground-grazing shot: is the player's foot ON the surface?
    "feet": """(() => {
        const p = window.__playerPos();
        return [[p[0] + 3.5, p[1] + 0.6, p[2] + 3.5], [p[0], p[1] + 0.5, p[2]]];
    })()""",
    "orbit_n": """(() => { const c = window.__contentCenter(), r = window.__contentRadius();
        return [[c[0], c[1] + r * 0.55, c[2] - r * 1.5], [c[0], c[1], c[2]]]; })()""",
    "orbit_e": """(() => { const c = window.__contentCenter(), r = window.__contentRadius();
        return [[c[0] + r * 1.5, c[1] + r * 0.55, c[2]], [c[0], c[1], c[2]]]; })()""",
    "orbit_s": """(() => { const c = window.__contentCenter(), r = window.__contentRadius();
        return [[c[0], c[1] + r * 0.55, c[2] + r * 1.5], [c[0], c[1], c[2]]]; })()""",
    "orbit_w": """(() => { const c = window.__contentCenter(), r = window.__contentRadius();
        return [[c[0] - r * 1.5, c[1] + r * 0.55, c[2]], [c[0], c[1], c[2]]]; })()""",
    "top": """(() => { const c = window.__contentCenter(), r = window.__contentRadius();
        return [[c[0], c[1] + r * 2.2, c[2] + 0.01], [c[0], c[1], c[2]]]; })()""",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--page", default="pc.html")
    ap.add_argument("--views", default="all")
    ap.add_argument("--extra", default="", help="extra URL params, e.g. 'showcol=1'")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds of physics before shooting")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    views = list(VIEWS) if args.views == "all" else args.views.split(",")
    port = free_port()
    server = start_server(port)
    report = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=[
                "--use-angle=default", "--enable-unsafe-swiftshader", "--hide-scrollbars"])
            ctx = browser.new_context(viewport={"width": 1280, "height": 720})
            page = ctx.new_page()
            errs = []
            page.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
            page.on("console", lambda m: m.type == "error" and errs.append(f"console: {m.text}"))
            url = (f"http://localhost:{port}/viewer/{args.page}"
                   f"?asset=/{args.asset.as_posix()}&shoot=1"
                   + (f"&{args.extra}" if args.extra else ""))
            print(f"[shoot] {url}")
            page.goto(url)
            t0 = time.time()
            while time.time() - t0 < 120:
                if page.evaluate("window.__loadError || null"):
                    raise RuntimeError(page.evaluate("window.__loadError")
                                       + " | stage=" + str(page.evaluate("window.__stage")))
                if page.evaluate("window.__ready === true"):
                    break
                time.sleep(0.25)
            else:
                raise TimeoutError(f"never ready, stage={page.evaluate('window.__stage')}")
            page.wait_for_timeout(int(args.settle * 1000))
            report["diag"] = page.evaluate("window.__diag ? window.__diag() : null")
            print("[shoot] diag:", json.dumps(report["diag"], indent=1))
            for name in views:
                expr = VIEWS[name]
                page.evaluate(f"window.__setCam(...{expr})")
                page.wait_for_timeout(350)
                page.screenshot(path=str(args.out / f"{name}.jpg"), type="jpeg", quality=88)
                print(f"[shoot] {name}.jpg")
            report["errors"] = errs
            browser.close()
    finally:
        server.terminate()
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if report.get("errors"):
        print("[shoot] PAGE ERRORS:")
        for e in report["errors"][:20]:
            print("   ", e)


if __name__ == "__main__":
    main()
