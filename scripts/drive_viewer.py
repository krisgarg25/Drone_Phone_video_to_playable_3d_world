"""Drive the walkable PlayCanvas viewer (viewer/pc.html) with Playwright.

  python drive_viewer.py walk --asset work/rocks/viewer_assets --out work/rocks/walktest

walk: load pc.html?asset=...&auto=1 headless, screenshot periodically,
save walk_log.json + walk.webm. Eval renders are NOT browser-based anymore;
produce them offline with scripts/render_evals_offline.py instead.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent


PORT = 8137


def start_server():
    """Serve the repo root; forces correct MIME for .mjs/.wasm (stock
    http.server on Windows serves them as text/plain, and browsers then
    refuse the ES module / wasm stream)."""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
            return None
    except (OSError, ConnectionRefusedError):
        pass

    helper = ROOT / "_serve.py"
    if not helper.exists():
        helper.write_text(
            "import functools, http.server, sys\n"
            "class H(http.server.SimpleHTTPRequestHandler):\n"
            "    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,\n"
            "                      '.mjs': 'text/javascript', '.wasm': 'application/wasm'}\n"
            "    def log_message(self, *a): pass\n"
            "port, root = int(sys.argv[1]), sys.argv[2]\n"
            "http.server.ThreadingHTTPServer(('127.0.0.1', port),\n"
            "    functools.partial(H, directory=root)).serve_forever()\n",
            encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(helper), str(PORT), str(ROOT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 5.0:
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=0.2):
                return proc
        except (OSError, ConnectionRefusedError):
            time.sleep(0.1)
    raise RuntimeError(f"_serve.py timed out binding to port {PORT}")


def open_page(pw, headless=False, record_dir: Path | None = None):
    from playwright.sync_api import BrowserType
    browser = pw.chromium.launch(headless=headless, args=[
        "--use-angle=default", "--enable-unsafe-swiftshader",
        "--disable-lcd-text", "--hide-scrollbars",
    ])
    ctx_kwargs = {"viewport": {"width": 1280, "height": 720},
                  "record_video_dir": str(record_dir) if record_dir else None}
    ctx = browser.new_context(**ctx_kwargs)
    page = ctx.new_page()
    page.on("pageerror", lambda e: print(f"[drive] page error: {e}"))
    page.on("console", lambda m: m.type == "error" and print(f"[drive] console: {m.text}"))
    return browser, ctx, page


def wait_ready(page, timeout_s=60):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if page.evaluate("window.__loadError || null"):
            raise RuntimeError(page.evaluate("window.__loadError"))
        if page.evaluate("window.__ready === true"):
            return
        time.sleep(0.25)
    raise TimeoutError("viewer never became ready")


def run_walk(asset: Path, out: Path, page_file: str = "pc.html"):
    out.mkdir(parents=True, exist_ok=True)
    frames = out / "frames"
    if frames.exists():
        for f in frames.glob("*.jpg"):
            f.unlink()
    frames.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        server = start_server()
        try:
            rel_asset = asset.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            rel_asset = asset.as_posix()
        url = (f"http://127.0.0.1:{PORT}/viewer/{page_file}?asset=/{rel_asset.lstrip('/')}"
               f"&auto=1")
        browser, ctx, page = open_page(pw, headless=True, record_dir=out)
        page.goto(url)
        wait_ready(page)
        t0 = time.time()
        shot_n = 0
        last_phase = None
        while time.time() - t0 < 240:
            phase = page.evaluate("window.__walk.phase")
            if phase != last_phase:
                print(f"[drive] phase -> {phase} (t={time.time()-t0:.0f}s)")
                last_phase = phase
            page.screenshot(path=str(frames / f"{shot_n:04d}.jpg"),
                            type="jpeg", quality=80)
            shot_n += 1
            if phase == "done":
                break
            page.wait_for_timeout(600)
        log = page.evaluate("window.__walk")
        (out / "walk_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
        dist = log["walked"]
        print(f"[drive] walk finished: phase={log['phase']} dist={dist:.1f}m "
              f"falls={log['violations']} samples={len(log['samples'])}")
        video = page.video
        video_path = video.path() if video else None
        browser.close()  # flushes/closes the recorded webm
        if server:
            server.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["walk"],
                    help="only 'walk' remains; eval renders come from render_evals_offline.py")
    ap.add_argument("--asset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--page", default="pc.html",
                    help="viewer page under viewer/ (default: pc.html)")
    args = ap.parse_args()
    run_walk(args.asset, args.out, page_file=args.page)


if __name__ == "__main__":
    main()
