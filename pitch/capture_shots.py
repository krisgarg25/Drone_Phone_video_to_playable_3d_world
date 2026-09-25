"""Capture clean 1920x1080 stills of the running app for the pitch deck."""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(exist_ok=True)
PORT = 8137
BASE = f"http://127.0.0.1:{PORT}"

HIDE_CHROME = """
() => {
  for (const id of ['hud-container','load','cam-inspector','cov-panel']) {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  }
}
"""


def wait_ready(page, timeout_s=120):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = page.evaluate("() => (window.__walk ? 'ok' : (document.getElementById('load')||{}).textContent || '')")
        if st == "ok":
            return True
        if isinstance(st, str) and st.startswith("ERROR"):
            print("  !", st)
            return False
        time.sleep(0.5)
    return False


def shoot(page, name, settle=3.0):
    time.sleep(settle)
    for attempt in range(3):
        try:
            page.screenshot(path=str(OUT / name), type="jpeg", quality=93, timeout=90000)
            print("  ->", name)
            return
        except Exception as exc:
            print(f"  retry {name}: {type(exc).__name__}")
            time.sleep(2)


def main():
    shots = sys.argv[1:] or ["all"]

    def want(s):
        return "all" in shots or s in shots

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=[
            "--use-angle=default", "--enable-unsafe-swiftshader",
            "--disable-lcd-text", "--hide-scrollbars",
        ])
        ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = ctx.new_page()
        page.on("pageerror", lambda e: print("  [page]", e))

        if want("viewer"):
            for scene, tag in [("auditorium", "hall"), ("rocks", "rocks")]:
                print(f"[viewer {scene}]")
                page.goto(f"{BASE}/viewer/pc.html?asset={scene}", wait_until="load")
                if not wait_ready(page):
                    print("  not ready, skipping")
                    continue
                page.evaluate(HIDE_CHROME)
                shoot(page, f"shot-{tag}-walk.jpg", 5)

                page.keyboard.press("KeyX")
                page.keyboard.press("KeyG")
                shoot(page, f"shot-{tag}-shell.jpg", 3)

                page.keyboard.press("KeyG")
                shoot(page, f"shot-{tag}-collider.jpg", 2.5)

                page.goto(f"{BASE}/viewer/pc.html?asset={scene}&drone=1", wait_until="load")
                wait_ready(page)
                page.evaluate(HIDE_CHROME)
                page.keyboard.press("KeyP")
                page.keyboard.press("KeyX")
                page.keyboard.press("KeyG")
                shoot(page, f"shot-{tag}-solve.jpg", 5)

        if want("gui"):
            print("[studio]")
            page.goto(f"{BASE}/viewer/pipeline_gui.html", wait_until="load")
            shoot(page, "shot-studio.jpg", 5)

        if want("scan"):
            print("[capture page]")
            page.goto(f"{BASE}/viewer/capture.html", wait_until="load")
            shoot(page, "shot-capture.jpg", 6)

        browser.close()


if __name__ == "__main__":
    main()
