"""Start the demo Chrome: its own profile, CDP on :9333, device scale factor 1.

  python video_production/rig/launch_chrome.py            # windowed, for rehearsal
  python video_production/rig/launch_chrome.py --record   # fullscreen 1920x1080, for takes

The profile lives next to this file (git-ignored), so the person's own Chrome and its
sign-ins are never touched. Scale factor 1 makes a fullscreen window exactly 1920x1080
CSS px on a 1920x1080 panel even with Windows set to 125 %.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9333
PROFILE = Path(__file__).resolve().parent / ".chrome-profile"


def alive() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=1) as r:
            return bool(json.load(r).get("webSocketDebuggerUrl"))
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="fullscreen for a take")
    ap.add_argument("--url", default="http://localhost:3000/")
    args = ap.parse_args()
    if alive():
        print(f"demo chrome already on :{PORT}")
        return
    PROFILE.mkdir(exist_ok=True)
    flags = [f"--remote-debugging-port={PORT}", f"--user-data-dir={PROFILE}",
             "--force-device-scale-factor=1", "--no-first-run", "--no-default-browser-check",
             "--disable-features=Translate,MediaRouter", "--hide-crash-restore-bubble",
             "--disable-session-crashed-bubble", "--password-store=basic",
             # A covered window must keep rendering, or WebGL never reports ready.
             "--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding",
             "--disable-background-timer-throttling",
             # Laptops default the browser to the integrated GPU: splats then render at ~28 fps.
             "--force_high_performance_gpu"]
    flags += ["--start-fullscreen", "--window-position=0,0"] if args.record else ["--window-position=40,40", "--window-size=1320,800"]
    subprocess.Popen([CHROME, *flags, args.url], creationflags=subprocess.DETACHED_PROCESS)
    for _ in range(40):
        if alive():
            print(f"demo chrome up on :{PORT} ({'fullscreen' if args.record else 'windowed'})")
            return
        time.sleep(0.25)
    raise SystemExit("demo chrome did not open its CDP port")


if __name__ == "__main__":
    main()
