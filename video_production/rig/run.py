"""Run the continuous demo take through browser-harness.

    python video_production/rig/launch_chrome.py [--record]
    set BH_TELEMETRY=0 & set BU_CDP_URL=http://127.0.0.1:9333 & set MODE=rehearse
    browser-harness < video_production/rig/run.py

MODE=rehearse: 1080p emulation in a normal window, no capture, a still after every step
              (video_production/raw/stills/).
MODE=record:   fullscreen Chrome (launch_chrome.py --record), one ffmpeg capture for the
              whole take -> video_production/raw/demo.mp4 + demo.json.
"""
import os
import traceback

from pathlib import Path

RIG_DIR = str(Path(__file__).resolve().parent)
exec(open(os.path.join(RIG_DIR, "demo.py"), encoding="utf-8").read())
exec(open(os.path.join(RIG_DIR, os.environ.get("TAKEFILE", "take.py")), encoding="utf-8").read())

MODE = os.environ.get("MODE", "rehearse")
NAME = os.environ.get("TAKE", "demo")
stills = OUT / "raw" / "stills"
stills.mkdir(parents=True, exist_ok=True)
ensure_window()
emulate_1080p(MODE == "rehearse")
try:
    setup()
    with Take(NAME, record=(MODE == "record"), stills=(stills if MODE == "rehearse" else None)) as t:
        run(t)
    print(f"ok  {t.now():.1f}s", flush=True)
except Exception:
    traceback.print_exc()
    shot(stills / f"{NAME}-FAILED.jpg")
finally:
    # Leave the workspaces as the person had them: the take's own saves go to the
    # archive, and rocks gets back the measurement and plan it had before any take.
    import shutil
    reset(ROCKS)
    reset(FLAT)
    keep = ROOT / "_archived_workspaces" / "rocks_quality_state_2026-09-28"
    for f in ("measurements.json", "plan.json"):
        if keep.is_dir() and (keep / f).is_file():
            shutil.copy2(keep / f, ROOT / "work" / ROCKS / f)
