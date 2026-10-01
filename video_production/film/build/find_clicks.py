"""Find every click in a rig recording: where the drawn cursor was, and when it pressed.

    .venv/Scripts/python.exe video_production/film/build/find_clicks.py raw/demo.mp4 out.json

The rig draws its own cursor (demo.py CURSOR_JS): a white arrow with a dark outline whose
tip is the pointer, and on every press a white ring that grows from the tip. This tracks
the arrow by template matching and reports a click wherever a ring appears around a
still cursor. Output: [{"t": seconds, "x": px, "y": px}] on the video's own clock.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
FFMPEG = str(ROOT / "tools/ffmpeg/ffmpeg-9.0.1-essentials_build/bin/ffmpeg_real.exe")
W, H, FPS = 1920, 1080, 60

# the cursor as demo.py draws it, tip at (3, 2) of a 24 px box
POLY = [(3, 2), (3, 19), (7.6, 14.8), (10.6, 21.6), (13.6, 20.3), (10.7, 13.6), (17, 13.6)]


def template(scale=1.0):
    """The arrow at `scale` about its box centre (demo.py scales it to 0.86 while pressed)."""
    k = 8
    img = np.full((24 * k, 24 * k), 128, np.uint8)
    mask = np.zeros_like(img)
    poly = [(12 + (x - 12) * scale, 12 + (y - 12) * scale) for x, y in POLY]
    pts = (np.array(poly) * k).astype(np.int32)
    cv2.fillPoly(img, [pts], 255)
    cv2.polylines(img, [pts], True, 20, thickness=int(1.4 * k))
    cv2.fillPoly(mask, [pts], 255)
    cv2.polylines(mask, [pts], True, 255, thickness=int(1.4 * k))
    img = cv2.resize(img, (24, 24), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (24, 24), interpolation=cv2.INTER_AREA)
    return img, (mask > 100).astype(np.uint8)


def frames(src, limit=None):
    lim = ["-t", str(limit)] if limit else []
    p = subprocess.Popen([FFMPEG, "-v", "error", *lim, "-i", str(src), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         stdout=subprocess.PIPE, bufsize=W * H * 4)
    while True:
        b = p.stdout.read(W * H)
        if len(b) < W * H:
            return
        yield np.frombuffer(b, np.uint8).reshape(H, W)


def main(src, out, limit=None):
    tpl, mask = template()
    tpl_p, mask_p = template(0.86)
    downs = []
    track = []
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    hist = []                      # (x, y) per frame, None when lost
    prev = []                      # last few frames' ROIs for the ring test
    clicks, last_click = [], -1e9
    pos = None
    for i, f in enumerate(frames(src, limit)):
        # track: search near the last position, the whole frame when lost
        if pos is not None:
            x0, y0 = max(0, pos[0] - 160), max(0, pos[1] - 160)
            roi = f[y0:y0 + 320, x0:x0 + 320]
        else:
            x0, y0, roi = 0, 0, f
        found = None
        if roi.shape[0] >= 24 and roi.shape[1] >= 24:
            r = cv2.matchTemplate(roi, tpl, cv2.TM_CCORR_NORMED, mask=mask)
            _, score, _, loc = cv2.minMaxLoc(r)
            if score > 0.9:
                found = (x0 + loc[0] + 3, y0 + loc[1] + 2)
        if found is None and pos is not None:
            r = cv2.matchTemplate(f, tpl, cv2.TM_CCORR_NORMED, mask=mask)
            _, score, _, loc = cv2.minMaxLoc(r)
            if score > 0.9:
                found = (loc[0] + 3, loc[1] + 2)
        pos = found
        hist.append(found)
        if i % 3 == 0 and found is not None:
            track.append([round(i / FPS, 3), int(found[0]), int(found[1])])
        # the press: the arrow shrinks to 0.86 while the button is down
        pressed = False
        if found is not None:
            cx, cy = found
            x1, y1 = max(0, cx - 3 - 10), max(0, cy - 2 - 10)
            win = f[y1:y1 + 44, x1:x1 + 44]
            if win.shape[0] >= 24 and win.shape[1] >= 24:
                sn = cv2.minMaxLoc(cv2.matchTemplate(win, tpl, cv2.TM_CCORR_NORMED, mask=mask))[1]
                sp = cv2.minMaxLoc(cv2.matchTemplate(win, tpl_p, cv2.TM_CCORR_NORMED, mask=mask_p))[1]
                pressed = sp > sn + 0.002
        downs.append(pressed)
        if pressed and not any(downs[-6:-1]) and i - last_click > 12:
            clicks.append({"t": round(i / FPS, 3), "x": int(found[0]), "y": int(found[1])})
            last_click = i
        prev.append(f)
        if len(prev) > 4:
            prev.pop(0)
        if i % 3000 == 0:
            print(f"{i / FPS:6.1f}s  clicks {len(clicks)}", flush=True)
    Path(out).write_text(json.dumps(clicks, indent=0))
    Path(out).with_name(Path(out).stem + "-track.json").write_text(json.dumps(track))
    print(f"{len(clicks)} clicks -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None)
