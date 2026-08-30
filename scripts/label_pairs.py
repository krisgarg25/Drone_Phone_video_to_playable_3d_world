"""Unblind results/blinded/*.jpg using pair_key.json -> labeled side-by-sides in
results/side_by_side/. TOP/BOTTOM panels get their true identity stamped on them."""
import json
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
key = json.loads((ROOT / "results" / "pair_key.json").read_text())
out = ROOT / "results" / "side_by_side"
out.mkdir(parents=True, exist_ok=True)

for k in sorted(key, key=lambda d: d["pair"]):
    pk = Path(k["pair"]).stem
    src = ROOT / "results" / "blinded" / k["pair"]
    img = cv2.imread(str(src))
    H = img.shape[0] // 2
    top_lab = "REAL DRONE FRAME" if k["A"] == "REAL" else "SPLAT RENDER (gsplat)"
    bot_lab = "REAL DRONE FRAME" if k["B"] == "REAL" else "SPLAT RENDER (gsplat)"
    for y0, lab in ((0, top_lab), (H, bot_lab)):
        bar_h = 34
        cv2.rectangle(img, (0, y0), (img.shape[1], y0 + bar_h), (16, 16, 16), -1)
        color = (80, 220, 90) if lab.startswith("REAL") else (90, 170, 255)
        cv2.putText(img, lab, (12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    color, 2, cv2.LINE_AA)
    # separator line
    cv2.line(img, (0, H), (img.shape[1], H), (255, 255, 255), 2)
    cv2.imwrite(str(out / f"{pk}_labeled.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[label] {pk}: A={k['A']} B={k['B']} -> {pk}_labeled.jpg")
print("[label] done")
