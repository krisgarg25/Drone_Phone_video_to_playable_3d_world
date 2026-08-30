"""Compose blind A/B comparison images for the critic.

For each eval timestamp we have:
  - the REAL frame (full-res, from frames_full/)
  - our RENDER (splat rendered from the matching camera pose)

Output per pair: results/pair_XX.jpg = vertical stack [A over B] where the order
is randomized per pair and NOT recorded in the image. The true mapping goes to
results/pair_key.json (kept away from the critic).

Usage:
  python make_pairs.py --real-dir work/rocks/frames_full \
      --render-dir work/rocks/eval_renders --pairs work/rocks/eval_pairs.json \
      --out results --tag rocks
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_resize(p: Path, width: int) -> np.ndarray:
    im = Image.open(p).convert("RGB")
    h = round(im.height * width / im.width)
    return np.asarray(im.resize((width, h), Image.LANCZOS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", required=True, type=Path)
    ap.add_argument("--render-dir", required=True, type=Path)
    ap.add_argument("--pairs", required=True, type=Path,
                    help="eval_pairs.json: [{t_sec, real_file, render_file}]")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", default="pair")
    ap.add_argument("--width", type=int, default=900)
    args = ap.parse_args()
    (args.out / "blinded").mkdir(parents=True, exist_ok=True)

    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
    rng = random.Random(1234)
    key = []
    for i, p in enumerate(pairs):
        real = load_resize(args.real_dir / p["real_file"], args.width)
        rend = load_resize(args.render_dir / p["render_file"], args.width)
        h = max(real.shape[0], rend.shape[0])
        a_first = rng.random() < 0.5
        top, bot = (real, rend) if a_first else (rend, real)
        canvas = np.full((h * 2 + 6, args.width, 3), 24, dtype=np.uint8)
        canvas[:top.shape[0]] = top
        canvas[h + 6:h + 6 + bot.shape[0]] = bot
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        # neutral position markers only — no hint which is which
        d.rectangle([0, h, args.width, h + 6], fill=(200, 200, 200))
        out_name = f"{args.tag}_AB_{i:02d}.jpg"
        im.save(args.out / "blinded" / out_name, quality=92)
        key.append({"pair": out_name, "A": "REAL" if a_first else "RENDER",
                    "B": "RENDER" if a_first else "REAL", **p})
    (args.out / "pair_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"[pairs] wrote {len(key)} blinded A/B stacks to {args.out/'blinded'}")


if __name__ == "__main__":
    main()
