"""Compose blind A/B comparison images for the critic.

For each eval timestamp we have:
  - the REAL frame (full-res, from frames_full/)
  - our RENDER (splat rendered from the matching camera pose)

Output per pair: results/pair_XX.jpg = vertical stack [A over B] where the order
is randomized per pair and NOT recorded in the image. The true mapping goes to
results/pair_key_<tag>.json (kept away from the critic).

Usage:
  python make_pairs.py --real-dir work/rocks/frames_full \
      --render-dir work/rocks/eval_renders --pairs work/rocks/eval_pairs.json \
      --out results --tag rocks
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402

# How many refused names one composite walks past before it is given up on. Bounded
# so that a directory refusing everything (disk full, read-only) costs N dropped
# pairs instead of N * slot_attempts writes.
MAX_NAME_TRIES = 4


def load_resize(p: Path, width: int):
    """None instead of raising: one unrendered camera must not cost every pair."""
    try:
        im = Image.open(p).convert("RGB")
    except (OSError, ValueError) as e:
        rb.warn(f"{p.name}: {type(e).__name__}: {e} - pair skipped")
        return None
    if im.width <= 0 or im.height <= 0:
        rb.warn(f"{p.name}: {im.width}x{im.height} is not an image - pair skipped")
        return None
    h = max(1, round(im.height * width / im.width))
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
    # Per tag, like the stacks themselves. A single shared key made the next take's
    # run overwrite the previous one's mapping, leaving every earlier stack on disk
    # with nothing to unblind it.
    key_path = args.out / f"pair_key_{args.tag}.json"

    pairs = rb.read_json(args.pairs, None)
    if not isinstance(pairs, list):
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{args.pairs} is missing or is not a JSON list - the eval step writes "
            "it, and it lists which real frame pairs with which render.",
            returncode=3)
    rng = random.Random(1234)
    key, skipped, dropped = [], 0, 0
    slot, no_room = 0, False
    for i, p in enumerate(pairs):
        if not isinstance(p, dict) or "real_file" not in p or "render_file" not in p:
            rb.warn(f"pair {i}: no real_file/render_file key - skipped")
            skipped += 1
            continue
        real = load_resize(args.real_dir / p["real_file"], args.width)
        rend = load_resize(args.render_dir / p["render_file"], args.width)
        if real is None or rend is None:
            skipped += 1
            continue
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
        out_name = None
        # A name the OS will not hand over must cost the name, not the rest of the
        # run: names come from a cursor that only moves forward, so a jpg some
        # viewer is holding open gets stepped over. Once a pair has walked past
        # every name it was allowed the directory itself is the problem, so the
        # rest get a single probe each instead of a wall of warnings.
        for _ in range(1 if no_room else MAX_NAME_TRIES):
            name = f"{args.tag}_AB_{slot:02d}.jpg"
            slot += 1
            if rb.save_image(im, args.out / "blinded" / name, quality=92):
                out_name = name
                no_room = False
                break
            rb.warn(f"{name}: could not be written - stepping over the name")
        if out_name is None:
            no_room = True
            rb.warn(f"pair {i}: no slot accepted the write - pair dropped, the "
                    "rest of the stacks still ship")
            dropped += 1
            continue
        key.append({"pair": out_name, "A": "REAL" if a_first else "RENDER",
                    "B": "RENDER" if a_first else "REAL", **p})
    if not key:
        # Exit 0: the world is already built and this is the evidence step. But a
        # run that produced no evidence must not look like it produced some, and
        # must not replace last run's key with an empty one while its stacks are
        # still on disk to be read as if they were scored.
        rb.warn(f"no blinded pairs written ({len(pairs)} requested, {skipped} "
                f"unreadable, {dropped} unwritable) - leaving "
                f"{key_path} as it was")
        return
    rb.write_json(key_path, key)
    print(f"[pairs] wrote {len(key)} blinded A/B stacks to {args.out / 'blinded'}"
          + (f" ({skipped} skipped)" if skipped else "")
          + (f", {dropped} unwritable" if dropped else ""))


if __name__ == "__main__":
    rb.configure_streams()
    try:
        main()
    except rb.StepError as e:
        print(f"\n[pairs] {e}", file=sys.stderr, flush=True)
        sys.exit(e.returncode)
