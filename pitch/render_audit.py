"""Render each slide of the deck at 1920x1080 and audit overflow + dead space."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

DECK = Path(sys.argv[1]).resolve()
OUT = DECK.parent / "render"
OUT.mkdir(exist_ok=True)

AUDIT = r"""
() => {
  const S = document.querySelector('.slide.active');
  let maxBottom = 0, maxRight = 0, minLeft = 1920, minTop = 1080;
  const offenders = [];
  const st = document.getElementById('deckStage').getBoundingClientRect();
  const f = st.width / 1920 || 1;
  const toStage = r => ({x:(r.left-st.left)/f, y:(r.top-st.top)/f,
                         w:r.width/f, h:r.height/f});
  const foot = S.querySelector('.foot'), head = S.querySelector('.head');
  const footTop = foot ? toStage(foot.getBoundingClientRect()).y : 1080;
  const headBot = head ? toStage(head.getBoundingClientRect()).y +
                         toStage(head.getBoundingClientRect()).h : 0;
  for (const el of S.querySelectorAll('*')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || +cs.opacity === 0) continue;
    if (el === foot || el === head || foot?.contains(el) || head?.contains(el)) continue;
    if (el.closest('.pagenum, .deck-controls')) continue;
    const r = el.getBoundingClientRect();
    if (!r.width && !r.height) continue;
    const {x, y, w, h} = toStage(r);
    if (el.children.length === 0 && (el.textContent||'').trim()) {
      maxBottom = Math.max(maxBottom, y+h);
      maxRight  = Math.max(maxRight,  x+w);
      minLeft   = Math.min(minLeft,   x);
      minTop    = Math.min(minTop,    y);
      const outOfStage = y+h > 1080 || x+w > 1920 || x < 0 || y < 0;
      const hitsFoot = y+h > footTop + 1;
      const hitsHead = y < headBot - 1;
      if (outOfStage || hitsFoot || hitsHead) {
        offenders.push({tag:el.tagName, txt:(el.textContent||'').trim().slice(0,44),
                        x:Math.round(x), y:Math.round(y), r:Math.round(x+w), b:Math.round(y+h),
                        why: outOfStage ? 'stage' : hitsFoot ? 'footer' : 'header'});
      }
    }
  }
  const STRUCT = '.body, .foot, .head, .pagenum';
  let maxBox = 0;
  for (const el of S.querySelectorAll('*')) {
    if (el.matches(STRUCT)) continue;
    if (el.closest('.pagenum, .deck-controls, .bloom, .ember, .tiles, .railwrap')) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || +cs.opacity === 0) continue;
    const bg = cs.backgroundColor, hasFill = bg && bg !== 'rgba(0, 0, 0, 0)';
    const hasBorder = parseFloat(cs.borderTopWidth) > 0 || parseFloat(cs.borderBottomWidth) > 0;
    if (!hasFill && !hasBorder) continue;
    const {y, h} = toStage(el.getBoundingClientRect());
    if (!h) continue;
    maxBox = Math.max(maxBox, y + h);
    if (y + h > footTop + 1)
      offenders.push({tag:el.tagName, cls:el.className.toString().slice(0,22), txt:'[box]',
                      x:0, y:Math.round(y), r:0, b:Math.round(y+h), why:'footer-box'});
  }
  return {maxBottom:Math.round(maxBottom), maxRight:Math.round(maxRight),
          minLeft:Math.round(minLeft), minTop:Math.round(minTop),
          maxBox:Math.round(maxBox),
          footTop:Math.round(footTop), headBot:Math.round(headBot),
          deadBand:1080-Math.round(maxBottom), offenders};
}
"""


def main():
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True, args=["--force-device-scale-factor=1"])
        ctx = b.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        pg = ctx.new_page()
        pg.goto(DECK.as_uri(), wait_until="networkidle")
        n = pg.evaluate("() => document.querySelectorAll('.slide').length")
        print(f"{n} slides\n")
        for i in range(n):
            pg.evaluate(f"() => document.querySelectorAll('.slide')[{i}].click()")
            pg.evaluate(f"""() => {{
              const s=document.querySelectorAll('.slide');
              s.forEach((e,k)=>{{e.classList.toggle('active',k==={i});e.classList.toggle('visible',k==={i});}});
            }}""")
            pg.wait_for_timeout(1400)
            a = pg.evaluate(AUDIT)
            img = OUT / f"{i+1:02d}.png"
            pg.screenshot(path=str(img), type="jpeg", quality=88)
            clear = a["footTop"] - max(a["maxBottom"], a["maxBox"])
            flag = ""
            if clear > 120: flag += " DEAD-BAND"
            if a["offenders"]: flag += f" COLLIDE x{len(a['offenders'])}"
            print(f"{i+1:>2}  text={a['maxBottom']:>4}  box={a['maxBox']:>4}  footTop={a['footTop']:>4}  "
                  f"clear={clear:>4}{flag}")
            for o in a["offenders"][:5]:
                print(f"      ! {o['why']:<6} {o['tag']} {o['txt']!r} "
                      f"box=({o['x']},{o['y']})-({o['r']},{o['b']})")
        b.close()


if __name__ == "__main__":
    main()
