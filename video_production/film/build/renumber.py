"""Renumber the chapters in the finished render after Twin (05), Walk (07) and Export (08) are cut.

The chapter numbers are baked into the picture, so they are repainted from the picture itself:

* slates: every slate plays the same animation, so a slate takes its numeral and "Chapter 0N"
  from the slate that used to carry that number, at the same frame of the move (Mission takes
  Twin's 05, Plan takes Mission's 06, Place takes Walk's 07). Every slate's "/ 10" becomes
  "/ 07" with the digits of Walk's "Chapter 07".
* chips ("09 Plan" + dots, bottom right) and the break headers ("09 · Plan"): the digits are
  swapped as layers, old out and new in, at the opacity the word beside them has in that frame.

Frames are yuv420p planes in one flat uint8 buffer. All boxes sit on even pixels.
"""
import subprocess

import numpy as np

W, H, FPS = 1920, 1080, 60
SPAN = 276                                   # frames of a slate, from its start to its text gone
SRC = {"twin": 11302, "mission": 12061, "walk": 15302, "export": 16014}   # first frame of the slates numerals come from
SLATES = {1422: None, 3265: None, 5355: None, 8020: None,                  # Explore .. Operations: only the total changes
          12061: "twin", 16932: "mission", 19839: "walk"}                  # Mission, Plan, Place
TOTAL = "walk"                                                             # whose "07" becomes every slate's total
NUM = (236, 432, 94, 370)                    # y0, y1, x0, x1: the big numeral
EYE = (462, 510, 268, 314)                   # the chapter digits in "Chapter 07 / 10"
TOT_DX = 76                                  # ...and the "10" sits five monospace characters right
BAND = (432, 466)                            # rows with no text: where the app window's edge is read


def planes(buf):
    y = buf[:H * W].reshape(H, W)
    u = buf[H * W:H * W * 5 // 4].reshape(H // 2, W // 2)
    v = buf[H * W * 5 // 4:].reshape(H // 2, W // 2)
    return y, u, v


def read_frames(ffmpeg, src, f0, n):
    """Frames f0 .. f0+n-1 of the source as writable flat yuv420p buffers."""
    p = subprocess.Popen([ffmpeg, "-v", "error", "-ss", f"{(f0 - 0.5) / FPS:.5f}", "-i", str(src), "-frames:v", str(n),
                          "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    size = W * H * 3 // 2
    for _ in range(n):
        raw = p.stdout.read(size)
        if len(raw) < size:
            break
        yield np.frombuffer(raw, np.uint8).copy()
    p.stdout.close()
    p.wait()


def crop(buf, box):
    y0, y1, x0, x1 = box
    y, u, v = planes(buf)
    return y[y0:y1, x0:x1].copy(), u[y0 // 2:y1 // 2, x0 // 2:x1 // 2].copy(), v[y0 // 2:y1 // 2, x0 // 2:x1 // 2].copy()


def paste(buf, box, src, x_lim):
    """Copy a crop into the frame, left of column x_lim only (the app window is right of it)."""
    y0, y1, x0, x1 = box
    w = max(0, min(x1, x_lim) - x0) // 2 * 2
    if w <= 0:
        return
    y, u, v = planes(buf)
    y[y0:y1, x0:x0 + w] = src[0][:, :w]
    u[y0 // 2:y1 // 2, x0 // 2:(x0 + w) // 2] = src[1][:, :w // 2]
    v[y0 // 2:y1 // 2, x0 // 2:(x0 + w) // 2] = src[2][:, :w // 2]


def get444(buf, box):
    """A box as float Y, U, V at luma resolution (so a layer can move by an odd number of pixels)."""
    y0, y1, x0, x1 = box
    y, u, v = planes(buf)
    ix = np.ix_(np.arange(y0, y1) // 2, np.arange(x0, x1) // 2)
    return np.stack([y[y0:y1, x0:x1], u[ix], v[ix]]).astype(np.float32)


def put444(buf, box, a):
    y0, y1, x0, x1 = box
    y, u, v = planes(buf)
    a = np.clip(a + 0.5, 0, 255)
    y[y0:y1, x0:x1] = a[0]
    for p, c in ((u, a[1]), (v, a[2])):
        p[y0 // 2:y1 // 2, x0 // 2:x1 // 2] = c.reshape((y1 - y0) // 2, 2, (x1 - x0) // 2, 2).mean((1, 3))


def layer(a):
    """What is drawn in a box, as the difference from its flat background (the border's median)."""
    edge = np.concatenate([a[:, 0], a[:, -1], a[:, :, 0], a[:, :, -1]], axis=1)
    return a - np.median(edge, axis=1)[:, None, None]


class Swap:
    """Swap a static label's digits, following the label's own fades: `word` is a box of text
    beside the digits that stays the same, and its strength in a frame is the label's opacity."""

    def __init__(self, first, last, word, pairs):
        self.first, self.last, self.word, self.pairs = first, last, word, pairs   # pairs: (box, old layer, new layer)

    def ready(self, ref):
        self.w = layer(get444(ref, self.word))[0]
        self.ww = float((self.w * self.w).sum())

    def apply(self, n, buf):
        if not self.first <= n < self.last:
            return
        f = layer(get444(buf, self.word))[0]
        o = float((f * self.w).sum()) / self.ww
        like = float((f * self.w).sum()) / (np.sqrt(float((f * f).sum()) * self.ww) + 1e-6)
        if o < 0.03 or like < 0.6:                # the label is not on screen (a break covers it, or it has faded)
            return
        o = min(o, 1.0)
        for box, old, new in self.pairs:
            put444(buf, box, get444(buf, box) + o * (new - old))


class Renumber:
    def __init__(self, ffmpeg, src):
        one = lambda f: next(read_frames(ffmpeg, src, f, 1))
        # ---- slates: the source slates' numerals, frame by frame, and where the window's edge is
        self.src = {k: [(crop(f, NUM), crop(f, EYE)) for f in read_frames(ffmpeg, src, f0, SPAN)] for k, f0 in SRC.items()}
        self.x_lim = []
        for a, b, c in zip(read_frames(ffmpeg, src, SRC["walk"], SPAN), read_frames(ffmpeg, src, SRC["export"], SPAN),
                           read_frames(ffmpeg, src, 16932, SPAN)):
            ya, yb, yc = (planes(x)[0][BAND[0]:BAND[1], :1100].astype(np.int16) for x in (a, b, c))
            # the three slates share one background; they differ only inside the window (different footage)
            d = np.maximum(np.abs(ya - yb), np.abs(ya - yc)).max(0) > 14
            hit = np.flatnonzero(d)
            self.x_lim.append(int(hit[0]) - 8 if len(hit) else 1100)
        # ---- chips and the break headers: still frames to lift the digits from
        twin_chip, mission_chip, walk_chip = one(11760), one(12900), one(15720)
        plan_chip, place_chip = one(18000), one(20220)
        CH = (998, 1030)                          # the chip's rows
        dots = (*CH, 1610, 1782)

        def chip(first, last, ref, src, x0, dx, word):
            box = (*CH, x0, x0 + 34)
            s = Swap(first, last, (*CH, *word), [
                (box, layer(get444(ref, box)), layer(get444(src, (*CH, x0 - dx, x0 - dx + 34)))),
                (dots, layer(get444(ref, dots)), layer(get444(src, dots)))])
            s.ready(ref)
            return s

        hb = (70, 100, 152, 188)                  # the digits in a break header; new ones come from a slate's eyebrow

        def header(first, last, ref, eye, word):
            dy, dx = self._offset(ref, hb, eye, (466, 506, 270, 306))
            s = Swap(first, last, (70, 100, *word), [
                (hb, layer(get444(ref, hb)), layer(get444(eye, (hb[0] + dy, hb[1] + dy, hb[2] + dx, hb[3] + dx))))])
            s.ready(ref)
            return s

        self.swaps = [chip(12270, 15340, mission_chip, twin_chip, 1450, -37, (1492, 1592)),   # 06 Mission -> 05
                      chip(17130, 19900, plan_chip, mission_chip, 1490, 40, (1532, 1592)),    # 09 Plan -> 06
                      chip(20010, 20381, place_chip, walk_chip, 1474, -7, (1518, 1592)),      # 10 Place -> 07
                      header(13896, 15310, one(14700), one(11450), (228, 336)),              # 06 · Mission -> 05
                      header(18435, 19850, one(18900), one(12209), (228, 292))]              # 09 · Plan -> 06

    @staticmethod
    def _offset(a, box_a, b, box_b):
        """How far the first digit in box_b of frame b sits from the first digit in box_a of frame a."""
        def corner(buf, box):
            g = layer(get444(buf, box))[0] > 60
            return box[0] + int(np.flatnonzero(g.any(1))[0]), box[2] + int(np.flatnonzero(g.any(0))[0])
        (ya, xa), (yb, xb) = corner(a, box_a), corner(b, box_b)
        return yb - ya, xb - xa

    def apply(self, n, buf):
        for start, kind in SLATES.items():
            p = n - start
            if 0 <= p < SPAN:
                lim = self.x_lim[p]
                if kind:
                    num, eye = self.src[kind][p]
                    paste(buf, NUM, num, lim)
                    paste(buf, EYE, eye, lim)
                tot = (EYE[0], EYE[1], EYE[2] + TOT_DX, EYE[3] + TOT_DX)
                paste(buf, tot, self.src[TOTAL][p][1], lim)
        for s in self.swaps:
            s.apply(n, buf)
