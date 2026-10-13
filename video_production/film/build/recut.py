"""Recut the finished render: shorter, with a sound-effects track.

    .venv/Scripts/python.exe video_production/film/build/recut.py [--draft] [--audio-only]

The raw footage behind build.py is gone, so this edits renders/Final Render.mp4 itself. Every
decision is in build/recut_edl.json, in the ORIGINAL render's seconds:

* remove: [start, end, why]            time taken out (a whole chapter, or the inside of a still hold)
* speed:  [start, end, 2, what]        played at 2x (every other frame dropped)
* joins:  [{t1, t2, xfade}]            t1..t2 is removed and the two sides cross-fade
* cues:   {t, sfx, gain_db, what}      a sound from assets/sfx/<sfx>.wav starting at t; optional
                                       rate (0.6 = slower and deeper), len (seconds, faded), pan
* music:  {file, gain_db}              optional bed, looped and faded to the film's length
* renumber: true                       repaint the chapter numbers for the cut (see renumber.py)

Cuts are frame-exact on the 60 fps grid. Cue times are mapped through the cuts, so a cue stays on
its picture however the cut list changes. Edit the EDL and run again.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

sys.path.insert(0, str(Path(__file__).resolve().parent))
import renumber  # noqa: E402

FILM = Path(__file__).resolve().parents[1]
ROOT = FILM.parents[1]
FFMPEG = str(ROOT / "tools/ffmpeg/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe")
FFPROBE = FFMPEG.replace("ffmpeg.exe", "ffprobe.exe")
EDL = FILM / "build" / "recut_edl.json"
SFX = FILM / "assets" / "sfx"
OUT = FILM / "renders"
FPS = 60
SR = 48000


def nf(t):
    return int(round(t * FPS))


def _run(args):
    subprocess.run([FFMPEG, "-hide_banner", "-v", "error", "-y", *args], check=True)


# ------------------------------------------------------------------ the cut, per frame
def frame_plan(edl, n_src):
    """keep[n]: is source frame n in the film; and the joins as (f1, f2, cross-fade frames)."""
    keep = np.ones(n_src, bool)
    for a, b, *_ in edl["remove"]:
        keep[nf(a):nf(b)] = False
    for a, b, factor, *_ in edl["speed"]:
        assert factor == 2, "only 2x drops frames cleanly at 60 fps"
        fa, fb = nf(a), nf(b)
        keep[fa + 1:fb:2] = False
    joins = sorted((nf(j["t1"]), nf(j["t2"]), nf(j["xfade"])) for j in edl.get("joins", []))
    for f1, f2, xf in joins:
        keep[f1:f2] = False
        keep[f1 - xf:f1] = keep[f2:f2 + xf] = True       # the frames that cross-fade are all there
    return keep, joins


class TimeMap:
    """Original seconds -> seconds in the recut."""

    def __init__(self, keep, joins):
        self.keep, self.joins = keep, joins
        self.cum = np.concatenate([[0], np.cumsum(keep)])

    def __call__(self, t):
        f = min(max(nf(t), 0), len(self.keep))
        out = int(self.cum[f])
        out -= sum(xf for f1, f2, xf in self.joins if f >= f2)   # side B starts under the cross-fade
        return out / FPS

    def dropped(self, t):
        return any(f1 <= nf(t) < f2 for f1, f2, _ in self.joins)

    @property
    def frames(self):
        return int(self.cum[-1]) - sum(xf for *_, xf in self.joins)


# ------------------------------------------------------------------ sound
_cache = {}


def sound(name):
    if name not in _cache:
        sr, x = wavfile.read(SFX / f"{name}.wav")
        assert sr == SR, f"{name}.wav is {sr} Hz, expected {SR}"
        x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
        _cache[name] = np.stack([x, x], 1) if x.ndim == 1 else x[:, :2]
    return _cache[name]


def decode(path):
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2)


def music_bed(spec, n):
    """The track looped (2 s cross-fades) to n samples, faded in and out."""
    x = decode(FILM / spec["file"])
    lo, hi = int(spec.get("from", 0) * SR), int(spec.get("to", len(x) / SR) * SR)
    x = x[lo:hi]
    xf = 2 * SR
    bed = x.copy()
    while len(bed) < n:
        ramp = np.linspace(0, 1, xf, dtype=np.float32)[:, None]
        bed = np.concatenate([bed[:-xf], bed[-xf:] * (1 - ramp) + x[:xf] * ramp, x[xf:]])
    bed = bed[:n] * 10 ** (spec.get("gain_db", -22) / 20)
    fi, fo = int(1.5 * SR), int(spec.get("fade_out", 4.0) * SR)
    bed[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]
    bed[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)[:, None]
    return bed


def mix(edl, tm):
    n = int(tm.frames / FPS * SR)
    sfx = np.zeros((n, 2), np.float32)
    placed, last = [], {}
    for c in sorted(edl["cues"], key=lambda c: c["t"]):
        if tm.dropped(c["t"]):
            continue
        t = tm(c["t"]) + c.get("nudge", 0)
        if t - last.get(c["sfx"], -9) < 0.12:      # the cut pushed two of the same sound together
            continue
        last[c["sfx"]] = t
        s = sound(c["sfx"])
        if c.get("rate", 1) != 1:                  # the kit is small: one whoosh, re-pitched, does several jobs
            s = resample(s, int(len(s) / c["rate"])).astype(np.float32)
        if c.get("len"):                           # a long boom cut down to a thud
            s = s[:int(c["len"] * SR)].copy()
            k = int(len(s) * 0.6)
            s[-k:] *= np.linspace(1, 0, k, dtype=np.float32)[:, None] ** 2
        s = s * 10 ** ((c.get("gain_db", -10) + edl.get("master_gain_db", 0)) / 20)
        pan = c.get("pan", 0.0)                    # -1 left .. 1 right, equal power
        s = s * np.array([np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)], np.float32) * 2 ** 0.5
        i = int(t * SR)
        s = s[:max(0, n - i)]
        sfx[i:i + len(s)] += s
        placed.append((t, c["sfx"], c.get("gain_db", -10), c.get("what", "")))
    out = sfx
    if edl.get("music"):
        out = out + music_bed(edl["music"], n)
    lim = 10 ** (-1 / 20)                          # soft ceiling at -1 dBFS
    over = np.abs(out) > 0.7
    out[over] = np.sign(out[over]) * (0.7 + (lim - 0.7) * np.tanh((np.abs(out[over]) - 0.7) / (lim - 0.7)))
    return out, placed


# ------------------------------------------------------------------ build
def main():
    draft, audio_only = "--draft" in sys.argv, "--audio-only" in sys.argv
    edl = json.loads(EDL.read_text(encoding="utf8"))
    src = FILM / edl["source"]
    n_src = int(subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
                                "-of", "csv=p=0", str(src)], check=True, capture_output=True, text=True).stdout.strip().strip(","))
    keep, joins = frame_plan(edl, n_src)
    tm = TimeMap(keep, joins)
    print(f"source {n_src / FPS:.2f} s -> recut {tm.frames / FPS:.2f} s "
          f"({int(tm.frames / FPS // 60)}:{tm.frames / FPS % 60:04.1f})")

    OUT.mkdir(exist_ok=True)
    audio, placed = mix(edl, tm)
    wav = OUT / "recut-audio.wav"
    wavfile.write(wav, SR, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    (OUT / "recut-cues.txt").write_text(
        "\n".join(f"{int(t // 60)}:{t % 60:06.3f}  {g:+5.1f} dB  {n:<18} {w}" for t, n, g, w in placed), encoding="utf8")
    print(f"{len(placed)} cues -> {wav.name}")
    if audio_only:
        return

    # frames go through Python: the cut, the cross-fade and the renumbering in one pass, one encode
    out = OUT / ("recut-draft.mp4" if draft else edl.get("output", "Final Render - short.mp4"))
    enc = ["-c:v", "libx264", "-preset", "medium", "-crf", "17"]
    if draft:                                      # half size, fast: for checking the cut and the sound
        enc = ["-vf", "scale=960:540", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26"]
    fix = renumber.Renumber(FFMPEG, src) if edl.get("renumber") else None
    size = renumber.W * renumber.H * 3 // 2
    dec = subprocess.Popen([FFMPEG, "-v", "error", "-i", str(src), "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"],
                           stdout=subprocess.PIPE)
    wr = subprocess.Popen([FFMPEG, "-hide_banner", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                           "-s", f"{renumber.W}x{renumber.H}", "-r", str(FPS), "-color_range", "tv", "-colorspace", "bt709",
                           "-i", "-", "-i", str(wav), "-map", "0:v", "-map", "1:a", *enc, "-pix_fmt", "yuv420p",
                           "-colorspace", "bt709", "-color_range", "tv", "-c:a", "aac", "-b:a", "256k",
                           "-movflags", "+faststart", "-shortest", str(out)], stdin=subprocess.PIPE)
    tails = {f2: [] for _, f2, _ in joins}         # side A's last frames wait here for side B's first
    for n in range(n_src):
        raw = dec.stdout.read(size)
        if len(raw) < size:
            break
        if not keep[n]:
            continue
        buf = np.frombuffer(raw, np.uint8)
        if fix:
            buf = buf.copy()
            fix.apply(n, buf)
        ja = next((j for j in joins if j[0] - j[2] <= n < j[0]), None)
        jb = next((j for j in joins if j[1] <= n < j[1] + j[2]), None)
        if ja:
            tails[ja[1]].append(buf.astype(np.float32))
        elif jb:
            k = n - jb[1]
            w = (k + 1) / (jb[2] + 1)
            wr.stdin.write((tails[jb[1]][k] * (1 - w) + buf * w + 0.5).astype(np.uint8).tobytes())
        else:
            wr.stdin.write(buf.tobytes())
    dec.stdout.close()
    wr.stdin.close()
    if wr.wait():
        raise SystemExit("encode failed")
    print("wrote", out)


if __name__ == "__main__":
    main()
