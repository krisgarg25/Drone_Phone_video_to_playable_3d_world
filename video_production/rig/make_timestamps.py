"""Build video_production/TIMESTAMPS.md from the take log the recorder writes.

    python video_production/rig/make_timestamps.py [take-name]

Times are on the video's own clock (0:00 = first frame), read from the ffmpeg capture
clock, so they line up with the MP4 to within a frame or two.
"""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
RAW = OUT / "raw"


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "demo"
    log = json.loads((RAW / f"{name}.json").read_text())
    if not log.get("recorded") or log.get("failed"):
        raise SystemExit(f"{name}.json is not a complete recorded take")
    lines = ["# Demo video timestamps", "",
             f"One continuous take: `raw/{name}.mp4`, {mmss(log['length'])} long, 1920x1080 at 60 fps. "
             "Times are on the video's own clock. The take starts with about 1 s of stillness and ends "
             "with 1.5 s, as handles.", "",
             "## Chapters", "", "| # | Chapter | Start | End | Length |", "|---|---|---|---|---|"]
    for i, c in enumerate(log["chapters"], 1):
        lines.append(f"| {i} | {c['title']} | {mmss(c['start'])} | {mmss(c['end'])} | {c['end'] - c['start']:.1f} s |")
    for c in log["chapters"]:
        rows = [e for e in log["events"] if e["chapter"] == c["title"]]
        lines += ["", f"## {c['title']}  ({mmss(c['start'])} – {mmss(c['end'])})", "",
                  "| # | Action | Start | End | Duration |", "|---|---|---|---|---|"]
        for i, e in enumerate(rows, 1):
            lines.append(f"| {i} | {e['action']} | {mmss(e['start'])} | {mmss(e['end'])} | {e['end'] - e['start']:.1f} s |")
    (OUT / "TIMESTAMPS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"TIMESTAMPS.md: {len(log['chapters'])} chapters, {len(log['events'])} actions, {mmss(log['length'])}")


if __name__ == "__main__":
    main()
