"""After-action review as a PDF (RH-6 / MIL-12): what the on-screen replay shows, on paper.

Page 1: the run's numbers (time, distance, exposure, hits, objective), exposure per
enemy post and the event log. Page 2: a top-down map - the scan's hillshade, the planned
route, the track walked coloured by how many enemies could see the player at that
moment, the enemy posts, and where each hit and casualty happened. Page 3: exposure over
time (who was watching when). Every page states that the terrain is reconstructed and
the enemy positions are the plan's.
"""
import io
from datetime import datetime, timezone

import numpy as np

NOTE = ("Rehearsal on reconstructed terrain. Enemy positions and behaviours are the plan's, "
        "not observations; unscanned ground was not assumed clear.")


def _exposure(record):
    p = np.asarray(record["player"], dtype=np.float64)
    t = p[:, 0]
    count = np.zeros(len(p), dtype=int)
    rows = []
    for b in record["bots"]:
        f = np.asarray(b["frames"], dtype=np.float64).reshape(-1, 7)
        if not len(f):
            continue
        sees = np.interp(t, f[:, 0], f[:, 5]) >= 0.5
        count += sees
        rows.append((b["label"] or f"Enemy {b['id']}", sees, f))
    return t, p, count, rows


def aar_pdf(record, *, mission_name, basemap_rgb=None, bounds=None):
    """PDF bytes. ``basemap_rgb`` (rows = +z) and ``bounds`` (x0, z0, x1, z1) are optional."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.collections import LineCollection

    s = record["summary"]
    t, p, count, bots = _exposure(record)
    footer = (f"{mission_name} · run {record['id']} · {record['created_at']} · conditions "
              f"{', '.join(f'{k} {v}' for k, v in record.get('conditions', {}).items()) or 'day'} · "
              f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, "After-action review", fontsize=17, weight="bold")
        fig.text(0.07, 0.925, mission_name, fontsize=11, color="#333")
        facts = [("Result", "objective reached" if s["completed"] else "not completed"),
                 ("Duration", f"{int(s['duration_s'] // 60)}:{int(s['duration_s'] % 60):02d}"),
                 ("Distance moved", f"{s['distance_m']:.0f} m"),
                 ("Time seen by the enemy", f"{s['exposure_s']:.0f} s ({s['exposed_pct']:.0f}%)"),
                 ("Hits taken / casualties", f"{s['hits_taken']} / {s['casualty']}"),
                 ("Enemies neutralised", f"{s['neutralised']} of {len(s['per_bot'])}"),
                 ("Waypoints reached", str(s["waypoints_reached"]))]
        y = 0.88
        for key, value in facts:
            fig.text(0.07, y, key, fontsize=10, color="#444")
            fig.text(0.45, y, value, fontsize=10, weight="bold")
            y -= 0.024
        y -= 0.02
        fig.text(0.07, y, "Exposure by enemy post", fontsize=11, weight="bold")
        rows = [[b["label"] or f"Enemy {b['id']}", f"{b['exposure_s']:.0f} s",
                 "never" if b["first_seen_s"] is None else f"{b['first_seen_s']:.0f} s",
                 "yes" if b["neutralised"] else "no"] for b in s["per_bot"]][:14]
        if rows:
            height = 0.021 * (len(rows) + 1)
            ax = fig.add_axes([0.07, y - 0.01 - height, 0.86, height])
            ax.axis("off")
            table = ax.table(cellText=rows, colLabels=["Enemy", "Saw you for", "First saw you at", "Neutralised"],
                             loc="upper left", cellLoc="left")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            y -= height + 0.03
        fig.text(0.07, y, "Events", fontsize=11, weight="bold")
        y -= 0.022
        for event in record["events"][:28]:
            fig.text(0.07, y, f"{int(event['t'] // 60)}:{int(event['t'] % 60):02d}", fontsize=8, family="monospace")
            fig.text(0.15, y, event["text"][:95] or event["type"], fontsize=8)
            y -= 0.017
            if y < 0.07:
                break
        fig.text(0.07, 0.035, NOTE, fontsize=7, color="#555")
        fig.text(0.07, 0.02, footer, fontsize=6.5, color="#777")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        if basemap_rgb is not None and bounds is not None:
            x0, z0, x1, z1 = bounds
            ax.imshow(basemap_rgb, extent=(x0, x1, z1, z0), origin="upper", interpolation="nearest")
        route = np.asarray(record.get("route") or [], dtype=float).reshape(-1, 2)
        if len(route) >= 2:
            ax.plot(route[:, 0], route[:, 1], "--", color="white", lw=1.6, label="planned route")
        xy = p[:, [1, 3]]
        jumps = np.linalg.norm(np.diff(xy, axis=0), axis=1) > 8.0
        segs = np.stack([xy[:-1], xy[1:]], axis=1)[~jumps]
        colours = np.array(["#3ec46d", "#f0a030", "#d63232"])[np.clip(count[:-1][~jumps], 0, 2)]
        ax.add_collection(LineCollection(segs, colors=colours, linewidths=2.6))
        for label, _, f in bots:
            ax.plot(f[0, 1], f[0, 3], "v", color="#d63232", ms=9, mec="black")
            ax.annotate(label, (f[0, 1], f[0, 3]), textcoords="offset points", xytext=(5, 5), fontsize=7, color="white")
        for event in record["events"]:
            if event.get("at") and event["type"] in ("hit_taken", "casualty"):
                ax.plot(event["at"][0], event["at"][2], "x", color="yellow" if event["type"] == "hit_taken" else "red", ms=9, mew=2)
        ax.plot(*xy[0], "o", color="#3ec46d", ms=8, mec="black")
        ax.plot(*xy[-1], "s", color="white", ms=7, mec="black")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m) - scene north is up")
        ax.set_title("Track: green unseen, amber seen by one, red by two or more; ▼ enemy posts; × hits")
        pad = 15.0
        ax.set_xlim(min(xy[:, 0].min(), *(route[:, 0] if len(route) else [xy[0, 0]])) - pad,
                    max(xy[:, 0].max(), *(route[:, 0] if len(route) else [xy[0, 0]])) + pad)
        ax.set_ylim(max(xy[:, 1].max(), *(route[:, 1] if len(route) else [xy[0, 1]])) + pad,
                    min(xy[:, 1].min(), *(route[:, 1] if len(route) else [xy[0, 1]])) - pad)
        fig.text(0.07, 0.02, footer, fontsize=6.5, color="#777")
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(len(bots) + 1, 1, figsize=(8.27, 11.69), sharex=True, squeeze=False)
        axes = axes[:, 0]
        axes[0].fill_between(t - t[0], count, step="post", color="#d63232", alpha=0.6)
        axes[0].set_ylabel("enemies\nseeing you", fontsize=8)
        for ax, (label, sees, _) in zip(axes[1:], bots):
            ax.fill_between(t - t[0], sees.astype(int), step="post", color="#f0a030", alpha=0.7)
            ax.set_yticks([])
            ax.set_ylabel(label[:14], fontsize=7, rotation=0, ha="right")
        for event in record["events"]:
            if event["type"] == "waypoint":
                for ax in axes:
                    ax.axvline(event["t"] - (t[0] if event["t"] >= t[0] else 0), color="#3070d0", lw=0.8, ls=":")
        axes[-1].set_xlabel("seconds from start (dotted: waypoints)")
        fig.suptitle("Who could see you, when")
        fig.text(0.07, 0.02, footer, fontsize=6.5, color="#777")
        pdf.savefig(fig)
        plt.close(fig)
    return buffer.getvalue()
