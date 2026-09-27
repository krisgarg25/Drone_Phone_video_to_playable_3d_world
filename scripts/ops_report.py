"""Field packs and one-page reports for the operations applications (DIS-09, CON-08).

A field team gets a folder (and a zip of it) that opens without our software:

* ``report.pdf`` - title, capture time, key numbers, tables and map panels, caveats
  (matplotlib's PDF backend; no other dependency);
* ``<name>.kmz`` - placemarks for every region/building/stretch with lat/lon (only when
  georeferenced - a local scene gets no KMZ rather than one placed off Africa);
* ``<name>.csv`` - the same rows for spreadsheets;
* ``overview.tif`` - a downsampled raster when a CRS is given;
* ``pack.json`` - what is in the pack and every caveat, machine-readable.

Every page and every placemark carries the capture time and "measured from drone imagery;
heuristic where stated", so a printed page cannot lose its provenance.
"""
import csv
import html
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

KINDS = ("change", "cutfill", "damage", "corridor", "access", "flood", "detections")
KML_COLOURS = {"gain": "ff0000ff", "loss": "ffff0000", "collapsed": "ff0000ff",
               "partial": "ff00a5ff", "intact": "ff00c800", "unknown": "ff969696",
               "blind": "ff00ffff"}


def _rows_csv(path, rows):
    keys = []
    for row in rows:
        for key, value in row.items():
            if key not in keys and not isinstance(value, (list, dict)):
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def write_kmz(path, name, placemarks, *, captured):
    """``placemarks``: dicts with lat, lon, name, style, description (plain text)."""
    styles = "".join(f'<Style id="{k}"><IconStyle><color>{v}</color></IconStyle></Style>'
                     for k, v in KML_COLOURS.items())
    marks = []
    for mark in placemarks:
        if mark.get("lat") is None or mark.get("lon") is None:
            continue
        text = html.escape(f"{mark.get('description', '')}\nCaptured {captured}. "
                           "Measured from drone imagery; heuristic where stated.")
        marks.append(f"<Placemark><name>{html.escape(str(mark['name']))}</name>"
                     f"<styleUrl>#{mark.get('style', 'unknown')}</styleUrl>"
                     f"<description>{text}</description>"
                     f"<Point><coordinates>{mark['lon']:.7f},{mark['lat']:.7f},0</coordinates>"
                     "</Point></Placemark>")
    kml = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
           f"<name>{html.escape(name)}</name>{styles}{''.join(marks)}</Document></kml>")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("doc.kml", kml)
    return len(marks)


def write_pdf(path, *, title, captured, facts, tables=(), panels=(), caveats=()):
    """One or more A4 pages: header, fact list, tables, raster panels, caveats.

    ``panels``: (title, 2D array, cmap, (vmin, vmax) or None, extent or None, colorbar label).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    footer = f"Captured {captured} · generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · " \
             "measured from drone imagery; heuristic where stated"
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, title, fontsize=16, weight="bold")
        fig.text(0.07, 0.925, f"Capture: {captured}", fontsize=9, color="#444")
        y = 0.89
        for key, value in facts:
            fig.text(0.07, y, str(key), fontsize=10, color="#333")
            fig.text(0.45, y, str(value), fontsize=10, weight="bold")
            y -= 0.022
        for table_title, header, rows in tables:
            y -= 0.015
            fig.text(0.07, y, table_title, fontsize=11, weight="bold")
            y -= 0.02
            shown = rows[:18]
            if shown:
                ax = fig.add_axes([0.07, max(0.05, y - 0.019 * (len(shown) + 1)), 0.86,
                                   0.019 * (len(shown) + 1)])
                ax.axis("off")
                t = ax.table(cellText=[[str(c) for c in r] for r in shown], colLabels=header,
                             loc="upper left", cellLoc="left")
                t.auto_set_font_size(False)
                t.set_fontsize(7.5)
                y -= 0.019 * (len(shown) + 1) + 0.01
            if len(rows) > len(shown):
                fig.text(0.07, y, f"... {len(rows) - len(shown)} more rows in the CSV", fontsize=8)
                y -= 0.02
        if caveats:
            y = min(y - 0.02, 0.2)
            fig.text(0.07, y, "Limits", fontsize=10, weight="bold")
            for note in caveats:
                y -= 0.018
                fig.text(0.07, y, "• " + str(note), fontsize=7.5, wrap=True)
        fig.text(0.07, 0.02, footer, fontsize=6.5, color="#666")
        pdf.savefig(fig)
        plt.close(fig)
        for panel_title, grid, cmap, limits, extent, label in panels:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            vmin, vmax = limits if limits else (None, None)
            image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=vmin, vmax=vmax,
                              extent=extent, interpolation="nearest")
            ax.set_title(panel_title)
            ax.set_xlabel("east (m)")
            ax.set_ylabel("north (m)")
            ax.set_facecolor("#dddddd")          # unobserved shows grey, never a colour
            fig.colorbar(image, ax=ax, shrink=0.6, label=label)
            fig.text(0.07, 0.02, footer + " · grey = not observed", fontsize=6.5, color="#666")
            pdf.savefig(fig)
            plt.close(fig)
    return Path(path)


def _extent(transform, shape):
    a, b, _, d, _, f = transform
    return (a, a + shape[1] * b, d + shape[0] * f, d)


def build_pack(out_dir, *, kind, title, captured, report, grids=None, crs_wkt=None,
               offset=(0.0, 0.0)):
    """Write a field pack for one analysis ``kind``: change, cutfill, damage, corridor."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grids = grids or {}
    facts, tables, panels, rows, marks = [], [], [], [], []
    caveats = list(report.get("notes", []))
    if kind == "change":
        reg, vol, cells = report["registration"], report["volume"], report["cells"]
        facts = [("Gain / loss volume", f"{vol['gain_m3']} / {vol['loss_m3']} m³"),
                 ("Median LoD95", f"{report['lod95_median_m']} m"),
                 ("Registration shift E/N/U", f"{reg['dx_m']} / {reg['dy_m']} / {reg['dz_m']} m"),
                 ("Registration sigma", f"{reg['sigma_reg_m']} m on {reg['stable_cells']} cells"),
                 ("Unobserved cells", f"{cells['unobserved']} of {cells['total']}")]
        rows = report["regions"]
        tables = [("Change regions", ["id", "kind", "area m²", "mean dh m", "volume m³", "±", "MGRS"],
                   [[r["id"], r["kind"], r["area_m2"], r["mean_dh_m"], r["volume_m3"],
                     r["volume_sigma_m3"], r.get("mgrs", "local")] for r in rows])]
        marks = [dict(r, name=f"{r['id']} {r['kind']} {r['volume_m3']} m³", style=r["kind"],
                      description=f"{r['area_m2']} m², mean {r['mean_dh_m']} m, "
                                  f"{r['volume_m3']} ± {r['volume_sigma_m3']} m³") for r in rows]
        if "dh" in grids:
            lim = float(np.nanpercentile(np.abs(grids["dh"]), 99)) or 1.0
            panels.append(("Height change (after − before)", grids["dh"], "RdBu", (-lim, lim),
                           _extent(grids["transform"], grids["dh"].shape), "m"))
    elif kind == "cutfill":
        facts = [("Design", report.get("design")),
                 ("Cut (to remove)", f"{report['cut_m3']} ± {report['cut_sigma_m3']} m³"),
                 ("Fill (to place)", f"{report['fill_m3']} ± {report['fill_sigma_m3']} m³"),
                 ("Net", f"{report['net_m3']} m³"),
                 ("On-grade tolerance", f"{report['tolerance_m']} m"),
                 ("Design area observed", f"{report['design_observed_fraction']:.0%}")]
        rows = [dict(item="area_" + k, value=v) for k, v in report["area_m2"].items()]
        if report.get("zones"):
            tables.append(("Zone progress", ["zone", "remaining m³", "start m³", "progress %"],
                           [[z["zone"], z["remaining_m3"], z.get("start_m3", "-"),
                             z.get("progress_pct", "-")] for z in report["zones"]]))
            rows = report["zones"]
        if "diff" in grids:
            lim = float(np.nanpercentile(np.abs(grids["diff"]), 99)) or 1.0
            panels.append(("As-built minus design (red = cut)", grids["diff"], "RdBu_r",
                           (-lim, lim), _extent(grids["transform"], grids["diff"].shape), "m"))
    elif kind == "damage":
        counts = report["counts"]
        facts = [(k.capitalize(), v) for k, v in counts.items()] + [("Footprints", report["footprints"])]
        rows = report["buildings"]
        tables = [("Buildings", ["id", "grade", "reason", "height m", "MGRS"],
                   [[b["id"], b["grade"], b.get("reason", "")[:48], b.get("height_m", "-"),
                     b.get("mgrs", "local")] for b in rows])]
        marks = [dict(b, name=f"{b['id']} {b['grade']}", style=b["grade"],
                      description=b.get("reason", "")) for b in rows]
    elif kind == "corridor":
        s = report
        facts = [("Line length", f"{s['line_length_m']} m"),
                 ("Covered", f"{s['covered_fraction']:.0%}"),
                 ("Blind stretches", len(s["blind_stretches"]))]
        rows = s["blind_stretches"]
        tables = [("Blind stretches", ["from m", "to m", "length m", "unobserved ground"],
                   [[b["from_m"], b["to_m"], b["length_m"], b["unobserved_ground"]] for b in rows])]
        marks = [dict(b, name=f"blind {b['from_m']}-{b['to_m']} m", style="blind",
                      description=f"{b['length_m']} m not seen by any post") for b in rows]
    elif kind == "access":
        route = report.get("route") or {}
        facts = [("Blocked road stretches", len(report.get("blocked", []))),
                 ("Blocked road area", f"{report.get('blocked_m2', 0)} m² of {report.get('road_m2', 0)} m²")]
        if route.get("length_m") is not None:
            facts += [("Vehicle route", f"{route['length_m']} m ({route['road_m']} m on road)"),
                      ("ETA", f"{route['eta_s'] / 60:.1f} min"), ("Vehicle width", f"{route['vehicle_width_m']} m")]
        elif report.get("route_error"):
            facts.append(("Vehicle route", "none: " + report["route_error"]))
        rows = report.get("blocked", [])
        tables = [("Blocked roads", ["id", "area m²", "max height m", "MGRS"],
                   [[b["id"], b["area_m2"], b["max_height_m"], b.get("mgrs", "local")] for b in rows])]
        marks = [dict(b, name=f"{b['id']} blocked road", style="collapsed",
                      description=f"{b['area_m2']} m², obstacle up to {b['max_height_m']} m") for b in rows]
        caveats += route.get("notes", [])
    elif kind == "flood":
        facts = [("Water level", f"{report['level_m']} m"), ("Flooded area", f"{report['flooded_m2']} m²"),
                 ("Water volume", f"{report['volume_m3']} m³"), ("Max depth", f"{report['max_depth_m']} m")] +                 [(f"Depth {k}", f"{v} m²") for k, v in report["depth_bands_m2"].items()]
        rows = report.get("buildings", [])
        tables = [("Buildings touching water", ["id", "max depth m", "MGRS"],
                   [[b["id"], b["max_depth_m"], b.get("mgrs", "local")] for b in rows])]
        marks = [dict(b, name=f"{b['id']} flooded {b['max_depth_m']} m", style="partial",
                      description=f"water up to {b['max_depth_m']} m at the walls") for b in rows]
        if "depth" in grids:
            panels.append(("Flood depth", grids["depth"], "Blues", (0, max(0.5, report["max_depth_m"])),
                           _extent(grids["transform"], grids["depth"].shape), "m"))
    elif kind == "detections":
        rows = report.get("objects", [])
        facts = [("Objects", len(rows)), ("Source", report.get("source", "")),
                 ("Unplaced detections", report.get("unplaced", 0))]
        tables = [("People and vehicles", ["id", "class", "sightings", "first s", "last s", "MGRS"],
                   [[o["id"], o["class"], o["sightings"], o.get("first_t"), o.get("last_t"), o.get("mgrs", "local")]
                    for o in rows])]
        marks = [dict(o, name=f"{o['id']} {o['class']}", style="partial",
                      description=f"seen {o['sightings']}x, t {o.get('first_t')}-{o.get('last_t')} s") for o in rows]
    else:
        raise ValueError(f"unknown pack kind {kind!r}")
    files = []
    stem = f"{kind}"
    _rows_csv(out_dir / f"{stem}.csv", rows)
    files.append(f"{stem}.csv")
    write_pdf(out_dir / "report.pdf", title=title, captured=captured, facts=facts,
              tables=tables, panels=panels, caveats=caveats)
    files.append("report.pdf")
    if any(m.get("lat") is not None for m in marks):
        count = write_kmz(out_dir / f"{stem}.kmz", title, marks, captured=captured)
        files.append(f"{stem}.kmz")
    else:
        count = 0
        caveats.append("No KMZ: the scene is not georeferenced (local coordinates only).")
    if crs_wkt and panels:
        import survey_formats
        grid = panels[0][1]
        step = max(1, int(np.ceil(max(grid.shape) / 2000)))
        a, b, c, d, e, f = grids["transform"]
        survey_formats.write_geotiff(np.asarray(grid[::step, ::step], np.float32),
                                     out_dir / "overview.tif",
                                     transform=(a + offset[0], b * step, 0.0, d + offset[1], 0.0, f * step),
                                     crs_wkt=crs_wkt, nodata=-9999.0)
        files.append("overview.tif")
    manifest = dict(kind=kind, title=title, captured=captured, files=files,
                    placemarks=count, caveats=caveats,
                    generated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (out_dir / "pack.json").write_text(json.dumps(manifest, indent=2, default=str))
    with zipfile.ZipFile(out_dir.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files + ["pack.json"]:
            archive.write(out_dir / name, name)
    return manifest
