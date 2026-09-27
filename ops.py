"""Operations analyses on measured clouds: disaster, construction, border (Phase 4).

  python ops.py change   BEFORE AFTER --out packs/change   [--cell 1] [--origin LAT LON ALT]
  python ops.py volume   BEFORE AFTER --polygon "x,y x,y x,y ..." [--cell 0.25]
  python ops.py cutfill  ASBUILT DESIGN(.tif|.xml|.dxf) --out packs/cutfill [--offset E N U]
                         [--baseline PRE.las] [--zones zones.json]
  python ops.py damage   POST --out packs/damage [--footprints fp.json] [--heights h.json]
  python ops.py corridor DSM_OR_CLOUD --line "x,y x,y" --posts posts.json --out packs/border
  python ops.py tiles    CLOUD --out tiles/ [--tile 1000] [--crs-wkt FILE] [--offset E N]

Clouds are .las (survey_formats), .ply (x/y/z vertex) or .npy (N x 3) in ENU metres (or
the same projected frame for every input). ``--origin`` makes results georeferenced
(WGS84 + MGRS on every region, KMZ in the pack). Every pack states its caveats.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402


def load_cloud(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)[:, :3].astype(np.float64)
    if suffix == ".las":
        import survey_formats
        pts = survey_formats.read_las(path)["points"]
        return np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float64)
    if suffix == ".ply":
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"].data
        return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    raise SystemExit(f"{path.name}: clouds are read from .las, .ply or .npy")


def _ring(text):
    try:
        return [tuple(float(v) for v in pair.split(",")) for pair in text.split()]
    except ValueError:
        raise SystemExit(f"could not read coordinates {text!r}; use 'x,y x,y ...'") from None


def _frame(args):
    if not getattr(args, "origin", None):
        return None
    import survey_measure
    return survey_measure.frame_for_origin(*args.origin)


def _captured(args):
    return args.captured or "capture time not given"


def _json(path):
    return json.loads(Path(path).read_text()) if path else None


def cmd_change(args):
    import ops_report
    import survey_change
    out = survey_change.detect_change(load_cloud(args.before), load_cloud(args.after),
                                      cell_m=args.cell, max_shift_m=args.max_shift,
                                      frame=_frame(args), register=not args.no_register)
    pack = ops_report.build_pack(args.out, kind="change", title=args.title or "Change since last flight",
                                 captured=_captured(args), report=out["report"],
                                 grids=dict(dh=out["dh"], transform=out["transform"]))
    return dict(report=out["report"], pack=pack)


def cmd_volume(args):
    import survey_change
    return survey_change.region_volume(load_cloud(args.before), load_cloud(args.after),
                                       _ring(args.polygon), cell_m=args.cell)


def cmd_cutfill(args):
    import ops_report
    import survey_design
    design = survey_design.read_design(args.design)
    cloud = load_cloud(args.asbuilt)
    out = survey_design.cut_fill(cloud, design, cell_m=args.cell, offset=tuple(args.offset),
                                 sigma_reg_m=args.sigma_reg)
    report = dict(out["report"])
    zones = _json(args.zones)
    if zones:
        base = load_cloud(args.baseline) if args.baseline else None
        report["zones"] = survey_design.zone_progress(cloud, design, zones, baseline=base,
                                                      cell_m=args.cell, offset=tuple(args.offset))
    pack = ops_report.build_pack(args.out, kind="cutfill", title=args.title or "Earthworks vs design",
                                 captured=_captured(args), report=report,
                                 grids=dict(diff=out["diff"], transform=out["transform"]))
    return dict(report=report, pack=pack)


def cmd_damage(args):
    import ops_report
    import survey_damage
    out = survey_damage.assess(load_cloud(args.post), footprints=_json(args.footprints),
                               reference_heights=_json(args.heights), cell_m=args.cell,
                               frame=_frame(args))
    out.pop("grids")
    pack = ops_report.build_pack(args.out, kind="damage", title=args.title or "Building damage triage",
                                 captured=_captured(args), report=out)
    return dict(report=out, pack=pack)


def cmd_corridor(args):
    import ops_report
    import survey_change
    import survey_corridor
    source = Path(args.surface)
    if source.suffix.lower() in (".tif", ".tiff"):
        import survey_formats
        data = survey_formats.read_geotiff(source)
        dsm, transform = data["raster"], tuple(data["transform"])
    else:
        cloud = load_cloud(source)
        lo, hi = cloud[:, :2].min(0), cloud[:, :2].max(0)
        g = survey_change.grid_surface(cloud, cell_m=args.cell,
                                       bounds=(lo[0], lo[1], hi[0] + 1e-9, hi[1] + 1e-9))
        dsm, transform = g["zmax"], g["transform"]
    out = survey_corridor.blind_spots(dsm, transform, _ring(args.line), _json(args.posts),
                                      step_m=args.step)
    frame = _frame(args)
    if frame is not None:
        for b in out["summary"]["blind_stretches"]:
            b.update(survey_change._where(b["centre_xy"], frame))
    pack = ops_report.build_pack(args.out, kind="corridor", title=args.title or "Border coverage",
                                 captured=_captured(args), report=out["summary"])
    return dict(report=out["summary"], pack=pack)


def cmd_tiles(args):
    import survey_corridor
    wkt = Path(args.crs_wkt).read_text() if args.crs_wkt else None
    return survey_corridor.tile_products(load_cloud(args.cloud), args.out, tile_m=args.tile,
                                         cell_m=args.cell, crs_wkt=wkt, offset=tuple(args.offset))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, cell):
        p.add_argument("--cell", type=float, default=cell, help="grid cell (m)")
        p.add_argument("--origin", type=float, nargs=3, metavar=("LAT", "LON", "ALT"))
        p.add_argument("--captured", help="capture time printed on every page")
        p.add_argument("--title")
        p.add_argument("--json", action="store_true", help="print the full result as JSON")

    p = sub.add_parser("change", help="two-epoch change with LoD95 (M3)")
    p.add_argument("before"), p.add_argument("after"), p.add_argument("--out", required=True)
    p.add_argument("--max-shift", type=float, default=3.0)
    p.add_argument("--no-register", action="store_true")
    common(p, 1.0)
    p.set_defaults(run=cmd_change)

    p = sub.add_parser("volume", help="volume change inside a polygon (debris, stockpile)")
    p.add_argument("before"), p.add_argument("after"), p.add_argument("--polygon", required=True)
    common(p, 0.25)
    p.set_defaults(run=cmd_volume)

    p = sub.add_parser("cutfill", help="as-built vs design surface (CON-02)")
    p.add_argument("asbuilt"), p.add_argument("design"), p.add_argument("--out", required=True)
    p.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--sigma-reg", type=float, default=0.0)
    p.add_argument("--zones"), p.add_argument("--baseline")
    common(p, 0.5)
    p.set_defaults(run=cmd_cutfill)

    p = sub.add_parser("damage", help="building damage triage (DIS-02)")
    p.add_argument("post"), p.add_argument("--out", required=True)
    p.add_argument("--footprints"), p.add_argument("--heights")
    common(p, 0.5)
    p.set_defaults(run=cmd_damage)

    p = sub.add_parser("corridor", help="post coverage and blind stretches (BOR-04)")
    p.add_argument("surface"), p.add_argument("--line", required=True)
    p.add_argument("--posts", required=True), p.add_argument("--out", required=True)
    p.add_argument("--step", type=float, default=5.0)
    common(p, 1.0)
    p.set_defaults(run=cmd_corridor)

    p = sub.add_parser("tiles", help="tiled DSM + LAS for long corridors (BOR-02)")
    p.add_argument("cloud"), p.add_argument("--out", required=True)
    p.add_argument("--tile", type=float, default=1000.0)
    p.add_argument("--crs-wkt"), p.add_argument("--offset", type=float, nargs=2, default=(0.0, 0.0))
    common(p, 0.5)
    p.set_defaults(run=cmd_tiles)

    args = parser.parse_args(argv)
    try:
        result = args.run(args)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        pack = result.get("pack") if isinstance(result, dict) else None
        print(json.dumps(pack or result, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
