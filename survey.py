import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import survey_workflow as survey


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare, align and evaluate single-pass surveys on CPU; reconstruction requires explicit GPU approval.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "align", "evaluate", "status", "reconstruct", "inputs"):
        command = commands.add_parser(name)
        command.add_argument("scene")
        if name == "reconstruct":
            command.add_argument("--allow-gpu", action="store_true", help="Explicitly authorize GPU reconstruction; never enabled by preparation or the dashboard.")
            command.add_argument("--dense-profile", choices=sorted(survey.DENSE_PROFILES), default="survey",
                                 help="survey keeps geometric consistency (cleanest, slowest); fast/budget trade density for the <15 min target.")
            command.add_argument("--progressive", action="store_true",
                                 help="Also execute the streaming window plan after the sparse solve: per-window submaps merged into an accumulated model, publishing progressive/checkpoints.json with measured per-window seconds. Diagnostic; a failed window never endangers the reconstruction.")
        if name == "inputs":
            command.add_argument("--telemetry", required=True, type=Path)
            command.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "reconstruct":
            result = survey.reconstruct_scene(ROOT, args.scene, allow_gpu=args.allow_gpu,
                                          dense_profile=args.dense_profile,
                                          progressive=args.progressive)
        elif args.command == "inputs":
            result = survey.save_inputs(ROOT, args.scene, args.telemetry.read_text(encoding="utf-8-sig"), survey.read_json(args.metadata))
        else:
            action = {"prepare": survey.prepare_scene, "align": survey.align_scene,
                      "evaluate": survey.evaluate_scene, "status": survey.scene_status}[args.command]
            result = action(ROOT, args.scene)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        print("[survey] " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
