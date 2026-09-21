"""Evaluate semantic runs; truth must use the same local NED frame and metres."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation.evaluation import evaluate_map, evaluate_identity_frames, summarize_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help='JSON {"instances": [{"instance_id", "class_name", "position_world"}]}',
    )
    parser.add_argument("--match-distance", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.match_distance < float("inf"):
        parser.error("--match-distance must be finite and positive")
    truth = (
        None
        if args.ground_truth is None
        else json.loads(args.ground_truth.read_text(encoding="utf-8"))["instances"]
    )
    runs, maps, missing_results = [], {}, []
    for directory in args.runs:
        result = directory / "evaluation" / "semantic_navigation.json"
        if result.exists():
            runs.append(json.loads(result.read_text(encoding="utf-8")))
        else:
            missing_results.append(str(directory))
        if truth is not None:
            records = json.loads(
                (directory / "maps" / "semantic_instances.json").read_text(
                    encoding="utf-8"
                )
            )["instances"]
            metrics = evaluate_map(records, truth, args.match_distance)
            frames = directory / "maps" / "semantic_observations.jsonl"
            if frames.exists():
                with frames.open(encoding="utf-8") as stream:
                    metrics.update(
                        evaluate_identity_frames(
                            [json.loads(line) for line in stream],
                            truth,
                            args.match_distance,
                        )
                    )
            else:
                metrics["id_switches"] = None
            maps[str(directory)] = metrics
    output = json.dumps(
        {
            "navigation": summarize_runs(runs),
            "maps": maps,
            "ground_truth_supplied": truth is not None,
            "missing_navigation_results": missing_results,
        },
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
