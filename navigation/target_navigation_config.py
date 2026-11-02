"""Configuration parser for independent saved-map target navigation."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from config.exploration_config import FrontierExplorationConfig

DEFAULT_CONFIG_FILE = Path(__file__).resolve().parents[1] / "config" / "explore.json"


def parse_config(argv=None, *, include_perception=False):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
    selection, _ = selector.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=selection.config,
                        help="Exploration startup JSON; command-line options override this file")
    defaults = FrontierExplorationConfig()
    parser.add_argument("--port", type=int, default=defaults.server_port)
    parser.add_argument("--timeout", type=float, default=defaults.timeout)
    parser.add_argument("--max-radius", type=float, default=defaults.max_radius)
    parser.add_argument("--height-step", type=float, default=defaults.height_step)
    parser.add_argument("--max-height-gain", type=float, default=defaults.max_height_gain)
    parser.add_argument("--max-viewpoints", type=int, default=defaults.max_viewpoints)
    parser.add_argument(
        "--viewpoint-spacing", type=float, default=defaults.viewpoint_spacing
    )
    parser.add_argument(
        "--information-radius", type=float, default=defaults.information_radius
    )
    parser.add_argument(
        "--frontier-empty-seconds",
        type=float,
        default=defaults.frontier_empty_min_duration,
    )
    parser.add_argument(
        "--completion-frontier-area",
        type=float,
        default=defaults.completion_frontier_area,
    )
    parser.add_argument(
        "--coverage-plateau-seconds", type=float,
        default=defaults.coverage_plateau_seconds,
        help="Seconds without meaningful map expansion before a frontier review",
    )
    parser.add_argument(
        "--minimum-map-gain-area",
        type=float,
        default=defaults.minimum_map_gain_area,
    )
    parser.add_argument(
        "--return-reserve",
        type=float,
        default=defaults.return_time_reserve,
    )
    perception = parser.add_mutually_exclusive_group()
    perception.add_argument("--sam2", dest="perception_backend", action="store_const",
                            const="yolo_sam2", help="Use YOLO-World + SAM2 (default)")
    perception.add_argument("--sam3", dest="perception_backend", action="store_const",
                            const="sam3", help="Use SAM3 with automatic service startup")
    parser.set_defaults(perception_backend="yolo_sam2")
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8878")
    parser.add_argument("--fastlio", choices=("off", "record", "shadow", "live"),
                        help="record: raw inputs; shadow: LIO poses; live: LIO poses and point map (navigation unchanged)")
    parser.add_argument("--lidar-scan-model", choices=("unverified", "snapshot", "snapshot_assumed"), default="unverified",
                        help="snapshot requires verified instantaneous scans; snapshot_assumed is an explicit experimental assumption")
    parser.add_argument("--object-prompts", nargs="+", help='Shared prompts, e.g. car "fire hose cabinet"')
    try:
        values = json.loads(selection.config.read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict):
            raise ValueError("configuration must be a JSON object")
        actions = {action.dest: action for action in parser._actions
                   if action.dest not in {"help", "config", "perception_backend"}}
        config_argv = []
        for key, value in values.items():
            if key == "_comment":
                continue
            if key == "perception_backend":
                if value not in {"sam3", "yolo_sam2"}:
                    raise ValueError("perception_backend must be sam3 or yolo_sam2")
                config_argv.append("--sam3" if value == "sam3" else "--sam2")
                continue
            if key not in actions:
                raise ValueError(f"Unknown startup option: {key}")
            if value is None:
                if key not in {"fastlio", "object_prompts"}:
                    raise ValueError(f"{key} cannot be null")
                continue
            action = actions[key]
            if key == "object_prompts":
                if not isinstance(value, list) or not value or not all(
                        isinstance(item, str) and item.strip() and not item.startswith("-") for item in value):
                    raise ValueError("object_prompts must be null or a nonempty list of prompt strings")
                config_argv.extend([action.option_strings[0], *value])
            else:
                if isinstance(value, (bool, list, dict)):
                    raise ValueError(f"Invalid value for {key}")
                config_argv.extend([action.option_strings[0], str(value)])
        # Parse separately so mutually exclusive perception flags in the CLI
        # override the file without creating a false --sam2/--sam3 conflict.
        file_args = parser.parse_args(config_argv)
        args = parser.parse_args(argv, namespace=file_args)
        if args.fastlio == "off":
            args.fastlio = None
    except (OSError, ValueError) as error:
        parser.error(f"Cannot load {selection.config}: {error}")
    if args.fastlio in {"shadow", "live"} and args.lidar_scan_model == "unverified":
        parser.error("FAST-LIO requires --lidar-scan-model snapshot (verified) or snapshot_assumed (experimental)")
    config = replace(
        defaults,
        server_port=args.port,
        timeout=args.timeout,
        max_radius=args.max_radius,
        height_step=args.height_step,
        max_height_gain=args.max_height_gain,
        max_viewpoints=args.max_viewpoints,
        viewpoint_spacing=args.viewpoint_spacing,
        information_radius=args.information_radius,
        frontier_empty_min_duration=args.frontier_empty_seconds,
        completion_frontier_area=args.completion_frontier_area,
        minimum_map_gain_area=args.minimum_map_gain_area,
        coverage_plateau_seconds=args.coverage_plateau_seconds,
        return_time_reserve=args.return_reserve,
    )

    return (config, args) if include_perception else config
