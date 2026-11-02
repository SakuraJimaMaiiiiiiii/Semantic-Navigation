"""Terminal 1: Utility frontier decisions + existing A* paths + EGO local avoidance, return/land, target service."""

import argparse
import errno
import json
from dataclasses import replace
from pathlib import Path
import sys
import uuid
import webbrowser

from config.exploration_config import FrontierExplorationConfig

DEFAULT_SESSION_FILE = (
    Path(__file__).resolve().parent / "data" / "exploration_session.json"
)

DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent / "config" / "explore.json"


def shutdown_session(server, runtime, primary_error=None):
    """Attempt every cleanup step without hiding the fault that ended flight."""
    steps = [("session state", lambda: server.set_state("STOPPING")),
             ("session server", server.close)]
    if runtime is not None:
        steps.extend([("landing request", lambda: runtime.land_if_armed(hover_first=True)),
                      ("runtime shutdown", runtime.close)])
    errors = []
    for name, stop in steps:
        try:
            stop()
        except Exception as error:
            errors.append(error)
            print(f"--- 清理失败 [{name}]: {type(error).__name__}: {error} ----", file=sys.stderr)
    if errors and primary_error is None:
        raise errors[0]


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


def main():
    config, args = parse_config(include_perception=True)
    print(f"--- 探索配置：{args.config.resolve()}；感知={args.perception_backend}；FAST-LIO={args.fastlio or 'off'} ----")
    from navigation.session_server import SessionServer
    from navigation.exploration_session import ExplorationSession

    # Acquire the single local control endpoint BEFORE sending any scene command.
    try:
        server = SessionServer(uuid.uuid4().hex, config.server_port)
    except OSError as error:
        if error.errno != errno.EADDRINUSE and getattr(error, "winerror", None) != 10048:
            raise
        print(
            f"探索会话端口 127.0.0.1:{config.server_port} 已被占用。\n"
            "旧探索进程在返航落地后仍会等待目标导航，因此不会自动退出。\n"
            "请先运行 python navigate_target.py --status 检查旧会话；\n"
            "若为 READY 或 ERROR，运行 python navigate_target.py --shutdown，"
            "等待旧进程完成清理后再启动。\n"
            "若旧会话仍在飞行，请等待任务完成，不要同时启动第二个控制进程。\n"
            f"若无法联系旧会话，用 PowerShell 查看占用者："
            f"Get-NetTCPConnection -LocalPort {config.server_port} | "
            "Select-Object State,OwningProcess",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    runtime = None
    sam3_service = None
    try:
        if args.perception_backend == "sam3":
            from detections.sam3_process import SAM3ServiceProcess
            sam3_service = SAM3ServiceProcess(args.sam3_url)
            try:
                sam3_service.start()
            except (RuntimeError, TimeoutError) as error:
                print(f"--- SAM3 启动失败：{error} ----", file=sys.stderr)
                raise SystemExit(1) from None
        from application import DemoRuntime, WebSocketRequestSender

        WebSocketRequestSender().send_when_ue_ready()
        from detections import DetectorConfig
        detector_config = replace(DetectorConfig(), backend=args.perception_backend, sam3_url=args.sam3_url)
        if args.object_prompts:
            detector_config = replace(detector_config, class_prompts=tuple(args.object_prompts),
                                      semantic_landmark_classes=tuple(args.object_prompts))
        if args.perception_backend == "sam3":
            detector_config = replace(detector_config, window_name="SAM3 Detection / Semantic Segmentation")
        runtime = DemoRuntime(exploration=True, exploration_config=config, detector_config=detector_config,
                              fastlio_mode=args.fastlio, lidar_scan_model=args.lidar_scan_model)
        runtime.start()
        session = ExplorationSession(runtime, config, session_id=server.session_id)
        server.map_path = str(session.map_path.resolve())
        server.publish_descriptor(session.directory / "session.json")
        server.publish_descriptor(DEFAULT_SESSION_FILE)
        print(f"--- Frontier exploration output: {session.directory} ----")
        session.annotate(state="准备起飞：雷达建图 / Utility 探索")
        map_url = session.map_path.resolve().as_uri()
        try:
            if not webbrowser.open(map_url, new=2):
                print(f"--- 浏览器未自动打开，请打开地图：{session.map_path} ----")
        except Exception as error:
            print(f"--- 打开地图失败：{error}；可手动打开：{session.map_path} ----")
        session.explore()
        server.set_state("READY")
        print("--- 探索结果已保存，后台录像、语义日志和回环重建已停止写入；等待目标导航 ----")
        print(f"--- 已返航并确认落地。保持本终端运行。地图：{session.map_path} ----")
        print("--- 可在本终端直接输入 vehicle_01 等目标 ID；也支持另开终端运行 navigate_target.py --client ----")
        from navigation.exploration_session import serve_target_commands
        serve_target_commands(session, server)
    except KeyboardInterrupt:
        print("--- Stopping exploration session ----")
    finally:
        try:
            shutdown_session(server, runtime, primary_error=sys.exc_info()[1])
        finally:
            if sam3_service is not None:
                sam3_service.close()


if __name__ == "__main__":
    main()
