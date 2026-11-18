"""Utility exploration, save after confirmed landing, then release background processes."""

import argparse
from dataclasses import replace
from pathlib import Path
import queue
import sys
import uuid
import webbrowser

from config.exploration_config import FrontierExplorationConfig

DEFAULT_SESSION_FILE = (
    Path(__file__).resolve().parent / "data" / "exploration_session.json"
)


def shutdown_session(server, runtime, primary_error=None, sam3_service=None):
    """Attempt every cleanup step without hiding the fault that ended flight."""
    steps = [("session state", lambda: server.set_state("STOPPING"))]
    if runtime is not None:
        steps.extend([("landing request", lambda: runtime.land_if_armed(hover_first=True)),
                      ("runtime shutdown", runtime.close)])
    if sam3_service is not None:
        steps.append(("SAM3 service", sam3_service.close))
    # Reserve the single-owner port until SDK, sensors and model cleanup finish.
    steps.append(("session server", server.close))
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
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = FrontierExplorationConfig()
    parser.add_argument("--port", type=int, default=defaults.server_port)
    parser.add_argument("--timeout", type=float, default=defaults.timeout)
    parser.add_argument("--max-radius", type=float, default=defaults.max_radius)
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
    perception.add_argument("--sam3", dest="perception_backend", action="store_const",
                            const="sam3", help="SAM3 detection and segmentation (default)")
    perception.add_argument("--sam2", dest="perception_backend", action="store_const",
                            const="yolo_sam2", help="YOLO-World detection and SAM2 segmentation")
    parser.set_defaults(perception_backend="sam3")
    parser.add_argument("--fastlio", action="store_true",
                        help="Experimental FAST-LIO point/semantic map alongside PX4 navigation (assumed snapshot scans)")
    parser.add_argument('--keep-session', action='store_true',
                        help='Keep the live target server and sensors after landing (default: save and exit)')
    args = parser.parse_args(argv)
    config = replace(
        defaults,
        server_port=args.port,
        timeout=args.timeout,
        max_radius=args.max_radius,
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
    backend = args.perception_backend
    from navigation.session_server import SessionServer
    from navigation.exploration_session import ExplorationSession, TargetUnavailableError

    # Acquire the single local control endpoint BEFORE sending any scene command.
    server = SessionServer(uuid.uuid4().hex, config.server_port)
    runtime = sam3_service = None
    try:
        from application import DemoRuntime, WebSocketRequestSender
        from detections import DetectorConfig

        detector = replace(DetectorConfig(), backend=backend)
        if backend == "sam3":
            from detections.sam3_process import SAM3ServiceProcess
            detector = replace(detector, window_name="SAM3 Detection / Semantic Segmentation")
            sam3_service = SAM3ServiceProcess(detector.sam3_url)
            sam3_service.start()

        WebSocketRequestSender().send_when_ue_ready()
        if args.fastlio:
            print("--- FAST-LIO 实验建图：假定雷达单帧瞬时采集；探索及控制仍使用 PX4 地图 ----")
        runtime = DemoRuntime(exploration=True, exploration_config=config, detector_config=detector,
                              fastlio_mode='live' if args.fastlio else None,
                              lidar_scan_model='snapshot_assumed' if args.fastlio else 'unverified')
        runtime.keep_session = args.keep_session
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
        if not args.keep_session:
            print(f'--- 已确认落地并保存探索地图：{session.map_path} ----')
            print('--- 正在关闭感知、SAM3 和建图后台，释放本次运行资源 ----')
            print('--- 后续导航可独立加载保存地图：python navigate_target.py ----')
            return
        server.set_state("READY")
        print("--- 探索结果已保存，后台录像、语义日志和回环重建已停止写入；等待目标导航 ----")
        print(f"--- 已返航并确认落地。保持本终端运行。地图：{session.map_path} ----")
        print("--- 另开终端执行：python navigate_target.py ----")
        while True:
            try:
                command, identity, request = server.commands.get(timeout=0.25)
            except queue.Empty:
                if server.state == "READY":
                    try:
                        session.watch.check()
                    except RuntimeError as error:
                        server.set_state("ERROR")
                        print(f"--- Session continuity lost: {error} ----")
                continue
            if command == "shutdown":
                if session.airborne_navigation:
                    session.return_home()
                break
            try:
                result = (session.return_home() if command == "return_home"
                          else session.navigate(request, identity))
                server.finish(identity, result)
            except TargetUnavailableError as error:
                server.finish(
                    identity,
                    {"success": False, "reason": str(error), "flight_started": False},
                )
            except Exception as error:
                # A flight/runtime fault requires recovery before another command.
                server.finish(
                    identity, {"success": False, "reason": str(error)}, fatal=True
                )
                runtime.land_if_armed(hover_first=True)
                session.annotate(
                    session.selected,
                    session.approach,
                    session.route,
                    "ERROR: " + str(error),
                )
    except KeyboardInterrupt:
        print("--- Stopping exploration session ----")
    finally:
        shutdown_session(server, runtime, primary_error=sys.exc_info()[1], sam3_service=sam3_service)


if __name__ == "__main__":
    main()
