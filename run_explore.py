"""Terminal 1: FUEL-inspired frontiers + fixed-height EGO, return/land, target service."""

import argparse
from dataclasses import replace
from pathlib import Path
import queue
import uuid

from config.exploration_config import FrontierExplorationConfig
from navigation.session_server import SessionServer
from navigation.exploration_session import ExplorationSession, TargetUnavailableError

DEFAULT_SESSION_FILE = (
    Path(__file__).resolve().parent / "data" / "exploration_session.json"
)


def main():
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
        "--minimum-map-gain-area",
        type=float,
        default=defaults.minimum_map_gain_area,
    )
    parser.add_argument(
        "--return-reserve",
        type=float,
        default=defaults.return_time_reserve,
    )
    args = parser.parse_args()
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
        return_time_reserve=args.return_reserve,
    )
    # Acquire the single local control endpoint BEFORE sending any scene command.
    server = SessionServer(uuid.uuid4().hex, config.server_port)
    runtime = None
    try:
        from application import DemoRuntime, WebSocketRequestSender

        WebSocketRequestSender().send_when_ue_ready()
        runtime = DemoRuntime(exploration=True)
        runtime.start()
        session = ExplorationSession(runtime, config, session_id=server.session_id)
        server.map_path = str(session.map_path.resolve())
        server.publish_descriptor(session.directory / "session.json")
        server.publish_descriptor(DEFAULT_SESSION_FILE)
        print(f"--- Frontier exploration output: {session.directory} ----")
        session.explore()
        server.set_state("READY")
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
                break
            try:
                result = session.navigate(request, identity)
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
        server.set_state("STOPPING")
        server.close()
        if runtime is not None:
            try:
                runtime.land_if_armed(hover_first=True)
            finally:
                runtime.close()


if __name__ == "__main__":
    main()
