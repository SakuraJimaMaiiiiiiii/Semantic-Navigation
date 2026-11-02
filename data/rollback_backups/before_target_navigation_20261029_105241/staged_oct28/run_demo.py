"""在 UE 已连接 WebSocket 后运行 RflySim 地下车库语义导航。"""

import argparse
from config.semantic_navigation_config import DEFAULT_TARGET_FILE
from navigation.target import load_target_request

from application import DemoRuntime, WebSocketRequestSender
from vehicle import CollisionDetectedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-target", default=str(DEFAULT_TARGET_FILE), help="JSON target request file")
    args = parser.parse_args()
    target = load_target_request(args.semantic_target)
    WebSocketRequestSender().send_when_ue_ready()
    runtime = DemoRuntime(semantic_target=target)
    try:
        runtime.start()
        runtime.run_mission()
    except CollisionDetectedError as error:
        print(f"--- Collision warning: {error} ----")
        print("--- Aborting mission and commanding AUTO.LAND... ----")
        runtime.land_if_armed()
    except TimeoutError as error:
        print(f"--- Mission timeout: {error} ----")
        print("--- Hovering, then commanding AUTO.LAND... ----")
        runtime.land_if_armed(hover_first=True)
    except Exception as error:
        print(f"--- Mission failed: {error} ----")
        runtime.land_if_armed(hover_first=True)
        raise
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
