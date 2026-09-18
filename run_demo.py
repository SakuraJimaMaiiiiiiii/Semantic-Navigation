"""在 UE 已连接 WebSocket 后运行 RflySim 地下车库语义导航。"""

from application import DemoRuntime, WebSocketRequestSender
from vehicle import CollisionDetectedError


def main() -> None:
    WebSocketRequestSender().send_when_ue_ready()
    runtime = DemoRuntime()
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
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
