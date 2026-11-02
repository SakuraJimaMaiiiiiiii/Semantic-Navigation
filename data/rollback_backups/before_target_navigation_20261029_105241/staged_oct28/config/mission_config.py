"""Flight-mission configuration for the RflySim demo."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MissionWaypoint:
    """One local-NED waypoint and its execution options."""

    position_ned: tuple[float, float, float]
    yaw: float = 0.0
    hold_time: float = 0.0


@dataclass(frozen=True)
class MissionConfig:
    """Coordinates and timing for one demo mission.
        face_path: bool = True 表示无人机在飞行过程中始终面向飞行路径的方向。
        face_path: bool = False 表示无人机在飞行过程中始终保持与起飞方向一致的朝向。
    """

    initial_yaw: float | None = None  # None 保持解锁前实测航向；显式数值单位 rad
    takeoff_height: float = 1.5
    takeoff_wait: float = 10.0

    waypoints: tuple[MissionWaypoint, ...] = (
        MissionWaypoint(
            position_ned=(6.0, 2.0, -1.5),
        ),
        MissionWaypoint(
            position_ned=(6.0, 28.0, -1.5),
        ),
        MissionWaypoint(
            position_ned=(-6.0, 28.0, -1.5),
        ),
        MissionWaypoint(
            position_ned=(-6.0, 2.0, -1.5),
        ),
    )

    return_to_origin: bool = True
    face_path: bool = True
    restore_initial_yaw: bool = True  # 返航到原点后恢复解锁前实测航向
    final_hover_time: float = 2.0
