"""只负责飞行控制、不负责传感器读取和数据保存的任务执行器。"""

import math

import numpy as np

from vehicle.rfly_multirotor import CollisionDetectedError


class RflyMissionController:
    """按照 MissionConfig 执行起飞、航点、返航和降落。"""

    def __init__(
        self,
        vehicle,
        mission,
        local_planner=None,
        global_return_planner=None,
        return_safety_planner=None,
        semantic_navigator=None,
    ) -> None:
        self.vehicle = vehicle
        self.mission = mission
        self.local_planner = local_planner
        self.global_return_planner = global_return_planner
        self.return_safety_planner = return_safety_planner
        self._return_origin_ned = None
        self.semantic_navigator = semantic_navigator

    def run(self) -> None:
        """执行完整飞行任务；碰撞异常继续交给上层统一处理。"""
        vehicle = self.vehicle
        mission = self.mission

        self._return_origin_ned = vehicle.get_position()
        pre_takeoff_yaw = float(vehicle.get_euler()[2])
        if not math.isfinite(pre_takeoff_yaw):
            raise ValueError("Pre-takeoff yaw must be finite.")
        print(
            "--- 本次任务返航原点："
            f"NED={self._return_origin_ned.round(3).tolist()}，"
            f"起飞前航向={math.degrees(pre_takeoff_yaw):.1f}° ----"
        )
        takeoff_yaw = pre_takeoff_yaw if mission.initial_yaw is None else float(mission.initial_yaw)
        if not math.isfinite(takeoff_yaw):
            raise ValueError("Takeoff yaw must be finite.")
        vehicle.enable_offboard(arm=True)
        vehicle.takeoff(mission.takeoff_height, yaw=takeoff_yaw)
        vehicle.wait(mission.takeoff_wait)
        # 升空后再用 yaw 角速度闭环精确对准，避免地面约束导致超时。
        if mission.initial_yaw is not None:
            vehicle.align_yaw(takeoff_yaw)

        if self.semantic_navigator is not None:
            self.semantic_navigator.run()
            configure = getattr(self.return_safety_planner, "set_navigation_plane", None)
            altitude = getattr(self.semantic_navigator, "altitude", None)
            if callable(configure) and altitude is not None:
                configure(altitude, self.semantic_navigator.config, self.semantic_navigator.global_map)
        else:
            for waypoint in mission.waypoints:
                vehicle.move_velocity_guided(
                    waypoint.position_ned,
                    yaw=waypoint.yaw,
                    face_direction=mission.face_path,
                    local_planner=self.local_planner,
                )
                if waypoint.hold_time > 0.0:
                    vehicle.wait(waypoint.hold_time)

        if mission.return_to_origin:
            if self.global_return_planner is None:
                self._return_with_local_planner()
            else:
                try:
                    self._return_on_global_path()
                except RuntimeError as error:
                    if isinstance(error, CollisionDetectedError):
                        raise
                    print(
                        "--- 全局记忆 A* 返航不可用，"
                        f"退回局部安全规划器：{error} ----"
                    )
                    self._return_with_local_planner()
            if mission.restore_initial_yaw:
                print(
                    "--- 已返回起飞点，恢复起飞前航向："
                    f"yaw={math.degrees(pre_takeoff_yaw):.1f}° ----"
                )
                vehicle.align_yaw(pre_takeoff_yaw)

        vehicle.wait(mission.final_hover_time)
        vehicle.land()

    def _return_with_local_planner(self) -> None:
        """朝向返航原点，并由返航局部规划器执行直达避障。"""
        vehicle = self.vehicle
        return_planner = self.return_safety_planner
        if return_planner is None:
            return_planner = self.local_planner
        origin = self._return_origin()
        position = vehicle.get_position()
        direction = origin[:2] - position[:2]
        if float(np.linalg.norm(direction)) <= 1e-6:
            print("--- 已位于本次起飞点，无需执行局部返航 ----")
            return
        return_yaw = math.atan2(
            float(direction[1]),
            float(direction[0]),
        )
        print(
            "--- 无全局路径，局部安全规划器直返：飞行中逐步转向，"
            f"目标 yaw={math.degrees(return_yaw):.1f}° ----"
        )
        target = position.copy()
        target[:2] = origin[:2]
        print(
            f"返回本次起飞点 North={origin[0]:.2f} m, "
            f"East={origin[1]:.2f} m……"
        )
        vehicle.move_velocity_guided(
            target,
            yaw=return_yaw,
            face_direction=True,
            local_planner=return_planner,
        )

    def _return_on_global_path(self) -> None:
        """一次性生成全局返航路径，并沿固定航点顺序执行。"""
        vehicle = self.vehicle
        origin = self._return_origin()
        current = vehicle.get_position()
        target = current.copy()
        target[:2] = origin[:2]
        path = self.global_return_planner.plan(current, target)
        print(
            f"--- 全局记忆 A* 返航路径已生成："
            f"{len(path)} 个稳定航点 ----"
        )
        yaw_aligned = False
        for index, waypoint in enumerate(path[1:], start=1):
            position = vehicle.get_position()
            direction = np.asarray(waypoint[:2]) - position[:2]
            if float(np.linalg.norm(direction)) <= 1e-6:
                continue
            segment_yaw = math.atan2(
                float(direction[1]),
                float(direction[0]),
            )
            print(
                f"--- 返航航点 {index}/{len(path) - 1}："
                f"NED={waypoint.round(2).tolist()} ----"
            )
            if not yaw_aligned:
                print(
                    "--- 先原地对准首个返航航段："
                    f"yaw={math.degrees(segment_yaw):.1f}° ----"
                )
                vehicle.align_yaw(segment_yaw)
                yaw_aligned = True
            elif index < len(path) - 1:
                print(
                    "--- 连续通过中间航点，飞行中平滑衔接下一航段 ----"
                )
            vehicle.move_velocity_guided(
                waypoint,
                yaw=segment_yaw,
                face_direction=False,
                local_planner=self.return_safety_planner,
                pass_through=index < len(path) - 1,
            )

    def _return_origin(self) -> np.ndarray:
        if self._return_origin_ned is None:
            raise RuntimeError("Return origin has not been recorded.")
        return np.asarray(self._return_origin_ned, dtype=float).copy()
