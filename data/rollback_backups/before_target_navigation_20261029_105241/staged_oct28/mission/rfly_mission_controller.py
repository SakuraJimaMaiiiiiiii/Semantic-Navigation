"""只负责飞行控制、不负责传感器读取和数据保存的任务执行器。"""

import math
import time

import numpy as np



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

    def run(self, *, hold_at_target=False) -> None:
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
        self._return_yaw = pre_takeoff_yaw
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
        else:
            for waypoint in mission.waypoints:
                vehicle.move_waypoint_guided(
                    waypoint.position_ned,
                    yaw=waypoint.yaw,
                    face_direction=mission.face_path,
                    local_planner=self.local_planner,
                )
                if waypoint.hold_time > 0.0:
                    vehicle.wait(waypoint.hold_time)

        if hold_at_target:
            vehicle.hover()
            self._set_navigation_phase("holding_at_target")
            return
        self.return_and_land()

    def run_target(self, *, resume=False):
        if resume:
            self.vehicle._require_offboard()
            if not self.vehicle.mav.isArmed:
                raise RuntimeError("Cannot resume target navigation: vehicle is disarmed")
            self.semantic_navigator.run()
            self.vehicle.hover()
            self._set_navigation_phase("holding_at_target")
        else:
            self.run(hold_at_target=True)

    def return_and_land(self, *, force_return=False):
        vehicle, mission = self.vehicle, self.mission
        pre_takeoff_yaw = self._return_yaw
        return_to_origin = force_return or mission.return_to_origin
        if return_to_origin:
            self._set_navigation_phase("returning")
            if self.global_return_planner is None:
                self._return_with_local_planner()
            else:
                self._return_on_global_path()
        vehicle.wait(mission.final_hover_time)
        if return_to_origin:
            self._confirm_return_position()
            if mission.restore_initial_yaw:
                print(
                    "--- 已返回起飞点，恢复起飞前航向："
                    f"yaw={math.degrees(pre_takeoff_yaw):.1f}° ----"
                )
                vehicle.align_yaw(pre_takeoff_yaw)
                # Rotation holds position, but verify that home was not lost.
                self._confirm_return_position()
        if return_to_origin:
            nav_config = getattr(self.semantic_navigator, "config", None)
            exploration_config = getattr(self.semantic_navigator, "exploration_config", None)
            tolerance = float(getattr(exploration_config, "home_tolerance",
                                      getattr(nav_config, "arrival_tolerance", .3)))
            vehicle.approach_home_ground(self._return_origin_ned, tolerance)
        # Hand off to PX4 before a map/HTML checkpoint can block this thread.
        vehicle.land()
        self._set_navigation_phase("landing")

    def _set_navigation_phase(self, phase):
        nav = self.semantic_navigator
        if nav is not None:
            nav.phase = phase
            nav.active_route = None
            checkpoint = getattr(nav, "checkpoint", None)
            if callable(checkpoint):
                checkpoint(nav)

    def _confirm_return_position(self):
        # Confirm horizontally before landing: home Z is on the ground.
        position = np.asarray(self.vehicle.get_position(), dtype=float)
        delta = position[:2] - self._return_origin()[:2]
        nav_config = getattr(self.semantic_navigator, "config", None)
        exploration_config = getattr(self.semantic_navigator, "exploration_config", None)
        tolerance = float(getattr(exploration_config, "home_tolerance",
                                  getattr(nav_config, "arrival_tolerance", .3)))
        if not np.all(np.isfinite(position)) or np.linalg.norm(delta) > tolerance:
            self.vehicle.hover()
            raise RuntimeError(f"Return position not confirmed before landing: horizontal_error={np.linalg.norm(delta):.3f} m")


    def _return_with_local_planner(self) -> None:
        """朝向返航原点，并由返航局部规划器执行直达避障。"""
        vehicle = self.vehicle
        return_planner = self.return_safety_planner
        if return_planner is None:
            return_planner = self.local_planner
        if self.semantic_navigator is not None:
            configure = getattr(return_planner, "set_navigation_plane", None)
            if callable(configure):
                nav = self.semantic_navigator
                configure(float(vehicle.get_position()[2]), nav.config, nav.global_map)
        origin = self._return_origin()
        position = vehicle.get_position()
        direction = origin[:2] - position[:2]
        nav_config = getattr(self.semantic_navigator, "config", None)
        exploration_config = getattr(self.semantic_navigator, "exploration_config", None)
        tolerance = float(getattr(exploration_config, "home_tolerance",
                                  getattr(nav_config, "arrival_tolerance", .3)))
        # Use the same home acceptance as the pre-landing confirmation. A
        # completed closed loop must not trigger a second centimetre-scale leg.
        if np.all(np.isfinite(position)) and float(np.linalg.norm(direction)) <= tolerance:
            print("--- 已到达起飞点附近，进入悬停、恢复航向与降落流程 ----")
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
        # The position controller requires its terminal command to equal the
        # requested target. Do not pass measured Z while the planner projects
        # that command back onto a different fixed navigation plane.
        plane = getattr(return_planner, "navigation_altitude", None)
        if plane is not None:
            if not math.isfinite(float(plane)):
                raise ValueError("Return navigation altitude must be finite")
            target[2] = float(plane)
        print(
            f"返回本次起飞点 North={origin[0]:.2f} m, "
            f"East={origin[1]:.2f} m……"
        )
        vehicle.move_waypoint_guided(
            target,
            yaw=return_yaw,
            face_direction=True,
            local_planner=return_planner,
        )

    def _return_on_global_path(self) -> None:
        """语义任务使用滚动安全航点返航；普通任务使用全局返航路线。"""
        if self.semantic_navigator is not None:
            return self._return_on_certified_path()
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
            vehicle.move_waypoint_guided(
                waypoint,
                yaw=segment_yaw,
                face_direction=False,
                local_planner=self.return_safety_planner,
                pass_through=index < len(path) - 1,
            )

    def _return_on_certified_path(self):
        """Return on the same observed body-clearance grid used for exploration."""
        from navigation.route import NavigationGrid
        from navigation.waypoint_cursor import WaypointCursor
        from navigation.mission import _FlightGuard
        from types import SimpleNamespace

        nav, vehicle = self.semantic_navigator, self.vehicle
        planner = self.return_safety_planner
        if planner is None:
            raise RuntimeError("Certified return requires a local safety planner")
        target = vehicle.get_position().copy()
        target[:2] = self._return_origin()[:2]
        altitude = float(target[2])
        configure = getattr(planner, "set_navigation_plane", None)
        if callable(configure):
            configure(altitude, nav.config, nav.global_map)
        # Reuse exploration's certified direct-segment/braking checks without
        # its expired exploration deadline or semantic-detection requirement.
        context = SimpleNamespace(
            phase="returning",
            local_planner=planner, local_map=nav.local_map, global_map=nav.global_map,
            config=nav.config, altitude=altitude, vehicle=vehicle,
            _check_health=nav._check_maps, _check_maps=nav._check_maps,
            _sample_position=lambda current: None,
            _event=getattr(nav, "_event", lambda *args, **kwargs: None),
        )
        if hasattr(nav, "output_path"):
            context.output_path = nav.output_path
        guard = _FlightGuard(context)
        map_revision = getattr(nav.global_map, 'revision', 0)
        path = None
        progress_time = time.monotonic()
        progress_endpoint = None
        best_distance = None
        aligned_endpoint = None
        start_blocked_since = None
        height_recovery_attempted = False
        holding_for_route = False
        cursor = WaypointCursor()
        tolerance = min(.1, nav.config.arrival_tolerance)
        home_radius = float(nav.config.arrival_tolerance)
        settling_since = None
        try:
            while True:
                current = vehicle.get_position()
                if not np.isfinite(current).all():
                    raise RuntimeError("Return position is not finite")
                if np.linalg.norm(current[:2] - target[:2]) <= home_radius:
                    velocity = vehicle.get_velocity()
                    if np.isfinite(velocity).all() and np.linalg.norm(velocity) <= .05:
                        print(f"--- 已到起点附近（半径 {home_radius:.2f} m），准备 PX4 降落 ----")
                        break
                    if settling_since is None:
                        settling_since = time.monotonic()
                        vehicle.hover()
                    if time.monotonic() - settling_since >= nav.config.leg_timeout:
                        raise TimeoutError("Return reached home area but did not settle")
                    vehicle.wait(.2)
                    continue
                settling_since = None
                nav._check_maps()
                current = vehicle.get_position()
                now = time.monotonic()
                grid = NavigationGrid.from_global(nav.global_map.snapshot(), altitude, nav.config,
                                                  nav.local_map.snapshot(), current)
                revision = getattr(nav.global_map, 'revision', 0)
                if revision != map_revision:
                    path = None
                    cursor = WaypointCursor()
                    map_revision = revision
                if path is not None:
                    while (len(path) > 2 and np.linalg.norm(current - path[1]) <= tolerance
                           and np.linalg.norm(vehicle.get_velocity()) <= .05):
                        path = path[1:]
                    path = [current, *path[1:]]
                    if not all(grid.segment_free(a, b) for a, b in zip(path, path[1:])):
                        path = None
                if path is None:
                    # Keep the first hold setpoint while waiting for a route.
                    # Re-capturing measured position each retry makes the target
                    # follow drift/residual motion, including upward motion.
                    if not holding_for_route:
                        vehicle.hover()
                        holding_for_route = True
                    path = grid.path(current, target)
                    if path is None:
                        if not height_recovery_attempted:
                            height_recovery_attempted = True
                            try:
                                restored = self._restore_return_height(nav, target)
                            except (RuntimeError, TimeoutError) as error:
                                current = vehicle.get_position()
                                guard._diagnostic_context = (nav.local_map.snapshot(), nav.global_map.snapshot(), current, target)
                                guard.capture_failure('return_height_recovery_failed', {
                                    'reason': str(error), 'current_ned': current.tolist(),
                                    'nominal_altitude': getattr(nav, '_nominal_altitude', None),
                                })
                                raise
                            if restored is not None:
                                altitude = restored
                                target[2] = altitude
                                context.altitude = altitude
                                if getattr(nav, 'memory', None) is not None:
                                    nav.memory.flight_altitude = altitude
                                if callable(configure):
                                    configure(altitude, nav.config, nav.global_map)
                                cursor = WaypointCursor()
                                aligned_endpoint = None
                                progress_endpoint = None
                                progress_time = time.monotonic()
                                continue
                        start_free = grid.cell(current) in grid.free
                        home_free = grid.cell(target) in grid.free
                        reason = ("return_start_not_free" if not start_free else
                                  "return_home_not_free" if not home_free else "return_route_disconnected")
                        # A rolling map update may temporarily invalidate the
                        # start. Hold and rebuild from fresh evidence, as during
                        # exploration, instead of aborting on the first frame.
                        if not start_free:
                            if start_blocked_since is None:
                                start_blocked_since = now
                                print("--- 返航起点暂不可通行：悬停等待地图更新并复查 ----")
                            wait_budget = min(nav.config.leg_timeout,
                                              2 * getattr(nav.config, "map_timeout", 3.0))
                            if now - start_blocked_since < wait_budget:
                                vehicle.wait(.2)
                                continue
                        detail = {"reason": reason, "start_free": start_free, "home_free": home_free,
                                  "current_ned": current.tolist(), "home_target_ned": target.tolist(),
                                  "start_recheck_seconds": (0.0 if start_blocked_since is None
                                                            else now-start_blocked_since)}
                        guard._diagnostic_context = (nav.local_map.snapshot(), nav.global_map.snapshot(), current, target)
                        guard.capture_failure(reason, detail)
                        raise RuntimeError(f"Return A* found no route with observed body clearance: {reason}; "
                                           f"current={current.round(3).tolist()}, home={target.round(3).tolist()}")
                    print(f"--- 返航 A* 安全路线：{len(path)-1} 个航点（与探索相同净空规则） ----")
                    start_blocked_since = None
                    aligned_endpoint = None
                    cursor = WaypointCursor()
                    nav.active_route = path
                    if callable(getattr(nav, "checkpoint", None)):
                        nav.checkpoint(nav)
                holding_for_route = False
                endpoint = np.asarray(path[1])
                distance = float(np.linalg.norm(endpoint-current))
                if progress_endpoint is None or not np.array_equal(progress_endpoint, endpoint):
                    progress_endpoint, best_distance = endpoint.copy(), distance
                    progress_time = now
                elif distance <= best_distance - min(.2, tolerance):
                    best_distance, progress_time = distance, now
                if now-progress_time >= nav.config.leg_timeout:
                    status = guard.last_status or getattr(getattr(planner, "_cached_plan", None), "status", "unknown")
                    detail = {"reason": "no_waypoint_progress", "local_status": status,
                              "current_ned": current.tolist(), "waypoint_ned": endpoint.tolist(),
                              "seconds_without_progress": now-progress_time,
                              "controller": getattr(vehicle, "waypoint_control_state", None)}
                    guard._diagnostic_context = (nav.local_map.snapshot(), nav.global_map.snapshot(), current, endpoint)
                    guard.capture_failure("return_stalled", detail)
                    raise TimeoutError(f"Return stalled: local_status={status}, current={current.round(3).tolist()}, "
                                       f"waypoint={endpoint.round(3).tolist()}, controller={detail['controller']}")
                direction = endpoint - current
                yaw = math.atan2(float(direction[1]), float(direction[0]))
                if aligned_endpoint is None or not np.array_equal(endpoint, aligned_endpoint):
                    vehicle.hover()
                    vehicle.align_yaw(yaw)
                    aligned_endpoint = endpoint.copy()
                    segment_yaw = yaw
                    # A deliberate turn is not a translational stall.
                    progress_time = time.monotonic()
                # Preserve A* corners; only shorten this one certified segment.
                local_goal, _ = cursor.select(
                    current, endpoint, max(2.4, float(nav.config.leg_length)), tolerance,
                    pass_radius=nav.config.arrival_tolerance,
                    z_tolerance=getattr(getattr(vehicle, "config", None), "guided_z_tolerance", tolerance)
                )
                vehicle.move_waypoint_guided(
                    local_goal, yaw=segment_yaw, face_direction=False,
                    local_planner=guard, pass_through=cursor.intermediate,
                    xy_tolerance=(home_radius if len(path) == 2 else
                                  nav.config.arrival_tolerance if cursor.intermediate else tolerance),
                    control_horizon=1.0, timeout=nav.config.leg_timeout,
                )
        except Exception as error:
            # Save collision context before normal cleanup can obscure the last
            # waypoint command. Failure of diagnostics must not mask the cause.
            try:
                vehicle.hover()
                guard._diagnostic_context = (nav.local_map.snapshot(), nav.global_map.snapshot(),
                                             vehicle.get_position(), target)
                guard.capture_failure("return_exception", {
                    "reason": type(error).__name__, "message": str(error),
                    "flight_context": getattr(error, "flight_context", None),
                    "planning_altitude": altitude,
                }, terminal=True)
            except Exception as diagnostic_error:
                print(f"--- 返航异常诊断保存失败：{diagnostic_error} ----")
            raise
        finally:
            vehicle.hover()
            guard.close_diagnostics()

    def _restore_return_height(self, nav, home):
        """Undo exploration's ascent only with a known route and clear descent."""
        from navigation.route import NavigationGrid
        from navigation.height_clearance import vertical_corridor_free
        vehicle = self.vehicle
        current = vehicle.get_position()
        nominal = getattr(nav, '_nominal_altitude', None)
        if nominal is None or not np.isfinite(nominal) or nominal-current[2] < .1:
            return None
        destination = current.copy()
        destination[2] = float(nominal)
        target = np.asarray(home).copy()
        target[2] = float(nominal)
        local = nav.local_map.snapshot()
        if not vertical_corridor_free(local, current, destination,
                                      nav.config.clearance_radius, nav.config.vertical_half_extent):
            return None
        grid = NavigationGrid.from_global(nav.global_map.snapshot(), nominal, nav.config, local, current)
        if grid.path(destination, target) is None:
            return None
        print(f'--- 当前高度无返航通路：垂直净空与原高度 A* 路线已确认，缓慢恢复至 NED z={nominal:.3f} ----')
        deadline = time.monotonic()+nav.config.leg_timeout
        vehicle.hover()
        try:
            while time.monotonic() < deadline:
                nav._check_maps()
                current = vehicle.get_position()
                if not np.isfinite(current).all():
                    raise RuntimeError('Return height recovery position is not finite')
                velocity = vehicle.get_velocity()
                if not np.isfinite(velocity).all():
                    raise RuntimeError('Return height recovery velocity is not finite')
                if np.linalg.norm(velocity[:2]) > .05:
                    vehicle.hover()
                    vehicle.wait(.1)
                    continue
                if abs(current[2]-nominal) <= .05:
                    nav.altitude = float(current[2])
                    return nav.altitude
                destination[:2] = current[:2]
                if not vertical_corridor_free(nav.local_map.snapshot(), current, destination,
                                              nav.config.clearance_radius, nav.config.vertical_half_extent):
                    raise RuntimeError('Return height recovery stopped: vertical corridor unknown or blocked')
                vehicle.raise_if_collision(include_ground=True)
                vehicle.send_velocity([0., 0., float(np.clip(nominal-current[2], -.15, .15))])
                vehicle.wait(.05)
            raise TimeoutError('Return height recovery timed out')
        finally:
            vehicle.hover()

    def _return_origin(self) -> np.ndarray:
        if self._return_origin_ned is None:
            raise RuntimeError("Return origin has not been recorded.")
        return np.asarray(self._return_origin_ned, dtype=float).copy()
