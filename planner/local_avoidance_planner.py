"""基于滑动局部占据栅格的固定高度A*主动避障。"""

from __future__ import annotations

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from config.planner_config import PlannerConfig


UNKNOWN = -1
OCCUPIED = 1


@dataclass(frozen=True)
class LocalPlan:
    """一次局部规划结果。"""

    status: str
    local_target_ned: np.ndarray | None
    path_ned: np.ndarray
    speed_limit: float
    obstacle_distance: float

    @property
    def can_move(self) -> bool:
        return self.local_target_ned is not None and self.status in {
            "clear",
            "avoid",
            "ego_clear",
            "ego_avoid",
            "sensor_clear_fallback",
            "depth_guard_clear",
        }


class LocalAvoidancePlanner:
    """在当前飞行高度投影局部地图，并生成一个短期NED航点。"""

    def __init__(
        self,
        occupancy_grid,
        config: PlannerConfig | None = None,
        visualizer=None,
        visualization_name: str = "local",
    ):
        self.grid = occupancy_grid
        self.config = config or PlannerConfig()
        self.visualizer = visualizer
        self.visualization_name = str(visualization_name)
        self._last_plan_time = -math.inf
        self._last_map_timestamp = None
        self._last_map_update_time = None
        self._cached_plan = None
        self._consecutive_no_path = 0
        self._consecutive_clear_path = 0
        self._last_avoidance_plan = None
        self._active_global_target = None
        self.last_map_obstacle_distance = math.inf
        self.last_forward_clearance = math.inf

    def plan(
        self,
        current_position_ned,
        global_target_ned,
        now: float | None = None,
    ) -> LocalPlan:
        target = np.asarray(global_target_ned, dtype=np.float64)
        if target.shape == (3,) and (
            self._active_global_target is None
            or np.linalg.norm(target - self._active_global_target) > 1e-6
        ):
            self._active_global_target = target.copy()
            self._last_plan_time = -math.inf
            self._cached_plan = None
            self._consecutive_no_path = 0
            self._consecutive_clear_path = 0
            self._last_avoidance_plan = None
        result = self._make_plan(
            current_position_ned,
            global_target_ned,
            now=now,
        )
        if (
            self.visualizer is not None
            and self.visualizer.should_save_local(
                self.visualization_name,
                now=now,
            )
        ):
            self.visualizer.submit_local(
                self.grid.snapshot(),
                current_position_ned,
                global_target_ned,
                result,
                inflation_radius=float(getattr(
                    getattr(self.grid, "config", None),
                    "inflation_radius",
                    0.3,
                )),
                source_name=self.visualization_name,
                now=now,
            )
        return result

    def _make_plan(
        self,
        current_position_ned,
        global_target_ned,
        now: float | None = None,
    ) -> LocalPlan:
        """返回可跟踪的局部目标；不安全时返回不可移动状态。"""
        now = time.monotonic() if now is None else float(now)
        period = 1.0 / float(self.config.replan_rate)
        if (
            self._cached_plan is not None
            and now - self._last_plan_time < period
        ):
            return self._cached_plan

        snapshot = self.grid.snapshot()
        self._last_plan_time = now
        timestamp = float(snapshot.timestamp)
        if timestamp <= 0.0:
            self._consecutive_no_path = 0
            return self._cache(self._stop_plan("waiting_for_map"))
        if timestamp != self._last_map_timestamp:
            self._last_map_timestamp = timestamp
            self._last_map_update_time = now
        if (
            self._last_map_update_time is None
            or now - self._last_map_update_time
            > float(self.config.map_timeout)
        ):
            self._consecutive_no_path = 0
            return self._cache(self._stop_plan("stale_map"))

        current = np.asarray(current_position_ned, dtype=np.float64)
        target = np.asarray(global_target_ned, dtype=np.float64)
        if current.shape != (3,) or target.shape != (3,):
            raise ValueError("Positions must be three-element NED vectors.")

        forward_clearance = float(
            getattr(snapshot, "forward_clearance", math.inf)
        )
        if self.config.navigation_mode == "depth_guard":
            self.last_map_obstacle_distance = math.inf
            self.last_forward_clearance = forward_clearance
            if not math.isfinite(forward_clearance):
                return self._cache(self._stop_plan(
                    "invalid_front_depth",
                ))
            if forward_clearance < float(
                self.config.emergency_stop_distance
            ):
                return self._cache(self._stop_plan(
                    "emergency_stop",
                    forward_clearance,
                ))
            local_target = self._direct_local_target(current, target)
            return self._accept_moving_plan(LocalPlan(
                status="depth_guard_clear",
                local_target_ned=local_target,
                path_ned=np.vstack((current, local_target)),
                speed_limit=self._clear_speed_limit(
                    forward_clearance
                ),
                obstacle_distance=forward_clearance,
            ))

        blocked = self._project_blocked(snapshot, current[2])
        map_obstacle_distance = self._nearest_obstacle_distance(
            snapshot,
            current,
            target,
        )
        self.last_map_obstacle_distance = map_obstacle_distance
        self.last_forward_clearance = forward_clearance
        obstacle_distance = min(
            map_obstacle_distance,
            forward_clearance,
        )
        if (
            forward_clearance
            < float(self.config.emergency_stop_distance)
            or map_obstacle_distance
            < float(self.config.map_emergency_stop_distance)
        ):
            self._consecutive_no_path = 0
            return self._cache(self._stop_plan(
                "emergency_stop",
                obstacle_distance,
            ))

        start = self.world_to_grid(snapshot, current)[:2]
        if not self._inside_2d(blocked, start):
            self._consecutive_no_path = 0
            return self._cache(self._stop_plan(
                "outside_map",
                obstacle_distance,
            ))
        blocked[tuple(start)] = False

        local_goal_world = self._lookahead_goal(current, target)
        raw_goal = self.world_to_grid(snapshot, local_goal_world)[:2]
        raw_goal_inside = self._inside_2d(blocked, raw_goal)
        checked_goal_world = local_goal_world.copy()
        if raw_goal_inside:
            direct_goal = raw_goal
        else:
            checked_goal_world = self._ray_to_map_boundary(
                snapshot,
                current,
                local_goal_world,
                blocked.shape,
            )
            if checked_goal_world is None:
                self._consecutive_no_path = 0
                return self._cache(self._stop_plan(
                    "map_boundary",
                    obstacle_distance,
                ))
            direct_goal = self.world_to_grid(
                snapshot,
                checked_goal_world,
            )[:2]

        # 只有原始局部目标本身可用且整条直线安全时才能直飞。不能先把
        # 被占据的目标挪到旁边，却继续向原始目标方向发送速度。
        direct_path_is_clear = (
            not blocked[tuple(direct_goal)]
            and self._line_is_free(blocked, start, direct_goal)
        )
        if direct_path_is_clear:
            if self._last_avoidance_plan is not None:
                self._consecutive_clear_path += 1
                if self._consecutive_clear_path < int(
                    self.config.clear_path_confirmation_count
                ):
                    reused = self._reuse_avoidance_target(
                        snapshot,
                        blocked,
                        start,
                        current,
                        target,
                        obstacle_distance,
                    )
                    if reused is not None:
                        self._consecutive_no_path = 0
                        self._last_avoidance_plan = reused
                        return self._cache(reused)
            self._consecutive_clear_path = 0
            self._last_avoidance_plan = None
            if raw_goal_inside:
                local_target = self._direct_local_target(current, target)
            else:
                local_target = self._bounded_local_target(
                    current,
                    target,
                    checked_goal_world,
                )
            path = np.vstack((current, local_target))
            return self._accept_moving_plan(LocalPlan(
                status="clear",
                local_target_ned=local_target,
                path_ned=path,
                speed_limit=self._clear_speed_limit(obstacle_distance),
                obstacle_distance=obstacle_distance,
            ))

        self._consecutive_clear_path = 0
        reused = self._reuse_avoidance_target(
            snapshot,
            blocked,
            start,
            current,
            target,
            obstacle_distance,
        )
        if reused is not None:
            return self._accept_moving_plan(reused)

        goal = direct_goal
        goal = self._nearest_reachable_free(
            blocked,
            start,
            goal,
            float(self.config.reachable_goal_tolerance)
            / float(snapshot.resolution),
        )
        cell_path = [] if goal is None else self._astar(blocked, start, goal)
        if not cell_path:
            fallback = self._sensor_clear_fallback(
                current,
                target,
                map_obstacle_distance,
                forward_clearance,
                obstacle_distance,
            )
            if fallback is not None:
                return fallback
            return self._handle_no_path(obstacle_distance)
        cell_path = self._smooth_path(blocked, cell_path)
        path = np.asarray([
            self.grid_to_world(snapshot, np.asarray(
                [cell[0], cell[1], start_z],
                dtype=int,
            ))
            for cell, start_z in (
                (cell, self.world_to_grid(snapshot, current)[2])
                for cell in cell_path
            )
        ])
        path[:, 2] = current[2]
        local_target = self._point_along_path(
            path,
            float(self.config.local_target_distance),
        )
        if not self._advances_toward_target(
            current,
            local_target,
            target,
            float(self.config.minimum_forward_progress),
        ):
            fallback = self._sensor_clear_fallback(
                current,
                target,
                map_obstacle_distance,
                forward_clearance,
                obstacle_distance,
            )
            if fallback is not None:
                return fallback
            return self._handle_no_path(obstacle_distance)
        local_target[2] = self._vertical_local_target(
            current[2],
            target[2],
        )
        return self._accept_moving_plan(LocalPlan(
            status="avoid",
            local_target_ned=local_target,
            path_ned=path,
            speed_limit=min(
                float(self.config.avoidance_max_speed),
                self._speed_limit(obstacle_distance),
            ),
            obstacle_distance=obstacle_distance,
        ))

    @staticmethod
    def world_to_grid(snapshot, point_ned) -> np.ndarray:
        return np.floor(
            (
                np.asarray(point_ned, dtype=np.float64)
                - np.asarray(snapshot.origin_ned, dtype=np.float64)
            )
            / float(snapshot.resolution)
        ).astype(int)

    @staticmethod
    def grid_to_world(snapshot, index) -> np.ndarray:
        return (
            np.asarray(snapshot.origin_ned, dtype=np.float64)
            + (np.asarray(index, dtype=np.float64) + 0.5)
            * float(snapshot.resolution)
        )

    def _project_blocked(self, snapshot, down: float) -> np.ndarray:
        center = self.world_to_grid(
            snapshot,
            [snapshot.origin_ned[0], snapshot.origin_ned[1], down],
        )[2]
        radius = max(
            0,
            int(math.ceil(
                float(self.config.vertical_band)
                / float(snapshot.resolution)
            )),
        )
        low = max(0, center - radius)
        high = min(snapshot.states.shape[2], center + radius + 1)
        if low >= high:
            return np.ones(snapshot.states.shape[:2], dtype=bool)
        blocked = np.any(
            snapshot.inflated_occupied[:, :, low:high],
            axis=2,
        )
        if self.config.unknown_is_occupied:
            blocked |= np.any(
                snapshot.states[:, :, low:high] == UNKNOWN,
                axis=2,
            )
        return blocked

    def _nearest_obstacle_distance(
        self,
        snapshot,
        current,
        target,
    ) -> float:
        """返回目标方向安全走廊内的最近原始障碍距离。"""
        indices = np.argwhere(snapshot.states == OCCUPIED)
        if indices.size == 0:
            return math.inf
        points = (
            np.asarray(snapshot.origin_ned)
            + (indices.astype(np.float64) + 0.5)
            * float(snapshot.resolution)
        )
        relative = points - current
        horizontal_target = np.asarray(target[:2]) - current[:2]
        target_distance = float(np.linalg.norm(horizontal_target))
        if target_distance <= 1e-6:
            return float(np.min(np.linalg.norm(relative, axis=1)))
        direction = horizontal_target / target_distance
        longitudinal = relative[:, :2] @ direction
        lateral = np.abs(
            relative[:, 0] * direction[1]
            - relative[:, 1] * direction[0]
        )
        vertical_limit = (
            float(self.config.vertical_band)
            + float(snapshot.resolution)
        )
        relevant = longitudinal >= 0.0
        relevant &= (
            lateral
            <= float(self.config.safety_corridor_half_width)
        )
        relevant &= np.abs(relative[:, 2]) <= vertical_limit
        if not np.any(relevant):
            return math.inf
        return float(np.min(
            np.linalg.norm(relative[relevant], axis=1)
        ))

    def _lookahead_goal(self, current, target) -> np.ndarray:
        delta = target - current
        horizontal = float(np.linalg.norm(delta[:2]))
        if horizontal <= float(self.config.planning_lookahead):
            result = target.copy()
        else:
            result = current.copy()
            result[:2] += (
                delta[:2] / horizontal
                * float(self.config.planning_lookahead)
            )
        result[2] = current[2]
        return result

    def _direct_local_target(self, current, target) -> np.ndarray:
        delta = target - current
        distance = float(np.linalg.norm(delta))
        limit = float(self.config.local_target_distance)
        if distance <= limit:
            result = target.copy()
        else:
            result = current + delta / distance * limit
        result[2] = self._vertical_local_target(
            current[2],
            target[2],
        )
        return result

    def _ray_to_map_boundary(
        self,
        snapshot,
        current,
        goal,
        shape,
    ) -> np.ndarray | None:
        """返回目标射线与预留余量后的二维地图边界交点。"""
        origin = np.asarray(snapshot.origin_ned[:2], dtype=np.float64)
        resolution = float(snapshot.resolution)
        margin = float(self.config.map_boundary_margin)
        lower = origin + 0.5 * resolution + margin
        upper = (
            origin
            + (np.asarray(shape, dtype=np.float64) - 0.5) * resolution
            - margin
        )
        current_xy = np.asarray(current[:2], dtype=np.float64)
        direction = np.asarray(goal[:2], dtype=np.float64) - current_xy
        if np.linalg.norm(direction) <= 1e-9:
            return None
        if np.any(current_xy < lower) or np.any(current_xy > upper):
            return None

        candidates = []
        for axis in range(2):
            if direction[axis] > 1e-12:
                candidates.append(
                    (upper[axis] - current_xy[axis]) / direction[axis]
                )
            elif direction[axis] < -1e-12:
                candidates.append(
                    (lower[axis] - current_xy[axis]) / direction[axis]
                )
        positive = [value for value in candidates if value > 1e-9]
        if not positive:
            return None
        scale = min(positive)
        result = np.asarray(current, dtype=np.float64).copy()
        result[:2] = current_xy + direction * scale
        result[:2] = np.clip(result[:2], lower, upper)
        result[2] = current[2]
        return result

    def _bounded_local_target(self, current, target, checked_goal) -> np.ndarray:
        """沿全局目标方向生成不超过已验证地图区域的短期航点。"""
        delta = np.asarray(checked_goal[:2]) - np.asarray(current[:2])
        checked_distance = float(np.linalg.norm(delta))
        if checked_distance <= 1e-9:
            result = np.asarray(current, dtype=np.float64).copy()
        else:
            travel = min(
                float(self.config.local_target_distance),
                checked_distance,
            )
            result = np.asarray(current, dtype=np.float64).copy()
            result[:2] += delta / checked_distance * travel
        result[2] = self._vertical_local_target(current[2], target[2])
        return result

    def _reuse_avoidance_target(
        self,
        snapshot,
        blocked,
        start,
        current,
        target,
        obstacle_distance,
    ) -> LocalPlan | None:
        """上一绕障航点仍安全且尚未到达时继续跟踪，抑制路径左右抖动。"""
        previous = self._last_avoidance_plan
        if previous is None or previous.local_target_ned is None:
            return None
        local_target = np.asarray(
            previous.local_target_ned,
            dtype=np.float64,
        ).copy()
        horizontal_distance = float(np.linalg.norm(
            local_target[:2] - np.asarray(current[:2])
        ))
        if horizontal_distance <= float(
            self.config.avoidance_reuse_min_distance
        ):
            return None
        if not self._advances_toward_target(
            current,
            local_target,
            target,
            float(self.config.minimum_forward_progress),
        ):
            return None
        target_cell = self.world_to_grid(snapshot, local_target)[:2]
        if (
            not self._inside_2d(blocked, target_cell)
            or blocked[tuple(target_cell)]
            or not self._line_is_free(blocked, start, target_cell)
        ):
            return None
        local_target[2] = self._vertical_local_target(
            current[2],
            target[2],
        )
        return LocalPlan(
            status="avoid",
            local_target_ned=local_target,
            path_ned=np.vstack((current, local_target)),
            speed_limit=min(
                float(self.config.avoidance_max_speed),
                self._speed_limit(obstacle_distance),
            ),
            obstacle_distance=obstacle_distance,
        )

    @staticmethod
    def _advances_toward_target(
        current,
        candidate,
        target,
        minimum_forward_progress: float = 0.0,
    ) -> bool:
        current = np.asarray(current[:2], dtype=np.float64)
        candidate = np.asarray(candidate[:2], dtype=np.float64)
        target = np.asarray(target[:2], dtype=np.float64)
        target_delta = target - current
        target_distance = float(np.linalg.norm(target_delta))
        if target_distance <= 1e-9:
            return True
        forward_progress = float(
            np.dot(candidate - current, target_delta / target_distance)
        )
        current_distance = float(np.linalg.norm(
            target - current
        ))
        candidate_distance = float(np.linalg.norm(
            target - candidate
        ))
        return (
            forward_progress >= float(minimum_forward_progress)
            and candidate_distance < current_distance - 1e-6
        )

    def _vertical_local_target(
        self,
        current_down: float,
        target_down: float,
    ) -> float:
        error = float(target_down) - float(current_down)
        step = float(self.config.vertical_target_step)
        return float(current_down) + float(np.clip(error, -step, step))

    def _sensor_clear_fallback(
        self,
        current,
        target,
        map_obstacle_distance,
        forward_clearance,
        obstacle_distance,
    ) -> LocalPlan | None:
        """A*假阻塞但两套前向安全证据均通过时低速直行。"""
        required = float(self.config.fallback_min_clearance)
        if (
            not math.isfinite(forward_clearance)
            or map_obstacle_distance
            < float(self.config.fallback_min_map_clearance)
            or forward_clearance < required
        ):
            return None
        local_target = self._direct_local_target(current, target)
        path = np.vstack((current, local_target))
        return self._accept_moving_plan(LocalPlan(
            status="sensor_clear_fallback",
            local_target_ned=local_target,
            path_ned=path,
            speed_limit=float(
                self.config.sensor_clear_fallback_speed
            ),
            obstacle_distance=obstacle_distance,
        ))

    def _speed_limit(self, obstacle_distance: float) -> float:
        maximum = float(self.config.avoidance_max_speed)
        if not math.isfinite(obstacle_distance):
            return maximum
        stop = float(self.config.emergency_stop_distance)
        slow = float(self.config.slowdown_distance)
        ratio = np.clip((obstacle_distance - stop) / (slow - stop), 0.0, 1.0)
        return max(0.1, maximum * float(ratio))

    def _clear_speed_limit(self, obstacle_distance: float) -> float:
        """障碍较远时使用安全巡航上限，接近后进一步减速。"""
        if obstacle_distance >= float(self.config.slowdown_distance):
            return float(self.config.cruise_max_speed)
        return min(
            float(self.config.cruise_max_speed),
            self._speed_limit(obstacle_distance),
        )

    def _handle_no_path(
        self,
        obstacle_distance: float,
    ) -> LocalPlan:
        """对无路结果计数，但任何失败帧都先悬停，绝不盲走旧路径。"""
        self._consecutive_no_path += 1
        confirmation_count = int(
            self.config.no_path_confirmation_count
        )
        status = (
            "transient_no_path"
            if self._consecutive_no_path < confirmation_count
            else "no_path"
        )
        return self._cache(self._stop_plan(
            status,
            obstacle_distance,
        ))

    def _accept_moving_plan(
        self,
        plan: LocalPlan,
    ) -> LocalPlan:
        self._consecutive_no_path = 0
        self._last_avoidance_plan = (
            plan if plan.status == "avoid" else None
        )
        self._consecutive_clear_path = 0
        return self._cache(plan)

    @staticmethod
    def _inside_2d(array, index) -> bool:
        return bool(np.all(index >= 0) and np.all(index < array.shape))

    def _nearest_reachable_free(
        self,
        blocked,
        start,
        goal,
        maximum_goal_distance,
    ):
        """从起点连通区中选择最接近期望局部终点的自由格。"""
        start_t = tuple(map(int, start))
        goal = np.asarray(goal, dtype=int)
        queue = deque([start_t])
        visited = {start_t}
        best = start_t
        best_distance = float(np.linalg.norm(np.asarray(best) - goal))
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.config.allow_diagonal:
            neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        while queue:
            current = queue.popleft()
            current_array = np.asarray(current)
            distance = float(np.linalg.norm(current_array - goal))
            if distance < best_distance:
                best = current
                best_distance = distance
            if distance == 0.0:
                return current_array
            for dx, dy in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if (
                    nxt in visited
                    or nxt[0] < 0 or nxt[1] < 0
                    or nxt[0] >= blocked.shape[0]
                    or nxt[1] >= blocked.shape[1]
                    or blocked[nxt]
                ):
                    continue
                if dx and dy and (
                    blocked[current[0] + dx, current[1]]
                    or blocked[current[0], current[1] + dy]
                ):
                    continue
                visited.add(nxt)
                queue.append(nxt)
        start_distance = float(np.linalg.norm(np.asarray(start_t) - goal))
        if (
            best == start_t
            or best_distance >= start_distance - 1.0
            or best_distance > float(maximum_goal_distance)
        ):
            return None
        return np.asarray(best, dtype=int)

    def _astar(self, blocked, start, goal):
        start_t = tuple(map(int, start))
        goal_t = tuple(map(int, goal))
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0),
            (0, -1, 1.0), (0, 1, 1.0),
        ]
        if self.config.allow_diagonal:
            root_two = math.sqrt(2.0)
            neighbors += [
                (-1, -1, root_two), (-1, 1, root_two),
                (1, -1, root_two), (1, 1, root_two),
            ]
        frontier = [(0.0, start_t)]
        came_from = {start_t: None}
        cost = {start_t: 0.0}
        expanded = 0
        while frontier and expanded < int(self.config.max_search_nodes):
            _, current = heapq.heappop(frontier)
            expanded += 1
            if current == goal_t:
                break
            for dx, dy, step_cost in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if (
                    nxt[0] < 0 or nxt[1] < 0
                    or nxt[0] >= blocked.shape[0]
                    or nxt[1] >= blocked.shape[1]
                    or blocked[nxt]
                ):
                    continue
                if dx and dy:
                    if (
                        blocked[current[0] + dx, current[1]]
                        or blocked[current[0], current[1] + dy]
                    ):
                        continue
                new_cost = cost[current] + step_cost
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    heuristic = math.hypot(
                        goal_t[0] - nxt[0],
                        goal_t[1] - nxt[1],
                    )
                    heapq.heappush(
                        frontier,
                        (new_cost + heuristic, nxt),
                    )
                    came_from[nxt] = current
        if goal_t not in came_from:
            return []
        path = []
        node = goal_t
        while node is not None:
            path.append(node)
            node = came_from[node]
        return list(reversed(path))

    def _smooth_path(self, blocked, path):
        if len(path) <= 2:
            return path
        smoothed = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            candidate = len(path) - 1
            while candidate > anchor + 1:
                if self._line_is_free(
                    blocked,
                    np.asarray(path[anchor]),
                    np.asarray(path[candidate]),
                ):
                    break
                candidate -= 1
            smoothed.append(path[candidate])
            anchor = candidate
        return smoothed

    @staticmethod
    def _line_is_free(blocked, start, goal) -> bool:
        delta = np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)
        count = max(2, int(math.ceil(np.linalg.norm(delta) * 2.0)) + 1)
        samples = np.rint(
            np.asarray(start)[None, :]
            + np.linspace(0.0, 1.0, count)[:, None] * delta[None, :]
        ).astype(int)
        return not bool(np.any(blocked[tuple(samples.T)]))

    @staticmethod
    def _point_along_path(path, distance):
        if len(path) == 1:
            return path[0].copy()
        remaining = float(distance)
        for start, end in zip(path[:-1], path[1:]):
            segment = end - start
            length = float(np.linalg.norm(segment))
            if length >= remaining and length > 0.0:
                return start + segment * (remaining / length)
            remaining -= length
        return path[-1].copy()

    @staticmethod
    def _stop_plan(status, obstacle_distance=math.inf):
        return LocalPlan(
            status=status,
            local_target_ned=None,
            path_ned=np.empty((0, 3), dtype=np.float64),
            speed_limit=0.0,
            obstacle_distance=float(obstacle_distance),
        )

    def _cache(self, plan):
        if not plan.can_move:
            self._last_avoidance_plan = None
            self._consecutive_clear_path = 0
        self._cached_plan = plan
        return plan
