"""不依赖 ROS 的 EGO 风格三维局部轨迹规划器。

保留 EGO-Planner 的关键结构：滚动局部目标、碰撞段 A* 拓扑种子、
均匀三次 B 样条，以及平滑性/障碍距离/动力学可行性联合优化。实现直接
消费项目已有的 NED 占据栅格，并输出现有飞控可跟踪的短期 NED 目标。
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np

from .local_avoidance_planner import (
    OCCUPIED,
    UNKNOWN,
    LocalAvoidancePlanner,
    LocalPlan,
)


@dataclass(frozen=True)
class UniformCubicBSpline:
    """带端点重复控制点的均匀三次 B 样条。"""

    control_points: np.ndarray
    knot_interval: float

    def __post_init__(self) -> None:
        points = np.asarray(self.control_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
            raise ValueError("control_points must have shape (N, 3), N >= 4")
        if not math.isfinite(self.knot_interval) or self.knot_interval <= 0.0:
            raise ValueError("knot_interval must be positive and finite")
        object.__setattr__(self, "control_points", points.copy())

    @property
    def duration(self) -> float:
        return (len(self.control_points) - 3) * float(self.knot_interval)

    def evaluate(self, trajectory_time: float) -> np.ndarray:
        interval = float(self.knot_interval)
        scaled = float(np.clip(trajectory_time, 0.0, self.duration)) / interval
        segment = min(int(math.floor(scaled)), len(self.control_points) - 4)
        u = 1.0 if trajectory_time >= self.duration else scaled - segment
        u2 = u * u
        u3 = u2 * u
        basis = np.asarray([
            (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0,
            (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0,
            (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0,
            u3 / 6.0,
        ])
        return basis @ self.control_points[segment:segment + 4]

    def evaluate_velocity(self, trajectory_time: float) -> np.ndarray:
        """Evaluate the analytic first derivative in world NED."""
        interval = float(self.knot_interval)
        scaled = float(np.clip(trajectory_time, 0.0, self.duration)) / interval
        segment = min(int(math.floor(scaled)), len(self.control_points) - 4)
        u = 1.0 if trajectory_time >= self.duration else scaled - segment
        basis = np.asarray([
            -0.5 * (1.0 - u) ** 2,
            -2.0 * u + 1.5 * u * u,
            0.5 + u - 1.5 * u * u,
            0.5 * u * u,
        ]) / interval
        return basis @ self.control_points[segment:segment + 4]

    def evaluate_acceleration(self, trajectory_time: float) -> np.ndarray:
        """Evaluate the analytic second derivative in world NED."""
        interval = float(self.knot_interval)
        scaled = float(np.clip(trajectory_time, 0.0, self.duration)) / interval
        segment = min(int(math.floor(scaled)), len(self.control_points) - 4)
        u = 1.0 if trajectory_time >= self.duration else scaled - segment
        basis = np.asarray([
            1.0 - u,
            -2.0 + 3.0 * u,
            1.0 - 3.0 * u,
            u,
        ]) / (interval * interval)
        return basis @ self.control_points[segment:segment + 4]

    def sample(self, step: float) -> np.ndarray:
        if step <= 0.0:
            raise ValueError("step must be greater than zero")
        count = max(2, int(math.ceil(self.duration / step)) + 1)
        times = np.linspace(0.0, self.duration, count)
        return np.asarray([self.evaluate(value) for value in times])


class EgoLocalPlanner(LocalAvoidancePlanner):
    """基于项目本地占据地图的无 ROS EGO 风格滚动规划器。"""

    _NEIGHBORS_3D = tuple(
        (dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz))
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if dx or dy or dz
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trajectory: UniformCubicBSpline | None = None
        self._trajectory_started_at = 0.0
        self._measured_velocity = np.zeros(3, dtype=np.float64)
        self._measured_acceleration = np.zeros(3, dtype=np.float64)
        self._planning_start_velocity = np.zeros(3, dtype=np.float64)
        self._planning_start_acceleration = np.zeros(3, dtype=np.float64)

    def set_motion_state(self, velocity_ned, acceleration_ned=None) -> None:
        """Update measured derivatives without changing the LocalPlan API."""
        velocity = np.asarray(velocity_ned, dtype=np.float64)
        acceleration = (
            np.zeros(3, dtype=np.float64)
            if acceleration_ned is None
            else np.asarray(acceleration_ned, dtype=np.float64)
        )
        if velocity.shape != (3,) or acceleration.shape != (3,):
            raise ValueError("Motion state must contain three-element NED vectors.")
        if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(acceleration)):
            raise ValueError("Motion state must be finite.")
        self._measured_velocity = velocity.copy()
        self._measured_acceleration = acceleration.copy()

    def _start_derivatives(self, now):
        trajectory = self._trajectory
        if trajectory is not None:
            elapsed = max(0.0, float(now) - self._trajectory_started_at)
            if elapsed < trajectory.duration:
                return (
                    trajectory.evaluate_velocity(elapsed),
                    trajectory.evaluate_acceleration(elapsed),
                )
        return self._measured_velocity.copy(), self._measured_acceleration.copy()

    def _make_plan(self, current_position_ned, global_target_ned, now=None):
        now = time.monotonic() if now is None else float(now)
        period = 1.0 / float(self.config.replan_rate)
        if self._cached_plan is not None and now - self._last_plan_time < period:
            return self._cached_plan

        snapshot = self.grid.snapshot()
        self._last_plan_time = now
        timestamp = float(snapshot.timestamp)
        if timestamp <= 0.0:
            self._trajectory = None
            return self._cache(self._stop_plan("waiting_for_map"))
        if timestamp != self._last_map_timestamp:
            self._last_map_timestamp = timestamp
            self._last_map_update_time = now
        if (
            self._last_map_update_time is None
            or now - self._last_map_update_time > float(self.config.map_timeout)
        ):
            self._trajectory = None
            return self._cache(self._stop_plan("stale_map"))

        current = np.asarray(current_position_ned, dtype=np.float64)
        target = np.asarray(global_target_ned, dtype=np.float64)
        if current.shape != (3,) or target.shape != (3,):
            raise ValueError("Positions must be three-element NED vectors.")

        forward_clearance = float(snapshot.forward_clearance)
        map_distance = self._nearest_obstacle_distance(snapshot, current, target)
        self.last_map_obstacle_distance = map_distance
        self.last_forward_clearance = forward_clearance
        obstacle_distance = min(map_distance, forward_clearance)
        if (
            forward_clearance < float(self.config.emergency_stop_distance)
            or map_distance < float(self.config.map_emergency_stop_distance)
        ):
            self._trajectory = None
            return self._cache(self._stop_plan("emergency_stop", obstacle_distance))

        (
            self._planning_start_velocity,
            self._planning_start_acceleration,
        ) = self._start_derivatives(now)

        blocked = np.asarray(snapshot.inflated_occupied, dtype=bool).copy()
        if self.config.unknown_is_occupied:
            blocked |= np.asarray(snapshot.states) == UNKNOWN
        start = self.world_to_grid(snapshot, current)
        if not self._inside_3d(blocked, start):
            self._trajectory = None
            return self._cache(self._stop_plan("outside_map", obstacle_distance))
        blocked[tuple(start)] = False

        requested_local_goal = self._local_goal_in_map(
            snapshot,
            current,
            target,
        )
        if requested_local_goal is None:
            self._trajectory = None
            return self._cache(self._stop_plan("map_boundary", obstacle_distance))
        requested_goal = self.world_to_grid(snapshot, requested_local_goal)
        goal = self._nearest_free_3d(blocked, requested_goal)
        if goal is None:
            self._trajectory = None
            return self._handle_no_path(obstacle_distance)
        # 目标体素本身可用时保留连续世界坐标，避免每次重规划都跳到
        # 体素中心而使直线路径方向和 yaw 发生周期性摆动。
        local_goal = (
            requested_local_goal
            if np.array_equal(goal, requested_goal)
            else self.grid_to_world(snapshot, goal)
        )

        direct_clear = self._line_is_free_3d(snapshot, blocked, current, local_goal)
        if direct_clear:
            seed_path = self._direct_seed_path(
                snapshot,
                blocked,
                current,
                local_goal,
                now,
            )
            status = "ego_clear"
        else:
            seed_path = self._collision_seed_path(
                snapshot,
                blocked,
                current,
                local_goal,
                now,
            )
            if seed_path is None:
                self._trajectory = None
                return self._handle_no_path(obstacle_distance)
            status = "ego_avoid"

        trajectory = self._build_trajectory(snapshot, blocked, seed_path)
        if trajectory is None:
            self._trajectory = None
            return self._handle_no_path(obstacle_distance)

        self._trajectory = trajectory
        self._trajectory_started_at = now
        check_step = float(self.config.ego_collision_check_step)
        sampled_path = trajectory.sample(check_step)
        target_time = min(
            float(self.config.ego_trajectory_lookahead_time),
            trajectory.duration,
        )
        local_target = trajectory.evaluate(target_time)
        if np.linalg.norm(local_target - current) < 0.05 and len(sampled_path) > 1:
            local_target = sampled_path[min(2, len(sampled_path) - 1)]
        speed_limit = self._clear_speed_limit(obstacle_distance)
        return self._accept_moving_plan(LocalPlan(
            status=status,
            local_target_ned=local_target,
            path_ned=sampled_path,
            speed_limit=speed_limit,
            obstacle_distance=obstacle_distance,
        ))

    @property
    def active_trajectory(self) -> UniformCubicBSpline | None:
        """返回最近一次安全轨迹，供调试、记录或后续控制器升级使用。"""
        return self._trajectory

    def tracking_target(self, now=None) -> np.ndarray | None:
        """返回随轨迹时间向前推进的跟踪点，而不是固定在一次规划结果上。"""
        trajectory = self._trajectory
        if trajectory is None:
            return None
        now = time.monotonic() if now is None else float(now)
        elapsed = max(0.0, now - self._trajectory_started_at)
        target_time = min(
            elapsed + float(self.config.ego_trajectory_lookahead_time),
            trajectory.duration,
        )
        return trajectory.evaluate(target_time)

    def _remaining_trajectory_seed(
        self,
        snapshot,
        blocked,
        current,
        local_goal,
        now,
    ):
        """用上一条轨迹尚未执行的部分初始化本轮，减小重规划跳变。"""
        trajectory = self._trajectory
        if trajectory is None:
            return None
        elapsed = max(0.0, float(now) - self._trajectory_started_at)
        if elapsed >= trajectory.duration:
            return None
        interval = max(
            float(self.config.ego_knot_interval),
            float(self.config.ego_collision_check_step),
        )
        times = np.arange(elapsed + interval, trajectory.duration, interval)
        points = [np.asarray(current, dtype=np.float64)]
        points.extend(trajectory.evaluate(value) for value in times)
        points.append(np.asarray(local_goal, dtype=np.float64))
        seed = np.asarray(points)
        if len(seed) < 2:
            return None
        for first, second in zip(seed[:-1], seed[1:]):
            if not self._line_is_free_3d(snapshot, blocked, first, second):
                return None
        return seed

    def _collision_seed_path(
        self,
        snapshot,
        blocked,
        current,
        local_goal,
        now,
    ):
        """Python fallback: reuse the remainder, then search a complete seed."""
        seed = self._remaining_trajectory_seed(
            snapshot,
            blocked,
            current,
            local_goal,
            now,
        )
        if seed is not None:
            return seed
        start = self.world_to_grid(snapshot, current)
        goal = self.world_to_grid(snapshot, local_goal)
        cell_path = self._astar_3d(blocked, start, goal)
        if not cell_path:
            return None
        cell_path = self._simplify_cells_3d(blocked, cell_path)
        seed = np.asarray([
            self.grid_to_world(snapshot, cell) for cell in cell_path
        ])
        seed[0] = current
        seed[-1] = local_goal
        return seed

    @staticmethod
    def _direct_seed_path(snapshot, blocked, current, local_goal, now):
        del snapshot, blocked, now
        return np.vstack((current, local_goal))

    def _local_goal_in_map(self, snapshot, current, target):
        delta = target - current
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            return target.copy()
        travel = min(distance, float(self.config.planning_lookahead))
        desired = current + delta / distance * travel

        resolution = float(snapshot.resolution)
        margin = float(self.config.map_boundary_margin)
        lower = np.asarray(snapshot.origin_ned) + 0.5 * resolution + margin
        upper = (
            np.asarray(snapshot.origin_ned)
            + (np.asarray(snapshot.states.shape) - 0.5) * resolution
            - margin
        )
        if np.any(current < lower) or np.any(current > upper):
            return None
        if np.all(desired >= lower) and np.all(desired <= upper):
            return desired
        direction = desired - current
        scales = [1.0]
        for axis in range(3):
            if direction[axis] > 1e-12:
                scales.append((upper[axis] - current[axis]) / direction[axis])
            elif direction[axis] < -1e-12:
                scales.append((lower[axis] - current[axis]) / direction[axis])
        scale = min(value for value in scales if value >= 0.0)
        if scale <= 1e-6:
            return None
        return np.clip(current + direction * scale, lower, upper)

    def _build_trajectory(self, snapshot, blocked, seed_path):
        spacing = float(self.config.ego_control_point_spacing)
        reference = self._resample_polyline(seed_path, spacing)
        if len(reference) < 2:
            return None
        padded = np.vstack((
            np.repeat(reference[:1], 3, axis=0),
            reference[1:-1],
            np.repeat(reference[-1:], 3, axis=0),
        ))
        optimized = self._optimize_control_points(snapshot, padded)
        for blend in (1.0, 0.7, 0.4, 0.0):
            points = padded + blend * (optimized - padded)
            interval = self._feasible_knot_interval(points)
            trajectory = UniformCubicBSpline(points, interval)
            if self._trajectory_is_free(snapshot, blocked, trajectory):
                return trajectory
        return None

    def _feasible_knot_interval(self, points):
        """像 EGO 的时间重分配一样放大节点间隔，硬满足速度/加速度上限。"""
        points = np.asarray(points, dtype=np.float64)
        interval = float(self.config.ego_knot_interval)
        if len(points) >= 2:
            max_delta = float(np.max(np.abs(np.diff(points, axis=0))))
            interval = max(
                interval,
                max_delta / float(self.config.cruise_max_speed),
            )
        if len(points) >= 3:
            second = np.diff(points, n=2, axis=0)
            max_second = float(np.max(np.abs(second)))
            interval = max(
                interval,
                math.sqrt(
                    max_second / float(self.config.ego_max_acceleration)
                ),
            )
        return interval

    def _optimize_control_points(self, snapshot, initial):
        points = np.asarray(initial, dtype=np.float64).copy()
        reference = points.copy()
        states = np.asarray(snapshot.states)
        origin = np.asarray(snapshot.origin_ned, dtype=np.float64)
        resolution = float(snapshot.resolution)
        search_radius = max(
            1,
            int(math.ceil(float(self.config.ego_clearance) / resolution)) + 1,
        )
        fixed = np.zeros(len(points), dtype=bool)
        fixed[:3] = True
        fixed[-3:] = True
        dt = float(self.config.ego_knot_interval)
        max_vel = float(self.config.cruise_max_speed)
        max_acc = float(self.config.ego_max_acceleration)

        for iteration in range(int(self.config.ego_optimization_iterations)):
            gradient = np.zeros_like(points)

            # EGO 使用三阶差分的 jerk 代价约束轨迹平滑性。
            for index in range(len(points) - 3):
                jerk = (
                    points[index + 3] - 3.0 * points[index + 2]
                    + 3.0 * points[index + 1] - points[index]
                )
                value = 2.0 * float(self.config.ego_smoothness_weight) * jerk
                gradient[index] -= value
                gradient[index + 1] += 3.0 * value
                gradient[index + 2] -= 3.0 * value
                gradient[index + 3] += value

            movable = np.flatnonzero(~fixed)
            for index in movable:
                nearest = self._nearby_obstacle_vector(
                    points[index],
                    states,
                    origin,
                    resolution,
                    search_radius,
                )
                if nearest is None:
                    continue
                offset, distance = nearest
                error = float(self.config.ego_clearance) - distance
                if error > 0.0:
                    direction = offset / max(distance, 1e-9)
                    gradient[index] -= (
                        3.0
                        * float(self.config.ego_collision_weight)
                        * error * error * direction
                    )

            feasibility = float(self.config.ego_feasibility_weight)
            for index in range(len(points) - 1):
                velocity = (points[index + 1] - points[index]) / dt
                excess = np.maximum(np.abs(velocity) - max_vel, 0.0)
                value = 2.0 * feasibility * np.sign(velocity) * excess / dt
                gradient[index] -= value
                gradient[index + 1] += value
            for index in range(len(points) - 2):
                acceleration = (
                    points[index + 2] - 2.0 * points[index + 1] + points[index]
                ) / (dt * dt)
                excess = np.maximum(np.abs(acceleration) - max_acc, 0.0)
                value = (
                    2.0 * feasibility * np.sign(acceleration) * excess
                    / (dt * dt)
                )
                gradient[index] += value
                gradient[index + 1] -= 2.0 * value
                gradient[index + 2] += value

            gradient += (
                2.0 * float(self.config.ego_fitness_weight)
                * (points - reference)
            )
            gradient[fixed] = 0.0
            norm = float(np.linalg.norm(gradient))
            if norm < 1e-7:
                break
            gradient /= max(1.0, norm / math.sqrt(3.0 * len(points)))
            step = float(self.config.ego_optimization_step) / math.sqrt(iteration + 1.0)
            points -= step * gradient
        return points

    @staticmethod
    def _nearby_obstacle_vector(
        point,
        states,
        origin,
        resolution,
        radius,
    ):
        """只搜索安全距离附近体素，避免每次迭代扫描整张三维地图。"""
        center = np.floor((point - origin) / resolution).astype(int)
        lower = np.maximum(center - radius, 0)
        upper = np.minimum(center + radius + 1, states.shape)
        if np.any(lower >= upper):
            return None
        region = states[
            lower[0]:upper[0],
            lower[1]:upper[1],
            lower[2]:upper[2],
        ]
        local_indices = np.argwhere(region == OCCUPIED)
        if not len(local_indices):
            return None
        indices = local_indices + lower
        obstacle_points = origin + (indices.astype(np.float64) + 0.5) * resolution
        offsets = point - obstacle_points
        squared = np.einsum("ij,ij->i", offsets, offsets)
        nearest = int(np.argmin(squared))
        return offsets[nearest], math.sqrt(max(float(squared[nearest]), 1e-18))

    def _trajectory_is_free(self, snapshot, blocked, trajectory):
        samples = trajectory.sample(float(self.config.ego_collision_check_step))
        indices = np.asarray([self.world_to_grid(snapshot, point) for point in samples])
        for index in indices:
            if not self._inside_3d(blocked, index) or blocked[tuple(index)]:
                return False
        return True

    @staticmethod
    def _resample_polyline(path, spacing):
        path = np.asarray(path, dtype=np.float64)
        if len(path) < 2:
            return path.copy()
        lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        total = float(cumulative[-1])
        if total <= 1e-9:
            return path[:1].copy()
        count = max(2, int(math.ceil(total / spacing)) + 1)
        distances = np.linspace(0.0, total, count)
        result = []
        segment = 0
        for distance in distances:
            while segment < len(lengths) - 1 and distance > cumulative[segment + 1]:
                segment += 1
            length = lengths[segment]
            ratio = 0.0 if length <= 1e-12 else (
                (distance - cumulative[segment]) / length
            )
            result.append(path[segment] + ratio * (path[segment + 1] - path[segment]))
        return np.asarray(result)

    def _astar_3d(self, blocked, start, goal):
        start_t = tuple(map(int, start))
        goal_t = tuple(map(int, goal))
        queue = [(0.0, 0.0, start_t)]
        parent = {start_t: None}
        g_score = {start_t: 0.0}
        expanded = 0
        while queue and expanded < int(self.config.max_search_nodes):
            _, cost, current = heapq.heappop(queue)
            if cost != g_score.get(current):
                continue
            expanded += 1
            if current == goal_t:
                path = []
                while current is not None:
                    path.append(np.asarray(current, dtype=int))
                    current = parent[current]
                return path[::-1]
            for dx, dy, dz, edge_cost in self._NEIGHBORS_3D:
                nxt = (current[0] + dx, current[1] + dy, current[2] + dz)
                if (
                    nxt[0] < 0 or nxt[1] < 0 or nxt[2] < 0
                    or nxt[0] >= blocked.shape[0]
                    or nxt[1] >= blocked.shape[1]
                    or nxt[2] >= blocked.shape[2]
                    or blocked[nxt]
                    or not self._transition_is_free(blocked, current, nxt)
                ):
                    continue
                new_cost = cost + edge_cost
                if new_cost >= g_score.get(nxt, math.inf):
                    continue
                g_score[nxt] = new_cost
                parent[nxt] = current
                heuristic = math.dist(nxt, goal_t)
                heapq.heappush(queue, (new_cost + heuristic, new_cost, nxt))
        return []

    def _simplify_cells_3d(self, blocked, cells):
        if len(cells) <= 2:
            return cells
        result = [cells[0]]
        anchor = 0
        while anchor < len(cells) - 1:
            candidate = len(cells) - 1
            while candidate > anchor + 1:
                if self._cell_line_is_free_3d(blocked, cells[anchor], cells[candidate]):
                    break
                candidate -= 1
            result.append(cells[candidate])
            anchor = candidate
        return result

    def _line_is_free_3d(self, snapshot, blocked, start_world, end_world):
        return self._cell_line_is_free_3d(
            blocked,
            self.world_to_grid(snapshot, start_world),
            self.world_to_grid(snapshot, end_world),
        )

    @staticmethod
    def _cell_line_is_free_3d(blocked, start, end):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        steps = max(1, int(np.max(np.abs(end - start))) * 2)
        indices = np.rint(np.linspace(start, end, steps + 1)).astype(int)
        previous = None
        for index in indices:
            if (
                np.any(index < 0)
                or np.any(index >= blocked.shape)
                or blocked[tuple(index)]
            ):
                return False
            if (
                previous is not None
                and not EgoLocalPlanner._transition_is_free(
                    blocked,
                    previous,
                    index,
                )
            ):
                return False
            previous = index
        return True

    @staticmethod
    def _transition_is_free(blocked, start, end):
        """检查对角移动跨过的面/棱相邻体素，禁止从障碍角落穿越。"""
        start = np.asarray(start, dtype=int)
        end = np.asarray(end, dtype=int)
        delta = end - start
        axes = np.flatnonzero(delta)
        if np.any(np.abs(delta) > 1):
            return False
        # 对于二维或三维对角步，所有非空真子集对应的中间体素都要空闲。
        for mask in range(1, (1 << len(axes)) - 1):
            intermediate = start.copy()
            for bit, axis in enumerate(axes):
                if mask & (1 << bit):
                    intermediate[axis] += delta[axis]
            if blocked[tuple(intermediate)]:
                return False
        return True

    @staticmethod
    def _inside_3d(array, index):
        index = np.asarray(index)
        return bool(np.all(index >= 0) and np.all(index < array.shape))

    @staticmethod
    def _nearest_free_3d(blocked, goal):
        goal = np.asarray(goal, dtype=int)
        if EgoLocalPlanner._inside_3d(blocked, goal) and not blocked[tuple(goal)]:
            return goal
        for radius in range(1, 6):
            lower = goal - radius
            upper = goal + radius
            candidates = []
            for x in range(lower[0], upper[0] + 1):
                for y in range(lower[1], upper[1] + 1):
                    for z in range(lower[2], upper[2] + 1):
                        index = np.asarray((x, y, z))
                        if (
                            EgoLocalPlanner._inside_3d(blocked, index)
                            and not blocked[x, y, z]
                        ):
                            candidates.append(index)
            if candidates:
                return min(candidates, key=lambda value: np.linalg.norm(value - goal))
        return None
