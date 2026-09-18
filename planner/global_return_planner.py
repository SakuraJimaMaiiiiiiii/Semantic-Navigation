"""基于固定世界稀疏占据地图的一次性全局返航 A*。"""

from __future__ import annotations

import heapq
import math

import numpy as np

from config.planner_config import PlannerConfig
from .vertical_projection import ned_height_slice_mask


class GlobalReturnPlanner:
    """在已观测自由区和去程轨迹走廊中生成稳定返航路径。"""

    def __init__(
        self,
        global_map,
        config: PlannerConfig | None = None,
        visualizer=None,
    ):
        self.global_map = global_map
        self.config = config or PlannerConfig()
        self.visualizer = visualizer

    def plan(self, current_ned, goal_ned) -> np.ndarray:
        snapshot = self.global_map.snapshot()
        current = np.asarray(current_ned, dtype=np.float64)
        goal = np.asarray(goal_ned, dtype=np.float64)
        if current.shape != (3,) or goal.shape != (3,):
            raise ValueError("Return endpoints must be three-element NED vectors.")
        if snapshot.timestamp <= 0.0:
            raise RuntimeError("Global navigation memory is not ready.")

        blocked, minimum, resolution, start, goal_cell = (
            self._build_navigation_grid(snapshot, current, goal)
        )
        cells = self._astar(blocked, start, goal_cell)
        if not cells:
            path = self._trace_fallback_path(
                snapshot.trajectory_ned,
                current,
                goal,
            )
            self._submit_visualization(
                snapshot,
                current,
                goal,
                path,
                "recorded_trace",
            )
            return path
        cells = self._smooth(blocked, cells)
        world_cells = np.asarray(cells, dtype=np.float64) + minimum
        path = np.empty((world_cells.shape[0], 3), dtype=np.float64)
        path[:, :2] = (world_cells + 0.5) * resolution
        path[:, 2] = current[2]
        path[0] = current
        path[-1] = goal
        path = self._limit_segment_length(
            path,
            float(self.config.planning_lookahead),
        )
        self._submit_visualization(
            snapshot,
            current,
            goal,
            path,
            "global_astar",
        )
        return path

    def _submit_visualization(
        self,
        snapshot,
        current,
        goal,
        path,
        source_name,
    ) -> None:
        if self.visualizer is None:
            return
        self.visualizer.submit_global(
            snapshot,
            current,
            goal,
            path,
            above_height=float(self.config.global_return_above_height),
            below_height=float(self.config.global_return_below_height),
            source_name=source_name,
        )

    def _trace_fallback_path(self, trajectory, current, goal) -> np.ndarray:
        """地图证据冲突时，沿真实飞过的去程轨迹反向返航。"""
        points = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)
        nearest_current = int(np.argmin(
            np.linalg.norm(points[:, :2] - current[:2], axis=1)
        ))
        nearest_goal = int(np.argmin(
            np.linalg.norm(points[:, :2] - goal[:2], axis=1)
        ))
        lower = min(nearest_current, nearest_goal)
        upper = max(nearest_current, nearest_goal)
        trace = points[lower:upper + 1]
        if nearest_current >= nearest_goal:
            trace = trace[::-1]
        trace = self._simplify_trace(
            trace,
            max(0.3, float(self.config.global_return_trace_radius)),
        )
        trace = self._thin_trace(
            trace,
            max(1.0, float(self.config.local_target_distance)),
        )
        result = np.vstack((current, trace, goal))
        keep = np.ones(len(result), dtype=bool)
        if len(result) > 1:
            keep[1:] = np.linalg.norm(
                np.diff(result[:, :2], axis=0),
                axis=1,
            ) > 1e-6
        result = result[keep]
        result[:, 2] = current[2]
        result[-1] = goal
        result = self._limit_segment_length(
            result,
            float(self.config.planning_lookahead),
        )
        print(
            "--- 全局占据 A* 无路，改用真实去程轨迹反向返航："
            f"{len(result)} 个航点 ----"
        )
        return result

    @staticmethod
    def _limit_segment_length(points, maximum_spacing) -> np.ndarray:
        """将长航段等距切开，使每个目标始终处于局部规划前视范围内。"""
        points = np.asarray(points, dtype=np.float64)
        if len(points) <= 1:
            return points.copy()
        result = [points[0].copy()]
        for start, end in zip(points[:-1], points[1:]):
            horizontal_distance = float(np.linalg.norm(end[:2] - start[:2]))
            segment_count = max(
                1,
                int(math.ceil(horizontal_distance / float(maximum_spacing))),
            )
            for index in range(1, segment_count + 1):
                ratio = index / segment_count
                result.append(start + (end - start) * ratio)
        return np.asarray(result, dtype=np.float64)

    @staticmethod
    def _thin_trace(points, minimum_spacing) -> np.ndarray:
        if len(points) <= 2:
            return points.copy()
        selected = [points[0]]
        for point in points[1:-1]:
            if np.linalg.norm(point[:2] - selected[-1][:2]) >= minimum_spacing:
                selected.append(point)
        selected.append(points[-1])
        return np.asarray(selected, dtype=np.float64)

    @classmethod
    def _simplify_trace(cls, points, tolerance) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if len(points) <= 2:
            return points.copy()
        segment = points[-1, :2] - points[0, :2]
        length = float(np.linalg.norm(segment))
        if length <= 1e-9:
            distances = np.linalg.norm(
                points[1:-1, :2] - points[0, :2],
                axis=1,
            )
        else:
            relative = points[1:-1, :2] - points[0, :2]
            distances = np.abs(
                segment[0] * relative[:, 1]
                - segment[1] * relative[:, 0]
            ) / length
        split = int(np.argmax(distances)) + 1
        if float(distances[split - 1]) <= tolerance:
            return points[[0, -1]]
        left = cls._simplify_trace(points[:split + 1], tolerance)
        right = cls._simplify_trace(points[split:], tolerance)
        return np.vstack((left[:-1], right))

    def _build_navigation_grid(self, snapshot, current, goal):
        resolution = float(snapshot.resolution)
        start_world = np.floor(current[:2] / resolution).astype(int)
        goal_world = np.floor(goal[:2] / resolution).astype(int)
        trajectory = np.asarray(snapshot.trajectory_ned, dtype=np.float64)
        if trajectory.shape[0] < 2:
            raise RuntimeError("Global navigation memory has no outbound trace.")

        height_center = 0.5 * (float(current[2]) + float(goal[2]))
        height_mask = ned_height_slice_mask(
            snapshot.indices,
            resolution,
            height_center,
            float(self.config.global_return_above_height),
            float(self.config.global_return_below_height),
        )
        horizontal = np.asarray(snapshot.indices[height_mask, :2], dtype=int)
        values = np.asarray(snapshot.log_odds[height_mask])
        free_world = horizontal[values <= float(snapshot.free_threshold)]
        occupied_world = horizontal[
            values >= float(snapshot.occupied_threshold)
        ]

        trace_world = self._rasterize_trace(trajectory[:, :2], resolution)
        bounds_points = np.vstack((
            start_world[None, :],
            goal_world[None, :],
            trace_world,
            free_world,
            occupied_world,
        ))
        padding = max(
            2,
            int(math.ceil(
                float(self.config.global_return_trace_radius) / resolution
            )) + 1,
        )
        minimum = bounds_points.min(axis=0) - padding
        maximum = bounds_points.max(axis=0) + padding
        shape = maximum - minimum + 1
        known_free = np.zeros(tuple(shape), dtype=bool)
        blocked = np.zeros(tuple(shape), dtype=bool)
        self._mark_cells(known_free, free_world - minimum)

        trace_radius = int(math.ceil(
            float(self.config.global_return_trace_radius) / resolution
        ))
        self._mark_dilated(
            known_free,
            trace_world - minimum,
            trace_radius,
            resolution,
            float(self.config.global_return_trace_radius),
        )

        inflation_radius = int(math.ceil(
            float(self.config.global_return_inflation_radius) / resolution
        ))
        self._mark_dilated(
            blocked,
            occupied_world - minimum,
            inflation_radius,
            resolution,
            float(self.config.global_return_inflation_radius),
        )
        trace_mask = np.zeros_like(blocked)
        trace_clearance = min(
            float(self.config.global_return_trace_radius),
            max(resolution, 0.5 * float(
                self.config.global_return_inflation_radius
            )),
        )
        self._mark_dilated(
            trace_mask,
            trace_world - minimum,
            int(math.ceil(trace_clearance / resolution)),
            resolution,
            trace_clearance,
        )
        blocked |= ~known_free
        blocked[trace_mask] = False

        start = start_world - minimum
        goal_cell = goal_world - minimum
        self._clear_endpoint(blocked, start)
        self._clear_endpoint(blocked, goal_cell)
        return blocked, minimum, resolution, start, goal_cell

    @staticmethod
    def _rasterize_trace(points_xy, resolution) -> np.ndarray:
        cells = []
        indices = np.floor(np.asarray(points_xy) / resolution).astype(int)
        for start, end in zip(indices[:-1], indices[1:]):
            delta = end - start
            count = max(2, int(np.max(np.abs(delta))) * 2 + 1)
            samples = np.rint(
                start[None, :]
                + np.linspace(0.0, 1.0, count)[:, None] * delta[None, :]
            ).astype(int)
            cells.append(samples)
        if not cells:
            return indices[:1]
        return np.unique(np.vstack(cells), axis=0)

    @staticmethod
    def _mark_cells(mask, cells) -> None:
        cells = np.asarray(cells, dtype=int).reshape(-1, 2)
        if cells.size:
            valid = np.all(cells >= 0, axis=1)
            valid &= np.all(cells < np.asarray(mask.shape), axis=1)
            cells = cells[valid]
            mask[tuple(cells.T)] = True

    @classmethod
    def _mark_dilated(
        cls,
        mask,
        cells,
        radius_cells,
        resolution,
        radius_m,
    ) -> None:
        cells = np.asarray(cells, dtype=int).reshape(-1, 2)
        if not cells.size:
            return
        values = np.arange(-radius_cells, radius_cells + 1)
        xx, yy = np.meshgrid(values, values, indexing="ij")
        offsets = np.column_stack((xx.ravel(), yy.ravel()))
        offsets = offsets[
            np.linalg.norm(offsets * resolution, axis=1) <= radius_m + 1e-9
        ]
        expanded = (cells[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        cls._mark_cells(mask, expanded)

    @staticmethod
    def _clear_endpoint(blocked, endpoint) -> None:
        """把实际起终点接入相邻的历史安全走廊。"""
        x, y = map(int, endpoint)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < blocked.shape[0] and 0 <= ny < blocked.shape[1]:
                    blocked[nx, ny] = False

    def _astar(self, blocked, start, goal):
        start_t = tuple(map(int, start))
        goal_t = tuple(map(int, goal))
        root_two = math.sqrt(2.0)
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0),
            (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, root_two), (-1, 1, root_two),
            (1, -1, root_two), (1, 1, root_two),
        ]
        frontier = [(0.0, 0.0, start_t)]
        came_from = {start_t: None}
        costs = {start_t: 0.0}
        expanded = 0
        maximum = int(self.config.global_return_max_search_nodes)
        while frontier and expanded < maximum:
            _, queued_cost, current = heapq.heappop(frontier)
            if queued_cost > costs.get(current, math.inf) + 1e-9:
                continue
            expanded += 1
            if current == goal_t:
                break
            for dx, dy, step in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if (
                    nxt[0] < 0 or nxt[1] < 0
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
                new_cost = costs[current] + step
                if new_cost >= costs.get(nxt, math.inf):
                    continue
                costs[nxt] = new_cost
                heuristic = math.hypot(goal_t[0] - nxt[0], goal_t[1] - nxt[1])
                heapq.heappush(
                    frontier,
                    (new_cost + heuristic, new_cost, nxt),
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

    def _smooth(self, blocked, path):
        if len(path) <= 2:
            return path
        result = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            candidate = len(path) - 1
            while candidate > anchor + 1:
                if self._line_is_free(blocked, path[anchor], path[candidate]):
                    break
                candidate -= 1
            result.append(path[candidate])
            anchor = candidate
        return result

    @staticmethod
    def _line_is_free(blocked, start, goal):
        start = np.asarray(start, dtype=float)
        delta = np.asarray(goal, dtype=float) - start
        count = max(2, int(math.ceil(np.linalg.norm(delta) * 3.0)) + 1)
        samples = np.rint(
            start[None, :]
            + np.linspace(0.0, 1.0, count)[:, None] * delta[None, :]
        ).astype(int)
        return not bool(np.any(blocked[tuple(samples.T)]))
