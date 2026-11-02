"""C++ EGO 核心与现有 Python 安全规划框架之间的适配层。"""

from __future__ import annotations

import importlib
import time

import numpy as np

from .ego_local_planner import EgoLocalPlanner, UniformCubicBSpline


class NativeEgoLocalPlanner(EgoLocalPlanner):
    """使用指定 C++ 核心；失败上报，不自动切换算法。"""

    def __init__(self, *args, native_backend=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._native_backend = native_backend
        if self._native_backend is None:
            try:
                module = importlib.import_module("._native_ego", package=__package__)
            except ImportError as error:
                raise RuntimeError("C++ EGO 扩展加载失败，请检查或编译 planner/native_ego") from error
            self._native_backend = module.NativeEgoPlanner(self._native_config())

    @property
    def backend_name(self) -> str:
        return "cpp"

    def _native_config(self) -> dict[str, float | int]:
        config = self.config
        return {
            "control_point_spacing": float(
                config.ego_control_point_spacing
            ),
            "clearance": float(config.ego_clearance),
            "knot_interval": float(config.ego_knot_interval),
            "max_velocity": float(config.cruise_max_speed),
            "max_acceleration": float(config.ego_max_acceleration),
            "feasibility_tolerance": float(
                config.ego_feasibility_tolerance
            ),
            "optimization_iterations": int(
                config.ego_optimization_iterations
            ),
            "lbfgs_memory": int(config.ego_lbfgs_memory),
            "rebound_max_restarts": int(
                config.ego_rebound_max_restarts
            ),
            "smoothness_weight": float(config.ego_smoothness_weight),
            "collision_weight": float(config.ego_collision_weight),
            "feasibility_weight": float(config.ego_feasibility_weight),
            "fitness_weight": float(config.ego_fitness_weight),
            "collision_check_step": float(
                config.ego_collision_check_step
            ),
            "max_search_nodes": int(config.max_search_nodes),
        }

    def _collision_seed_path(
        self,
        snapshot,
        blocked,
        current,
        local_goal,
        now,
    ):
        return self._direct_seed_path(snapshot, blocked, current, local_goal, now)

    def _direct_seed_path(
        self,
        snapshot,
        blocked,
        current,
        local_goal,
        now,
    ):
        # Both native paths reuse a certified remainder or initialize the same
        # boundary-state polynomial. C++ applies A* to colliding segments.
        remaining = self._remaining_trajectory_seed(
            snapshot,
            blocked,
            current,
            local_goal,
            now,
        )
        if remaining is not None:
            return remaining
        return np.vstack((current, local_goal))

    def _build_trajectory(self, snapshot, blocked, seed_path):
        backend = self._native_backend
        started = time.monotonic()
        result = backend.rebound_replan(
            np.ascontiguousarray(seed_path, dtype=np.float64),
            np.ascontiguousarray(blocked, dtype=np.uint8),
            np.ascontiguousarray(snapshot.origin_ned, dtype=np.float64),
            float(snapshot.resolution),
            np.ascontiguousarray(
                self._planning_start_velocity,
                dtype=np.float64,
            ),
            np.ascontiguousarray(
                self._planning_start_acceleration,
                dtype=np.float64,
            ),
            np.zeros(3, dtype=np.float64),
        )
        self.last_plan_timings['native_solver_seconds'] = time.monotonic()-started
        if not bool(result["success"]):
            return None
        points = np.asarray(result["control_points"], dtype=np.float64)
        interval = float(result["knot_interval"])
        if self.navigation_altitude is not None:
            points = points.copy()
            points[:, 2] = self.navigation_altitude
            interval = max(interval, self._feasible_knot_interval(points))
        trajectory = UniformCubicBSpline(points, interval)
        # C++ 已做密集碰撞检测；这里保留 Python 边界的独立复核。
        started = time.monotonic()
        free = self._trajectory_is_free(snapshot, blocked, trajectory)
        self.last_plan_timings['trajectory_clearance_seconds'] = time.monotonic()-started
        if free:
            return trajectory
        return None
