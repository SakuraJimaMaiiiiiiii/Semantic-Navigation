"""C++ EGO 核心与现有 Python 安全规划框架之间的适配层。"""

from __future__ import annotations

import importlib
import warnings

import numpy as np

from .ego_local_planner import EgoLocalPlanner, UniformCubicBSpline


class NativeEgoLocalPlanner(EgoLocalPlanner):
    """优先使用 C++ 核心，异常时安全回退到 Python EGO 实现。"""

    def __init__(self, *args, native_backend=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._native_backend = native_backend
        self._native_error: str | None = None
        if self._native_backend is None:
            self._load_native_backend()

    @property
    def native_available(self) -> bool:
        """当前实例是否正在使用已编译的 C++ 扩展。"""
        return self._native_backend is not None

    @property
    def backend_name(self) -> str:
        return "cpp" if self.native_available else "python_fallback"

    @property
    def native_error(self) -> str | None:
        """原生扩展加载或执行失败的原因；正常时为 ``None``。"""
        return self._native_error

    def _load_native_backend(self) -> None:
        try:
            module = importlib.import_module("._native_ego", package=__package__)
            self._native_backend = module.NativeEgoPlanner(
                self._native_config()
            )
        except Exception as error:  # 扩展未编译或 ABI 不匹配都要可诊断
            self._disable_native(error)

    def _disable_native(self, error: Exception) -> None:
        self._native_backend = None
        self._native_error = f"{type(error).__name__}: {error}"
        warnings.warn(
            "C++ EGO 核心不可用，已回退到 Python EGO："
            f"{self._native_error}",
            RuntimeWarning,
            stacklevel=3,
        )

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
        if self._native_backend is None:
            return super()._collision_seed_path(
                snapshot,
                blocked,
                current,
                local_goal,
                now,
            )
        remaining = self._remaining_trajectory_seed(
            snapshot,
            blocked,
            current,
            local_goal,
            now,
        )
        if remaining is not None:
            return remaining
        # Match official EGO first initialization: construct a boundary-state
        # polynomial; C++ then runs A* only for colliding control-point segments.
        return np.vstack((current, local_goal))

    def _direct_seed_path(
        self,
        snapshot,
        blocked,
        current,
        local_goal,
        now,
    ):
        if self._native_backend is not None:
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
        if backend is None:
            return super()._build_trajectory(snapshot, blocked, seed_path)
        try:
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
            if not bool(result["success"]):
                return self._build_python_fallback_trajectory(
                    snapshot, blocked, seed_path
                )
            points = np.asarray(result["control_points"], dtype=np.float64)
            interval = float(result["knot_interval"])
            trajectory = UniformCubicBSpline(points, interval)
            # C++ 已做密集碰撞检测；这里保留 Python 边界的独立复核。
            if self._trajectory_is_free(snapshot, blocked, trajectory):
                return trajectory
        except Exception as error:
            self._disable_native(error)
        return self._build_python_fallback_trajectory(
            snapshot, blocked, seed_path
        )

    def _build_python_fallback_trajectory(
        self,
        snapshot,
        blocked,
        seed_path,
    ):
        seed = np.asarray(seed_path, dtype=np.float64)
        if not self._line_is_free_3d(snapshot, blocked, seed[0], seed[-1]):
            start = self.world_to_grid(snapshot, seed[0])
            goal = self.world_to_grid(snapshot, seed[-1])
            cell_path = super()._astar_3d(blocked, start, goal)
            if not cell_path:
                return None
            cell_path = self._simplify_cells_3d(blocked, cell_path)
            seed = np.asarray([
                self.grid_to_world(snapshot, cell) for cell in cell_path
            ])
            seed[0] = seed_path[0]
            seed[-1] = seed_path[-1]
        return super()._build_trajectory(snapshot, blocked, seed)
