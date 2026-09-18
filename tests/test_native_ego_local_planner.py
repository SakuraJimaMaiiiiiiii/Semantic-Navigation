"""C++ EGO 适配层的接口与安全回退测试。"""

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np

from config import PlannerConfig
from planner import NativeEgoLocalPlanner


@dataclass(frozen=True)
class _Snapshot:
    timestamp: float
    origin_ned: np.ndarray
    resolution: float
    states: np.ndarray
    inflated_occupied: np.ndarray
    drone_position_ned: np.ndarray
    forward_clearance: float = float("inf")


class _Grid:
    def __init__(self, snapshot):
        self.value = snapshot

    def snapshot(self):
        return self.value


class _Backend:
    def __init__(self):
        self.replan_calls = 0
        self.last_start_velocity = None
        self.last_start_acceleration = None

    def rebound_replan(
        self,
        seed,
        blocked,
        origin,
        resolution,
        start_velocity,
        start_acceleration,
        target_velocity,
    ):
        del blocked, origin, resolution, target_velocity
        self.replan_calls += 1
        self.last_start_velocity = np.asarray(start_velocity).copy()
        self.last_start_acceleration = np.asarray(start_acceleration).copy()
        points = np.vstack((
            np.repeat(seed[:1], 3, axis=0),
            np.repeat(seed[-1:], 3, axis=0),
        ))
        return {
            "success": True,
            "status": "optimized",
            "control_points": points,
            "knot_interval": 0.4,
        }


def _snapshot():
    shape = (31, 31, 15)
    return _Snapshot(
        timestamp=1.0,
        origin_ned=np.asarray([-3.1, -3.1, -1.5]),
        resolution=0.2,
        states=np.zeros(shape, dtype=np.int8),
        inflated_occupied=np.zeros(shape, dtype=bool),
        drone_position_ned=np.zeros(3),
    )


def _config():
    return PlannerConfig(
        navigation_mode="ego_native",
        emergency_stop_distance=0.3,
        map_emergency_stop_distance=0.15,
        slowdown_distance=0.8,
        planning_lookahead=2.6,
    )


class NativeEgoLocalPlannerTest(unittest.TestCase):
    def test_missing_extension_falls_back_during_initialization(self):
        with patch(
            "planner.native_ego_local_planner.importlib.import_module",
            side_effect=ImportError("extension not built"),
        ), self.assertWarns(RuntimeWarning):
            planner = NativeEgoLocalPlanner(_Grid(_snapshot()), _config())

        self.assertEqual(planner.backend_name, "python_fallback")
        self.assertIn("extension not built", planner.native_error)

    def test_native_backend_builds_active_trajectory(self):
        backend = _Backend()
        planner = NativeEgoLocalPlanner(
            _Grid(_snapshot()),
            _config(),
            native_backend=backend,
        )

        result = planner.plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            now=1.0,
        )

        self.assertTrue(result.can_move)
        self.assertEqual(result.status, "ego_clear")
        self.assertEqual(planner.backend_name, "cpp")
        self.assertEqual(backend.replan_calls, 1)
        np.testing.assert_allclose(backend.last_start_velocity, 0.0)
        np.testing.assert_allclose(backend.last_start_acceleration, 0.0)
        self.assertIsNotNone(planner.active_trajectory)

    def test_measured_motion_state_reaches_native_boundary_conditions(self):
        backend = _Backend()
        planner = NativeEgoLocalPlanner(
            _Grid(_snapshot()),
            _config(),
            native_backend=backend,
        )
        planner.set_motion_state([0.4, -0.1, 0.0], [0.2, 0.0, 0.0])

        planner.plan([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], now=1.0)

        np.testing.assert_allclose(
            backend.last_start_velocity,
            [0.4, -0.1, 0.0],
        )
        np.testing.assert_allclose(
            backend.last_start_acceleration,
            [0.2, 0.0, 0.0],
        )

    def test_invalid_native_result_disables_backend_and_falls_back(self):
        class InvalidBackend(_Backend):
            def rebound_replan(self, *args):
                raise RuntimeError("native failure")

        planner = NativeEgoLocalPlanner(
            _Grid(_snapshot()),
            _config(),
            native_backend=InvalidBackend(),
        )
        with self.assertWarns(RuntimeWarning):
            result = planner.plan(
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                now=1.0,
            )

        self.assertTrue(result.can_move)
        self.assertEqual(planner.backend_name, "python_fallback")
        self.assertIn("native failure", planner.native_error)


if __name__ == "__main__":
    unittest.main()
