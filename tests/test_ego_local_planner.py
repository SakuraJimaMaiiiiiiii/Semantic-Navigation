"""原生 EGO 风格规划器的无仿真测试。"""

import unittest
from dataclasses import dataclass

import numpy as np

from config import PlannerConfig
from planner import EgoLocalPlanner, UniformCubicBSpline


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


def _snapshot(obstacles=(), timestamp=1.0, forward_clearance=float("inf")):
    shape = (31, 31, 15)
    states = np.zeros(shape, dtype=np.int8)
    inflated = np.zeros(shape, dtype=bool)
    for north, east, down in obstacles:
        states[north, east, down] = 1
        inflated[
            max(0, north - 1):north + 2,
            max(0, east - 1):east + 2,
            max(0, down - 1):down + 2,
        ] = True
    return _Snapshot(
        timestamp=timestamp,
        origin_ned=np.asarray([-3.1, -3.1, -1.5]),
        resolution=0.2,
        states=states,
        inflated_occupied=inflated,
        drone_position_ned=np.zeros(3),
        forward_clearance=forward_clearance,
    )


def _config(**changes):
    values = dict(
        navigation_mode="ego",
        emergency_stop_distance=0.3,
        map_emergency_stop_distance=0.15,
        slowdown_distance=0.8,
        planning_lookahead=2.6,
        ego_clearance=0.35,
        ego_optimization_iterations=20,
        max_search_nodes=30_000,
    )
    values.update(changes)
    return PlannerConfig(**values)


class UniformCubicBSplineTest(unittest.TestCase):
    def test_repeated_control_points_clamp_both_endpoints(self):
        start = np.asarray([0.0, 0.0, 0.0])
        end = np.asarray([2.0, 1.0, -0.5])
        points = np.vstack((
            np.repeat(start[None, :], 3, axis=0),
            np.repeat(end[None, :], 3, axis=0),
        ))
        trajectory = UniformCubicBSpline(points, 0.2)

        np.testing.assert_allclose(trajectory.evaluate(0.0), start)
        np.testing.assert_allclose(
            trajectory.evaluate(trajectory.duration),
            end,
        )


class EgoLocalPlannerTest(unittest.TestCase):
    def test_clear_space_returns_bspline_target(self):
        planner = EgoLocalPlanner(_Grid(_snapshot()), _config())

        result = planner.plan([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], now=1.0)

        self.assertEqual(result.status, "ego_clear")
        self.assertTrue(result.can_move)
        self.assertIsNotNone(planner.active_trajectory)
        self.assertGreater(result.local_target_ned[0], 0.0)
        self.assertGreater(len(result.path_ned), 3)

    def test_clear_path_keeps_continuous_goal_instead_of_voxel_center(self):
        snapshot = _snapshot()
        planner = EgoLocalPlanner(_Grid(snapshot), _config())
        current = np.asarray([0.0, 0.0, 0.0])
        target = np.asarray([4.0, 1.3, 0.0])
        expected = planner._local_goal_in_map(
            snapshot,
            current,
            target,
        )

        result = planner.plan(current, target, now=1.0)

        self.assertEqual(result.status, "ego_clear")
        np.testing.assert_allclose(result.path_ned[-1], expected, atol=1e-9)

    def test_3d_obstacle_generates_collision_free_rebound_trajectory(self):
        # 当前点索引约为 (15, 15, 7)，前方放置三维膨胀障碍。
        snapshot = _snapshot(obstacles=[(20, 15, 7)])
        planner = EgoLocalPlanner(_Grid(snapshot), _config())

        result = planner.plan([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], now=1.0)

        self.assertEqual(result.status, "ego_avoid")
        self.assertTrue(result.can_move)
        indices = np.asarray([
            planner.world_to_grid(snapshot, point) for point in result.path_ned
        ])
        self.assertTrue(all(
            not snapshot.inflated_occupied[tuple(index)] for index in indices
        ))
        lateral_or_vertical = np.abs(result.path_ned[:, 1:])
        self.assertGreater(float(np.max(lateral_or_vertical)), 0.05)

    def test_depth_emergency_guard_still_has_priority(self):
        planner = EgoLocalPlanner(
            _Grid(_snapshot(forward_clearance=0.2)),
            _config(),
        )

        result = planner.plan([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], now=1.0)

        self.assertEqual(result.status, "emergency_stop")
        self.assertFalse(result.can_move)

    def test_tracking_target_advances_between_replanning_ticks(self):
        planner = EgoLocalPlanner(_Grid(_snapshot()), _config())
        planner.plan([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], now=1.0)

        first = planner.tracking_target(now=1.0)
        later = planner.tracking_target(now=1.1)

        self.assertGreater(later[0], first[0])

    def test_diagonal_transition_cannot_cut_an_occupied_corner(self):
        blocked = np.zeros((3, 3, 3), dtype=bool)
        blocked[1, 0, 0] = True

        self.assertFalse(EgoLocalPlanner._transition_is_free(
            blocked,
            (0, 0, 0),
            (1, 1, 0),
        ))


if __name__ == "__main__":
    unittest.main()
