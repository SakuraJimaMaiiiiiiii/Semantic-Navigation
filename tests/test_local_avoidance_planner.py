"""局部A*避障规划器的无仿真测试。"""

import unittest
from dataclasses import dataclass

import numpy as np

from config.planner_config import PlannerConfig
from planner import LocalAvoidancePlanner


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


class _LocalVisualizationCapture:
    def __init__(self):
        self.calls = []

    @staticmethod
    def should_save_local(source_name, now=None):
        return True

    def submit_local(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return True


def _snapshot(obstacles=(), timestamp=1.0):
    shape = (30, 30, 5)
    states = np.zeros(shape, dtype=np.int8)
    inflated = np.zeros(shape, dtype=bool)
    for north, east in obstacles:
        states[north, east, 2] = 1
        inflated[
            max(0, north - 1):north + 2,
            max(0, east - 1):east + 2,
            1:4,
        ] = True
    return _Snapshot(
        timestamp=timestamp,
        origin_ned=np.asarray([-3.0, -3.0, -0.5]),
        resolution=0.2,
        states=states,
        inflated_occupied=inflated,
        drone_position_ned=np.zeros(3),
    )


class LocalAvoidancePlannerTest(unittest.TestCase):
    def _planner(self, snapshot, visualizer=None):
        return LocalAvoidancePlanner(
            _Grid(snapshot),
            PlannerConfig(
                navigation_mode="astar",
                emergency_stop_distance=0.3,
                slowdown_distance=0.8,
                vertical_band=0.2,
            ),
            visualizer=visualizer,
        )

    def test_direct_path_returns_short_term_target(self):
        visualizer = _LocalVisualizationCapture()
        planner = self._planner(_snapshot(), visualizer=visualizer)
        result = planner.plan([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], now=1.0)

        self.assertEqual(result.status, "clear")
        self.assertTrue(result.can_move)
        self.assertAlmostEqual(
            np.linalg.norm(result.local_target_ned),
            1.2,
            places=6,
        )
        self.assertAlmostEqual(
            result.speed_limit,
            planner.config.cruise_max_speed,
        )
        self.assertEqual(len(visualizer.calls), 1)
        self.assertEqual(
            visualizer.calls[0][1]["source_name"],
            "local",
        )

    def test_target_change_invalidates_cached_local_plan(self):
        planner = self._planner(_snapshot())
        first = planner.plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            now=1.0,
        )
        changed = planner.plan(
            [0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            now=1.01,
        )

        self.assertGreater(first.local_target_ned[0], 0.0)
        self.assertAlmostEqual(first.local_target_ned[1], 0.0)
        self.assertAlmostEqual(changed.local_target_ned[0], 0.0)
        self.assertGreater(changed.local_target_ned[1], 0.0)

    def test_outside_goal_uses_ray_boundary_and_stays_in_checked_map(self):
        snapshot = _snapshot()
        planner = self._planner(snapshot)
        current = np.asarray([2.4, 0.0, 0.0])
        target = np.asarray([10.0, 4.0, 0.0])

        result = planner.plan(current, target, now=1.0)

        self.assertEqual(result.status, "clear")
        self.assertTrue(result.can_move)
        expected_direction = (target[:2] - current[:2])
        actual_direction = result.local_target_ned[:2] - current[:2]
        self.assertAlmostEqual(
            actual_direction[0] * expected_direction[1]
            - actual_direction[1] * expected_direction[0],
            0.0,
            places=6,
        )
        map_upper = (
            snapshot.origin_ned[:2]
            + (np.asarray(snapshot.states.shape[:2]) - 0.5)
            * snapshot.resolution
            - planner.config.map_boundary_margin
        )
        self.assertTrue(np.all(result.local_target_ned[:2] <= map_upper + 1e-9))

    def test_outward_goal_stops_inside_boundary_margin(self):
        planner = self._planner(_snapshot())
        result = planner.plan(
            [2.8, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            now=1.0,
        )

        self.assertEqual(result.status, "map_boundary")
        self.assertFalse(result.can_move)

    def test_astar_routes_around_inflated_obstacle(self):
        # 当前点在索引(15,15)，目标沿north正向；障碍位于正前方。
        planner = self._planner(_snapshot(obstacles=[(20, 15)]))
        result = planner.plan([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], now=1.0)

        self.assertEqual(result.status, "avoid")
        self.assertTrue(result.can_move)
        self.assertGreater(len(result.path_ned), 2)
        self.assertNotAlmostEqual(result.local_target_ned[1], 0.0)

    def test_avoidance_target_is_held_until_clear_path_is_confirmed(self):
        grid = _Grid(_snapshot(obstacles=[(20, 15)], timestamp=1.0))
        planner = self._planner(grid.value)
        planner.grid = grid
        current = [0.0, 0.0, 0.0]
        target = [4.0, 0.0, 0.0]

        avoid = planner.plan(current, target, now=1.0)
        self.assertEqual(avoid.status, "avoid")

        grid.value = _snapshot(timestamp=2.0)
        held_once = planner.plan(current, target, now=1.3)
        grid.value = _snapshot(timestamp=3.0)
        held_twice = planner.plan(current, target, now=1.6)
        grid.value = _snapshot(timestamp=4.0)
        clear = planner.plan(current, target, now=1.9)

        self.assertEqual(held_once.status, "avoid")
        self.assertEqual(held_twice.status, "avoid")
        np.testing.assert_allclose(
            held_once.local_target_ned,
            avoid.local_target_ned,
        )
        self.assertEqual(clear.status, "clear")

    def test_missing_map_stops_vehicle(self):
        snapshot = _snapshot()
        snapshot = _Snapshot(
            timestamp=0.0,
            origin_ned=snapshot.origin_ned,
            resolution=snapshot.resolution,
            states=snapshot.states,
            inflated_occupied=snapshot.inflated_occupied,
            drone_position_ned=snapshot.drone_position_ned,
        )
        result = self._planner(snapshot).plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            now=1.0,
        )

        self.assertEqual(result.status, "waiting_for_map")
        self.assertFalse(result.can_move)

    def test_stale_map_stops_vehicle(self):
        planner = self._planner(_snapshot())
        first = planner.plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            now=1.0,
        )
        stale = planner.plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            now=2.0,
        )

        self.assertTrue(first.can_move)
        self.assertEqual(stale.status, "stale_map")
        self.assertFalse(stale.can_move)

    def test_transient_no_path_stops_immediately_then_confirms(self):
        grid = _Grid(_snapshot())
        planner = LocalAvoidancePlanner(
            grid,
            PlannerConfig(
                navigation_mode="astar",
                emergency_stop_distance=0.3,
                slowdown_distance=0.8,
                vertical_band=0.2,
                no_path_confirmation_count=3,
            ),
        )
        first = planner.plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            now=1.0,
        )
        wall = [(20, east) for east in range(30)]

        grid.value = _snapshot(wall, timestamp=2.0)
        transient_once = planner.plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            now=1.3,
        )
        grid.value = _snapshot(wall, timestamp=3.0)
        transient_twice = planner.plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            now=1.6,
        )
        grid.value = _snapshot(wall, timestamp=4.0)
        stopped = planner.plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            now=1.9,
        )

        self.assertEqual(first.status, "clear")
        self.assertEqual(transient_once.status, "transient_no_path")
        self.assertFalse(transient_once.can_move)
        self.assertEqual(transient_twice.status, "transient_no_path")
        self.assertFalse(transient_twice.can_move)
        self.assertEqual(stopped.status, "no_path")
        self.assertFalse(stopped.can_move)

    def test_blocked_raw_goal_is_not_misreported_as_clear(self):
        # 三米前视终点为索引(29,15)，该点被占据时必须走A*。
        planner = self._planner(_snapshot(obstacles=[(29, 15)]))
        result = planner.plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            now=1.0,
        )

        self.assertNotEqual(result.status, "clear")

    def test_forward_depth_guard_stops_even_with_empty_map(self):
        snapshot = _snapshot()
        snapshot = _Snapshot(
            timestamp=snapshot.timestamp,
            origin_ned=snapshot.origin_ned,
            resolution=snapshot.resolution,
            states=snapshot.states,
            inflated_occupied=snapshot.inflated_occupied,
            drone_position_ned=snapshot.drone_position_ned,
            forward_clearance=0.2,
        )
        result = self._planner(snapshot).plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            now=1.0,
        )

        self.assertEqual(result.status, "emergency_stop")
        self.assertFalse(result.can_move)

    def test_depth_guard_mode_ignores_map_ghost_and_tracks_target(self):
        snapshot = _snapshot(obstacles=[(19, 15)])
        snapshot = _Snapshot(
            timestamp=snapshot.timestamp,
            origin_ned=snapshot.origin_ned,
            resolution=snapshot.resolution,
            states=snapshot.states,
            inflated_occupied=snapshot.inflated_occupied,
            drone_position_ned=snapshot.drone_position_ned,
            forward_clearance=3.0,
        )
        planner = LocalAvoidancePlanner(
            _Grid(snapshot),
            PlannerConfig(
                navigation_mode="depth_guard",
                emergency_stop_distance=0.8,
                slowdown_distance=1.5,
            ),
        )
        result = planner.plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, -1.0],
            now=1.0,
        )

        self.assertEqual(result.status, "depth_guard_clear")
        self.assertTrue(result.can_move)
        self.assertAlmostEqual(
            result.speed_limit,
            planner.config.cruise_max_speed,
        )
        self.assertAlmostEqual(result.local_target_ned[2], -0.5)

    def test_safe_sensor_fallback_moves_when_only_projection_is_blocked(self):
        snapshot = _snapshot()
        inflated = snapshot.inflated_occupied.copy()
        inflated[19:22, :, 1:4] = True
        snapshot = _Snapshot(
            timestamp=snapshot.timestamp,
            origin_ned=snapshot.origin_ned,
            resolution=snapshot.resolution,
            states=snapshot.states,
            inflated_occupied=inflated,
            drone_position_ned=snapshot.drone_position_ned,
            forward_clearance=5.0,
        )
        result = self._planner(snapshot).plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            now=1.0,
        )

        self.assertEqual(result.status, "sensor_clear_fallback")
        self.assertTrue(result.can_move)
        self.assertAlmostEqual(result.speed_limit, 0.4)

    def test_sensor_fallback_ignores_noncritical_local_map_ghost(self):
        snapshot = _snapshot(obstacles=[(19, 15)])
        inflated = snapshot.inflated_occupied.copy()
        inflated[19:22, :, 1:4] = True
        snapshot = _Snapshot(
            timestamp=snapshot.timestamp,
            origin_ned=snapshot.origin_ned,
            resolution=snapshot.resolution,
            states=snapshot.states,
            inflated_occupied=inflated,
            drone_position_ned=snapshot.drone_position_ned,
            forward_clearance=3.0,
        )
        result = self._planner(snapshot).plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            now=1.0,
        )

        self.assertEqual(result.status, "sensor_clear_fallback")
        self.assertTrue(result.can_move)

    def test_very_close_map_obstacle_still_emergency_stops(self):
        snapshot = _snapshot(obstacles=[(16, 15)])
        snapshot = _Snapshot(
            timestamp=snapshot.timestamp,
            origin_ned=snapshot.origin_ned,
            resolution=snapshot.resolution,
            states=snapshot.states,
            inflated_occupied=snapshot.inflated_occupied,
            drone_position_ned=snapshot.drone_position_ned,
            forward_clearance=3.0,
        )
        result = self._planner(snapshot).plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            now=1.0,
        )

        self.assertEqual(result.status, "emergency_stop")
        self.assertFalse(result.can_move)

    def test_vertical_target_converges_during_horizontal_navigation(self):
        result = self._planner(_snapshot()).plan(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, -1.0],
            now=1.0,
        )

        self.assertTrue(result.can_move)
        self.assertAlmostEqual(result.local_target_ned[2], -0.5)

    def test_progress_check_rejects_overshooting_local_target(self):
        current = np.asarray([19.5, -0.5, -2.0])
        target = np.asarray([20.0, 2.0, -2.0])

        self.assertFalse(LocalAvoidancePlanner._advances_toward_target(
            current,
            np.asarray([20.7, -0.5, -2.0]),
            target,
        ))
        self.assertTrue(LocalAvoidancePlanner._advances_toward_target(
            current,
            np.asarray([19.5, 0.7, -2.0]),
            target,
        ))

    def test_far_avoidance_target_that_moves_backward_is_rejected(self):
        planner = self._planner(_snapshot(obstacles=[(20, 15)]))
        planner._point_along_path = lambda path, distance: np.asarray(
            [-1.0, 0.0, 0.0]
        )

        result = planner.plan(
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            now=1.0,
        )

        self.assertEqual(result.status, "transient_no_path")
        self.assertFalse(result.can_move)


if __name__ == "__main__":
    unittest.main()
