"""固定全局记忆地图返航 A* 测试。"""

import unittest
from dataclasses import dataclass

import numpy as np

from config import PlannerConfig
from planner import GlobalReturnPlanner
from planner.vertical_projection import ned_height_slice_mask


@dataclass(frozen=True)
class _Snapshot:
    timestamp: float
    resolution: float
    indices: np.ndarray
    log_odds: np.ndarray
    occupied_threshold: float
    free_threshold: float
    drone_position_ned: np.ndarray
    dropped_new_voxels: int
    trajectory_ned: np.ndarray


class _MemoryMap:
    def __init__(self, snapshot):
        self.value = snapshot

    def snapshot(self):
        return self.value


class _GlobalVisualizationCapture:
    def __init__(self):
        self.calls = []

    def submit_global(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return True


def _snapshot(indices, values, trajectory, resolution=1.0):
    return _Snapshot(
        timestamp=1.0,
        resolution=resolution,
        indices=np.asarray(indices, dtype=np.int32),
        log_odds=np.asarray(values, dtype=np.float32),
        occupied_threshold=0.7,
        free_threshold=-0.2,
        drone_position_ned=np.asarray(trajectory[-1], dtype=float),
        dropped_new_voxels=0,
        trajectory_ned=np.asarray(trajectory, dtype=float),
    )


class GlobalReturnPlannerTest(unittest.TestCase):
    def test_shortens_outbound_trace_through_known_free_space(self):
        free = []
        for north in range(0, 7):
            for east in range(0, 4):
                free.append((north, east, -2))
        trajectory = [
            [0.5, 0.5, -1.5],
            [0.5, 3.5, -1.5],
            [6.5, 3.5, -1.5],
        ]
        snapshot = _snapshot(
            free,
            [-1.0] * len(free),
            trajectory,
        )
        visualizer = _GlobalVisualizationCapture()
        planner = GlobalReturnPlanner(
            _MemoryMap(snapshot),
            PlannerConfig(
                global_return_inflation_radius=0.3,
                global_return_trace_radius=0.45,
                global_return_above_height=1.0,
                global_return_below_height=1.0,
            ),
            visualizer=visualizer,
        )

        path = planner.plan(trajectory[-1], trajectory[0])

        self.assertEqual(path.shape[1], 3)
        np.testing.assert_allclose(path[0], trajectory[-1])
        np.testing.assert_allclose(path[-1], trajectory[0])
        self.assertTrue(np.all(
            np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
            <= planner.config.planning_lookahead + 1e-9
        ))
        direct_length = float(np.sum(np.linalg.norm(
            np.diff(path[:, :2], axis=0),
            axis=1,
        )))
        outbound_length = float(np.sum(np.linalg.norm(
            np.diff(np.asarray(trajectory)[:, :2], axis=0),
            axis=1,
        )))
        self.assertLess(direct_length, outbound_length)
        self.assertEqual(len(visualizer.calls), 1)
        self.assertEqual(
            visualizer.calls[0][1]["source_name"],
            "global_astar",
        )
        self.assertEqual(
            visualizer.calls[0][1]["above_height"],
            1.0,
        )
        self.assertEqual(
            visualizer.calls[0][1]["below_height"],
            1.0,
        )

    def test_uses_remembered_trace_when_unknown_space_surrounds_it(self):
        trajectory = [
            [0.5, 0.5, -1.5],
            [0.5, 2.5, -1.5],
            [2.5, 2.5, -1.5],
        ]
        snapshot = _snapshot(
            [(0, 0, -2)],
            [-1.0],
            trajectory,
        )
        planner = GlobalReturnPlanner(
            _MemoryMap(snapshot),
            PlannerConfig(
                global_return_inflation_radius=0.3,
                global_return_trace_radius=1.1,
                global_return_above_height=1.0,
                global_return_below_height=1.0,
            ),
        )

        path = planner.plan(trajectory[-1], trajectory[0])

        self.assertGreaterEqual(len(path), 2)
        np.testing.assert_allclose(path[0], trajectory[-1])
        np.testing.assert_allclose(path[-1], trajectory[0])

    def test_outbound_trace_overrides_conflicting_occupied_voxels(self):
        trajectory = [
            [0.5, 0.5, -1.5],
            [3.5, 0.5, -1.5],
            [6.5, 0.5, -1.5],
        ]
        occupied = [(north, 0, -2) for north in range(7)]
        snapshot = _snapshot(
            occupied,
            [1.0] * len(occupied),
            trajectory,
        )
        planner = GlobalReturnPlanner(
            _MemoryMap(snapshot),
            PlannerConfig(
                global_return_inflation_radius=0.3,
                global_return_trace_radius=0.6,
                global_return_above_height=1.0,
                global_return_below_height=1.0,
            ),
        )

        path = planner.plan(trajectory[-1], trajectory[0])

        self.assertGreaterEqual(len(path), 2)
        np.testing.assert_allclose(path[0], trajectory[-1])
        np.testing.assert_allclose(path[-1], trajectory[0])

    def test_disconnected_current_position_falls_back_to_reverse_trace(self):
        trajectory = [
            [0.5, 0.5, -1.5],
            [3.5, 0.5, -1.5],
            [6.5, 0.5, -1.5],
        ]
        snapshot = _snapshot(
            [(0, 0, -2)],
            [-1.0],
            trajectory,
        )
        planner = GlobalReturnPlanner(_MemoryMap(snapshot))

        path = planner.plan([10.5, 0.5, -1.5], trajectory[0])

        np.testing.assert_allclose(path[0], [10.5, 0.5, -1.5])
        np.testing.assert_allclose(path[-1], trajectory[0])
        self.assertTrue(np.all(np.diff(path[:, 0]) <= 0.0))
        self.assertTrue(np.all(
            np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
            <= planner.config.planning_lookahead + 1e-9
        ))

    def test_height_slice_excludes_ceiling_and_low_obstacles(self):
        indices = np.asarray([
            [0, 0, -8],  # Down=-2.25，位于飞行平面上方0.50m
            [0, 0, -7],  # Down=-1.95，位于允许的上方范围内
            [0, 0, -6],  # Down=-1.65，位于允许的下方范围内
            [0, 0, -5],  # Down=-1.35，柱体的较低采样层
            [0, 0, -4],  # Down=-1.05，柱体的更低采样层
            [0, 0, -3],  # Down=-0.75，低于允许范围
        ])

        mask = ned_height_slice_mask(
            indices,
            resolution=0.3,
            center_down=-1.75,
            above_height=0.25,
            below_height=0.9,
        )

        np.testing.assert_array_equal(
            mask,
            [False, True, True, True, True, False],
        )


if __name__ == "__main__":
    unittest.main()
