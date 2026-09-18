"""Mission-level tests that do not require the RflySim SDK."""

import unittest
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from mission.rfly_mission_controller import RflyMissionController
from config.mission_config import MissionConfig


class _Vehicle:
    def __init__(self):
        self.config = SimpleNamespace()
        self.calls = []
        self.yaw_calls = []
        self.position = np.asarray([4.0, 2.0, -1.5])

    @staticmethod
    def _vector3(value, _name):
        return np.asarray(value, dtype=float)

    def get_position(self):
        return self.position.copy()

    def align_yaw(self, yaw):
        self.yaw_calls.append(yaw)

    def move_velocity_guided(self, waypoint, **options):
        waypoint = np.asarray(waypoint)
        self.calls.append((waypoint, options))
        self.position = waypoint.copy()

class _GlobalPlanner:
    @staticmethod
    def plan(current, target):
        return np.vstack((current, [4.0, 0.0, -1.5], target))


class GlobalReturnExecutionTest(unittest.TestCase):
    def test_return_restores_measured_pre_takeoff_yaw_before_landing(self):
        for return_mode in ("local", "global", "fallback"):
            with self.subTest(return_mode=return_mode):
                vehicle = MagicMock()
                vehicle.get_position.return_value = np.array([1.0, 2.0, 0.0])
                vehicle.get_euler.return_value = np.array([0.0, 0.0, 1.2])
                mission = MissionConfig(waypoints=(), initial_yaw=-0.4)
                controller = RflyMissionController(
                    vehicle, mission,
                    global_return_planner=None if return_mode == "local" else object(),
                )
                events = []
                vehicle.enable_offboard.side_effect = lambda **_: events.append("arm")
                vehicle.get_euler.side_effect = lambda: (
                    events.append("read_yaw") or np.array([0.0, 0.0, 1.2])
                )
                vehicle.align_yaw.side_effect = lambda yaw: events.append(("yaw", yaw))
                vehicle.land.side_effect = lambda: events.append("land")
                controller._return_with_local_planner = MagicMock(
                    side_effect=lambda: events.append("returned")
                )
                controller._return_on_global_path = MagicMock(
                    side_effect=RuntimeError("No path") if return_mode == "fallback"
                    else lambda: events.append("returned")
                )

                controller.run()

                self.assertEqual(events, [
                    "read_yaw", "arm", ("yaw", -0.4),
                    "returned", ("yaw", 1.2), "land",
                ])
                vehicle.get_euler.assert_called_once_with()

    def test_global_return_passes_intermediate_points_without_stopping(self):
        vehicle = _Vehicle()
        mission = SimpleNamespace()
        safety_planner = object()
        controller = RflyMissionController(
            vehicle,
            mission,
            global_return_planner=_GlobalPlanner(),
            return_safety_planner=safety_planner,
        )
        controller._return_origin_ned = np.asarray([0.0, 0.0, -1.5])

        controller._return_on_global_path()

        self.assertEqual(len(vehicle.calls), 2)
        np.testing.assert_allclose(vehicle.yaw_calls, [-math.pi / 2.0])
        expected_yaws = [-math.pi / 2.0, math.pi]
        for (_, options), expected_yaw in zip(vehicle.calls, expected_yaws):
            self.assertFalse(options["face_direction"])
            self.assertEqual(options["yaw"], expected_yaw)
            self.assertIs(options["local_planner"], safety_planner)
        self.assertTrue(vehicle.calls[0][1]["pass_through"])
        self.assertFalse(vehicle.calls[1][1]["pass_through"])

    def test_local_fallback_turns_toward_origin_before_flying(self):
        vehicle = _Vehicle()
        mission = SimpleNamespace()
        local_planner = object()
        controller = RflyMissionController(
            vehicle,
            mission,
            local_planner=local_planner,
        )
        controller._return_origin_ned = np.asarray([1.0, -1.0, 0.0])

        controller._return_with_local_planner()

        self.assertAlmostEqual(
            vehicle.yaw_calls[0],
            math.atan2(-3.0, -3.0),
        )
        self.assertEqual(len(vehicle.calls), 1)
        waypoint, options = vehicle.calls[0]
        np.testing.assert_allclose(waypoint, [1.0, -1.0, -1.5])
        self.assertFalse(options["face_direction"])
        self.assertEqual(options["yaw"], vehicle.yaw_calls[0])
        self.assertIs(options["local_planner"], local_planner)

    def test_local_return_prefers_dedicated_return_safety_planner(self):
        vehicle = _Vehicle()
        mission = SimpleNamespace()
        outbound_planner = object()
        return_planner = object()
        controller = RflyMissionController(
            vehicle,
            mission,
            local_planner=outbound_planner,
            return_safety_planner=return_planner,
        )
        controller._return_origin_ned = np.asarray([0.0, 0.0, -1.5])

        controller._return_with_local_planner()

        self.assertIs(
            vehicle.calls[0][1]["local_planner"],
            return_planner,
        )


if __name__ == "__main__":
    unittest.main()
