"""速度引导中局部轨迹点与最终航点容差的回归测试。"""

import math
import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np

from config import Config


for module_name in ("PX4MavCtrlV4", "ReqCopterSim", "UE4CtrlAPI"):
    sys.modules.setdefault(module_name, types.ModuleType(module_name))

from vehicle.rfly_multirotor import RflyMultirotorInterface


class _CommandCaptured(Exception):
    pass


class _LocalPlanner:
    last_map_obstacle_distance = math.inf
    last_forward_clearance = math.inf

    @staticmethod
    def plan(_position, _target):
        return SimpleNamespace(
            status="ego_clear",
            obstacle_distance=math.inf,
            speed_limit=0.8,
            local_target_ned=np.asarray([0.18, 0.0, -1.5]),
            can_move=True,
        )

    @staticmethod
    def tracking_target(now=None):
        del now
        return np.asarray([0.18, 0.0, -1.5])


class _NoisyStraightPlanner(_LocalPlanner):
    @staticmethod
    def plan(_position, _target):
        plan = _LocalPlanner.plan(_position, _target)
        plan.local_target_ned = np.asarray([0.18, 0.10, -1.5])
        return plan

    @staticmethod
    def tracking_target(now=None):
        del now
        return np.asarray([0.18, 0.10, -1.5])


class _VerticallyNoisyStraightPlanner(_LocalPlanner):
    @staticmethod
    def plan(_position, _target):
        plan = _LocalPlanner.plan(_position, _target)
        plan.local_target_ned = np.asarray([0.18, 0.0, -1.15])
        return plan

    @staticmethod
    def tracking_target(now=None):
        del now
        return np.asarray([0.18, 0.0, -1.15])


class VelocityGuidanceTest(unittest.TestCase):
    @staticmethod
    def _capture_first_world_velocity(pass_through):
        vehicle = RflyMultirotorInterface(Config(
            guided_max_forward_speed=0.8,
            guided_max_horizontal_acceleration=1000.0,
        ))
        vehicle.connected = True
        vehicle.offboard_enabled = True
        vehicle.mav = SimpleNamespace(
            uavPosNED=np.asarray([0.0, 0.0, -1.5]),
            uavVelNED=np.zeros(3),
            uavAngEular=np.zeros(3),
        )
        vehicle.raise_if_collision = lambda **_kwargs: None
        commands = []

        def capture_command(velocity, yaw_rate=0.0):
            commands.append((np.asarray(velocity), float(yaw_rate)))
            raise _CommandCaptured

        vehicle.send_velocity = capture_command
        with unittest.TestCase().assertRaises(_CommandCaptured):
            vehicle.move_velocity_guided(
                [0.7, 0.0, -1.5],
                face_direction=False,
                pass_through=pass_through,
            )
        return commands[0][0]

    def test_close_local_tracking_point_does_not_trigger_global_tolerance(self):
        vehicle = RflyMultirotorInterface(Config(
            guided_max_forward_speed=0.8,
            guided_max_horizontal_acceleration=1000.0,
        ))
        vehicle.connected = True
        vehicle.offboard_enabled = True
        vehicle.mav = SimpleNamespace(
            uavPosNED=np.asarray([0.0, 0.0, -1.5]),
            uavVelNED=np.zeros(3),
            uavAngEular=np.zeros(3),
        )
        vehicle.raise_if_collision = lambda **_kwargs: None
        commands = []

        def capture_command(velocity, yaw_rate=0.0):
            commands.append((np.asarray(velocity), float(yaw_rate)))
            raise _CommandCaptured

        vehicle.send_body_velocity = capture_command

        with self.assertRaises(_CommandCaptured):
            vehicle.move_velocity_guided(
                [20.0, 0.0, -1.5],
                local_planner=_LocalPlanner(),
            )

        self.assertEqual(len(commands), 1)
        self.assertAlmostEqual(commands[0][0][0], 0.8)
        self.assertAlmostEqual(commands[0][0][1], 0.0)

    def test_clear_route_yaw_uses_stable_final_target_direction(self):
        vehicle = RflyMultirotorInterface(Config(
            guided_max_forward_speed=0.8,
        ))
        vehicle.connected = True
        vehicle.offboard_enabled = True
        vehicle.mav = SimpleNamespace(
            uavPosNED=np.asarray([0.0, 0.0, -1.5]),
            uavVelNED=np.zeros(3),
            uavAngEular=np.zeros(3),
        )
        vehicle.raise_if_collision = lambda **_kwargs: None
        commands = []

        def capture_command(velocity, yaw_rate=0.0):
            commands.append((np.asarray(velocity), float(yaw_rate)))
            raise _CommandCaptured

        vehicle.send_body_velocity = capture_command

        with self.assertRaises(_CommandCaptured):
            vehicle.move_velocity_guided(
                [20.0, 0.0, -1.5],
                local_planner=_NoisyStraightPlanner(),
            )

        self.assertEqual(len(commands), 1)
        self.assertAlmostEqual(commands[0][1], 0.0, places=9)

    def test_clear_route_ignores_replanned_vertical_target_noise(self):
        vehicle = RflyMultirotorInterface(Config(
            guided_max_forward_speed=0.8,
        ))
        vehicle.connected = True
        vehicle.offboard_enabled = True
        vehicle.mav = SimpleNamespace(
            uavPosNED=np.asarray([0.0, 0.0, -1.5]),
            uavVelNED=np.zeros(3),
            uavAngEular=np.zeros(3),
        )
        vehicle.raise_if_collision = lambda **_kwargs: None
        commands = []

        def capture_command(velocity, yaw_rate=0.0):
            commands.append((np.asarray(velocity), float(yaw_rate)))
            raise _CommandCaptured

        vehicle.send_body_velocity = capture_command

        with self.assertRaises(_CommandCaptured):
            vehicle.move_velocity_guided(
                [20.0, 0.0, -1.5],
                local_planner=_VerticallyNoisyStraightPlanner(),
            )

        self.assertEqual(len(commands), 1)
        self.assertAlmostEqual(commands[0][0][2], 0.0, places=9)

    def test_pass_through_point_keeps_cruise_speed(self):
        pass_through = self._capture_first_world_velocity(True)
        stopping = self._capture_first_world_velocity(False)

        self.assertAlmostEqual(pass_through[0], 0.8, places=6)
        self.assertAlmostEqual(stopping[0], 0.42, places=6)


if __name__ == "__main__":
    unittest.main()
