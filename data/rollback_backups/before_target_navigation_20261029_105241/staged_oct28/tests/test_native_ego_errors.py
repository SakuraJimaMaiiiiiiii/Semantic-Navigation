"""Native planner failures keep their cause; unsafe trajectories remain rejected."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from planner.native_ego_local_planner import NativeEgoLocalPlanner


class NativeEgoErrorsTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()
        self.planner = NativeEgoLocalPlanner.__new__(NativeEgoLocalPlanner)
        self.planner._native_backend = self.backend
        self.planner._planning_start_velocity = np.zeros(3)
        self.planner._planning_start_acceleration = np.zeros(3)
        self.planner.navigation_altitude = None
        self.planner.last_plan_timings = {}
        self.snapshot = SimpleNamespace(origin_ned=np.zeros(3), resolution=.2)

    def plan(self):
        return self.planner._build_trajectory(
            self.snapshot, np.zeros((3, 3, 3), dtype=bool), np.zeros((2, 3)))

    def test_backend_error_is_not_relabelled(self):
        error = ValueError('invalid native input shape')
        self.backend.rebound_replan.side_effect = error
        with self.assertRaises(ValueError) as caught:
            self.plan()
        self.assertIs(caught.exception, error)

    def test_no_route_remains_a_normal_result(self):
        self.backend.rebound_replan.return_value = {'success': False}
        self.assertIsNone(self.plan())

    def test_success_still_requires_trajectory_clearance(self):
        self.backend.rebound_replan.return_value = {
            'success': True, 'control_points': np.zeros((4, 3)), 'knot_interval': .5}
        self.planner._trajectory_is_free = Mock(return_value=False)
        self.assertIsNone(self.plan())
        self.planner._trajectory_is_free.assert_called_once()
