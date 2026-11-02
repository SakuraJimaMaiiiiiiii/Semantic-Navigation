"""Closed test loop must not get stuck on an unnecessary, altitude-mismatched return."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from mission.rfly_mission_controller import RflyMissionController
from config.mission_config import MissionConfig


class LocalReturnArrivalTests(unittest.TestCase):
    def controller(self, position, plane=-1.516735864803195):
        vehicle=SimpleNamespace(get_position=lambda:np.asarray(position),move_waypoint_guided=Mock())
        planner=SimpleNamespace(navigation_altitude=plane)
        controller=RflyMissionController(vehicle,MissionConfig(),local_planner=planner,
                                         return_safety_planner=planner)
        controller._return_origin_ned=np.array([-.049,.025,-.0167])
        return controller,vehicle

    def test_recorded_centimetre_error_skips_redundant_leg(self):
        controller,vehicle=self.controller([-.035,.038,-1.519])
        controller._return_with_local_planner()
        vehicle.move_waypoint_guided.assert_not_called()
        controller._confirm_return_position()

    def test_outside_home_uses_exact_planner_altitude(self):
        controller,vehicle=self.controller([-.6,.025,-1.519])
        controller._return_with_local_planner()
        target=vehicle.move_waypoint_guided.call_args.args[0]
        np.testing.assert_array_equal(target[:2],controller._return_origin_ned[:2])
        self.assertEqual(target[2],controller.return_safety_planner.navigation_altitude)
        self.assertIs(vehicle.move_waypoint_guided.call_args.kwargs['local_planner'],
                      controller.return_safety_planner)

    def test_nonplanar_planner_retains_current_height(self):
        controller,vehicle=self.controller([-.6,.025,-1.519],None)
        controller._return_with_local_planner()
        self.assertEqual(vehicle.move_waypoint_guided.call_args.args[0][2],-1.519)

    def test_landing_confirmation_is_not_loosened(self):
        controller,vehicle=self.controller([-.45,.025,-1.519])
        vehicle.hover=Mock()
        with self.assertRaisesRegex(RuntimeError,'Return position not confirmed'):
            controller._confirm_return_position()
        vehicle.hover.assert_called_once()
