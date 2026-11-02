from types import SimpleNamespace
from unittest.mock import Mock
import unittest
import numpy as np

from navigation.frontier_explorer import FrontierExplorer


class ContinuousExplorationTests(unittest.TestCase):
    def test_certified_intermediate_turn_does_not_hover_or_align_in_place(self):
        vehicle = SimpleNamespace(get_position=lambda:np.zeros(3), get_euler=lambda:np.zeros(3),
            config=SimpleNamespace(guided_z_tolerance=.15), move_waypoint_guided=Mock())
        nav = SimpleNamespace(vehicle=vehicle, _check_health=lambda:None,
            _sample_position=lambda:None, _hold_for_planning=Mock(), _align=Mock(),
            _travel_heading=0., _guard=object(), deadline=100., clock=lambda:0.,
            config=SimpleNamespace(arrival_tolerance=.4,leg_length=1.2,leg_timeout=30.),
            exploration_config=SimpleNamespace(rolling_replan_period=.5))
        FrontierExplorer._move(nav, [np.zeros(3),np.array([0.,1.,0.]),np.array([0.,2.,0.])],
                               continuous=True)
        nav._hold_for_planning.assert_not_called()
        nav._align.assert_not_called()
        options = vehicle.move_waypoint_guided.call_args.kwargs
        self.assertTrue(options['pass_through'])
        self.assertAlmostEqual(options['yaw'], np.pi/2)
        self.assertEqual(options['xy_tolerance'], .4)
        self.assertEqual(options['control_horizon'], 10.)

    def test_uncertified_next_segment_still_slows_at_corner(self):
        vehicle = SimpleNamespace(get_position=lambda:np.zeros(3), get_euler=lambda:np.zeros(3),
            config=SimpleNamespace(guided_z_tolerance=.15), move_waypoint_guided=Mock())
        nav = SimpleNamespace(vehicle=vehicle, _check_health=lambda:None,
            _sample_position=lambda:None, _travel_heading=0., _guard=object(),
            deadline=100., clock=lambda:0.,
            config=SimpleNamespace(arrival_tolerance=.4,leg_length=1.2,leg_timeout=30.),
            exploration_config=SimpleNamespace(rolling_replan_period=.5))
        FrontierExplorer._move(nav, [np.zeros(3),np.array([0.,1.,0.]),np.array([0.,2.,0.])],
                               continuous=False)
        self.assertFalse(vehicle.move_waypoint_guided.call_args.kwargs['pass_through'])


if __name__ == '__main__':
    unittest.main()
