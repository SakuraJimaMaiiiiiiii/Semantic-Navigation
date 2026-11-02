from types import SimpleNamespace
from unittest.mock import Mock
import unittest
import numpy as np
from navigation.path_smoothing import advance_route
from navigation.frontier_explorer import FrontierExplorer


class ExplorationSmoothingTests(unittest.TestCase):
    def test_arc_advances_while_moving(self):
        path, modes = advance_route([np.zeros(3), np.array([1.,0,0]), np.array([2.,.1,0])],
                                    [(True,True),(False,False)], np.array([.98,0,0]),
                                    np.array([.18,0,0]), lambda a,b:True, .15)
        self.assertEqual(len(path),2)
        self.assertEqual(modes,[(False,False)])

    def test_blocked_shortcut_and_hard_corner_do_not_advance(self):
        for flowing, free in [(True,False),(False,True)]:
            path, _ = advance_route([np.zeros(3), np.array([1.,0,0]), np.array([2.,.1,0])],
                                     [(flowing,True),(False,False)], np.array([.98,0,0]),
                                     np.array([.18,0,0]), lambda a,b:free, .15)
            self.assertEqual(len(path),3)

    def test_wrong_height_does_not_advance(self):
        path, _ = advance_route([np.zeros(3), np.array([1.,0,0]), np.array([2.,.1,0])],
                                [(True,True),(False,False)], np.array([1.,0,.3]),
                                np.zeros(3), lambda a,b:True, .15)
        self.assertEqual(len(path),3)

    def test_curve_executes_without_stop_and_align(self):
        vehicle = SimpleNamespace(get_position=lambda:np.zeros(3), get_euler=lambda:np.zeros(3),
            config=SimpleNamespace(guided_z_tolerance=.15,guided_max_forward_speed=.3),
            move_waypoint_guided=Mock())
        nav = SimpleNamespace(vehicle=vehicle, _check_health=lambda:None,
            _sample_position=lambda:None, _hold_for_planning=Mock(), _align=Mock(),
            _travel_heading=0., _guard=object(), deadline=100, clock=lambda:0,
            config=SimpleNamespace(arrival_tolerance=.4,leg_length=1.2,leg_timeout=30),
            exploration_config=SimpleNamespace(rolling_replan_period=.5))
        FrontierExplorer._move(nav, [np.zeros(3),np.array([.2,.06,0])], flowing=True, turning=True)
        nav._hold_for_planning.assert_not_called()
        nav._align.assert_not_called()
        options = vehicle.move_waypoint_guided.call_args.kwargs
        self.assertTrue(options['pass_through'])
        self.assertEqual(options['control_horizon'], .5)
        self.assertEqual(options['xy_tolerance'], .25)
        self.assertAlmostEqual(nav._motion_speed_limit, .18)


if __name__ == '__main__': unittest.main()
