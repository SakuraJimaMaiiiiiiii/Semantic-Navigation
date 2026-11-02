from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
import numpy as np
from navigation.height_clearance import ascent_levels, vertical_corridor_free
from navigation.frontier_explorer import FrontierExplorer
from config.semantic_navigation_config import SemanticNavigationConfig
from config.exploration_config import FrontierExplorationConfig


class StairTests(unittest.TestCase):
    def descent_fixture(self):
        position = np.array([4., 0., -2.3])
        velocity = np.zeros(3)
        clock = [0.]
        source = SimpleNamespace(states=np.zeros((100, 60, 50), dtype=np.int8),
                                 origin_ned=np.array([-2., -3., -4.]), resolution=.1)
        vehicle = Mock()
        vehicle.get_position.side_effect = lambda: position.copy()
        def send(v): velocity[:] = v
        def wait(dt):
            position[:] += velocity*dt
            clock[0] += dt
        vehicle.send_velocity.side_effect = send
        vehicle.wait.side_effect = wait
        nav = SimpleNamespace(vehicle=vehicle, local_map=SimpleNamespace(snapshot=lambda: source),
            global_map=Mock(), altitude=-2.3, _nominal_altitude=-1.5,
            _height_probe_position=np.array([0., 0., -2.3]),
            config=SemanticNavigationConfig(), _motion_progress=(position.copy(), 0.),
            _hold_for_planning=Mock(), _event=Mock(), _check_health=lambda: None,
            deadline=100, clock=lambda: clock[0], memory=SimpleNamespace(),
            _guard=SimpleNamespace(), active_viewpoint=None,
            exploration_waypoints=[{"position": [6., 0., -2.3]}])
        return nav, position, source

    def test_clear_descent_restores_goal_memory_and_route(self):
        nav, position, _ = self.descent_fixture()
        goal = np.array([6., 0., -2.3])
        grid = Mock(); grid.path.return_value = [position, goal]
        with patch('navigation.frontier_explorer.NavigationGrid.from_global', return_value=grid):
            self.assertTrue(FrontierExplorer._try_restore_cruise_height(nav, goal))
        self.assertLess(abs(position[2]+1.5), .05)
        self.assertEqual(nav.memory.flight_altitude, nav.altitude)
        self.assertEqual(goal[2], nav.altitude)
        self.assertEqual(nav.exploration_waypoints[-1]['position'], goal.tolist())
        self.assertIsNone(nav.active_route)
        self.assertTrue(all(0. < c.args[0][2] <= .15 for c in nav.vehicle.send_velocity.call_args_list))

    def test_unknown_or_vehicle_below_prevents_descent(self):
        for state in (-1, 1):
            nav, _, source = self.descent_fixture()
            source.states[:, :, 24] = state
            self.assertFalse(FrontierExplorer._try_restore_cruise_height(nav, np.array([6., 0., -2.3])))
            nav.vehicle.send_velocity.assert_not_called()

    def test_unreachable_lower_goal_keeps_upper_height(self):
        nav, _, _ = self.descent_fixture()
        grid = Mock(); grid.path.return_value = None
        with patch('navigation.frontier_explorer.NavigationGrid.from_global', return_value=grid):
            self.assertFalse(FrontierExplorer._try_restore_cruise_height(nav, np.array([6., 0., -2.3])))
        nav.vehicle.send_velocity.assert_not_called()

    def test_no_repeated_probe_without_horizontal_progress(self):
        nav, position, _ = self.descent_fixture()
        nav._height_probe_position = position.copy()
        with patch('navigation.frontier_explorer.NavigationGrid.from_global') as build:
            self.assertFalse(FrontierExplorer._try_restore_cruise_height(nav, np.array([6., 0., -2.3])))
        build.assert_not_called()

    def test_changed_clearance_stops_descent_and_retains_actual_height(self):
        nav, position, _ = self.descent_fixture()
        grid = Mock(); grid.path.return_value = [position]
        goal = np.array([6., 0., -2.3])
        with patch('navigation.frontier_explorer.NavigationGrid.from_global', return_value=grid), \
             patch('navigation.height_clearance.vertical_corridor_free', side_effect=[True, True, False]):
            self.assertTrue(FrontierExplorer._try_restore_cruise_height(nav, goal))
        self.assertGreater(position[2], -2.3)
        self.assertLess(position[2], -2.2)
        self.assertEqual(goal[2], position[2])
        nav.vehicle.hover.assert_called_once()

    def test_missing_lower_frontier_can_trigger_upper_recovery(self):
        source=SimpleNamespace(states=np.zeros((40,40,40),dtype=np.int8),
                               origin_ned=np.array([-2.,-2.,-4.]),resolution=.1)
        nav=SimpleNamespace(vehicle=SimpleNamespace(get_position=lambda:np.array([0.,0.,-1.5])),
             local_map=SimpleNamespace(snapshot=lambda:source), global_map=Mock(),
             altitude=-1.5,_nominal_altitude=-1.5,config=SemanticNavigationConfig(),
             exploration_config=FrontierExplorationConfig(),exploration_origin=np.zeros(3),
             _try_raise_for_obstacle=Mock(return_value=True),_reset_empty_frontier_evidence=Mock(),
             _event=Mock())
        lower=Mock();lower.path.return_value=None
        upper=Mock();upper.frontier.return_value=(np.array([2.,0.,-1.9]), [])
        with patch('navigation.frontier_explorer.NavigationGrid.from_global',return_value=upper):
            self.assertTrue(FrontierExplorer._try_upper_frontier(nav,lower))
        np.testing.assert_allclose(nav._try_raise_for_obstacle.call_args.args[0],[2.,0.,-1.5])
        nav._reset_empty_frontier_evidence.assert_called_once()

    def test_bounded_levels_include_final_partial_step(self):
        np.testing.assert_allclose(ascent_levels(-1.5,-1.5,.4,1.),[-1.9,-2.3,-2.5])
        self.assertEqual(ascent_levels(-2.5,-1.5,.4,1.),[])
        self.assertEqual(ascent_levels(-1.5,-1.5,.4,0.),[])

    def test_ceiling_and_unknown_column_block_ascent(self):
        states=np.zeros((40,40,40),dtype=np.int8)
        source=SimpleNamespace(states=states,origin_ned=np.array([-2.,-2.,-4.]),resolution=.1)
        start,end=np.array([0.,0.,-1.5]),np.array([0.,0.,-2.3])
        self.assertTrue(vertical_corridor_free(source,start,end,.35,.25))
        for value in [1,-1]:
            states[:,:,20]=value
            self.assertFalse(vertical_corridor_free(source,start,end,.35,.25))

    def test_tall_step_can_use_second_height_candidate(self):
        position=np.array([0.,0.,-1.5]); velocity=np.zeros(3); clock=[0.]
        states=np.zeros((40,40,40),dtype=np.int8)
        states[26,20,23]=1
        source=SimpleNamespace(states=states,origin_ned=np.array([-2.,-2.,-4.]),resolution=.1)
        vehicle=Mock()
        vehicle.get_position.side_effect=lambda:position.copy()
        vehicle.get_velocity.return_value=np.zeros(3)
        def send(v): velocity[:]=v
        def wait(dt):
            position[:]+=velocity*dt
            clock[0]+=dt
        vehicle.send_velocity.side_effect=send
        vehicle.wait.side_effect=wait
        nav=SimpleNamespace(vehicle=vehicle,local_map=SimpleNamespace(snapshot=lambda:source),
             global_map=Mock(),altitude=-1.5,_nominal_altitude=-1.5,
             config=SemanticNavigationConfig(),exploration_config=FrontierExplorationConfig(),
             _height_unavailable=Mock(return_value=False),_event=Mock(),_check_health=lambda:None,
             deadline=100,clock=lambda:clock[0],_motion_progress=None,memory=SimpleNamespace(),_guard=SimpleNamespace())
        grid=Mock();grid.path.return_value=[np.zeros(3)]
        with patch('navigation.frontier_explorer.NavigationGrid.from_global',return_value=grid) as build:
            self.assertTrue(FrontierExplorer._try_raise_for_obstacle(nav,np.array([2.,0.,-1.5])))
        self.assertAlmostEqual(build.call_args.args[1],-2.3)
        self.assertLess(position[2],-2.24)


if __name__ == '__main__': unittest.main()
