from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
import numpy as np
from navigation.height_clearance import ascent_levels, vertical_corridor_free
from navigation.frontier_explorer import FrontierExplorer
from config.semantic_navigation_config import SemanticNavigationConfig
from config.exploration_config import FrontierExplorationConfig


class StairTests(unittest.TestCase):
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
