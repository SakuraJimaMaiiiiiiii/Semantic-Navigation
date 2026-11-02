import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from navigation.vehicle_inspection import needs_inspection, observation_route
from navigation.frontier_explorer import FrontierExplorer


class VehicleInspectionTests(unittest.TestCase):
    def record(self):
        return dict(class_name='car', runtime_instance_id='vehicle_01', confidence=.9,
                    last_seen=10., position_world=[0.,0.,0.],
                    bbox_3d={'min':[-2.,-1.,-1.], 'max':[2.,1.,0.]},
                    position_status='partial_surface', position_covariance=np.eye(3)*.01)

    def test_only_incomplete_or_uncertain_vehicles(self):
        r=self.record()
        self.assertTrue(needs_inspection(r,.4))
        r['position_status']='multi_surface_estimate'
        self.assertFalse(needs_inspection(r,.4))
        r['position_covariance']=np.eye(3)
        self.assertTrue(needs_inspection(r,.4))
        r['class_name']='column'
        self.assertFalse(needs_inspection(r,.4))

    def test_candidates_require_existing_free_astar_routes(self):
        grid=SimpleNamespace(cell=lambda p:0, free={0}, path=Mock(return_value=None))
        self.assertIsNone(observation_route(grid,np.array([4.,0.,-1.5]),self.record(),-1.5,1.2,[]))
        grid.path.side_effect=lambda a,b:np.array([a,b])
        route=observation_route(grid,np.array([4.,0.,-1.5]),self.record(),-1.5,1.2,[])
        self.assertEqual(route[1][2],-1.5)
        self.assertGreater(abs(route[3]),.5)
        grid.free.clear()
        self.assertIsNone(observation_route(grid,np.array([4.,0.,-1.5]),self.record(),-1.5,1.2,[]))

    def test_unreachable_vehicle_is_not_repeated(self):
        n=object.__new__(FrontierExplorer)
        n._inspected_vehicle_ids=set()
        n.vehicle=Mock();n.vehicle.get_position.return_value=np.array([4.,0.,-1.5])
        n._frame=lambda:SimpleNamespace(timestamp=10.,object_inspection_candidates=[self.record()])
        n.config=SimpleNamespace(perception_timeout=5.,clearance_radius=.4,standoff_distance=1.2)
        n.request=None  # Exploration has no semantic navigation target.
        n.exploration_config=SimpleNamespace(information_radius=3.,candidate_radius_max=2.5,
                                            vehicle_inspection_timeout=45.,vehicle_inspection_views=2)
        n.clock=lambda:10.;n.deadline=100.;n._motion_progress=None;n.altitude=-1.5
        n._check_health=Mock();n._checkpoint=Mock()
        n._grid=Mock(return_value=SimpleNamespace(free=set(),cell=lambda p:0))
        n._inspect_nearby_vehicle()
        self.assertEqual(n.deadline,100.)
        self.assertIn('vehicle_01',n._inspected_vehicle_ids)
        n._inspect_nearby_vehicle()
        self.assertEqual(n._grid.call_count,1)
