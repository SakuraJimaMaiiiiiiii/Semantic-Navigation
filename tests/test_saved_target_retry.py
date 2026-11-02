from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
import numpy as np
from navigation.mission import SemanticNavigator
from navigation.target import SemanticTargetRequest
from config.semantic_navigation_config import SemanticNavigationConfig


class SavedTargetRetryTests(unittest.TestCase):
    def execute(self, results, require_visual_confirmation=True):
        clock=[0.]
        current=np.array([1.,2.,-1.5])
        record={'runtime_instance_id':'vehicle_03','position_world':[3.,2.,-.5]}
        frame=SimpleNamespace(timestamp=10,records=[record])
        world=Mock()
        world.snapshot.return_value=SimpleNamespace(timestamp=10)
        vehicle=Mock()
        vehicle.get_position.return_value=current
        perception=Mock()
        perception.semantic_snapshot.return_value=frame
        with TemporaryDirectory() as directory:
            nav=SemanticNavigator(vehicle,perception,world,world,Mock(),
                 SemanticTargetRequest(instance_id='vehicle_03'),SemanticNavigationConfig(),
                 Path(directory)/'result.json',clock=lambda:clock[0])
            nav.allow_search=False
            nav.require_visual_confirmation=require_visual_confirmation
            nav._sample_position=lambda *args:current
            nav._check_health=lambda:None
            nav._frame=lambda:frame
            nav._query=lambda *args:[record]
            nav._wait=lambda duration:clock.__setitem__(0,clock[0]+duration)
            nav._verify=Mock(side_effect=[record if result else None for result in results])
            grid=Mock()
            grid.approach.return_value=(0,current,[current])
            with patch('navigation.mission.NavigationGrid.from_global',return_value=grid):
                nav.run()
            return nav.result, nav._verify.call_count, clock[0]

    def test_cooldown_does_not_misreport_unreachable(self):
        result,calls,elapsed=self.execute([False,True])
        self.assertTrue(result['success'])
        self.assertEqual(calls,2)
        self.assertGreaterEqual(elapsed,15)

    def test_saved_arrival_succeeds_without_visual_confirmation(self):
        result,calls,_=self.execute([False], require_visual_confirmation=False)
        self.assertTrue(result['success'])
        self.assertTrue(result['navigation_arrived'])
        self.assertFalse(result['visual_confirmed'])
        self.assertEqual(result['reason'],'arrived_visual_unconfirmed')
        self.assertEqual(calls,1)

    def test_saved_arrival_preserves_visual_confirmation(self):
        result,calls,_=self.execute([True], require_visual_confirmation=False)
        self.assertTrue(result['visual_confirmed'])
        self.assertEqual(result['reason'],'visually_confirmed')

    def test_failed_confirmation_exhausts_bounded_budget(self):
        result,calls,elapsed=self.execute([False,False,False])
        self.assertFalse(result['success'])
        self.assertEqual(calls,3)
        self.assertEqual(result['reason'],'verification_attempts_exhausted')
        self.assertLess(elapsed,40)


if __name__ == '__main__': unittest.main()
