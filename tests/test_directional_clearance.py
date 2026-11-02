import unittest
from types import SimpleNamespace
import numpy as np
from planner.directional_clearance import motion_forward_clearance


class DirectionalClearanceTests(unittest.TestCase):
    def test_retreat_only_without_forward_velocity(self):
        pose = np.eye(4)
        pose[:3, 2] = [1, 0, 0]
        snapshot = SimpleNamespace(forward_clearance=.3,
                                   observation=SimpleNamespace(camera_pose_world=pose))
        self.assertTrue(np.isinf(motion_forward_clearance(snapshot, [0,0,0], [-2,0,0], [0,0,0])))
        for target, velocity in [([2,0,0],[0,0,0]), ([0,2,0],[0,0,0]),
                                 ([-2,0,0],[.1,0,0])]:
            self.assertEqual(motion_forward_clearance(snapshot,[0,0,0],target,velocity),.3)

    def test_missing_pose_preserves_stop(self):
        self.assertEqual(motion_forward_clearance(SimpleNamespace(forward_clearance=.3),
                                                  [0,0,0],[-2,0,0],[0,0,0]),.3)
