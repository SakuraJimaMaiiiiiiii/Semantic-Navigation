import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from localization.fastlio_semantics import FastLioSemantics
from mapping.loop_semantic_replay import transform_measurement


class SemanticAlignmentTests(unittest.TestCase):
    def test_full_rotation_translation_covariance(self):
        bridge = FastLioSemantics()
        a = np.eye(4); b = np.eye(4)
        b[:3,:3] = Rotation.from_euler('x', 90, degrees=True).as_matrix()
        b[:3,3] = [2, 0, 0]
        bridge.add_pose(10., a); bridge.add_pose(10.2, b)
        t = bridge.correction_at(10.1)
        expected = Rotation.from_euler('x', 45, degrees=True).as_matrix()
        np.testing.assert_allclose(t[:3,:3], expected, atol=1e-12)
        sample = dict(position_world=[0,1,0], bbox_3d_min=[0,0,0],
                      bbox_3d_max=[1,1,1], measurement_covariance=np.diag([1,2,3]),
                      surface_points_world=[[0,1,0]])
        transformed = transform_measurement(sample,t)
        np.testing.assert_allclose(transformed.position_world, expected@[0,1,0]+[1,0,0])
        np.testing.assert_allclose(transformed.surface_points_world[0], transformed.position_world)
        np.testing.assert_allclose(transformed.measurement_covariance, expected@np.diag([1,2,3])@expected.T)

    def test_no_extrapolation_or_interpolation_across_gap(self):
        bridge = FastLioSemantics()
        bridge.add_pose(10., np.eye(4));bridge.add_pose(11., np.eye(4))
        for stamp in (9.,10.5,12.):
            self.assertIsNone(bridge.correction_at(stamp))

    def test_pending_drain_and_closed_sink(self):
        bridge = FastLioSemantics()
        bridge.submit(dict(timestamp=10.1,samples=[]))
        bridge.add_pose(10., np.eye(4))
        self.assertEqual(len(bridge.pending),1)
        bridge.add_pose(10.2,np.eye(4))
        self.assertEqual(bridge.counts['matched_frames'],1)
        bridge.submit(dict(timestamp=12.,samples=[]))
        bridge.drain(final=True)
        self.assertEqual(bridge.counts['unmatched_time'],1)
        bridge.submit(dict(timestamp=13.,samples=[]))
        self.assertEqual(len(bridge.pending),0)

    def test_delayed_pose_does_not_drop_semantic_frames(self):
        bridge = FastLioSemantics()
        bridge.add_pose(10., np.eye(4))
        for i in range(200):
            bridge.submit(dict(timestamp=10.+.001*(i+1), samples=[]))
        self.assertEqual(len(bridge.pending), 200)
        bridge.add_pose(10.3, np.eye(4))
        self.assertEqual(bridge.counts['matched_frames'], 200)
        self.assertEqual(bridge.counts['pending_peak'], 200)
