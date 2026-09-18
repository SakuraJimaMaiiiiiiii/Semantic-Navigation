"""用有限差分验证右乘旋转雅可比、杠杆臂和完整协方差传播。"""

import unittest
from types import SimpleNamespace
import numpy as np

from sensors.pose_uncertainty import (
    skew, camera_pose_covariance, world_point_pose_jacobian, world_point_covariance,
)


def rotation(vector):
    angle = np.linalg.norm(vector)
    if angle == 0:
        return np.eye(3)
    axis = skew(np.asarray(vector) / angle)
    return np.eye(3) + np.sin(angle) * axis + (1 - np.cos(angle)) * axis @ axis


class PoseUncertaintyTest(unittest.TestCase):
    def test_synchronized_stream_supplies_covariance_to_geometry(self):
        from config.camera_config import CameraConfig
        from sensors.synchronized_rgbd import SynchronizedRGBDStream
        from detections.object_detect import SAM2Segmenter, DetectorConfig
        frame = SimpleNamespace(timestamp=1., rgb_bgr=np.zeros((20, 20, 3), np.uint8),
                                depth_m=np.full((20, 20), 5.), rgb_timestamp=1.,
                                depth_timestamp=1., received_at=1.)
        intrinsics = np.array([[100., 0, 10], [0, 100., 10], [0, 0, 1.]])
        camera = SimpleNamespace(config=CameraConfig(), get_frame=lambda **_: frame,
                                 camera_intrinsics=intrinsics, camera_extrinsics=np.eye(4))
        stream = SynchronizedRGBDStream(object(), camera)
        stream._thread = SimpleNamespace(is_alive=lambda: True)
        state = SimpleNamespace(quaternion=np.array([1., 0, 0, 0]), position_xyz=np.zeros(3),
                                linear_velocity=np.ones(3), angular_velocity=np.zeros(3))
        stream._state_at = lambda *_: (state, 0.0)
        observation = stream.get_observation()
        self.assertEqual(observation.camera_pose_covariance.shape, (6, 6))
        segmenter = object.__new__(SAM2Segmenter)
        segmenter.config = DetectorConfig(device="cpu", instance_depth_stride=1)
        args = (np.ones((20, 20), bool), frame.depth_m, intrinsics, observation.camera_pose_world)
        before = segmenter._mask_world_geometry(*args)
        after = segmenter._mask_world_geometry(*args, observation.camera_pose_covariance)
        np.testing.assert_array_equal(before.position_world, after.position_world)
        representative = np.array([-0.025, -0.025, 5.])
        jacobian = world_point_pose_jacobian(np.eye(3), representative)
        np.testing.assert_allclose(after.measurement_covariance - before.measurement_covariance,
                                   jacobian @ observation.camera_pose_covariance @ jacobian.T,
                                   atol=1e-10)

    def test_point_pose_jacobian_matches_central_difference(self):
        rot = rotation([0.3, -0.2, 0.5])
        point = np.array([1., 2., 10.])
        eps = 1e-6
        numerical = np.empty((3, 6))
        for i in range(6):
            perturb = np.eye(6)[i] * eps
            plus = rot @ rotation(perturb[3:]) @ point + perturb[:3]
            minus = rot @ rotation(-perturb[3:]) @ point - perturb[:3]
            numerical[:, i] = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(world_point_pose_jacobian(rot, point), numerical, atol=1e-8)

    def test_lever_arm_and_full_cross_covariance_match_direct_body_derivative(self):
        rot = rotation([0.2, -0.3, 0.4])
        extr = np.eye(4)
        extr[:3, :3] = rotation([0.6, 0.2, -0.4])
        extr[:3, 3] = [0.4, -0.2, 0.1]
        point = np.array([1., -0.5, 8.])
        random = np.random.default_rng(7)
        factor = random.normal(size=(6, 6)) * 0.01
        body_cov = factor @ factor.T
        cam_cov = camera_pose_covariance(rot, extr, body_cov, np.zeros(3), np.zeros(3))
        propagated = world_point_covariance(rot @ extr[:3, :3], point, np.zeros((3, 3)), cam_cov)
        body_point = extr[:3, :3] @ point + extr[:3, 3]
        eps = 1e-6
        numerical = np.empty((3, 6))
        for i in range(6):
            perturb = np.eye(6)[i] * eps
            numerical[:, i] = (rot @ rotation(perturb[3:]) @ body_point + perturb[:3]
                               - rot @ rotation(-perturb[3:]) @ body_point + perturb[:3]) / (2 * eps)
        np.testing.assert_allclose(propagated, numerical @ body_cov @ numerical.T, atol=1e-9)
        self.assertGreater(np.linalg.norm(cam_cov[:3, 3:]), 0)

    def test_angular_uncertainty_grows_with_range_and_translation_is_additive(self):
        pose = np.diag([0.01] * 3 + [0.001] * 3)
        near = world_point_covariance(np.eye(3), [0, 0, 5], np.zeros((3, 3)), pose)
        far = world_point_covariance(np.eye(3), [0, 0, 10], np.zeros((3, 3)), pose)
        self.assertAlmostEqual(far[0, 0] - 0.01, 4 * (near[0, 0] - 0.01))
        self.assertAlmostEqual(far[2, 2], 0.01)

    def test_timestamp_error_keeps_velocity_rotation_cross_terms(self):
        rate = np.array([1., 2., 0., 0., 0., 0.5])
        actual = camera_pose_covariance(np.eye(3), np.eye(4), np.zeros((6, 6)),
                                        rate[:3], rate[3:], 0.02)
        np.testing.assert_allclose(actual, 0.02**2 * np.outer(rate, rate))

    def test_invalid_covariance_is_rejected_and_zero_pose_preserves_old_result(self):
        rot = rotation([0.1, 0.2, -0.3])
        covariance = np.diag([0.1, 0.2, 0.3])
        np.testing.assert_allclose(world_point_covariance(rot, [1, 2, 3], covariance, np.zeros((6, 6))),
                                   rot @ covariance @ rot.T)
        for invalid in (np.eye(3), -np.eye(6), np.full((6, 6), np.nan)):
            with self.assertRaises(ValueError):
                world_point_covariance(rot, [1, 2, 3], covariance, invalid)


if __name__ == "__main__":
    unittest.main()
