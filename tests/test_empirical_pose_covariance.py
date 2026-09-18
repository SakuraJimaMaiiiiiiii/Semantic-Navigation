import unittest
from types import SimpleNamespace

import numpy as np

from sensors.empirical_pose_covariance import (
    pose_error, estimate_profile, position_covariance_from_odometry,
    combine_pose_covariance,
)


class EmpiricalPoseCovarianceTest(unittest.TestCase):
    def test_quaternion_sign_and_yaw_wrap(self):
        def yaw(deg):
            a = np.deg2rad(deg)/2
            return [np.cos(a), 0, 0, np.sin(a)]
        a = pose_error([0]*3, yaw(179), [1, 2, 3], yaw(-179))
        b = pose_error([0]*3, yaw(179), [1, 2, 3], -np.array(yaw(-179)))
        np.testing.assert_allclose(a, b)
        np.testing.assert_allclose(a, [1, 2, 3, 0, 0, np.deg2rad(2)])

    def test_bias_is_not_discarded(self):
        e = np.zeros((100, 6)); e[:, 0] = 2
        p = estimate_profile(e)
        self.assertEqual(p['covariance'][0][0], 0)
        self.assertEqual(p['effective_covariance'][0][0], 4)
        with self.assertRaises(ValueError):
            estimate_profile(e[:10])

    def test_stationary_truth_error_not_vehicle_motion(self):
        errors = [pose_error([x, 0, 0], [1, 0, 0, 0], [x+.2, 0, 0], [1, 0, 0, 0]) for x in range(100)]
        p = estimate_profile(errors)
        self.assertAlmostEqual(p['effective_covariance'][0][0], .04)

    def test_position_validity_and_zero_attitude(self):
        values = np.zeros(21); values[[0, 6, 11]] = [.01, .02, .03]
        msg = SimpleNamespace(frame_id=1, estimator_type=8, _timestamp=10, pose_covariance=values)
        np.testing.assert_allclose(position_covariance_from_odometry(msg, 10.1), np.diag([.01, .02, .03]))
        self.assertIsNone(position_covariance_from_odometry(msg, 11))
        self.assertIsNone(position_covariance_from_odometry(msg, 9))
        values[0] = np.nan
        self.assertIsNone(position_covariance_from_odometry(msg, 10.1))

    def test_profile_cross_terms_remain_psd(self):
        v = np.arange(1, 7)*.01
        base = np.outer(v, v)
        output = combine_pose_covariance(base, np.eye(3), calibrated=True)
        self.assertGreaterEqual(np.linalg.eigvalsh(output).min(), -1e-12)
        np.testing.assert_allclose(output[:3, 3:], base[:3, 3:])
        np.testing.assert_allclose(np.diag(output)[:3], 1)

    def test_decode_sdk_truth_format(self):
        import struct
        from tools.calibrate_pose_covariance import decode_truth
        values = [123456789, 1, 3, 0] + [0.]*24 + [1., 2., 3., 4., 0., 0., 0.]
        values[10] = 1.
        data = struct.pack('<4i24f7d', *values)
        stamp, pos, q = decode_truth(data, 1)
        np.testing.assert_allclose(pos, [2, 3, 4])
        np.testing.assert_allclose(q, [1, 0, 0, 0])
        self.assertIsNone(decode_truth(data, 2))


if __name__ == '__main__':
    unittest.main()
