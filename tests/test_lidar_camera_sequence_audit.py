from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from tools.audit_lidar_camera_sequence import audit


class LidarCameraSequenceAuditTests(unittest.TestCase):
    def test_detects_forward_extrinsic_error(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run/'fastlio').mkdir()
            (run/'recordings').mkdir()
            with h5py.File(run/'fastlio/inputs.h5', 'w') as lidar:
                scan = lidar.create_group('scans').create_group('00000000')
                scan.attrs['shared_timestamp'] = 1.
                scan.create_dataset('px4_pose', data=[0.,0.,0.,1.,0.,0.,0.])
                scan.create_dataset('points_body', data=[[0.,0.,2.],[.1,0.,2.]])
            with h5py.File(run/'recordings/flight_rgbd_test.h5', 'w') as camera:
                camera.create_dataset('timestamps/timestamp', data=[1.])
                camera.create_dataset('camera/intrinsics', data=[[10.,0.,5.],[0.,10.,5.],[0.,0.,1.]])
                camera.create_dataset('camera/extrinsics', data=np.eye(4))
                camera.create_dataset('state/quaternion_wxyz', data=[[1.,0.,0.,0.]])
                camera.create_dataset('state/position_xyz', data=[[0.,0.,0.]])
                camera.create_dataset('sensors/depth', data=np.full((1,10,10),2.))
            correct = audit(run)
            with h5py.File(run/'recordings/flight_rgbd_test.h5', 'r+') as camera:
                extrinsics = np.eye(4)
                extrinsics[2,3] = .5
                camera['camera/extrinsics'][:] = extrinsics
            shifted = audit(run)
            self.assertEqual(correct['matched_pairs'], 1)
            self.assertAlmostEqual(correct['all_point_depth_residual_m']['median'], 0.)
            self.assertAlmostEqual(shifted['all_point_depth_residual_m']['median'], .5)
            self.assertFalse(correct['scan_model_verified'])


if __name__ == '__main__':
    unittest.main()
