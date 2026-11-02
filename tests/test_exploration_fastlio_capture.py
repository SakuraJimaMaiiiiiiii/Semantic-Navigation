from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
import threading
import time
import numpy as np

from mapping.explore_v3.lidar_mapping_worker import LidarMappingWorker


class ExplorationFastLioCaptureTests(unittest.TestCase):
    def test_both_mapping_threads_start_and_stop(self):
        frame = SimpleNamespace(timestamp=1., points_body=np.array([[1., 0., 0.]]))
        state = SimpleNamespace(position_xyz=np.zeros(3), quaternion=np.array([1., 0., 0., 0.]))
        reader = Mock()
        def read():
            time.sleep(.005)
            return frame
        reader.read.side_effect = read
        stream = SimpleNamespace(_state_at=Mock(return_value=(state, .01)), max_sync_error=.05)
        local, world = Mock(), Mock()
        with patch('mapping.explore_v3.lidar_mapping_worker.RflyLidarReader', return_value=reader):
            worker = LidarMappingWorker(Mock(), stream, local, world,
                                        SimpleNamespace(update_rate=100., stale_timeout=3.))
        worker.start(timeout=2.)
        worker.close()
        local.integrate_lidar_points.assert_called()
        world.integrate_lidar_points.assert_called()
        self.assertFalse(worker._thread.is_alive())
        self.assertFalse(worker._global_thread.is_alive())
        self.assertFalse(worker._acquisition_thread.is_alive())

    def test_slow_global_fusion_does_not_block_local_updates(self):
        global_entered, release_global, local_updated = threading.Event(), threading.Event(), threading.Event()
        global_calls, local_calls = [], []
        def global_integrate(origin, points, stamp, position):
            global_calls.append(stamp)
            if len(global_calls) == 1:
                global_entered.set()
                if not release_global.wait(5.):
                    raise TimeoutError('global test worker blocked')
        def local_integrate(origin, points, stamp, position):
            local_calls.append(stamp)
            if len(local_calls) >= 2:
                local_updated.set()
        local = SimpleNamespace(integrate_lidar_points=local_integrate)
        world = SimpleNamespace(integrate_lidar_points=global_integrate)
        with patch('mapping.explore_v3.lidar_mapping_worker.RflyLidarReader'):
            worker = LidarMappingWorker(Mock(), Mock(), local, world,
                                        SimpleNamespace(update_rate=1000., stale_timeout=3.))
        threads = [threading.Thread(target=worker._integrate, args=(name,)) for name in ('local', 'global')]
        for thread in threads:
            thread.start()
        try:
            sample = (np.zeros(3), np.ones((3, 3)), 1.)
            with worker._cloud_condition:
                worker._pending_cloud = worker._pending_global_cloud = sample
                worker._cloud_condition.notify_all()
            self.assertTrue(global_entered.wait(2.))
            deadline = time.monotonic()+2.
            while not local_calls and time.monotonic() < deadline:
                time.sleep(.01)
            with worker._cloud_condition:
                worker._pending_cloud = worker._pending_global_cloud = (sample[0], sample[1], 2.)
                worker._cloud_condition.notify_all()
            self.assertTrue(local_updated.wait(2.))
            self.assertEqual(local_calls, [1., 2.])
            self.assertEqual(global_calls, [1.])
            release_global.set()
            deadline = time.monotonic()+2.
            while len(global_calls) < 2 and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(global_calls, [1., 2.])
        finally:
            release_global.set()
            worker._stop.set()
            with worker._cloud_condition:
                worker._cloud_condition.notify_all()
            for thread in threads:
                thread.join(5.)
                self.assertFalse(thread.is_alive())

    def test_only_synchronized_valid_scans_reach_recorder_and_existing_map(self):
        frame = SimpleNamespace(timestamp=1., points_body=np.array([[1., 0., 0.]]))
        state = SimpleNamespace(position_xyz=np.array([2., 3., -1.]),
                                quaternion=np.array([1., 0., 0., 0.]))
        reader = Mock()
        stream = SimpleNamespace(_state_at=Mock(return_value=(state, .01)), max_sync_error=.05)
        with patch('mapping.explore_v3.lidar_mapping_worker.RflyLidarReader', return_value=reader):
            worker = LidarMappingWorker(Mock(), stream, Mock(), Mock(), Mock())
        worker.fastlio_recorder = Mock()
        def read():
            if reader.read.call_count == 1:
                return frame
            worker._stop.set()
            return None
        reader.read.side_effect = read
        worker._acquire()
        worker.fastlio_recorder.submit_scan.assert_called_once_with(frame, state, .01)
        self.assertEqual(worker.alignment_statistics['received_lidar_frames'], 1)
        self.assertEqual(worker.alignment_statistics['accepted_lidar_frames'], 1)
        origin, points, stamp = worker._pending_cloud
        np.testing.assert_allclose(origin, [2., 3., -1.])
        np.testing.assert_allclose(points, [[3., 3., -1.]])
        self.assertEqual(stamp, 1.)

    def test_unsynchronized_scan_is_not_recorded(self):
        frame = SimpleNamespace(timestamp=1., points_body=np.array([[1., 0., 0.]]))
        reader = Mock()
        stream = SimpleNamespace(_state_at=Mock(return_value=(Mock(), .1)), max_sync_error=.05)
        with patch('mapping.explore_v3.lidar_mapping_worker.RflyLidarReader', return_value=reader):
            worker = LidarMappingWorker(Mock(), stream, Mock(), Mock(), Mock())
        worker.fastlio_recorder = Mock()
        def read():
            if reader.read.call_count == 1:
                return frame
            worker._stop.set()
            return None
        reader.read.side_effect = read
        worker._acquire()
        worker.fastlio_recorder.submit_scan.assert_not_called()
        self.assertEqual(worker.alignment_statistics['received_lidar_frames'], 1)
        self.assertEqual(worker.alignment_statistics['accepted_lidar_frames'], 0)
        self.assertIsNone(worker._pending_cloud)


if __name__ == '__main__':
    unittest.main()
