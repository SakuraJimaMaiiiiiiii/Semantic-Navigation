from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from sensors.synchronized_rgbd import SynchronizedRGBDStream


class PoseTelemetryRateTests(unittest.TestCase):
    def test_receiver_burst_preserves_intermediate_pose_packets(self):
        import queue
        import threading
        from collections import deque
        stream = object.__new__(SynchronizedRGBDStream)
        stream.sample_rate = 100.
        stream.max_sync_error = .03
        stream._stop = threading.Event()
        stream._condition = threading.Condition()
        stream._states = deque()
        stream._pose_packets = queue.Queue(maxsize=10)
        stream.pose_statistics = dict(received=0, overflow=0, accepted=0, source_skew_rejected=0)
        connection = SimpleNamespace(messages={})
        stream._pose_connection = connection
        message = SimpleNamespace(get_type=lambda: 'ATTITUDE')
        for stamp in (1., 1.02, 1.04):
            connection.messages = {'ATTITUDE': SimpleNamespace(timestamp=stamp),
                                   'LOCAL_POSITION_NED': object()}
            stream._capture_pose(connection, message)
        def read(*, pose_messages):
            stamp = pose_messages['ATTITUDE'].timestamp
            if stamp == 1.04:
                stream._stop.set()
            return SimpleNamespace(px4_boot_timestamp=stamp, timestamp=stamp, pose_sync_error=0.)
        stream.vehicle = SimpleNamespace(get_drone_state=read)
        stream._sample_loop()
        self.assertEqual([s.timestamp for s in stream._states], [1., 1.02, 1.04])
        self.assertEqual(stream.pose_statistics['accepted'], 3)

    def test_pose_messages_requested_and_restored(self):
        mav = SimpleNamespace(SendMavCmdLong=Mock())
        stream = object.__new__(SynchronizedRGBDStream)
        stream.vehicle = SimpleNamespace(connected=True, mav=mav)
        stream.camera = SimpleNamespace(config=SimpleNamespace(copter_id=1))
        stream.sample_rate = 100.
        stream._stop = Mock()
        stream._condition = __import__('threading').Condition()
        stream._thread = None
        stream._sample_loop = Mock()
        thread = Mock()
        thread.is_alive.return_value = False
        with patch('sensors.synchronized_rgbd.threading.Thread', return_value=thread):
            stream.start()
            stream.close()
        self.assertEqual(mav.SendMavCmdLong.call_args_list, [
            unittest.mock.call(511, 30, 20000), unittest.mock.call(511, 32, 20000),
            unittest.mock.call(511, 331, 100000), unittest.mock.call(511, 30, 0),
            unittest.mock.call(511, 32, 0), unittest.mock.call(511, 331, 0)])


if __name__ == '__main__':
    unittest.main()
