"""RGB-D 启动等待时间的回归测试。"""

import unittest
import threading
import time
from types import SimpleNamespace

import numpy as np

from application.demo_runtime import DemoRuntime
from sensors.rfly_rgbd_camera import RflyRGBDCamera
from sensors.synchronized_rgbd import SynchronizedRGBDStream


class _AliveThread:
    @staticmethod
    def is_alive():
        return True


class _Frame:
    timestamp = 123.0


class _Camera:
    def __init__(self):
        self.config = SimpleNamespace(
            state_sample_rate=100.0,
            state_history_seconds=5.0,
            max_state_sync_error=0.03,
            pose_position_std_ned=(0.1, 0.1, 0.15),
            pose_rotation_std_body_deg=(0.5, 0.5, 1.0),
            pose_timestamp_std=0.01,
        )
        self.timeouts = []

    def get_frame(self, timeout, after_timestamp=None):
        self.timeouts.append((float(timeout), after_timestamp))
        return _Frame()


class SynchronizedStartupTimeoutTest(unittest.TestCase):
    def test_state_timeout_does_not_retry_camera_with_zero_budget(self):
        camera = _Camera()
        stream = SynchronizedRGBDStream(object(), camera)
        stream._thread = _AliveThread()

        def state_timeout(_timestamp, _deadline):
            raise TimeoutError("No state sample after image timestamp.")

        stream._state_at = state_timeout

        with self.assertRaisesRegex(
            TimeoutError,
            "No state sample after image timestamp",
        ):
            stream.get_observation(timeout=0.2)

        self.assertEqual(len(camera.timeouts), 1)
        self.assertGreater(camera.timeouts[0][0], 0.1)

    def test_rejects_nonpositive_timeout(self):
        camera = _Camera()
        stream = SynchronizedRGBDStream(object(), camera)
        stream._thread = _AliveThread()

        with self.assertRaisesRegex(ValueError, "positive finite"):
            stream.get_observation(timeout=0.0)


class _StartComponent:
    def __init__(self):
        self.calls = 0

    def start(self):
        self.calls += 1


class _CameraStartComponent(_StartComponent):
    def __init__(self, vehicle):
        super().__init__()
        self.vehicle = vehicle
        self.vehicle_was_connected = False

    def start(self):
        self.vehicle_was_connected = self.vehicle.connect_calls > 0
        super().start()


class _Vehicle:
    def __init__(self):
        self.setup_calls = 0
        self.connect_calls = 0

    def setup_ue(self):
        self.setup_calls += 1

    def connect(self):
        self.connect_calls += 1


class _DataStream(_StartComponent):
    def __init__(self):
        super().__init__()
        self.observation_timeouts = []

    def get_observation(self, timeout):
        self.observation_timeouts.append(float(timeout))


class DemoRuntimeStartupTest(unittest.TestCase):
    def test_uses_camera_first_frame_timeout_for_sync_validation(self):
        runtime = DemoRuntime.__new__(DemoRuntime)
        runtime.vehicle = _Vehicle()
        runtime.camera = _CameraStartComponent(runtime.vehicle)
        runtime.camera.config = SimpleNamespace(first_frame_timeout=7.5)
        runtime.data_stream = _DataStream()
        runtime.recording_config = SimpleNamespace(enabled=False)
        runtime.mapping_config = SimpleNamespace(enabled=False)
        runtime.global_map_config = SimpleNamespace(enabled=False)
        runtime.detector_config = SimpleNamespace(enabled=False)
        runtime.observation_hub = _StartComponent()
        runtime.recorder = None
        runtime.mapping_worker = None
        runtime.global_mapping_worker = None
        runtime.detection_worker = None
        runtime._reset_started_flags()

        runtime.start()

        self.assertEqual(runtime.data_stream.observation_timeouts, [7.5])
        self.assertEqual(runtime.observation_hub.calls, 0)
        self.assertTrue(runtime.camera.vehicle_was_connected)


class _Capture:
    def __init__(self, timestamp):
        self.Img = [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2), dtype=np.float32),
            np.zeros((2, 2, 3), dtype=np.uint8),
        ]
        self.hasData = [True, True, True]
        self.Img_lock = [threading.Lock() for _ in self.Img]
        self.rflyStartStmp = [timestamp - 1.0] * 3
        self.timeStmp = [1.0] * 3


class CameraFreshFrameTest(unittest.TestCase):
    def _camera(self, timestamp):
        camera = RflyRGBDCamera.__new__(RflyRGBDCamera)
        camera.config = SimpleNamespace(
            max_rgb_depth_skew=0.02,
            poll_interval=0.001,
            depth_unit_scale=0.001,
            depth_min=0.3,
            depth_max=30.0,
        )
        camera._started = True
        camera._timestamp_offset = None
        camera._latest_frame_lock = threading.Lock()
        camera._rgb_capture_index = 0
        camera._depth_capture_index = 1
        camera._segmentation_capture_index = 2
        camera._capture = _Capture(timestamp)
        return camera

    def test_rejects_static_cached_frame_after_baseline(self):
        camera = self._camera(time.time() - 300.0)
        baseline = camera.get_frame(timeout=0.01)

        with self.assertRaisesRegex(TimeoutError, "No complete RGB-D"):
            camera.get_frame(
                timeout=0.01,
                after_timestamp=baseline.timestamp,
            )

    def test_calibrates_advancing_sdk_timestamp_with_fixed_offset(self):
        camera = self._camera(time.time() - 300.0)
        baseline = camera.get_frame(timeout=0.01)
        camera._capture.timeStmp = [1.05, 1.05, 1.05]

        current = camera.get_frame(
            timeout=0.01,
            after_timestamp=baseline.timestamp,
        )

        self.assertAlmostEqual(current.timestamp - baseline.timestamp, 0.05)
        self.assertLess(abs(current.received_at - current.timestamp), 0.1)

    def test_uses_per_frame_sdk_timestamp(self):
        camera = self._camera(time.time() - 300.0)
        camera._capture.rflyStartStmp = [1000.0, 1000.0, 1000.0]
        camera._capture.timeStmp = [10.0, 10.0, 10.0]
        baseline = camera.get_frame(timeout=0.01)

        camera._capture.timeStmp = [10.05, 10.05, 10.05]
        current = camera.get_frame(
            timeout=0.01,
            after_timestamp=baseline.timestamp,
        )

        self.assertAlmostEqual(current.timestamp - baseline.timestamp, 0.05)


class CameraSensorRequestTest(unittest.TestCase):
    @staticmethod
    def _sensor(seq_id, type_id):
        return SimpleNamespace(SeqID=seq_id, TypeID=type_id)

    def _camera(self, show_segmentation):
        camera = RflyRGBDCamera.__new__(RflyRGBDCamera)
        camera.config = SimpleNamespace(
            show_segmentation=show_segmentation,
        )
        camera._rgb_sensor = {"SeqID": 0}
        camera._depth_sensor = {"SeqID": 1}
        camera._segmentation_sensor = {"SeqID": 2}
        camera._capture = SimpleNamespace(
            VisSensor=[
                self._sensor(0, 1),
                self._sensor(1, 2),
                self._sensor(2, 4),
            ],
            VisSensorNew=[
                self._sensor(0, 1),
                self._sensor(1, 2),
                self._sensor(2, 4),
            ],
        )
        return camera

    def test_hidden_segmentation_is_not_requested(self):
        camera = self._camera(show_segmentation=False)
        camera._configure_requested_sensors()

        self.assertEqual(
            [sensor.TypeID for sensor in camera._capture.VisSensor],
            [1, 2],
        )
        self.assertEqual(
            [sensor.TypeID for sensor in camera._capture.VisSensorNew],
            [1, 2],
        )
        self.assertEqual(camera._rgb_capture_index, 0)
        self.assertEqual(camera._depth_capture_index, 1)
        self.assertIsNone(camera._segmentation_capture_index)

    def test_visible_segmentation_is_requested(self):
        camera = self._camera(show_segmentation=True)
        camera._configure_requested_sensors()

        self.assertEqual(
            [sensor.TypeID for sensor in camera._capture.VisSensor],
            [1, 2, 4],
        )
        self.assertEqual(camera._segmentation_capture_index, 2)


if __name__ == "__main__":
    unittest.main()
