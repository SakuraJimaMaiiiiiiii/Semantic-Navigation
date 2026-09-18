"""HDF5 同步飞行数据记录的无仿真往返测试。"""

import tempfile
import time
import unittest
from pathlib import Path

import h5py
import numpy as np

from config import RecordingConfig
from recording import FlightDataReader, FlightDataRecorder
from sensors import SensorFrame, SynchronizedObservation
from vehicle import DroneState


class _FakeSynchronizedStream:
    def __init__(self):
        self.index = 0

    def get_observation(self, timeout=1.0):
        self.index += 1
        timestamp = time.time() + self.index * 1e-4
        state = DroneState(
            timestamp=timestamp,
            position_xyz=np.asarray([1.0, 2.0, -3.0]),
            quaternion=np.asarray([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.asarray([0.1, 0.2, 0.3]),
            angular_velocity=np.asarray([0.01, 0.02, 0.03]),
            euler_rpy=np.asarray([0.0, 0.0, 0.0]),
            px4_boot_timestamp=float(self.index),
        )
        frame = SensorFrame(
            timestamp=timestamp,
            rgb=np.full((8, 10, 3), self.index % 255, dtype=np.uint8),
            depth=np.full((8, 10), 2.5, dtype=np.float32),
            camera_intrinsics=np.asarray(
                [[5.0, 0.0, 4.5], [0.0, 5.0, 3.5], [0.0, 0.0, 1.0]]
            ),
            camera_extrinsics=np.eye(4),
            rgb_timestamp=timestamp,
            depth_timestamp=timestamp,
            received_at=timestamp,
        )
        return SynchronizedObservation(
            drone_state=state,
            sensor_frame=frame,
            state_sync_error=0.001,
        )


class HDF5FlightDataRecorderTest(unittest.TestCase):
    def test_hdf5_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FlightDataRecorder(
                _FakeSynchronizedStream(),
                RecordingConfig(
                    output_dir=Path(directory),
                    record_rate=30.0,
                    observation_timeout=0.1,
                    flush_interval=1,
                    hdf5_compression="gzip",
                    hdf5_compression_level=1,
                ),
            )
            recorder.start()
            time.sleep(0.12)
            recorder.close()
            recorder.raise_if_failed()

            self.assertGreater(recorder.frame_count, 0)
            self.assertEqual(recorder.output_path.suffix, ".h5")
            with FlightDataReader(recorder.output_path) as reader:
                rows = reader.read_state_rows()
                restored = reader.read_frame(rows[0]["frame_id"])
                batch = reader.read_batch(0, min(2, reader.frame_count))

            self.assertEqual(len(rows), recorder.frame_count)
            self.assertEqual(batch["rgb"].ndim, 4)
            np.testing.assert_array_equal(
                restored.rgb,
                np.full(
                    restored.rgb.shape,
                    restored.rgb[0, 0, 0],
                    dtype=np.uint8,
                ),
            )
            np.testing.assert_allclose(restored.depth, 2.5)
            np.testing.assert_allclose(
                restored.state["position_x"],
                1.0,
            )
            np.testing.assert_allclose(
                restored.camera_extrinsics,
                np.eye(4),
            )

    def test_expected_hdf5_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FlightDataRecorder(
                _FakeSynchronizedStream(),
                RecordingConfig(
                    output_dir=Path(directory),
                    flush_interval=1,
                    hdf5_compression="none",
                ),
            )
            recorder.start()
            time.sleep(0.05)
            recorder.close()
            recorder.raise_if_failed()

            with h5py.File(recorder.output_path, "r") as h5_file:
                self.assertIn("timestamps/timestamp", h5_file)
                self.assertIn("state/position_xyz", h5_file)
                self.assertIn("state/quaternion_wxyz", h5_file)
                self.assertIn("sensors/rgb", h5_file)
                self.assertIn("sensors/depth", h5_file)
                self.assertIn("camera/intrinsics", h5_file)
                self.assertIn("camera/extrinsics", h5_file)


if __name__ == "__main__":
    unittest.main()
