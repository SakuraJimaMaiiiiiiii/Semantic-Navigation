"""局部三维占据栅格与观测扇出的无仿真测试。"""

import tempfile
import time
import unittest
from pathlib import Path

import h5py
import numpy as np

from config import (
    GlobalSparseMapConfig,
    OccupancyGridConfig,
)
from mapping import (
    FREE,
    OCCUPIED,
    GlobalSparseOccupancyMap,
    LocalOccupancyGrid,
    PointCloudLoopClosure,
)
from sensors import (
    SensorFrame,
    SynchronizedObservation,
    SynchronizedObservationHub,
)
from vehicle import DroneState


def _make_plane_observation(timestamp=1.0, position=None):
    depth = np.full((12, 16), 2.0, dtype=np.float32)
    intrinsics = np.asarray(
        [[10.0, 0.0, 7.5], [0.0, 10.0, 5.5], [0.0, 0.0, 1.0]]
    )
    # 相机光学(x右,y下,z前)转机体FRD/世界NED(x前,y右,z下)。
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    state = DroneState(
        timestamp=timestamp,
        position_xyz=(
            np.zeros(3)
            if position is None
            else np.asarray(position, dtype=float)
        ),
        quaternion=np.asarray([1.0, 0.0, 0.0, 0.0]),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        euler_rpy=np.zeros(3),
        px4_boot_timestamp=timestamp,
    )
    frame = SensorFrame(
        timestamp=timestamp,
        rgb=np.zeros((12, 16, 3), dtype=np.uint8),
        depth=depth,
        camera_intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
        rgb_timestamp=timestamp,
        depth_timestamp=timestamp,
        received_at=timestamp,
    )
    return SynchronizedObservation(state, frame, state_sync_error=0.0)


class LocalOccupancyGridTest(unittest.TestCase):
    def test_depth_plane_creates_free_rays_and_occupied_surface(self):
        grid = LocalOccupancyGrid(
            OccupancyGridConfig(
                size_ned=(6.0, 6.0, 4.0),
                resolution=0.2,
                depth_stride=1,
                min_depth=0.2,
                max_depth=3.0,
                inflation_radius=0.4,
            )
        )
        observation = _make_plane_observation()
        grid.update(observation)
        grid.update(observation)
        snapshot = grid.snapshot()

        self.assertGreater(np.count_nonzero(snapshot.states == FREE), 0)
        self.assertGreater(np.count_nonzero(snapshot.states == OCCUPIED), 0)
        occupied_indices = np.argwhere(snapshot.states == OCCUPIED)
        occupied = (
            snapshot.origin_ned
            + (occupied_indices.astype(np.float64) + 0.5)
            * snapshot.resolution
        )
        self.assertTrue(np.all((occupied[:, 0] > 1.8) & (occupied[:, 0] < 2.2)))
        self.assertGreater(
            np.count_nonzero(snapshot.inflated_occupied),
            np.count_nonzero(snapshot.states == OCCUPIED),
        )

class GlobalSparseOccupancyMapTest(unittest.TestCase):
    def test_old_world_voxels_remain_after_drone_moves(self):
        global_map = GlobalSparseOccupancyMap(
            GlobalSparseMapConfig(
                resolution=0.2,
                update_rate=5.0,
                depth_stride=1,
                min_depth=0.2,
                max_depth=3.0,
                ray_step=0.2,
                visualize=False,
                save_on_close=False,
            )
        )
        global_map.update(
            _make_plane_observation(
                timestamp=1.0,
                position=[0.0, 0.0, 0.0],
            )
        )
        global_map.update(
            _make_plane_observation(
                timestamp=2.0,
                position=[4.0, 0.0, 0.0],
            )
        )
        occupied = global_map.snapshot().occupied_points_ned

        self.assertTrue(np.any(
            (occupied[:, 0] > 1.8) & (occupied[:, 0] < 2.2)
        ))
        self.assertTrue(np.any(
            (occupied[:, 0] > 5.8) & (occupied[:, 0] < 6.2)
        ))

        with tempfile.TemporaryDirectory() as directory:
            output_path = global_map.save(
                Path(directory) / "global_map.h5"
            )
            ply_path = global_map.export_occupied_ply(
                Path(directory) / "global_map_occupied.ply"
            )
            dense_ply_path = global_map.export_dense_occupied_ply(
                Path(directory) / "global_map_occupied_dense.ply",
                subdivisions=4,
                max_points=1_500_000,
            )
            with h5py.File(output_path, mode="r") as h5_file:
                self.assertEqual(
                    h5_file["voxel_indices_ned"].shape[1],
                    3,
                )
                self.assertEqual(
                    h5_file["log_odds"].shape[0],
                    global_map.snapshot().voxel_count,
                )
            ply_bytes = ply_path.read_bytes()
            header_end = ply_bytes.index(b"end_header\n") + len(
                b"end_header\n"
            )
            self.assertIn(
                b"format binary_little_endian 1.0",
                ply_bytes[:header_end],
            )
            ply_dtype = np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                    ("log_odds", "<f4"),
                ]
            )
            vertices = np.frombuffer(
                ply_bytes[header_end:],
                dtype=ply_dtype,
            )
            expected_ned = global_map.snapshot().occupied_points_ned
            expected_neu = expected_ned.copy()
            expected_neu[:, 2] *= -1.0
            exported_neu = np.column_stack(
                (vertices["x"], vertices["y"], vertices["z"])
            )
            self.assertTrue(np.allclose(exported_neu, expected_neu))
            dense_bytes = dense_ply_path.read_bytes()
            dense_header_end = dense_bytes.index(
                b"end_header\n"
            ) + len(b"end_header\n")
            dense_header = dense_bytes[:dense_header_end]
            # A 4 x 4 x 4 grid has 56 samples on its outer surface.
            expected_dense_count = expected_neu.shape[0] * 56
            self.assertIn(
                f"element vertex {expected_dense_count}".encode("ascii"),
                dense_header,
            )
            dense_vertices = np.frombuffer(
                dense_bytes[dense_header_end:],
                dtype=ply_dtype,
            )
            self.assertEqual(dense_vertices.shape[0], expected_dense_count)

    def test_revisit_is_verified_as_point_cloud_loop(self):
        config = GlobalSparseMapConfig(
            resolution=0.2,
            update_rate=5.0,
            depth_stride=1,
            min_depth=0.2,
            max_depth=3.0,
            ray_step=0.2,
            visualize=False,
            save_on_close=False,
            keyframe_translation=0.5,
            loop_min_separation=4,
            loop_search_radius=0.6,
            loop_max_candidates=1,
            icp_voxel_size=0.2,
            icp_max_correspondence=0.7,
            icp_min_fitness=0.8,
            icp_max_rmse=0.1,
        )
        global_map = GlobalSparseOccupancyMap(config)
        loop_closure = PointCloudLoopClosure(global_map, config)

        # 最后一帧估计位置带有0.4m漂移，但相机重新看到了首帧平面。
        positions = [0.0, 1.0, 2.0, 1.0, 0.4]
        for index, north in enumerate(positions):
            loop_closure.process(_make_plane_observation(
                timestamp=float(index + 1),
                position=[north, 0.0, 0.0],
            ))

        self.assertEqual(len(loop_closure.keyframes), 5)
        self.assertEqual(loop_closure.loop_count, 1)
        self.assertLess(
            loop_closure.keyframes[-1].corrected_camera_pose[0, 3],
            loop_closure.keyframes[-1].raw_camera_pose[0, 3],
        )
        self.assertGreater(global_map.snapshot().occupied_count, 0)
        with tempfile.TemporaryDirectory() as directory:
            output_path = global_map.save(
                Path(directory) / "loop_corrected_map.h5"
            )
            loop_closure.write_hdf5_metadata(output_path)
            with h5py.File(output_path, mode="r") as h5_file:
                self.assertEqual(
                    h5_file["loop_closure/corrected_camera_poses"].shape,
                    (5, 4, 4),
                )
                self.assertEqual(
                    int(h5_file["loop_closure"].attrs[
                        "accepted_loop_count"
                    ]),
                    1,
                )


class _SlowFakeSource:
    def __init__(self):
        self.timestamp = 0.0

    def get_observation(self, timeout=1.0):
        time.sleep(0.01)
        self.timestamp += 1.0
        return _make_plane_observation(self.timestamp)


class ObservationHubTest(unittest.TestCase):
    def test_same_observation_is_fanned_out(self):
        hub = SynchronizedObservationHub(_SlowFakeSource())
        first = hub.subscribe("first", max_queue=4)
        second = hub.subscribe("second", max_queue=4)
        hub.start()
        try:
            observation_a = first.get_observation(timeout=0.5)
            observation_b = second.get_observation(timeout=0.5)
        finally:
            hub.close()
        self.assertIs(observation_a, observation_b)


if __name__ == "__main__":
    unittest.main()
