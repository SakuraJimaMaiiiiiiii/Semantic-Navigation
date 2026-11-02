"""Corrected historical geometry expressed in the current PX4 control frame."""
from collections import deque
from dataclasses import replace
import threading
import numpy as np
from .global_sparse_occupancy_map import GlobalSparseOccupancyMap


class OnlineLoopNavigationMap:
    def __init__(self, raw):
        self.raw = raw
        self.config = raw.config
        self._lock = threading.RLock()
        self._recent = deque(maxlen=256)
        self._state = (raw, 0)

    @property
    def revision(self):
        return self._state[1]

    @property
    def last_observation_completed_at(self):
        return self._state[0].last_observation_completed_at

    def snapshot_with_revision(self):
        world, revision = self._state
        return world.snapshot(), revision

    def snapshot(self):
        return self.snapshot_with_revision()[0]

    def integrate_lidar_points(self, origin, points, timestamp, position):
        self.raw.integrate_lidar_points(origin, points, timestamp, position)
        with self._lock:
            self._recent.append((origin.copy(), points.copy(), timestamp, position.copy()))
            active = self._state[0]
            if active is not self.raw:
                active.integrate_lidar_points(origin, points, timestamp, position)

    def publish(self, frames, corrections, check_cancelled=None):
        """Recast rays, not rotated voxel centers: preserve observed free-space evidence."""
        transform = np.linalg.inv(corrections[-1])
        world = GlobalSparseOccupancyMap(replace(self.config, vehicle_free_radius=0., visualize=False))
        cutoff = frames[-1].timestamp
        # Check history coverage before spending time on a reconstruction.
        with self._lock:
            initial_oldest = self._recent[0][2] if self._recent else cutoff
        if initial_oldest > cutoff:
            return False
        for frame in frames:
            if check_cancelled is not None:
                check_cancelled()
            pose = transform @ frame.corrected_camera_pose
            points = frame.points_camera @ pose[:3,:3].T + pose[:3,3]
            world.integrate_lidar_points(pose[:3,3], points, frame.timestamp, pose[:3,3])
        # Catch up outside the shared lock. Replaying a backlog while holding
        # it blocks the LiDAR worker's global update, hence its NEXT local frame.
        # A busy map may reject this revision rather than stall flight mapping.
        for _ in range(8):
            if check_cancelled is not None:
                check_cancelled()
            with self._lock:
                if self._recent and self._recent[0][2] > cutoff:
                    return False
                pending = tuple(sample for sample in self._recent if sample[2] > cutoff)
                if not pending:
                    self._state = (world, self.revision + 1)
                    return True
            for origin, points, stamp, position in pending:
                if check_cancelled is not None:
                    check_cancelled()
                world.integrate_lidar_points(origin, points, stamp, position)
            cutoff = pending[-1][2]
        return False
