"""Read the LiDAR owned by the existing RGB-D capture; never open another SDK owner."""
from dataclasses import dataclass
import time
import numpy as np


@dataclass(frozen=True)
class LidarFrame:
    timestamp: float
    points_body: np.ndarray


def filter_lidar_points(points, min_range, max_range, max_points):
    data = np.asarray(points, dtype=float)
    if data.ndim != 2 or data.shape[1] not in (3,4):
        raise ValueError('LiDAR must contain Nx3 or Nx4 XYZ data')
    xyz = data[:, :3]
    distance = np.linalg.norm(xyz, axis=1)
    valid = np.isfinite(xyz).all(axis=1) & (distance >= min_range) & (distance < max_range-.01)
    xyz = xyz[valid]
    # Never turn missing/invalid/max-range returns into occupied endpoints or free rays.
    if len(xyz) > max_points:
        xyz = xyz[np.linspace(0, len(xyz)-1, max_points, dtype=int)]
    return xyz.copy()


class RflyLidarReader:
    def __init__(self, camera, config):
        self.camera, self.config = camera, config
        self.last_stamp = -float('inf')

    def read(self):
        capture = self.camera._capture
        matches = [i for i,s in enumerate(capture.VisSensor)
                   if int(s.SeqID) == self.config.seq_id and int(s.TypeID) == 20]
        if len(matches) != 1:
            raise RuntimeError('Requested LiDAR sensor is missing or duplicated')
        i = matches[0]
        if not capture.hasData[i] or self.camera._timestamp_offset is None:
            return None
        with capture.Img_lock[i]:
            before = self.camera._sensor_timestamp(i)
            points = np.array(capture.Img[i], copy=True)
            header = np.asarray(capture.ImgData[i], dtype=float).copy()
            after = self.camera._sensor_timestamp(i)
        if before <= 0 or before != after:
            return None
        stamp = before + self.camera._timestamp_offset
        if not np.isfinite(stamp) or stamp <= self.last_stamp:
            return None
        if header.shape != (9,) or not np.isfinite(header).all():
            raise RuntimeError('Unsupported LiDAR header (expected CopterID, axis, pose, count)')
        if int(header[0]) != self.camera.config.copter_id or int(header[1]) != 0:
            raise RuntimeError('LiDAR must use the selected copter and body FRD coordinates (axis=0)')
        if len(points) != int(header[8]):
            return None  # Header/point update in progress; never combine mismatched frames.
        self.last_stamp = stamp
        if abs(time.time()-stamp) > self.config.stale_timeout:
            return None
        xyz = filter_lidar_points(points, self.config.min_range, self.config.max_range, self.config.max_points)
        if len(xyz) == 0:
            return None
        if self.config.point_axes == "flu":
            xyz[:, 1:] *= -1.0
        elif self.config.point_axes != "frd":
            raise ValueError("LiDAR point_axes must be flu or frd")
        return LidarFrame(stamp, xyz)
