"""Time-aligned camera/LiDAR point-cloud fusion in camera optical coordinates."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AlignedLidar:
    points_camera: np.ndarray  # measured XYZ after image-time pose transformation
    rows: np.ndarray
    columns: np.ndarray
    reference_depth: np.ndarray
    consistent: np.ndarray
    time_skew: float
    usable: bool
    min_points: int

    @property
    def depth(self):
        """Read-only optical Z view of the measured points, not radial distance."""
        depth = self.points_camera[:, 2]
        depth.setflags(write=False)
        return depth

    def mask_points(self, mask, object_depth, depth_gate, *, mask_is_interior=False):
        mask = np.asarray(mask, dtype=bool)
        # Reject mask boundaries, where slight projection error associates background.
        interior = mask.copy()
        if not mask_is_interior:
            interior[[0,-1],:] = False
            interior[:,[0,-1]] = False
            interior[1:-1,1:-1] &= (mask[:-2,1:-1] & mask[2:,1:-1]
                                          & mask[1:-1,:-2] & mask[1:-1,2:])
        if not self.usable:
            return np.empty((0, 3), dtype=float)
        valid = interior[self.rows,self.columns] & self.consistent
        valid &= np.abs(self.reference_depth-object_depth) <= depth_gate
        if valid.sum() < self.min_points:
            return np.empty((0, 3), dtype=float)
        return self.points_camera[valid]


def fuse_camera_lidar_points(camera_points, lidar_points):
    """Concatenate measurements in one frame into an independent point cloud.

    As in the reference preprocessing pipeline, neither cloud is shifted to
    fit the other. Projection/occlusion filtering happens before fusion.
    """
    camera = np.asarray(camera_points, dtype=float).reshape(-1, 3)
    lidar = np.asarray(lidar_points, dtype=float).reshape(-1, 3)
    if not np.isfinite(camera).all() or not np.isfinite(lidar).all():
        raise ValueError("Fusion requires finite point clouds")
    return np.vstack((camera, lidar))


def project_lidar(points_world, camera_pose_world, intrinsics, depth_image,
                  time_skew, max_skew=.05, min_points=6, absolute_gate=.3, relative_gate=.02):
    if not np.isfinite(time_skew) or abs(time_skew)>max_skew:
        return None
    points = np.asarray(points_world,dtype=float)
    pose = np.asarray(camera_pose_world,dtype=float)
    k = np.asarray(intrinsics,dtype=float)
    image = np.asarray(depth_image,dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or pose.shape != (4,4)
            or k.shape != (3,3) or image.ndim != 2 or image.size == 0
            or not np.isfinite(pose).all() or not np.isfinite(k).all()
            or k[0,0]<=0 or k[1,1]<=0):
        raise ValueError('Invalid LiDAR/camera alignment inputs')
    # World points already contain the LiDAR-time pose. The inverse of the
    # image-time camera pose compensates vehicle translation and rotation.
    optical = (points-pose[:3,3]) @ pose[:3,:3]
    optical = optical[np.isfinite(optical).all(axis=1) & (optical[:,2]>.01)]
    uv = optical @ k.T
    uv = uv[:,:2]/uv[:,2,None]
    h,w = image.shape
    valid = (uv[:,0]>=0) & (uv[:,0]<w-1) & (uv[:,1]>=0) & (uv[:,1]<h-1)
    optical, uv = optical[valid], uv[valid]
    cols,rows = np.rint(uv).astype(int).T
    # Z-buffer: retain the nearest return if several points project to one pixel.
    order = np.argsort(optical[:,2],kind='stable')
    _, first = np.unique(rows[order]*w+cols[order],return_index=True)
    selected = order[first]
    rows,cols,optical = rows[selected],cols[selected],optical[selected]
    z = optical[:,2]
    reference = image[rows,cols]
    available = np.isfinite(reference) & (reference>0)
    consistent = available & (np.abs(z-reference) <= absolute_gate+relative_gate*reference)
    enough = available.sum() >= min_points
    usable = bool(enough and consistent.sum()/max(1,available.sum()) >= .6)
    return AlignedLidar(optical,rows,cols,reference,consistent,float(time_skew),usable,min_points)
