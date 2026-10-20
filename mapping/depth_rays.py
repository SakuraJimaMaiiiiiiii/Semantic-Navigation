"""Limit integration range without throwing away measured free-space rays."""

import numpy as np


def clip_depth_rays(points_camera, max_depth, depth_is_range):
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    distance = np.linalg.norm(points, axis=1) if depth_is_range else points[:, 2]
    hit = distance <= float(max_depth)
    scale = np.minimum(1., float(max_depth) / np.maximum(distance, 1e-12))
    return points * scale[:, None], hit
