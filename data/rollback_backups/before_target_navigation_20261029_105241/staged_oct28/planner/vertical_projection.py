"""NED 体素地图投影到水平规划面时使用的高度筛选。"""

from __future__ import annotations

import math
import numpy as np


def ned_height_slice_mask(
    voxel_indices,
    resolution: float,
    center_down: float,
    above_height: float,
    below_height: float,
) -> np.ndarray:
    """返回位于飞行平面上下指定高度内的体素掩码。

    NED 的 Down 轴向下为正，因此机体上方的体素 Down 值更小，
    机体下方的体素 Down 值更大。
    """
    indices = np.asarray(voxel_indices)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("voxel_indices must have shape (N, 3).")

    voxel_centers_down = (
        indices[:, 2].astype(np.float64) + 0.5
    ) * float(resolution)
    minimum_down = float(center_down) - float(above_height)
    maximum_down = float(center_down) + float(below_height)
    tolerance = 1e-9
    return (
        (voxel_centers_down >= minimum_down - tolerance)
        & (voxel_centers_down <= maximum_down + tolerance)
    )


def height_band_indices(center_down, half_height, resolution, origin_down=0.0):
    """Voxel layers with positive-volume overlap with the body's height band.

    A voxel merely touching the top/bottom boundary contributes no volume.
    Tolerance is in voxel units and only absorbs floating-point roundoff.
    """
    low = (float(center_down)-float(half_height)-float(origin_down))/resolution
    high = (float(center_down)+float(half_height)-float(origin_down))/resolution
    return math.floor(low+1e-8), math.ceil(high-1e-8)-1
