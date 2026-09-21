"""Rasterized current vehicle volume, not an unknown-space travel allowance."""

import math
import numpy as np


def vehicle_footprint_indices(position, origin, resolution, radius, half_height):
    if radius <= 0:
        return np.empty((0, 3), dtype=np.int64)
    relative = (np.asarray(position) - np.asarray(origin)) / resolution
    center = np.floor(relative).astype(int)
    extent = math.ceil(radius / resolution)
    low = math.floor(relative[2] - half_height / resolution)
    high = math.floor(relative[2] + half_height / resolution)
    # Same discrete horizontal footprint as NavigationGrid's clearance test.
    return np.asarray(
        [
            (center[0] + i, center[1] + j, z)
            for i in range(-extent, extent + 1)
            for j in range(-extent, extent + 1)
            if math.hypot(i, j) * resolution <= radius + resolution * 0.5
            for z in range(low, high + 1)
        ],
        dtype=np.int64,
    ).reshape(-1, 3)
