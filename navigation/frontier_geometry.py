"""Viewpoint data and occlusion-aware information gain for exploration."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class Viewpoint:
    key: tuple
    point: np.ndarray
    yaw: float
    cost: float
    visible_unknown: frozenset
    score: float


def visible_unknown(grid, cell, yaw, radius, fov_degrees):
    """Approximate camera rays: known obstacles occlude expected unknown gain."""
    visible = set()
    half = math.radians(fov_degrees) / 2
    for angle in np.linspace(yaw - half, yaw + half, 25):
        for distance in np.arange(0.5, radius / grid.resolution + 0.5, 0.5):
            point = (
                int(math.floor(cell[0] + 0.5 + math.cos(angle) * distance)),
                int(math.floor(cell[1] + 0.5 + math.sin(angle) * distance)),
            )
            if point in grid.occupied:
                break
            if point not in grid.known:
                visible.add(point)
    return frozenset(visible)
