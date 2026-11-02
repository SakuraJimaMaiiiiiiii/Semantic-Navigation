"""Bounded union of repeatedly measured surfaces for one associated instance."""
from collections import OrderedDict

import numpy as np
from scipy.spatial import cKDTree


class InstanceSurface:
    # Geometry memory only: these cells never mark navigation space occupied.
    resolution = .10
    capacity = 4096

    def __init__(self):
        self.cells = OrderedDict()
        self._bounds = None
        self._dirty = True
        self._tree = None

    def update(self, points):
        if points is None:
            return
        self._dirty = True
        self._tree = None
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        keys, inverse = np.unique(np.floor(points/self.resolution).astype(np.int64),
                                  axis=0, return_inverse=True)
        counts = np.bincount(inverse, minlength=len(keys))
        sums = np.zeros((len(keys), 3))
        np.add.at(sums, inverse, points)
        for key, point in zip(keys, sums/np.maximum(counts[:, None], 1)):
            key = tuple(key)
            previous, count = self.cells.get(key, (point, 0))
            self.cells[key] = (previous+(point-previous)/(count+1), count+1)
            self.cells.move_to_end(key)
        while len(self.cells) > self.capacity:
            self.cells.popitem(last=False)

    def bounds(self):
        # Density within a frame is not independent evidence. Require repeated
        # observations before a surface may expand the persistent object extent.
        if self._dirty:
            points = self.supported_points()
            self._bounds = None if len(points) < 12 else np.percentile(points, [2.5, 97.5], axis=0)
            self._dirty = False
        return self._bounds

    def supported_points(self):
        """One sample per voxel, supported by at least two different frames."""
        return np.asarray([p for p, count in self.cells.values() if count >= 2],
                          dtype=float).reshape(-1, 3)

    def supports(self, points):
        """Check measured overlap in the map frame, without fitting/ICP.

        A shared bounding box alone is not surface evidence. Require 12 distinct
        observed voxels and 70% agreement within 1.5 geometry voxels. This is an
        association tolerance, not an obstacle clearance or navigation setting.
        """
        if points is None:
            return False
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        _, indices = np.unique(np.floor(points/self.resolution).astype(np.int64),
                               axis=0, return_index=True)
        points = points[indices]
        if len(points) < 12:
            return False
        if self._tree is None:
            stable = self.supported_points()
            if len(stable) < 12:
                return False
            self._tree = cKDTree(stable)
        tolerance = 1.5*self.resolution
        if (np.any(points.max(axis=0) < self._tree.mins-tolerance)
                or np.any(points.min(axis=0) > self._tree.maxes+tolerance)):
            return False
        distances, _ = self._tree.query(points, distance_upper_bound=tolerance)
        matched = np.count_nonzero(np.isfinite(distances))
        return matched >= 12 and matched/len(points) >= .7

    def merge(self, other):
        self._dirty = True
        self._tree = None
        for key, value in other.cells.items():
            if key not in self.cells or self.cells[key][1] < value[1]:
                self.cells[key] = value
        while len(self.cells) > self.capacity:
            self.cells.popitem(last=False)

    def shares_patch(self, other):
        """Repeated side/end surfaces can overlap locally, not by 70% globally.

        Require reciprocal support over a two-dimensional patch. A single edge,
        a handful of touching voxels, or a fitted/translated cloud is not enough.
        This is only geometry evidence; identity needs independent checks.
        """
        first, second = self.supported_points(), other.supported_points()
        if min(len(first), len(second)) < 24:
            return False
        tolerance = 1.5*self.resolution
        a = cKDTree(second).query(first)[0] <= tolerance
        b = cKDTree(first).query(second)[0] <= tolerance
        matched = min(int(a.sum()), int(b.sum()))
        if matched < 24 or matched/min(len(first), len(second)) < .25:
            return False
        return all(np.count_nonzero(np.ptp(points, axis=0) >= 3*self.resolution) >= 2
                   for points in (first[a], second[b]))
