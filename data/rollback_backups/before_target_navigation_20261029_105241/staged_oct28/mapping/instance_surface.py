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
        self._model = None

    def update(self, points):
        if points is None:
            return
        self._dirty = True
        self._tree = None
        self._model = None
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

    def vehicle_geometry(self):
        """Observed oriented extent, never extrapolated to unseen car surfaces."""
        if self._model is not None:
            return self._model
        points = self.supported_points()
        if len(points) < 24:
            return {"position_status": "partial_surface"}
        # Fit measured extent, not a PCA axis weighted by which face is visible.
        # Each voxel contributes once regardless of frame count or pixel density.
        origin = np.mean(points[:, :2], axis=0)
        angles = np.deg2rad(np.arange(90.))
        axes = np.column_stack((np.cos(angles), np.sin(angles)))
        transverse = np.column_stack((-axes[:, 1], axes[:, 0]))
        xy = points[:, :2]-origin
        along, across = xy @ axes.T, xy @ transverse.T
        lo_x, hi_x = np.percentile(along, [2.5, 97.5], axis=0)
        lo_y, hi_y = np.percentile(across, [2.5, 97.5], axis=0)
        # Minimum area alone rotates an L-shaped side/end observation onto
        # its diagonal. Prefer edges actually supported by the measured faces.
        residual = np.minimum.reduce((np.abs(along-lo_x), np.abs(along-hi_x),
                                      np.abs(across-lo_y), np.abs(across-hi_y)))
        cutoff = np.percentile(residual, 90, axis=0)
        loss = np.minimum(residual, cutoff).mean(axis=0)
        best = int(np.lexsort(((hi_x-lo_x)*(hi_y-lo_y), loss))[0])
        axis = axes[best]
        if hi_y[best]-lo_y[best] > hi_x[best]-lo_x[best]:
            axis = transverse[best]
        basis = np.column_stack((axis, [-axis[1], axis[0]]))
        local = xy @ basis
        low, high = np.percentile(local, [2.5, 97.5], axis=0)
        zlow, zhigh = np.percentile(points[:, 2], [2.5, 97.5])
        center = np.r_[origin + basis @ ((low+high)/2), (zlow+zhigh)/2]
        size = np.r_[high-low, zhigh-zlow]
        # A single broad face must not become a complete-car center estimate.
        # Count measured vertical patches on the four sides of the fitted box.
        faces = []
        for dimension, name in ((0, "length"), (1, "width")):
            for side, bound in (("min", low[dimension]), ("max", high[dimension])):
                patch = points[np.abs(local[:, dimension]-bound) <= 1.5*self.resolution]
                support = local[np.abs(local[:, dimension]-bound) <= 1.5*self.resolution]
                if (len(patch) >= 12 and np.ptp(patch[:, 2]) >= .6
                        and np.ptp(support[:, 1-dimension]) >= .5*size[1-dimension]):
                    faces.append(f"{name}_{side}")
        supported = bool(size[0] >= 2.5 and size[1] >= 1.0 and size[2] >= .6
                         and size[0] >= 1.5*size[1] and len(faces) >= 2)
        self._model = {"position_status": "multi_surface_estimate" if supported else "partial_surface",
                       "supported_box_faces": faces,
                       "geometry_fit": "measured_face_rectangle",
                       "observed_oriented_box": {"center": center.tolist(), "size": size.tolist(),
                                                 "yaw": float(np.arctan2(axis[1], axis[0]))}}
        return self._model

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

    def observed_patch(self, points):
        """A new view must actually cover part of this stable measured surface."""
        if points is None:
            return False
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        stable = self.supported_points()
        if min(len(points), len(stable)) < 12:
            return False
        matched = stable[cKDTree(points).query(stable)[0] <= 1.5*self.resolution]
        return (len(matched) >= max(12, .25*len(stable))
                and np.count_nonzero(np.ptp(matched, axis=0) >= 2*self.resolution) >= 2)

    def merge(self, other):
        self._dirty = True
        self._tree = None
        self._model = None
        for key, value in other.cells.items():
            if key not in self.cells or self.cells[key][1] < value[1]:
                self.cells[key] = value
        while len(self.cells) > self.capacity:
            self.cells.popitem(last=False)

    def shares_patch(self, other, *, minimum_fraction=.25):
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
        if matched < 24 or matched/min(len(first), len(second)) < minimum_fraction:
            return False
        return all(np.count_nonzero(np.ptp(points, axis=0) >= 3*self.resolution) >= 2
                   for points in (first[a], second[b]))
