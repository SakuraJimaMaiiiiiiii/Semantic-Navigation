"""Known-free fixed-altitude routing, approach viewpoints and frontier search.

Global return trace fallback is deliberately not used for arbitrary object goals.
NED altitude stays fixed; sparse free voxels must cover the vehicle's vertical band.
"""

import heapq
import math
from collections import deque
import numpy as np
from planner.vertical_projection import height_band_indices


class NavigationGrid:
    def __init__(self, free, occupied, resolution, origin, altitude, config):
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=float)[:2]
        self.altitude = float(altitude)
        self.config = config
        self.occupied = set(occupied)
        raw_free = set(free) - self.occupied
        self.known = raw_free | self.occupied
        radius = int(math.ceil(config.clearance_radius / self.resolution))
        offsets = [
            (i, j)
            for i in range(-radius, radius + 1)
            for j in range(-radius, radius + 1)
            if math.hypot(i, j) * self.resolution
            <= config.clearance_radius + self.resolution * 0.5
        ]
        self.free = {
            cell
            for cell in raw_free
            if all((cell[0] + i, cell[1] + j) in raw_free for i, j in offsets)
        }

    @classmethod
    def from_global(cls, snapshot, altitude, config, local_snapshot=None, current=None):
        r = float(snapshot.resolution)
        zmin, zmax = height_band_indices(altitude, config.vertical_half_extent, r)
        mask = (snapshot.indices[:, 2] >= zmin) & (snapshot.indices[:, 2] <= zmax)
        indices = snapshot.indices[mask, :2]
        odds = snapshot.log_odds[mask]
        occupied = set(map(tuple, indices[odds >= snapshot.occupied_threshold].tolist()))
        free_cells, counts = np.unique(indices[odds <= snapshot.free_threshold], axis=0, return_counts=True)
        free = set(map(tuple, free_cells[counts == zmax - zmin + 1].tolist()))
        if local_snapshot is not None and current is not None:
            # Temporary evidence fusion near the vehicle only. A coarse world
            # cell is certified only when its ENTIRE volume is locally free;
            # center sampling would miss walls with different grid resolutions.
            local = local_snapshot
            if abs(local.timestamp - snapshot.timestamp) <= config.map_timeout:
                # Fresh local hits must veto stale global free space before
                # clearance erosion. Project occupied voxel volumes, not just
                # centers (the two maps can have different resolutions/origins).
                lr = float(local.resolution)
                lzmin, lzmax = height_band_indices(
                    altitude, config.vertical_half_extent, lr, local.origin_ned[2]
                )
                lzmin, lzmax = max(0, lzmin), min(local.states.shape[2]-1, lzmax)
                if lzmin <= lzmax:
                    hits = np.argwhere(np.any(local.states[:, :, lzmin:lzmax+1] == 1, axis=2))
                    for hit in hits:
                        low = local.origin_ned[:2] + hit*lr
                        first = np.floor(low/r + 1e-8).astype(int)
                        last = np.ceil((low+lr)/r - 1e-8).astype(int)-1
                        occupied.update((x,y) for x in range(first[0],last[0]+1)
                                        for y in range(first[1],last[1]+1))
                    free.difference_update(occupied)
                center = np.floor(np.asarray(current)[:2] / r).astype(int)
                extent = math.ceil((2.4 + config.clearance_radius) / r)
                zlo = math.floor((zmin*r-local.origin_ned[2])/lr + 1e-8)
                zhi = math.ceil(((zmax+1)*r-local.origin_ned[2])/lr - 1e-8)
                if 0 <= zlo < zhi <= local.states.shape[2]:
                    for x in range(center[0]-extent, center[0]+extent+1):
                        for y in range(center[1]-extent, center[1]+extent+1):
                            cell = (x, y)
                            if cell in occupied or cell in free:
                                continue
                            low = np.floor((np.array(cell)*r-local.origin_ned[:2])/lr + 1e-8).astype(int)
                            high = np.ceil(((np.array(cell)+1)*r-local.origin_ned[:2])/lr - 1e-8).astype(int)
                            if np.any(low < 0) or np.any(high > local.states.shape[:2]):
                                continue
                            if np.all(local.states[low[0]:high[0], low[1]:high[1], zlo:zhi] == 0):
                                free.add(cell)
        return cls(free, occupied, r, (0, 0), altitude, config)

    @classmethod
    def from_local(cls, snapshot, altitude, config, global_snapshot=None):
        r = float(snapshot.resolution)
        zmin, zmax = height_band_indices(
            altitude, config.vertical_half_extent, r, snapshot.origin_ned[2]
        )
        if zmin < 0 or zmax >= snapshot.states.shape[2]:
            return cls(set(), set(), r, snapshot.origin_ned, altitude, config)
        band = snapshot.states[:, :, zmin : zmax + 1].copy()
        observed = getattr(snapshot, "observed", None)
        if global_snapshot is not None and observed is not None:
            # The rolling camera map can forget previously observed free space.
            # Fill only never-observed local voxels, never local hits/ambiguity.
            missing = np.argwhere((band == -1) & ~observed[:, :, zmin : zmax + 1])
            gr = float(global_snapshot.resolution)
            lower = np.floor(
                (snapshot.origin_ned + np.array([0, 0, zmin]) * r) / gr
            ).astype(int)
            upper = np.floor(
                (snapshot.origin_ned + np.array([*band.shape[:2], zmax + 1]) * r) / gr
            ).astype(int)
            mask = np.all(
                (global_snapshot.indices >= lower) & (global_snapshot.indices <= upper),
                axis=1,
            )
            global_free = {
                tuple(index)
                for index, odds in zip(
                    global_snapshot.indices[mask], global_snapshot.log_odds[mask]
                )
                if odds <= global_snapshot.free_threshold
            }
            if len(missing) and global_free:
                # Check all coarse cells covering each missing local voxel in
                # batches; the previous Python loop repeated this per voxel.
                index3 = missing + np.array([0, 0, zmin])
                low = snapshot.origin_ned + index3*r
                first = np.floor(low/gr + 1e-8).astype(np.int64)
                last = np.floor((low+r)/gr - 1e-8).astype(np.int64)
                free_indices = np.asarray(sorted(global_free), dtype=np.int64)
                row_dtype = np.dtype((np.void, 3*free_indices.dtype.itemsize))
                free_keys = np.ascontiguousarray(free_indices).view(row_dtype).ravel()
                covered = np.ones(len(missing), dtype=bool)
                span = (last-first).max(axis=0)+1
                for offset in np.ndindex(tuple(span)):
                    cells = first + offset
                    active = np.all(cells <= last, axis=1)
                    keys = np.ascontiguousarray(cells).view(row_dtype).ravel()
                    covered &= ~active | np.isin(keys, free_keys)
                band[tuple(missing[covered].T)] = 0
        free = set(map(tuple, np.argwhere(np.all(band == 0, axis=2))))
        occupied = set(map(tuple, np.argwhere(np.any(band == 1, axis=2))))
        return cls(free, occupied, r, snapshot.origin_ned, altitude, config)

    def cell(self, point):
        return tuple(
            np.floor((np.asarray(point)[:2] - self.origin) / self.resolution).astype(
                int
            )
        )

    def point(self, cell):
        xy = self.origin + (np.asarray(cell) + 0.5) * self.resolution
        return np.array([*xy, self.altitude])

    def segment_free(self, start, end):
        start, end = np.asarray(start), np.asarray(end)
        if (
            abs(start[2] - self.altitude) > self.resolution
            or abs(end[2] - self.altitude) > self.resolution
        ):
            return False
        count = max(
            1, int(math.ceil(np.linalg.norm(end - start) / (self.resolution * 0.2)))
        )
        previous = None
        for i in range(count + 1):
            point = start + (end - start) * i / count
            cell = self.cell(point)
            if cell not in self.free:
                return False
            if (
                previous is not None
                and cell[0] != previous[0]
                and cell[1] != previous[1]
            ):
                if (cell[0], previous[1]) not in self.free or (
                    previous[0],
                    cell[1],
                ) not in self.free:
                    return False
            previous = cell
        return True

    def path(self, start, goal):
        first, last = self.cell(start), self.cell(goal)
        if first not in self.free or last not in self.free:
            return None
        queue, costs, parents = [(0.0, first)], {first: 0.0}, {}
        expanded = 0
        while queue and expanded < self.config.max_plan_nodes:
            _, cell = heapq.heappop(queue)
            expanded += 1
            if cell == last:
                cells = [cell]
                while cell != first:
                    cell = parents[cell]
                    cells.append(cell)
                points = [
                    np.asarray(start, dtype=float),
                    *[self.point(c) for c in reversed(cells)],
                    np.asarray(goal, dtype=float),
                ]
                # Only smooth segments certified in known free space.
                result, index = [points[0]], 0
                while index < len(points) - 1:
                    end = min(index + 30, len(points) - 1)
                    while end > index + 1 and not self.segment_free(
                        points[index], points[end]
                    ):
                        end -= 1
                    if not self.segment_free(points[index], points[end]):
                        return None
                    result.append(points[end])
                    index = end
                return np.asarray(result)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cell[0] + dx, cell[1] + dy)
                value = costs[cell] + 1
                if nxt in self.free and value < costs.get(nxt, math.inf):
                    costs[nxt] = value
                    parents[nxt] = cell
                    heapq.heappush(
                        queue,
                        (value + abs(nxt[0] - last[0]) + abs(nxt[1] - last[1]), nxt),
                    )
        return None

    def approach(self, record, current):
        low, high = np.array(record["bbox_3d"]["min"]), np.array(
            record["bbox_3d"]["max"]
        )
        center = (low + high) / 2
        candidates = []
        for angle in np.linspace(
            0, 2 * math.pi, self.config.approach_samples, endpoint=False
        ):
            direction = np.array([math.cos(angle), math.sin(angle)])
            # Sample side centres/corners and offset outward from the AABB surface.
            half = (high - low)[:2] / 2
            direction[np.abs(direction) < 1e-8] = 0.0
            surface_xy = center[:2] + np.sign(direction) * half
            xy = surface_xy + direction * self.config.standoff_distance
            goal = np.array([*xy, self.altitude])
            # Observe the near surface; the interior of the object is not free.
            surface = goal.copy()
            surface[:2] -= direction * max(
                0,
                self.config.standoff_distance
                - self.config.clearance_radius
                - self.resolution,
            )
            if not self.segment_free(goal, surface):
                continue
            path = self.path(current, goal)
            if path is not None:
                length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                candidates.append((length, goal, path))
        return min(candidates, key=lambda c: c[0]) if candidates else None

    def frontier(self, current, origin, visited):
        start = self.cell(current)
        if start not in self.free:
            return None
        queue, seen, choices = deque([start]), {start}, []
        probe = int(math.ceil(self.config.clearance_radius / self.resolution)) + 2
        while queue and len(seen) <= self.config.max_plan_nodes:
            cell = queue.popleft()
            point = self.point(cell)
            distance = float(np.linalg.norm(point[:2] - np.asarray(current)[:2]))
            unknown = sum(
                (cell[0] + dx * probe, cell[1] + dy * probe) not in self.known
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            if (
                unknown
                and distance >= self.config.leg_length
                and np.linalg.norm(point[:2] - np.asarray(origin)[:2])
                <= self.config.max_search_radius
                and all(
                    np.linalg.norm(point[:2] - p[:2]) > self.config.leg_length
                    for p in visited
                )
            ):
                choices.append((distance / unknown, point))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = cell[0] + dx, cell[1] + dy
                if nxt in self.free and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        for _, point in sorted(choices, key=lambda c: c[0])[:32]:
            path = self.path(current, point)
            if path is not None:
                return point, path
        return None




def observed_body_segment_free(snapshot, start, end, config):
    """Certify a height-recovery chord using actual observed 3D body volumes."""
    return observed_body_segment_rejection(snapshot, start, end, config) is None


def _body_volume_rejection(snapshot, cells, low, high):
    if (low < 0 or high >= snapshot.states.shape[2] or np.any(cells < 0)
            or np.any(cells >= np.asarray(snapshot.states.shape[:2]))):
        return 'outside_map'
    states = snapshot.states[cells[:,0],cells[:,1],low:high+1]
    if np.any(states == 1):
        return 'occupied_body_volume'
    if np.any(states != 0):
        return 'unknown_body_volume'
    return None


def _vertical_body_cells(snapshot, position, radius):
    """All voxel squares with positive-area overlap with the actual body disk.

    Using offsets from floor(position) both overchecks one side of a cell and
    misses diagonal intersections on the other side as the UAV moves in-cell.
    """
    r = float(snapshot.resolution)
    xy = np.asarray(position)[:2]
    first = np.floor((xy-radius-snapshot.origin_ned[:2])/r+1e-8).astype(int)
    last = np.ceil((xy+radius-snapshot.origin_ned[:2])/r-1e-8).astype(int)
    cells = np.array([(x,y) for x in range(first[0],last[0])
                      for y in range(first[1],last[1])],dtype=int).reshape(-1,2)
    low = snapshot.origin_ned[:2]+cells*r
    distance = np.maximum(np.maximum(low-xy,xy-(low+r)),0.)
    return cells[np.einsum('ij,ij->i',distance,distance) < radius*radius]


def observed_body_segment_rejection(snapshot, start, end, config):
    """Explain a failed observed-volume check without discarding its cause."""
    if snapshot is None:
        return 'missing_map'
    r = float(snapshot.resolution)
    start, end = np.asarray(start, float), np.asarray(end, float)
    if (start.shape != (3,) or end.shape != (3,)
            or not np.isfinite(start).all() or not np.isfinite(end).all()):
        return 'invalid_position'
    if np.array_equal(start[:2],end[:2]):
        cells = _vertical_body_cells(snapshot,start,config.clearance_radius)
        low,high = height_band_indices((start[2]+end[2])*.5,
            config.vertical_half_extent+abs(end[2]-start[2])*.5,r,snapshot.origin_ned[2])
        return _body_volume_rejection(snapshot,cells,low,high)
    count = max(2, int(math.ceil(np.linalg.norm(end-start)/(.5*r)))+1)
    extent = math.ceil(config.clearance_radius/r)
    offsets = np.array([(x, y) for x in range(-extent, extent+1)
                        for y in range(-extent, extent+1)
                        if math.hypot(x, y)*r <= config.clearance_radius+.5*r])
    for point in np.linspace(start, end, count):
        center = np.floor((point[:2]-snapshot.origin_ned[:2])/r).astype(int)
        cells = center+offsets
        low, high = height_band_indices(point[2], config.vertical_half_extent, r, snapshot.origin_ned[2])
        reason = _body_volume_rejection(snapshot,cells,low,high)
        if reason is not None:
            return reason
    return None


def execution_segment_free(snapshot, start, end, config, certified_grid):
    """Recheck only the commanded chord's body volume in the newest local map.

    Never replace a new hit or observed ambiguity with old free evidence.
    Never-observed cells may retain the preceding grid's certified history.
    """
    if not certified_grid.segment_free(start, end):
        return False
    r = float(snapshot.resolution)
    low, high = height_band_indices(certified_grid.altitude, config.vertical_half_extent,
                                    r, snapshot.origin_ned[2])
    if low < 0 or high >= snapshot.states.shape[2]:
        return False
    start, end = np.asarray(start), np.asarray(end)
    count = max(2, int(math.ceil(np.linalg.norm(end-start)/(.5*r)))+1)
    centers = np.floor((np.linspace(start, end, count)[:, :2]-snapshot.origin_ned[:2])/r).astype(int)
    extent = math.ceil(config.clearance_radius/r)
    offsets = np.array([(x,y) for x in range(-extent,extent+1) for y in range(-extent,extent+1)
                        if math.hypot(x,y)*r <= config.clearance_radius+.5*r])
    cells = (centers[:,None,:]+offsets).reshape(-1,2)
    if np.any(cells < 0) or np.any(cells >= np.asarray(snapshot.states.shape[:2])):
        return False
    # The planar certificate covers only its own height band. Tracking drift
    # and sloping connectors can sweep additional layers above/below it.
    points = np.linspace(start, end, count)
    observed = getattr(snapshot, 'observed', None)
    for point, center in zip(points, centers):
        body_low, body_high = height_band_indices(
            point[2], config.vertical_half_extent, r, snapshot.origin_ned[2])
        if body_low < 0 or body_high >= snapshot.states.shape[2]:
            return False
        body_cells = center+offsets
        states = snapshot.states[body_cells[:,0], body_cells[:,1], body_low:body_high+1]
        if np.any(states == 1):
            return False
        unknown = states != 0
        if observed is None:
            if np.any(unknown):
                return False
        else:
            seen = observed[body_cells[:,0], body_cells[:,1], body_low:body_high+1]
            layers = np.arange(body_low, body_high+1)
            certified = (layers >= low) & (layers <= high)
            if np.any(unknown & (seen | ~certified)):
                return False
    return True
