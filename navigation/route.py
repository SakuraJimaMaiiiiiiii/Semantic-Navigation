"""Known-free fixed-altitude routing, approach viewpoints and frontier search.

Global return trace fallback is deliberately not used for arbitrary object goals.
NED altitude stays fixed; sparse free voxels must cover the vehicle's vertical band.
"""

import heapq
import math
from collections import deque
import numpy as np


class NavigationGrid:
    def __init__(self, free, occupied, resolution, origin, altitude, config):
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=float)[:2]
        self.altitude = float(altitude)
        self.config = config
        self.occupied = set(occupied)
        self.known = set(free) | set(occupied)
        radius = int(math.ceil(config.clearance_radius / self.resolution))
        offsets = [
            (i, j)
            for i in range(-radius, radius + 1)
            for j in range(-radius, radius + 1)
            if math.hypot(i, j) * self.resolution
            <= config.clearance_radius + self.resolution * 0.5
        ]
        raw_free = set(free) - set(occupied)
        self.free = {
            cell
            for cell in raw_free
            if all((cell[0] + i, cell[1] + j) in raw_free for i, j in offsets)
        }

    @classmethod
    def from_global(cls, snapshot, altitude, config):
        r = float(snapshot.resolution)
        zmin = math.floor((altitude - config.vertical_half_extent) / r)
        zmax = math.floor((altitude + config.vertical_half_extent) / r)
        mask = (snapshot.indices[:, 2] >= zmin) & (snapshot.indices[:, 2] <= zmax)
        counts, occupied = {}, set()
        for index, odds in zip(snapshot.indices[mask], snapshot.log_odds[mask]):
            cell = (int(index[0]), int(index[1]))
            if odds >= snapshot.occupied_threshold:
                occupied.add(cell)
            elif odds <= snapshot.free_threshold:
                counts[cell] = counts.get(cell, 0) + 1
        free = {cell for cell, count in counts.items() if count == zmax - zmin + 1}
        return cls(free, occupied, r, (0, 0), altitude, config)

    @classmethod
    def from_local(cls, snapshot, altitude, config, global_snapshot=None):
        r = float(snapshot.resolution)
        zmin = math.floor(
            (altitude - config.vertical_half_extent - snapshot.origin_ned[2]) / r
        )
        zmax = math.floor(
            (altitude + config.vertical_half_extent - snapshot.origin_ned[2]) / r
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
            for index in missing:
                index3 = index + np.array([0, 0, zmin])
                low = snapshot.origin_ned + index3 * r
                high = low + r
                first = np.floor(low / gr + 1e-8).astype(int)
                last = np.floor(high / gr - 1e-8).astype(int)
                if all(
                    (x, y, z) in global_free
                    for x in range(first[0], last[0] + 1)
                    for y in range(first[1], last[1] + 1)
                    for z in range(first[2], last[2] + 1)
                ):
                    band[tuple(index)] = 0
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


def next_leg(path, length):
    for start, end in zip(path[:-1], path[1:]):
        distance = float(np.linalg.norm(end - start))
        if distance >= length:
            return start + (end - start) * length / distance
        length -= distance
    return path[-1].copy()
