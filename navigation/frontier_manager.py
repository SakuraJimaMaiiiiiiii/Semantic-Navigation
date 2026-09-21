"""Connected frontiers, occlusion-aware viewpoints and a bounded visit tour.

FUEL-inspired exploration layer; this is not a port of the FUEL planner.
Global grid search supplies connectivity/costs only. EGO executes all motion.
"""

from collections import deque
from dataclasses import dataclass
import math
import numpy as np

NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class Viewpoint:
    key: tuple
    point: np.ndarray
    yaw: float
    cost: float
    visible_unknown: frozenset
    score: float


def reachable_costs(grid, start, origin, radius):
    if start not in grid.free:
        return {}, False, False
    queue, costs = deque([start]), {start: 0.0}
    limited = False
    while queue and len(costs) < grid.config.max_plan_nodes:
        cell = queue.popleft()
        for dx, dy in NEIGHBORS:
            nxt = cell[0] + dx, cell[1] + dy
            if nxt in grid.free and nxt not in costs:
                if np.linalg.norm(grid.point(nxt)[:2] - origin[:2]) > radius:
                    limited = True
                    continue
                costs[nxt] = costs[cell] + grid.resolution
                queue.append(nxt)
    return costs, bool(queue), limited


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


def frontier_regions(grid, reachable, max_span):
    # Free cells are clearance-eroded, so look outward through the observed
    # footprint margin until the first unknown cell. An occupied ray is not a frontier.
    probe = math.ceil(grid.config.clearance_radius / grid.resolution) + 2
    frontier = set()
    for x, y in reachable:
        for dx, dy in NEIGHBORS:
            for step in range(1, probe + 1):
                other = x + dx * step, y + dy * step
                if other in grid.occupied:
                    break
                if other not in grid.known:
                    frontier.add((x, y))
                    break
            if (x, y) in frontier:
                break
    components = []
    while frontier:
        first = min(frontier)
        frontier.remove(first)
        component, queue = [first], deque([first])
        while queue:
            x, y = queue.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nxt = x + dx, y + dy
                    if nxt in frontier:
                        frontier.remove(nxt)
                        component.append(nxt)
                        queue.append(nxt)
        pending = [component]
        while pending:
            part = pending.pop()
            points = np.asarray(part)
            spans = np.ptp(points, axis=0) * grid.resolution
            if max(spans) > max_span and len(part) > 2:
                axis = int(np.argmax(spans))
                ordered = sorted(part, key=lambda p: (p[axis], p[1 - axis]))
                middle = len(part) // 2
                pending.extend((ordered[:middle], ordered[middle:]))
            else:
                components.append(frozenset(part))
    return components


class FrontierManager:
    def __init__(self, config):
        self.config = config
        self.observed_views = {}
        self.candidates = {}
        self.regions = []
        self.visit_order = []
        self._map_key = None
        self.identities = {}

    def identity(self, key):
        if key not in self.identities:
            self.identities[key] = f"frontier_{len(self.identities):04d}"
        return self.identities[key]

    def observe_view(self, candidate):
        if candidate is not None:
            self.observed_views[candidate.key] = candidate.visible_unknown

    def update(self, grid, current, origin, yaw, excluded):
        current, origin = np.asarray(current), np.asarray(origin)
        cfg = self.config
        costs, truncated, outside = reachable_costs(
            grid, grid.cell(current), origin, cfg.max_radius
        )
        # Recompute regions when occupancy evidence/connectivity changes. Failed
        # or reached viewpoints never delete unknown frontier cells from the map.
        map_key = (frozenset(costs), frozenset(grid.known), frozenset(grid.occupied))
        if map_key != self._map_key:
            self.regions = frontier_regions(grid, costs, 2 * cfg.viewpoint_spacing)
            self._map_key = map_key
        candidates = []
        for region in self.regions:
            center = np.mean(list(region), axis=0)
            ordered = sorted(
                region, key=lambda c: (np.linalg.norm(np.asarray(c) - center), c)
            )
            # Bounded representative sampling across each connected component.
            indices = np.linspace(0, len(ordered) - 1, min(6, len(ordered))).astype(int)
            best = None
            for i in indices:
                cell = ordered[i]
                point = grid.point(cell)
                if np.linalg.norm(point[:2] - current[:2]) < max(
                    0.8, cfg.progress_distance * 2
                ):
                    continue
                for direction in range(8):
                    angle = direction * math.pi / 4
                    key = (int(cell[0]), int(cell[1]), direction)
                    if key in excluded:
                        continue
                    visible = visible_unknown(
                        grid,
                        cell,
                        angle,
                        cfg.information_radius,
                        cfg.camera_fov_degrees,
                    )
                    if not visible:
                        continue
                    previous = self.observed_views.get(key)
                    if previous is not None and visible == previous:
                        continue
                    gain = len(visible) * grid.resolution**2
                    turn = abs((angle - yaw + math.pi) % (2 * math.pi) - math.pi)
                    score = (
                        cfg.information_weight * gain
                        - cfg.path_cost_weight * costs[cell]
                        - cfg.heading_cost_weight * turn
                    )
                    view = Viewpoint(key, point, angle, costs[cell], visible, score)
                    if best is None or view.score > best.score:
                        best = view
            if best is not None:
                candidates.append(best)
        shortlist = sorted(candidates, key=lambda c: (-c.score, c.key))[
            : cfg.tour_frontiers
        ]
        # Evaluate a short nearest-neighbour tour for each possible first region.
        # Pairwise costs use the connected free grid, not distance through walls.
        pair_cost = {}
        for first in shortlist:
            distances, _, _ = reachable_costs(
                grid, grid.cell(first.point), origin, cfg.max_radius
            )
            for other in shortlist:
                pair_cost[first.key, other.key] = distances.get(
                    grid.cell(other.point), math.inf
                )
        tours = []
        for first in shortlist:
            remaining = [v for v in shortlist if v is not first]
            order, length, previous = [first], first.cost, first
            while remaining:
                nxt = min(
                    remaining, key=lambda v: (pair_cost[previous.key, v.key], v.key)
                )
                cost = pair_cost[previous.key, nxt.key]
                if not math.isfinite(cost):
                    break
                length += cost
                order.append(nxt)
                remaining.remove(nxt)
                previous = nxt
            utility = first.score - cfg.tour_cost_weight * length
            tours.append((utility, order))
        chosen = max(tours, key=lambda t: t[0])[1] if tours else []
        self.candidates = {v.key: v for v in candidates}
        self.visit_order = [v.key for v in chosen]
        # Include candidates outside the short tour for later re-evaluation.
        keys = set(self.visit_order)
        chosen += [
            v for v in sorted(candidates, key=lambda c: -c.score) if v.key not in keys
        ]
        return [(v.key, v.point) for v in chosen], truncated, outside
