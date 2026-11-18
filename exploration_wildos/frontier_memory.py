"""Spatial exploration history, independent of graph node IDs and camera yaw."""
from collections import OrderedDict, deque
import math
import numpy as np


class FrontierMemory:
    repeat_weight = 2.0

    def __init__(self, spacing):
        self.spacing = float(spacing)
        self.cell_size = .5
        self.traversed = OrderedDict()
        self.observations = deque(maxlen=1024)
        self.last_position = self.heading_anchor = self.heading = None

    def cell(self, position):
        return tuple(np.floor(np.asarray(position)[:2]/self.cell_size).astype(int))

    def record_position(self, position):
        point = np.asarray(position, dtype=float)[:2].copy()
        if point.shape != (2,) or not np.isfinite(point).all():
            return
        previous = self.last_position
        self.last_position = point
        # Sparse telemetry must not paint an unobserved chord as flown space.
        distance = 0. if previous is None else float(np.linalg.norm(point-previous))
        if previous is None or distance > 2.:
            samples = [point]
            self.heading_anchor = point
            self.heading = None
        else:
            samples = np.linspace(previous, point, max(2, int(math.ceil(distance/.25))+1))
        for sample in samples:
            cell = self.cell(sample)
            self.traversed[cell] = None
            self.traversed.move_to_end(cell)
        while len(self.traversed) > 50000:
            self.traversed.popitem(last=False)
        delta = point-self.heading_anchor
        if np.linalg.norm(delta) >= .5:
            self.heading = delta/np.linalg.norm(delta)
            self.heading_anchor = point

    def reused_distance(self, points):
        total = 0.
        for a, b in zip(points, points[1:]):
            a, b = np.asarray(a)[:2], np.asarray(b)[:2]
            distance = float(np.linalg.norm(b-a))
            steps = max(1, int(math.ceil(distance/self.cell_size)))
            for step in range(steps):
                middle = a+(b-a)*((step+.5)/steps)
                if self.cell(middle) in self.traversed:
                    total += distance/steps
        return total

    def turn_cost(self, path, yaw):
        forward = self.heading
        if forward is None:
            forward = np.array([math.cos(yaw), math.sin(yaw)])
        for point in path[1:]:
            delta = np.asarray(point)[:2]-np.asarray(path[0])[:2]
            distance = np.linalg.norm(delta)
            if distance >= .75:
                return self.spacing*(1.-float(np.clip(delta @ forward/distance, -1., 1.)))
        return 0.

    def novel_cells(self, point, yaw, cells):
        covered = set()
        for previous, heading, observed in self.observations:
            difference = math.atan2(math.sin(yaw-heading), math.cos(yaw-heading))
            if np.linalg.norm(np.asarray(point)[:2]-previous) <= self.spacing and abs(difference) <= math.pi/4:
                covered.update(observed)
        return frozenset(cells-covered)

    def record_visit(self, view):
        self.observations.append((np.asarray(view.point)[:2].copy(), float(view.yaw), view.visible_unknown))
