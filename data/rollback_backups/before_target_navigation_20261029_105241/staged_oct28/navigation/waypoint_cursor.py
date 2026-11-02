"""Keep position setpoints fixed across rolling control horizons."""
import math
import numpy as np


class WaypointCursor:
    def __init__(self):
        self.endpoint = None
        self.goal = None
        self.yaw = None

    @property
    def intermediate(self):
        return self.goal is not None and not np.array_equal(self.goal, self.endpoint)

    def select(self, current, endpoint, length, tolerance, pass_radius=None, z_tolerance=None):
        current, endpoint = np.asarray(current), np.asarray(endpoint)
        new_segment = self.endpoint is None or not np.array_equal(endpoint, self.endpoint)
        if new_segment:
            self.endpoint = endpoint.copy()
            direction = endpoint-current
            self.yaw = math.atan2(float(direction[1]), float(direction[0]))
        radius = pass_radius if self.intermediate and pass_radius is not None else tolerance
        reached = False
        if not new_segment:
            delta = current-self.goal
            reached = (np.linalg.norm(delta[:2]) <= radius
                       and abs(delta[2]) <= (tolerance if z_tolerance is None else z_tolerance))
        if new_segment or reached:
            direction = endpoint-current
            distance = float(np.linalg.norm(direction))
            # Preserve the exact endpoint: arithmetic reconstruction can differ
            # by an ulp and incorrectly mark a real A* corner as pass-through.
            self.goal = (endpoint.copy() if distance <= length else
                         current + direction*(length/distance))
        return self.goal.copy(), self.yaw
