"""Bounded multiview acquisition; navigation remains the existing guarded A*."""
import math

import numpy as np


VEHICLES = {'vehicle', 'car', 'truck', 'bus', 'van'}
LANDMARKS = {'fire extinguisher', 'fire cabinet', 'exit door', 'exit sign', 'signboard'}


class ObjectInspectionMemory:
    def __init__(self):
        self.sessions = {}
        self.pending = set()

    def resolve(self, records, identity):
        return next((r for r in records.values()
                     if r['runtime_instance_id'] == identity
                     or identity in r.get('merged_from', [])), None)

    def notice(self, records, timestamp, current):
        """Remember nearby sightings across a flight leg, without copying geometry."""
        for record in records.values():
            if (record['class_name'] in VEHICLES | LANDMARKS
                    and 0 <= timestamp - max([record['last_seen'], *[t['last_seen']
                        for t in record.get('connection_targets', [])]]) <= 1.0
                    and np.linalg.norm(np.asarray(record['position_world'])[:2]-current[:2]) <= 8.0):
                self.pending.add(record['runtime_instance_id'])

    def eligible(self, records, timestamp, current):
        self.notice(records, timestamp, current)
        candidates = []
        for record in records.values():
            ids = {record['runtime_instance_id'], *record.get('merged_from', [])}
            previous = [self.sessions[i] for i in ids.intersection(self.sessions)]
            if previous and any(s.get('status') in {
                    'no_fresh_observation', 'navigation_failed', 'time_budget_reached',
                    'observation_timeout'} for s in previous):
                self.pending.difference_update(ids)
                continue
            if previous and (not record.get('connection_targets')
                             or any(s.get('connection_review', False) for s in previous)):
                self.pending.difference_update(ids)
                continue
            if not ids.intersection(self.pending):
                continue
            distance = float(np.linalg.norm(np.asarray(record['position_world'])[:2] - current[:2]))
            if distance <= 8.0:
                candidates.append((distance, record))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    @staticmethod
    def focus(record):
        bounds = record['bbox_3d']
        centers = [(np.asarray(bounds['min'])+bounds['max'])/2]
        targets = record.get('connection_targets', [])
        if targets:
            fragment = max(targets, key=lambda t: t['last_seen'])
            centers.append((np.asarray(fragment['min'])+fragment['max'])/2)
        return np.mean(centers, axis=0)

    @staticmethod
    def _visible(grid, point, bounds):
        low, high = np.asarray(bounds['min']), np.asarray(bounds['max'])
        center = (low+high)/2
        delta = center[:2]-point[:2]
        distance = np.linalg.norm(delta)
        if distance <= 1e-9:
            return False
        for t in np.arange(0, distance, grid.resolution/2):
            p = point.copy()
            p[:2] += delta*t/distance
            if np.all(p[:2] >= low[:2]) and np.all(p[:2] <= high[:2]):
                return True
            if grid.cell(p) not in grid.known or grid.cell(p) in grid.occupied:
                return False
        return True

    @staticmethod
    def candidate(grid, record, current, attempted, origin, max_radius):
        low = np.asarray(record['bbox_3d']['min'], float)
        high = np.asarray(record['bbox_3d']['max'], float)
        targets = [record['bbox_3d']]
        if record.get('connection_targets'):
            targets.append(max(record['connection_targets'], key=lambda t: t['last_seen']))
            low = np.minimum(low, targets[-1]['min'])
            high = np.maximum(high, targets[-1]['max'])
        center = ObjectInspectionMemory.focus(record)
        if not np.all(np.isfinite([low, high])) or np.any(high < low):
            return None
        # Stay outside the observed footprint; actual clearance is certified by
        # the grid, including obstacles not belonging to the vehicle.
        radius = np.linalg.norm(high[:2] - low[:2]) / 2 + 1.5
        start_angle = math.atan2(current[1]-center[1], current[0]-center[0])
        candidates = []
        is_vehicle = record['class_name'] in VEHICLES
        # Wall-mounted objects: remain on the initially visible side. Their
        # back face is neither informative nor generally reachable.
        if not is_vehicle and attempted:
            start_angle = attempted[0]
        offsets = (np.linspace(0, 2*math.pi, 8, endpoint=False) if is_vehicle
                   else np.deg2rad([0, 30, -30, 60, -60]))
        separation = math.pi/3 if is_vehicle else math.pi/9
        for offset in offsets:
            angle = start_angle + offset
            if any(abs((angle-a+math.pi) % (2*math.pi)-math.pi) < separation
                   for a in attempted):
                continue
            point = np.array([center[0]+radius*math.cos(angle),
                              center[1]+radius*math.sin(angle), grid.altitude])
            if np.linalg.norm(point[:2]-origin[:2]) > max_radius:
                continue
            # A connecting view must see BOTH measured regions from one pose.
            if not all(ObjectInspectionMemory._visible(grid, point, bounds) for bounds in targets):
                continue
            if len(targets) > 1:
                bearings = [math.atan2((np.asarray(t['min'])+t['max'])[1]/2-point[1],
                                      (np.asarray(t['min'])+t['max'])[0]/2-point[0]) for t in targets]
                if abs((bearings[0]-bearings[1]+math.pi) % (2*math.pi)-math.pi) > math.radians(80):
                    continue
            path = grid.path(current, point)
            if path is None:
                continue
            cost = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            candidates.append((cost, point, angle))
        if not candidates:
            return None
        _, point, angle = min(candidates, key=lambda item: item[0])
        return point, angle
