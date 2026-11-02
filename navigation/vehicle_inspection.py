"""Bounded extra viewpoints for incomplete static vehicle surfaces."""
import math
import numpy as np


def needs_inspection(record, uncertainty_limit):
    if record['class_name'] not in ('car', 'truck', 'bus', 'van'):
        return False
    covariance = np.asarray(record['position_covariance'])[:2, :2]
    return (record['position_status'] != 'multi_surface_estimate'
            or np.max(np.linalg.eigvalsh(covariance)) > uncertainty_limit ** 2)


def observation_route(grid, current, record, altitude, standoff, used_yaws):
    low, high = np.asarray(record['bbox_3d']['min']), np.asarray(record['bbox_3d']['max'])
    center = (low + high) * .5
    half = (high - low) * .5
    initial = math.atan2(current[1]-center[1], current[0]-center[0])
    routes = []
    for offset in (math.pi/3, -math.pi/3, 2*math.pi/3, -2*math.pi/3):
        angle = initial + offset
        if any(abs(math.atan2(math.sin(angle-y), math.cos(angle-y))) < math.pi/4 for y in used_yaws):
            continue
        direction = np.array([math.cos(angle), math.sin(angle)])
        radius = min((half[i]+standoff)/abs(direction[i]) for i in range(2) if abs(direction[i]) > 1e-6)
        point = np.r_[center[:2] + radius*direction, altitude]
        if grid.cell(point) not in grid.free:
            continue
        path = grid.path(current, point)
        if path is not None:
            length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            routes.append((length, point, path, angle))
    return min(routes, key=lambda row: row[0]) if routes else None
