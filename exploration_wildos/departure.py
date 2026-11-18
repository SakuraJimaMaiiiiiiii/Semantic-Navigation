"""Certified graph departure and transient failure attribution (no flight I/O)."""
import math
import numpy as np


def trim_reached_prefix(path, current, grid, tolerance, z_tolerance):
    """Skip only reached intermediate points with a certified next chord."""
    result = [np.asarray(current).copy(), *[np.asarray(p).copy() for p in path[1:]]]
    skipped = []
    while (len(result) > 2
           and np.linalg.norm(result[1][:2]-current[:2]) <= tolerance
           and abs(result[1][2]-current[2]) <= z_tolerance
           and grid.segment_free(current, result[2])):
        skipped.append(result[1].tolist())
        result.pop(1)
    return result, skipped


def heading_difference(a, b):
    return abs(math.atan2(math.sin(a-b), math.cos(a-b)))


def transient_failure(failure, start, current, tolerance):
    reason = failure.get('reason')
    if reason == 'graph_route_changed_before_execution':
        return 'route_revalidation'
    if (np.linalg.norm(np.asarray(current)-start) <= tolerance
            and ((reason == 'exploration_no_motion_progress'
                  and failure.get('local_status') == 'emergency_stop')
                 or reason in {'graph_departure_without_clearance', 'graph_departure_turn_timeout'})):
        return 'departure_blocked'
    return None


def local_geometry_signature(grid, position):
    # Ignore timestamps/remote map churn: only changes near this hold matter.
    center = grid.cell(position)
    extent = int(math.ceil(2. / grid.resolution))
    return frozenset((x, y) for x in range(center[0]-extent, center[0]+extent+1)
                     for y in range(center[1]-extent, center[1]+extent+1)
                     if (x, y) in grid.free)
