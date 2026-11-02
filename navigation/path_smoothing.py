"""Collision-certified corner rounding; output is executed as checked chords."""
import math
import numpy as np


def rounded_path(path, segment_free, radius=1.0):
    """Return (point, pass_through, turn) without changing either endpoint.

    Quadratic corner arcs use angle-adaptive sampling and
    <= 0.2 m nominal spacing. Every emitted chord, not just its endpoints, is
    checked against the caller's clearance-inflated grid. A rejected corner
    stays a stopping waypoint; no smoothing through unknown space.
    """
    points = [np.asarray(p, dtype=float) for p in path]
    if len(points) < 2:
        return []
    result = []
    previous = points[0]
    for a, b, c in zip(points, points[1:], points[2:]):
        incoming, outgoing = b-a, c-b
        before, after = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        if min(before, after) < 1e-8:
            continue
        u, v = incoming/before, outgoing/after
        angle = math.acos(float(np.clip(u@v, -1, 1)))
        accepted = False
        # Nearly reversing direction has no useful bounded-radius rounding.
        if math.radians(5) < angle < math.radians(150):
            trim = min(radius, .45*before, .45*after)
            entry, exit = b-u*trim, b+v*trim
            count = max(4, int(math.ceil(2*trim/.2)),
                        int(math.ceil(angle/math.radians(5))))
            arc = [(entry + 2*(1-t)*t*(b-entry) + t*t*(exit-entry))
                   for t in np.linspace(0, 1, count+1)]
            chain = [previous, *arc, c]
            if all(segment_free(x,y) for x,y in zip(chain, chain[1:])):
                result.extend((p, True, True) for p in arc)
                previous = exit
                accepted = True
        if not accepted:
            result.append((b.copy(), angle < math.radians(5), False))
            previous = b
    result.append((points[-1].copy(), False, False))
    return tracking_points(points[0], result, segment_free)


def tracking_points(start, route, segment_free, spacing=.6):
    """Keep dense validation samples out of the position controller.

    Merge only flowing points, re-certifying every longer chord. Real stopping
    corners and the destination remain unchanged.
    """
    output = []
    anchor = np.asarray(start)
    index = 0
    while index < len(route):
        end = index
        distance = float(np.linalg.norm(route[index][0]-anchor))
        while end+1 < len(route) and route[end][1] and route[end+1][1]:
            distance += float(np.linalg.norm(route[end+1][0]-route[end][0]))
            if distance > spacing or not segment_free(anchor, route[end+1][0]):
                break
            end += 1
        point, flow, _ = route[end]
        turning = any(item[2] for item in route[index:end+1])
        # Avoid leaving a tiny final arc fragment that would shrink the lead.
        if (flow and output and output[-1][1]
                and np.linalg.norm(point-output[-1][0]) < spacing/2
                and segment_free(output[-2][0] if len(output)>1 else start, point)):
            turning = turning or output[-1][2]
            output[-1] = (point, flow, turning)
        else:
            output.append((point, flow, turning))
        anchor = point
        index = end+1
    return output


def advance_route(path, modes, current, velocity, segment_free, z_tolerance):
    """Advance certified flowing points without imposing a zero-speed gate.

    Real corners still require a slow arrival. A skipped point's replacement
    tracking chord is always checked, including when tolerance cuts an arc.
    """
    path, modes = list(path), list(modes)
    if len(modes) != len(path)-1:
        raise ValueError('Route point/mode count mismatch')
    current = np.asarray(current)
    while len(path) > 2:
        flowing, _ = modes[0]
        delta = current - path[1]
        if (np.linalg.norm(delta[:2]) > (.25 if flowing else .1)
                or abs(delta[2]) > z_tolerance
                or (not flowing and np.linalg.norm(velocity) > .05)
                or not segment_free(current, path[2])):
            break
        path.pop(1)
        modes.pop(0)
    path[0] = current.copy()
    return path, modes
