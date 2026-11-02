"""Temporary decision exclusions; never alter occupancy or flight safety checks."""
import numpy as np


def filter_failed_routes(choices, paths, waypoints, current, now, spacing, cooldown):
    recent = [w for w in waypoints if w.get('status') == 'blocked'
              and w.get('source') != 'object_inspection'
              and 0 <= now-w.get('failed_at', -float('inf')) < cooldown
              and (w.get('failure') or {}).get('local_status') != 'stale_map']
    kept = []
    for key, goal in choices:
        reject = False
        for waypoint in recent:
            if np.linalg.norm(np.asarray(goal)[:2]-np.asarray(waypoint['position'])[:2]) < spacing:
                reject = True
                break
            failure = waypoint.get('failure') or {}
            target = (failure.get('controller') or {}).get('target_ned')
            start = failure.get('current_ned')
            if failure.get('local_status') != 'no_path' or target is None or start is None:
                continue
            if np.linalg.norm(np.asarray(current)[:2]-np.asarray(start)[:2]) > spacing:
                continue
            # Same directed departure through the failed local target, even if
            # the final frontier/cluster changed. Other departures remain open.
            path = paths.get(key, [])
            direction = np.asarray(target)[:2]-np.asarray(start)[:2]
            if np.linalg.norm(direction) < .1:
                continue
            for a, b in zip(path, path[1:]):
                a, b = np.asarray(a), np.asarray(b)
                delta = b[:2]-a[:2]
                squared = float(delta @ delta)
                if squared < 1e-9 or delta @ direction <= 0:
                    continue
                t = float((np.asarray(target)[:2]-a[:2]) @ delta / squared)
                if (0 <= t <= 1 and
                        np.linalg.norm(a[:2]+t*delta-np.asarray(target)[:2]) <= .3 and
                        abs(a[2]+t*(b[2]-a[2])-target[2]) <= .15):
                    reject = True
                    break
            if reject:
                break
        if not reject:
            kept.append((key, goal))
    return kept, len(choices)-len(kept)
