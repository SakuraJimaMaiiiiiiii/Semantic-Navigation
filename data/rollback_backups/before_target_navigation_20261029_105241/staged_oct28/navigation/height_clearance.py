"""Conservative full-body clearance for bounded vertical exploration moves."""
import math
import numpy as np
from planner.vertical_projection import height_band_indices


def vertical_corridor_free(snapshot, start, end, radius, half_height):
    start, end = np.asarray(start,dtype=float), np.asarray(end,dtype=float)
    if not np.isfinite([start,end]).all() or np.linalg.norm(start[:2]-end[:2]) > 1e-6:
        return False
    r=float(snapshot.resolution)
    origin=np.asarray(snapshot.origin_ned)
    center=np.floor((start-origin)/r).astype(int)
    extent=math.ceil(radius/r)
    center_down = .5*(start[2]+end[2])
    swept_half_height = half_height + .5*abs(start[2]-end[2])
    low, high = height_band_indices(center_down, swept_half_height, r, origin[2])
    if low<0 or high>=snapshot.states.shape[2]:
        return False
    for dx in range(-extent,extent+1):
        for dy in range(-extent,extent+1):
            if math.hypot(dx,dy)*r>radius+.5*r:
                continue
            x,y=center[0]+dx,center[1]+dy
            if x<0 or y<0 or x>=snapshot.states.shape[0] or y>=snapshot.states.shape[1]:
                return False
            if not np.all(snapshot.states[x,y,low:high+1]==0):
                return False
    return True


def ascent_levels(current_down, nominal_down, step, max_gain):
    """Ordered bounded NED heights, including a final partial step."""
    if not all(math.isfinite(v) for v in (current_down, nominal_down, step, max_gain)) or step < .1 or max_gain < 0:
        raise ValueError('Invalid ascent limits')
    limit = nominal_down-max_gain
    levels = []
    proposed = current_down
    while proposed-limit >= .1-1e-9:
        proposed = max(limit, proposed-step)
        levels.append(proposed)
    return levels
