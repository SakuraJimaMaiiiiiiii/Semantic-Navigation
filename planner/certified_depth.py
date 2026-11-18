"""Resolve a broad front-depth stop only with measured corridor evidence."""
import numpy as np


def _hits_body_sweep(points, start, end, radius, half_height):
    low = np.minimum(start,end)-[radius,radius,half_height]
    high = np.maximum(start,end)+[radius,radius,half_height]
    inside = ((points[:,0] >= low[0]) & (points[:,0] <= high[0])
              & (points[:,1] >= low[1]) & (points[:,1] <= high[1])
              & (points[:,2] >= low[2]) & (points[:,2] <= high[2]))
    points = points[inside]
    if not len(points):
        return False
    delta = end-start
    squared = float(np.dot(delta[:2],delta[:2]))
    if squared > 1e-12:
        fraction = np.clip(((points[:,0]-start[0])*delta[0]
                            +(points[:,1]-start[1])*delta[1])/squared,0.,1.)
    elif abs(delta[2]) > 1e-12:
        fraction = np.clip((points[:,2]-start[2])/delta[2],0.,1.)
    else:
        fraction = np.zeros(len(points))
    dx = points[:,0]-start[0]-fraction*delta[0]
    dy = points[:,1]-start[1]-fraction*delta[1]
    dz = points[:,2]-start[2]-fraction*delta[2]
    return bool(np.any((dx*dx+dy*dy <= radius*radius) & (np.abs(dz) <= half_height)))


def refine_front_clearance(snapshot, current, target, stopping_point, clearance,
                           body_config, planner_config, map_config=None, *, cache=None):
    """Keep hard stops; allow slow certified travel past an off-corridor hit.

    All nearby depth pixels are checked, including hits too thin for the map.
    Missing evidence, unknown body volumes and unsafe braking retain the stop.
    """
    stop = float(planner_config.emergency_stop_distance)
    if not planner_config.map_emergency_stop_distance < clearance < stop:
        return clearance
    observation = getattr(snapshot,'observation',None)
    frame = getattr(observation,'sensor_frame',None)
    if frame is None:
        return clearance
    stamps = [getattr(snapshot,'timestamp',float('nan')),
              getattr(frame,'depth_timestamp',float('nan'))]
    if not np.isfinite(stamps).all() or abs(stamps[0]-stamps[1]) > planner_config.map_timeout:
        return clearance
    current,target,stopping_point = map(lambda p:np.asarray(p,dtype=float),
                                      (current,target,stopping_point))
    if any(p.shape != (3,) or not np.isfinite(p).all() for p in (current,target,stopping_point)):
        return clearance
    from navigation.route import observed_body_segment_free
    if (not observed_body_segment_free(snapshot,current,target,body_config)
            or not observed_body_segment_free(snapshot,current,stopping_point,body_config)):
        return clearance
    depth = np.asarray(frame.depth)
    k = np.asarray(frame.camera_intrinsics,dtype=float)
    pose = np.asarray(observation.camera_pose_world,dtype=float)
    if (depth.ndim != 2 or k.shape != (3,3) or pose.shape != (4,4)
            or not np.isfinite(k).all() or not np.isfinite(pose).all()
            or k[0,0] <= 0 or k[1,1] <= 0):
        return clearance
    # A point intersecting either swept body cannot be farther than this bound
    # from the camera. Process every nearby pixel without stride decimation.
    margin = max(.05,.5*float(snapshot.resolution))
    radius = body_config.clearance_radius+margin
    half = body_config.vertical_half_extent+margin
    reach = max(np.linalg.norm(p-pose[:3,3]) for p in (current,target,stopping_point))
    reach += np.hypot(radius,half)
    settings = (float(getattr(map_config,'min_depth',.3)),bool(getattr(map_config,'depth_is_range',False)),
                float(np.ceil(max(2.,reach))))
    cached = None if cache is None else cache.get('depth_points')
    if cached is not None and cached[0] is observation and cached[1] == settings:
        points = cached[2]
    else:
        valid = np.isfinite(depth) & (depth >= settings[0]) & (depth <= settings[2])
        rows,columns = np.nonzero(valid)
        if not len(rows):
            return clearance
        values = depth[rows,columns].astype(float)
        rays = np.column_stack(((columns-k[0,2])/k[0,0],(rows-k[1,2])/k[1,1],np.ones(len(rows))))
        if settings[1]:
            rays /= np.linalg.norm(rays,axis=1)[:,None]
        points = np.einsum('ij,kj->ki',pose[:3,:3],rays*values[:,None])+pose[:3,3]
        points.setflags(write=False)
        if cache is not None:
            # One immutable observation only; replace atomically across the
            # planning/control threads instead of accumulating image history.
            cache['depth_points'] = (observation,settings,points)
    if any(_hits_body_sweep(points,current,end,radius,half) for end in (target,stopping_point)):
        return clearance
    # The normal near-obstacle speed limiter at this value permits <=0.1 m/s.
    return stop
