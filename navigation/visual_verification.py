"""Bounded saved-target verification paced by real completed observations."""
import math
import numpy as np


def verify_saved_target(nav, record, goal):
    nav._event('VERIFY',target_id=record['runtime_instance_id'])
    started = nav.clock()
    hard_deadline = min(nav.deadline,started+max(45.,nav.config.verification_timeout))
    delta = np.asarray(record['position_world'])-nav.vehicle.get_position()
    heading = math.atan2(delta[1],delta[0])
    nav._align(heading,deadline=hard_deadline)
    baseline = nav._frame(deadline=hard_deadline)
    last_timestamp, last_completed = baseline.timestamp,baseline.completed_at
    period = max(.2,float(baseline.inference_seconds))
    deadline = min(hard_deadline,nav.clock()+max(nav.config.verification_timeout,
                                               (nav.config.verification_frames+3)*period))
    nav._event('VERIFICATION_BUDGET',target_id=record['runtime_instance_id'],
               frame_period_seconds=period,budget_seconds=deadline-started)
    observed,matched,accepted = set(),set(),set()
    last_distance = None
    views,missed = 1,0
    while nav.clock() < deadline:
        nav._check_health()
        try:
            frame = nav._frame(deadline=deadline)
        except RuntimeError as error:
            if str(error) != 'semantic_perception_stale':
                raise
            break
        if frame.timestamp > last_timestamp:
            interval = frame.completed_at-last_completed
            last_timestamp,last_completed = frame.timestamp,frame.completed_at
            period = max(period,float(frame.inference_seconds),interval)
            # A slower new completion can extend the window, never the hard cap.
            deadline = min(hard_deadline,max(deadline,started+(nav.config.verification_frames+3)*period))
            observed.add(frame.timestamp)
            current = nav.vehicle.get_position()
            match = next((r for r in nav._verification_candidates(frame,current)
                          if r['runtime_instance_id'] == record['runtime_instance_id']
                          and abs(r['last_seen']-frame.timestamp) < 1e-6),None)
            if match is not None:
                matched.add(frame.timestamp)
                missed = 0
                low,high = np.asarray(match['bbox_3d']['min']),np.asarray(match['bbox_3d']['max'])
                last_distance = float(np.linalg.norm(np.maximum(
                    np.maximum(low[:2]-current[:2],current[:2]-high[:2]),0.)))
                if (np.linalg.norm(current-goal) <= nav.config.arrival_tolerance
                        and abs(last_distance-nav.config.standoff_distance) <= nav.config.arrival_tolerance):
                    accepted.add(frame.timestamp)
                    if len(accepted) >= nav.config.verification_frames:
                        return match
            else:
                missed += 1
            if missed >= 2 and views < 3 and deadline-nav.clock() >= period+1.:
                yaw = heading+math.radians(20. if views == 1 else -20.)
                try:
                    nav._align(yaw,deadline=deadline)
                except TimeoutError:
                    if nav.clock() < deadline or nav.clock() >= nav.deadline:
                        raise
                    break
                views += 1
                missed = 0
                # Skip frames already published during the turn. Every counted
                # confirmation still needs its own fresh observation timestamp.
                turned_frame = nav._frame(deadline=deadline)
                last_timestamp = max(last_timestamp,turned_frame.timestamp)
                last_completed = max(last_completed,turned_frame.completed_at)
                nav._event('VERIFICATION_VIEW',target_id=record['runtime_instance_id'],view=views)
        remaining = deadline-nav.clock()
        if remaining > 0:
            nav.vehicle.wait(min(.2,remaining))
    nav._event('VERIFICATION_INCOMPLETE',target_id=record['runtime_instance_id'],
               observed_frames=len(observed),matched_frames=len(matched),accepted_frames=len(accepted),
               required_frames=nav.config.verification_frames,last_surface_distance_m=last_distance,
               views=views,elapsed_seconds=nav.clock()-started,
               reason=('insufficient_fresh_frames' if len(observed) < nav.config.verification_frames
                       else 'target_not_reobserved' if not matched else 'verification_geometry_or_frame_count'))
    return None
