"""Apply the camera front-depth scalar to motion toward its viewing direction."""
import math
import numpy as np


def motion_forward_clearance(snapshot, current, target, velocity):
    clearance = float(snapshot.forward_clearance)
    observation = getattr(snapshot, 'observation', None)
    if observation is None:
        return clearance
    forward = np.asarray(observation.camera_pose_world)[:3, 2]
    direction = np.asarray(target) - np.asarray(current)
    velocity = np.asarray(velocity)
    if not (np.isfinite(forward).all() and np.isfinite(direction).all()
            and np.isfinite(velocity).all()):
        return clearance
    distance = np.linalg.norm(direction)
    # Only retreating chords may ignore front depth. Lateral/forward motion,
    # or residual forward velocity, retains the original emergency check.
    # Occupancy clearance and braking validation remain mandatory at callers.
    if distance > 1e-6 and np.dot(direction / distance, forward) < -.1 and np.dot(velocity, forward) <= 0.:
        return math.inf
    return clearance
