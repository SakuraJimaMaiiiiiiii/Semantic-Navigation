"""One planner owner publishes results; only the flight thread sends commands."""
from dataclasses import dataclass, replace
import threading
import time
import numpy as np


@dataclass(frozen=True)
class WaypointPlanUpdate:
    plan: object
    certificate: object
    duration: float
    target: np.ndarray


class WaypointPlanningWorker:
    def __init__(self, planner):
        self.planner = planner
        self._condition = threading.Condition()
        self._target = None
        self._generation = 0
        self._pending = None
        self._latest = None
        self._error = None
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name='WaypointPlanning', daemon=True)
        self._thread.start()

    def poll(self, current, target, velocity):
        with self._condition:
            if self._stopped:
                raise RuntimeError('Waypoint planner is closed')
            if self._target is None or not np.array_equal(target, self._target):
                self._generation += 1
                self._target = np.asarray(target).copy()
                # Keep a completed certificate during a rolling waypoint
                # handoff; a late solve from the OLD generation is still dropped.
            self._pending = (self._generation, np.asarray(current).copy(),
                             self._target.copy(), np.asarray(velocity).copy())
            self._condition.notify_all()
            if self._error is not None:
                raise RuntimeError('Background waypoint planning failed') from self._error
            update = self._latest
            if update is not None and not np.array_equal(update.target, self._target):
                old_point = self.tracking_plan(update, current).local_target_ned
                if old_point is not None and (np.linalg.norm(old_point-current) <= .1
                                             or np.dot(old_point-current, self._target-current) <= 0):
                    retarget = getattr(self.planner, 'retarget_clear_plan', None)
                    if callable(retarget):
                        plan = retarget(update.plan, current, self._target, update.certificate)
                        if plan is not None:
                            return WaypointPlanUpdate(plan, update.certificate, update.duration,
                                                      self._target.copy())
                    return None  # Never pull backwards during a waypoint handoff.
            return update

    def tracking_plan(self, update, current):
        """Move the lookahead along the published curve without asking the solver."""
        plan = update.plan
        path = np.asarray(plan.path_ned)
        if plan.status not in {'ego_clear', 'ego_avoid'} or len(path) < 2:
            return plan
        segments = np.diff(path, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        squared = np.einsum('ij,ij->i', segments, segments)
        fraction = np.divide(np.einsum('ij,ij->i', np.asarray(current)-path[:-1], segments),
                             squared, out=np.zeros_like(squared), where=squared>1e-12)
        fraction = np.clip(fraction, 0., 1.)
        projection = path[:-1]+fraction[:,None]*segments
        closest = int(np.argmin(np.linalg.norm(projection-current, axis=1)))
        distance = np.r_[0., np.cumsum(lengths)]
        progress = distance[closest]+fraction[closest]*lengths[closest]
        lookahead = min(self.planner.config.local_target_distance,
                        max(.25, plan.speed_limit*self.planner.config.ego_trajectory_lookahead_time))
        goal = min(distance[-1], progress+lookahead)
        point = np.array([np.interp(goal, distance, path[:,axis]) for axis in range(3)])
        return replace(plan, local_target_ned=point)

    def invalidate(self):
        with self._condition:
            self._generation += 1
            self._pending = self._latest = None
            self._error = None
            self._condition.notify_all()

    def close(self):
        with self._condition:
            self._stopped = True
            self._pending = self._latest = None
            self._condition.notify_all()
        self._thread.join(timeout=10.)
        if self._thread.is_alive():
            raise TimeoutError('Waypoint planning worker did not stop')

    def _run(self):
        next_update = 0.
        while True:
            with self._condition:
                while not self._stopped:
                    remaining = next_update-time.monotonic()
                    if self._pending is not None and remaining <= 0:
                        break
                    self._condition.wait(remaining if self._pending is not None else None)
                if self._stopped:
                    return
                generation, current, target, velocity = self._pending
                self._pending = None
            started = time.monotonic()
            try:
                updater = getattr(self.planner, 'set_motion_state', None)
                if callable(updater):
                    updater(velocity)
                compute = getattr(self.planner, 'compute_plan', self.planner.plan)
                plan = compute(current, target)
                certify = getattr(self.planner, 'execution_certificate', None)
                certificate = certify() if callable(certify) else None
                points = np.asarray(plan.path_ned).copy()
                points.setflags(write=False)
                target_point = None if plan.local_target_ned is None else np.asarray(plan.local_target_ned).copy()
                if target_point is not None:
                    target_point.setflags(write=False)
                plan = replace(plan, path_ned=points, local_target_ned=target_point)
                result = WaypointPlanUpdate(plan, certificate, time.monotonic()-started, target.copy())
                error = None
            except Exception as failure:
                result, error = None, failure
            with self._condition:
                if self._stopped:
                    return
                if generation == self._generation:
                    self._latest, self._error = result, error
            next_update = started+1./self.planner.config.replan_rate
