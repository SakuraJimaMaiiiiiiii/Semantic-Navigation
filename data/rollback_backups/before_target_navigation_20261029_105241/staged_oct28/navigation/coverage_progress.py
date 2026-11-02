"""Sliding-window expansion evidence, not a ground-truth coverage percentage."""
from collections import deque


class CoverageProgress:
    def __init__(self, minimum_gain, plateau_seconds, relative_gain=.01):
        self.minimum_gain = minimum_gain
        self.plateau_seconds = plateau_seconds
        self.relative_gain = relative_gain
        self.level = None
        self.known = set()
        self.samples = deque()
        self.metrics = {}
        self.review_started = None
        self.review_area = 0.
        self.review_threshold = 0.

    def update(self, grid, altitude, now, visits):
        level = (round(float(altitude), 3), float(grid.resolution))
        if (self.level is None or level[1] != self.level[1]
                or abs(level[0]-self.level[0]) >= .5*level[1]):
            self.level = level
            self.known = set()
            self.samples.clear()
            self.review_started = None
        self.known.update(grid.known)
        area = len(self.known) * grid.resolution ** 2
        self.samples.append((now, area, visits))
        # Keep the sample just before the window boundary, even when decisions
        # are sparse. Compare growth rate over the actual interval, not a reset
        # timer that tiny increments can indefinitely postpone.
        while len(self.samples) > 1 and self.samples[1][0] <= now-self.plateau_seconds:
            self.samples.popleft()
        start, start_area, start_visits = self.samples[0]
        elapsed = max(0., now-start)
        gain = max(0., area-start_area)
        normalized_gain = gain*self.plateau_seconds/elapsed if elapsed else 0.
        threshold = max(self.minimum_gain, start_area*self.relative_gain)
        ready = (elapsed >= self.plateau_seconds
                 and normalized_gain <= threshold)
        # Latch review: threshold jitter must not restart its budget forever.
        if self.review_started is not None and area-self.review_area >= 2*self.review_threshold:
            self.review_started = None
            self.samples.clear()
            self.samples.append((now, area, visits))
            ready = False
        if ready and self.review_started is None:
            self.review_started, self.review_area, self.review_threshold = now, area, threshold
        reviewing = self.review_started is not None
        remaining = max(0., self.plateau_seconds-(now-self.review_started)) if reviewing else None
        review_due = reviewing and remaining <= 0.
        stage = ('review_due' if review_due else 'review' if reviewing else
                 'collecting' if elapsed < self.plateau_seconds else 'expanding')
        self.metrics = dict(observed_area_m2=area,
                            arrivals_without_meaningful_expansion=max(0, visits-start_visits),
                            window_seconds=elapsed, window_new_area_m2=gain,
                            window_remaining_seconds=max(0., self.plateau_seconds-elapsed),
                            review_duration_seconds=self.plateau_seconds,
                            normalized_window_gain_m2=normalized_gain,
                            low_gain_threshold_m2=threshold,
                            stage=stage, review_remaining_seconds=remaining,
                            review_gain_m2=max(0., area-self.review_area) if reviewing else None,
                            review_resume_gain_m2=2*self.review_threshold if reviewing else None,
                            review_due=review_due,
                            full_map_coverage=None)
        return reviewing
