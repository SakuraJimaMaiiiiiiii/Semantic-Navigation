"""Background corrected mapping; PX4 navigation keeps its continuous odometry frame."""
from concurrent.futures import CancelledError
from copy import deepcopy
import numpy as np
import threading
import time
from pathlib import Path

from navigation.atomic_file import atomic_write_text
from navigation.scene_memory import write_json
from .lidar_loop_map import optimize_frames, rebuild
from .loop_semantic_replay import export_bundle, correction_at


class LiveLoopMapping:
    def __init__(self, detector, run, home, yaw, interval=10., navigation_map=None):
        self.detector = detector
        self.navigation_map = navigation_map
        self.run = Path(run)
        self.output = self.run / 'maps/live_loop_corrected'
        self.source = dict(home_ned=list(home), initial_yaw_rad=float(yaw))
        self.interval = max(1., float(interval))
        self._stop = threading.Event()
        self._thread = None
        self._published = None
        self._navigation_version = None
        self.map_path = self.run / 'exploration/map.html'
        self._display_lock = threading.RLock()
        self._display = dict(state='exploring', vehicle=None, route=None, approach=None, waypoints=[])
        self._waypoint_stamps = {}
        self._scene = None

    def start(self):
        atomic_write_text(self.map_path,
            '<meta charset="utf-8"><meta http-equiv="refresh" content="10">'
            '<p>实时回环地图：等待雷达关键帧和语义观测。有效回环后，校正几何转换到 PX4 坐标参与导航。</p>')
        self._thread = threading.Thread(target=self._run, name='LiveLoopMapping', daemon=True)
        self._thread.start()
        print(f'--- 实时回环校正地图：{self.map_path} ----')

    def update(self):
        # Only the latest immutable snapshot is consumed: no growing job queue.
        frames, constraints = self.detector.mapping_snapshot
        history = self.run / 'maps/semantic_observations.jsonl'
        if not frames:
            return
        version = (len(frames), len(constraints), history.stat().st_size if history.exists() else 0)
        if version == self._published:
            return
        started = time.monotonic()
        optimized, corrections = optimize_frames(frames, constraints, self.detector.config, check_cancelled=self._check_cancelled)
        navigation_applied = False
        geometry_version = (len(frames), len(constraints))
        if constraints and self.navigation_map is not None:
            if geometry_version != self._navigation_version:
                navigation_applied = self.navigation_map.publish(optimized, corrections, check_cancelled=self._check_cancelled)
                if navigation_applied:
                    self._navigation_version = geometry_version
            else:
                navigation_applied = True
        if not history.exists():
            write_json(self.output/'status.json', dict(state='waiting_for_semantics',
                online_navigation_applied=navigation_applied,
                navigation_revision=getattr(self.navigation_map, 'revision', 0)))
            return
        snapshot = rebuild(optimized, self.detector.config, check_cancelled=self._check_cancelled)
        destination, count = export_bundle(self.run, self.output, snapshot,
            [f[2] for f in frames], corrections, source=self.source, live=True, check_cancelled=self._check_cancelled,
            on_publish=lambda snapshot, scene: self._publish_display(snapshot, scene, corrections, [f[2] for f in frames]))
        write_json(self.output/'status.json', dict(state='ready', revision=destination.name,
            keyframes=len(frames), loops=len(constraints), measurements=count,
            through_timestamp=frames[-1][2], duration_seconds=time.monotonic()-started,
            coordinate_frame='loop_corrected_ned', navigation_frame='px4_ned',
            online_navigation_applied=navigation_applied,
            navigation_revision=getattr(self.navigation_map, 'revision', 0)))
        self._published = version

    def update_display(self, *, state, vehicle=None, route=None, approach=None, waypoints=None, timestamp=None):
        with self._display_lock:
            points = deepcopy(list(waypoints or []))
            for point in points:
                key = (point['number'], tuple(point['position']))
                self._waypoint_stamps.setdefault(key, timestamp)
                point['_display_timestamp'] = self._waypoint_stamps[key]
            self._display = deepcopy(dict(state=state, vehicle=vehicle, route=route, approach=approach, waypoints=points))
            self._render_display()

    def _publish_display(self, snapshot, scene, corrections, timestamps):
        with self._display_lock:
            self._scene = (snapshot, scene, np.asarray(corrections).copy(), list(timestamps))
            self._render_display()

    def _render_display(self):
        if self._scene is None:
            return
        from navigation.map_view import render_map
        snapshot, scene, corrections, timestamps = self._scene
        correction = corrections[-1]
        def transform(value):
            if value is None:
                return None
            points = np.asarray(value, dtype=float)
            if points.size == 0:
                return np.empty((0, 3))
            return points @ correction[:3,:3].T + correction[:3,3]
        waypoints = deepcopy(self._display['waypoints'])
        for point in waypoints:
            stamp = point.pop('_display_timestamp')
            transform_at_selection = correction_at(stamp, timestamps, corrections) if stamp is not None else correction
            point['position'] = (np.asarray(point['position']) @ transform_at_selection[:3,:3].T
                                 + transform_at_selection[:3,3]).tolist()
        render_map(self.map_path, snapshot, scene,
            exploration_waypoints=waypoints,
            state='回环校正地图 · ' + self._display['state'],
            vehicle=transform(self._display['vehicle']),
            route=transform(self._display['route']),
            approach=transform(self._display['approach']))

    def _check_cancelled(self):
        if self._stop.is_set():
            raise CancelledError()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.update()
            except CancelledError:
                return
            except Exception as error:
                # Keep the last complete map; report failure rather than publish mixed frames.
                print(f'--- 实时回环地图更新失败：{type(error).__name__}: {error} ----')
                try:
                    write_json(self.output/'status.json', dict(state='failed', error=str(error)))
                except OSError:
                    pass
            self._stop.wait(self.interval)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=60.)
            if self._thread.is_alive():
                raise TimeoutError('Live loop map reconstruction did not stop')

    def finalize(self):
        """Publish the closed input logs once; never restart background writes.

        Call after semantic recording and the keyframe detector have stopped.
        Canceling the background job alone can leave late identity merges out
        of the HTML even though the raw semantic map already contains them.
        """
        self.close()
        self._stop.clear()
        try:
            self.update()
        finally:
            self._stop.set()
