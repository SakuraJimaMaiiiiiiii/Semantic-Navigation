"""Asynchronous LiDAR revisit verification in PX4 NED; no flight/map corrections."""
import threading
import numpy as np
from scipy.spatial import cKDTree

from navigation.scene_memory import write_json


def verify_clouds(source, target, maximum_distance, minimum_fitness, maximum_rmse):
    """Rigid ICP plus symmetric overlap and non-planar support checks."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if min(len(source), len(target)) < 50:
        return None
    for cloud in (source, target):
        spectrum = np.linalg.eigvalsh(np.cov(cloud.T))
        if spectrum[0] < .01*spectrum[-1]:
            return None
    moved, transform = source.copy(), np.eye(4)
    tree = cKDTree(target)
    for _ in range(30):
        distance, indices = tree.query(moved)
        mask = distance < maximum_distance
        if mask.sum() < 50:
            return None
        a, b = moved[mask], target[indices[mask]]
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        u, _, vt = np.linalg.svd((a-ca).T@(b-cb))
        r = vt.T@u.T
        if np.linalg.det(r) < 0:
            vt[-1] *= -1
            r = vt.T@u.T
        delta = np.eye(4)
        delta[:3,:3], delta[:3,3] = r, cb-r@ca
        moved = moved@r.T+delta[:3,3]
        transform = delta@transform
        if np.linalg.norm(delta-np.eye(4)) < 1e-5:
            break
    forward, _ = tree.query(moved)
    backward, _ = cKDTree(moved).query(target)
    fitness = min(float(np.mean(forward < maximum_distance)), float(np.mean(backward < maximum_distance)))
    residual = np.concatenate((forward[forward < maximum_distance], backward[backward < maximum_distance]))
    rmse = float(np.sqrt(np.mean(residual**2))) if len(residual) else float('inf')
    if fitness < minimum_fitness or rmse > maximum_rmse:
        return None
    return transform, fitness, rmse


class LidarLoopDetector:
    def __init__(self, config, output):
        self.config, self.output = config, output
        self.frames, self.constraints = [], []
        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self._thread = None
        self.error = None
        self.mapping_snapshot = ((), ())  # Atomic immutable container, clouds never modified.

    def start(self):
        self._save()
        self._thread = threading.Thread(target=self._run, name='LidarLoopDetection', daemon=True)
        self._thread.start()
        print(f'--- 雷达回环检测已启动：生成约束供后台位姿图优化；结果={self.output} ----')

    def submit(self, position, points, timestamp):
        with self._condition:
            if not self._closed:
                self._pending = (position.copy(), points.copy(), float(timestamp))
                self._condition.notify()

    def close(self):
        if getattr(self, '_archive_saved', False):
            return
        with self._condition:
            self._closed = True
            self._condition.notify()
        if self._thread is not None:
            self._thread.join(timeout=10.)
            if self._thread.is_alive():
                raise TimeoutError('LiDAR loop detector did not stop')
        if self.frames:
            from dataclasses import asdict
            import json
            from pathlib import Path
            archive = Path(self.output).with_name('lidar_loop_keyframes.npz')
            archive.parent.mkdir(parents=True, exist_ok=True)
            lengths = [len(cloud) for _, cloud, _ in self.frames]
            with archive.with_suffix('.npz.tmp').open('wb') as stream:
                np.savez_compressed(stream,
                    origins=np.asarray([p for p, _, _ in self.frames]),
                    timestamps=np.asarray([t for _, _, t in self.frames]),
                    offsets=np.r_[0, np.cumsum(lengths)],
                    points_world=np.concatenate([cloud for _, cloud, _ in self.frames]),
                    constraints_json=json.dumps(self.constraints),
                    config_json=json.dumps(asdict(self.config), default=str))
            archive.with_suffix('.npz.tmp').replace(archive)
        self._save()
        self._archive_saved = True

    def _save(self):
        write_json(self.output, {'mode': 'lidar_geometric_loop_detection',
            'pose_source': 'px4', 'map_correction_applied': False,
            'keyframes': len(self.frames), 'constraints': self.constraints, 'error': self.error})

    def _run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._pending is not None)
                    if self._closed:
                        break
                    sample, self._pending = self._pending, None
                self.process(*sample)
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
            with self._condition:
                self._closed = True
                self._pending = None
            print(f'--- LiDAR loop detection stopped: {self.error} ----')
            try:
                self._save()
            except Exception as save_error:
                print(f'--- LiDAR loop report failed: {save_error} ----')

    def process(self, position, points, timestamp):
        c = self.config
        if len(self.frames) >= c.max_keyframes:
            return
        if self.frames and np.linalg.norm(position-self.frames[-1][0]) < c.keyframe_translation:
            return  # Stationary spinning/hovering is not a revisit.
        points = np.asarray(points, float)
        points = points[np.isfinite(points).all(axis=1)]
        _, index = np.unique(np.floor(points/c.icp_voxel_size).astype(np.int64), axis=0, return_index=True)
        cloud = points[index]
        if len(cloud) < 50:
            return
        if len(cloud) > 4000:
            cloud = cloud[np.linspace(0, len(cloud)-1, 4000, dtype=int)]
        current = len(self.frames)
        candidates = []
        for i, (origin, _, stamp) in enumerate(self.frames):
            if current-i < c.loop_min_separation or timestamp-stamp < 30.:
                continue
            distance = np.linalg.norm(position-origin)
            if distance > c.loop_search_radius:
                continue
            if max(np.linalg.norm(p-origin) for p, _, _ in self.frames[i:]) < 2*c.loop_search_radius:
                continue
            candidates.append((distance, i))
        for _, i in sorted(candidates)[:c.loop_max_candidates]:
            result = verify_clouds(cloud, self.frames[i][1], c.icp_max_correspondence,
                                   max(.65, c.icp_min_fitness), c.icp_max_rmse)
            if result is None:
                continue
            transform, fitness, rmse = result
            origin = cloud.mean(axis=0)
            displacement = np.linalg.norm(transform[:3,:3]@origin+transform[:3,3]-origin)
            angle = np.degrees(np.arccos(np.clip((np.trace(transform[:3,:3])-1)/2, -1, 1)))
            if displacement > c.loop_max_translation_correction or angle > c.loop_max_yaw_correction_degrees:
                continue
            self.constraints.append({'source': current, 'target': i, 'timestamp': timestamp,
                'source_to_target_world_transform': transform.tolist(), 'symmetric_fitness': fitness, 'rmse_m': rmse})
            print(f'--- LiDAR loop verified: K{current} -> K{i}, RMSE={rmse:.3f}m; constraint accepted ----')
            break
        self.frames.append((position.copy(), cloud, timestamp))
        self.mapping_snapshot = (tuple(self.frames), tuple(self.constraints))
        self._save()
