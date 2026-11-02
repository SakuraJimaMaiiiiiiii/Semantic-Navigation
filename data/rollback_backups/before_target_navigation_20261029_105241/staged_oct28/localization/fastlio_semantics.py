"""Timestamp-matched semantic measurements in the LIO map, never navigation."""
from collections import Counter, deque
import threading

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from mapping.loop_semantic_replay import transform_measurement
from mapping.semantic_instance_mapper import SemanticInstanceMapper, instance_class_name
from localization.scene_graph_adapter import SceneGraphAdapter


class FastLioSemantics:
    def __init__(self):
        self.mapper = SemanticInstanceMapper(sign_min_observations=10)
        self.poses = deque(maxlen=4096)
        self.pending = deque()
        self.lock = threading.Lock()
        self.closed = False
        self.counts = Counter()
        self.last_stamp = -float('inf')
        self.records = []
        self.scene_graph = SceneGraphAdapter()
        self.last_transform = None

    def submit(self, packet):
        with self.lock:
            if self.closed:
                return
            if len(self.pending) >= 128:
                self.pending.popleft()
                self.counts['queue_overflow'] += 1
            self.pending.append(packet)

    def add_pose(self, stamp, correction):
        if not np.isfinite(stamp) or (self.poses and stamp <= self.poses[-1][0]):
            raise ValueError('LIO shared timestamps must increase')
        self.poses.append((stamp, correction.copy()))
        self.drain()

    def correction_at(self, stamp):
        times = np.array([p[0] for p in self.poses])
        if not len(times) or stamp < times[0] or stamp > times[-1]:
            return None
        i = int(np.searchsorted(times, stamp))
        if times[i] == stamp:
            return self.poses[i][1].copy()
        if times[i] - times[i-1] > .5:
            return None
        a, b = self.poses[i-1][1], self.poses[i][1]
        fraction = (stamp-times[i-1])/(times[i]-times[i-1])
        result = np.eye(4)
        result[:3, :3] = Slerp([0., 1.], Rotation.from_matrix(np.stack([a[:3,:3], b[:3,:3]])))(fraction).as_matrix()
        result[:3, 3] = a[:3, 3]*(1-fraction) + b[:3, 3]*fraction
        return result

    def drain(self, final=False):
        with self.lock:
            if final:
                self.closed = True
            packets = []
            while self.pending:
                stamp = float(self.pending[0]['timestamp'])
                if not final and np.isfinite(stamp) and (not self.poses or stamp > self.poses[-1][0]):
                    break
                packets.append(self.pending.popleft())
        for packet in packets:
            stamp = float(packet['timestamp'])
            if not np.isfinite(stamp) or stamp <= self.last_stamp:
                self.counts['nonmonotonic_or_duplicate'] += 1
                continue
            self.last_stamp = stamp
            transform = self.correction_at(stamp)
            if transform is None:
                self.counts['unmatched_time'] += 1
                if not self.poses:
                    reason = 'no_tracking_pose'
                elif stamp < self.poses[0][0]:
                    reason = 'before_pose_history'
                elif stamp > self.poses[-1][0]:
                    reason = 'after_pose_history'
                else:
                    reason = 'pose_gap_over_0_5_seconds'
                self.counts[reason] += 1
                continue
            samples = packet['samples']
            measurements = [transform_measurement(s['measurement'], transform) for s in samples]
            identities = self.mapper.update([s['class_name'] for s in samples],
                               [s['confidence'] for s in samples], measurements, stamp)
            self.scene_graph.observe(stamp, samples, measurements, identities, transform, instance_class_name)
            self.counts['matched_frames'] += 1
            self.counts['measurements'] += len(samples)
            self.last_transform = transform
        if packets:
            self.records = self.mapper.navigation_records(3)
