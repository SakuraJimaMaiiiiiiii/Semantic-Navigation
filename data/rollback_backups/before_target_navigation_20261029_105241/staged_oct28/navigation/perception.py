"""Publish complete perception frames without sharing mutable mapper state."""

from copy import deepcopy
from dataclasses import dataclass
import threading
import time
from mapping.semantic_voxels import SemanticVoxelSnapshot


@dataclass(frozen=True)
class SemanticFrame:
    timestamp: float
    completed_at: float
    inference_seconds: float
    records: tuple[dict, ...]
    semantic_voxels: SemanticVoxelSnapshot | None = None
    vehicle_observation_pending: bool = False
    object_inspection_candidates: tuple[dict, ...] = ()


class SemanticFrameStore:
    def __init__(self, clock=time.monotonic):
        self._lock = threading.Lock()
        self._frame = None
        self._clock = clock

    def publish(self, timestamp, records, inference_seconds, semantic_voxels=None, *, vehicle_observation_pending=False, object_inspection_candidates=()):
        with self._lock:
            if self._frame is not None and timestamp <= self._frame.timestamp:
                return
            self._frame = SemanticFrame(
                float(timestamp),
                self._clock(),
                float(inference_seconds),
                tuple(deepcopy(records)),
                deepcopy(semantic_voxels),
                bool(vehicle_observation_pending),
                tuple(deepcopy(object_inspection_candidates)),
            )

    def snapshot(self):
        with self._lock:
            return deepcopy(self._frame)
