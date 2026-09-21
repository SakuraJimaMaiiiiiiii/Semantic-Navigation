"""Publish complete perception frames without sharing mutable mapper state."""

from copy import deepcopy
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class SemanticFrame:
    timestamp: float
    completed_at: float
    inference_seconds: float
    records: tuple[dict, ...]


class SemanticFrameStore:
    def __init__(self, clock=time.monotonic):
        self._lock = threading.Lock()
        self._frame = None
        self._clock = clock

    def publish(self, timestamp, records, inference_seconds):
        with self._lock:
            if self._frame is not None and timestamp <= self._frame.timestamp:
                return
            self._frame = SemanticFrame(
                float(timestamp),
                self._clock(),
                float(inference_seconds),
                tuple(deepcopy(records)),
            )

    def snapshot(self):
        with self._lock:
            return deepcopy(self._frame)
