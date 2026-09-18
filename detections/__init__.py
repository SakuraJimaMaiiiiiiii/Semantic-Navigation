"""目标检测接口。"""

from .object_detect import (
    Detection,
    DetectorConfig,
    TiledVehicleDetector,
    SAM2Segmenter,
)
from .realtime_detection_worker import RealtimeDetectionWorker

__all__ = [
    "Detection",
    "DetectorConfig",
    "RealtimeDetectionWorker",
    "TiledVehicleDetector",
    "SAM2Segmenter",
]
