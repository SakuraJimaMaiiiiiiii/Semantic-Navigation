"""Runtime wiring and perception publication without loading model weights."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import json
import sys
import time

import numpy as np
import pytest

from application.demo_runtime import DemoRuntime
from config.semantic_navigation_config import SemanticNavigationConfig
from navigation.target import SemanticTargetRequest
from detections.object_detect import DetectorConfig
from detections.realtime_detection_worker import RealtimeDetectionWorker
from mapping.semantic_instance_mapper import (
    SemanticInstanceMapper,
    SemanticObservation3D,
)


def test_semantic_mode_enables_consistent_global_memory(tmp_path):
    runtime = DemoRuntime.__new__(DemoRuntime)
    runtime.run_output_dir = tmp_path
    runtime.semantic_target = SemanticTargetRequest(class_name="car")
    runtime.semantic_navigation_config = SemanticNavigationConfig()
    runtime._create_configs()
    assert runtime.global_map_config.enabled
    assert not runtime.global_map_config.loop_closure_enabled
    assert runtime.planner_config.unknown_is_occupied
    assert runtime.mapping_config.vehicle_free_radius == 0.35
    assert runtime.global_map_config.vehicle_free_radius == 0.35
    runtime.semantic_target = None
    runtime._create_configs()
    assert not runtime.global_map_config.enabled
    assert runtime.mapping_config.vehicle_free_radius == 0


def test_runtime_rejects_unstructured_target_before_allocating_resources():
    with pytest.raises(TypeError):
        DemoRuntime(semantic_target={"class_name": "car"})


def test_entry_passes_requested_target_and_lands_on_failure(tmp_path, monkeypatch):
    import run_demo

    request = tmp_path / "target.json"
    request.write_text('{"target": {"instance_id": "vehicle_02"}}')
    monkeypatch.setattr(sys, "argv", ["run_demo.py", "--semantic-target", str(request)])
    runtime = MagicMock()
    runtime.run_mission.side_effect = ValueError("invalid map")
    with patch.object(run_demo, "WebSocketRequestSender"), patch.object(
        run_demo, "DemoRuntime", return_value=runtime
    ) as constructor:
        with pytest.raises(ValueError, match="invalid map"):
            run_demo.main()
    assert constructor.call_args.kwargs["semantic_target"].instance_id == "vehicle_02"
    runtime.land_if_armed.assert_called_once_with(hover_first=True)
    runtime.close.assert_called_once()


def test_worker_publishes_complete_map_and_writes_history(tmp_path):
    stamp = 100.0
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    mapper = SemanticInstanceMapper()
    obs = SemanticObservation3D(
        np.array([5.0, 0.0, -1.5]),
        np.array([4.8, -0.2, -3.0]),
        np.array([5.2, 0.2, 0.0]),
        np.eye(3) * 0.1,
    )
    for timestamp in (98.0, 99.0, 100.0):
        mapper.update(["column"], [0.9], [obs], timestamp)
    observation = SimpleNamespace(
        sensor_frame=SimpleNamespace(
            rgb=image,
            depth=np.ones((32, 48)),
            camera_intrinsics=np.eye(3),
            timestamp=stamp,
        ),
        camera_pose_world=np.eye(4),
    )

    class Source:
        delivered = False

        def get_observation(self, timeout):
            if not self.delivered:
                self.delivered = True
                return observation
            time.sleep(0.01)
            raise TimeoutError()

    detector = SimpleNamespace(
        detect=lambda image: [], draw=lambda image, detections: image.copy()
    )
    segmenter = SimpleNamespace(
        mapper=mapper,
        segment=lambda image, *args, **kwargs: (image.copy(), 0),
        save_semantic_map=lambda: None,
    )
    config = DetectorConfig(
        device="cpu",
        save_videos=False,
        semantic_map_path=tmp_path / "semantic_instances.json",
    )
    with patch(
        "detections.realtime_detection_worker.TiledVehicleDetector",
        return_value=detector,
    ), patch(
        "detections.realtime_detection_worker.SAM2Segmenter", return_value=segmenter
    ):
        worker = RealtimeDetectionWorker(Source(), config)
    worker.start()
    try:
        deadline = time.monotonic() + 2
        while worker.processed_frames == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        worker.close()
    worker.raise_if_failed()
    frame = worker.semantic_snapshot()
    assert frame.timestamp == stamp
    assert frame.records[0]["runtime_instance_id"] == "column_01"
    payload = json.loads((tmp_path / "semantic_observations.jsonl").read_text())
    assert payload["records"][0]["position_world"] == [5.0, 0.0, -1.5]
