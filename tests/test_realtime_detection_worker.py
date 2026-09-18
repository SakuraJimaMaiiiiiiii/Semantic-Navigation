"""实时 YOLO worker 的无模型回归测试。"""

import json
import time
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import cv2

from detections.object_detect import (
    Detection,
    DetectorConfig,
    SAM2Segmenter,
    TiledVehicleDetector,
    canonical_class_name,
    instance_color,
    semantic_class_color,
)
from detections.realtime_detection_worker import RealtimeDetectionWorker
from mapping.semantic_instance_mapper import (
    SemanticInstanceMapper,
    SemanticObservation3D,
)


def _geometry(position, half_extent=(0.2, 0.2, 0.5)):
    position = np.asarray(position, dtype=np.float64)
    half_extent = np.asarray(half_extent, dtype=np.float64)
    return SemanticObservation3D(
        position_world=position,
        bbox_3d_min=position - half_extent,
        bbox_3d_max=position + half_extent,
        measurement_covariance=np.diag((0.09, 0.09, 0.36)),
    )


def _update(mapper, detections, observations, timestamp):
    return mapper.update(
        [detection.class_name for detection in detections],
        [detection.confidence for detection in detections],
        observations,
        timestamp=timestamp,
    )


class _OneFrameSource:
    def __init__(self, image):
        self.depth = np.full(image.shape[:2], 2.0, dtype=np.float32)
        self.camera_intrinsics = np.eye(3)
        self.camera_pose_world = np.eye(4)
        self.observation = SimpleNamespace(
            sensor_frame=SimpleNamespace(
                rgb=image,
                depth=self.depth,
                camera_intrinsics=self.camera_intrinsics,
                timestamp=1788400000.0,
            ),
            camera_pose_world=self.camera_pose_world,
        )
        self.delivered = False

    def get_observation(self, timeout=0.5):
        if not self.delivered:
            self.delivered = True
            return self.observation
        time.sleep(min(float(timeout), 0.01))
        raise TimeoutError


class _FakeDetector:
    def __init__(self):
        self.received = None

    def detect(self, image):
        self.received = image.copy()
        return []

    @staticmethod
    def draw(image, detections):
        del detections
        return image.copy()


class _FakeSegmenter:
    def __init__(self):
        self.received = None
        self.latest_memory = ()

    def segment(self, image, detections, **sensor_data):
        self.received = image.copy()
        self.detections = list(detections)
        self.sensor_data = sensor_data
        return image.copy(), 0


class RealtimeDetectionWorkerTest(unittest.TestCase):
    def test_column_prompt_aliases_use_one_canonical_class(self):
        for alias in (
            "column",
            "concrete pillar",
            "parking garage pillar",
            "support column",
            "support pillar",
            "structural column",
            "concrete column",
        ):
            self.assertEqual(canonical_class_name(alias), "column")
        self.assertEqual(canonical_class_name("car"), "car")
        self.assertEqual(
            semantic_class_color("concrete pillar"),
            semantic_class_color("column"),
        )
        self.assertNotEqual(
            semantic_class_color("column"),
            semantic_class_color("person"),
        )

    def test_nms_suppresses_same_class_but_preserves_overlapping_classes(self):
        detector = object.__new__(TiledVehicleDetector)
        detector.config = SimpleNamespace(nms_iou=0.45)
        boxes = torch.tensor([
            [0.0, 0.0, 20.0, 20.0],
            [1.0, 1.0, 19.0, 19.0],
            [0.0, 0.0, 20.0, 20.0],
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])

        keep = detector._nms(
            boxes, scores, ["column", "column", "person"]
        )

        self.assertEqual(keep.tolist(), [0, 2])

    def test_3d_tracker_keeps_column_ids_when_order_changes(self):
        tracker = SemanticInstanceMapper(
            duplicate_merge_distance_xy=1.5
        )
        left = Detection((0, 0, 20, 40), 0.9, "column")
        right = Detection((40, 0, 60, 40), 0.8, "column")
        first_ids = _update(
            tracker,
            [left, right],
            [
                _geometry((2.0, 1.0, -1.0)),
                _geometry((5.0, 1.0, -1.0)),
            ],
            timestamp=100.0,
        )
        second_ids = _update(
            tracker,
            [
                Detection((400, 0, 440, 80), 0.95, "column"),
                Detection((300, 0, 340, 80), 0.7, "column"),
            ],
            [
                _geometry((5.1, 1.0, -2.0)),
                _geometry((2.1, 1.0, -2.0)),
            ],
            timestamp=101.0,
        )

        self.assertEqual(first_ids, [1, 2])
        self.assertEqual(second_ids, [2, 1])
        self.assertNotEqual(instance_color(1), instance_color(2))

    def test_world_position_recovers_column_id_after_viewpoint_change(self):
        tracker = SemanticInstanceMapper(
            landmark_classes=("column",),
            duplicate_merge_distance_xy=1.5,
        )
        first_id = _update(
            tracker,
            [Detection((0, 0, 20, 40), 0.9, "column")],
            [_geometry((5.0, 2.0, -1.0))],
            timestamp=100.0,
        )
        recovered_id = _update(
            tracker,
            [Detection((400, 100, 430, 180), 0.8, "column")],
            [_geometry((5.3, 2.1, -2.0))],
            timestamp=101.0,
        )

        self.assertEqual(first_id, [1])
        self.assertEqual(recovered_id, [1])
        memory = tracker.snapshot()
        self.assertEqual(len(memory), 1)
        self.assertEqual(memory[0].label, "column_01")

    def test_kalman_filter_smooths_position_and_mahalanobis_rejects_outlier(self):
        mapper = SemanticInstanceMapper(
            process_noise_std=0.0,
            mahalanobis_gate=7.815,
            duplicate_merge_distance_xy=1.5,
        )
        detection = Detection((0, 0, 20, 40), 0.9, "column")

        first_id = _update(
            mapper,
            [detection],
            [_geometry((0.0, 0.0, -1.0))],
            timestamp=100.0,
        )
        second_id = _update(
            mapper,
            [detection],
            [_geometry((0.6, 0.0, -1.0))],
            timestamp=101.0,
        )

        self.assertEqual(first_id, [1])
        self.assertEqual(second_id, [1])
        filtered = mapper.snapshot()[0]
        self.assertAlmostEqual(filtered.position_world[0], 0.3, places=6)
        self.assertAlmostEqual(
            filtered.position_covariance[0][0], 0.045, places=6
        )

        outlier_id = _update(
            mapper,
            [detection],
            [_geometry((5.0, 0.0, -1.0))],
            timestamp=102.0,
        )
        self.assertEqual(outlier_id, [2])
        self.assertEqual(len(mapper.snapshot()), 2)

    def test_online_merge_uses_ned_position_without_bbox_overlap(self):
        tracker = SemanticInstanceMapper(
            duplicate_merge_distance_xy=1.5
        )
        detections = [
            Detection((0, 0, 20, 40), 0.9, "column"),
            Detection((20, 0, 40, 40), 0.7, "column"),
        ]
        ids = _update(
            tracker,
            detections,
            [
                _geometry((0.0, 0.0, -1.0), (0.7, 0.5, 1.5)),
                _geometry((1.0, 0.0, -1.0), (0.7, 0.5, 1.5)),
            ],
            timestamp=100.0,
        )

        self.assertEqual(ids, [1, 1])
        memory = tracker.snapshot()
        self.assertEqual(len(memory), 1)
        self.assertEqual(memory[0].label, "column_01")
        self.assertEqual(memory[0].observation_count, 2)
        self.assertAlmostEqual(memory[0].confidence, 0.8)
        np.testing.assert_allclose(memory[0].position_world, [0.5, 0.0, -1.0])
        self.assertIsNone(
            tracker.confirmed_instance_id("column", 1, min_observations=3)
        )
        _update(
            tracker,
            [detections[0]],
            [_geometry((0.4, 0.0, -1.0), (0.7, 0.5, 1.5))],
            timestamp=101.0,
        )
        self.assertEqual(
            tracker.confirmed_instance_id(
                "column", 1, min_observations=3
            ),
            1,
        )

        # 即使两个单帧包围盒不重叠，只要 N/E 世界坐标足够接近，
        # 仍应视为同一根柱子，而不是创建长期重复ID。
        coordinate_tracker = SemanticInstanceMapper(
            duplicate_merge_distance_xy=1.5
        )
        coordinate_ids = _update(
            coordinate_tracker,
            detections,
            [
                _geometry((0.0, 0.0, -1.0), (0.2, 0.2, 1.5)),
                _geometry((1.0, 0.0, -1.0), (0.2, 0.2, 1.5)),
            ],
            timestamp=100.0,
        )
        self.assertEqual(coordinate_ids, [1, 1])
        self.assertEqual(len(coordinate_tracker.snapshot()), 1)

    def test_semantic_map_filters_non_vertical_column_geometry(self):
        tracker = SemanticInstanceMapper()
        detections = [
            Detection((40, 0, 80, 10), 0.8, "column"),
            Detection((0, 0, 20, 40), 0.9, "column"),
            Detection((80, 0, 100, 40), 0.85, "column"),
        ]
        observations = [
            _geometry((5.0, 0.0, 0.0), (1.0, 0.3, 0.05)),
            _geometry((0.0, 0.0, -1.0), (0.5, 0.5, 1.5)),
            _geometry((10.0, 0.0, -1.0), (0.5, 0.5, 1.5)),
        ]
        for timestamp in range(3):
            _update(tracker, detections, observations, timestamp)

        self.assertEqual(
            tracker.confirmed_instance_id(
                "column", 2, min_observations=3
            ),
            1,
        )
        self.assertEqual(
            tracker.confirmed_instance_id(
                "column", 3, min_observations=3
            ),
            2,
        )
        records = tracker.semantic_map_records(min_observations=3)

        self.assertEqual(len(records), 2)
        self.assertEqual(tracker.snapshot()[1].label, "column_02")
        self.assertEqual(
            [record["instance_id"] for record in records],
            ["column_01", "column_02"],
        )
        self.assertEqual(
            [record["runtime_instance_id"] for record in records],
            ["column_02", "column_03"],
        )
        self.assertEqual(
            [record["video_instance_id"] for record in records],
            ["column_01", "column_02"],
        )

    def test_tracker_does_not_assign_ids_without_world_positions(self):
        tracker = SemanticInstanceMapper(landmark_classes=("column",))

        detections = [
            Detection((0, 0, 20, 40), 0.9, "column"),
            Detection((20, 0, 40, 40), 0.8, "person"),
        ]
        ids = _update(tracker, detections, [None, None], timestamp=100.0)

        self.assertEqual(ids, [None, None])
        self.assertEqual(tracker.snapshot(), ())

    def test_semantic_map_is_saved_but_next_run_starts_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            weights = directory / "sam2.1_t.pt"
            weights.touch()
            map_path = directory / "semantic_instances.json"
            config = DetectorConfig(
                sam_weights=weights,
                semantic_map_path=map_path,
                semantic_map_min_observations=3,
                device="cpu",
            )
            with patch("detections.object_detect.SAM", return_value=MagicMock()):
                first_flight = SAM2Segmenter(config)
                for timestamp in range(3):
                    ids = _update(
                        first_flight.mapper,
                        [Detection((0, 0, 20, 40), 0.9, "column")],
                        [
                            _geometry(
                                (5.0, 2.0, -1.0),
                                (0.2, 0.2, 1.5),
                            )
                        ],
                        timestamp=1788400000.0 + timestamp,
                    )
                first_flight.save_semantic_map()
                second_flight = SAM2Segmenter(config)

            self.assertEqual(second_flight.mapper.snapshot(), ())
            new_run_ids = _update(
                second_flight.mapper,
                [Detection((300, 50, 340, 150), 0.8, "column")],
                [_geometry((5.2, 2.1, -2.0))],
                timestamp=1788400100.0,
            )

            self.assertEqual(ids, [1])
            self.assertEqual(new_run_ids, [1])
            self.assertTrue(map_path.is_file())
            payload = json.loads(map_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["format_version"], 5)
            self.assertEqual(payload["coordinate_frame"], "NED")
            self.assertEqual(payload["instance_count"], 1)
            self.assertEqual(len(payload["instances"]), 1)
            instance = payload["instances"][0]
            self.assertEqual(instance["instance_id"], "column_01")
            self.assertEqual(
                instance["runtime_instance_id"], "column_01"
            )
            self.assertIsNone(instance["video_instance_id"])
            self.assertEqual(instance["class_name"], "column")
            self.assertIn("support pillar", instance["aliases"])
            self.assertEqual(instance["position_world"], [5.0, 2.0, -1.0])
            self.assertEqual(len(instance["position_covariance"]), 3)
            self.assertEqual(instance["observation_count"], 3)
            self.assertEqual(instance["last_seen"], 1788400002.0)
            self.assertTrue(instance["is_static"])
            self.assertTrue(instance["is_obstacle"])
            self.assertEqual(
                instance["bbox_3d"],
                {
                    "min": [4.8, 1.8, -2.5],
                    "max": [5.2, 2.2, 0.5],
                },
            )

    def test_sam_prompt_budget_prioritizes_columns(self):
        segmenter = object.__new__(SAM2Segmenter)
        segmenter.config = DetectorConfig(
            device="cpu",
            max_sam_prompts=2,
        )
        segmenter.mapper = SimpleNamespace(landmark_classes={"column"})
        detections = [
            Detection((0, 0, 20, 20), 0.99, "floor"),
            Detection((20, 0, 40, 20), 0.20, "column"),
            Detection((40, 0, 60, 20), 0.95, "person"),
        ]

        selected = segmenter._select_sam_detections(detections)

        self.assertEqual([item[0] for item in selected], [1, 0])

    def test_mask_depth_is_transformed_to_world_geometry(self):
        segmenter = object.__new__(SAM2Segmenter)
        segmenter.config = DetectorConfig(
            device="cpu",
            instance_depth_max=10.0,
            instance_depth_stride=1,
            instance_depth_min_points=4,
            instance_mask_pixel_noise_std=2.0,
            instance_depth_noise_std=0.5,
        )
        mask = np.zeros((10, 10), dtype=bool)
        mask[4:7, 4:7] = True
        depth = np.full((10, 10), 2.0, dtype=np.float32)
        intrinsics = np.asarray([
            [10.0, 0.0, 5.0],
            [0.0, 10.0, 5.0],
            [0.0, 0.0, 1.0],
        ])
        camera_pose_world = np.eye(4)
        camera_pose_world[:3, 3] = [1.0, 2.0, 3.0]

        geometry = segmenter._mask_world_geometry(
            mask,
            depth,
            intrinsics,
            camera_pose_world,
        )

        np.testing.assert_allclose(
            geometry.position_world, [1.0, 2.0, 5.0]
        )
        np.testing.assert_allclose(
            geometry.bbox_3d_min, [0.8, 1.8, 5.0]
        )
        np.testing.assert_allclose(
            geometry.bbox_3d_max, [1.2, 2.2, 5.0]
        )
        self.assertEqual(geometry.measurement_covariance.shape, (3, 3))
        self.assertTrue(np.all(np.linalg.eigvalsh(
            geometry.measurement_covariance
        ) > 0.0))
        np.testing.assert_allclose(
            geometry.measurement_covariance,
            np.diag((0.16, 0.16, 0.36)),
            atol=1e-9,
        )

        moved_mask = np.zeros((10, 10), dtype=bool)
        moved_mask[4:7, 3:6] = True
        moved_camera_pose = camera_pose_world.copy()
        moved_camera_pose[0, 3] += 0.2
        moved_geometry = segmenter._mask_world_geometry(
            moved_mask,
            depth,
            intrinsics,
            moved_camera_pose,
        )
        np.testing.assert_allclose(
            moved_geometry.position_world,
            geometry.position_world,
            atol=1e-9,
        )
        self.assertAlmostEqual(
            moved_geometry.measurement_covariance[0, 2],
            -0.036,
            places=9,
        )

    def test_draw_hides_floor_and_ceiling_without_removing_detections(self):
        detector = object.__new__(TiledVehicleDetector)
        detector.config = SimpleNamespace(
            hidden_box_classes=("floor", "ceiling"),
        )
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        floor = Detection((2, 2, 30, 30), 0.9, "floor")
        car = Detection((5, 5, 25, 25), 0.8, "car")

        hidden_render = detector.draw(image, [floor])
        visible_render = detector.draw(image, [floor, car])

        self.assertTrue(np.array_equal(hidden_render, image))
        self.assertFalse(np.array_equal(visible_render, image))

    def test_fp16_is_only_requested_for_cuda_devices(self):
        self.assertEqual(
            DetectorConfig().instance_duplicate_merge_distance_xy,
            1.5,
        )
        self.assertEqual(
            DetectorConfig().instance_display_min_observations,
            10,
        )
        self.assertEqual(
            DetectorConfig(device=0, use_fp16=True).prediction_quantization,
            16,
        )
        self.assertEqual(
            DetectorConfig(
                device="cuda:0",
                use_fp16=True,
            ).prediction_quantization,
            16,
        )
        self.assertIsNone(
            DetectorConfig(
                device="cpu",
                use_fp16=True,
            ).prediction_quantization
        )
        self.assertIsNone(
            DetectorConfig(device=0, use_fp16=False).prediction_quantization
        )

    def test_world_model_sets_text_prompts_and_does_not_use_coco_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "world.pt"
            weights.touch()
            model = MagicMock()
            model.predict.return_value = [SimpleNamespace(boxes=[])]
            config = DetectorConfig(
                weights=weights,
                class_prompts=("car", "parking barrier"),
                device="cpu",
            )

            with patch(
                "detections.object_detect.YOLOWorld",
                return_value=model,
            ) as model_type:
                detector = TiledVehicleDetector(config)
                detections = detector.detect(
                    np.zeros((405, 720, 3), dtype=np.uint8)
                )

            model_type.assert_called_once_with(str(weights))
            model.set_classes.assert_called_once_with(
                ["car", "parking barrier"]
            )
            self.assertEqual(detections, [])
            predict_options = model.predict.call_args.kwargs
            self.assertNotIn("classes", predict_options)
            self.assertIsNone(predict_options["quantize"])
            self.assertEqual(model.predict.call_count, 1)

    def test_sam2_receives_yolo_world_boxes_as_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "sam2.1_t.pt"
            weights.touch()
            image = np.zeros((32, 48, 3), dtype=np.uint8)
            result = SimpleNamespace(masks=None)
            model = MagicMock()
            model.predict.return_value = [result]
            config = DetectorConfig(
                sam_weights=weights,
                class_prompts=("car",),
                device="cpu",
                save_semantic_map=False,
            )
            detections = [Detection((1, 2, 20, 25), 0.8, "car")]

            with patch(
                "detections.object_detect.SAM",
                return_value=model,
            ) as model_type:
                segmenter = SAM2Segmenter(config)
                rendered, count = segmenter.segment(image, detections)

            model_type.assert_called_once_with(str(weights))
            self.assertEqual(
                model.predict.call_args.kwargs["bboxes"],
                [[1, 2, 20, 25]],
            )
            self.assertIsNone(model.predict.call_args.kwargs["quantize"])
            self.assertEqual(count, 0)
            self.assertEqual(rendered.shape, image.shape)

    def test_sam2_is_not_run_without_yolo_world_boxes(self):
        model = MagicMock()
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        with patch(
            "detections.object_detect.SAM",
            return_value=model,
        ):
            segmenter = SAM2Segmenter(
                DetectorConfig(
                    sam_weights="sam2.1_t.pt",
                    device="cpu",
                    save_semantic_map=False,
                )
            )
        rendered, count = segmenter.segment(image, [])
        model.predict.assert_not_called()
        self.assertEqual(count, 0)
        self.assertEqual(rendered.shape, image.shape)

    def test_sam_budget_includes_cars_even_with_many_high_confidence_columns(self):
        segmenter = object.__new__(SAM2Segmenter)
        segmenter.config = DetectorConfig(max_sam_prompts=4, device="cpu")
        segmenter.mapper = SemanticInstanceMapper()
        detections = [Detection((i, 0, i + 10, 20), 0.99, "column") for i in range(10)]
        detections.extend([
            Detection((30, 0, 50, 20), 0.40, "car"),
            Detection((60, 0, 80, 20), 0.35, "van"),
        ])
        selected = segmenter._select_sam_detections(detections)
        self.assertEqual([d.class_name for _, d in selected], ["column", "car", "column", "van"])

    def test_signs_and_fire_cabinets_receive_sam_budget(self):
        segmenter = object.__new__(SAM2Segmenter)
        segmenter.config = DetectorConfig(max_sam_prompts=4, device="cpu")
        segmenter.mapper = SemanticInstanceMapper()
        detections = [Detection((0, 0, 10, 20), 0.99, "column") for _ in range(12)]
        detections.extend([Detection((0, 0, 10, 20), score, name) for score, name in
                           ((0.9, "car"), (0.2, "wall mounted sign"),
                            (0.15, "fire hose cabinet"))])
        selected = segmenter._select_sam_detections(detections)
        self.assertEqual([d.class_name for _, d in selected],
                         ["column", "car", "wall mounted sign", "fire hose cabinet"])

    def test_vehicle_duplicate_subtype_boxes_are_suppressed(self):
        detector = object.__new__(TiledVehicleDetector)
        detector.config = DetectorConfig(device="cpu")
        boxes = torch.tensor([[0., 0., 40., 40.], [0., 0., 40., 40.],
                              [25., 0., 65., 40.], [0., 0., 40., 40.]])
        keep = detector._nms(boxes, torch.tensor([0.9, 0.8, 0.7, 0.6]),
                             ["car", "van", "car", "column"])
        self.assertEqual(keep.tolist(), [0, 2, 3])

    def test_vehicle_masks_produce_stable_labels_and_saved_geometry(self):
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        image[4:40, 4:28] = (20, 20, 200)
        image[4:40, 40:64] = (200, 20, 20)
        boxes = [(4, 4, 28, 40), (40, 4, 64, 40)]
        model = MagicMock()

        def predict(_image, bboxes, **kwargs):
            masks = torch.zeros((len(bboxes), 64, 96))
            for index, (x1, y1, x2, y2) in enumerate(bboxes):
                masks[index, y1:y2, x1:x2] = 1
            return [SimpleNamespace(masks=SimpleNamespace(data=masks))]

        model.predict.side_effect = predict
        with tempfile.TemporaryDirectory() as directory:
            config = DetectorConfig(
                sam_weights="sam2.1_t.pt", device="cpu",
                instance_depth_stride=1, instance_display_min_observations=2,
                vehicle_confirmation_observations=2,
                semantic_map_min_observations=2,
                semantic_map_path=Path(directory) / "map.json",
            )
            with patch("detections.object_detect.SAM", return_value=model):
                segmenter = SAM2Segmenter(config)
            sensor_data = dict(depth=np.full((64, 96), 4., dtype=np.float32),
                               camera_intrinsics=np.array([[50., 0, 48], [0, 50., 32], [0, 0, 1.]]),
                               camera_pose_world=np.eye(4))
            segmenter.segment(image, [Detection(box, 0.9, "car") for box in boxes],
                              timestamp=1., **sensor_data)
            with patch("detections.object_detect.cv2.putText", wraps=cv2.putText) as text:
                rendered, count = segmenter.segment(
                    image, [Detection(boxes[1], 0.9, "van"), Detection(boxes[0], 0.8, "car")],
                    timestamp=1.1, **sensor_data,
                )
            self.assertEqual(count, 2)
            self.assertFalse(np.array_equal(rendered, image))
            self.assertEqual([call.args[1] for call in text.call_args_list], ["vehicle_01", "vehicle_02"])
            # 无人机平移引起图像左移；同步NED位姿补偿后车辆身份仍相同。
            moved_pose = np.eye(4)
            moved_pose[0, 3] = 0.16  # 2 px * 4 m / fx=50
            moved_boxes = [(x1 - 2, y1, x2 - 2, y2) for x1, y1, x2, y2 in boxes]
            with patch("detections.object_detect.cv2.putText", wraps=cv2.putText) as text:
                segmenter.segment(
                    np.roll(image, -2, axis=1),
                    [Detection(box, 0.9, "car") for box in moved_boxes],
                    timestamp=1.2,
                    **{**sensor_data, "camera_pose_world": moved_pose},
                )
            self.assertEqual([call.args[1] for call in text.call_args_list], ["vehicle_02", "vehicle_01"])
            segmenter.save_semantic_map()
            records = json.loads(config.semantic_map_path.read_text(encoding="utf-8"))["instances"]
            self.assertEqual([r["instance_id"] for r in records], ["vehicle_01", "vehicle_02"])
            self.assertEqual([r["video_instance_id"] for r in records], ["vehicle_01", "vehicle_02"])
            self.assertEqual(len(records[0]["position_world"]), 3)
            self.assertTrue(records[0]["is_obstacle"])
            self.assertEqual([r["color"] for r in records], ["blue", "red"])

    def test_mask_appearance_ignores_background(self):
        image = np.full((48, 64, 3), 20, dtype=np.uint8)
        mask = np.zeros((48, 64), dtype=bool)
        mask[8:40, 16:48] = True
        image[mask] = (10, 20, 200)
        first = SAM2Segmenter._mask_appearance(image, mask)
        image[~mask] = (200, 255, 10)
        np.testing.assert_allclose(SAM2Segmenter._mask_appearance(image, mask), first)

    def test_worker_reads_sensor_frame_rgb_and_publishes_preview(self):
        image = np.full((32, 48, 3), 17, dtype=np.uint8)
        source = _OneFrameSource(image)
        detector = _FakeDetector()
        segmenter = _FakeSegmenter()
        config = DetectorConfig(
            device="cpu",
            inference_fps=100.0,
            save_videos=False,
            save_semantic_map=False,
        )

        with patch(
            "detections.realtime_detection_worker.TiledVehicleDetector",
            return_value=detector,
        ), patch(
            "detections.realtime_detection_worker.SAM2Segmenter",
            return_value=segmenter,
        ):
            worker = RealtimeDetectionWorker(source, config)
        worker.start()
        deadline = time.monotonic() + 1.0
        while worker.processed_frames == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        worker.close()

        worker.raise_if_failed()
        self.assertEqual(worker.processed_frames, 1)
        self.assertTrue(np.array_equal(detector.received, image))
        self.assertTrue(np.array_equal(segmenter.received, image))
        self.assertTrue(np.array_equal(
            segmenter.sensor_data["depth"], source.depth
        ))
        self.assertTrue(np.array_equal(
            segmenter.sensor_data["camera_intrinsics"],
            source.camera_intrinsics,
        ))
        self.assertTrue(np.array_equal(
            segmenter.sensor_data["camera_pose_world"],
            source.camera_pose_world,
        ))
        self.assertEqual(segmenter.sensor_data["timestamp"], 1788400000.0)
        self.assertEqual(
            worker.get_latest_preview(image).shape,
            (image.shape[0] * 2, image.shape[1], image.shape[2]),
        )


if __name__ == "__main__":
    unittest.main()
