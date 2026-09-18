"""静止车辆身份关联回归：相邻车、遮挡、观测偏移和类别抖动。"""

import itertools
import math
from dataclasses import replace
import unittest

import numpy as np

from mapping.semantic_instance_mapper import SemanticInstanceMapper, SemanticObservation3D
from mapping.vehicle_instance_tracker import _minimum_cost_assignment


def observation(position, appearance=None):
    position = np.asarray(position, dtype=float)
    return SemanticObservation3D(
        position_world=position,
        bbox_3d_min=position - [0.8, 0.4, 0.5],
        bbox_3d_max=position + [0.8, 0.4, 0.5],
        measurement_covariance=np.eye(3) * 0.09,
        appearance=None if appearance is None else np.asarray(appearance, dtype=float),
    )


def update(mapper, observations, timestamp, names=None):
    return mapper.update(
        names or ["car"] * len(observations),
        [0.9] * len(observations), observations, timestamp,
    )


class VehicleInstanceTrackingTest(unittest.TestCase):
    def test_pending_requires_three_distinct_frames_and_exports_only_confirmed(self):
        mapper = SemanticInstanceMapper()
        obs = observation([0, 0, -1])
        self.assertEqual(update(mapper, [obs], 1), [None])
        self.assertEqual(update(mapper, [obs], 1), [None])
        self.assertEqual(update(mapper, [obs], 1.1), [None])
        self.assertEqual(mapper.semantic_map_records(), [])
        self.assertEqual(update(mapper, [obs], 1.2), [1])
        self.assertEqual(mapper.snapshot()[0].observation_count, 3)

    def test_expired_false_positive_does_not_consume_formal_id(self):
        mapper = SemanticInstanceMapper()
        update(mapper, [observation([0, 0, -1])], 1)
        update(mapper, [], 4)
        self.assertEqual(mapper._vehicles._pending, [])
        for stamp in (5, 5.1):
            self.assertEqual(update(mapper, [observation([5, 0, -1])], stamp), [None])
        self.assertEqual(update(mapper, [observation([5, 0, -1])], 5.2), [1])

    def test_gallery_retains_older_view_and_color_is_multiframe(self):
        mapper = SemanticInstanceMapper()
        obs = replace(observation([0, 0, -1], [1, 0]), color_scores={"red": 0.9, "black": 0.1})
        for stamp in (1, 1.1, 1.2):
            update(mapper, [obs], stamp)
        record = mapper.semantic_map_records()[0]
        self.assertEqual(record["color"], "red")
        self.assertAlmostEqual(record["color_confidence"], 0.9)
        changed = replace(obs, appearance=np.array([0., 1.]), color_scores={"blue": 1.0})
        self.assertEqual(update(mapper, [changed], 1.3), [1])
        self.assertEqual(mapper.vehicle_color(1)[0], "red")
        self.assertEqual(len(mapper._vehicles.tracks[1].appearance_gallery), 2)
        self.assertEqual(update(mapper, [obs], 1.4), [1])
        self.assertEqual(len(mapper.snapshot()), 1)

    def test_missing_or_conflicting_colors_remain_unknown(self):
        mapper = SemanticInstanceMapper()
        for stamp in (1, 1.1, 1.2):
            update(mapper, [observation([0, 0, -1])], stamp)
        self.assertEqual(mapper.vehicle_color(1), ("unknown", 0.0))
        for i in range(6):
            obs = replace(observation([0, 0, -1]), color_scores={"red" if i % 2 else "blue": 1.0})
            update(mapper, [obs], 2 + i * 0.1)
        self.assertEqual(mapper.vehicle_color(1), ("unknown", 0.0))

    def test_converged_filter_accepts_visible_surface_shift(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        for frame in range(100):
            obs = observation([0, 0, -1], [1, 0])
            obs.measurement_covariance[:] = np.eye(3) * 0.0001
            self.assertEqual(update(mapper, [obs], frame * 0.1), [1])
        shifted = observation([0.6, 0.0, -0.8], [1, 0])
        shifted.measurement_covariance[:] = np.eye(3) * 0.0001
        self.assertEqual(update(mapper, [shifted], 10), [1])
        self.assertEqual(len(mapper.snapshot()), 1)

    def test_unique_close_geometry_recovers_after_color_change(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        update(mapper, [observation([0, 0, -1], [1, 0])], 1)
        self.assertEqual(update(mapper, [observation([0.1, 0, -1], [0, 1])], 2), [1])
        self.assertEqual(len(mapper.snapshot()), 1)

    def test_color_rejection_near_existing_vehicle_does_not_spawn_id(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        update(mapper, [observation([0, 0, -1], [1, 0])], 1)
        for timestamp in (2, 3, 4):
            self.assertEqual(update(mapper, [observation([0.9, 0, -1], [0, 1])],
                                    timestamp), [None])
        self.assertEqual(len(mapper.snapshot()), 1)
        self.assertEqual(mapper.snapshot()[0].observation_count, 1)
        self.assertEqual(update(mapper, [observation([5, 0, -1], [0, 1])], 5), [2])

    def test_color_recovery_is_disabled_when_neighbor_also_plausible(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        update(mapper, [observation([0, 0, -1], [1, 0]),
                        observation([1, 0, -1], [1, 0])], 1)
        self.assertEqual(update(mapper, [observation([0.1, 0, -1], [0, 1])], 2), [None])
        self.assertEqual([m.observation_count for m in mapper.snapshot()], [1, 1])

    def test_close_parked_cars_are_not_merged_and_reordering_keeps_ids(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        self.assertEqual(update(mapper, [observation([0, 0, -1]),
                                         observation([1, 0, -1])], 1), [1, 2])
        self.assertEqual(update(mapper, [observation([1.02, 0, -1]),
                                         observation([0.02, 0, -1])], 1.1), [2, 1])
        self.assertEqual(len(mapper.snapshot()), 2)

    def test_appearance_keeps_ids_through_viewpoint_and_class_flicker(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        for frame in range(13):
            x = -1.5 + 0.15 * math.sin(frame)
            items = [observation([x, 0, -1], [1, 0]),
                     observation([-x, 0, -1], [0, 1])]
            names = ["van" if frame % 2 else "car", "truck"]
            expected = [1, 2]
            if frame % 2:
                items.reverse()
                names.reverse()
                expected.reverse()
            self.assertEqual(update(mapper, items, frame * 0.5, names), expected)
        self.assertEqual([m.label for m in mapper.snapshot()], ["vehicle_01", "vehicle_02"])

    def test_viewpoint_drift_does_not_extrapolate_vehicle_after_occlusion(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        for frame in range(5):
            self.assertEqual(update(mapper, [observation([frame * 0.1, 0, -1])],
                                    frame * 0.5), [1])
        track = mapper._vehicles.tracks[1]
        saved_position = track.state.copy()
        saved_covariance = track.covariance.copy()
        self.assertEqual(track.state.shape, (3,))
        self.assertEqual(track.covariance.shape, (3, 3))
        for timestamp in (3.5, 120.0):
            predicted, covariance = mapper._vehicles._predict(track, timestamp)
            np.testing.assert_array_equal(predicted, saved_position)
            elapsed = min(timestamp - track.last_seen, 5.0)
            np.testing.assert_allclose(
                covariance, saved_covariance + np.eye(3) * 0.05**2 * elapsed
            )
        np.testing.assert_array_equal(track.covariance, saved_covariance)
        self.assertEqual(update(mapper, [], 2.5), [])
        self.assertEqual(update(mapper, [observation([0, 0, -1])], 120), [1])
        memory = mapper.snapshot()[0]
        self.assertLess(memory.bbox_3d_max[0] - memory.bbox_3d_min[0], 1.7)

    def test_identical_parked_cars_with_ambiguous_view_do_not_swap_ids(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        update(mapper, [observation([-0.5, 0, -1]), observation([0.5, 0, -1])], 1)
        self.assertEqual(update(mapper, [observation([0, 0, -1]),
                                        observation([0, 0, -1])], 2), [None, None])
        self.assertEqual(update(mapper, [observation([0.5, 0, -1]),
                                        observation([-0.5, 0, -1])], 3), [2, 1])
        self.assertEqual(len(mapper.snapshot()), 2)

    def test_ambiguous_match_does_not_create_or_update_id(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        update(mapper, [observation([-0.5, 0, -1]), observation([0.5, 0, -1])], 1)
        self.assertEqual(update(mapper, [observation([0, 0, -1])], 1.1), [None])
        self.assertEqual([m.observation_count for m in mapper.snapshot()], [1, 1])

    def test_two_measurements_cannot_claim_one_vehicle(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        update(mapper, [observation([0, 0, -1])], 1)
        self.assertEqual(update(mapper, [observation([0, 0, -1]),
                                         observation([0, 0, -1])], 1.1), [None, None])
        self.assertEqual(len(mapper.snapshot()), 1)

    def test_parked_vehicle_survives_long_absence(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        update(mapper, [observation([2, 2, -1], [1, 0])], 1)
        update(mapper, [], 20)
        self.assertEqual(update(mapper, [observation([2.1, 2, -1], [1, 0])], 120), [1])

    def test_map_and_video_do_not_renumber_vehicle_after_filtering(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        update(mapper, [observation([0, 0, -1]), observation([5, 0, -1])], 1)
        for timestamp in (2, 3):
            self.assertEqual(update(mapper, [observation([5, 0, -1])], timestamp,
                                    ["van"]), [2])
        self.assertEqual(mapper.confirmed_instance_id("van", 2, 3), 2)
        record = mapper.semantic_map_records(min_observations=3)[0]
        self.assertEqual(record["instance_id"], "vehicle_02")
        self.assertEqual(record["runtime_instance_id"], "vehicle_02")
        self.assertEqual(record["video_instance_id"], "vehicle_02")
        self.assertTrue(record["is_obstacle"])
        self.assertTrue(record["is_static"])
        self.assertEqual(record["class_name"], "car")

    def test_missing_geometry_and_repeated_frames_do_not_confirm_id(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        self.assertEqual(update(mapper, [None], 1), [None])
        self.assertEqual(update(mapper, [observation([0, 0, -1])], 2), [1])
        for timestamp in (2, 1.5):
            self.assertEqual(update(mapper, [observation([0, 0, -1])], timestamp), [None])
        self.assertEqual(mapper.snapshot()[0].observation_count, 1)

    def test_all_vehicle_classes_share_ids_but_do_not_receive_column_geometry_filter(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        names = ["car", "truck", "bus", "van"]
        observations = [observation([i * 5, 0, -1]) for i in range(4)]
        self.assertEqual(update(mapper, observations, 1, names), [1, 2, 3, 4])
        self.assertEqual([m.label for m in mapper.snapshot()],
                         ["vehicle_01", "vehicle_02", "vehicle_03", "vehicle_04"])
        self.assertEqual(len(mapper.semantic_map_records()), 4)

    def test_large_measurement_uncertainty_does_not_take_a_far_vehicle_id(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        update(mapper, [observation([0, 0, -1])], 1)
        far = observation([20, 0, -1])
        far.measurement_covariance[:] = np.eye(3) * 1000
        self.assertEqual(update(mapper, [far], 2), [2])

    def test_unseen_parked_vehicle_id_is_not_recycled_for_distant_vehicle(self):
        mapper = SemanticInstanceMapper(vehicle_confirmation_observations=1)
        for frame in range(5):
            self.assertEqual(update(mapper, [observation([0, 0, -1])],
                                    frame * 0.5), [1])
        self.assertEqual(update(mapper, [observation([20, 0, -1])], 20), [2])
        self.assertEqual([m.label for m in mapper.snapshot()], ["vehicle_01", "vehicle_02"])

    def test_assignment_finds_global_optimum_including_unmatched_columns(self):
        random = np.random.default_rng(42)
        for rows in range(1, 5):
            for _ in range(5):
                costs = random.uniform(0, 10, (rows, rows + 2))
                columns = _minimum_cost_assignment(costs)
                actual = sum(costs[row, col] for row, col in enumerate(columns))
                expected = min(sum(costs[row, col] for row, col in enumerate(cols))
                               for cols in itertools.permutations(range(rows + 2), rows))
                self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
