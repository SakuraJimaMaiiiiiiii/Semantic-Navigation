"""Column identity evidence shared by current and v3 runtime adapters."""
import numpy as np
from scipy.spatial import cKDTree

from .instance_surface import measured_patch_overlap


class ColumnIdentityPolicy:
    def _strict_column_geometry(self, track):
        points = track.surface.supported_points()
        box = track.surface.vehicle_geometry().get('observed_oriented_box')
        if box is None or len(points) < 24:
            return False
        length, width, height = box['size']
        if not (height >= self.column_min_vertical_extent and .30 <= width <= length <= 1.60
                and length <= 2.*width):
            return False
        yaw = box['yaw']
        basis = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        local = (points[:, :2]-np.asarray(box['center'])[:2]) @ basis
        low, high = np.percentile(local, [2.5, 97.5], axis=0)
        supported_axes = set()
        tolerance = min(.12, .18*width)
        for axis in (0, 1):
            for bound in (low[axis], high[axis]):
                patch = points[np.abs(local[:, axis]-bound) <= tolerance]
                tangent = local[np.abs(local[:, axis]-bound) <= tolerance, 1-axis]
                if (len(patch) >= 24 and np.ptp(patch[:, 2]) >= .65*height
                        and np.ptp(tangent) >= max(.30, .6*(high[1-axis]-low[1-axis]))):
                    supported_axes.add(axis)
        return len(supported_axes) == 2

    def _strict_column_records(self, records):
        return [r for r in records if r['class_name'] != 'column' or self._strict_column_geometry(
            self._landmarks['column', int(r['runtime_instance_id'].rsplit('_', 1)[1])])]

    def _pending_column_records(self, published):
        existing = {r['runtime_instance_id'] for r in published}
        result = []
        for (category, identity), track in self._landmarks.items():
            name = f'column_{identity:02d}'
            size = track.bbox_3d_max-track.bbox_3d_min
            if (category != 'column' or name in existing or track.observation_count < 3
                    or track.confidence < .5 or size[2] < self.column_min_vertical_extent
                    or not .4 <= max(size[:2]) <= 1.6):
                continue
            result.append(dict(runtime_instance_id=name, class_name='column',
                position_world=((track.bbox_3d_min+track.bbox_3d_max)*.5).tolist(),
                bbox_3d=dict(min=track.bbox_3d_min.tolist(), max=track.bbox_3d_max.tolist()),
                position_covariance=track.position_covariance.tolist(), last_seen=track.last_seen,
                observation_count=track.observation_count, confidence=track.confidence,
                geometry_status='needs_second_face', merged_from=[]))
        return result

    @staticmethod
    def _column_observed_patch(surface, points):
        # A small visible face need not cover 25% of a model accumulated around
        # all sides. Require a measured vertical patch relative to the smaller
        # cloud, so occlusion/point density does not create another identity.
        return measured_patch_overlap(surface.supported_points(), points, minimum_fraction=.5,
                                      axis_spans=(.15, .15, .5), require_vertical=True)

    @staticmethod
    def _column_boxes_overlap(first, second):
        # Nearby parallel surfaces alone do not prove one physical column.
        low = np.maximum(first.bbox_3d_min, second.bbox_3d_min)
        high = np.minimum(first.bbox_3d_max, second.bbox_3d_max)
        small = np.minimum(first.bbox_3d_max-first.bbox_3d_min,
                           second.bbox_3d_max-second.bbox_3d_min)
        return bool(np.all(high-low+.15 >= .5*small))

    def _column_vertical_parts(self, first, second):
        low = np.maximum(first.bbox_3d_min, second.bbox_3d_min)
        high = np.minimum(first.bbox_3d_max, second.bbox_3d_max)
        small = np.minimum(first.bbox_3d_max-first.bbox_3d_min,
                           second.bbox_3d_max-second.bbox_3d_min)
        union_low = np.minimum(first.bbox_3d_min, second.bbox_3d_min)
        union_high = np.maximum(first.bbox_3d_max, second.bbox_3d_max)
        centers = [(p.bbox_3d_min[:2]+p.bbox_3d_max[:2])*.5 for p in (first, second)]
        return bool(np.linalg.norm(centers[0]-centers[1]) <= .15
                    and np.all(high[:2]-low[:2]+.05 >= .75*small[:2])
                    and -.15 <= high[2]-low[2] <= .25*small[2]
                    and small[2] >= .5
                    and self.column_min_vertical_extent <= union_high[2]-union_low[2]
                    <= 2*self.column_min_vertical_extent)

    @staticmethod
    def _column_continuous_height(observation, first, second):
        points = observation.surface_points_world
        if points is None or len(points) < 24:
            return False
        heights = np.unique(np.asarray(points)[:, 2])
        return bool(heights[0] <= min(first.bbox_3d_min[2], second.bbox_3d_min[2])+.15
                    and heights[-1] >= max(first.bbox_3d_max[2], second.bbox_3d_max[2])-.15
                    and np.max(np.diff(heights), initial=0.) <= .3)

    def _prepare_column_observations(self, names, scores, observations, timestamp):
        observations = list(observations)
        accepted = []
        for index in sorted(range(len(names)), key=lambda i: scores[i], reverse=True):
            obs = observations[index]
            if names[index] != 'column' or obs is None:
                continue
            if any(self._column_boxes_overlap(obs, observations[other]) and measured_patch_overlap(
                    obs.surface_points_world, observations[other].surface_points_world)
                   for other in accepted):
                observations[index] = None
            else:
                accepted.append(index)
        self._column_current = [observations[i] for i in accepted]
        previous = getattr(self, '_column_positive_stamp', None)
        if accepted:
            interval = 0. if previous is None else timestamp-previous
            self._column_positive_stamp = timestamp
            self._column_evidence_window = max(90., min(180., 3.*interval))
        self._column_ready_pairs = set()
        evidence = getattr(self, '_column_merge_evidence', {})
        window = getattr(self, '_column_evidence_window', 90.)
        evidence = {pair: stamps for pair, stamps in evidence.items()
                    if timestamp-stamps[-1] <= window and all(k in self._landmarks for k in pair)}
        keys = [k for k, t in self._landmarks.items() if k[0] == 'column' and t.observation_count >= 2]
        if len(keys) > 1 and self._column_current:
            centers = [(self._landmarks[k].bbox_3d_min[:2]+self._landmarks[k].bbox_3d_max[:2])*.5 for k in keys]
            pairs = cKDTree(centers).query_pairs(self.duplicate_merge_distance_xy)
            for i, j in pairs:
                pair = tuple(sorted((keys[i], keys[j])))
                a, b = (self._landmarks[k] for k in pair)
                # A fresh observation must explain both repeatedly measured
                # surfaces. Stored geometry/missed frames cannot count again.
                fresh = [obs for obs in self._column_current
                         if self._column_boxes_overlap(a, obs) and self._column_boxes_overlap(b, obs)]
                vertical = self._column_vertical_parts(a, b)
                if not fresh or not (self._column_boxes_overlap(a, b) or vertical):
                    continue
                # One real common face can be only a quarter of a pillar model
                # accumulated around four sides. Fresh vertical support below
                # is still required in three different frames.
                shared = a.surface.shares_patch(b.surface, minimum_fraction=.25)
                if not shared and not vertical:
                    continue
                if not any(self._column_observed_patch(a.surface, obs.surface_points_world)
                           and self._column_observed_patch(b.surface, obs.surface_points_world)
                           and (shared or self._column_continuous_height(obs, a, b)) for obs in fresh):
                    continue
                stamps = (*evidence.get(pair, ()), timestamp)[-3:]
                evidence[pair] = stamps
                if len(stamps) >= 3:
                    self._column_ready_pairs.add(frozenset(pair))
        self._column_merge_evidence = evidence
        return observations

    def _column_surface_cost(self, landmark, observation):
        low = np.maximum(landmark.bbox_3d_min, observation.bbox_3d_min)
        high = np.minimum(landmark.bbox_3d_max, observation.bbox_3d_max)
        if np.any(low-high > .15):
            return None
        if not self._column_observed_patch(landmark.surface, observation.surface_points_world):
            return None
        # Do not use different visible heights/face centroids as identity gates.
        # World-frame surface overlap supplies the association, without ICP.
        return min(self._innovation_distance_squared(landmark, observation, self._last_timestamp), 1.)

    def _column_inspection_records(self, records):
        """Spatial hypotheses request a view; they never authorize a merge."""
        columns = [dict(r, connection_targets=[], identity_status='observed')
                   for r in records if r['class_name'] == 'column']
        columns.sort(key=lambda r: int(r['runtime_instance_id'].rsplit('_', 1)[1]))
        for index, parent in enumerate(columns):
            a = self._landmarks.get(('column', int(parent['runtime_instance_id'].rsplit('_', 1)[1])))
            if a is None:
                continue
            for fragment in columns[index+1:]:
                b = self._landmarks.get(('column', int(fragment['runtime_instance_id'].rsplit('_', 1)[1])))
                if b is None or not (self._column_boxes_overlap(a, b) or self._column_vertical_parts(a, b)):
                    continue
                centers = [(t.bbox_3d_min[:2]+t.bbox_3d_max[:2])*.5 for t in (a, b)]
                if np.linalg.norm(centers[0]-centers[1]) > self.duplicate_merge_distance_xy:
                    continue
                parent['connection_targets'].append(dict(fragment['bbox_3d'],
                    last_seen=fragment['last_seen'], target_id=fragment['runtime_instance_id']))
                parent['identity_status'] = fragment['identity_status'] = 'needs_connection_view'
        return columns

    def _with_column_identity_status(self, records):
        records = self._strict_column_records(records)
        columns = {r['runtime_instance_id']: r for r in self._column_inspection_records(records)}
        return [columns.get(r['runtime_instance_id'], r) for r in records]

