"""Measured surface evidence keyed by stable instance ID, independent of occupancy.

This layer never changes free/occupied states and never rasterizes bounding boxes.
Only exact world-grid intersections with occupied voxels receive exported labels.
"""

from collections import OrderedDict
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SemanticVoxelSnapshot:
    resolution: float
    indices: np.ndarray
    instance_ids: np.ndarray
    support: np.ndarray
    confidence: np.ndarray


class SemanticVoxelEvidence:
    def __init__(self, resolution, max_voxels):
        if not np.isfinite(resolution) or resolution <= 0 or max_voxels < 1:
            raise ValueError("Semantic evidence requires positive resolution and capacity")
        self.resolution = float(resolution)
        self.max_voxels = int(max_voxels)
        self._cells = OrderedDict()
        self._last_timestamp = None

    def update(self, observations, timestamp):
        """One vote per instance/voxel/frame, regardless of point density."""
        if not np.isfinite(timestamp):
            raise ValueError("Semantic evidence timestamp must be finite")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            return
        self._last_timestamp = timestamp
        votes = set()
        for identity, points in observations:
            if points is None:
                continue
            points = np.asarray(points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("Semantic surface points must have shape (N, 3)")
            points = points[np.isfinite(points).all(axis=1)]
            cells = np.unique(np.floor(points / self.resolution).astype(np.int64), axis=0)
            votes.update((tuple(cell), identity) for cell in cells)
        for cell, identity in sorted(votes):
            counts = self._cells.setdefault(cell, {})
            counts[identity] = counts.get(identity, 0) + 1
            self._cells.move_to_end(cell)
        while len(self._cells) > self.max_voxels:
            self._cells.popitem(last=False)

    def snapshot(self, records):
        """Cumulative evidence; consumers may skip perception frames safely.

        Merge aliases conservatively using max support, since the two tracks may
        have observed a voxel in the same frame. Unconfirmed identities cannot win.
        Ambiguous votes remain in the denominator rather than becoming free votes.
        """
        records = {r["runtime_instance_id"]: r for r in records}
        aliases = {old: identity for identity, r in records.items()
                   for old in r.get("merged_from", [])}
        cells, identities, support, confidence = [], [], [], []
        for cell, counts in self._cells.items():
            for old in counts.keys() & aliases.keys():
                count = counts.pop(old)
                identity = aliases[old]
                counts[identity] = max(counts.get(identity, 0), count)
            winner = max(counts, key=counts.get)
            count, total = counts[winner], sum(counts.values())
            if winner in records and count >= 2 and count * 2 > total:
                cells.append(cell)
                identities.append(winner)
                support.append(count)
                confidence.append(count / total)
        return SemanticVoxelSnapshot(
            self.resolution, np.asarray(cells, dtype=np.int64).reshape(-1, 3),
            np.asarray(identities, dtype=str), np.asarray(support, dtype=np.int32),
            np.asarray(confidence, dtype=np.float32),
        )


def project_semantic_voxels(snapshot, evidence, records):
    """Label rows of the existing global map. Code 0 means unclassified, not free."""
    records = {r["runtime_instance_id"]: r for r in records}
    identities = [""] + sorted(records)
    codes = {identity: i for i, identity in enumerate(identities) if i}
    aliases = {old: identity for identity, r in records.items()
               for old in r.get("merged_from", [])}
    result = {
        "semantic_labels": np.zeros(len(snapshot.indices), dtype=np.int32),
        "semantic_support": np.zeros(len(snapshot.indices), dtype=np.int32),
        "semantic_confidence": np.zeros(len(snapshot.indices), dtype=np.float32),
        "semantic_instance_ids": np.asarray(identities, dtype=str),
        "semantic_class_names": np.asarray([""] + [records[i]["class_name"] for i in identities[1:]], dtype=str),
    }
    if evidence is None:
        return result
    if not np.isclose(evidence.resolution, snapshot.resolution, rtol=0, atol=1e-9):
        raise ValueError("Semantic evidence and occupancy voxel resolution must match")
    if not len(evidence.indices):
        return result
    lookup = {tuple(cell): (codes.get(aliases.get(identity, identity), 0), count, confidence)
              for cell, identity, count, confidence in zip(
                  evidence.indices, evidence.instance_ids, evidence.support, evidence.confidence)}
    for row in np.flatnonzero(snapshot.log_odds >= snapshot.occupied_threshold):
        label, count, confidence = lookup.get(tuple(snapshot.indices[row]), (0, 0, 0.))
        if label:
            result["semantic_labels"][row] = label
            result["semantic_support"][row] = count
            result["semantic_confidence"][row] = confidence
    return result


def semantic_height_mask(snapshot, labels, altitude, half_height):
    from planner.vertical_projection import height_band_indices
    low, high = height_band_indices(altitude, half_height, snapshot.resolution)
    return ((labels > 0) & (snapshot.indices[:, 2] >= low)
            & (snapshot.indices[:, 2] <= high))
