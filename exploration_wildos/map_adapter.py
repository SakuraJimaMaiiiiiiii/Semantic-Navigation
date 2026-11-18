"""Read existing occupancy archives without loading a control session or pickle."""
from dataclasses import asdict, dataclass
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from config.semantic_navigation_config import SemanticNavigationConfig
from navigation.route import NavigationGrid


def _scalar(data, key):
    value = np.asarray(data[key])
    if value.size != 1 or not np.isfinite(value).all():
        raise ValueError(f'{key} must be one finite number')
    return float(value.reshape(-1)[0])


def _indices(value, dimensions):
    value = np.asarray(value)
    if (value.ndim != 2 or value.shape[1] != dimensions
            or not np.isfinite(value).all() or not np.equal(value, np.floor(value)).all()
            or np.any(np.abs(value) > 2**40)):
        raise ValueError(f'indices must be finite integer N x {dimensions} coordinates')
    result = value.astype(np.int64, copy=True)
    if len(np.unique(result, axis=0)) != len(result):
        raise ValueError('Duplicate voxel/cell indices would corrupt height-band evidence')
    return result


@dataclass(frozen=True)
class MapAdapter:
    grid: NavigationGrid
    metadata: dict
    raw_free: frozenset

    @classmethod
    def from_grid(cls, grid, metadata=None):
        """Also usable with synthetic fixtures; preserve raw versus eroded free cells."""
        return cls(grid, dict(metadata or {}), frozenset(grid.known - grid.occupied))

    def state(self, point):
        """Projected height-band state, not a 3-D voxel query."""
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('point must be finite NED xyz')
        if abs(point[2] - self.grid.altitude) > 1e-8:
            raise ValueError('This adapter only represents its selected altitude')
        cell = self.grid.cell(point)
        return 'OCCUPIED' if cell in self.grid.occupied else 'FREE' if cell in self.raw_free else 'UNKNOWN'


def load_map(path, *, altitude=None, clearance=None, vertical_half_extent=None):
    """Load global sparse NPZ/navigation_cache or exported occupancy_2d NPZ.

    This is offline analysis, not load_bundle: no landing/session authorization
    is inferred. For global maps the caller must choose the analysis altitude.
    """
    path = Path(path)
    archive_bytes = path.read_bytes()
    with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as data:
        resolution = _scalar(data, 'resolution')
        timestamp = _scalar(data, 'timestamp')
        if resolution <= 0:
            raise ValueError('resolution must be positive')
        projected = 'states' in data and 'log_odds' not in data
        if projected:
            saved_altitude = _scalar(data, 'altitude_ned')
            saved_vertical = _scalar(data, 'vertical_half_extent')
            if altitude is not None and not np.isclose(altitude, saved_altitude, rtol=0, atol=1e-8):
                raise ValueError('A projected map cannot be moved to a different altitude')
            if vertical_half_extent is not None and not np.isclose(vertical_half_extent, saved_vertical, rtol=0, atol=1e-8):
                raise ValueError('A projected map cannot be assigned a different height band')
            altitude = saved_altitude
            vertical_half_extent = saved_vertical
            clearance = _scalar(data, 'clearance_radius') if clearance is None else clearance
        else:
            if altitude is None:
                raise ValueError('Global 3-D archive requires explicit --altitude (NED metres)')
            clearance = 0.40 if clearance is None else clearance
            vertical_half_extent = 0.25 if vertical_half_extent is None else vertical_half_extent
        if not np.isfinite(altitude):
            raise ValueError('altitude must be finite')
        config = SemanticNavigationConfig(clearance_radius=float(clearance),
                                          vertical_half_extent=float(vertical_half_extent))
        metadata = dict(source=str(path.resolve()), source_sha256=hashlib.sha256(archive_bytes).hexdigest(),
                        source_kind='exported_2d' if projected else 'global_sparse_3d',
                        frame='NED', frame_evidence='existing project archive convention; not relocalized',
                        offline_only=True, timestamp=timestamp, snapshot_id=None, loop_revision=None,
                        resolution=resolution, altitude_ned=float(altitude),
                        navigation_config=asdict(config), local_snapshot_fused=False)
        metadata['snapshot_id'] = metadata['source_sha256']
        if 'dataset_label' in data:
            label = np.asarray(data['dataset_label'])
            if label.size != 1 or label.dtype.kind not in 'US':
                raise ValueError('dataset_label must be a scalar string')
            metadata['dataset_label'] = str(label.reshape(-1)[0])
        if projected:
            indices = _indices(data['indices'], 2)
            states = np.asarray(data['states'])
            if states.shape != (len(indices),) or not np.isin(states, [-1, 0, 1]).all():
                raise ValueError('states must have one UNKNOWN/FREE/OCCUPIED value per cell')
            free = set(map(tuple, indices[states == 0].tolist()))
            occupied = set(map(tuple, indices[states == 1].tolist()))
            # Recompute erosion from raw states: never erode saved safe_indices twice.
            grid = NavigationGrid(free, occupied, resolution, (0, 0), altitude, config)
        else:
            indices = _indices(data['indices'], 3)
            # Keep the archive dtype: NavigationGrid compares thresholds in that
            # dtype. Promoting float32(0.7) to float64 changes boundary membership.
            odds = np.asarray(data['log_odds'])
            if odds.dtype.kind not in 'fiu' or odds.shape != (len(indices),) or not np.isfinite(odds).all():
                raise ValueError('log_odds must be a finite value per voxel')
            free_threshold = _scalar(data, 'free_threshold')
            occupied_threshold = _scalar(data, 'occupied_threshold')
            if free_threshold >= occupied_threshold:
                raise ValueError('free_threshold must be below occupied_threshold')
            snapshot = SimpleNamespace(indices=indices, log_odds=odds.copy(), resolution=resolution,
                                       free_threshold=free_threshold, occupied_threshold=occupied_threshold,
                                       timestamp=timestamp)
            grid = NavigationGrid.from_global(snapshot, float(altitude), config)
            metadata.update(free_threshold=free_threshold, occupied_threshold=occupied_threshold)
    return MapAdapter.from_grid(grid, metadata)
