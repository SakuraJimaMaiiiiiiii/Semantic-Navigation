"""Keep v3 occupancy/landmarks while restoring current vehicle association."""
from .local_occupancy_grid import LocalOccupancyGrid as V3LocalMap
from .online_loop_navigation import OnlineLoopNavigationMap as V3NavigationMap
from .semantic_instance_mapper import SemanticInstanceMapper as V3Mapper
from .global_sparse_occupancy_map import GlobalSparseOccupancyMap
from .lidar_mapping_worker import LidarMappingWorker
from .lidar_loop_map import rebuild
from ..vehicle_instance_tracker import VehicleInstanceTracker
from ..semantic_instance_mapper import SemanticInstanceMapper as CurrentMapper


class LocalOccupancyGrid(V3LocalMap):
    def __init__(self, config=None):
        super().__init__(config)
        self._execution_snapshot = None

    def integrate_lidar_points(self, origin, endpoints, timestamp, drone_position):
        super().integrate_lidar_points(origin, endpoints, timestamp, drone_position)
        self._execution_snapshot = self.snapshot()

    @property
    def execution_snapshot(self):
        return self._execution_snapshot

    @property
    def forward_clearance(self):
        return self._forward_clearance


class OnlineLoopNavigationMap(V3NavigationMap):
    @property
    def last_observation_completed_at(self):
        return self._state[0].last_observation_completed_at


class SemanticInstanceMapper(V3Mapper):
    def __init__(self, **options):
        super().__init__(**options)
        vehicle_options = {
            "mahalanobis_gate": self.mahalanobis_gate,
            "process_noise_std": self.process_noise_std,
        }
        for source, target in (
            ("vehicle_max_association_distance", "max_distance"),
            ("vehicle_appearance_gate", "appearance_gate"),
            ("vehicle_ambiguity_margin", "ambiguity_margin"),
            ("vehicle_viewpoint_noise_std", "viewpoint_noise_std"),
            ("vehicle_confirmation_observations", "confirmation_observations"),
        ):
            if source in options:
                vehicle_options[target] = options[source]
        self._vehicles = VehicleInstanceTracker(**vehicle_options)

    def semantic_map_records(self, min_observations=1):
        records = super().semantic_map_records(min_observations)
        for record in records:
            if record["runtime_instance_id"].startswith("vehicle_"):
                track = self._vehicles.tracks[int(record["runtime_instance_id"].split("_")[-1])]
                record.update(**track.surface.vehicle_geometry(),
                              connected_fragment_count=track.connected_fragment_count)
        return records

    def object_inspection_records(self, published_records):
        return CurrentMapper.object_inspection_records(self, published_records)
