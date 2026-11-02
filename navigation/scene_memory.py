"""Coordinate-free semantic graph, with a separate binary navigation cache."""

from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .atomic_file import atomic_write_text
from mapping.semantic_voxels import project_semantic_voxels, semantic_height_mask


def _numpy_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path, value):
    atomic_write_text(path, json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False, default=_numpy_json
    ))


class SceneMemory:
    def __init__(self, session_id, home, initial_yaw, config):
        self.session_id = session_id
        self.home = np.asarray(home, dtype=float)
        self.initial_yaw = float(initial_yaw)
        self.config = config
        self.records = {}
        self.places = {}
        self.observed_at = defaultdict(set)
        self.coobserved = Counter()
        self._last_frame = None
        self.semantic_voxels = None
        self.exploration = {"state": "exploring", "returned_home": False}

    def add_place(self, identity, position, parent=None):
        self.places[identity] = {
            "position": np.asarray(position).tolist(),
            "parent": parent,
        }

    def observe(self, frame, place_id=None):
        if self._last_frame is not None and frame.timestamp <= self._last_frame:
            return
        self._last_frame = frame.timestamp
        self.records = {r["runtime_instance_id"]: deepcopy(r) for r in frame.records}
        self.semantic_voxels = deepcopy(getattr(frame, "semantic_voxels", None))
        redirects = {old: identity for identity, record in self.records.items()
                     for old in record.get("merged_from", [])}
        for old, identity in redirects.items():
            self.observed_at[identity].update(self.observed_at.pop(old, set()))
        if redirects:
            merged = Counter()
            for (first, second), count in self.coobserved.items():
                first, second = redirects.get(first, first), redirects.get(second, second)
                if first != second:
                    merged[tuple(sorted((first, second)))] += count
            self.coobserved = merged
        fresh = sorted(
            r["runtime_instance_id"]
            for r in frame.records
            if abs(r["last_seen"] - frame.timestamp) < 1e-6
        )
        if place_id is not None:
            for identity in fresh:
                self.observed_at[identity].add(place_id)
        self.coobserved.update(combinations(fresh, 2))

    def graph(self):
        objects = []
        relations = []
        for identity, record in sorted(self.records.items()):
            objects.append(
                {
                    "id": identity,
                    "class_name": record["class_name"],
                    "aliases": record.get("aliases", []),
                    "merged_from": record.get("merged_from", []),
                    "classification_status": record.get("classification_status", "single_category"),
                    "category_evidence": record.get("category_evidence", {}),
                    "position_status": record.get("position_status", "unassessed"),
                    "confidence": record["confidence"],
                    "observation_count": record["observation_count"],
                    "color": record.get("color", "unknown"),
                    "color_confidence": record.get("color_confidence", 0.0),
                    "observed_at": sorted(self.observed_at[identity]),
                }
            )
        c, s = math.cos(self.initial_yaw), math.sin(self.initial_yaw)
        for first, second in combinations(sorted(self.records), 2):
            a, b = self.records[first], self.records[second]
            delta = np.asarray(a["position_world"]) - b["position_world"]
            distance = float(np.linalg.norm(delta))
            if distance <= self.config.relation_radius:
                forward, right = (
                    c * delta[0] + s * delta[1],
                    -s * delta[0] + c * delta[1],
                )
                if distance <= self.config.near_distance:
                    relations.append(
                        {
                            "source": first,
                            "relation": "near",
                            "target": second,
                            "symmetric": True,
                            "distance_m": round(distance, 2),
                        }
                    )
                if max(abs(forward), abs(right)) >= 0.3:
                    direction = (
                        ("in_front_of" if forward > 0 else "behind")
                        if abs(forward) >= abs(right)
                        else ("right_of" if right > 0 else "left_of")
                    )
                    relations.append(
                        {
                            "source": first,
                            "relation": direction,
                            "target": second,
                            "basis": "initial_heading",
                            "evidence": "estimated_geometry",
                        }
                    )
                if abs(delta[2]) >= 1.0:
                    relations.append(
                        {
                            "source": first,
                            "relation": "above" if delta[2] < 0 else "below",
                            "target": second,
                        }
                    )
            count = self.coobserved[(first, second)]
            if count:
                relations.append(
                    {
                        "source": first,
                        "relation": "co_observed",
                        "target": second,
                        "symmetric": True,
                        "frame_count": count,
                    }
                )
        nodes = [
            {"id": obj["id"], "type": "object",
             "attributes": {key: value for key, value in obj.items()
                            if key not in {"id", "observed_at"}}}
            for obj in objects
        ]
        nodes.extend({"id": identity, "type": "place", "attributes": {}}
                     for identity in sorted(self.places))
        relations.extend(
            {"source": node["parent"], "relation": "traversed_to", "target": identity}
            for identity, node in sorted(self.places.items())
            if node["parent"] in self.places
        )
        relations.extend(
            {"source": obj["id"], "relation": "observed_at", "target": place}
            for obj in objects for place in obj["observed_at"] if place in self.places
        )
        edges = [
            {"id": f"{edge['source']}:{edge['relation']}:{edge['target']}", **edge}
            for edge in relations
        ]
        return {
            "version": 2,
            "map_id": self.session_id,
            "representation": "semantic_topological_graph",
            "directed": True,
            "direction_reference": "Forward is the initial heading; left/right do not change with later drone turns.",
            "exploration": deepcopy(self.exploration),
            "nodes": nodes,
            "edges": edges,
        }

    def checkpoint(self, directory):
        write_json(Path(directory) / "semantic_graph.partial.json", self.graph())

    def number_final_instances(self):
        """Finalize public IDs after recording stops; retain source provenance."""
        from mapping.semantic_numbering import number_exported_instances
        records, self.semantic_voxels, mapping = number_exported_instances(
            self.records.values(), self.semantic_voxels)
        self.records = {record['runtime_instance_id']: record for record in records}
        self.observed_at = defaultdict(set, {
            mapping.get(identity, identity): places for identity, places in self.observed_at.items()})
        self.coobserved = Counter({
            tuple(sorted(mapping.get(identity, identity) for identity in pair)): count
            for pair, count in self.coobserved.items()})
        return mapping

    def save_bundle(self, directory, snapshot, navigation_config=None):
        """Commit matching semantics and binary geometry; manifest is written last."""
        directory = Path(directory)
        private = directory / "private"
        private.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.session_id,
            "records": list(self.records.values()),
            "places": self.places,
        }
        temporary = private / "navigation_cache.tmp"
        semantic_labels = project_semantic_voxels(snapshot, self.semantic_voxels, self.records.values())
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                payload=np.frombuffer(
                    json.dumps(payload, allow_nan=False, default=_numpy_json).encode(),
                    dtype=np.uint8,
                ),
                home=self.home,
                initial_yaw=self.initial_yaw,
                indices=snapshot.indices,
                log_odds=snapshot.log_odds,
                resolution=snapshot.resolution,
                occupied_threshold=snapshot.occupied_threshold,
                free_threshold=snapshot.free_threshold,
                timestamp=snapshot.timestamp,
                **semantic_labels,
            )
        destination = private / "navigation_cache.npz"
        temporary.replace(destination)
        write_json(directory / "semantic_graph.json", self.graph())
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        self._save_semantic_map(directory, snapshot, digest, navigation_config, semantic_labels)
        write_json(
            directory / "bundle.json",
            {
                "version": 1,
                "session_id": self.session_id,
                "navigation_cache": "private/navigation_cache.npz",
                "sha256": digest,
                "semantic_graph": "semantic_graph.json",
                "semantic_map": "semantic_map.json",
                "returned_home": bool(self.exploration.get("returned_home")),
            },
        )

    def _save_semantic_map(self, directory, snapshot, geometry_digest, navigation_config, semantic_labels):
        """Export semantic locations and a height-specific sparse 2-D map.

        Occupancy remains sensor geometry. Object bounding boxes are annotations,
        never filled as obstacles or treated as precise per-voxel class labels.
        """
        from types import SimpleNamespace
        from config.semantic_navigation_config import SemanticNavigationConfig
        from .route import NavigationGrid

        config = navigation_config or SemanticNavigationConfig()
        altitude = float(getattr(self, "flight_altitude", self.home[2]-1.5))
        raw = NavigationGrid.from_global(snapshot, altitude, SimpleNamespace(vertical_half_extent=config.vertical_half_extent, clearance_radius=0.))
        safe = NavigationGrid(raw.free, raw.occupied, raw.resolution, raw.origin, altitude, config)
        cells = sorted(raw.free) + sorted(raw.occupied)
        indices = np.asarray(cells, dtype=np.int32).reshape(-1, 2)
        states = np.concatenate((np.zeros(len(raw.free), dtype=np.int8),
                                 np.ones(len(raw.occupied), dtype=np.int8)))
        cell_rows = {cell: i for i, cell in enumerate(cells)}
        in_band = semantic_height_mask(snapshot, semantic_labels["semantic_labels"],
                                       altitude, config.vertical_half_extent)
        # Several objects at different Z may share one XY cell: retain all IDs.
        relations = sorted({(cell_rows[tuple(cell[:2])], int(label))
                            for cell, label in zip(snapshot.indices[in_band],
                                                  semantic_labels["semantic_labels"][in_band])})
        relations = np.asarray(relations, dtype=np.int32).reshape(-1, 2)
        path = directory / "occupancy_2d.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, indices=indices, states=states,
                                safe_indices=np.asarray(sorted(safe.free), dtype=np.int32).reshape(-1, 2),
                                semantic_cell_rows=relations[:, 0],
                                semantic_instance_labels=relations[:, 1],
                                semantic_instance_ids=semantic_labels["semantic_instance_ids"],
                                semantic_class_names=semantic_labels["semantic_class_names"],
                                resolution=snapshot.resolution, altitude_ned=altitude,
                                vertical_half_extent=config.vertical_half_extent,
                                clearance_radius=config.clearance_radius, timestamp=snapshot.timestamp)
        temporary.replace(path)
        write_json(directory / "semantic_map.json", {
            "version": 2, "session_id": self.session_id,
            "coordinate_frame": "world_NED", "units": "metres",
            "axes": ["north", "east", "down"], "home_ned": self.home,
            "initial_yaw_rad": self.initial_yaw, "timestamp": snapshot.timestamp,
            "returned_home": bool(self.exploration.get("returned_home")),
            "exploration": {"reason": self.exploration.get("reason"),
                            "frontiers_exhausted": self.exploration.get("frontiers_exhausted", False),
                            "coverage_guaranteed": False},
            "occupancy_3d": {"file": "private/navigation_cache.npz", "sha256": geometry_digest,
                             "index_key": "indices", "value_key": "log_odds",
                             "resolution_m": snapshot.resolution,
                             "free_threshold": snapshot.free_threshold,
                             "occupied_threshold": snapshot.occupied_threshold,
                             "voxel_center_formula": "(index + 0.5) * resolution_m",
                             "missing_or_between_thresholds": "unknown"},
            "semantic_voxels": {
                "file": "private/navigation_cache.npz", "label_key": "semantic_labels",
                "row_alignment": "occupancy_3d.indices", "unclassified_label": 0,
                "instance_ids_key": "semantic_instance_ids", "class_names_key": "semantic_class_names",
                "support_key": "semantic_support", "confidence_key": "semantic_confidence",
                "evidence": "masked measured surfaces; at least two frames and strict majority; exact occupied voxel intersection only",
                "confidence_meaning": "winning instance vote fraction, not calibrated geometric accuracy",
            },
            "occupancy_2d": {"file": path.name,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "index_key": "indices", "value_key": "states",
                             "axes": ["north", "east"],
                             "resolution_m": snapshot.resolution,
                             "cell_center_formula": "(index + 0.5) * resolution_m",
                             "altitude_ned": altitude, "states": {"0": "free", "1": "occupied"},
                             "missing_cell": "unknown", "safe_cells_key": "safe_indices",
                             "semantic_relation_keys": ["semantic_cell_rows", "semantic_instance_labels"],
                             "semantic_relation_meaning": "row of indices -> instance label; multiple instances allowed; only flight height band",
                             "instance_ids_key": "semantic_instance_ids", "class_names_key": "semantic_class_names"},
            "object_id_key": "runtime_instance_id",
            "object_geometry": "position_world and bbox_3d use world_NED; position_world is an estimated anchor from accumulated visible surfaces, not a guaranteed object center; bbox_3d bounds observed geometry, not guaranteed full object extent; annotations are not voxel class labels",
            "objects": list(self.records.values()),
            "relations_file": "semantic_graph.json",
        })


def load_bundle(directory, expected_session_id):
    """Read geometry only for its still-running control session; never pickle."""
    directory = Path(directory)
    manifest = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest["session_id"] != expected_session_id:
        raise ValueError(
            "Map belongs to another control session; relocalization is required"
        )
    if manifest.get("coordinate_frame", "NED") != "NED" or manifest.get("navigation_enabled") is False:
        raise ValueError("Corrected map requires PX4-to-map relocalization before navigation")
    if not manifest.get("returned_home"):
        raise ValueError("Exploration has not confirmed return and landing")
    path = directory / "private" / "navigation_cache.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Navigation cache checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        payload = json.loads(data["payload"].tobytes().decode())
        if payload["session_id"] != expected_session_id:
            raise ValueError("Navigation cache session mismatch")
        payload.update(home=data["home"].copy(), initial_yaw=float(data["initial_yaw"]))
        payload["occupancy_map"] = {
            "indices": data["indices"].copy(), "log_odds": data["log_odds"].copy(),
            "resolution": float(data["resolution"]),
            "occupied_threshold": float(data["occupied_threshold"]),
            "free_threshold": float(data["free_threshold"]),
            "timestamp": float(data["timestamp"]),
        }
        # Old bundles remain readable and do not acquire fabricated labels.
        if "semantic_labels" in data:
            payload["semantic_voxels"] = {key: data[key].copy() for key in (
                "semantic_labels", "semantic_support", "semantic_confidence",
                "semantic_instance_ids", "semantic_class_names")}
    return payload
