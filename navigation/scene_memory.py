"""Coordinate-free semantic graph, with a separate binary navigation cache."""

from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def _numpy_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            value, ensure_ascii=False, indent=2, allow_nan=False, default=_numpy_json
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


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
        self.exploration = {"state": "exploring", "returned_home": False}

    def add_place(self, identity, position, parent=None):
        self.places[identity] = {
            "position": np.asarray(position).tolist(),
            "parent": parent,
            "scanned": False,
        }

    def observe(self, frame, place_id=None):
        if self._last_frame is not None and frame.timestamp <= self._last_frame:
            return
        self._last_frame = frame.timestamp
        self.records = {r["runtime_instance_id"]: deepcopy(r) for r in frame.records}
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
        places = [
            {"id": identity, "scan_completed": node["scanned"]}
            for identity, node in self.places.items()
        ]
        connections = [
            {"source": node["parent"], "relation": "traversed_to", "target": identity}
            for identity, node in self.places.items()
            if node["parent"] is not None
        ]
        return {
            "version": 1,
            "map_id": self.session_id,
            "representation": "semantic_topological_graph",
            "direction_reference": "Forward is the initial heading; left/right do not change with later drone turns.",
            "exploration": deepcopy(self.exploration),
            "objects": objects,
            "relations": relations,
            "places": places,
            "connections": connections,
        }

    def checkpoint(self, directory):
        write_json(Path(directory) / "semantic_graph.partial.json", self.graph())

    def save_bundle(self, directory, snapshot):
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
            )
        destination = private / "navigation_cache.npz"
        temporary.replace(destination)
        write_json(directory / "semantic_graph.json", self.graph())
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        write_json(
            directory / "bundle.json",
            {
                "version": 1,
                "session_id": self.session_id,
                "navigation_cache": "private/navigation_cache.npz",
                "sha256": digest,
                "semantic_graph": "semantic_graph.json",
                "returned_home": bool(self.exploration.get("returned_home")),
            },
        )


def load_bundle(directory, expected_session_id):
    """Read geometry only for its still-running control session; never pickle."""
    directory = Path(directory)
    manifest = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest["session_id"] != expected_session_id:
        raise ValueError(
            "Map belongs to another control session; relocalization is required"
        )
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
    return payload
