"""Explicit target interface; IDs are stable runtime IDs, never export row numbers."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class SemanticTargetRequest:
    class_name: str | None = None
    instance_id: str | None = None
    color: str | None = None
    near_instance_id: str | None = None
    near_distance: float = 3.0
    min_confidence: float = 0.25
    min_observations: int = 3
    max_age: float = 120.0

    def __post_init__(self):
        for name in ("class_name", "instance_id", "color", "near_instance_id"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be a nonempty string")
                object.__setattr__(self, name, value.strip().lower())
        if not self.class_name and not self.instance_id:
            raise ValueError(
                "Specify class_name or instance_id; no implicit target is chosen"
            )
        for name in ("near_distance", "max_age"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        if (
            isinstance(self.min_observations, bool)
            or not isinstance(self.min_observations, int)
            or self.min_observations < 1
        ):
            raise ValueError("min_observations must be a positive integer")


def load_target_request(path):
    """Read JSON with optional standalone // comment lines (not inline comments)."""
    text = Path(path).read_text(encoding="utf-8")
    # Keep line numbers for JSON errors; never strip // inside a JSON string.
    text = "\n".join(
        "" if line.lstrip().startswith("//") else line for line in text.splitlines()
    )
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != {"target"}:
        raise ValueError('Expected {"target": null or a target object}')
    return (
        None
        if payload["target"] is None
        else SemanticTargetRequest(**payload["target"])
    )


def select_targets(records, request, current_ned, frame_timestamp, *, check_age=True):
    """Filter confirmed map records and rank by distance, confidence, stable ID."""
    import numpy as np

    by_id = {r["runtime_instance_id"]: r for r in records}
    anchor = by_id.get(request.near_instance_id) if request.near_instance_id else None
    if request.near_instance_id and (
        anchor is None
        or (
            check_age
            and not 0 <= frame_timestamp - anchor["last_seen"] <= request.max_age
        )
    ):
        return []
    result = []
    for record in records:
        age = frame_timestamp - record["last_seen"]
        if check_age and not 0 <= age <= request.max_age:
            continue
        if (
            record["confidence"] < request.min_confidence
            or record["observation_count"] < request.min_observations
        ):
            continue
        if request.instance_id and record["runtime_instance_id"] != request.instance_id:
            continue
        names = {record["class_name"], *record.get("aliases", [])}
        if record["class_name"] in ("car", "truck", "bus", "van"):
            names.add("vehicle")
        if request.class_name and request.class_name not in names:
            continue
        if request.color and (
            record.get("color") != request.color
            or record.get("color_confidence", 0) < 0.6
        ):
            continue
        if (
            anchor is not None
            and np.linalg.norm(
                np.array(record["position_world"]) - anchor["position_world"]
            )
            > request.near_distance
        ):
            continue
        result.append(record)
    return sorted(
        result,
        key=lambda r: (
            float(np.linalg.norm(np.array(r["position_world"]) - current_ned)),
            -r["confidence"],
            r["runtime_instance_id"],
        ),
    )
