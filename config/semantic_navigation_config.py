"""Semantic task budgets and approach/search geometry (metres, seconds, NED)."""

from dataclasses import dataclass, fields
from pathlib import Path
import math

DEFAULT_TARGET_FILE = Path(__file__).resolve().parent / "semantic_target.json"


@dataclass(frozen=True)
class SemanticNavigationConfig:
    mission_timeout: float = 300.0
    perception_timeout: float = 5.0
    map_timeout: float = 3.0
    verification_timeout: float = 8.0
    verification_frames: int = 3
    standoff_distance: float = 1.2
    arrival_tolerance: float = 0.4
    clearance_radius: float = 0.35
    vertical_half_extent: float = 0.25
    approach_samples: int = 24
    leg_length: float = 1.2
    leg_timeout: float = 30.0
    max_search_steps: int = 30
    max_target_attempts: int = 3
    max_search_radius: float = 30.0
    max_plan_nodes: int = 100000
    scan_wait: float = 1.0
    retry_cooldown: float = 15.0

    def __post_init__(self):
        integers = {
            "verification_frames",
            "approach_samples",
            "max_search_steps",
            "max_target_attempts",
            "max_plan_nodes",
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be finite and positive")
            if field.name in integers and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValueError(f"{field.name} must be an integer")
        if self.standoff_distance <= self.clearance_radius + self.arrival_tolerance:
            raise ValueError(
                "standoff_distance must exceed clearance plus arrival tolerance"
            )
