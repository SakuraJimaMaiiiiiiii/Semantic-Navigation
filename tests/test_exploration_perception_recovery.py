"""Slow semantic inference pauses exploration without bypassing map health."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from config.exploration_config import FrontierExplorationConfig
from config.semantic_navigation_config import SemanticNavigationConfig
from navigation.dfs_explorer import DFSExplorer
from navigation.perception import SemanticFrame


def explorer(tmp_path, recover_after=None, frozen_map=False):
    clock = SimpleNamespace(now=100.0)
    vehicle = MagicMock()
    vehicle.get_position.return_value = np.zeros(3)
    vehicle.wait.side_effect = lambda seconds: setattr(clock, "now", clock.now + seconds)
    perception = MagicMock()
    perception.semantic_snapshot.side_effect = lambda: SemanticFrame(
        clock.now, clock.now if recover_after is not None and clock.now >= 100 + recover_after else 90.0,
        6.0, (),
    )
    maps = SimpleNamespace(snapshot=lambda: SimpleNamespace(timestamp=100 if frozen_map else clock.now))
    nav = DFSExplorer(
        vehicle, perception, maps, maps, SimpleNamespace(config=None), None,
        SemanticNavigationConfig(), tmp_path / "result.json",
        clock=lambda: clock.now,
        exploration_config=FrontierExplorationConfig(perception_recovery_timeout=8),
        memory=MagicMock(),
    )
    nav.started, nav.deadline = 100.0, 200.0
    nav._motion_progress = np.zeros(3), 95.0
    return nav, clock


def test_slow_inference_hovers_then_resumes_and_resets_motion_watchdog(tmp_path):
    nav, clock = explorer(tmp_path, recover_after=2)
    nav._check_health()
    nav.vehicle.hover.assert_called_once()
    nav.vehicle.move_velocity_guided.assert_not_called()
    nav.memory.observe.assert_called_once()
    assert nav._motion_progress[1] == clock.now
    assert [e["state"] for e in nav.events] == ["PERCEPTION_WAIT", "PERCEPTION_RECOVERED"]


def test_dead_perception_has_bounded_wait_and_never_records_stale_frame(tmp_path):
    nav, clock = explorer(tmp_path)
    with pytest.raises(RuntimeError, match="semantic_perception_stale"):
        nav._frame()
    assert clock.now == pytest.approx(108)
    nav.memory.observe.assert_not_called()
    assert nav.events[-1]["state"] == "PERCEPTION_RECOVERY_TIMEOUT"


def test_map_staleness_still_interrupts_perception_wait(tmp_path):
    nav, clock = explorer(tmp_path, frozen_map=True)
    with pytest.raises(RuntimeError, match="semantic_local_map_stale"):
        nav._frame()
    assert clock.now < 108
    nav.memory.observe.assert_not_called()


def test_worker_failure_is_not_treated_as_slow_inference(tmp_path):
    nav, _ = explorer(tmp_path)
    nav.perception.raise_if_failed.side_effect = RuntimeError("detector_failed")
    with pytest.raises(RuntimeError, match="detector_failed"):
        nav._frame()
    nav.vehicle.wait.assert_not_called()
