"""Prefer new corridors without forbidding necessary graph backtracking."""
import math
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from test_wildos_live_explore import grid, manager, explorer, FrontierExplorer
from exploration_wildos.frontier_memory import FrontierMemory


def fork():
    m, g = manager(), grid()
    nodes = [dict(id=i, stable_id=f'nav:{i}', position_ned=p, frontier_ids=f)
             for i, p, f in [(0, [0., 0., -1.5], []),
                             (1, [-4., 0., -1.5], [0, 1, 2]),
                             (2, [4., 0., -1.5], [3, 4, 5])]]
    m.graph = dict(nodes=nodes, edges=[dict(source=0, target=i, length=4.) for i in (1, 2)],
                   frontiers=[dict(id=i, cell=[-7 if i < 3 else 7, i % 3]) for i in range(6)],
                   components=[[0, 1, 2]], diagnostics={'node_budget_reached': False})
    return m, g


def flew_east(m):
    for x in np.arange(-4., .01, .5):
        m.travel_memory.record_position([x, 0., -1.5])


def test_continues_into_new_corridor_even_when_camera_faces_backwards():
    m, g = fork()
    flew_east(m)
    choices, _, _ = m.update(g, np.array([0., 0., -1.5]), np.zeros(3), math.pi, set())
    assert choices[0][0][0] == 2
    assert m.scores[choices[0][0]]['repeated_distance_m'] < m.scores[choices[1][0]]['repeated_distance_m']
    assert m.scores[choices[0][0]]['turn_cost_m'] == 0.


def test_remaining_branch_still_allows_required_backtracking():
    m, g = fork()
    flew_east(m)
    current = np.array([0., 0., -1.5])
    choices, _, _ = m.update(g, current, np.zeros(3), 0., set())
    m.record_visit(m.candidates[choices[0][0]])
    remaining, _, _ = m.update(g, current, np.zeros(3), 0., set())
    assert [key[0] for key, _ in remaining] == [1]
    assert m.route(g, current, remaining[0][1], 'nav:1') is not None
    assert m.regions  # Penalized/observed boundaries never mean completed coverage.


def test_node_replacement_cannot_reoffer_same_view_but_new_unknown_is_kept():
    m, g = fork()
    current = np.array([0., 0., -1.5])
    choices, _, _ = m.update(g, current, np.zeros(3), 0., set())
    key = next(key for key, _ in choices if key[0] == 2)
    m.record_visit(m.candidates[key])
    m.graph['nodes'][2]['stable_id'] = 'nav:20'
    choices, _, _ = m.update(g, current, np.zeros(3), 0., set())
    assert all(key[0] != 20 for key, _ in choices)
    m.graph['frontiers'].append(dict(id=6, cell=[7, 3]))
    m.graph['nodes'][2]['frontier_ids'].append(6)
    choices, _, _ = m.update(g, current, np.zeros(3), 0., set())
    key = next(key for key, _ in choices if key[0] == 20)
    assert m.scores[key]['novel_frontier_cells'] == 1


def test_spatial_visits_keep_distinct_angles_and_distant_views():
    memory = FrontierMemory(3.)
    cells = frozenset({(8, 0), (8, 1)})
    memory.record_visit(SimpleNamespace(point=np.zeros(3), yaw=0., visible_unknown=cells))
    assert not memory.novel_cells([.5, .5, 0.], .1, cells)
    assert memory.novel_cells([0., 0., 0.], math.pi, cells) == cells
    assert memory.novel_cells([4., 0., 0.], 0., cells) == cells


def loop_route(detour_height):
    m, g = fork()
    positions = [[0., 0., -1.5], [2., 0., -1.5], [4., 0., -1.5], [2., detour_height, -1.5]]
    m.graph['nodes'] = [dict(id=i, stable_id=f'nav:{i}', position_ned=p, frontier_ids=[])
                        for i, p in enumerate(positions)]
    m.graph['components'] = [[0, 1, 2, 3]]
    m.graph['edges'] = [dict(source=a, target=b, length=float(np.linalg.norm(np.array(positions[a])-positions[b])))
                        for a, b in [(0, 1), (1, 2), (0, 3), (3, 2)]]
    for x in np.arange(0., 4.01, .5):
        m.travel_memory.record_position([x, 0., -1.5])
    before = dict(m.travel_memory.traversed)
    result = m.routes(g, np.array(positions[0]))['nav:2']
    assert dict(m.travel_memory.traversed) == before  # Planning adds no flown history.
    assert all(g.segment_free(a, b) for a, b in zip(result[1], result[1][1:]))
    return result


def test_routes_can_take_untraveled_loop_and_report_physical_length():
    result = loop_route(2.)
    assert result[2] == ['nav:0', 'nav:3', 'nav:2']
    assert math.isclose(result[0], 2*math.sqrt(8))


def test_large_loop_does_not_replace_short_necessary_retrace():
    result = loop_route(4.)
    assert result[2] == ['nav:0', 'nav:1', 'nav:2']
    assert result[0] == 4.


def test_sparse_telemetry_does_not_mark_unflown_straight_chord():
    memory = FrontierMemory(3.)
    memory.record_position([0., 0., 0.])
    memory.record_position([10., 0., 0.])
    assert memory.cell([5., 0., 0.]) not in memory.traversed


def test_new_coordinate_epoch_resets_route_and_view_history():
    m, g = manager(), grid()
    current = np.array([0., 0., -1.5])
    m.refresh(g, current, frame_key='old')
    flew_east(m)
    m.travel_memory.record_visit(SimpleNamespace(point=current, yaw=0., visible_unknown=frozenset({(7, 0)})))
    previous = m.travel_memory
    m.refresh(g, current, frame_key='new')
    assert m.travel_memory is not previous
    assert not m.travel_memory.traversed and not m.travel_memory.observations
    assert previous.traversed  # Refreshing a staged copy must not erase live history.


def test_live_health_samples_record_partial_motion_and_reject_stale_frame(tmp_path):
    e = explorer(tmp_path)
    # Restore the real graph health wrapper; the offline fixture replaces it.
    from exploration_wildos.explore import GraphExplorer
    e.frontiers.epoch = ('test:PX4-raw:0', -1.5, 1., (0., 0.))
    e.frontiers.travel_memory.record_position([0., 0., -1.5])
    e.vehicle.get_position.return_value = np.array([1., 0., -1.5])
    with patch.object(FrontierExplorer, '_check_health'):
        GraphExplorer._check_health(e)
        assert e.frontiers.travel_memory.cell([.5, 0.]) in e.frontiers.travel_memory.traversed
        assert e.frontiers.travel_memory.cell([5., 0.]) not in e.frontiers.travel_memory.traversed
        e.global_map.revision = 1
        e.vehicle.get_position.return_value = np.array([5., 0., -1.5])
        GraphExplorer._check_health(e)
        assert e.frontiers.travel_memory.cell([5., 0.]) not in e.frontiers.travel_memory.traversed
