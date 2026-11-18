"""Geometry and archive boundary tests for the independent offline graph."""
import json

import numpy as np
import pytest

from config.semantic_navigation_config import SemanticNavigationConfig
from exploration_wildos.graph_export import export_graph
from exploration_wildos.map_adapter import MapAdapter, load_map
from exploration_wildos.static_graph import DistanceField, GraphConfig, build_graph, validate_graph
from navigation.route import NavigationGrid


def scene(free, occupied=(), resolution=.25):
    config = SemanticNavigationConfig(clearance_radius=.40, vertical_half_extent=.25)
    return MapAdapter.from_grid(NavigationGrid(set(free), set(occupied), resolution, (0, 0), -1.5, config),
                                {'source': 'synthetic test fixture', 'snapshot_id': 'test'})


def save_2d(path, indices=None, states=None, **overrides):
    indices = np.array([(x, y) for x in range(-8, 9) for y in range(-8, 9)]) if indices is None else np.asarray(indices)
    payload = dict(indices=indices, states=np.zeros(len(indices), dtype=np.int8) if states is None else states,
                   resolution=.25, timestamp=1., altitude_ned=-1.5, vertical_half_extent=.25, clearance_radius=.4)
    payload.update(overrides)
    np.savez(path, **payload)


def save_3d(path, **overrides):
    # At z=-1.5 and half extent .25, the project conservatively includes z indices -7..-5.
    from planner.vertical_projection import height_band_indices
    low, high = height_band_indices(-1.5, .25, .25)
    indices = np.array([(x, y, z) for x in range(-8, 9) for y in range(-8, 9) for z in range(low, high+1)])
    payload = dict(indices=indices, log_odds=np.full(len(indices), -2.), resolution=.25, timestamp=1.,
                   free_threshold=-.5, occupied_threshold=.5)
    payload.update(overrides)
    np.savez(path, **payload)
    return payload


def test_projected_read_is_read_only_and_does_not_double_erode(tmp_path):
    path = tmp_path/'map.npz'
    save_2d(path, safe_indices=np.empty((0, 2), dtype=int))
    before = path.read_bytes()
    adapter = load_map(path)
    assert path.read_bytes() == before
    assert len(adapter.grid.free) > 0  # Deliberately incorrect saved safe list was not used.
    assert adapter.grid.free == scene(adapter.raw_free).grid.free
    p = adapter.grid.point((-3, -2))
    assert adapter.grid.cell(p) == (-3, -2)
    assert adapter.state(p) == 'FREE'
    assert adapter.state([100., 100., -1.5]) == 'UNKNOWN'
    with pytest.raises(ValueError, match='selected altitude'):
        adapter.state([0., 0., 2.])


def test_3d_unknown_height_and_obstacle_veto(tmp_path):
    path = tmp_path/'map.npz'
    payload = save_3d(path)
    indices, odds = payload['indices'], payload['log_odds']
    # Missing one layer, ambiguous evidence in another column, and one occupied hit.
    keep = np.ones(len(indices), dtype=bool)
    keep[np.flatnonzero(np.all(indices[:, :2] == [0, 0], axis=1))[0]] = False
    odds[np.flatnonzero(np.all(indices[:, :2] == [3, 0], axis=1))[0]] = 0.
    odds[np.flatnonzero(np.all(indices[:, :2] == [-3, 0], axis=1))[0]] = 2.
    payload.update(indices=indices[keep], log_odds=odds[keep])
    np.savez(path, **payload)
    with pytest.raises(ValueError, match='explicit --altitude'):
        load_map(path)
    adapter = load_map(path, altitude=-1.5)
    assert adapter.state(adapter.grid.point((0, 0))) == 'UNKNOWN'
    assert adapter.state(adapter.grid.point((3, 0))) == 'UNKNOWN'
    assert adapter.state(adapter.grid.point((-3, 0))) == 'OCCUPIED'
    assert (0, 1) not in adapter.grid.free  # Safety erosion includes unknown volume.


@pytest.mark.parametrize('override', [dict(altitude=-2.), dict(vertical_half_extent=.5)])
def test_projected_map_cannot_claim_other_height(tmp_path, override):
    path = tmp_path/'map.npz'
    save_2d(path)
    with pytest.raises(ValueError, match='projected map'):
        load_map(path, **override)


@pytest.mark.parametrize('bad', ['duplicate', 'fractional', 'nan', 'threshold_order'])
def test_invalid_archive_rejected(tmp_path, bad):
    path = tmp_path/'map.npz'
    payload = save_3d(path)
    if bad == 'duplicate':
        payload['indices'][1] = payload['indices'][0]
    elif bad == 'fractional':
        payload['indices'] = payload['indices'].astype(float)
        payload['indices'][0, 0] += .1
    elif bad == 'nan':
        payload['log_odds'][0] = np.nan
    else:
        payload['free_threshold'] = 2.
    np.savez(path, **payload)
    with pytest.raises(ValueError):
        load_map(path, altitude=-1.5)


def test_distance_boundaries_and_no_obstacles():
    adapter = scene([(x, y) for x in range(-5, 6) for y in range(-5, 6)])
    field = DistanceField(adapter, 1000)
    obstacle, unknown = field.at((5, 0))
    assert np.isinf(obstacle)
    assert 0 < unknown <= .125  # Distance to the neighboring unknown cell boundary.
    assert field.at((0, 0))[1] > unknown
    assert field.at((500, 0)) == (0., 0.)


def test_large_sparse_bounding_box_is_refused():
    adapter = scene([(0, 0), (100000, 100000)])
    with pytest.raises(ValueError, match='max_dense_cells'):
        DistanceField(adapter, 1000)


@pytest.mark.parametrize('barrier', ['occupied', 'unknown'])
def test_wall_or_unknown_strip_splits_graph(barrier):
    area = {(x, y) for x in range(-16, 17) for y in range(-10, 11)}
    strip = {c for c in area if c[0] == 0}
    adapter = scene(area-strip, strip if barrier == 'occupied' else ())
    graph = build_graph(adapter, GraphConfig(free_radius_cap=.6, edge_radius=2.))
    assert validate_graph(adapter, graph)['passed']
    assert len(graph['components']) == 2
    # Independent analytic check: no exported chord crosses x=0 wall/unknown column.
    for edge in graph['edges']:
        a, b = (graph['nodes'][edge[k]]['position_ned'] for k in ('source', 'target'))
        assert a[0]*b[0] > 0
    assert adapter.grid.path([-2., 0., -1.5], [2., 0., -1.5]) is None


def test_open_door_connects_rooms_and_robot():
    area = {(x, y) for x in range(-16, 17) for y in range(-12, 13)}
    wall = {c for c in area if c[0] == 0 and abs(c[1]) > 4}
    adapter = scene(area-wall, wall)
    graph = build_graph(adapter, GraphConfig(free_radius_cap=.5, edge_radius=1.5), [-2., 0., -1.5])
    assert len(graph['components']) == 1
    assert graph['robot_attachment']['status'] == 'attached'
    assert adapter.grid.path([-2., 0., -1.5], [2., 0., -1.5]) is not None
    assert validate_graph(adapter, graph)['passed']
    for edge in graph['edges']:
        a, b = (np.array(graph['nodes'][edge[k]]['position_ned']) for k in ('source', 'target'))
        if a[0] < 0 < b[0] or b[0] < 0 < a[0]:
            y = (a + (-a[0]/(b[0]-a[0]))*(b-a))[1]
            assert -1. <= y <= 1.25  # Crossing occurs inside the actual door opening.


def test_deterministic_and_unsafe_start_not_attached():
    adapter = scene([(x, y) for x in range(-10, 11) for y in range(-10, 11)])
    options = GraphConfig(max_samples=10, coverage_samples=10)
    first = build_graph(adapter, options, [100., 100., -1.5])
    second = build_graph(adapter, options, [100., 100., -1.5])
    assert first['nodes'] == second['nodes'] and first['edges'] == second['edges']
    assert first['robot_attachment']['status'] == 'unsafe_start'
    assert build_graph(adapter, options, [0., 0., -2.])['robot_attachment']['status'] == 'wrong_altitude'
    assert first['diagnostics']['untested_safe_cells'] > 0


def test_empty_map_export_and_no_overwrite(tmp_path):
    adapter = scene([])
    graph = build_graph(adapter)
    assert graph['nodes'] == [] and graph['components'] == []
    result = validate_graph(adapter, graph)
    export_graph(tmp_path/'output', adapter, graph, result)
    loaded = json.loads((tmp_path/'output/graph.json').read_text())
    assert loaded['nodes'] == []
    assert '<svg' in (tmp_path/'output/graph.html').read_text()
    with pytest.raises(FileExistsError):
        export_graph(tmp_path/'output', adapter, graph, result)


def test_validator_catches_injected_wall_crossing():
    area = {(x, y) for x in range(-12, 13) for y in range(-10, 11)}
    wall = {c for c in area if c[0] == 0}
    adapter = scene(area-wall, wall)
    graph = build_graph(adapter)
    left = next(n['id'] for n in graph['nodes'] if n['position_ned'][0] < 0)
    right = next(n['id'] for n in graph['nodes'] if n['position_ned'][0] > 0)
    graph['edges'].append(dict(id=9999, source=left, target=right))
    assert validate_graph(adapter, graph)['invalid_edges'] == [9999]


def test_cli_export_roundtrip_and_refuse_existing_output(tmp_path):
    from exploration_wildos.build_snapshot_graph import main
    source = tmp_path/'map.npz'
    save_3d(source, dataset_label='SYNTHETIC <script>bad()</script>')
    target = tmp_path/'result'
    args = [str(source), '--altitude', '-1.5', '--output', str(target)]
    assert main(args) == 0
    graph = json.loads((target/'graph.json').read_text())
    assert graph['nodes'] and graph['edges']
    assert graph['metadata']['offline_only']
    assert graph['metadata']['loop_revision'] is None  # Not invented from archive timestamps.
    html = (target/'graph.html').read_text()
    assert '&lt;script&gt;bad()&lt;/script&gt;' in html
    before = (target/'graph.json').read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert (target/'graph.json').read_bytes() == before


def test_graph_export_preserves_geometry(tmp_path):
    adapter = scene([(x, y) for x in range(-10, 11) for y in range(-10, 11)])
    graph = build_graph(adapter)
    export_graph(tmp_path/'graph', adapter, graph, validate_graph(adapter, graph))
    restored = json.loads((tmp_path/'graph/graph.json').read_text())
    assert restored['nodes'] == graph['nodes']
    assert restored['edges'] == graph['edges']
    assert validate_graph(adapter, restored)['passed']


def test_float32_threshold_boundary_matches_existing_projection(tmp_path):
    from types import SimpleNamespace
    path = tmp_path/'threshold.npz'
    payload = save_3d(path)
    odds = payload['log_odds'].astype(np.float32)
    column = np.all(payload['indices'][:, :2] == [0, 0], axis=1)
    odds[column] = np.float32(.7)
    payload.update(log_odds=odds, occupied_threshold=.7)
    np.savez(path, **payload)
    adapter = load_map(path, altitude=-1.5)
    expected = NavigationGrid.from_global(SimpleNamespace(**payload), -1.5, adapter.grid.config)
    assert (0, 0) in expected.occupied
    assert adapter.grid.occupied == expected.occupied
    assert adapter.grid.free == expected.free


def test_reduction_preserves_bridge_and_rejects_long_detour():
    from types import SimpleNamespace
    from exploration_wildos.node_reduction import reduce_nodes
    grid = SimpleNamespace(free={(0, 0)}, resolution=1.)
    nodes = [dict(id=i, cell=[0, 0], position_ned=[0., 0., -1.5], free_radius=1.) for i in range(4)]
    def edge(a, b, length=1.):
        return dict(source=a, target=b, length=length)
    for edges in [[edge(0, 1), edge(1, 2)],
                  [edge(0, 1), edge(1, 2), edge(0, 3, 10.), edge(3, 2, 10.)]]:
        kept, _, result = reduce_nodes(nodes, edges, grid, protected=[0, 2, 3])
        assert len(kept) == 4
        assert result['removed_sample_ids'] == []


def test_reduction_removes_redundant_node_but_keeps_coverage_and_protected_anchor():
    from types import SimpleNamespace
    from exploration_wildos.node_reduction import reduce_nodes
    grid = SimpleNamespace(free={(0, 0), (3, 0)}, resolution=1.)
    nodes = [dict(id=i, cell=[0, 0], position_ned=[0., 0., -1.5], free_radius=1.) for i in range(3)]
    nodes.append(dict(id=3, cell=[3, 0], position_ned=[3., 0., -1.5], free_radius=.3))
    edges = [dict(source=a, target=b, length=1.) for a,b in [(0,1),(1,2),(0,2),(2,3)]]
    kept, kept_edges, result = reduce_nodes(nodes, edges, grid, protected=[0,2])
    assert result['removed_sample_ids'] == [1]
    assert result['covered_cells_before'] == result['covered_cells_after'] == 2
    assert [n['sample_id'] for n in kept] == [0,2,3]
    assert all(e['source'] < len(kept) and e['target'] < len(kept) for e in kept_edges)


def test_reduction_keeps_frontier_nodes():
    from types import SimpleNamespace
    from exploration_wildos.node_reduction import reduce_nodes
    grid = SimpleNamespace(free={(0, 0)}, resolution=1.)
    nodes = [dict(id=i, cell=[0,0], free_radius=1., frontier_ids=['frontier']) for i in range(2)]
    kept, _, result = reduce_nodes(nodes, [dict(source=0,target=1,length=1.)], grid)
    assert len(kept) == 2 and not result['removed_sample_ids']


def test_build_reduction_preserves_coverage_and_start_connection():
    from exploration_wildos.node_reduction import covered_cells
    adapter = scene([(x,y) for x in range(-12,13) for y in range(-12,13)])
    start = [0., 0., -1.5]
    before = build_graph(adapter, GraphConfig(prune_redundant=False), start)
    after = build_graph(adapter, GraphConfig(prune_redundant=True), start)
    assert len(after['nodes']) <= len(before['nodes'])
    assert set().union(*(covered_cells(adapter.grid,n) for n in before['nodes'])) == set().union(*(covered_cells(adapter.grid,n) for n in after['nodes']))
    assert len(after['components']) == len(before['components'])
    assert after['robot_attachment']['status'] == 'attached'
    anchor = after['nodes'][after['robot_attachment']['node_id']]
    assert anchor['sample_id'] == before['robot_attachment']['node_id']
    assert validate_graph(adapter,after)['passed']
