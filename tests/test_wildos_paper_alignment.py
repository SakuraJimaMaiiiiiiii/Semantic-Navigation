from dataclasses import replace
import numpy as np

from exploration_wildos.static_graph import GraphConfig, DistanceField, validate_graph
from exploration_wildos.incremental_graph import update_graph, _frontiers
from exploration_wildos.map_adapter import MapAdapter
from navigation.route import NavigationGrid
from config.semantic_navigation_config import SemanticNavigationConfig


def fixture():
    grid=NavigationGrid({(x,y) for x in range(-10,11) for y in range(-10,11)},set(),1.,(0,0),-1.5,
                        SemanticNavigationConfig(clearance_radius=.4))
    return MapAdapter.from_grid(grid,{'snapshot_id':'fixture','timestamp':1})


def test_continuous_sampling_previous_batch_only_and_clearance():
    a=fixture();options=GraphConfig(paper_aligned=True,max_samples=100,coverage_samples=1000,free_radius_cap=4,edge_radius=8,prune_redundant=False)
    g,s=update_graph(a,frame_key='test',options=options)
    assert g['diagnostics']['sampled_cells']==100
    assert validate_graph(a,g)['passed']
    assert all(min(DistanceField(a,10000).at_point(n['position_ned']))>.5 for n in g['nodes'])
    assert any(not np.allclose(n['position_ned'][:2],a.grid.point(n['cell'])[:2]) for n in g['nodes'])
    points=np.asarray([n['position_ned'] for n in g['nodes']])
    assert any(np.linalg.norm(points[i]-points[j])<g['nodes'][i]['free_radius'] for i in range(len(points)) for j in range(i))
    a2=MapAdapter.from_grid(a.grid,{'snapshot_id':'next','timestamp':2})
    next_graph,_=update_graph(a2,frame_key='test',options=options,state=s)
    old_ids={n['stable_id'] for n in g['nodes']}
    for n in next_graph['nodes']:
        if n['stable_id'] not in old_ids:
            assert all(np.linalg.norm(np.asarray(n['position_ned'])-p['position_ned'])>p['free_radius'] for p in g['nodes'])


def test_paper_frontier_keeps_owner_and_unknown_in_bounds_without_boundary():
    a=fixture()
    # Unknown hole is not adjacent to raw free after removing its neighbors.
    hole=(0,0);grid=a.grid
    grid.known-= {(x,y) for x in range(-2,3) for y in range(-2,3)}
    grid.free-= {(x,y) for x in range(-2,3) for y in range(-2,3)}
    a=MapAdapter.from_grid(grid)
    nodes=[dict(id=0,stable_id='old',position_ned=[-6.5,.5,-1.5],explored_radius=0),
           dict(id=1,stable_id='nearer',position_ned=[3.5,.5,-1.5],explored_radius=0)]
    prior=dict(id='frontier:0:0',cell=[0,0],position_ned=[.5,.5,-1.5],node_stable_id='old',node_id=0)
    records,events=_frontiers(a,nodes,[prior],True)
    assert next(f for f in records if f['id']==prior['id'])['node_stable_id']=='old'
    assert not any(e['id']==prior['id'] for e in events)
    grid.known.add(hole);grid.occupied.add(hole)
    records,events=_frontiers(a,nodes,[prior],True)
    assert prior['id'] not in {f['id'] for f in records}
    assert any(e.get('reason')=='became_known' for e in events)


def test_paper_node_becomes_unknown_inside_map_is_removed():
    a=fixture();options=GraphConfig(paper_aligned=True,max_samples=20,coverage_samples=1)
    g,s=update_graph(a,frame_key='test',options=options)
    victim=g['nodes'][0];cell=tuple(victim['cell'])
    a.grid.known.remove(cell);a.grid.free.discard(cell)
    b=MapAdapter.from_grid(a.grid,{'snapshot_id':'unknown-hole','timestamp':2})
    new,_=update_graph(b,frame_key='test',state=s,options=options)
    assert any(e['id']==victim['stable_id'] and e.get('reason')=='zero_free_radius_in_map' for e in new['incremental']['node_events'])
    assert victim['stable_id'] not in {n['stable_id'] for n in new['nodes']}


def test_continuous_circle_coverage_uses_actual_center():
    from exploration_wildos.node_reduction import covered_cells
    a=fixture()
    node={'cell':[0,0],'position_ned':[.99,.5,-1.5],'free_radius':.55}
    assert covered_cells(a.grid,node)=={(0,0),(1,0)}
