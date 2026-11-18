import copy
import numpy as np
from config.semantic_navigation_config import SemanticNavigationConfig
from navigation.route import NavigationGrid
from exploration_wildos.target_graph import TargetGraphGrid
from exploration_wildos.map_adapter import MapAdapter
from exploration_wildos.vehicle_constraints import constrain_adapter


def fixture():
    g=NavigationGrid({(x,y) for x in range(-3,16) for y in range(-5,6)},set(),.5,(0.,0.),-1.5,
                     SemanticNavigationConfig(clearance_radius=.1))
    positions=[g.point((x,0)) for x in (0,4,8,12)]
    graph=dict(nodes=[dict(id=i,stable_id=f'nav:{i}',position_ned=p.tolist()) for i,p in enumerate(positions)],
               edges=[dict(source=i,target=i+1) for i in range(3)], options=dict(edge_radius=2.5),components=[[0,1,2,3]])
    return g,graph,positions


def test_target_route_uses_graph_edges_and_safe_terminal_connector(monkeypatch):
    g,graph,p=fixture()
    def forbid(*args):
        raise AssertionError('Must never fall back to grid A*')
    monkeypatch.setattr(NavigationGrid,'path',forbid)
    grid=TargetGraphGrid(g,graph)
    goal=p[-1]+np.array([.1,0,0])
    route=grid.path(p[0],goal)
    assert route is not None
    assert grid.routes[tuple(goal)]['node_path']==['nav:0','nav:1','nav:2','nav:3']
    assert all(grid.segment_free(a,b) for a,b in zip(route,route[1:]))
    assert np.allclose(route[-1],goal)


def test_unreachable_graph_and_new_obstacle_do_not_get_grid_fallback():
    g,graph,p=fixture()
    graph['edges']=[dict(source=0,target=1),dict(source=2,target=3)]
    assert TargetGraphGrid(g,graph).path(p[0],p[-1]) is None
    g,graph,p=fixture()
    g.free.discard((6,0))
    assert TargetGraphGrid(g,graph).path(p[0],p[-1]) is None


def test_target_center_forbidden_even_above_car_and_records_unchanged():
    g,graph,p=fixture()
    vehicles=[dict(id='vehicle',bbox_3d=dict(min=[5.8,-.2,-.4],max=[6.8,.8,0.]))]
    before=copy.deepcopy(vehicles)
    adapter,_=constrain_adapter(MapAdapter.from_grid(g),vehicles)
    assert TargetGraphGrid(adapter.grid,graph).path(p[0],p[-1]) is None
    assert vehicles==before
