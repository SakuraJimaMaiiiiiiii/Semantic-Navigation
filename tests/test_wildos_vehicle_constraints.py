import numpy as np
from exploration_wildos.vehicle_constraints import segment_hits_box, constrain_adapter, validate_vehicles
from exploration_wildos.map_adapter import MapAdapter
from exploration_wildos.static_graph import build_graph, GraphConfig, validate_graph
from navigation.route import NavigationGrid
from config.semantic_navigation_config import SemanticNavigationConfig


def test_segment_sweep_tangency_and_vertical_clearance():
    low=np.array([0,0,-1.5]);high=np.array([1,1,0])
    assert segment_hits_box([-1,.5,-1],[2,.5,-1],low,high)
    assert segment_hits_box([-1,0,-1.5],[2,0,-1.5],low,high)
    assert not segment_hits_box([-1,.5,-1.6],[2,.5,-1.6],low,high)
    assert segment_hits_box([.5,.5,-1],[.5,.5,-1],low,high)
    assert not segment_hits_box([2,2,-1],[2,2,-1],low,high)


def test_constrained_rebuild_and_above_vehicle():
    config=SemanticNavigationConfig(clearance_radius=.4,vertical_half_extent=.25)
    grid=NavigationGrid({(x,y) for x in range(-15,16) for y in range(-15,16)},set(),.3,(0,0),-1.5,config)
    adapter=MapAdapter.from_grid(grid)
    objects=[{'id':'high','bbox_3d':{'min':[-.4,-.4,-1.4],'max':[.4,.4,0]}},
             {'id':'low','bbox_3d':{'min':[-3,-3,-.7],'max':[3,3,0]}}]
    constrained,report=constrain_adapter(adapter,objects)
    assert report['active_vehicle_ids']==['high','low']
    assert len(constrained.grid.free)<len(grid.free)
    assert (0,0) not in constrained.grid.free and (0,0) in grid.free
    graph=build_graph(constrained,GraphConfig(max_samples=2000,coverage_samples=2000),[0,0,-1.5])
    assert graph['robot_attachment']['status']=='unsafe_start'
    assert validate_graph(constrained,graph)['passed']
    assert validate_vehicles(graph,objects,config)['passed']
    assert not any(n['above_vehicle_ids'] for n in graph['nodes'])
    assert not graph['diagnostics']['uncovered_safe_cells']
    graph['nodes'][0]['position_ned']=[0,0,-1.5]
    assert not validate_vehicles(graph,objects,config)['passed']


def test_edge_and_robot_attachment_sweeps_are_checked():
    config=SemanticNavigationConfig(clearance_radius=.4,vertical_half_extent=.25)
    objects=[{'id':'vehicle','bbox_3d':{'min':[0,0,-1.4],'max':[1,1,0]}}]
    graph={'nodes':[{'id':0,'position_ned':[-1,.5,-1.5]},
                    {'id':1,'position_ned':[2,.5,-1.5]}],
           'edges':[{'id':0,'source':0,'target':1}],
           'robot_attachment':{'status':'attached','position_ned':[-1,.5,-1.5],'node_id':1}}
    result=validate_vehicles(graph,objects,config)
    assert result['invalid_nodes']==[]
    assert result['invalid_edges']==[0]
    assert result['invalid_robot_attachment']


def test_vehicle_margin_is_smaller_than_occupancy_clearance_and_consistent():
    from exploration_wildos.vehicle_constraints import expanded_boxes
    config=SemanticNavigationConfig(clearance_radius=.4,vertical_half_extent=.25)
    objects=[{'id':'car','bbox_3d':{'min':[0,0,-1.4],'max':[1,1,0]}}]
    _,low,high=expanded_boxes(objects,config)[0]
    assert np.allclose(low,[-.2,-.2,-1.65])
    assert np.allclose(high,[1.2,1.2,.25])
    g=NavigationGrid({(x,y) for x in range(-20,21) for y in range(-20,21)},set(),.1,(0,0),-1.5,config)
    constrained,report=constrain_adapter(MapAdapter.from_grid(g),objects)
    assert report['horizontal_margin_m']==.2
    assert (-4,5) in constrained.grid.free  # outside new buffer, excluded by old .4 m buffer
    assert (-1,5) not in constrained.grid.free
    assert config.clearance_radius==.4  # raw occupancy/body clearance unchanged


def test_no_overflight_owns_full_footprint_without_changing_localization():
    import copy
    from exploration_wildos.vehicle_constraints import expanded_boxes, segment_hits_vehicle
    config=SemanticNavigationConfig(clearance_radius=.4,vertical_half_extent=.25)
    objects=[dict(id='object:car',position_ned=[.5,.5,-.4],
                  bbox_3d={'min':[0,0,-.7],'max':[1,1,0]})]
    before=copy.deepcopy(objects)
    grid=NavigationGrid({(x,y) for x in range(-15,16) for y in range(-15,16)},set(),.3,(0,0),-3.,config)
    original=(set(grid.known),set(grid.occupied),set(grid.free))
    constrained,report=constrain_adapter(MapAdapter.from_grid(grid),objects)
    assert report['forbid_overflight'] and not report['localization_modified']
    assert constrained.grid.cell([.5,.5,-3]) in constrained.grid.occupied
    assert len(report['cell_owners'])==len(report['occupied_cells'])
    assert all(row['owners'] and row['owners'][0]['vehicle_id']=='object:car' for row in report['cell_owners'])
    assert {o['zone'] for row in report['cell_owners'] for o in row['owners']}=={'observed_footprint','clearance_buffer'}
    _,low,high=expanded_boxes(objects,config)[0]
    assert segment_hits_vehicle([-1,.5,-3],[2,.5,-3],low,high)
    assert not segment_hits_box([-1,.5,-3],[2,.5,-3],low,high)
    assert objects==before and (grid.known,grid.occupied,grid.free)==original


def test_overlapping_footprints_preserve_multiple_owners():
    config=SemanticNavigationConfig()
    grid=NavigationGrid({(x,y) for x in range(-5,6) for y in range(-5,6)},set(),1.,(0,0),-5.,config)
    objects=[dict(id=name,bbox_3d={'min':[0,0,-1],'max':[1,1,0]}) for name in ['a','b']]
    _,report=constrain_adapter(MapAdapter.from_grid(grid),objects)
    assert all({o['vehicle_id'] for o in row['owners']}=={'a','b'} for row in report['cell_owners'])
