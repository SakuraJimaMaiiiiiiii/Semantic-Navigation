import copy

from exploration_wildos.semantic_graph import attach_objects, ray_status


def test_ray_checks_obstacles_unknown_and_endpoint():
    start,end = [.5,.5,.5],[3.5,.5,.5]
    free={(x,0,0):'free' for x in range(3)}
    assert ray_status(start,end,1,free)=='clear_to_surface'
    assert ray_status(start,end,1,{**free,(3,0,0):'occupied'})=='clear_to_surface'
    assert ray_status(start,end,1,{**free,(1,0,0):'occupied'})=='blocked'
    del free[(1,0,0)]
    assert ray_status(start,end,1,free)=='unknown'


def test_ray_on_boundary_checks_both_sides():
    assert ray_status([.5,1,.5],[2.5,1,.5],1,{(1,0,0):'occupied'})=='blocked'


def test_objects_do_not_change_navigation_and_keep_unverified_separate():
    graph={'nodes':[{'id':0,'position_ned':[.5,.5,.5]}], 'edges':[], 'components':[[0]]}
    before=copy.deepcopy(graph)
    meta={'objects':[{'instance_id':'vehicle_1','class_name':'car','position_world':[3.5,.5,.5],
                     'bbox_3d':{'min':[3,0,0],'max':[4,1,1]},'confidence':.9}]}
    attach_objects(graph,meta,{},1)
    assert all(graph[k]==v for k,v in before.items())
    assert len(graph['semantic_nodes'])==len(graph['semantic_edges'])==1
    edge=graph['semantic_edges'][0]
    assert edge['ray_status']=='unknown' and not edge['visibility_confirmed'] and not edge['traversable']
    attach_objects(graph,meta,{(1,0,0):'occupied'},1)
    assert graph['semantic_edges']==[]
    assert graph['semantic_nodes'][0]['association_count']==0
