from types import SimpleNamespace
from exploration_wildos.graph_compaction import compact_graph


def graph(edges,frontier=2):
    nodes=[dict(id=i,stable_id=f'nav:{i}',cell=[0,0],position_ned=[0,0,-1.5],free_radius=1,
                frontier_ids=['f'] if i==frontier else [],is_frontier=i==frontier) for i in range(5)]
    return dict(nodes=nodes,edges=[dict(id=i,source=a,target=b,length=d) for i,(a,b,d) in enumerate(edges)],
                robot_attachment=dict(node_id=0,component_id=0),components=[[0,1,2,3,4]],
                frontiers=[dict(id='f',node_id=frontier,node_stable_id=f'nav:{frontier}')],diagnostics={})


def test_compaction_preserves_coverage_anchor_frontier_and_restores_short_paths():
    g=graph([(0,1,1),(1,2,1),(0,3,2),(3,4,2),(4,2,2)])
    grid=SimpleNamespace(free={(0,0)},resolution=1)
    result=compact_graph(g,grid)
    report=result['node_compaction']
    assert report['covered_cells_before']==report['covered_cells_after']==1
    assert report['path_repair_restored_nodes']>0
    assert result['nodes'][result['robot_attachment']['node_id']]['stable_id']=='nav:0'
    assert result['nodes'][result['frontiers'][0]['node_id']]['stable_id']=='nav:2'
    assert len(g['nodes'])==5


def test_compaction_removes_redundant_complete_graph_nodes():
    g=graph([(a,b,1) for a in range(5) for b in range(a+1,5)])
    result=compact_graph(g,SimpleNamespace(free={(0,0)},resolution=1))
    assert len(result['nodes'])==2 and len(result['edges'])==1
    assert len(result['components'])==1
    assert compact_graph(g,SimpleNamespace(free={(0,0)},resolution=1))==result
