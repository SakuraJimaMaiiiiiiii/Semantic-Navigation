from exploration_wildos.edge_sparsification import sparsify_edges


def edge(a,b,d):return dict(id=0,source=a,target=b,length=d,stable_id=f'{a}|{b}')


def test_local_clusters_keep_long_bridge_and_isolated_components():
    edges=[edge(0,1,1),edge(1,2,1),edge(0,2,2),edge(3,4,1),edge(4,5,1),edge(3,5,2),edge(2,3,6),edge(1,4,7),edge(6,7,1)]
    reduced,report=sparsify_edges(9,edges,neighbors=1,local_radius=3)
    assert report['components_preserved'] and report['long_bridge_edges']==1
    assert '2|3' in {e['stable_id'] for e in reduced}
    assert '1|4' not in {e['stable_id'] for e in reduced}
    assert {e['stable_id'] for e in reduced}<={e['stable_id'] for e in edges}
    assert sparsify_edges(9,edges,neighbors=1,local_radius=3)==(reduced,report)


def test_dense_graph_shrinks_and_no_new_edges_or_weights():
    edges=[edge(a,b,abs(b-a)) for a in range(12) for b in range(a+1,12)]
    reduced,report=sparsify_edges(12,edges,neighbors=2,local_radius=3)
    assert len(reduced)<len(edges)/2
    assert report['components_preserved'] and report['long_bridge_edges']==0
    assert all(e['length']==abs(e['source']-e['target']) for e in reduced)


def test_empty_and_singleton_graphs():
    for count in (0,1,3):
        reduced,report=sparsify_edges(count,[])
        assert reduced==[] and report['components_preserved']


def test_detour_repairs_bound_all_pair_shortest_paths():
    import numpy as np
    from scipy.sparse.csgraph import shortest_path
    points=np.array([(0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,1)],dtype=float)
    edges=[edge(a,b,float(np.linalg.norm(points[a]-points[b]))) for a in range(8) for b in range(a+1,8)]
    reduced,report=sparsify_edges(8,edges,neighbors=1,local_radius=1.01,max_stretch=1.5)
    def distances(es):
        matrix=np.zeros((8,8))
        for e in es:matrix[e['source'],e['target']]=matrix[e['target'],e['source']]=e['length']
        return shortest_path(matrix,directed=False)
    a,b=distances(edges),distances(reduced)
    assert np.all(b<=1.5*a+1e-9)
    assert report['detour_shortcut_edges']>0
