import hashlib
import json

import pytest

from exploration_wildos.incremental_graph import update_graph
from exploration_wildos.map_adapter import MapAdapter
from exploration_wildos.static_graph import GraphConfig, validate_graph
from config.semantic_navigation_config import SemanticNavigationConfig
from navigation.route import NavigationGrid


def snapshot(size=6, occupied=(), timestamp=1):
    cells={(x,y) for x in range(-size,size+1) for y in range(-size,size+1)}
    grid=NavigationGrid(cells,set(occupied),1.,(0,0),-1.5,
        SemanticNavigationConfig(clearance_radius=.4,vertical_half_extent=.25))
    digest=hashlib.sha256(repr((sorted(cells),sorted(occupied),timestamp)).encode()).hexdigest()
    return MapAdapter.from_grid(grid,{'snapshot_id':digest,'timestamp':timestamp})


def update(a,state=None,**kw):
    return update_graph(a,frame_key='fixture-world',state=state,
                        options=GraphConfig(max_samples=2000,coverage_samples=2000),**kw)


def test_idempotent_resume_history_and_frontier_lifecycle():
    a=snapshot();g,s=update(a,start=[0,0,-1.5])
    assert g['frontiers'] and any(n['is_frontier'] for n in g['nodes'])
    clone=json.loads(json.dumps(s));g2,s2=update(a,clone,start=[0,0,-1.5])
    assert g2==g and s2==s
    b=snapshot(10,timestamp=2);new,state=update(b,s,start=[0,0,-1.5])
    old_nodes={n['stable_id']:n for n in g['nodes']}
    retained={n['stable_id']:n for n in new['nodes']}
    assert old_nodes.keys()<=retained.keys()
    assert all(retained[k]['explored_radius']>=v['explored_radius'] for k,v in old_nodes.items())
    assert any(retained[k]['explored_radius']>v['explored_radius'] for k,v in old_nodes.items())
    assert any(e.get('reason')=='became_known' for e in new['incremental']['frontier_events'])
    assert validate_graph(b,new)['passed']
    assert not new['diagnostics']['uncovered_safe_cells']
    assert all(n['is_frontier']==bool(n['frontier_ids']) for n in new['nodes'])


def test_obstacle_retires_node_and_edges_never_reuses_stable_id():
    g,s=update(snapshot())
    victim=g['nodes'][0]
    b=snapshot(occupied=[tuple(victim['cell'])],timestamp=2)
    newer,state=update(b,s)
    assert victim['stable_id'] not in {n['stable_id'] for n in newer['nodes']}
    assert any(e['id']==victim['stable_id'] and e['event']=='removed' for e in newer['incremental']['node_events'])
    assert all(victim['stable_id'] not in [e['source_stable_id'],e['target_stable_id']] for e in newer['edges'])
    restored,state=update(snapshot(timestamp=3),state)
    assert victim['stable_id'] not in {n['stable_id'] for n in restored['nodes']}
    assert validate_graph(b,newer)['passed']


def test_unknown_dormancy_and_reactivation_keeps_memory():
    g,s=update(snapshot(10))
    small,state=update(snapshot(6,timestamp=2),s)
    assert small['incremental']['dormant_nodes']>0
    dormant={n['stable_id']:n for n in state['nodes'] if not n['active']}
    big,state=update(snapshot(10,timestamp=3),state)
    active={n['stable_id']:n for n in big['nodes']}
    assert dormant.keys()<=active.keys()
    assert all(active[k]['explored_radius']>=v['explored_radius'] for k,v in dormant.items())


def test_frame_and_order_rejected_without_mutation():
    a=snapshot(timestamp=3);g,s=update(a);before=json.dumps(s,sort_keys=True)
    with pytest.raises(ValueError,match='Out-of-order'):
        update(snapshot(timestamp=2),s)
    with pytest.raises(ValueError,match='Frame'):
        update_graph(a,frame_key='different',state=s,options=GraphConfig(max_samples=2000,coverage_samples=2000))
    assert json.dumps(s,sort_keys=True)==before


def test_empty_map_frontiers_cleared_and_nodes_dormant():
    g,s=update(snapshot())
    a=snapshot(timestamp=2)
    grid=NavigationGrid(set(),set(),1,(0,0),-1.5,a.grid.config)
    empty=MapAdapter.from_grid(grid,{'snapshot_id':'empty','timestamp':2})
    result,state=update(empty,s)
    assert result['nodes']==result['edges']==result['frontiers']==[]
    assert result['incremental']['dormant_nodes']==len(g['nodes'])


def test_replay_disk_roundtrip_and_reject_mixed_frames(tmp_path):
    import numpy as np
    from exploration_wildos.replay_graph import replay
    entries=[]
    for i,size in enumerate((6,9)):
        a=snapshot(size,timestamp=i+1)
        path=tmp_path/f'map{i}.npz'
        np.savez(path,indices=np.asarray(sorted(a.raw_free)),states=np.zeros(len(a.raw_free),dtype=np.int8),
            resolution=1.,altitude_ned=-1.5,vertical_half_extent=.25,clearance_radius=.4,timestamp=float(i+1))
        entries.append({'map':path.name})
    manifest={'frame_key':'test-world','snapshots':entries}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    summaries=replay(path,tmp_path/'output')
    assert len(summaries)==2 and summaries[1]['events']['step']==1
    assert (tmp_path/'output/incremental_state.json').is_file()
    with pytest.raises(ValueError,match='already exists'):
        replay(path,tmp_path/'output')
    entries[1]['frame_key']='corrected-other-world'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='coordinate frame changed'):
        replay(path,tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def test_saved_workflow_accepts_next_snapshot_and_preserves_repeat_audit(tmp_path):
    import numpy as np
    from exploration_wildos.map_adapter import load_map
    from exploration_wildos.static_graph import build_graph
    from exploration_wildos.graph_export import export_graph
    from exploration_wildos.semantic_graph import refresh_semantics
    archive=tmp_path/'navigation_cache.npz'
    np.savez(archive,indices=np.empty((0,3),dtype=int),log_odds=np.empty(0,dtype=np.float32),
             resolution=1.,free_threshold=-.2,occupied_threshold=.7,timestamp=1.)
    archive_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
    def save(size,timestamp):
        a=snapshot(size,timestamp=timestamp);path=tmp_path/'occupancy_2d.npz'
        np.savez(path,indices=np.asarray(sorted(a.raw_free)),states=np.zeros(len(a.raw_free),dtype=np.int8),
                 resolution=1.,altitude_ned=-1.5,vertical_half_extent=.25,clearance_radius=.4,timestamp=timestamp)
        meta={'coordinate_frame':'world_NED','session_id':'fixture','home_ned':[0,0,0],'initial_yaw_rad':0.,
              'objects':[], 'occupancy_2d':{'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
              'occupancy_3d':{'file':archive.name,'sha256':archive_hash}}
        (tmp_path/'semantic_map.json').write_text(json.dumps(meta))
        return load_map(path)
    a=save(6,1.);g=build_graph(a);export_graph(tmp_path/'navigation_graph',a,g,validate_graph(a,g))
    (tmp_path/'map.html').write_text('<html>original</html>')
    initial=refresh_semantics(tmp_path)
    marker=tmp_path/'navigation_graph/audit.json';marker.write_text('{}')
    repeat=refresh_semantics(tmp_path)
    assert initial==repeat and marker.exists()
    save(9,2.)
    next_graph=refresh_semantics(tmp_path)
    assert next_graph['incremental']['step']==1
    assert {n['stable_id'] for n in initial['nodes']} <= {n['stable_id'] for n in next_graph['nodes']}
    assert not marker.exists()  # old audit archived, not left claiming to describe the new graph

    sparse = refresh_semantics(tmp_path, sparse_edges=True)
    repeat_sparse = refresh_semantics(tmp_path, sparse_edges=True)
    assert sparse == repeat_sparse
    assert sparse['nodes'] == next_graph['nodes']
    restored = refresh_semantics(tmp_path, sparse_edges=False)
    assert restored['edges'] == next_graph['edges']

    compact = refresh_semantics(tmp_path, sparse_edges=True, compact_nodes=True)
    repeat_compact = refresh_semantics(tmp_path, sparse_edges=True, compact_nodes=True)
    assert compact == repeat_compact
    assert compact['diagnostics']['covered_safe_cells'] == next_graph['diagnostics']['covered_safe_cells']
    restored = refresh_semantics(tmp_path, sparse_edges=False, compact_nodes=False)
    assert restored['nodes'] == next_graph['nodes']
    assert restored['edges'] == next_graph['edges']
