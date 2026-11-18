import copy
import json
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen
import pytest

from exploration_wildos.live_map import LiveMapServer
from exploration_wildos.map_adapter import MapAdapter
from exploration_wildos.graph_export import render_graph_html
from exploration_wildos.combined_map import compose_map_html


def fixture():
    grid = SimpleNamespace(known={(x,y) for x in range(8) for y in range(8)},
                           free={(x,y) for x in range(8) for y in range(8)},
                           occupied=set(), resolution=1., origin=[0.,0.], altitude=-1.5)
    nodes=[dict(id=i,stable_id=f'nav:{i}',position_ned=[i+1.,2.,-1.5],
                free_radius=1.,explored_radius=2.) for i in range(3)]
    graph=dict(nodes=nodes,edges=[dict(source=0,target=1),dict(source=1,target=2)],
               components=[[0,1,2]],robot_attachment=dict(position_ned=[1.,2.,-1.5],node_id=0,node_stable_id='nav:0'),
               diagnostics=dict(uncovered_safe_cells=[]))
    return MapAdapter.from_grid(grid), graph, dict(passed=True)


def test_live_http_snapshot_versioning_and_read_only_routes(tmp_path):
    adapter, graph, validation = fixture()
    page=tmp_path/'map.html'
    page.write_text('<html><meta http-equiv="refresh" content="2"><svg></svg></html>')
    import threading
    adapter.grid.runtime_lock = threading.Lock()
    server=LiveMapServer(page)
    try:
        server.robot(graph['robot_attachment'])
        server.publish(adapter, graph, validation, 1)
        graph['nodes'][0]['position_ned'][0]=999
        server.future.result(timeout=10)
        root=server.url.rsplit('/',1)[0]
        with urlopen(root+'/live.json') as response:
            data=json.load(response)
        assert data['version']==1 and data['nodes']==3
        assert data['positions']['nav:0'][0]==1
        assert data['robot']['node_stable_id']=='nav:0'
        with urlopen(root+'/live.json?since=1') as response:
            unchanged=json.load(response)
        assert 'svg' not in unchanged and 'positions' not in unchanged
        assert unchanged['robot']==data['robot']
        with urlopen(server.url) as response:
            html=response.read().decode()
        assert 'http-equiv="refresh"' not in html and 'wildos-live-status' in html
        with pytest.raises(HTTPError) as error:
            urlopen(root+'/../semantic_map.json')
        assert error.value.code==404
        assert (tmp_path/'navigation_graph/live_graph.json').exists()
    finally:
        server.close()


def test_initial_empty_graph_page_has_both_panels():
    page=compose_map_html('<html><svg viewBox="-2 -2 4 4"></svg></html>',
                          '<svg viewBox="-2 -2 4 4"></svg>',
                          dict(nodes=[],edges=[],components=[]),[0,0,0],0)
    assert page.count('id="graph-panel"')==1
    assert page.count('id="exploration-panel"')==1
    assert '_wildosUpdateBounds' in page


def test_navigation_updates_without_graph_version_change_and_survives_render(tmp_path):
    adapter,graph,validation=fixture()
    server=LiveMapServer(tmp_path/'map.html')
    state=dict(target_id='vehicle_02',phase='沿图路径执行',
        route=dict(node_path=['nav:0','nav:1'],points_ned=[[0,2,-1.5],[1,2,-1.5],[2,2,-1.5],[3,2,-1.5]]),
        active_segment=[[1,2,-1.5],[2,2,-1.5]],trace_ned=[[0,2,-1.5],[.5,2,-1.5]])
    try:
        server.navigation(state)
        server.publish(adapter,graph,validation,1)
        server.future.result(timeout=10)
        state['trace_ned'].append([.8,2,-1.5])
        assert len(server.navigation_snapshot()['trace_ned'])==2
        server.navigation(state)
        with urlopen(server.url.replace('/map.html','/live.json?since=1')) as response:
            update=json.load(response)
        assert 'svg' not in update
        assert len(update['navigation']['trace_ned'])==3
        assert update['navigation']['route']['node_path']==['nav:0','nav:1']
        graph['navigation']=state
        html=render_graph_html(adapter,graph,validation)
        assert 'data-planned-segment="1"' in html
        assert 'data-measured-trace' in html and 'data-active-segment' in html
        page=compose_map_html('<html><svg viewBox="0 0 8 8"></svg></html>',html,graph,[0,0,0],0)
        assert 'data-navigation-status' in page and 'vehicle_02' in page
        assert 'data-measured-trace' in page
    finally:
        server.close()
