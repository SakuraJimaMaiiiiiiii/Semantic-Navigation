"""Load a completed raw-NED WildOS archive without starting flight resources."""
import hashlib
import json
from pathlib import Path
import numpy as np

from config.semantic_navigation_config import SemanticNavigationConfig
from navigation.route import NavigationGrid
from navigation.standalone_navigation import read_archive, SavedMapNotFound
from .identity_overrides import corrected_records
from .map_adapter import MapAdapter
from .static_graph import validate_graph


def read_graph_archive(directory):
    source, archive = read_archive(directory)  # landing, frame and cache hash gates
    paths = sorted((source/'navigation_graph').glob('update_*/graph.json'))
    if not paths:
        raise SavedMapNotFound('保存地图没有 WildOS 增量图；请用 --map 指定 WildOS 探索目录')
    path = paths[-1]  # A broken latest snapshot is not silently replaced by an older one.
    graph = json.loads(path.read_text(encoding='utf-8'))
    state = json.loads(path.with_name('incremental_state.json').read_text(encoding='utf-8'))
    identity = state['identity']
    frame = identity['frame_key']
    if (not frame.startswith(archive['session_id']+':PX4-raw:')
            or graph.get('incremental',{}).get('frame_key') != frame
            or graph.get('metadata',{}).get('frame') != 'NED'):
        raise ValueError('保存图与探索地图不属于同一个 PX4 原始坐标系')
    nodes = graph['nodes']
    if not nodes or [n['id'] for n in nodes] != list(range(len(nodes))):
        raise ValueError('保存图节点编号不连续或图为空')
    points = np.asarray([n['position_ned'] for n in nodes],dtype=float)
    if points.shape != (len(nodes),3) or not np.isfinite(points).all():
        raise ValueError('保存图包含无效节点坐标')
    if len({n['stable_id'] for n in nodes}) != len(nodes):
        raise ValueError('保存图包含重复稳定节点 ID')
    for edge in graph['edges']:
        if any(type(edge[k]) is not int or not 0 <= edge[k] < len(nodes) for k in ('source','target')):
            raise ValueError('保存图边引用了无效节点')
    with np.load(path.with_name('planning_grid.npz'),allow_pickle=False) as data:
        grid = object.__new__(NavigationGrid)
        for name in ('known','free','occupied'):
            setattr(grid,name,set(map(tuple,data[name+'_indices'])))
        grid.resolution = float(data['resolution'])
        grid.origin = data['origin'].copy()
        grid.altitude = float(data['altitude_ned'])
    if (not np.isfinite(grid.altitude) or not np.isfinite(grid.origin).all()
            or not np.isclose(grid.resolution,archive['occupancy_map']['resolution'])
            or not np.isclose(grid.resolution,identity['resolution'])
            or not np.allclose(grid.origin,identity['origin'])
            or not np.isclose(grid.altitude,identity['altitude'])):
        raise ValueError('保存图的高度、分辨率或原点不一致')
    grid.config = SemanticNavigationConfig(**identity['navigation_config'])
    adapter = MapAdapter.from_grid(grid,graph['metadata'])
    validation = validate_graph(adapter,graph)
    if not validation['passed']:
        raise ValueError('保存图未通过节点与边的几何复核')
    archive['records'] = corrected_records(archive['records'],source,archive['session_id'])
    files = [path,path.with_name('planning_grid.npz'),path.with_name('incremental_state.json'),source/'bundle.json']
    provenance = dict(source=str(source),source_session_id=archive['session_id'],
        graph_source=str(path),source_frame=frame,
        sha256={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    return source, archive, dict(graph=graph,adapter=adapter,validation=validation,provenance=provenance)


def latest_graph_archive(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]/'data/runs'
    for run in sorted(root.glob('run_*'),reverse=True):
        directory = run/'exploration'
        if not (directory/'bundle.json').is_file():
            continue
        try:
            _, archive, _ = read_graph_archive(directory)
            if not archive['records']:
                raise ValueError('地图中没有可导航物体')
        except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
            print(f'--- 跳过不可用 WildOS 地图 {run.name}：{error} ----')
            continue
        return directory
    raise SavedMapNotFound('没有可用的已完成 WildOS 探索地图，请用 --map 指定完整存档')
