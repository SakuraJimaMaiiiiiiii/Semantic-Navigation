"""Object annotations and non-traversable associations for saved navigation graphs."""
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np


def ray_status(start, end, resolution, states):
    """Check all voxels touched by a segment, excluding its surface endpoint voxel.

    This is geometric evidence only: no camera frustum or object visibility claim.
    """
    a, b = np.asarray(start)/resolution, np.asarray(end)/resolution
    delta = b-a
    ts = [0., 1.]
    for k in range(3):
        if abs(delta[k]) > 1e-12:
            ts.extend((v-a[k])/delta[k] for v in range(int(np.floor(min(a[k], b[k])))+1,
                                                       int(np.ceil(max(a[k], b[k])))))
    ts = sorted(set(t for t in ts if 0 <= t <= 1))
    samples = ts[:-1]+[(x+y)/2 for x,y in zip(ts, ts[1:])]
    cells = set()
    for t in samples:
        point = a+t*delta
        axes = []
        for v in point:
            axes.append([int(round(v))-1, int(round(v))] if abs(v-round(v)) < 1e-9 else [int(np.floor(v))])
        cells.update(itertools.product(*axes))
    cells.discard(tuple(np.floor(b).astype(int)))
    values = [states.get(c, 'unknown') for c in cells]
    if 'occupied' in values:
        return 'blocked'
    return 'unknown' if not values or 'unknown' in values else 'clear_to_surface'


def attach_objects(graph, meta, states, resolution, max_distance=6., max_links=3):
    """Keep navigation topology intact; semantic edges are never routing edges."""
    objects, links = [], []
    seen = set()
    for obj in meta.get('objects', []):
        if obj.get('class_name') not in ('car', 'truck', 'bus', 'van', 'vehicle'):
            continue
        identity = str(obj.get('runtime_instance_id') or obj['instance_id'])
        if identity in seen:
            raise ValueError('Duplicate vehicle instance: '+identity)
        seen.add(identity)
        position = np.asarray(obj['position_world'], dtype=float)
        anchor = np.asarray(obj.get('anchor_position_world', position), dtype=float)
        box = obj.get('planning_bbox_3d', obj['bbox_3d'])
        low, high = np.asarray(box['min'], dtype=float), np.asarray(box['max'], dtype=float)
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (position,anchor,low,high)) or np.any(low>high):
            raise ValueError('Invalid vehicle geometry: '+identity)
        node = dict(id='object:'+identity, instance_id=identity, node_type='vehicle', traversable=False,
                    position_ned=position.tolist(), observed_anchor_ned=anchor.tolist(), bbox_3d=box,
                    class_name=obj['class_name'], color=obj.get('color','unknown'),
                    confidence=obj.get('confidence'), observation_count=obj.get('observation_count'),
                    geometry_evidence='observed surfaces; not guaranteed complete extent')
        node['observed_bbox_3d'] = obj['bbox_3d']
        node['merged_from'] = list(obj.get('merged_from', []))
        node['identity_evidence'] = obj.get('identity_evidence')
        objects.append(node)
        candidates = []
        if max_links == 0:
            node['association_count'] = 0
            continue
        for nav in graph['nodes']:
            distance = float(np.linalg.norm(np.asarray(nav['position_ned'])-anchor))
            if distance <= max_distance:
                status = ray_status(nav['position_ned'], anchor, resolution, states)
                candidates.append(({'clear_to_surface':0,'unknown':1,'blocked':2}[status], distance, nav['id'], status))
        # Blocked rays cannot supply approach/observation associations.
        for _, distance, nav_id, status in sorted(candidates)[:max_links]:
            if status == 'blocked':
                continue
            links.append(dict(source=node['id'], target=nav_id, edge_type='object_association',
                              traversable=False, distance_m=distance, ray_status=status,
                              visibility_confirmed=False))
        node['association_count'] = sum(e['source']==node['id'] for e in links)
    graph['semantic_nodes'], graph['semantic_edges'] = objects, links
    graph['semantic_policy'] = dict(max_distance_m=max_distance, max_links=max_links,
        visibility='3-D ray to observed anchor; endpoint voxel excluded; camera visibility unverified',
        routing='Only nodes/edges are traversable; semantic_nodes/semantic_edges are annotations')
    return graph


def refresh_semantics(directory, *, incremental=True, paper_aligned=False, sparse_edges=False, compact_nodes=False):
    """Upgrade an existing graph, retaining old artifacts and original navigation data."""
    from .map_adapter import load_map
    from .graph_export import export_graph
    from navigation.atomic_file import atomic_write_text
    directory = Path(directory)
    output = directory/'navigation_graph'
    meta_bytes = (directory/'semantic_map.json').read_bytes()
    meta = json.loads(meta_bytes)
    graph = json.loads((output/'graph.json').read_text())
    if meta.get('coordinate_frame') != 'world_NED':
        raise ValueError('Semantic objects require world_NED coordinates')
    source = next((meta[k] for k in ('occupancy_2d','occupancy_3d')
                   if meta[k]['sha256']==graph['metadata']['source_sha256']), None)
    if source is None and incremental and (output/'incremental_state.json').exists():
        source = meta['occupancy_2d' if graph['metadata']['source_kind']=='exported_2d' else 'occupancy_3d']
    if source is None or hashlib.sha256((directory/source['file']).read_bytes()).hexdigest()!=source['sha256']:
        raise ValueError('Graph occupancy source does not match this exploration')
    archive = directory/meta['occupancy_3d']['file']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=meta['occupancy_3d']['sha256']:
        raise ValueError('3-D ray-check archive hash mismatch')
    with np.load(archive, allow_pickle=False) as data:
        states = {tuple(c): 'occupied' if v>=data['occupied_threshold'] else 'free' if v<=data['free_threshold'] else 'unknown'
                  for c,v in zip(data['indices'],data['log_odds'])}
        attach_objects(graph, meta, states, float(data['resolution']), max_links=0)
    graph['semantic_source_sha256'] = hashlib.sha256(meta_bytes).hexdigest()
    config = graph['metadata']['navigation_config']
    adapter = load_map(directory/source['file'], altitude=graph['metadata']['altitude_ned'],
                       clearance=config['clearance_radius'], vertical_half_extent=config['vertical_half_extent'])
    from .vehicle_constraints import constrain_adapter, validate_vehicles
    from .static_graph import GraphConfig, build_graph, validate_graph
    objects = graph['semantic_nodes']
    original_graph = graph
    adapter, constraints = constrain_adapter(adapter, objects)
    incremental_state = None
    if incremental:
        from .incremental_graph import update_graph
        state_path = output/'incremental_state.json'
        previous_state = json.loads(state_path.read_text()) if state_path.exists() else None
        from dataclasses import replace
        options = GraphConfig(**graph['options'])
        options = replace(options, paper_aligned=paper_aligned)
        if paper_aligned:
            options = replace(options, max_samples=1000, max_nodes=10000, free_radius_cap=4.,
                              edge_radius=8., traversability_clearance=.5, prune_redundant=False)
        profile_changed = previous_state is not None and previous_state['options'].get('paper_aligned', False)!=paper_aligned
        if profile_changed:
            previous_state = None  # Explicit profile change starts a new archived epoch.
        elif previous_state is not None and 'paper_aligned' not in previous_state['options']:
            previous_state['options'] = dict(previous_state['options'], paper_aligned=False, traversability_clearance=.5)
        if not paper_aligned and previous_state is None and not validate_graph(adapter, graph)['passed']:
            graph = build_graph(adapter, GraphConfig(**graph['options']), graph['robot_attachment'].get('position_ned'))
        adapter.metadata['semantic_source_sha256'] = hashlib.sha256(meta_bytes).hexdigest()
        frame_key = json.dumps(dict(session=meta['session_id'], home=meta['home_ned'],
                                    yaw=meta['initial_yaw_rad'], frame=meta['coordinate_frame']), sort_keys=True)
        graph, incremental_state = update_graph(adapter, frame_key=frame_key, state=previous_state,
            options=options, start=graph['robot_attachment'].get('position_ned'),
            bootstrap=graph if previous_state is None and not paper_aligned else None)
    else:
        static_options = dict(graph['options'], paper_aligned=False)
        graph = build_graph(adapter, GraphConfig(**static_options), graph['robot_attachment'].get('position_ned'))
    import copy
    from .edge_sparsification import apply_sparsification
    if graph.get('node_compaction'):
        if incremental_state is None or 'uncompacted_graph' not in incremental_state:
            raise ValueError('Compact state is missing its sampling graph')
        graph = copy.deepcopy(incremental_state['uncompacted_graph'])
    dense_edges = graph['edges']
    if graph.get('edge_sparsification'):
        if incremental_state is None or 'candidate_edges' not in incremental_state:
            raise ValueError('Sparse state is missing its certified candidate edges')
        dense_edges = incremental_state['candidate_edges']
    graph['edges'] = dense_edges
    graph.pop('edge_sparsification', None)
    if incremental_state is not None:
        incremental_state['uncompacted_graph'] = copy.deepcopy(graph)
    if compact_nodes:
        from .graph_compaction import compact_graph
        graph = compact_graph(graph, adapter.grid)
        dense_edges = graph['edges']
    if incremental_state is not None:
        incremental_state['candidate_edges'] = dense_edges
    apply_sparsification(graph, dense_edges, enabled=sparse_edges, max_stretch=1.2 if compact_nodes else 1.5)
    if graph.get('incremental'):
        old_ids = {e.get('stable_id') for e in original_graph['edges']}
        new_ids = {e.get('stable_id') for e in graph['edges']}
        if old_ids!=new_ids or original_graph.get('incremental',{}).get('step')!=graph['incremental']['step']:
            graph['incremental']['added_edge_ids'] = sorted(new_ids-old_ids-{None})
            graph['incremental']['removed_edge_ids'] = sorted(old_ids-new_ids-{None})
        else:
            graph['incremental'] = copy.deepcopy(original_graph['incremental'])
    # Preserve provenance while associations are recomputed for the rebuilt IDs.
    attach_objects(graph, meta, states, adapter.grid.resolution)
    graph['semantic_source_sha256'] = hashlib.sha256(meta_bytes).hexdigest()
    previous = original_graph.get('vehicle_constraints', {})
    constraints['previous_nodes'] = previous.get('previous_nodes', len(original_graph['nodes']))
    constraints['previous_edges'] = previous.get('previous_edges', len(original_graph['edges']))
    constraints['previous_components'] = previous.get('previous_components', len(original_graph['components']))
    graph['vehicle_constraints'] = constraints
    validation = validate_graph(adapter, graph)
    validation['vehicle_geometry'] = validate_vehicles(graph, objects, adapter.grid.config)
    validation['passed'] = validation['passed'] and validation['vehicle_geometry']['passed']
    if not validation['passed']:
        raise ValueError('Constrained graph failed geometry validation; original output preserved')
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        stage = Path(temp)/'export'
        export_graph(stage, adapter, graph, validation)
        if incremental_state is not None:
            incremental_state['graph'] = graph
            (stage/'incremental_state.json').write_text(json.dumps(incremental_state, ensure_ascii=False, indent=2, allow_nan=False))
        digest = hashlib.sha256((output/'graph.json').read_bytes()).hexdigest()[:12]
        backup = output/'history'/('before_semantics_'+digest)
        changed = any((output/name).read_bytes() != (stage/name).read_bytes() for name in ('graph.json','graph.html'))
        if changed:
            backup.mkdir(parents=True, exist_ok=True)
            for name in ('graph.json','graph.html','validation.json','incremental_state.json'):
                if (output/name).exists() and not (backup/name).exists():
                    shutil.copy2(output/name, backup/name)
            if not (backup/'map.html').exists():
                shutil.copy2(directory/'map.html', backup/'map.html')
        for name in ('audit.json', 'comparison.json', 'vehicle_constraint_comparison.json', 'paper_alignment_report.json', 'sparsification_comparison.json', 'compaction_comparison.json', 'graph_preview.png', 'ui_verification', 'README.md'):
            if changed and (output/name).exists():
                backup.mkdir(parents=True, exist_ok=True)
                shutil.move(str(output/name), str(backup/name))
        if not incremental and (output/'incremental_state.json').exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.move(str(output/'incremental_state.json'), str(backup/'incremental_state.json'))
        for name in ('graph.json','graph.html','validation.json') + (('incremental_state.json',) if incremental else ()):
            atomic_write_text(output/name, (stage/name).read_text(encoding='utf-8'))
    return graph
