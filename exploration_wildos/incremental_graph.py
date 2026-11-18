"""Persistent single-height graph updates for ordered, same-frame map snapshots.

Stable IDs are separate from dense export indices. This module consumes complete
saved snapshots, not moving local windows or unaligned loop-closure revisions.
"""
import copy
from dataclasses import asdict
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree

from .static_graph import DistanceField, GraphConfig, _components, validate_graph
from .node_reduction import covered_cells


def _identity(adapter, frame_key):
    g=adapter.grid
    return dict(frame_key=frame_key, resolution=g.resolution, origin=g.origin.tolist(),
                altitude=g.altitude, navigation_config=asdict(g.config))


def _frontiers(adapter, nodes, previous, paper_aligned=False, explored_memory=()):
    """Unknown-side boundary cells; links express geometric observation, not travel.

    The final unknown cell is not traversed. A raw-free ray to its neighboring
    known-free cell associates it with a safe node. This does not certify a
    camera frustum or a body-safe path all the way to the boundary.
    """
    grid=adapter.grid
    boundary={}
    for x,y in sorted(adapter.raw_free):
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            cell=(x+dx,y+dy)
            if cell not in grid.known:
                boundary.setdefault(cell,[]).append((x,y))
    raw_grid=copy.copy(grid)
    raw_grid.free=set(adapter.raw_free)
    old={tuple(f['cell']):f for f in previous}
    records=[];events=[]
    positions=np.array([n['position_ned'] for n in nodes])
    owners={n['stable_id']:n for n in nodes}
    retained_cells=set()
    if paper_aligned and grid.known:
        known_array=np.asarray(sorted(grid.known))
        lower,upper=known_array.min(axis=0)-1,known_array.max(axis=0)+1
        for f in previous:
            cell=tuple(f['cell'])
            if cell not in grid.known and np.all(np.asarray(cell)>=lower) and np.all(np.asarray(cell)<=upper) and f['node_stable_id'] in owners:
                record=copy.deepcopy(f)
                record['node_id']=owners[f['node_stable_id']]['id']
                records.append(record);retained_cells.add(cell)
    for cell, approaches in sorted(boundary.items()):
        if cell in retained_cells:
            continue
        point=grid.point(cell)
        prior=old.get(cell)
        distances=np.linalg.norm(positions-point,axis=1) if len(nodes) else np.array([])
        if (paper_aligned or prior is None) and any(d<=n['explored_radius']+1e-9 for d,n in zip(distances,nodes)):
            continue
        if any(np.linalg.norm(np.asarray(m['position_ned'])-point) <= m['explored_radius']+1e-9
               for m in explored_memory):
            continue
        assigned=None;approach=None
        order=sorted(range(len(nodes)),key=lambda i:(distances[i],nodes[i]['stable_id']))
        for i in order:
            for free_cell in approaches:
                if raw_grid.segment_free(positions[i],grid.point(free_cell)):
                    assigned=nodes[i];approach=free_cell;break
            if assigned is not None:
                break
        if assigned is None:
            continue
        record=dict(id=f'frontier:{cell[0]}:{cell[1]}',cell=list(cell),position_ned=point.tolist(),
                    node_stable_id=assigned['stable_id'],node_id=assigned['id'],
                    known_side_cell=list(approach),traversable=False,
                    association='raw-free sightline to known side; camera visibility unverified')
        records.append(record)
        if prior is None:
            events.append(dict(id=record['id'],event='created'))
        elif prior['node_stable_id']!=record['node_stable_id']:
            events.append(dict(id=record['id'],event='reassigned',previous_owner=prior['node_stable_id']))
    retained={f['id'] for f in records}
    for f in previous:
        if f['id'] not in retained:
            cell=tuple(f['cell'])
            reason='became_known' if cell in grid.known else 'boundary_disappeared' if cell not in boundary else 'no_valid_owner'
            events.append(dict(id=f['id'],event='removed',reason=reason))
    for n in nodes:
        n['frontier_ids']=[f['id'] for f in records if f['node_stable_id']==n['stable_id']]
        n['is_frontier']=bool(n['frontier_ids'])
    return records,events


def update_graph(adapter, *, frame_key, state=None, options=None, start=None, bootstrap=None):
    """Return (graph, state), without mutating inputs or performing disk writes.

    A changed frame/layer/config or an out-of-order snapshot is rejected. Unknown
    nodes become dormant; nodes invalidated in known space are retired forever.
    A repeated snapshot is idempotent. Retained explored radii never decrease.
    """
    if not isinstance(frame_key,str) or not frame_key.strip():
        raise ValueError('An explicit stable coordinate frame key is required')
    options=options or GraphConfig()
    identity=_identity(adapter,frame_key)
    timestamp=float(adapter.metadata.get('timestamp',0.))
    fingerprint=hashlib.sha256(json.dumps(dict(metadata=adapter.metadata,identity=identity,
        options=asdict(options),start=start),sort_keys=True,allow_nan=False).encode()).hexdigest()
    if state:
        if state['identity']!=identity or asdict(GraphConfig(**state['options']))!=asdict(options):
            raise ValueError('Frame, height, resolution or configuration changed; explicit new graph epoch required')
        if timestamp < state['timestamp']:
            raise ValueError('Out-of-order snapshot timestamp')
        if state['fingerprint']==fingerprint:
            return copy.deepcopy(state['graph']),copy.deepcopy(state)
    field=DistanceField(adapter,options.max_dense_cells)
    grid=adapter.grid
    history=copy.deepcopy(state['nodes']) if state else []
    next_id=state['next_id'] if state else 0
    events=[]
    if not state and bootstrap:
        if not validate_graph(adapter,bootstrap)['passed']:
            raise ValueError('Bootstrap nodes or edges do not match the constrained snapshot')
        for node in bootstrap['nodes']:
            n=copy.deepcopy(node);n['stable_id']=f'nav:{next_id}';next_id+=1
            n['active']=True;history.append(n)
    active=[];survivors=[]
    bounds = (np.asarray(sorted(grid.known)).min(axis=0)-1, np.asarray(sorted(grid.known)).max(axis=0)+1) if grid.known else None
    for n in history:
        cell=tuple(n['cell'])
        if cell in grid.free:
            obstacle,unknown=field.at_point(n['position_ned']) if options.paper_aligned else field.at(cell)
            n['free_radius']=min(obstacle,unknown,options.free_radius_cap)
            n['explored_radius']=max(n['explored_radius'],unknown)
            if not n.get('active',True):
                events.append(dict(id=n['stable_id'],event='reactivated'))
            n['active']=True;active.append(n);survivors.append(n)
        elif cell not in grid.known:
            if options.paper_aligned and bounds is not None and np.all(np.asarray(cell)>=bounds[0]) and np.all(np.asarray(cell)<=bounds[1]):
                events.append(dict(id=n['stable_id'],event='removed',reason='zero_free_radius_in_map'))
                continue
            if n.get('active',True):
                events.append(dict(id=n['stable_id'],event='dormant'))
            n['active']=False;n['frontier_ids']=[];n['is_frontier']=False;survivors.append(n)
        else:
            events.append(dict(id=n['stable_id'],event='removed',reason='occupied_or_insufficient_clearance'))
    cells=sorted(grid.free)
    rng=np.random.default_rng(options.seed+(state['step']+1 if state else 0))
    positions=[np.asarray(n['position_ned']) for n in active]
    radii=[n['free_radius'] for n in active]
    previous_positions=np.asarray(positions).copy()
    previous_radii=np.asarray(radii).copy()
    if options.paper_aligned:
        # Algorithm 3 draws independently from the entire free region. Only
        # V_(t-1) participates in circle rejection; V_new does not.
        raw_cells=sorted(adapter.raw_free)
        order=[]
        for _ in range(options.max_samples if raw_cells else 0):
            cell=raw_cells[int(rng.integers(len(raw_cells)))]
            xy=grid.origin+(np.asarray(cell)+rng.random(2))*grid.resolution
            order.append((cell,np.array([xy[0],xy[1],grid.altitude])))
    else:
        selected_cells=[cells[int(i)] for i in rng.permutation(len(cells))[:options.max_samples]]
        selected=set(selected_cells)
        selected_cells+=sorted((c for c in cells if c not in selected),key=lambda c:(min(field.at(c)),c))[:options.coverage_samples]
        order=[(cell,grid.point(cell)) for cell in selected_cells]
    tested=0
    for cell,point in order:
        if len(survivors)>=options.max_nodes:
            break
        tested+=1
        obstacle,unknown=field.at_point(point) if options.paper_aligned else field.at(cell)
        if options.paper_aligned:
            if cell not in grid.free or min(obstacle,unknown)<=options.traversability_clearance:
                continue
            rejection_positions = np.asarray(positions) if options.incremental_cleanup else previous_positions
            rejection_radii = np.asarray(radii) if options.incremental_cleanup else previous_radii
            if len(rejection_positions) and np.any(np.linalg.norm(rejection_positions-point,axis=1)<=rejection_radii):
                continue
        elif positions and np.any(np.linalg.norm(np.asarray(positions)-point,axis=1)<=radii):
            continue
        n=dict(stable_id=f'nav:{next_id}',position_ned=point.tolist(),cell=list(cell),
               free_radius=min(obstacle,unknown,options.free_radius_cap),explored_radius=unknown,
               active=True,valid=True,frontier_ids=[],is_frontier=False)
        next_id+=1;survivors.append(n);active.append(n);positions.append(point);radii.append(n['free_radius'])
        events.append(dict(id=n['stable_id'],event='created'))
    for i,n in enumerate(active):
        n['id']=i;n['validated_snapshot']=adapter.metadata.get('snapshot_id')
    pairs=sorted(cKDTree(positions).query_pairs(options.edge_radius)) if positions else []
    edges=[]
    for a,b in pairs:
        if grid.segment_free(positions[a],positions[b]):
            edge_id='|'.join(sorted((active[a]['stable_id'],active[b]['stable_id'])))
            edges.append(dict(id=len(edges),stable_id=edge_id,source=a,target=b,
                              source_stable_id=active[a]['stable_id'],target_stable_id=active[b]['stable_id'],
                              length=float(np.linalg.norm(positions[a]-positions[b])),valid=True,
                              validated_snapshot=adapter.metadata.get('snapshot_id')))
    components=_components(len(active),edges)
    attachment=dict(status='not_requested',node_id=None,component_id=None)
    if start is not None:
        start=np.asarray(start,dtype=float)
        if start.shape!=(3,) or not np.isfinite(start).all():
            raise ValueError('start must be finite NED xyz')
        attachment['position_ned']=start.tolist()
        attachment['status']='no_certified_connection'
        if abs(start[2]-grid.altitude)>1e-8:
            attachment['status']='wrong_altitude'
        elif grid.cell(start) not in grid.free:
            attachment['status']='unsafe_start'
        else:
            for i in sorted(range(len(active)),key=lambda i:(np.linalg.norm(positions[i]-start),i)):
                if np.linalg.norm(positions[i]-start)<=options.edge_radius and grid.segment_free(start,positions[i]):
                    attachment.update(status='attached',node_id=i,node_stable_id=active[i]['stable_id'],
                                      component_id=next(k for k,c in enumerate(components) if i in c));break
    frontiers,frontier_events=_frontiers(adapter,active,state['graph']['frontiers'] if state else [],options.paper_aligned,
        state.get('explored_memory', []) if state else [])
    covered=set().union(*(covered_cells(grid,n) for n in active)) if active else set()
    previous_edges={e['stable_id'] for e in state['graph']['edges']} if state else set()
    current_edges={e['stable_id'] for e in edges}
    graph=dict(schema_version=2,implementation_stage='B_incremental_snapshot',metadata=adapter.metadata,
        algorithm_profile='paper_aligned' if options.paper_aligned else 'coverage_adapted',
        options=asdict(options),nodes=copy.deepcopy(active),edges=edges,components=components,
        robot_attachment=attachment,frontiers=frontiers,
        incremental=dict(step=state['step']+1 if state else 0,frame_key=frame_key,
            node_events=events,frontier_events=frontier_events,
            added_edge_ids=sorted(current_edges-previous_edges),removed_edge_ids=sorted(previous_edges-current_edges),
            dormant_nodes=sum(not n['active'] for n in survivors)),
        reduction=dict(enabled=False,note='Persistent incremental identities; static bootstrap may already be pruned'),
        limitations=['Ordered complete snapshots in one explicit frame only; not a live exploration hook',
                     'Frontier association is geometric, not a certified route into unknown space',
                     'Edges are revalidated each snapshot; no stale edge reuse',
                     'No automatic loop-closure graph deformation'],
        diagnostics=dict(raw_free_cells=len(adapter.raw_free),safe_free_cells=len(cells),occupied_cells=len(grid.occupied),
            node_count=len(active),edge_count=len(edges),component_count=len(components),
            covered_safe_cells=len(covered),uncovered_safe_cells=[list(c) for c in cells if c not in covered],
            sampled_cells=tested,node_budget_reached=len(survivors)>=options.max_nodes,
            frontier_points=len(frontiers),frontier_nodes=sum(n['is_frontier'] for n in active)))
    result=dict(schema_version=1,identity=identity,options=asdict(options),timestamp=timestamp,
                fingerprint=fingerprint,step=graph['incremental']['step'],next_id=next_id,
                nodes=survivors,graph=copy.deepcopy(graph),
                explored_memory=copy.deepcopy(state.get('explored_memory', [])) if state else [])
    return graph,result
