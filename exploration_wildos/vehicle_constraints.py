"""Vehicle-owned planning footprints; observed geometry is never modified."""
import copy
import numpy as np
from .map_adapter import MapAdapter


VEHICLE_HORIZONTAL_MARGIN = 0.20
VEHICLE_POLICY = "no_overflight_v1"


def horizontal_margin(config):
    """Vehicle semantic uncertainty buffer, separate from occupancy/body clearance."""
    return min(config.clearance_radius, VEHICLE_HORIZONTAL_MARGIN)


def expanded_boxes(objects, config):
    horizontal = horizontal_margin(config)
    margin = np.array([horizontal, horizontal, config.vertical_half_extent])
    result=[]
    for obj in objects:
        low, high = (np.asarray(obj['bbox_3d'][key], dtype=float) for key in ('min','max'))
        if low.shape!=(3,) or high.shape!=(3,) or not np.isfinite([low,high]).all() or np.any(low>high):
            raise ValueError('Invalid vehicle box')
        result.append((obj['id'], low-margin, high+margin))
    return result


def segment_hits_box(a, b, low, high):
    """Closed slab intersection, including tangency and zero-length segments."""
    a,b=np.asarray(a),np.asarray(b)
    enter,leave=0.,1.
    for k in range(len(a)):
        delta=b[k]-a[k]
        if abs(delta)<1e-12:
            if a[k]<low[k]-1e-9 or a[k]>high[k]+1e-9:
                return False
        else:
            t0,t1=sorted(((low[k]-a[k])/delta,(high[k]-a[k])/delta))
            enter=max(enter,t0);leave=min(leave,t1)
            if enter>leave+1e-9:
                return False
    return True


def segment_hits_vehicle(a, b, low, high):
    """An observed vehicle footprint is forbidden at EVERY flight altitude."""
    return segment_hits_box(np.asarray(a)[:2], np.asarray(b)[:2], low[:2], high[:2])


def footprint_cells(low, high, resolution, origin):
    """All cells touching the closed XY footprint, including its edge."""
    first = np.ceil((np.asarray(low)[:2]-origin)/resolution-1e-9).astype(int)-1
    last = np.floor((np.asarray(high)[:2]-origin)/resolution+1e-9).astype(int)
    if np.prod(last-first+1) > 1_000_000:
        raise ValueError('Vehicle footprint exceeds planning cell budget')
    return {(x,y) for x in range(first[0],last[0]+1) for y in range(first[1],last[1]+1)}


def constrain_adapter(adapter, objects):
    """Fill vehicle-owned planning obstacles without feeding them to perception.

    Occupied footprints are height-independent. Buffer cells are attributed to
    their vehicle as planning constraints, NOT measured semantic voxel labels.
    No object centers, observed bounds or source occupancy arrays are mutated.
    """
    grid=copy.copy(adapter.grid)
    boxes=expanded_boxes(objects, grid.config)
    owners={};body=set()
    for obj,(identity,low,high) in zip(objects,boxes):
        footprint=footprint_cells(obj['bbox_3d']['min'],obj['bbox_3d']['max'],grid.resolution,grid.origin)
        body.update(footprint)
        for cell in footprint_cells(low,high,grid.resolution,grid.origin):
            owners.setdefault(cell,[]).append(dict(vehicle_id=identity,
                zone='observed_footprint' if cell in footprint else 'clearance_buffer'))
    blocked=set(owners)
    grid.free=grid.free-blocked
    grid.occupied=grid.occupied|blocked
    grid.known=set(grid.known)|blocked
    metadata=dict(adapter.metadata, vehicle_geometry_constraints=True,
                  vehicle_constraint_policy=VEHICLE_POLICY,
                  vehicle_horizontal_margin_m=horizontal_margin(grid.config))
    constrained=MapAdapter(grid,metadata,frozenset(adapter.raw_free-blocked))
    report=dict(policy=VEHICLE_POLICY, forbid_overflight=True,
        horizontal_margin_m=horizontal_margin(grid.config),
        vertical_margin_m=grid.config.vertical_half_extent, height_filter_applied=False,
        active_vehicle_ids=[identity for identity,_,_ in boxes],
        safe_cells_before=len(adapter.grid.free), safe_cells_after=len(grid.free),
        excluded_safe_cells=[list(c) for c in sorted(adapter.grid.free-grid.free)],
        footprint_cells=[list(c) for c in sorted(body)],
        occupied_cells=[list(c) for c in sorted(blocked)],
        cell_owners=[dict(cell=list(c),owners=owners[c]) for c in sorted(owners)],
        ownership_evidence='Derived planning footprint/buffer ownership, not relabeled sensor evidence',
        localization_modified=False,
        evidence='Observed bounds define the footprint; no inference of complete hidden body dimensions')
    grid.vehicle_constraint_report=report
    return constrained,report


def validate_vehicles(graph, objects, config):
    boxes=expanded_boxes(objects,config)
    nodes=graph['nodes']
    def hits(a,b):
        return [identity for identity,low,high in boxes if segment_hits_vehicle(a,b,low,high)]
    invalid_nodes=[n['id'] for n in nodes if hits(n['position_ned'],n['position_ned'])]
    invalid_edges=[e['id'] for e in graph['edges'] if hits(nodes[e['source']]['position_ned'],nodes[e['target']]['position_ned'])]
    start=graph['robot_attachment'];invalid_attachment=False
    if start.get('status')=='attached':
        invalid_attachment=bool(hits(start['position_ned'],nodes[start['node_id']]['position_ned']))
    for n in nodes:
        p=np.asarray(n['position_ned'])
        n['above_vehicle_ids']=[identity for identity,low,high in boxes
                                if np.all(p[:2]>=low[:2]) and np.all(p[:2]<=high[:2]) and p[2]<low[2]]
    return dict(passed=not invalid_nodes and not invalid_edges and not invalid_attachment,
                checked_nodes=len(nodes),checked_edges=len(graph['edges']),invalid_nodes=invalid_nodes,
                invalid_edges=invalid_edges,invalid_robot_attachment=invalid_attachment)
