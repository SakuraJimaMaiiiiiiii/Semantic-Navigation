"""Spawn-safe graph computation. No flight SDK, sensor, runtime or vehicle imports."""
import os
import traceback


def build_live_graph(payload):
    import numpy as np
    from config.semantic_navigation_config import SemanticNavigationConfig
    from navigation.route import NavigationGrid
    from .map_adapter import MapAdapter
    from .static_graph import GraphConfig, validate_graph
    from .incremental_graph import update_graph
    from .graph_compaction import compact_graph
    from .edge_sparsification import apply_sparsification
    from .vehicle_constraints import validate_vehicles

    data = payload['grid']
    # Restore the already constrained snapshot exactly, without eroding twice.
    grid = object.__new__(NavigationGrid)
    grid.known = set(map(tuple, data['known']))
    grid.free = set(map(tuple, data['free']))
    grid.occupied = set(map(tuple, data['occupied']))
    grid.origin = np.asarray(data['origin'])
    grid.resolution, grid.altitude = data['resolution'], data['altitude']
    grid.config = SemanticNavigationConfig(**data['config'])
    adapter = MapAdapter.from_grid(grid, payload['metadata'])
    graph, state = update_graph(adapter, frame_key=payload['frame_key'],
        state=payload['state'], options=GraphConfig(**payload['options']), start=payload['start'])
    graph = compact_graph(graph, grid)
    apply_sparsification(graph, graph['edges'], max_stretch=1.2)
    graph['semantic_nodes'] = payload['objects']
    graph['semantic_edges'] = []
    if payload.get('vehicle_constraints') is not None:
        graph['vehicle_constraints'] = payload['vehicle_constraints']
    graph['implementation_stage'] = 'live_frontier_exploration'
    graph['limitations'] = ['Geometric frontier gain/distance scoring; no ExploRFM',
        'Online sampling batch is smaller than paper 1000 draws to bound startup latency',
        'Fixed-height PX4 frame; graph updated at decision/replan boundaries and asynchronously during motion',
        'Compaction and sparse edges are project adaptations, not paper Algorithm 5']
    validation = validate_graph(adapter, graph)
    validation['vehicles'] = validate_vehicles(graph, payload['objects'], grid.config)
    if not validation['passed'] or not validation['vehicles']['passed']:
        raise RuntimeError('Live graph validation failed; motion withheld')
    if payload['options'].get('incremental_cleanup', False):
        from .node_lifecycle import retire_compacted_nodes
        state = retire_compacted_nodes(state, graph)
        graph['node_retirement'] = state['retirement']
        graph['limitations'].append('Same-batch rejection and retired-node memory are project adaptations')
    return graph, state, validation


def graph_process(connection, payload):
    # Root launcher imports only runpy; under Windows spawn the numerical stack
    # is first loaded here, after capping its internal worker counts.
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    try:
        connection.send(('ok', build_live_graph(payload)))
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()
