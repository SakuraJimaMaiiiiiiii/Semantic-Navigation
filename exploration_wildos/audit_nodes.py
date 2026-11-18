"""Offline node-density audit and conservative cleanup preview; no flight imports.

python -m exploration_wildos.audit_nodes SNAPSHOT_DIRECTORY OUTPUT_DIRECTORY
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import numpy as np
from config.semantic_navigation_config import SemanticNavigationConfig
from navigation.route import NavigationGrid
from .map_adapter import MapAdapter
from .node_reduction import covered_cells
from .graph_compaction import compact_graph
from .edge_sparsification import apply_sparsification
from .static_graph import validate_graph
from .vehicle_constraints import validate_vehicles
from .node_lifecycle import retire_compacted_nodes
from .graph_export import export_graph


def audit(source, destination):
    source, destination = Path(source), Path(destination)
    before = json.loads((source/'graph.json').read_text())
    state = json.loads((source/'incremental_state.json').read_text())
    with np.load(source/'planning_grid.npz') as data:
        arrays = {key: data[key].copy() for key in data.files}
    grid = object.__new__(NavigationGrid)
    for name in ('known', 'free', 'occupied'):
        setattr(grid, name, set(map(tuple, arrays[name+'_indices'])))
    grid.resolution = float(arrays['resolution'])
    grid.altitude = float(arrays['altitude_ned'])
    grid.origin = arrays['origin']
    grid.config = SemanticNavigationConfig(**state['identity']['navigation_config'])
    adapter = MapAdapter.from_grid(grid, before['metadata'])
    created = {}
    for path in sorted(source.parent.glob('update_*/graph.json')):
        historical = json.loads(path.read_text())
        for event in historical.get('incremental', {}).get('node_events', []):
            if event['event'] == 'created':
                created.setdefault(event['id'], path.parent.name)
    nodes = before['nodes']
    covers = [covered_cells(grid, node) for node in nodes]
    counts = Counter(cell for cells in covers for cell in cells)
    positions = np.array([node['position_ned'] for node in nodes])
    distances = np.linalg.norm(positions[:,None]-positions[None,:], axis=-1)
    np.fill_diagonal(distances, np.inf)
    details = []
    for i, node in enumerate(nodes):
        j = int(distances[i].argmin())
        details.append(dict(id=node['stable_id'], nearest=nodes[j]['stable_id'],
            distance_m=float(distances[i,j]), unique_cells=sum(counts[c]==1 for c in covers[i]),
            frontier=node['is_frontier'], anchor=node['id']==before['robot_attachment']['node_id'],
            created=created.get(node['stable_id']), nearest_created=created.get(nodes[j]['stable_id'])))
    # Reuse the exact sampling history and geometry, not a new random realization.
    raw = copy.deepcopy(state['graph'])
    for key in ('semantic_nodes', 'semantic_edges', 'vehicle_constraints'):
        if key in before:
            raw[key] = copy.deepcopy(before[key])
    after = compact_graph(raw, grid)
    apply_sparsification(after, after['edges'], max_stretch=1.2)
    after['implementation_stage'] = 'offline_node_cleanup_preview'
    after['metadata'] = dict(after['metadata'], offline_only=True)
    validation = validate_graph(adapter, after)
    validation['vehicles'] = validate_vehicles(after, after.get('semantic_nodes', []), grid.config)
    if not validation['passed'] or not validation['vehicles']['passed']:
        raise RuntimeError('Cleanup preview failed validation')
    next_state = retire_compacted_nodes(state, after)
    kept = {n['stable_id'] for n in after['nodes']}
    for row in details:
        row['removed_in_preview'] = row['id'] not in kept
        row['interpretation'] = ('redundant_after_path_repair' if row['removed_in_preview'] else
            'protected_frontier' if row['frontier'] else 'protected_robot_attachment' if row['anchor'] else
            'unique_safe_cell_coverage' if row['unique_cells'] else 'connectivity_or_path_constraint')
    report = dict(source=str(source), before_nodes=len(nodes), after_nodes=len(after['nodes']),
        before_edges=len(before['edges']), after_edges=len(after['edges']),
        near_pairs_under_1m=int(np.sum(distances<1)/2),
        coverage_before=len(counts), coverage_after=len(set().union(*(covered_cells(grid,n) for n in after['nodes']))),
        components_before=len(before['components']), components_after=len(after['components']),
        retirement=next_state['retirement'], nodes=sorted(details, key=lambda r:r['distance_m']))
    export_graph(destination, adapter, after, validation)
    for name, value in (('audit.json',report), ('incremental_state.json',next_state)):
        (destination/name).write_text(json.dumps(value, ensure_ascii=False, indent=2))
    np.savez_compressed(destination/'planning_grid.npz', **arrays)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    report = audit(args.source, args.destination)
    print(json.dumps({k:v for k,v in report.items() if k not in ('nodes','retirement')}, ensure_ascii=False))
