"""Reproduce the real-map consistency/connectivity audit without flight side effects."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

from exploration_wildos.map_adapter import load_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exploration', type=Path, required=True)
    parser.add_argument('--graph-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.exploration
    meta = json.loads((directory/'semantic_map.json').read_text(encoding='utf-8'))
    report = json.loads((directory/'exploration_report.json').read_text(encoding='utf-8'))
    graph = json.loads((args.graph_dir/'graph.json').read_text(encoding='utf-8'))
    a = load_map(directory/'occupancy_2d.npz')
    b = load_map(directory/'private/navigation_cache.npz', altitude=a.grid.altitude,
                 clearance=a.grid.config.clearance_radius,
                 vertical_half_extent=a.grid.config.vertical_half_extent)
    with np.load(directory/'occupancy_2d.npz', allow_pickle=False) as data:
        saved_safe = set(map(tuple, data['safe_indices'].tolist()))
    with np.load(directory/'private/navigation_cache.npz', allow_pickle=False) as data:
        start = json.loads(data['payload'].tobytes())['places']['place_0000']['position']
    checks = dict(raw_free_matches_3d=a.raw_free == b.raw_free,
                  occupied_matches_3d=a.grid.occupied == b.grid.occupied,
                  safe_free_matches_3d=a.grid.free == b.grid.free,
                  safe_free_matches_saved=saved_safe == a.grid.free,
                  graph_source_matches=a.metadata['source_sha256'] == graph['metadata']['source_sha256'])
    for name in ['occupancy_2d', 'occupancy_3d']:
        entry = meta[name]
        checks[name+'_checksum'] = hashlib.sha256((directory/entry['file']).read_bytes()).hexdigest() == entry['sha256']

    if graph.get('vehicle_constraints'):
        from exploration_wildos.vehicle_constraints import constrain_adapter, validate_vehicles
        from exploration_wildos.node_reduction import covered_cells
        a, constraints = constrain_adapter(a, graph['semantic_nodes'])
        checks['semantic_source_matches'] = hashlib.sha256((directory/'semantic_map.json').read_bytes()).hexdigest() == graph['semantic_source_sha256']
        checks['constraint_cells_match'] = constraints['excluded_safe_cells'] == graph['vehicle_constraints']['excluded_safe_cells']
        checks['vehicle_geometry_passed'] = validate_vehicles(graph, graph['semantic_nodes'], a.grid.config)['passed']
        covered = set().union(*(covered_cells(a.grid,n) for n in graph['nodes']))
        checks['safe_cell_coverage_matches'] = len(covered)==graph['diagnostics']['covered_safe_cells']
        checks['uncovered_cells_match'] = {tuple(c) for c in graph['diagnostics']['uncovered_safe_cells']} == a.grid.free-covered

    if graph.get('incremental'):
        state = json.loads((args.graph_dir/'incremental_state.json').read_text())
        ids = [n['stable_id'] for n in graph['nodes']]
        frontier_ids = {f['id'] for f in graph['frontiers']}
        checks['incremental_state_matches_graph'] = state['graph'] == graph
        if graph.get('edge_sparsification'):
            from exploration_wildos.static_graph import _components
            candidates = state['candidate_edges']
            candidate_by_id = {e['stable_id']:e for e in candidates}
            checks['sparse_edges_are_certified_subset'] = all(e['stable_id'] in candidate_by_id and all(e[k]==candidate_by_id[e['stable_id']][k] for k in ('source','target','length')) for e in graph['edges'])
            checks['sparse_components_preserved'] = _components(len(graph['nodes']),candidates)==graph['components']
        if graph.get('node_compaction'):
            sampling = state['uncompacted_graph']
            samples_by_id = {n['stable_id']:n for n in sampling['nodes']}
            checks['compact_nodes_keep_geometry_and_history'] = all(all(n[k]==samples_by_id[n['stable_id']][k] for k in ('position_ned','free_radius','explored_radius')) for n in graph['nodes'])
            checks['sampling_memory_retained'] = {n['stable_id'] for n in sampling['nodes']} <= {n['stable_id'] for n in state['nodes']}
            checks['compaction_coverage_preserved'] = graph['diagnostics']['covered_safe_cells']==sampling['diagnostics']['covered_safe_cells']
            checks['frontier_owners_protected'] = {f['node_stable_id'] for f in sampling['frontiers']} <= set(ids)
        checks['stable_node_ids_unique'] = len(ids)==len(set(ids))
        checks['frontier_cells_unknown'] = all(tuple(f['cell']) not in a.grid.known for f in graph['frontiers'])
        checks['frontier_owners_valid'] = all(f['node_stable_id']==graph['nodes'][f['node_id']]['stable_id'] for f in graph['frontiers'])
        checks['frontier_node_flags_consistent'] = all(n['is_frontier']==bool(n['frontier_ids']) and set(n['frontier_ids'])<=frontier_ids for n in graph['nodes'])

    # Exact cell connectivity under the grid's 8-neighbor/no-corner-cut rules.
    left, labels, sizes = set(a.grid.free), {}, []
    while left:
        seed = min(left)
        left.remove(seed)
        queue, count = deque([seed]), 0
        while queue:
            x, y = queue.popleft()
            labels[(x, y)] = len(sizes)
            count += 1
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cell = (x+dx, y+dy)
                    if cell not in left:
                        continue
                    if dx and dy and ((x+dx, y) not in a.grid.free or (x, y+dy) not in a.grid.free):
                        continue
                    left.remove(cell)
                    queue.append(cell)
        sizes.append(count)
    mapping = [sorted({labels[tuple(graph['nodes'][i]['cell'])] for i in group})
               for group in graph['components']]
    checks['one_graph_component_per_grid_component'] = (
        all(len(group) == 1 for group in mapping)
        and len(mapping) == len(sizes)
        and sorted(group[0] for group in mapping) == list(range(len(sizes))))
    probes = []
    for group in graph['components']:
        goal = graph['nodes'][group[0]]['position_ned']
        path = a.grid.path(start, goal)
        probes.append(dict(node_id=group[0], astar_path_found=path is not None,
                           same_grid_component=labels.get(a.grid.cell(start)) == labels[a.grid.cell(goal)]))
    result = dict(checks=checks, all_checks_passed=all(checks.values()), source=str(directory),
                  exploration_summary={k: report[k] for k in ['reason', 'frontiers_exhausted', 'elapsed_seconds',
                                                               'viewpoint_count', 'blocked_frontiers']},
                  returned_home=meta['returned_home'], coordinate_frame=meta['coordinate_frame'],
                  grid_component_sizes=sizes, graph_component_sizes=list(map(len, graph['components'])),
                  graph_to_grid_components=mapping, astar_probes=probes,
                  unknown_note='A missing A* path alone may also mean a search budget limit; grid BFS is exhaustive.')
    (args.graph_dir/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
