"""One offline command per exploration: create/reuse graph and compose map.html."""
import argparse
from pathlib import Path

from .build_snapshot_graph import main as build_main
from .combine_map import main as combine_main
from .semantic_graph import refresh_semantics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exploration-dir', type=Path, required=True)
    parser.add_argument('--static-rebuild', action='store_true', help='Rebuild without incremental history; archive prior state')
    parser.add_argument('--coverage-adapted', action='store_true', help='Use the earlier coverage-oriented sampling instead of paper sampling')
    parser.add_argument('--dense-edges', action='store_true', help='Keep every safe candidate edge instead of the default sparse graph')
    parser.add_argument('--keep-all-nodes', action='store_true', help='Keep every sampling node rather than compacting the planning graph')
    args = parser.parse_args(argv)
    directory = args.exploration_dir
    graph_directory = directory/'navigation_graph'
    for name in ['map.html', 'semantic_map.json', 'occupancy_2d.npz']:
        if not (directory/name).is_file():
            parser.error(f'Missing saved exploration file: {directory/name}')
    if not graph_directory.exists():
        status = build_main([str(directory/'occupancy_2d.npz'), '--max-samples', '12000',
                             '--coverage-samples', '12000', '--output', str(graph_directory)])
        if status:
            return status
    refresh_semantics(directory, incremental=not args.static_rebuild,
                      paper_aligned=not args.coverage_adapted and not args.static_rebuild,
                      sparse_edges=not args.dense_edges, compact_nodes=not args.keep_all_nodes)
    # The merger checks the graph's source hash before reusing an existing graph.
    return combine_main(['--exploration-dir', str(directory)])


if __name__ == '__main__':
    raise SystemExit(main())
