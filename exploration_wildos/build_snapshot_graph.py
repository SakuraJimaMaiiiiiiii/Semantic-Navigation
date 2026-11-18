"""Build a static, offline WildOS-style graph from a saved occupancy NPZ."""
import argparse
import json
from pathlib import Path


from exploration_wildos.map_adapter import load_map
from exploration_wildos.static_graph import GraphConfig, build_graph, validate_graph
from exploration_wildos.graph_export import export_graph


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map', type=Path, help='navigation_cache.npz, global occupancy NPZ, or occupancy_2d.npz')
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing directories are refused')
    parser.add_argument('--altitude', type=float, help='NED z in metres; required for 3-D archives')
    parser.add_argument('--clearance', type=float, help='Horizontal grid clearance; 3-D default 0.40 m')
    parser.add_argument('--vertical-half-extent', type=float, help='Vertical body half extent; 3-D default 0.25 m')
    parser.add_argument('--start-ned', nargs=3, type=float, help='Optional robot position for certified graph attachment')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=2000)
    parser.add_argument('--coverage-samples', type=int, default=2000)
    parser.add_argument('--max-nodes', type=int, default=1000)
    parser.add_argument('--free-radius-cap', type=float, default=2.)
    parser.add_argument('--edge-radius', type=float, default=3.)
    parser.add_argument('--max-dense-cells', type=int, default=4_000_000)
    parser.add_argument('--keep-redundant-nodes', action='store_true', help='Disable conservative node reduction for comparison')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    try:
        options = GraphConfig(seed=args.seed, max_samples=args.max_samples, coverage_samples=args.coverage_samples,
                              max_nodes=args.max_nodes, free_radius_cap=args.free_radius_cap,
                              edge_radius=args.edge_radius, max_dense_cells=args.max_dense_cells,
                              prune_redundant=not args.keep_redundant_nodes)
        adapter = load_map(args.map, altitude=args.altitude, clearance=args.clearance,
                           vertical_half_extent=args.vertical_half_extent)
        graph = build_graph(adapter, options, args.start_ned)
        validation = validate_graph(adapter, graph)
        export_graph(args.output, adapter, graph, validation)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(dict(output=str(args.output.resolve()), nodes=len(graph['nodes']), edges=len(graph['edges']),
                          components=len(graph['components']), safety_validation=validation['passed'],
                          robot_attachment=graph['robot_attachment']['status']), ensure_ascii=False))
    return 0 if validation['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
