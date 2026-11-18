"""Add a zoomable static navigation graph below a saved exploration map."""
import argparse
import hashlib
import json
from pathlib import Path

from exploration_wildos.combined_map import compose_map_html
from navigation.atomic_file import atomic_write_text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exploration-dir', type=Path, required=True)
    parser.add_argument('--graph-dir', type=Path, help='Defaults to exploration-dir/navigation_graph')
    args = parser.parse_args(argv)
    directory = args.exploration_dir
    graph_directory = args.graph_dir or directory/'navigation_graph'
    destination = directory/'map.html'
    backup = directory/'map.exploration.original.html'
    try:
        meta = json.loads((directory/'semantic_map.json').read_text(encoding='utf-8'))
        graph = json.loads((graph_directory/'graph.json').read_text(encoding='utf-8'))
        matching = [meta[k] for k in ['occupancy_2d','occupancy_3d']
                    if meta[k]['sha256'] == graph['metadata']['source_sha256']]
        if not matching or hashlib.sha256((directory/matching[0]['file']).read_bytes()).hexdigest() != matching[0]['sha256']:
            raise ValueError('Graph and exploration map must reference the same occupancy archive')
        current = destination.read_text(encoding='utf-8')
        original = backup.read_text(encoding='utf-8') if 'id="wildos-combined-style"' in current else current
        content = compose_map_html(original, (graph_directory/'graph.html').read_text(encoding='utf-8'),
                                   graph, meta['home_ned'], meta['initial_yaw_rad'])
        # If the existing renderer has produced a fresh map since the last merge,
        # preserve the previous original by content hash before updating the backup.
        if backup.exists() and 'id="wildos-combined-style"' not in current:
            previous = backup.read_bytes()
            if previous != destination.read_bytes():
                archived = backup.with_name('map.exploration.'+hashlib.sha256(previous).hexdigest()[:12]+'.html')
                if not archived.exists():
                    with archived.open('xb') as stream:
                        stream.write(previous)
                atomic_write_text(backup, original)
        # Back up exact bytes once, including original newline conventions.
        if not backup.exists():
            with backup.open('xb') as stream:
                stream.write(destination.read_bytes())
        atomic_write_text(destination, content)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({'map':str(destination.resolve()),'original_backup':str(backup.resolve())},ensure_ascii=False))


if __name__ == '__main__':
    main()
