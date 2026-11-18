"""Bound generated live graph history without deleting user-added artifacts."""
from config.output_retention import prune_generated_bundles


SNAPSHOT_FILES = frozenset({
    'graph.json', 'validation.json', 'graph.html',
    'incremental_state.json', 'planning_grid.npz',
})


def prune_graph_snapshots(directory, current, keep=2):
    """Prune only older, complete generated snapshots after a successful save."""
    return prune_generated_bundles(directory, current, pattern=r'update_\d{6,}',
        files=SNAPSHOT_FILES, keep=keep, order=lambda path: int(path.name[7:]))
