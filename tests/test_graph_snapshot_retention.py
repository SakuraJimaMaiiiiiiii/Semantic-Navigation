from exploration_wildos.snapshot_retention import SNAPSHOT_FILES, prune_graph_snapshots


def snapshot(root, number):
    path = root / f'update_{number:06d}'
    path.mkdir()
    for name in SNAPSHOT_FILES:
        (path / name).write_text('test', encoding='utf-8')
    return path


def test_long_run_retains_two_latest_complete_snapshots(tmp_path):
    for number in range(1, 21):
        current = snapshot(tmp_path, number)
        prune_graph_snapshots(tmp_path, current)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['update_000019', 'update_000020']


def test_failed_save_does_not_prune_previous_snapshots(tmp_path):
    previous = [snapshot(tmp_path, n) for n in range(1, 4)]
    current = snapshot(tmp_path, 4)
    (current / 'planning_grid.npz').unlink()
    prune_graph_snapshots(tmp_path, current)
    assert all(p.exists() for p in previous)


def test_user_files_and_other_directories_are_preserved(tmp_path):
    protected = snapshot(tmp_path, 1)
    (protected / 'notes.txt').write_text('keep')
    unrelated = tmp_path / 'restored'
    unrelated.mkdir()
    for n in range(2, 6):
        current = snapshot(tmp_path, n)
    prune_graph_snapshots(tmp_path, current)
    assert (protected / 'notes.txt').read_text() == 'keep'
    assert len(list(protected.iterdir())) == len(SNAPSHOT_FILES) + 1
    assert unrelated.exists()
    assert not (tmp_path / 'update_000002').exists()
