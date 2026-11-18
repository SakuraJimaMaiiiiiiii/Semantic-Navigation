import json
from pathlib import Path
from concurrent.futures import CancelledError
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from config.output_retention import prune_generated_bundles
from navigation.atomic_file import atomic_write_text


FILES = {'bundle.json', 'private/cache.npz'}


def bundle(root, number):
    path = root / f'revision_{number:012x}'
    (path / 'private').mkdir(parents=True)
    for name in FILES:
        (path / name).write_bytes(b'generated')
    return path


def prune(root, current):
    return prune_generated_bundles(root, current, pattern=r'revision_[0-9a-f]{12}', files=FILES)


def test_nested_bundles_are_bounded_and_foreign_files_are_preserved(tmp_path):
    protected = bundle(tmp_path, 0)
    (protected / 'private/notes.txt').write_text('user data')
    for number in range(1, 10):
        current = bundle(tmp_path, number)
        prune(tmp_path, current)
    assert len(list(tmp_path.iterdir())) == 3
    assert (protected / 'private/notes.txt').read_text() == 'user data'
    assert (current / 'private/cache.npz').is_file()


def test_incomplete_or_external_current_never_triggers_deletion(tmp_path):
    root = tmp_path / 'output'
    previous = [bundle(root, n) for n in range(3)]
    current = bundle(root, 4)
    (current / 'private/cache.npz').unlink()
    assert prune(root, current) == []
    external = bundle(tmp_path / 'elsewhere', 5)
    assert prune(root, external) == []
    assert all(path.exists() for path in previous)


def test_linked_bundle_is_never_deleted(tmp_path):
    external = bundle(tmp_path / 'outside', 0)
    root = tmp_path / 'output'
    root.mkdir()
    link = root / 'revision_000000000000'
    try:
        link.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip('Creating symlinks is unavailable')
    for number in range(1, 5):
        current = bundle(root, number)
        prune(root, current)
    assert link.is_symlink()
    assert (external / 'private/cache.npz').read_bytes() == b'generated'


def test_cleanup_failure_keeps_published_bundle_usable(tmp_path, monkeypatch):
    for number in range(3):
        current = bundle(tmp_path, number)
    original = Path.unlink
    def locked(path, *args, **kwargs):
        if 'revision_000000000000' in path.parts:
            raise PermissionError('file is open')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', locked)
    with pytest.warns(UserWarning, match='Could not prune'):
        prune(tmp_path, current)
    assert (current / 'private/cache.npz').read_bytes() == b'generated'


def test_identical_text_is_not_rewritten(tmp_path, monkeypatch):
    import navigation.atomic_file as module
    path = tmp_path / 'map.html'
    atomic_write_text(path, '地图\n')
    replace = Mock(wraps=module.os.replace)
    monkeypatch.setattr(module.os, 'replace', replace)
    atomic_write_text(path, '地图\n')
    replace.assert_not_called()
    atomic_write_text(path, '更新地图\n')
    replace.assert_called_once()
    assert path.read_text(encoding='utf-8') == '更新地图\n'
    assert list(tmp_path.iterdir()) == [path]


def empty_snapshot():
    return SimpleNamespace(indices=np.empty((0, 3), dtype=int), log_odds=np.empty(0),
        resolution=.3, occupied_threshold=.7, free_threshold=-.2, timestamp=1.,
        trajectory_ned=np.array([[0., 0., -1.5]]))


def test_live_loop_export_retains_readable_latest_two_bundles(tmp_path, monkeypatch):
    from mapping import loop_semantic_replay as module
    monkeypatch.setattr(module, 'replay', lambda *a, **kw: ([], None, 0))
    output = tmp_path / 'corrected'
    for _ in range(6):
        destination, _ = module.export_bundle(tmp_path, output, empty_snapshot(), [1.], [np.eye(4)],
            source={'home_ned': [0., 0., 0.], 'initial_yaw_rad': 0.}, live=True)
    assert len(list(output.glob('revision_*'))) == 2
    assert json.loads((output / 'latest.json').read_text())['directory'] == destination.name
    manifest = json.loads((destination / 'bundle.json').read_text())
    with np.load(destination / manifest['navigation_cache'], allow_pickle=False) as data:
        assert data['indices'].shape == (0, 3)
    assert not list(output.glob('.*'))


def test_cancelled_loop_export_removes_partial_files(tmp_path, monkeypatch):
    from mapping import loop_semantic_replay as module
    from navigation.scene_memory import SceneMemory
    monkeypatch.setattr(module, 'replay', lambda *a, **kw: ([], None, 0))
    def cancelled(self, directory, snapshot):
        directory.mkdir(parents=True)
        (directory / 'partial.npz').write_bytes(b'partial')
        raise CancelledError()
    monkeypatch.setattr(SceneMemory, 'save_bundle', cancelled)
    output = tmp_path / 'corrected'
    with pytest.raises(CancelledError):
        module.export_bundle(tmp_path, output, empty_snapshot(), [1.], [np.eye(4)],
            source={'home_ned': [0., 0., 0.], 'initial_yaw_rad': 0.}, live=True)
    assert not list(output.iterdir())


def test_failed_latest_pointer_does_not_accumulate_orphan_revisions(tmp_path, monkeypatch):
    from mapping import loop_semantic_replay as module
    from navigation import scene_memory
    monkeypatch.setattr(module, 'replay', lambda *a, **kw: ([], None, 0))
    output = tmp_path / 'corrected'
    arguments = dict(source={'home_ned': [0., 0., 0.], 'initial_yaw_rad': 0.}, live=True)
    previous, _ = module.export_bundle(tmp_path, output, empty_snapshot(), [1.], [np.eye(4)], **arguments)
    original = scene_memory.write_json
    def fail_pointer(path, value):
        if Path(path).name == 'latest.json':
            raise OSError('publication failed')
        original(path, value)
    monkeypatch.setattr(scene_memory, 'write_json', fail_pointer)
    for _ in range(3):
        with pytest.raises(OSError, match='publication failed'):
            module.export_bundle(tmp_path, output, empty_snapshot(), [1.], [np.eye(4)], **arguments)
    assert list(output.glob('revision_*')) == [previous]
    assert not list(output.glob('.*'))
    assert json.loads((output / 'latest.json').read_text())['directory'] == previous.name
