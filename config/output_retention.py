"""Retention for complete generated bundles, confined to their output directory."""
from pathlib import Path
import re
import stat
import warnings


def _regular_tree(directory, files):
    """Require the exact generated layout; reject links, junctions and extra files."""
    directories = {Path('.')}
    for name in files:
        directories.update(Path(name).parents)
    for relative in directories:
        path = directory / relative
        info = path.lstat()
        if (not stat.S_ISDIR(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            return False
        expected = {p.name for p in files | directories if p.parent == relative and p != relative}
        if {p.name for p in path.iterdir()} != expected:
            return False
    for relative in files:
        info = (directory / relative).lstat()
        if (not stat.S_ISREG(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            return False
    return True


def prune_generated_bundles(directory, current, *, pattern, files, keep=2, order=None):
    """Keep the published bundle and recent predecessors; ignore foreign contents.

    Call only after publishing the new complete bundle. Deletion failures must
    not interrupt navigation or discard the successfully published output.
    """
    if keep < 1:
        raise ValueError('At least one bundle must be retained')
    root = Path(directory).resolve()
    current = Path(current).absolute()
    files = {Path(name) for name in files}
    if any(p.is_absolute() or '..' in p.parts for p in files):
        raise ValueError('Bundle filenames must be relative to the bundle')
    removed = []
    try:
        if current.parent.resolve() != root:
            return removed
        current = root / current.name
        if (not re.fullmatch(pattern, current.name)
                or not _regular_tree(current, files)):
            return removed
        # Inspect only immediate children of the explicitly named output root.
        candidates = []
        for path in root.iterdir():
            if path == current or not re.fullmatch(pattern, path.name):
                continue
            if path.resolve().parent != root:
                continue
            try:
                if _regular_tree(path, files):
                    candidates.append(path)
            except (FileNotFoundError, NotADirectoryError):
                continue
        key = order or (lambda path: path.stat().st_mtime_ns)
        candidates.sort(key=key, reverse=True)
        for path in candidates[keep - 1:]:
            # Recheck immediately before deletion; never recurse into unknown data.
            if not _regular_tree(path, files):
                continue
            for relative in files:
                (path / relative).unlink()
            parents = {parent for name in files for parent in name.parents if parent != Path('.')}
            for relative in sorted(parents, key=lambda p: len(p.parts), reverse=True):
                (path / relative).rmdir()
            path.rmdir()
            removed.append(path)
    except OSError as error:
        warnings.warn(f'Could not prune generated bundles in {root}: {error}')
    return removed
