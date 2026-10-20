"""Atomic text replacement with bounded retries for Windows sharing locks."""
import os
from pathlib import Path
import tempfile
import time


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name+".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        # Close our handle before rename. Readers/antivirus may briefly retain theirs.
        for delay in (0.02, 0.05, 0.1, None):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if delay is None:
                    raise
                time.sleep(delay)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
