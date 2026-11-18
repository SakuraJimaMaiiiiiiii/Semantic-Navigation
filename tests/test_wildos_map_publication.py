from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.request import urlopen
import pytest

from exploration_wildos.map_publication import publish_map_page
from exploration_wildos.live_map import LiveMapServer
import navigation.atomic_file as atomic


def test_windows_replace_denial_preserves_live_page_and_retries_later(tmp_path, monkeypatch):
    path=tmp_path/'map.html'
    path.write_text('old complete page')
    session=SimpleNamespace(map_path=path, live_map=Mock())
    original=atomic.os.replace
    attempts=[]
    def sharing_lock(source,destination):
        if Path(destination)==path:
            attempts.append(source)
            raise PermissionError(5,'Windows sharing lock')
        return original(source,destination)
    monkeypatch.setattr(atomic.os,'replace',sharing_lock)
    monkeypatch.setattr(atomic.time,'sleep',lambda _:None)
    assert not publish_map_page(session,'new complete page')
    assert len(attempts)==4
    assert path.read_text()=='old complete page'
    assert (tmp_path/'map.pending.html').read_text()=='new complete page'
    session.live_map.set_page.assert_called_with('new complete page')
    assert not list(tmp_path.glob('*.tmp'))
    monkeypatch.setattr(atomic.os,'replace',original)
    assert publish_map_page(session,'latest complete page')
    assert path.read_text()=='latest complete page'
    assert not (tmp_path/'map.pending.html').exists()
    assert not session._map_write_blocked


def test_http_reads_cached_page_even_when_disk_is_locked(tmp_path,monkeypatch):
    path=tmp_path/'map.html';path.write_text('<html>old</html>')
    server=LiveMapServer(path)
    try:
        def deny(*args,**kwargs):
            raise PermissionError('Disk page must not be opened by HTTP reader')
        monkeypatch.setattr(Path,'read_text',deny)
        server.set_page('<html>new complete page</html>')
        with urlopen(server.url) as response:
            html=response.read().decode()
        assert 'new complete page' in html and 'wildos-live-status' in html
    finally:
        server.close()


def test_unrelated_storage_error_is_not_misreported_as_file_lock(tmp_path,monkeypatch):
    import exploration_wildos.map_publication as publication
    monkeypatch.setattr(publication,'atomic_write_text',Mock(side_effect=OSError('disk full')))
    with pytest.raises(OSError,match='disk full'):
        publish_map_page(SimpleNamespace(map_path=tmp_path/'map.html'),'data')
