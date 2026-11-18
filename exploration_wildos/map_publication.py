"""Best-effort HTML persistence; a Windows display-file lock is not a flight fault."""
from navigation.atomic_file import atomic_write_text


def publish_map_page(session, content):
    viewer = getattr(session, 'live_map', None)
    if viewer is not None:
        viewer.set_page(content)
    pending = session.map_path.with_name('map.pending.html')
    try:
        atomic_write_text(session.map_path, content)
    except PermissionError as error:
        if not getattr(session, '_map_write_blocked', False):
            print(f'--- 地图文件被占用，实时网页继续更新；稍后检查点将重试保存：{error} ----')
        session._map_write_blocked = True
        try:
            atomic_write_text(pending, content)
        except PermissionError as pending_error:
            print(f'--- 地图备用快照也无法写入：{pending_error}；实时网页保留最新画面 ----')
        return False
    session._map_write_blocked = False
    try:
        pending.unlink(missing_ok=True)
    except PermissionError:
        pass  # Old fallback cleanup must not abort navigation either.
    return True
