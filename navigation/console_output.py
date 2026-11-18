"""Quiet only the foreground navigation task; preserve reports and other threads."""
from contextlib import contextmanager, redirect_stdout
import sys
import threading


class _QuietThreadStream:
    def __init__(self, stream):
        self.stream = stream
        self.owner = threading.get_ident()
        self.active = True

    @property
    def quiet_current_thread(self):
        return ((self.active and threading.get_ident() == self.owner)
                or getattr(self.stream, 'quiet_current_thread', False) is True)

    def write(self, text):
        if self.quiet_current_thread:
            return len(text)
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def quiet_navigation(vehicle=None):
    progress = None
    if vehicle is not None:
        from .terminal_progress import shared_progress
        progress = getattr(vehicle, '_motion_progress_display', None)
        if progress is None:
            progress = shared_progress()
            vehicle._motion_progress_display = progress
    enabled = getattr(progress, 'enabled', True)
    if progress is not None:
        progress.finish()
        progress.enabled = False
    stream = _QuietThreadStream(sys.stdout)
    try:
        with redirect_stdout(stream):
            yield
    finally:
        # Cached progress streams must work normally after this task ends.
        stream.active = False
        if progress is not None:
            progress.enabled = enabled


def navigation_result_text(result, target_id=None, *, command='navigate'):
    title = ('返航' if command == 'return_home' else
             str(target_id or result.get('target_id') or result.get('selected_target_id') or '目标'))
    if result.get('success'):
        return f'{title}：导航成功' if command != 'return_home' else '返航成功'
    reason = ' '.join(str(result.get('reason') or '未完成').split())[:100]
    return f'{title}：导航失败（{reason}）' if command != 'return_home' else f'返航失败（{reason}）'
