"""Bound idle inference load while keeping active navigation perception responsive."""
from functools import wraps
from .console_output import quiet_navigation


def active_perception(method):
    @wraps(method)
    def wrapped(session, *args, **kwargs):
        worker = getattr(getattr(session, 'runtime', None), 'detection_worker', None)
        set_idle = getattr(worker, 'set_idle_mode', None)
        with quiet_navigation(getattr(getattr(session, 'runtime', None), 'vehicle', None)):
            if callable(set_idle):
                set_idle(False)
            try:
                return method(session, *args, **kwargs)
            finally:
                if callable(set_idle):
                    set_idle(True)
    return wrapped
