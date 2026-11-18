"""Bounded background graph computation; never calls flight-control APIs."""
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from .graph_worker import graph_process


class MovingGraph:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='WildOSGraphWait')
        self.cancel = threading.Event()
        self.future = None

    def build(self, payload):
        context = multiprocessing.get_context('spawn')
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=graph_process, args=(sender, payload), daemon=True)
        try:
            process.start()
            sender.close()
            started = time.monotonic()
            while not self.cancel.is_set():
                if receiver.poll(.05):
                    status, result = receiver.recv()
                    if status != 'ok':
                        raise RuntimeError(result)
                    return result
                if not process.is_alive():
                    raise RuntimeError('Moving graph worker exited without result')
                if time.monotonic()-started > 30:
                    raise TimeoutError('Moving graph build exceeded 30 seconds')
            raise RuntimeError('Moving graph build cancelled')
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
                process.close()
            receiver.close()
            sender.close()

    def submit(self, pending, grid, current, frame, objects):
        def work():
            changed = pending.refresh(grid, current, frame_key=frame,
                                      objects=objects, builder=self.build)
            return changed, pending
        self.future = self.executor.submit(work)

    def submit_from_maps(self, pending, global_map, local_map, current, frame,
                         objects, altitude, config, origin, max_radius):
        # Snapshot APIs are already used by the local planning worker. Keep their
        # copying and grid projection off the flight health-check callback.
        def work():
            import numpy as np
            from navigation.route import NavigationGrid
            from .map_adapter import MapAdapter
            from .vehicle_constraints import constrain_adapter
            started = time.monotonic()
            grid = NavigationGrid.from_global(global_map.snapshot(), altitude, config,
                local_snapshot=local_map.snapshot(), current=current)
            grid = constrain_adapter(MapAdapter.from_grid(grid), objects)[0].grid
            if origin is not None:
                grid.free = {c for c in grid.free if np.linalg.norm(
                    grid.point(c)[:2]-origin[:2]) <= max_radius}
            preparation_seconds = time.monotonic()-started
            if self.cancel.is_set():
                raise RuntimeError('Moving graph build cancelled')
            changed = pending.refresh(grid, current, frame_key=frame,
                                      objects=objects, builder=self.build)
            pending.preparation_seconds = preparation_seconds
            return changed, pending
        self.future = self.executor.submit(work)

    def close(self):
        self.cancel.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
