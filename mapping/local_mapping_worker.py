"""在独立线程中消费同步观测并更新局部三维占据栅格。"""

from __future__ import annotations

import threading
import time


class LocalMappingWorker:
    """使建图、HDF5记录和飞行控制能够同时运行。"""

    def __init__(
        self,
        observation_source,
        grid,
        update_rate: float = 5.0,
        observation_timeout: float = 1.0,
    ) -> None:
        self.observation_source = observation_source
        self.grid = grid
        self.update_rate = float(update_rate)
        self.observation_timeout = float(observation_timeout)
        self.update_count = 0
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        if self.update_rate <= 0.0:
            raise ValueError("update_rate must be greater than zero.")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._mapping_loop,
            name="LocalOccupancyMapping",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, self.observation_timeout + 1.0))

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Local occupancy mapping failed.") from self._error

    def _mapping_loop(self) -> None:
        period = 1.0 / self.update_rate
        next_update = time.monotonic()
        try:
            while not self._stop.is_set():
                try:
                    observation = self.observation_source.get_observation(
                        timeout=self.observation_timeout
                    )
                except TimeoutError:
                    continue
                self.grid.update(observation)
                self.update_count += 1

                next_update += period
                wait_time = next_update - time.monotonic()
                if wait_time > 0.0:
                    self._stop.wait(wait_time)
                else:
                    next_update = time.monotonic()
        except Exception as error:
            if not self._stop.is_set():
                self._error = error
                print(f"--- WARNING: local occupancy mapping stopped: {error} ----")
