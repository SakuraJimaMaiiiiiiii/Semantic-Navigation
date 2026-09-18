"""将单一同步RGB-D数据流扇出给记录器和建图器。"""

from __future__ import annotations

import queue
import threading


_CLOSED = object()


class ObservationSubscription:
    """一个具有独立缓冲队列的同步观测订阅端。"""

    def __init__(self, name: str, max_queue: int) -> None:
        self.name = str(name)
        self._queue = queue.Queue(maxsize=int(max_queue))
        self._hub_error = None
        self.dropped_count = 0

    def get_observation(self, timeout: float = 1.0):
        """返回下一组同步观测，接口与SynchronizedRGBDStream一致。"""
        try:
            value = self._queue.get(timeout=float(timeout))
        except queue.Empty as error:
            raise TimeoutError(
                f"No observation for subscriber {self.name!r}."
            ) from error
        if value is _CLOSED:
            if self._hub_error is not None:
                raise RuntimeError("Observation hub stopped unexpectedly.") from (
                    self._hub_error
                )
            raise RuntimeError("Observation hub is closed.")
        return value

    def _publish(self, observation) -> None:
        try:
            self._queue.put_nowait(observation)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
            self.dropped_count += 1
        except queue.Empty:
            pass
        self._queue.put_nowait(observation)

    def _close(self, error=None) -> None:
        self._hub_error = error
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(_CLOSED)
        except queue.Full:
            pass


class SynchronizedObservationHub:
    """只读取一次相机流，再把同一观测发布给多个运行模块。"""

    def __init__(self, source, observation_timeout: float = 1.0) -> None:
        self.source = source
        self.observation_timeout = float(observation_timeout)
        self._subscriptions = []
        self._stop = threading.Event()
        self._thread = None
        self._error = None

    def subscribe(
        self,
        name: str,
        max_queue: int = 4,
    ) -> ObservationSubscription:
        """在启动前创建订阅端；队列满时丢弃最旧帧以保持实时性。"""
        if self._thread is not None:
            raise RuntimeError("Create subscriptions before starting the hub.")
        if max_queue <= 0:
            raise ValueError("max_queue must be greater than zero.")
        subscription = ObservationSubscription(name, max_queue)
        self._subscriptions.append(subscription)
        return subscription

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self._subscriptions:
            raise RuntimeError("Observation hub has no subscribers.")
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._publish_loop,
            name="SynchronizedObservationHub",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, self.observation_timeout + 1.0))
        for subscription in self._subscriptions:
            subscription._close(self._error)

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Observation hub failed.") from self._error

    def _publish_loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    observation = self.source.get_observation(
                        timeout=self.observation_timeout
                    )
                except TimeoutError:
                    continue
                for subscription in self._subscriptions:
                    subscription._publish(observation)
        except Exception as error:
            if not self._stop.is_set():
                self._error = error
        finally:
            for subscription in self._subscriptions:
                subscription._close(self._error)
