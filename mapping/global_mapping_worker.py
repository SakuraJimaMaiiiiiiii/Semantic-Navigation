"""在独立线程中更新、显示并保存全局稀疏地图。"""

from __future__ import annotations

import threading
import time


class GlobalMappingWorker:
    """消费同步观测；关闭时可把累积地图保存到HDF5。"""

    def __init__(
        self,
        observation_source,
        global_map,
        config,
        visualizer=None,
        loop_closure=None,
        observation_timeout: float = 1.0,
    ) -> None:
        self.observation_source = observation_source
        self.global_map = global_map
        self.config = config
        self.visualizer = visualizer
        self.loop_closure = loop_closure
        self.observation_timeout = float(observation_timeout)
        self.update_count = 0
        self.output_path = None
        self.foxglove_output_path = None
        self._stop = threading.Event()
        self._thread = None
        self._error = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._error = None
        if self.visualizer is not None:
            self.visualizer.start()
        self._thread = threading.Thread(
            target=self._mapping_loop,
            name="GlobalSparseOccupancyMapping",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, self.observation_timeout + 1.0))
        if self.visualizer is not None:
            self.visualizer.close()
        if (
            self.config.save_on_close
            and self.update_count > 0
            and self._error is None
        ):
            try:
                self.output_path = self.global_map.save()
                if self.loop_closure is not None:
                    self.loop_closure.write_hdf5_metadata(
                        self.output_path
                    )
                if self.config.export_foxglove_mcap_on_close:
                    from tools.export_foxglove_voxel_map import export

                    snapshot = self.global_map.snapshot()
                    temporary_ply = self.output_path.with_name(
                        f".{self.output_path.stem}_foxglove_conversion.ply"
                    )
                    self.foxglove_output_path = self.output_path.with_name(
                        f"{self.output_path.stem}_foxglove.mcap"
                    )
                    try:
                        self.global_map.export_dense_occupied_ply(
                            temporary_ply,
                            subdivisions=self.config.dense_export_subdivisions,
                            max_points=self.config.dense_export_max_points,
                        )
                        export(
                            temporary_ply,
                            self.output_path,
                            self.foxglove_output_path,
                            resolution=snapshot.resolution,
                            drone_down=float(
                                snapshot.drone_position_ned[2]
                            ),
                        )
                    finally:
                        temporary_ply.unlink(missing_ok=True)
            except Exception as error:
                self._error = error

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                f"Global sparse mapping failed: {self._error!r}"
            ) from self._error
        if self.visualizer is not None:
            self.visualizer.raise_if_failed()

    def _mapping_loop(self) -> None:
        period = 1.0 / float(self.config.update_rate)
        next_update = time.monotonic()
        try:
            while not self._stop.is_set():
                try:
                    observation = self.observation_source.get_observation(
                        timeout=self.observation_timeout
                    )
                except TimeoutError:
                    continue
                if self.loop_closure is None:
                    integrated = True
                    self.global_map.update(observation)
                else:
                    integrated = self.loop_closure.process(observation)
                if integrated:
                    self.update_count += 1
                self.global_map.mark_observation_completed()

                next_update += period
                wait_time = next_update - time.monotonic()
                if wait_time > 0.0:
                    self._stop.wait(wait_time)
                else:
                    next_update = time.monotonic()
        except Exception as error:
            if not self._stop.is_set():
                self._error = error
                print(f"--- WARNING: global sparse mapping stopped: {error} ----")
