"""异步保存局部和全局A*二维规划图。"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from planner.vertical_projection import ned_height_slice_mask


@dataclass(frozen=True)
class AStarVisualizationConfig:
    """A*二维规划图的保存参数。"""

    enabled: bool = True
    local_save_interval: float = 2.0
    local_above_height: float = 0.25
    local_below_height: float = 0.9
    image_dpi: int = 140
    local_cell_pixels: int = 16
    local_max_image_pixels: int = 2400
    global_cell_pixels: int = 7
    global_max_image_pixels: int = 2400
    global_max_grid_cells: int = 2_000_000
    pending_image_limit: int = 8
    output_root: str | Path | None = None

    def __post_init__(self) -> None:
        if float(self.local_save_interval) <= 0.0:
            raise ValueError("local_save_interval must be greater than zero.")
        if float(self.local_above_height) <= 0.0:
            raise ValueError("local_above_height must be greater than zero.")
        if float(self.local_below_height) <= 0.0:
            raise ValueError("local_below_height must be greater than zero.")
        if int(self.image_dpi) <= 0:
            raise ValueError("image_dpi must be greater than zero.")
        if int(self.local_cell_pixels) <= 0:
            raise ValueError("local_cell_pixels must be greater than zero.")
        if int(self.local_max_image_pixels) < 800:
            raise ValueError(
                "local_max_image_pixels must be at least 800."
            )
        if int(self.global_cell_pixels) <= 0:
            raise ValueError("global_cell_pixels must be greater than zero.")
        if int(self.global_max_image_pixels) < 800:
            raise ValueError(
                "global_max_image_pixels must be at least 800."
            )
        if int(self.global_max_grid_cells) <= 0:
            raise ValueError(
                "global_max_grid_cells must be greater than zero."
            )
        if int(self.pending_image_limit) <= 0:
            raise ValueError("pending_image_limit must be greater than zero.")


@dataclass(frozen=True)
class _LocalPlotTask:
    sequence: int
    source_name: str
    timestamp: datetime
    origin_ned: np.ndarray
    resolution: float
    unknown: np.ndarray
    occupied: np.ndarray
    inflated: np.ndarray
    current_ned: np.ndarray
    global_target_ned: np.ndarray
    local_target_ned: np.ndarray | None
    path_ned: np.ndarray
    status: str
    speed_limit: float


@dataclass(frozen=True)
class _GlobalPlotTask:
    sequence: int
    timestamp: datetime
    source_name: str
    current_ned: np.ndarray
    goal_ned: np.ndarray
    path_ned: np.ndarray
    trajectory_ned: np.ndarray
    origin_ned: np.ndarray
    resolution: float
    states: np.ndarray


class AStar2DVisualizer:
    """在后台线程中将规划快照保存为PNG，不阻塞飞行控制循环。"""

    def __init__(
        self,
        config: AStarVisualizationConfig | None = None,
        run_timestamp: str | None = None,
    ) -> None:
        self.config = config or AStarVisualizationConfig()
        self.output_dir: Path | None = None
        self._closed = False
        self._error: Exception | None = None
        self._last_local_save: dict[str, float] = {}
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._stop_token = object()
        self._queue: queue.Queue = queue.Queue(
            maxsize=int(self.config.pending_image_limit)
        )
        self._worker: threading.Thread | None = None
        if not self.config.enabled:
            return

        root = (
            Path(self.config.output_root)
            if self.config.output_root is not None
            else Path(__file__).resolve().parent
        )
        directory_name = run_timestamp or datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        self.output_dir = root / directory_name
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._worker = threading.Thread(
            target=self._run,
            name="astar-2d-visualizer",
            daemon=True,
        )
        self._worker.start()

    def submit_local(
        self,
        snapshot,
        current_ned,
        global_target_ned,
        plan,
        inflation_radius: float,
        source_name: str = "local",
        now: float | None = None,
    ) -> bool:
        """按时间间隔提交一张局部规划图；队列忙时丢弃本帧。"""
        monotonic_now = time.monotonic() if now is None else float(now)
        if not self.should_save_local(source_name, monotonic_now):
            return False

        current = np.asarray(current_ned, dtype=np.float64)
        target = np.asarray(global_target_ned, dtype=np.float64)
        states = np.asarray(snapshot.states)
        if current.shape != (3,) or target.shape != (3,) or states.ndim != 3:
            return False
        occupied, unknown, inflated = self._project_local_grid(
            snapshot,
            current[2],
            float(self.config.local_above_height),
            float(self.config.local_below_height),
            float(inflation_radius),
        )
        if occupied is None:
            return False
        local_target = getattr(plan, "local_target_ned", None)
        task = _LocalPlotTask(
            sequence=self._next_sequence(),
            source_name=str(source_name),
            timestamp=datetime.now(),
            origin_ned=np.asarray(snapshot.origin_ned, dtype=np.float64).copy(),
            resolution=float(snapshot.resolution),
            unknown=unknown.copy(),
            occupied=occupied.copy(),
            inflated=inflated.copy(),
            current_ned=current.copy(),
            global_target_ned=target.copy(),
            local_target_ned=(
                None
                if local_target is None
                else np.asarray(local_target, dtype=np.float64).copy()
            ),
            path_ned=np.asarray(plan.path_ned, dtype=np.float64).copy(),
            status=str(plan.status),
            speed_limit=float(plan.speed_limit),
        )
        if not self._put_nowait(task):
            return False
        self._last_local_save[source_name] = monotonic_now
        return True

    @classmethod
    def _project_local_grid(
        cls,
        snapshot,
        center_down: float,
        above_height: float,
        below_height: float,
        inflation_radius: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """先过滤高度，再投影并重新计算二维膨胀。"""
        states = np.asarray(snapshot.states)
        if states.ndim != 3:
            return None, None, None

        resolution = float(snapshot.resolution)
        down_centers = (
            float(snapshot.origin_ned[2])
            + (np.arange(states.shape[2], dtype=np.float64) + 0.5)
            * resolution
        )
        selected = (
            down_centers >= float(center_down) - float(above_height) - 1e-9
        )
        selected &= (
            down_centers <= float(center_down) + float(below_height) + 1e-9
        )
        if not np.any(selected):
            return None, None, None

        band = states[:, :, selected]
        occupied = np.any(band == 1, axis=2)
        unknown = np.all(band == -1, axis=2)
        inflated = cls._inflate_2d(
            occupied,
            resolution,
            inflation_radius,
        )
        inflated &= ~occupied
        return occupied, unknown, inflated

    @staticmethod
    def _inflate_2d(
        occupied: np.ndarray,
        resolution: float,
        inflation_radius: float,
    ) -> np.ndarray:
        """按米制半径膨胀二维障碍，不引入切片外的三维障碍。"""
        result = np.zeros_like(occupied, dtype=bool)
        occupied_cells = np.argwhere(occupied)
        if occupied_cells.size == 0:
            return result

        radius_cells = max(
            0,
            int(math.ceil(float(inflation_radius) / float(resolution))),
        )
        coordinates = np.arange(-radius_cells, radius_cells + 1)
        north_offsets, east_offsets = np.meshgrid(
            coordinates,
            coordinates,
            indexing="ij",
        )
        offsets = np.column_stack((
            north_offsets.ravel(),
            east_offsets.ravel(),
        ))
        offsets = offsets[
            np.linalg.norm(offsets * float(resolution), axis=1)
            <= float(inflation_radius) + 1e-9
        ]
        expanded = (
            occupied_cells[:, None, :] + offsets[None, :, :]
        ).reshape(-1, 2)
        valid = np.all(expanded >= 0, axis=1)
        valid &= np.all(expanded < np.asarray(result.shape), axis=1)
        expanded = expanded[valid]
        result[tuple(expanded.T)] = True
        return result

    def should_save_local(
        self,
        source_name: str = "local",
        now: float | None = None,
    ) -> bool:
        """在复制地图快照前快速判断当前是否达到保存间隔。"""
        if not self._can_submit():
            return False
        monotonic_now = time.monotonic() if now is None else float(now)
        last_save = self._last_local_save.get(source_name, -math.inf)
        return monotonic_now - last_save >= float(
            self.config.local_save_interval
        )

    def submit_global(
        self,
        snapshot,
        current_ned,
        goal_ned,
        path_ned,
        above_height: float,
        below_height: float,
        source_name: str = "global_astar",
    ) -> bool:
        """提交一次全局返航路径及其记忆地图投影。"""
        if not self._can_submit():
            return False
        current = np.asarray(current_ned, dtype=np.float64)
        goal = np.asarray(goal_ned, dtype=np.float64)
        indices = np.asarray(snapshot.indices, dtype=np.int64)
        values = np.asarray(snapshot.log_odds)
        if current.shape != (3,) or goal.shape != (3,):
            return False
        resolution = float(snapshot.resolution)
        if indices.size != 0:
            height_center = 0.5 * (float(current[2]) + float(goal[2]))
            vertical = ned_height_slice_mask(
                indices,
                resolution,
                height_center,
                above_height,
                below_height,
            )
            indices = indices[vertical]
            values = values[vertical]
        path = np.asarray(path_ned, dtype=np.float64).reshape(-1, 3)
        trajectory = np.asarray(
            snapshot.trajectory_ned,
            dtype=np.float64,
        ).reshape(-1, 3)
        origin, display_resolution, states = self._build_global_grid(
            indices,
            values,
            resolution,
            float(snapshot.occupied_threshold),
            float(snapshot.free_threshold),
            np.vstack((current, goal, path, trajectory)),
        )
        task = _GlobalPlotTask(
            sequence=self._next_sequence(),
            timestamp=datetime.now(),
            source_name=str(source_name),
            current_ned=current.copy(),
            goal_ned=goal.copy(),
            path_ned=path.copy(),
            trajectory_ned=trajectory.copy(),
            origin_ned=origin,
            resolution=display_resolution,
            states=states,
        )
        return self._put_nowait(task)

    def _build_global_grid(
        self,
        indices,
        values,
        resolution,
        occupied_threshold,
        free_threshold,
        support_points,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """将稀疏三维体素投影成连续的二维鸟瞰占据栅格。"""
        indices = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        values = np.asarray(values).reshape(-1)
        support_points = np.asarray(support_points, dtype=np.float64).reshape(
            -1,
            3,
        )
        support_indices = np.floor(
            support_points[:, :2] / float(resolution)
        ).astype(np.int64)
        if len(indices) > 0:
            all_indices = np.vstack((indices[:, :2], support_indices))
        else:
            all_indices = support_indices
        padding = 2
        minimum = all_indices.min(axis=0) - padding
        maximum = all_indices.max(axis=0) + padding
        native_shape = maximum - minimum + 1
        native_cell_count = int(native_shape[0]) * int(native_shape[1])
        scale = max(
            1,
            int(math.ceil(math.sqrt(
                native_cell_count / int(self.config.global_max_grid_cells)
            ))),
        )
        shape = np.ceil(native_shape / scale).astype(int)
        states = np.full(tuple(shape), -1, dtype=np.int8)
        if len(indices) > 0:
            projected = np.floor_divide(indices[:, :2] - minimum, scale)
            projected[:, 0] = np.clip(projected[:, 0], 0, shape[0] - 1)
            projected[:, 1] = np.clip(projected[:, 1], 0, shape[1] - 1)
            free = values <= float(free_threshold)
            occupied = values >= float(occupied_threshold)
            states[tuple(projected[free].T)] = 0
            states[tuple(projected[occupied].T)] = 1
        origin = minimum.astype(np.float64) * float(resolution)
        return origin, float(resolution) * scale, states

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            self._queue.put(self._stop_token)
            self._worker.join()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("A* 2D visualization failed") from self._error

    def _can_submit(self) -> bool:
        return bool(
            self.config.enabled
            and not self._closed
            and self.output_dir is not None
        )

    def _next_sequence(self) -> int:
        with self._sequence_lock:
            self._sequence += 1
            return self._sequence

    def _put_nowait(self, task) -> bool:
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            return False
        return True

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is self._stop_token:
                    return
                if isinstance(task, _LocalPlotTask):
                    self._render_local(task)
                else:
                    self._render_global(task)
            except Exception as error:  # 可视化失败不能中断飞行控制。
                if self._error is None:
                    self._error = error
            finally:
                self._queue.task_done()

    def _render_local(self, task: _LocalPlotTask) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch

        rows, columns = task.occupied.shape
        north_min = float(task.origin_ned[0])
        east_min = float(task.origin_ned[1])
        north_max = north_min + rows * task.resolution
        east_max = east_min + columns * task.resolution

        image = np.empty((rows, columns, 4), dtype=np.float32)
        image[:] = (0.92, 0.96, 1.0, 1.0)
        image[task.unknown] = (0.82, 0.84, 0.87, 1.0)
        image[task.inflated] = (1.0, 0.72, 0.25, 0.78)
        image[task.occupied] = (0.12, 0.13, 0.15, 1.0)

        width_pixels = int(np.clip(
            columns * int(self.config.local_cell_pixels) + 300,
            1000,
            int(self.config.local_max_image_pixels),
        ))
        height_pixels = int(np.clip(
            rows * int(self.config.local_cell_pixels) + 180,
            900,
            int(self.config.local_max_image_pixels),
        ))
        figure, axis = plt.subplots(figsize=(
            width_pixels / int(self.config.image_dpi),
            height_pixels / int(self.config.image_dpi),
        ))
        try:
            axis.imshow(
                image,
                origin="lower",
                extent=[east_min, east_max, north_min, north_max],
                interpolation="nearest",
            )
            path = task.path_ned
            if len(path) > 0:
                axis.plot(
                    path[:, 1],
                    path[:, 0],
                    color="#1677ff",
                    linewidth=2.2,
                    marker="o",
                    markersize=3.0,
                    label="Local trajectory",
                )
            axis.scatter(
                task.current_ned[1],
                task.current_ned[0],
                s=75,
                marker="^",
                color="#16a34a",
                edgecolors="white",
                linewidths=0.8,
                label="Drone",
                zorder=5,
            )
            if task.local_target_ned is not None:
                axis.scatter(
                    task.local_target_ned[1],
                    task.local_target_ned[0],
                    s=95,
                    marker="*",
                    color="#0ea5e9",
                    edgecolors="#082f49",
                    linewidths=0.7,
                    label="Local target",
                    zorder=6,
                )
            clipped_goal = np.asarray([
                np.clip(
                    task.global_target_ned[0],
                    north_min + 0.5 * task.resolution,
                    north_max - 0.5 * task.resolution,
                ),
                np.clip(
                    task.global_target_ned[1],
                    east_min + 0.5 * task.resolution,
                    east_max - 0.5 * task.resolution,
                ),
            ])
            goal_is_inside = bool(
                north_min <= task.global_target_ned[0] <= north_max
                and east_min <= task.global_target_ned[1] <= east_max
            )
            axis.scatter(
                clipped_goal[1],
                clipped_goal[0],
                s=85,
                marker="X",
                color="#dc2626",
                label=(
                    "Segment target"
                    if goal_is_inside
                    else "Target (outside)"
                ),
                zorder=6,
            )
            map_handles = [
                Patch(facecolor="#d1d5db", label="Unknown"),
                Patch(facecolor="#fbbf24", label="Inflated obstacle"),
                Patch(facecolor="#1f2937", label="Occupied"),
            ]
            handles, labels = axis.get_legend_handles_labels()
            axis.legend(
                map_handles + handles,
                [item.get_label() for item in map_handles] + labels,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                fontsize=8,
            )
            axis.set_xlim(east_min, east_max)
            axis.set_ylim(north_min, north_max)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("East (m)")
            axis.set_ylabel("North (m)")
            axis.set_title(
                f"Local plan | source={task.source_name} | "
                f"status={task.status} | waypoints={len(path)} | "
                f"speed={task.speed_limit:.2f} m/s"
            )
            axis.grid(
                True,
                color="#475569",
                alpha=0.22,
                linewidth=0.55,
            )
            figure.tight_layout()
            output_path = self.output_dir / (
                f"{task.source_name}_{task.sequence:06d}_"
                f"{task.timestamp.strftime('%Y%m%d_%H%M%S_%f')}.png"
            )
            figure.savefig(
                output_path,
                dpi=int(self.config.image_dpi),
                bbox_inches="tight",
            )
        finally:
            plt.close(figure)

    def _render_global(self, task: _GlobalPlotTask) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch

        rows, columns = task.states.shape
        north_min = float(task.origin_ned[0])
        east_min = float(task.origin_ned[1])
        north_max = north_min + rows * task.resolution
        east_max = east_min + columns * task.resolution
        width_pixels = int(np.clip(
            columns * int(self.config.global_cell_pixels) + 300,
            1000,
            int(self.config.global_max_image_pixels),
        ))
        height_pixels = int(np.clip(
            rows * int(self.config.global_cell_pixels) + 180,
            900,
            int(self.config.global_max_image_pixels),
        ))
        image = np.empty((rows, columns, 4), dtype=np.float32)
        image[:] = (0.82, 0.84, 0.87, 1.0)
        image[task.states == 0] = (0.92, 0.96, 1.0, 1.0)
        image[task.states == 1] = (0.10, 0.12, 0.15, 1.0)

        figure, axis = plt.subplots(figsize=(
            width_pixels / int(self.config.image_dpi),
            height_pixels / int(self.config.image_dpi),
        ))
        try:
            axis.imshow(
                image,
                origin="lower",
                extent=[east_min, east_max, north_min, north_max],
                interpolation="nearest",
            )
            trajectory = task.trajectory_ned
            if len(trajectory) > 0:
                axis.plot(
                    trajectory[:, 1],
                    trajectory[:, 0],
                    color="#64748b",
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.85,
                    label="Outbound trace",
                    zorder=4,
                )
            path = task.path_ned
            axis.plot(
                path[:, 1],
                path[:, 0],
                color="#1677ff",
                linewidth=2.8,
                marker="o",
                markersize=4.5,
                label="Return path",
                zorder=5,
            )
            axis.scatter(
                task.current_ned[1],
                task.current_ned[0],
                s=85,
                marker="^",
                color="#16a34a",
                edgecolors="white",
                linewidths=0.8,
                label="Return start",
                zorder=6,
            )
            axis.scatter(
                task.goal_ned[1],
                task.goal_ned[0],
                s=90,
                marker="X",
                color="#dc2626",
                label="Home",
                zorder=6,
            )
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlim(east_min, east_max)
            axis.set_ylim(north_min, north_max)
            axis.set_xlabel("East (m)")
            axis.set_ylabel("North (m)")
            axis.set_title(
                f"Global return plan | source={task.source_name} | "
                f"waypoints={len(path)}"
            )
            axis.grid(True, color="#475569", alpha=0.22, linewidth=0.55)
            map_handles = [
                Patch(facecolor="#d1d5db", label="Unknown"),
                Patch(facecolor="#ebf5ff", label="Remembered free"),
                Patch(facecolor="#1f2937", label="Occupied"),
            ]
            handles, labels = axis.get_legend_handles_labels()
            axis.legend(
                map_handles + handles,
                [item.get_label() for item in map_handles] + labels,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                fontsize=8,
            )
            figure.tight_layout()
            output_path = self.output_dir / (
                f"global_{task.source_name}_{task.sequence:06d}_"
                f"{task.timestamp.strftime('%Y%m%d_%H%M%S_%f')}.png"
            )
            figure.savefig(
                output_path,
                dpi=int(self.config.image_dpi),
                bbox_inches="tight",
            )
        finally:
            plt.close(figure)
