"""Open3D全局稀疏占据地图实时窗口。"""

from __future__ import annotations

import threading

import numpy as np


class Open3DGlobalMapVisualizer:
    """显示不断扩展且不会随无人机移动清除的全局障碍地图。"""

    def __init__(
        self,
        global_map,
        show_free_voxels: bool = False,
        point_size: float = 2.0,
        min_point_spacing: float = 0.45,
        max_occupied_voxels: int = 20000,
        max_free_voxels: int = 10000,
        refresh_rate: float = 5.0,
    ) -> None:
        self.global_map = global_map
        self.show_free_voxels = bool(show_free_voxels)
        self.point_size = float(point_size)
        self.min_point_spacing = float(min_point_spacing)
        self.max_occupied_voxels = int(max_occupied_voxels)
        self.max_free_voxels = int(max_free_voxels)
        self.refresh_rate = float(refresh_rate)
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._validate()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._visualization_loop,
            name="Open3DGlobalSparseMapVisualizer",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                "Global sparse-map visualization failed."
            ) from self._error

    def _visualization_loop(self) -> None:
        try:
            import open3d as o3d
            o3d.utility.set_verbosity_level(
                o3d.utility.VerbosityLevel.Error
            )

            visualizer = o3d.visualization.Visualizer()
            if not visualizer.create_window(
                window_name="Garage Global Sparse Map (N-E-Up)",
                width=1180,
                height=800,
            ):
                return

            occupied_cloud = o3d.geometry.PointCloud()
            free_cloud = o3d.geometry.PointCloud()
            drone_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.18)
            drone_marker.paint_uniform_color([0.10, 0.95, 0.25])
            bounds = o3d.geometry.LineSet()
            geometry_added = {
                "occupied": False,
                "free": False,
                "drone": False,
                "bounds": False,
            }
            drone_position_neu = np.zeros(3, dtype=np.float64)

            render = visualizer.get_render_option()
            render.point_size = self.point_size
            render.background_color = np.asarray([0.04, 0.04, 0.04])
            period = 1.0 / self.refresh_rate

            while not self._stop.is_set():
                snapshot = self.global_map.snapshot()
                if snapshot.timestamp <= 0.0:
                    if not visualizer.poll_events():
                        break
                    visualizer.update_renderer()
                    self._stop.wait(period)
                    continue

                occupied = self._prepare_points(
                    snapshot.occupied_points_ned,
                    self.max_occupied_voxels,
                )
                free = np.empty((0, 3), dtype=np.float64)
                if self.show_free_voxels:
                    free = self._prepare_points(
                        snapshot.free_points_ned,
                        self.max_free_voxels,
                    )
                self._set_cloud(
                    occupied_cloud,
                    occupied,
                    [0.92, 0.12, 0.12],
                    o3d,
                )
                self._set_cloud(
                    free_cloud,
                    free,
                    [0.22, 0.52, 0.92],
                    o3d,
                )
                self._set_bounds(bounds, snapshot, o3d)

                if not geometry_added["bounds"]:
                    visualizer.add_geometry(bounds, True)
                    geometry_added["bounds"] = True
                else:
                    visualizer.update_geometry(bounds)
                geometry_added["occupied"] = self._add_or_update(
                    visualizer,
                    occupied_cloud,
                    geometry_added["occupied"],
                    occupied.shape[0] > 0,
                )
                geometry_added["free"] = self._add_or_update(
                    visualizer,
                    free_cloud,
                    geometry_added["free"],
                    free.shape[0] > 0,
                )

                new_drone_position = self._ned_to_neu(
                    snapshot.drone_position_ned.reshape(1, 3)
                )[0]
                drone_marker.translate(
                    new_drone_position - drone_position_neu,
                    relative=True,
                )
                drone_position_neu = new_drone_position
                if geometry_added["drone"]:
                    visualizer.update_geometry(drone_marker)
                else:
                    visualizer.add_geometry(drone_marker, False)
                    geometry_added["drone"] = True

                if not visualizer.poll_events():
                    break
                visualizer.update_renderer()
                self._stop.wait(period)

            visualizer.destroy_window()
        except Exception as error:
            if not self._stop.is_set():
                self._error = error
                print(f"--- WARNING: global map visualization stopped: {error} ----")

    def _prepare_points(
        self,
        points_ned: np.ndarray,
        maximum: int,
    ) -> np.ndarray:
        points = self._ned_to_neu(points_ned)
        points = self._thin_by_spacing(points, self.min_point_spacing)
        if points.shape[0] <= maximum:
            return points
        indices = np.linspace(
            0,
            points.shape[0] - 1,
            num=maximum,
            dtype=int,
        )
        return points[indices]

    @staticmethod
    def _add_or_update(
        visualizer,
        geometry,
        already_added: bool,
        has_points: bool,
    ) -> bool:
        if already_added:
            visualizer.update_geometry(geometry)
            return True
        if has_points:
            visualizer.add_geometry(geometry, False)
            return True
        return False

    @staticmethod
    def _set_cloud(cloud, points, color, o3d) -> None:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        cloud.points = o3d.utility.Vector3dVector(points)
        colors = np.tile(np.asarray(color, dtype=float), (points.shape[0], 1))
        cloud.colors = o3d.utility.Vector3dVector(colors)

    @staticmethod
    def _ned_to_neu(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).copy()
        if points.size:
            points[:, 2] *= -1.0
        return points

    @staticmethod
    def _thin_by_spacing(
        points: np.ndarray,
        spacing: float,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if points.shape[0] == 0:
            return points
        cells = np.floor(points / float(spacing)).astype(np.int64)
        _, first_indices = np.unique(cells, axis=0, return_index=True)
        return points[np.sort(first_indices)]

    def _set_bounds(self, line_set, snapshot, o3d) -> None:
        minimum, maximum = snapshot.bounds_ned
        corners = np.asarray(
            [
                [minimum[0], minimum[1], minimum[2]],
                [maximum[0], minimum[1], minimum[2]],
                [maximum[0], maximum[1], minimum[2]],
                [minimum[0], maximum[1], minimum[2]],
                [minimum[0], minimum[1], maximum[2]],
                [maximum[0], minimum[1], maximum[2]],
                [maximum[0], maximum[1], maximum[2]],
                [minimum[0], maximum[1], maximum[2]],
            ],
            dtype=np.float64,
        )
        corners = self._ned_to_neu(corners)
        lines = np.asarray(
            [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7],
            ],
            dtype=np.int32,
        )
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(
            np.tile([0.68, 0.68, 0.68], (len(lines), 1))
        )

    def _validate(self) -> None:
        if self.refresh_rate <= 0.0:
            raise ValueError("refresh_rate must be greater than zero.")
        if self.point_size <= 0.0:
            raise ValueError("point_size must be greater than zero.")
        if self.min_point_spacing <= 0.0:
            raise ValueError("min_point_spacing must be greater than zero.")
        if min(
            self.max_occupied_voxels,
            self.max_free_voxels,
        ) <= 0:
            raise ValueError("Visualization voxel limits must be positive.")
