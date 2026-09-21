"""RflySim语义导航演示所需组件的创建、启动与清理。"""

from dataclasses import replace
from config.semantic_navigation_config import SemanticNavigationConfig
from navigation.mission import SemanticNavigator
from navigation.target import SemanticTargetRequest

import config as cfg
from detections import DetectorConfig, RealtimeDetectionWorker
from Visual import AStar2DVisualizer, AStarVisualizationConfig
from mapping import (
    GlobalMappingWorker,
    GlobalSparseOccupancyMap,
    LocalMappingWorker,
    LocalOccupancyGrid,
    Open3DGlobalMapVisualizer,
    PointCloudLoopClosure,
)
from mission import RflyMissionController
from planner import (
    EgoLocalPlanner,
    GlobalReturnPlanner,
    LocalAvoidancePlanner,
    NativeEgoLocalPlanner,
)
from recording import FlightDataRecorder
from sensors import (
    RflyRGBDCamera,
    SynchronizedObservationHub,
    SynchronizedRGBDStream,
)
from vehicle import RflyMultirotorInterface


class DemoRuntime:
    """集中管理演示程序的配置、组件和生命周期。"""

    def __init__(
        self, semantic_target: SemanticTargetRequest | None = None, *, exploration=False
    ) -> None:
        if semantic_target is not None and not isinstance(
            semantic_target, SemanticTargetRequest
        ):
            raise TypeError("semantic_target must be a SemanticTargetRequest or None")
        self.exploration_mode = bool(exploration)
        self.semantic_target = semantic_target
        self.semantic_navigation_config = SemanticNavigationConfig()
        self._create_run_output_directory()
        self._create_configs()
        self._create_core()
        self._create_subscribers()
        self._create_processing()
        self._reset_started_flags()

    def _create_run_output_directory(self) -> None:
        self.run_output_dir = cfg.create_run_output_directory()
        print(f"--- 本次运行数据目录：{self.run_output_dir} ----")

    def _create_configs(self) -> None:
        self.mission_config = cfg.MissionConfig()
        self.drone_config = cfg.Config()
        self.recording_config = replace(
            cfg.RecordingConfig(),
            output_dir=self.run_output_dir / "recordings",
        )
        self.mapping_config = cfg.OccupancyGridConfig()
        self.global_map_config = replace(
            cfg.GlobalSparseMapConfig(),
            output_dir=self.run_output_dir / "maps",
        )
        self.planner_config = cfg.PlannerConfig()
        if self.semantic_target is not None or getattr(self, "exploration_mode", False):
            self.global_map_config = replace(
                self.global_map_config,
                enabled=True,
                loop_closure_enabled=False,
                visualize=False,
            )
            self.planner_config = replace(self.planner_config, unknown_is_occupied=True)
            body = dict(
                vehicle_free_radius=self.semantic_navigation_config.clearance_radius,
                vehicle_free_half_height=self.semantic_navigation_config.vertical_half_extent,
            )
            self.mapping_config = replace(self.mapping_config, **body)
            self.global_map_config = replace(self.global_map_config, **body)
            if (
                not self.mapping_config.enabled
                or self.planner_config.navigation_mode == "depth_guard"
            ):
                raise ValueError(
                    "Semantic navigation requires occupancy mapping and EGO/A* planning"
                )
        self.astar_visualization_config = replace(
            AStarVisualizationConfig(),
            output_root=self.run_output_dir,
        )
        self.detector_config = replace(
            DetectorConfig(),
            video_output_dir=self.run_output_dir / "detections",
            semantic_map_path=(
                self.run_output_dir / "maps" / "semantic_instances.json"
            ),
        )

        if getattr(self, "exploration_mode", False):
            if self.planner_config.navigation_mode not in {"ego", "ego_native"}:
                raise ValueError("Frontier exploration requires EGO trajectory execution")
            self.planner_config = replace(self.planner_config, return_navigation_mode=self.planner_config.navigation_mode)
            # Public semantic output is the coordinate-free scene graph; geometry
            # is committed separately by the exploration session as a binary cache.
            self.detector_config = replace(
                self.detector_config, save_semantic_map=False
            )
            self.global_map_config = replace(
                self.global_map_config,
                save_on_close=False,
                export_foxglove_mcap_on_close=False,
            )
            self.mission_config = replace(self.mission_config, return_to_origin=True)
            if not self.detector_config.enabled:
                raise ValueError("Exploration requires semantic perception")

    def _create_core(self) -> None:
        self.vehicle = RflyMultirotorInterface(self.drone_config)
        self.camera = RflyRGBDCamera(
            cfg.CameraConfig(copter_id=self.drone_config.copter_id)
        )
        self.data_stream = SynchronizedRGBDStream(
            self.vehicle,
            self.camera,
        )
        self.observation_hub = SynchronizedObservationHub(self.data_stream)

    def _create_subscribers(self) -> None:
        hub = self.observation_hub
        self.recorder_source = (
            hub.subscribe("hdf5_recorder", max_queue=90)
            if self.recording_config.enabled
            else None
        )
        self.mapping_source = (
            hub.subscribe("local_mapping", max_queue=2)
            if self.mapping_config.enabled
            else None
        )
        self.global_mapping_source = (
            hub.subscribe("global_sparse_mapping", max_queue=2)
            if self.global_map_config.enabled
            else None
        )
        self.detection_source = (
            hub.subscribe("realtime_yolo_detection", max_queue=1)
            if self.detector_config.enabled
            else None
        )

    def _create_processing(self) -> None:
        self.recorder = (
            FlightDataRecorder(
                self.recorder_source,
                self.recording_config,
            )
            if self.recording_config.enabled
            else None
        )
        self.occupancy_grid = (
            LocalOccupancyGrid(self.mapping_config)
            if self.mapping_config.enabled
            else None
        )
        self.astar_visualizer = AStar2DVisualizer(
            self.astar_visualization_config,
            run_timestamp="planning",
        )
        if self.astar_visualizer.output_dir is not None:
            print(
                "--- 局部规划二维投影目录：" f"{self.astar_visualizer.output_dir} ----"
            )
        planner_type = self._local_planner_type(self.planner_config.navigation_mode)
        self.local_planner = (
            planner_type(
                self.occupancy_grid,
                self.planner_config,
                visualizer=self.astar_visualizer,
                visualization_name="local",
            )
            if self.occupancy_grid is not None
            else None
        )
        self.global_sparse_map = (
            GlobalSparseOccupancyMap(self.global_map_config)
            if self.global_map_config.enabled
            else None
        )
        self.global_return_planner = (
            GlobalReturnPlanner(
                self.global_sparse_map,
                self.planner_config,
                visualizer=self.astar_visualizer,
            )
            if self.global_sparse_map is not None
            else None
        )
        return_planner_type = self._local_planner_type(
            self.planner_config.return_navigation_mode
        )
        self.return_safety_planner = (
            return_planner_type(
                self.occupancy_grid,
                replace(
                    self.planner_config,
                    navigation_mode=(self.planner_config.return_navigation_mode),
                ),
                visualizer=self.astar_visualizer,
                visualization_name="return_local",
            )
            if self.occupancy_grid is not None
            else None
        )
        if self.return_safety_planner is not None:
            return_mode_label = {
                "ego_native": "C++ EGO 三维 B 样条",
                "ego": "Python EGO 三维 B 样条",
                "astar": "局部栅格 A*",
                "depth_guard": "深度估计直线路径",
            }[self.planner_config.return_navigation_mode]
            print(f"--- 返航航段规划模式：{return_mode_label} ----")
        self.controller = RflyMissionController(
            self.vehicle,
            self.mission_config,
            local_planner=self.local_planner,
            global_return_planner=self.global_return_planner,
            return_safety_planner=self.return_safety_planner,
        )
        self.mapping_worker = (
            LocalMappingWorker(
                self.mapping_source,
                self.occupancy_grid,
                update_rate=self.mapping_config.update_rate,
            )
            if self.mapping_config.enabled
            else None
        )
        self.global_visualizer = self._create_global_visualizer()
        self.loop_closure = (
            PointCloudLoopClosure(
                self.global_sparse_map,
                self.global_map_config,
            )
            if self.global_map_config.enabled
            and self.global_map_config.loop_closure_enabled
            else None
        )
        self.global_mapping_worker = (
            GlobalMappingWorker(
                self.global_mapping_source,
                self.global_sparse_map,
                self.global_map_config,
                visualizer=self.global_visualizer,
                loop_closure=self.loop_closure,
            )
            if self.global_map_config.enabled
            else None
        )
        self.detection_worker = (
            RealtimeDetectionWorker(
                self.detection_source,
                self.detector_config,
            )
            if self.detector_config.enabled
            else None
        )

        if self.semantic_target is not None:
            if self.detection_worker is None:
                raise ValueError("Semantic navigation requires the detector")
            self.controller.semantic_navigator = SemanticNavigator(
                self.vehicle,
                self.detection_worker,
                self.occupancy_grid,
                self.global_sparse_map,
                self.local_planner,
                self.semantic_target,
                self.semantic_navigation_config,
                self.run_output_dir / "evaluation" / "semantic_navigation.json",
            )

        if self.detection_worker is not None:
            self.camera.set_detection_preview_provider(
                self.detection_worker.get_latest_preview,
                self.detector_config.window_name,
            )

    @staticmethod
    def _local_planner_type(mode):
        if mode == "ego_native":
            return NativeEgoLocalPlanner
        if mode == "ego":
            return EgoLocalPlanner
        return LocalAvoidancePlanner

    def _create_global_visualizer(self):
        config = self.global_map_config
        if not config.enabled or not config.visualize:
            return None
        return Open3DGlobalMapVisualizer(
            self.global_sparse_map,
            show_free_voxels=config.show_free_voxels,
            point_size=config.visualization_point_size,
            min_point_spacing=config.visualization_min_spacing,
            max_occupied_voxels=config.max_visualized_occupied_voxels,
            max_free_voxels=config.max_visualized_free_voxels,
        )

    def _reset_started_flags(self) -> None:
        self.recorder_started = False
        self.mapping_started = False
        self.global_mapping_started = False
        self.detection_started = False
        self.hub_started = False

    def start(self) -> None:
        """连接仿真、验证同步数据并启动后台处理线程。"""
        self.vehicle.setup_ue()
        self.vehicle.connect()
        # CopterSim/PX4 就绪后再请求绑定到该载具的视觉传感器。
        self.camera.start()
        self.data_stream.start()
        # PX4 状态采样线程刚启动时，需要等待一帧时间戳晚于首个状态
        # 样本的 RGB-D 数据。沿用相机首帧超时，避免用硬编码 2 秒把
        # 正常的 UE/PX4 启动延迟误判为任务失败。
        self.data_stream.get_observation(
            timeout=float(self.camera.config.first_frame_timeout)
        )

        if any(
            (
                self.recording_config.enabled,
                self.mapping_config.enabled,
                self.global_map_config.enabled,
                self.detector_config.enabled,
            )
        ):
            self.observation_hub.start()
            self.hub_started = True
        if self.recorder is not None:
            self.recorder.start()
            self.recorder_started = True
        if self.mapping_worker is not None:
            self.mapping_worker.start()
            self.mapping_started = True
        if self.global_mapping_worker is not None:
            self.global_mapping_worker.start()
            self.global_mapping_started = True
        if self.detection_worker is not None:
            self.detection_worker.start()
            self.detection_started = True

    def run_mission(self) -> None:
        """在主线程执行飞行任务。"""
        self.controller.run()

    def land_if_armed(self, hover_first: bool = False) -> None:
        """任务异常时安全悬停并降落。"""
        if not self.vehicle.connected or not bool(self.vehicle.mav.isArmed):
            return
        if hover_first:
            self.vehicle.hover()
        self.vehicle.land()

    def close(self) -> None:
        """按依赖关系逆序停止组件并输出运行结果。"""
        if self.detection_started:
            self.detection_worker.close()
        if self.global_mapping_started:
            self.global_mapping_worker.close()
        if self.mapping_started:
            self.mapping_worker.close()
        if self.recorder_started:
            self.recorder.close()
        if self.hub_started:
            self.observation_hub.close()
        self.data_stream.close()
        self.camera.close()
        self.vehicle.close(disarm=True)
        self.astar_visualizer.close()
        self._report_results()

    def _report_results(self) -> None:
        if self.astar_visualizer.output_dir is not None:
            try:
                self.astar_visualizer.raise_if_failed()
            except RuntimeError as error:
                print(f"--- WARNING: 局部规划二维投影保存失败：{error} ----")
            else:
                print(
                    "--- 局部规划二维投影已保存："
                    f"{self.astar_visualizer.output_dir} ----"
                )

        if self.recorder_started:
            try:
                self.recorder.raise_if_failed()
            except RuntimeError as error:
                print(f"--- WARNING: flight data recording failed: {error} ----")
            else:
                print(
                    f"--- Flight data saved: {self.recorder.output_path} "
                    f"({self.recorder.frame_count} frames) ----"
                )

        if self.mapping_started:
            try:
                self.mapping_worker.raise_if_failed()
            except RuntimeError as error:
                print(f"--- WARNING: local occupancy mapping failed: {error} ----")

        if self.global_mapping_started:
            self._report_global_mapping()

        if self.detection_started:
            try:
                self.detection_worker.raise_if_failed()
            except RuntimeError as error:
                print("--- WARNING: real-time YOLO-World/SAM2 failed: " f"{error} ----")
            else:
                print(
                    "--- YOLO-World/SAM2 perception summary: "
                    f"{self.detection_worker.processed_frames} frames processed ----"
                )
                for path in self.detection_worker.video_output_paths:
                    if path.is_file():
                        print(f"--- Detection video saved: {path} ----")
                semantic_map_path = self.detection_worker.semantic_map_output_path
                if semantic_map_path is not None and semantic_map_path.is_file():
                    print(
                        "--- Persistent semantic map saved: "
                        f"{semantic_map_path} ----"
                    )

    def _report_global_mapping(self) -> None:
        try:
            self.global_mapping_worker.raise_if_failed()
        except RuntimeError as error:
            print(f"--- WARNING: global sparse mapping failed: {error} ----")
            return
        if self.global_mapping_worker.output_path is None:
            return

        snapshot = self.global_sparse_map.snapshot()
        print(
            "--- Global sparse map saved: "
            f"{self.global_mapping_worker.output_path} "
            f"({snapshot.voxel_count} voxels, "
            f"{snapshot.occupied_count} occupied) ----"
        )
        if self.global_mapping_worker.foxglove_output_path is not None:
            print(
                "--- Foxglove MCAP saved: "
                f"{self.global_mapping_worker.foxglove_output_path} ----"
            )
        if self.loop_closure is not None:
            print(
                "--- Loop closure summary: "
                f"{self.loop_closure.loop_count} accepted, "
                f"{self.loop_closure.rejected_loop_count} rejected, "
                f"{len(self.loop_closure.keyframes)} keyframes ----"
            )
