"""RflySim语义导航演示所需组件的创建、启动与清理。"""

from dataclasses import replace
import json
from config.lidar_config import LidarConfig
from mapping.lidar_mapping_worker import LidarMappingWorker
from mapping.semantic_voxels import SemanticVoxelEvidence
from config.exploration_config import FrontierExplorationConfig
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
        self,
        semantic_target: SemanticTargetRequest | None = None,
        *,
        exploration=False,
        detector_config: DetectorConfig | None = None,
        exploration_config: FrontierExplorationConfig | None = None,
        fastlio_mode: str | None = None,
        lidar_scan_model: str = "unverified",
    ) -> None:
        if semantic_target is not None and not isinstance(
            semantic_target, SemanticTargetRequest
        ):
            raise TypeError("semantic_target must be a SemanticTargetRequest or None")
        self.exploration_mode = bool(exploration)
        self.exploration_config = (
            exploration_config or FrontierExplorationConfig()
            if self.exploration_mode
            else None
        )
        self._requested_detector_config = detector_config
        self.fastlio_mode = fastlio_mode
        self.lidar_scan_model = lidar_scan_model
        self.fastlio_recorder = None
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
            # Depth-only keyframe rebuilding would discard fused LiDAR evidence.
            loop_closure_enabled=False,
        )
        self.planner_config = cfg.PlannerConfig()
        if self.semantic_target is not None or self.exploration_mode:
            self.global_map_config = replace(
                self.global_map_config,
                enabled=True,
                visualize=False,
            )
            self.planner_config = replace(self.planner_config, unknown_is_occupied=True)
            clearance = self.semantic_navigation_config.clearance_radius
            if self.exploration_config is not None:
                clearance = max(clearance, self.exploration_config.flight_clearance)
            body = dict(
                vehicle_free_radius=clearance,
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
            self._requested_detector_config or DetectorConfig(),
            video_output_dir=self.run_output_dir / "detections",
            semantic_map_path=(
                self.run_output_dir / "maps" / "semantic_instances.json"
            ),
        )

        if self.exploration_mode:
            if self.planner_config.navigation_mode not in {"ego", "ego_native"}:
                raise ValueError("Waypoint exploration requires ego/ego_native local avoidance")
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
        camera_config = cfg.CameraConfig(copter_id=self.drone_config.copter_id)
        self.lidar_config = LidarConfig()
        sensors = json.loads(camera_config.sensor_json.read_text(encoding="utf-8"))
        if any(int(s["SeqID"]) == self.lidar_config.seq_id for s in sensors["VisionSensors"]):
            raise ValueError("LiDAR SeqID conflicts with an existing sensor")
        sensors["VisionSensors"].append(self.lidar_config.sensor(self.drone_config.copter_id))
        combined = self.run_output_dir / "sensors_rgbd_lidar.json"
        combined.write_text(json.dumps(sensors, indent=2), encoding="utf-8")
        camera_config = replace(camera_config, sensor_json=combined)
        self.camera = RflyRGBDCamera(camera_config)
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
        from mapping.online_loop_navigation import OnlineLoopNavigationMap
        local_map_type = LocalOccupancyGrid
        global_map_type = GlobalSparseOccupancyMap
        lidar_worker_type = LidarMappingWorker
        navigation_map_type = OnlineLoopNavigationMap
        detector_options = {}
        if self.exploration_mode:
            from mapping.explore_v3 import adapters as v3
            local_map_type = v3.LocalOccupancyGrid
            global_map_type = v3.GlobalSparseOccupancyMap
            lidar_worker_type = v3.LidarMappingWorker
            navigation_map_type = v3.OnlineLoopNavigationMap
            detector_options["mapper_type"] = v3.SemanticInstanceMapper
            print("--- 探索建图：v3 / PX4 + LiDAR；车辆使用跨视角关联；识别后端沿用启动配置 ----")
        self.recorder = (
            FlightDataRecorder(
                self.recorder_source,
                self.recording_config,
            )
            if self.recording_config.enabled
            else None
        )
        self.occupancy_grid = (
            local_map_type(self.mapping_config)
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
        local_config = self.planner_config
        if self.exploration_mode:
            local_config = replace(
                local_config,
                ego_max_acceleration=min(local_config.ego_max_acceleration,
                                         self.drone_config.guided_max_horizontal_acceleration),
            )
        self.local_planner = (
            planner_type(
                self.occupancy_grid,
                local_config,
                visualizer=self.astar_visualizer,
                visualization_name="local",
            )
            if self.occupancy_grid is not None
            else None
        )
        self.global_sparse_map = (
            global_map_type(self.global_map_config)
            if self.global_map_config.enabled
            else None
        )
        self.navigation_map = (navigation_map_type(self.global_sparse_map)
                               if self.global_sparse_map is not None else None)
        self.global_return_planner = (
            GlobalReturnPlanner(
                self.navigation_map,
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
                integrate_depth=False,
            )
            if self.mapping_config.enabled
            else None
        )
        self.global_visualizer = self._create_global_visualizer()
        self.global_mapping_worker = (
            GlobalMappingWorker(
                self.global_mapping_source,
                self.global_sparse_map,
                self.global_map_config,
                visualizer=self.global_visualizer,
                integrate_depth=False,
            )
            if self.global_map_config.enabled
            else None
        )
        self.detection_worker = (
            RealtimeDetectionWorker(
                self.detection_source,
                self.detector_config,
                semantic_voxels=SemanticVoxelEvidence(
                    self.global_map_config.resolution, self.global_map_config.max_voxels
                ) if self.global_map_config.enabled else None,
                **detector_options,
            )
            if self.detector_config.enabled
            else None
        )

        from mapping.lidar_loop_detector import LidarLoopDetector
        self.lidar_worker = lidar_worker_type(
            self.camera, self.data_stream, self.occupancy_grid,
            self.navigation_map, self.lidar_config,
            loop_detector=LidarLoopDetector(self.global_map_config,
                self.run_output_dir / 'maps' / 'lidar_loop_closures.json'),
        )
        self.lidar_worker.alignment_preview_path = self.run_output_dir / "lidar_camera_alignment.png"
        if self.detection_worker is not None:
            self.detection_worker.lidar_alignment_provider = self.lidar_worker.align_to_camera

        if self.semantic_target is not None:
            if self.detection_worker is None:
                raise ValueError("Semantic navigation requires the detector")
            self.controller.semantic_navigator = SemanticNavigator(
                self.vehicle,
                self.detection_worker,
                self.occupancy_grid,
                self.navigation_map,
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
        if self.fastlio_mode is not None:
            from localization.fastlio_recording import FastLioRecorder
            self.fastlio_recorder = FastLioRecorder(
                self.run_output_dir / 'fastlio', scan_model=self.lidar_scan_model,
                shadow=self.fastlio_mode == 'shadow', live_map=self.fastlio_mode == 'live')
            self.fastlio_recorder.start(self.vehicle.mav)
            self.lidar_worker.fastlio_recorder = self.fastlio_recorder
            if self.fastlio_recorder.live_map is not None and self.detection_worker is not None:
                self.detection_worker.fastlio_semantic_sink = self.fastlio_recorder.live_map.semantics.submit
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
        self.lidar_worker.start(timeout=self.camera.config.first_frame_timeout)
        if self.fastlio_mode in {'shadow', 'live'}:
            print("--- FAST-LIO 静止初始化中，初始化完成后才开始任务 ----", flush=True)
            self.fastlio_recorder.wait_ready()
            if self.fastlio_mode == 'live':
                print(f"--- FAST-LIO 实时实验地图：{self.run_output_dir / 'fastlio' / 'map.html'}；导航仍使用 PX4 ----", flush=True)
        print("--- LiDAR enabled: Type20 / 360° / body FRD / LiDAR-only occupancy / camera semantic association ----")
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
        if getattr(self.vehicle, "_landing_commanded", False):
            return  # Do not overwrite an active AUTO.LAND with a hover request.
        if hover_first:
            try:
                self.vehicle.hover()
            except Exception as error:
                print(f"--- 降落前悬停失败，继续发送降落请求：{error} ----")
        self.vehicle.land()

    def finish_exploration_recording(self) -> None:
        """Finalize exploration files while keeping live navigation sensors running."""
        if getattr(self, "_exploration_recording_finished", False):
            return
        if self.fastlio_recorder is not None:
            self.fastlio_recorder.close()
        if getattr(self, "live_loop_mapping", None) is not None:
            self.live_loop_mapping.close()
        if self.recorder_started:
            self.recorder.close()
            self.recorder.raise_if_failed()
        lidar = getattr(self, "lidar_worker", None)
        if lidar is not None:
            lidar._preview_saved = True
            lidar._alignment_sample_saved = True
        if self.detection_started:
            self.detection_worker.finish_recording()
        if lidar is not None:
            if lidar.loop_detector is not None:
                lidar.loop_detector.close()
            self._save_alignment_report()
        self.astar_visualizer.close()
        if getattr(self, "live_loop_mapping", None) is not None:
            print("--- 正在保存最终回环校正地图与语义关联；完成后停止后台写入 ----")
            self.live_loop_mapping.finalize()
        self._exploration_recording_finished = True

    def _save_alignment_report(self) -> None:
        report = dict(self.lidar_worker.alignment_statistics)
        report['pose_sampling'] = dict(self.lidar_worker.stream.pose_statistics)
        report["semantic_fused_objects"] = getattr(
            self.detection_worker.segmenter, "lidar_fused_objects", 0) if self.detection_worker else 0
        report["semantic_lidar_points"] = getattr(
            self.detection_worker.segmenter, "lidar_fused_points", 0) if self.detection_worker else 0
        (self.run_output_dir / "lidar_camera_alignment.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def close(self) -> None:
        """按依赖关系逆序停止组件并输出运行结果。"""
        errors = []

        def stop(name, action):
            try:
                action()
                return True
            except Exception as error:
                errors.append(error)
                print(f"--- 组件关闭失败 [{name}]：{type(error).__name__}: {error} ----")
                return False

        stop("waypoint planning", self.vehicle.stop_waypoint_planning)
        if getattr(self, "live_loop_mapping", None) is not None:
            stop("live loop mapping", self.live_loop_mapping.close)
        lidar_closed = True
        detection_closed = True
        if getattr(self, "lidar_worker", None) is not None:
            lidar_closed = stop("lidar", self.lidar_worker.close)
            print(f"--- LiDAR mapping frames: {self.lidar_worker.update_count} ----")
        if self.fastlio_recorder is not None:
            stop("FAST-LIO recording", self.fastlio_recorder.close)
        if self.detection_started:
            detection_closed = stop("detection", self.detection_worker.close)
        if (getattr(self, "lidar_worker", None) is not None
                and not getattr(self, "_exploration_recording_finished", False)):
            stop("lidar report", self._save_alignment_report)
        if self.global_mapping_started:
            stop("global mapping", self.global_mapping_worker.close)
        if self.mapping_started:
            stop("local mapping", self.mapping_worker.close)
        if self.recorder_started:
            stop("recorder", self.recorder.close)
        if self.hub_started:
            stop("observation hub", self.observation_hub.close)
        stop("data stream", self.data_stream.close)
        stop("camera", self.camera.close)
        # Each remaining component still gets cleanup if touchdown times out.
        # Vehicle.close retains its no-disarm/no-disconnect protection.
        stop("vehicle", lambda: self.vehicle.close(disarm=True))
        stop("planner visualizer", self.astar_visualizer.close)
        # Finalize closed logs even when the flight/landing failed. Do this
        # after vehicle shutdown so offline reconstruction cannot delay landing.
        if (getattr(self, "live_loop_mapping", None) is not None
                and lidar_closed and detection_closed
                and not getattr(self, "_exploration_recording_finished", False)):
            stop("final loop map", self.live_loop_mapping.finalize)
        stop("results", self._report_results)
        if errors:
            raise errors[0]

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
                print("--- WARNING: real-time perception backend failed: " f"{error} ----")
            else:
                print(
                    "--- Perception summary: "
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
