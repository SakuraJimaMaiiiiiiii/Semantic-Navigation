"""Independent WildOS navigation; startup adapted from navigation/standalone_navigation.py.

One SDK owner, same-origin consistency gate, existing graph target execution.
Source archives are read-only; the new run owns all outputs and descriptors.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import sys
import time
import uuid
import numpy as np

from navigation.standalone_navigation import (check_same_frame, associate_live_records,
                                             NavigationPortInUse, SavedMapNotFound as SavedMapNotFound)
from navigation.exploration_session import TargetUnavailableError, serve_target_commands
from navigation.scene_memory import write_json
from navigation.target import SemanticTargetRequest, select_targets
from .explore import GraphExplorationSession
from .target_navigation import GraphTargetNavigator
from .vehicle_motion import VehicleAwareMissionController
from .saved_graph import read_graph_archive, latest_graph_archive as latest_graph_archive
from .session_service import WildOSSessionServer


class IndependentGraphNavigator(GraphTargetNavigator):
    def _verification_candidates(self, frame, current):
        # New detector IDs must never be equated with the saved run's IDs.
        records = associate_live_records(self.records,frame.records)
        return select_targets(records,replace(self.request,near_instance_id=None),current,frame.timestamp)


class IndependentGraphSession(GraphExplorationSession):
    navigator_type = IndependentGraphNavigator

    def __init__(self, runtime, config, archive, saved, *, session_id, evidence):
        if evidence.get('same_origin_consistency_passed') is not True:
            raise ValueError('独立图导航必须先完成场景与起点一致性核对')
        super().__init__(runtime,config,session_id=session_id)
        self.archive = deepcopy(archive)
        self.archive['home'] = self.home.copy()
        self.memory.records = {r['runtime_instance_id']:deepcopy(r) for r in archive['records']}
        self.memory.places = deepcopy(archive.get('places',{}))
        self.show_place_links = False
        self._restored_revision = getattr(runtime.global_sparse_map,'revision',0)
        self._restored_altitude = saved['adapter'].grid.altitude
        self.graph_explorer = SimpleNamespace(frontiers=SimpleNamespace(
            graph=deepcopy(saved['graph']),adapter=saved['adapter'],validation=saved['validation']))
        # Final map numbering may differ from detector IDs in the last live
        # graph snapshot. Render object annotations from the selected archive.
        from .semantic_graph import attach_objects
        attach_objects(self.graph_explorer.frontiers.graph, {'objects':self.archive['records']},
                       {}, saved['adapter'].grid.resolution)
        self.graph_explorer.frontiers.graph['robot_attachment'] = dict(
            position_ned=self.home.tolist(),node_id=None,node_stable_id=None,
            component_id=None,status='awaiting_route')
        runtime.controller = VehicleAwareMissionController.wrap(runtime.controller,self._target_objects)
        self.directory.mkdir(parents=True,exist_ok=True)
        write_json(self.directory/'source_map.json',dict(**saved['provenance'],**evidence,
            active_session_id=self.session_id,active_revision=self._restored_revision,
            binding='verified same-origin identity transform; old node IDs retained'))
        write_json(self.directory/'navigation_graph/restored/graph.json',self.graph_explorer.frontiers.graph)
        write_json(self.directory/'navigation_graph/restored/validation.json',saved['validation'])

    def navigation_archive(self):
        return self.archive

    def _validate_target_frame(self):
        if getattr(self.runtime.global_sparse_map,'revision',0) != self._restored_revision:
            raise TargetUnavailableError('独立导航坐标系已变化，禁止继续使用旧图路径')
        return self.graph_explorer.frontiers.graph

    def _target_grid(self, altitude, current):
        from .height_recovery import (MAX_RECOVERY_METRES, RECOVERY_START_METRES,
                                      height_detail, vertical_rejection)
        tolerance = self.runtime.vehicle.config.guided_z_tolerance
        if self.airborne_navigation:
            tolerance = min(tolerance,RECOVERY_START_METRES)
        if abs(altitude-self._restored_altitude) > tolerance:
            detail = height_detail(current,self._restored_altitude,tolerance)
            if not self.airborne_navigation or abs(altitude-self._restored_altitude) > MAX_RECOVERY_METRES:
                raise TargetUnavailableError('飞行高度与保存图高度不一致，未执行新目标；'+detail)
            projected = np.asarray(current).copy();projected[2] = self._restored_altitude
            reason = vertical_rejection(self.runtime.occupancy_grid.execution_snapshot,current,projected,
                                        self.runtime.semantic_navigation_config,self._target_objects())
            if reason is not None:
                raise TargetUnavailableError('无法安全恢复图高度：'+reason+'；'+detail)
        return super()._target_grid(self._restored_altitude,current)

    def _arguments(self, request, output):
        arguments = super()._arguments(request,output)
        if request is not None:
            arguments['graph_altitude'] = self._restored_altitude
        return arguments


def run_standalone(directory, request, *, config_path, open_map=False):
    source, archive, saved = read_graph_archive(directory)
    if request is not None and not select_targets(archive['records'], request, archive['home'], 0, check_age=False):
        raise ValueError('保存地图中没有符合条件的目标；未启动无人机连接')
    from navigation.target_navigation_config import parse_config
    SessionServer = WildOSSessionServer
    config, args = parse_config(['--config', str(config_path)], include_perception=True)
    # Reserve the same control port as exploration before touching the simulator.
    try:
        server = SessionServer(uuid.uuid4().hex, config.server_port)
    except OSError as error:
        import errno
        if error.errno != errno.EADDRINUSE and getattr(error, 'winerror', None) != 10048:
            raise
        raise NavigationPortInUse(
            f'控制端口 {config.server_port} 已占用，未启动新的无人机连接。\n'
            '若旧探索/导航终端已就绪，直接运行 python navigate_wildos_target.py 即可自动连接并输入目标。\n'
            '用 python navigate_wildos_target.py --status --no-open-map 查看已有会话。\n'
            '若要改用独立导航，请在客户端输入 quit，等待原控制端返航降落并退出后，再启动本程序。'
        ) from error
    runtime = service = live_map = None
    try:
        if args.perception_backend == 'sam3':
            from detections.sam3_process import SAM3ServiceProcess
            service = SAM3ServiceProcess(args.sam3_url)
            service.start()
        from application import DemoRuntime, WebSocketRequestSender
        from detections import DetectorConfig
        detector = replace(DetectorConfig(), backend=args.perception_backend, sam3_url=args.sam3_url)
        if args.perception_backend == 'sam3':
            detector = replace(detector, window_name='SAM3 Detection / Semantic Segmentation')
        if args.object_prompts:
            detector = replace(detector, class_prompts=tuple(args.object_prompts),
                               semantic_landmark_classes=tuple(args.object_prompts))
        WebSocketRequestSender().send_when_ue_ready()
        # Navigation uses the existing PX4 NED map. Independent experimental
        # FAST-LIO recording/maps are deliberately not substituted for it.
        # Enable the semantic navigation infrastructure without starting a
        # mission. Interactive requests replace this navigator before flying.
        runtime = DemoRuntime(semantic_target=request or SemanticTargetRequest(class_name='vehicle'),
                              detector_config=detector)
        runtime.start()
        print('--- WildOS 独立图导航：等待实时感知，并核对同场景、同起点雷达地图；尚未起飞 ----')
        deadline = time.monotonic() + 60
        error = None
        while time.monotonic() < deadline:
            runtime.detection_worker.raise_if_failed()
            frame = runtime.detection_worker.semantic_snapshot()
            live = runtime.global_sparse_map.snapshot()
            if frame is not None and time.monotonic() - frame.completed_at < runtime.semantic_navigation_config.perception_timeout:
                try:
                    if abs(live.timestamp - frame.timestamp) > runtime.semantic_navigation_config.map_timeout:
                        raise ValueError('实时地图尚未同步')
                    evidence = check_same_frame(archive, runtime.vehicle.get_position(),
                                                runtime.vehicle.get_euler()[2], live)
                    break
                except ValueError as exc:
                    error = exc
            time.sleep(.2)
        else:
            raise RuntimeError(f'独立导航起飞前核对失败：{error or "没有新感知帧"}')
        runtime.global_sparse_map.seed_saved_map(archive['occupancy_map'])
        session = IndependentGraphSession(runtime, config, archive, saved,
                                          session_id=server.session_id, evidence=evidence)
        from navigate_wildos_target import DEFAULT_SESSION_FILE
        server.map_path = str(session.map_path.resolve())
        server.publish_descriptor(session.directory / 'session.json')
        server.publish_descriptor(DEFAULT_SESSION_FILE)
        # Provenance and the graph copy were saved by IndependentGraphSession.
        session.annotate(state='保存地图已加载：准备目标导航')
        print(f'--- WildOS 独立图导航地图：{session.map_path} ----')
        from .live_map import LiveMapServer
        live_map = LiveMapServer(session.map_path)
        session.live_map = live_map
        server.live_map_url = live_map.url
        frontiers = session.graph_explorer.frontiers
        live_map.publish(frontiers.adapter,frontiers.graph,frontiers.validation,0)
        server.publish_descriptor(session.directory / 'session.json')
        server.publish_descriptor(DEFAULT_SESSION_FILE)
        if open_map:
            import webbrowser
            webbrowser.open(live_map.url, new=2)
        result = {'success': True}
        if request is not None:
            result = session.navigate(request, uuid.uuid4().hex)
            from navigation.console_output import navigation_result_text
            print(navigation_result_text(result, request.instance_id))
        print('--- 保持本终端运行，接下来可直接输入目标 ID 或 home 返航 ----')
        serve_target_commands(session, server)
        return result
    finally:
        # Keep the control port reserved until flight/runtime cleanup finishes.
        primary_error = sys.exc_info()[1]
        steps = []
        if runtime is not None:
            steps.extend([('landing', lambda: runtime.land_if_armed(hover_first=True)),
                          ('runtime', runtime.close)])
        if live_map is not None:
            steps.append(('live map', live_map.close))
        steps.append(('control port', server.close))
        if service is not None:
            steps.append(('SAM3', service.close))
        errors = []
        for name, stop in steps:
            try:
                stop()
            except Exception as error:
                errors.append(error)
                print(f'--- 清理失败 [{name}]: {error} ----', file=sys.stderr)
        if errors and primary_error is None:
            raise errors[0]
