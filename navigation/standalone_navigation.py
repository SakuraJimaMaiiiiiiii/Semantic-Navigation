"""Saved-map navigation from the same scene/origin, without an exploration flight."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import uuid
import webbrowser

import numpy as np
from scipy.spatial import cKDTree

from .scene_memory import load_bundle, write_json
from .target import select_targets, SemanticTargetRequest
from .exploration_session import ExplorationSession, SavedMapNavigator


class SavedMapNotFound(FileNotFoundError):
    """The selected directory has no saved navigation bundle."""


class NavigationPortInUse(RuntimeError):
    """Another process owns the navigation control port."""


def read_archive(directory):
    directory = Path(directory).resolve()
    if not (directory / 'bundle.json').exists() and (directory / 'exploration').is_dir():
        directory /= 'exploration'
    bundle = directory / 'bundle.json'
    if not bundle.is_file():
        raise SavedMapNotFound(
            f'指定地图缺少文件：{bundle}。该目录可能已删除、移动或未完成地图保存。\n'
            '请恢复完整 exploration 目录，或去掉 --map 自动选择最新可用地图。'
            '不同地图的物体 ID 可能不同，请先核对目标。'
        )
    manifest = json.loads(bundle.read_text(encoding='utf-8'))
    # Retain checksum/frame/landing validation; session identity is read from
    # this explicitly selected archive, not from the old command descriptor.
    return directory, load_bundle(directory, manifest['session_id'])


def latest_archive(root=None):
    """Choose by run creation name, not HTML refresh or navigation timestamps."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / 'data' / 'runs'
    for run in sorted(root.glob('run_*'), key=lambda p: p.name, reverse=True):
        directory = run / 'exploration'
        if not (directory / 'bundle.json').is_file():
            continue  # Navigation-only runs and unfinished exploration aren't maps.
        try:
            _, archive = read_archive(directory)
            if not archive.get('records'):
                raise ValueError('地图没有可导航的语义物体')
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f'--- 跳过不可用地图 {run.name}：{error} ----')
            continue
        return directory.resolve()
    raise FileNotFoundError('没有找到可导航的已完成探索地图。请先完成一次探索，或用 --map 指定保存地图。')


def check_same_frame(archive, current, yaw, live):
    """Conservative consistency gate, NOT general scene relocalization."""
    if not np.isfinite(current).all() or np.linalg.norm(current - archive['home']) > .5:
        raise ValueError('请把无人机放回建图时的起点；当前位置与保存的 NED 原点不一致')
    delta = np.arctan2(np.sin(yaw - archive['initial_yaw']), np.cos(yaw - archive['initial_yaw']))
    if not np.isfinite(delta) or abs(delta) > np.deg2rad(15):
        raise ValueError('请恢复建图时的初始朝向；独立模式不自动重定位坐标系')
    saved = archive['occupancy_map']
    points = (saved['indices'][saved['log_odds'] >= saved['occupied_threshold']] + .5) * saved['resolution']
    observed = live.occupied_points_ned
    if len(points) < 100 or len(observed) < 100:
        raise ValueError('起点雷达障碍观测不足，不能核对保存地图坐标')
    observed = observed[::max(1, len(observed) // 10000)]
    if np.linalg.norm(np.ptp(observed, axis=0)) < 2:
        raise ValueError('雷达观测范围不足以核对地图')
    distances = cKDTree(points).query(observed)[0]
    tolerance = max(.30, 2 * saved['resolution'])
    fraction = float(np.mean(distances <= tolerance))
    if fraction < .75:
        raise ValueError(f'实时雷达与旧地图不一致：匹配率={fraction:.1%}；未起飞。请检查场景、起点和定位基准')
    return {'same_origin_consistency_passed': True, 'matched_fraction': fraction,
            'tolerance_m': tolerance, 'samples': len(observed),
            'registration': 'identity; same scene and origin required; not general relocalization'}


def associate_live_records(saved_records, live_records):
    """Cross-session IDs are unrelated. Accept only unambiguous geometry matches."""
    result = []
    for live in live_records:
        candidates = []
        for saved in saved_records:
            if saved['class_name'] != live['class_name']:
                continue
            a, b = saved['bbox_3d'], live['bbox_3d']
            low = np.maximum(a['min'], b['min'])
            high = np.minimum(a['max'], b['max'])
            if np.any(high <= low):
                continue
            distance = float(np.linalg.norm(np.asarray(saved['position_world']) - live['position_world']))
            if distance <= 1.0:
                candidates.append((distance, saved))
        candidates.sort(key=lambda pair: pair[0])
        if candidates and (len(candidates) == 1 or candidates[1][0] - candidates[0][0] > .5):
            result.append({**live, 'live_instance_id': live['runtime_instance_id'],
                           'runtime_instance_id': candidates[0][1]['runtime_instance_id']})
    # Multiple fragments must not masquerade as independent confirmations.
    counts = {}
    for record in result:
        identity = record['runtime_instance_id']
        counts[identity] = counts.get(identity, 0) + 1
    return [record for record in result if counts[record['runtime_instance_id']] == 1]


class IndependentNavigator(SavedMapNavigator):
    def _verification_candidates(self, frame, current):
        records = associate_live_records(self.records, frame.records)
        return select_targets(records, replace(self.request, near_instance_id=None), current, frame.timestamp)


class IndependentSession(ExplorationSession):
    navigator_type = IndependentNavigator

    def __init__(self, runtime, config, archive):
        super().__init__(runtime, config)
        self.show_place_links = False
        self.archive = deepcopy(archive)
        # Geometry remains in the verified NED frame. Return to THIS flight's
        # measured launch point, not an old telemetry sample.
        self.archive['home'] = self.home.copy()
        self.memory.records = {r['runtime_instance_id']: deepcopy(r) for r in archive['records']}
        self.memory.places = deepcopy(archive.get('places', {}))

    def navigation_archive(self):
        return self.archive


def run_standalone(directory, request, *, config_path, open_map=False):
    source, archive = read_archive(directory)
    if request is not None and not select_targets(archive['records'], request, archive['home'], 0, check_age=False):
        raise ValueError('保存地图中没有符合条件的目标；未启动无人机连接')
    from .target_navigation_config import parse_config
    from navigation.session_server import SessionServer
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
            '若旧探索/导航终端已就绪，可直接在其中输入多个目标，例如 vehicle_03 vehicle_01。\n'
            '用 python navigate_target.py --status --no-open-map 查看已有会话。\n'
            '若要改用独立导航，请先在原终端输入 quit，等待返航降落并退出后，再启动本程序。'
        ) from error
    runtime = service = None
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
        print('--- 独立导航：等待实时感知，并核对同场景、同起点雷达地图；尚未起飞 ----')
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
        session = IndependentSession(runtime, config, archive)
        from navigate_target import DEFAULT_SESSION_FILE
        server.map_path = str(session.map_path.resolve())
        server.publish_descriptor(session.directory / 'session.json')
        server.publish_descriptor(DEFAULT_SESSION_FILE)
        write_json(session.directory / 'source_map.json', {'source': str(source), **evidence})
        session.annotate(state='保存地图已加载：准备目标导航')
        print(f'--- 独立导航地图：{session.map_path} ----')
        if open_map:
            webbrowser.open(session.map_path.resolve().as_uri())
        result = {'success': True}
        if request is not None:
            result = session.navigate(request, uuid.uuid4().hex)
            from .console_output import navigation_result_text
            print(navigation_result_text(result, request.instance_id))
        from .exploration_session import serve_target_commands
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
