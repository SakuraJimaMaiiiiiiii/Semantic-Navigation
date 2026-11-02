"""Short PX4-controlled rectangular flight for FAST-LIO recording and loop validation."""
import argparse
from dataclasses import replace
import json
import math
import sys
import time
import uuid

import numpy as np
from config.mission_config import MissionConfig, MissionWaypoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=float, default=3., help='Forward side in metres (0.5..6)')
    parser.add_argument('--width', type=float, default=1.5, help='Right side in metres (0.5..4)')
    parser.add_argument('--height', type=float, default=1.5, help='Height above takeoff point (0.8..2)')
    parser.add_argument('--dry-run', action='store_true', help='Print relative waypoints without connecting or flying')
    args = parser.parse_args(argv)
    for name, low, high in [('length', .5, 6.), ('width', .5, 4.), ('height', .8, 2.)]:
        value = getattr(args, name)
        if not math.isfinite(value) or not low <= value <= high:
            parser.error(f'--{name} must be finite and between {low} and {high}')
    return args


def loop_waypoints(home, yaw, length, width, height):
    """Body-horizontal forward/right rectangle; each yaw follows its own leg."""
    home = np.asarray(home, dtype=float)
    if home.shape != (3,) or not np.isfinite(home).all() or not math.isfinite(yaw):
        raise ValueError('Invalid home pose')
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.])
    right = np.array([-math.sin(yaw), math.cos(yaw), 0.])
    origin = home + [0., 0., -height]
    points = [origin+length*forward, origin+length*forward+width*right,
              origin+width*right, origin]
    headings = [yaw, yaw+math.pi/2, yaw+math.pi, yaw-math.pi/2]
    return tuple(MissionWaypoint(tuple(p), math.atan2(math.sin(a), math.cos(a)), 2.)
                 for p,a in zip(points, headings))


def check_capture(runtime):
    recorder = runtime.fastlio_recorder
    if recorder is None or recorder.error or recorder.thread is None or not recorder.thread.is_alive():
        raise RuntimeError(f'FAST-LIO capture unavailable: {getattr(recorder, "error", None)}')


def record_static_start(runtime, seconds=10.):
    vehicle = runtime.vehicle
    origin = vehicle.get_position()
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        check_capture(runtime)
        state = vehicle.get_drone_state()
        if (vehicle.mav.isArmed or not np.isfinite(state.position_xyz).all()
                or not np.isfinite(state.linear_velocity).all()
                or np.linalg.norm(state.linear_velocity) > .05
                or not np.isfinite(state.angular_velocity).all()
                or np.linalg.norm(state.angular_velocity) > .1
                or not math.isfinite(state.timestamp) or abs(time.time()-state.timestamp) > 2.
                or np.linalg.norm(state.position_xyz-origin) > .1):
            raise RuntimeError('Static initialization requires a disarmed, stationary aircraft')
        vehicle.wait(.1)
    if runtime.fastlio_recorder.counts['imu'] < 100 or runtime.fastlio_recorder.counts['scans'] < 3:
        raise RuntimeError('Insufficient recorded IMU/LiDAR; flight was not started')


class CaptureCheckedPlanner:
    """Abort a flight leg if its validation recording has failed."""
    def __init__(self, runtime, planner):
        self.runtime, self.planner = runtime, planner

    def __getattr__(self, name):
        return getattr(self.planner, name)

    def plan(self, current, target):
        check_capture(self.runtime)
        return self.planner.plan(current, target)


def configure_mission(runtime, args):
    vehicle = runtime.vehicle
    home, yaw = vehicle.get_position(), float(vehicle.get_euler()[2])
    waypoints = loop_waypoints(home, yaw, args.length, args.width, args.height)
    mission = replace(MissionConfig(), initial_yaw=None, takeoff_height=args.height,
                      takeoff_wait=10., waypoints=waypoints, final_hover_time=10.,
                      return_to_origin=True, restore_initial_yaw=True, face_path=False)
    for planner in (runtime.local_planner, runtime.return_safety_planner):
        if planner is None:
            raise RuntimeError('Short flight requires the existing local avoidance planner')
        planner.set_navigation_plane(home[2]-args.height, runtime.semantic_navigation_config,
                                     runtime.navigation_map)
    runtime.controller.mission = mission
    runtime.controller.local_planner = CaptureCheckedPlanner(runtime, runtime.local_planner)
    runtime.controller.return_safety_planner = CaptureCheckedPlanner(runtime, runtime.return_safety_planner)
    # The rectangle already ends at home. Reuse the existing short local return
    # correction and PX4 landing flow, without a second exploration/return tour.
    runtime.controller.global_return_planner = None
    return dict(home_ned=home.tolist(), initial_yaw_rad=yaw,
                airborne_home_ned=(home+[0.,0.,-args.height]).tolist(),
                waypoints=[dict(position_ned=list(w.position_ned), yaw=w.yaw) for w in waypoints])


def fly_and_finish_recording(runtime, report):
    runtime.run_mission()
    # Keep capture running through touchdown, not just through LAND command.
    runtime.vehicle.finish_flight_session()
    runtime.vehicle.wait(3.)
    report['flight_completed'] = True
    report['final_position_ned'] = runtime.vehicle.get_position().tolist()
    report['landing'] = runtime.vehicle.landing_confirmation
    runtime.fastlio_recorder.close()
    capture = json.loads((runtime.run_output_dir/'fastlio/capture_report.json').read_text())
    report['capture_completed'] = bool(capture.get('completed'))
    if not report['capture_completed']:
        raise RuntimeError(f'Flight finished but capture is incomplete: {capture}')


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps([dict(forward_right_down=list(w.position_ned), yaw_degrees=math.degrees(w.yaw))
                          for w in loop_waypoints([0.,0.,0.],0.,args.length,args.width,args.height)], indent=2))
        return
    # Imports that load SDK/planner libraries are deliberately after --dry-run.
    from application import DemoRuntime, WebSocketRequestSender
    from detections import DetectorConfig
    from navigation.session_server import SessionServer
    from run_explore import shutdown_session

    class TestRuntime(DemoRuntime):
        def _create_configs(self):
            super()._create_configs()
            self.recording_config = replace(self.recording_config, enabled=False)
            self.global_map_config = replace(self.global_map_config, enabled=True, visualize=False,
                                            export_foxglove_mcap_on_close=False)
            self.planner_config = replace(self.planner_config, unknown_is_occupied=True)
            self.drone_config.guided_max_forward_speed = min(.3, self.drone_config.guided_max_forward_speed)
            body = dict(vehicle_free_radius=self.semantic_navigation_config.clearance_radius,
                        vehicle_free_half_height=self.semantic_navigation_config.vertical_half_extent)
            self.mapping_config = replace(self.mapping_config, **body)
            self.global_map_config = replace(self.global_map_config, **body)

    # Same exclusive endpoint as exploration: never create a second controller.
    try:
        server = SessionServer(uuid.uuid4().hex)
    except OSError as error:
        raise SystemExit(f'无法取得控制端口 8877；请先正常关闭旧探索终端。原始错误：{error}') from error
    runtime = None
    report = dict(test='fastlio_short_rectangle', flight_completed=False, capture_completed=False,
                  scan_model='unverified', controls_navigation='PX4 + existing local planner', error=None)
    try:
        WebSocketRequestSender().send_when_ue_ready()
        runtime = TestRuntime(detector_config=replace(DetectorConfig(), enabled=False), fastlio_mode='record')
        runtime.start()
        print('--- FAST-LIO 短程验证：地面静止录制 10 秒，请保持仿真运行 ----', flush=True)
        record_static_start(runtime)
        report.update(configure_mission(runtime,args))
        (runtime.run_output_dir/'fastlio_test.json').write_text(
            json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'--- 前进 {args.length:g} m，右移 {args.width:g} m，矩形返回；飞行高度 {args.height:g} m ----', flush=True)
        fly_and_finish_recording(runtime, report)
        from tools.audit_fastlio_inputs import audit
        audit_report = audit(runtime.run_output_dir)
        (runtime.run_output_dir/'fastlio/input_audit.json').write_text(
            json.dumps(audit_report,indent=2),encoding='utf-8')
        print(f'--- 短程验证录制完成：{runtime.run_output_dir} ----', flush=True)
        print(f'--- 输入审计：{audit_report["issues"]}；扫描时序未验证属于预期待核查项 ----',flush=True)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        primary = sys.exc_info()[1]
        try:
            # On a failed leg, request landing and retain recording while waiting
            # for touchdown; do not command an unverified blind return.
            if runtime is not None and primary is not None and runtime.vehicle.connected:
                try:
                    runtime.land_if_armed(hover_first=True)
                    if runtime.vehicle.mav.isArmed or getattr(runtime.vehicle,'_landing_commanded',False):
                        runtime.vehicle.finish_flight_session()
                except Exception as error:
                    report['cleanup_landing_error'] = f'{type(error).__name__}: {error}'
            try:
                shutdown_session(server, runtime, primary_error=primary)
            except Exception as error:
                report['cleanup_error'] = f'{type(error).__name__}: {error}'
                raise
        finally:
            if runtime is not None:
                # Failed flight and incomplete recording are separate outcomes:
                # cleanup may still drain a complete, useful closed-loop capture.
                capture_path = runtime.run_output_dir/'fastlio/capture_report.json'
                if capture_path.exists():
                    try:
                        capture = json.loads(capture_path.read_text())
                        report['capture_completed'] = bool(capture.get('completed'))
                    except (OSError, ValueError) as error:
                        report['capture_report_error'] = str(error)
                (runtime.run_output_dir/'fastlio_test.json').write_text(
                    json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
