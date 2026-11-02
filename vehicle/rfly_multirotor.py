"""RflySim/PX4 多旋翼统一控制接口。"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from copy import deepcopy

import PX4MavCtrlV4 as PX4MavCtrl
import ReqCopterSim
import UE4CtrlAPI


ROUTE_SOURCE_LABELS = {
    "braking_corridor_blocked": "停止：当前速度的刹停通路不足",
    "route_clear": "A* 路线航点跟随（当前航段畅通，EGO待命）",
    "ego_clear": "EGO B样条生成的直达轨迹",
    "ego_avoid": "EGO A*种子优化后的三维绕障轨迹",
    "height_recovery": "恢复巡航高度（三维净空已确认）",
    "height_recovery_blocked": "停止：高度恢复通路未确认空闲",
    "depth_guard_clear": "深度估计生成的直线路径",
    "avoid": "A* 生成的绕障路径",
    "clear": "A* 地图验证后的直线路径",
}

@dataclass(frozen=True)
class DroneState:
    """同一采样时刻的 PX4 飞行状态。

    坐标约定：
        position_xyz、linear_velocity 使用世界 NED 坐标系；
        quaternion 使用 [w, x, y, z]，表示机体 FRD 到世界 NED 的旋转；
        angular_velocity 使用机体 FRD 坐标系，单位 rad/s；
        euler_rpy 使用 [roll, pitch, yaw]，单位 rad。
    """

    timestamp: float
    position_xyz: np.ndarray
    quaternion: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    euler_rpy: np.ndarray
    px4_boot_timestamp: float
    position_covariance_ned: np.ndarray | None = None
    pose_sync_error: float = 0.0  # ATTITUDE / LOCAL_POSITION_NED source-time difference, seconds


class CollisionDetectedError(RuntimeError):
    """Raised when RflySim reports a collision during a flight task."""

    def __init__(self, target_id, position, object_name="", flight_context=None):
        self.target_id = int(target_id)
        self.position = np.asarray(position, dtype=float).copy()
        self.object_name = str(object_name)
        self.flight_context = flight_context
        object_text = f", object={self.object_name}" if self.object_name else ""
        super().__init__(
            f"Collision detected: target_id={self.target_id}, "
            f"position_ned={self.position.tolist()}{object_text}"
        )


class RflyMultirotorInterface:
    """RflySim/PX4 多旋翼控制接口。

    坐标系采用 NED：
        North：北向为正
        East：东向为正
        Down：向下为正

    因此：
        高度 5 m 对应 down = -5
        上升速度 1 m/s 对应 vz = -1

    Args:
        config: 无人机配置对象，通常使用 ``config.Config``。
    """

    def __init__(self, config):
        """初始化控制接口，但不建立仿真连接。

        Args:
            config: 包含无人机编号、超时时间和返航原点等参数的配置对象。
        """
        self.config = config

        self.ue = None
        self.mav = None

        self.connected = False
        self.offboard_enabled = False
        self.armed = False
        self._ground_down = 0.0
        self._landing_commanded = False
        self._landing_requested_at = None
        self._shutdown_disarmed = False
        self.landing_confirmation = None
        self._waypoint_planning_worker = None

    def setup_ue(self):
        """初始化 RflySim3D 显示接口。"""
        self.ue = UE4CtrlAPI.UE4CtrlAPI()
        copter_id = int(self.config.copter_id)

        self.ue.CopterID = copter_id
        self.ue.isVehicleCrash = False
        self.ue.isVehicleCrashID = -10

        self.ue.sendUE4Cmd("r.setres 720x405w", 0)
        self.ue.sendUE4Cmd("t.MaxFPS 30", 0)

        # Follow the SDK collision example: request continuous vehicle data,
        # enable RflySim3D collision mode, then listen for reqVeCrashData on
        # multicast 224.0.0.10:20006.
        self.ue.sendUE4Cmd("RflyReqVehicleData 1", 0)
        time.sleep(0.2)
        self.ue.sendUE4Cmd("RflyChangeViewKeyCmd P", 0)
        time.sleep(0.5)
        self.ue.initUE4MsgRec()
        time.sleep(2)

        if self.ue.getUE4Data(copter_id) == 0:
            print(
                "WARNING: no UE vehicle/crash data received on UDP 20006. "
                "Do not press P again because setup_ue() already enabled it; "
                "check Windows Firewall and UDP multicast 224.0.0.10."
            )
        else:
            print(f"--- UE collision data connected for CopterID={copter_id} ----")

    def connect(self):
        """连接 CopterSim 并启动 MAVLink 数据接收。"""
        if self.connected:
            return

        copter_id = int(self.config.copter_id)

        print(f"--- 正在连接 {copter_id} 号多旋翼…… ----")

        request = ReqCopterSim.ReqCopterSim()
        target_ip = request.getSimIpID(copter_id)

        # 只重置地面位置和航向。CopterSim 会自动将 Z 贴合地形，
        # 避免强制修改完整 XYZ 后造成 PX4 状态估计异常。
        if self.config.reset_initial_pose:
            initial_xy_yaw = self._vector3(
                self.config.initial_xy_yaw,
                "initial_xy_yaw",
            )
            request.sendReSimXYyaw(copter_id, initial_xy_yaw.tolist())
            time.sleep(5)

        request.sendReSimIP(copter_id)

        self.mav = PX4MavCtrl.PX4MavCtrler(copter_id, target_ip)
        self.mav.InitMavLoop()

        # 重置位置后必须等待 PX4 的 EKF 重新完成三维定位，否则飞控可能拒绝执行解锁或 Offboard 位置指令。
        ekf_timeout = float(self.config.ekf_timeout)
        deadline = time.time() + ekf_timeout
        while not self.mav.isPX4Ekf3DFixed:
            if time.time() >= deadline:
                self.mav.stopRun()
                raise TimeoutError(
                    f"等待 PX4 EKF 定位完成超时（{ekf_timeout:.1f} 秒）"
                )
            print("--- 等待 PX4 EKF 定位完成…… ----")
            time.sleep(1)

        self.connected = True
        # 以连接时的实际地面位置为降落判定基准，避免依赖手工配置原点。
        self._ground_down = float(self.get_position()[2])
        self._landing_ground_sources = self._landing_height_diagnostics()
        print(f"--- {copter_id} 号多旋翼连接成功，IP: {target_ip} ----")

    def enable_offboard(self, arm=True):
        """进入 Offboard 模式，并可选择是否解锁。

        Args:
            arm: ``True`` 时进入 Offboard 后自动解锁；``False`` 时仅切换
                控制模式。

        Raises:
            RuntimeError: 尚未连接无人机，或 PX4 拒绝解锁。
        """
        self._require_connection()

        if not self.offboard_enabled:
            position = self.get_position()
            yaw = float(self.get_euler()[2])
            if not np.all(np.isfinite(position)) or not math.isfinite(yaw):
                raise ValueError("Offboard initial position and yaw must be finite")
            self._prepare_offboard_sender()
            print("--- 进入 Offboard 模式…… ----")
            # 在启动 SDK 的指令发送循环前准备保持当前位置/航向的设定值。
            self.mav.SendPosNED(*position, yaw)
            self.mav.initOffboard()
            self.mav.SendPosNED(*position, yaw)
            time.sleep(0.5)
            self.offboard_enabled = True

            # RflySimSDK 的 initOffboard() 在 MAVLink 模式下会自动发送
            # 解锁命令，因此应同步真实状态，避免再次解锁。
            self.armed = bool(self.mav.isArmed)

        if arm and not self.armed:
            self.arm()

    def _prepare_offboard_sender(self):
        """SDK initOffboard starts t2 but does not replace a completed thread."""
        sender = self.mav.t2
        if sender.is_alive():
            raise RuntimeError("Offboard sender is still running; refusing a second sender")
        if sender.ident is not None:
            # AUTO.LAND joins the old sender while retaining the MAVLink receiver.
            # Never reinitialize the whole SDK or steal its receive connection.
            self.mav.isInOffboard = False
            self.mav.t2 = threading.Thread(
                target=self.mav.OffboardSendMode,
                name=sender.name, daemon=sender.daemon,
            )

    def arm(self):
        """解锁无人机。"""
        self._require_connection()

        if self.armed:
            return

        print("--- 无人机解锁…… ----")
        self.mav.SendMavArm(True)

        arm_timeout = float(self.config.arm_timeout)
        deadline = time.time() + arm_timeout
        while not self.mav.isArmed:
            if time.time() >= deadline:
                raise RuntimeError(
                    "PX4 拒绝解锁，请检查 QGroundControl 中的解锁提示"
                )
            time.sleep(0.2)

        self.armed = True

    def disarm(self, force=False):
        """上锁无人机。

        Args:
            force: ``True`` 时忽略接口记录的解锁状态并发送上锁命令。

        Warning:
            仅应在确认无人机落地后调用。RflySimSDK 使用强制上锁参数。
        """
        self._require_connection()

        if not self.armed and not force:
            return

        self.mav.SendMavArm(False)
        time.sleep(1)
        self.armed = False
        self._shutdown_disarmed = True

    def takeoff(self, height, yaw=None):
        """通过 Offboard 位置控制起飞。

        Args:
            height: 相对起飞点高度，单位 m，必须为正数。
            yaw: 偏航角，单位 rad；None 保持当前实测航向。
        """
        height = float(height)

        if height <= 0:
            raise ValueError("起飞高度必须大于 0")

        self._require_offboard()

        yaw = float(self.get_euler()[2]) if yaw is None else float(yaw)
        if not math.isfinite(yaw):
            raise ValueError("Takeoff yaw must be finite")
        position = self.get_position()
        target = np.array(
            [position[0], position[1], position[2] - height],
            dtype=float,
        )

        print(f"--- 起飞到 {height:.1f} m…… ----")
        self.send_position(target, yaw)

    def send_position(self, waypoint, yaw=0.0):
        """发送 NED 位置指令。

        Args:
            waypoint: [north, east, down]，单位 m。
            yaw: 偏航角，单位 rad。
        """
        self._require_offboard()

        north, east, down = self._vector3(waypoint, "waypoint")

        self.mav.SendPosNED(
            float(north),
            float(east),
            float(down),
            float(yaw),
        )
        self._last_position_command = {
            "position_ned": [float(north), float(east), float(down)],
            "yaw": float(yaw), "monotonic_time": time.monotonic(),
        }

    def wait(self, duration, check_interval=0.1):
        """Wait while monitoring RflySim collision state."""
        duration = float(duration)
        check_interval = float(check_interval)
        if duration < 0.0:
            raise ValueError("duration must not be negative")
        if check_interval <= 0.0:
            raise ValueError("check_interval must be positive")

        deadline = time.monotonic() + duration
        while True:
            self.raise_if_collision()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(check_interval, remaining))

    def move_waypoint_guided(
        self, waypoint, yaw=None, face_direction=False, timeout=None,
        local_planner=None, pass_through=False, control_horizon=None, xy_tolerance=None,
    ):
        """Execute certified NED position waypoints using the PX4 position loop.

        No Python velocity outer loop is used. PX4 position-mode parameters
        govern physical speed/acceleration. Planner speed limits bound position
        lead using the configured guidance gain, not a MAVLink velocity command.
        A zero limit latches a hold until stopped and continuously clear.
        A horizon return leaves only the last certified position target active.
        """
        self._require_offboard()
        target = self._vector3(waypoint, "waypoint")
        timeout = float(self.config.guided_timeout if timeout is None else timeout)
        tolerance = float(self.config.guided_xy_tolerance if xy_tolerance is None else xy_tolerance)
        period = 1.0 / float(self.config.guided_update_rate)
        if not np.isfinite(target).all() or timeout <= 0 or tolerance <= 0:
            raise ValueError("Waypoint and tolerances must be finite and valid")
        if not math.isfinite(timeout) or not math.isfinite(tolerance):
            raise ValueError("Waypoint timeout and tolerance must be finite")
        if control_horizon is not None and (not math.isfinite(control_horizon) or control_horizon <= 0):
            raise ValueError("control_horizon must be positive and finite")
        initial = self.get_position()
        if yaw is None:
            delta = target[:2] - initial[:2]
            yaw = (math.atan2(delta[1], delta[0]) if face_direction and np.linalg.norm(delta) > tolerance
                   else float(self.get_euler()[2]))
        yaw = float(yaw)
        if not math.isfinite(yaw):
            raise ValueError("Waypoint yaw must be finite")
        command_yaw = float(self.get_euler()[2])
        started = time.monotonic()
        stable_since = None
        last_status = None
        hold_target = getattr(self, "_waypoint_hold_target", None)
        hold_reason = getattr(self, "_waypoint_hold_reason", None)
        clear_since = getattr(self, "_waypoint_clear_since", None)
        braking = bool(getattr(self, "_waypoint_braking", False))
        # Preserve the command ramp across rolling horizons; measured speed
        # noise must not re-seed the position lead on every planning cycle.
        lead_speed = float(getattr(self, "_waypoint_lead_speed", 0.0))
        lead_acceleration = float(getattr(self, "_waypoint_lead_acceleration", 0.0))
        lead_jerk = float(self.config.guided_position_lead_jerk)
        from navigation.terminal_progress import shared_progress
        progress = getattr(self, '_motion_progress_display', None)
        if progress is None:
            progress = self._motion_progress_display = shared_progress()
        if not math.isfinite(lead_jerk) or lead_jerk <= 0:
            raise ValueError("guided_position_lead_jerk must be positive and finite")
        from planner.waypoint_planning_worker import WaypointPlanningWorker
        worker = getattr(self, '_waypoint_planning_worker', None)
        if local_planner is not None and (worker is None or worker.planner is not local_planner):
            if worker is not None:
                self.hover()
                worker.close()
            worker = self._waypoint_planning_worker = WaypointPlanningWorker(local_planner)
        try:
            while True:
                self.raise_if_collision(include_ground=True)
                now = time.monotonic()
                if now-started >= timeout:
                    raise TimeoutError(f"Position waypoint timed out: target={target.round(3).tolist()}, status={last_status}")
                if control_horizon is not None and now-started >= control_horizon:
                    return now-started
                current = self.get_position()
                delta = target-current
                reached = (np.linalg.norm(delta[:2]) <= tolerance
                           and abs(delta[2]) <= self.config.guided_z_tolerance)
                if not pass_through and reached and np.linalg.norm(self.get_velocity()) <= .05:
                    # Arrival is measured, not conditional on a new moving plan.
                    # Hold where we are rather than chasing the last centimetres
                    # into a close obstacle merely to finish the reference.
                    stable_since = now if stable_since is None else stable_since
                    self.send_position(current, command_yaw)
                    if now-stable_since >= self.config.guided_stable_time:
                        self._waypoint_lead_speed = self._waypoint_lead_acceleration = 0.
                        self._waypoint_position_reference = current.copy()
                        self._waypoint_hold_target = self._waypoint_hold_reason = None
                        self._waypoint_clear_since = None
                        self._waypoint_braking = False
                        progress.finish()
                        return now-started
                    time.sleep(period)
                    continue
                stable_since = None
                command = target
                status = "position_waypoint"
                can_move = True
                speed_limit = float(self.config.guided_max_forward_speed)
                planner_seconds = revalidation_seconds = 0.
                if local_planner is not None:
                    health = getattr(local_planner, 'check_execution_health', None)
                    if callable(health):
                        health()
                    update = worker.poll(current, target, self.get_velocity())
                    if update is None:
                        from planner.local_avoidance_planner import LocalPlan
                        plan = LocalPlan('waiting_for_plan', None, np.empty((0,3)), 0.)
                    else:
                        plan = worker.tracking_plan(update, current)
                        planner_seconds = update.duration
                        revalidate = getattr(local_planner, 'revalidate_plan', None)
                        if callable(revalidate):
                            validation_started = time.monotonic()
                            plan = revalidate(plan, current, certificate=update.certificate)
                            revalidation_seconds = time.monotonic()-validation_started
                    notify = getattr(local_planner, 'notify_execution', None)
                    if callable(notify):
                        notify(plan, certificate=None if update is None else update.certificate)
                    status = plan.status
                    speed_limit = min(speed_limit, float(plan.speed_limit))
                    can_move = plan.can_move and math.isfinite(speed_limit) and speed_limit > 0
                    if can_move:
                        command = self._vector3(plan.local_target_ned, "local waypoint")
                        if not np.isfinite(command).all():
                            raise ValueError("Planner returned a non-finite waypoint")
                command_time = time.monotonic()
                last_command_time = getattr(self, "_waypoint_command_time", None)
                control_dt = min(1.0, max(period, command_time - (
                    started-period if last_command_time is None else last_command_time)))
                self._waypoint_command_time = command_time
                # Bound heading changes when a new A* segment or avoidance
                # target is selected. Position safety checks remain unchanged.
                yaw_delta = math.atan2(math.sin(yaw-command_yaw), math.cos(yaw-command_yaw))
                command_yaw += float(np.clip(yaw_delta,
                    -self.config.guided_max_yaw_rate*control_dt,
                    self.config.guided_max_yaw_rate*control_dt))
                if (not can_move and status != 'waiting_for_plan'
                        and status != getattr(self, '_waypoint_warning_status', None)):
                    progress.finish()
                    print(f"--- 位置航点控制：{ROUTE_SOURCE_LABELS.get(status, status)}；目标NED={target.round(3).tolist()} ----")
                self._waypoint_warning_status = None if can_move else status
                last_status = status
                if not can_move:
                    if hold_target is None:
                        hold_target = current.copy()
                        hold_reason = status
                        braking = status == "braking_corridor_blocked"
                    elif status != 'waiting_for_plan':
                        hold_reason = status
                    clear_since = None
                elif hold_target is not None:
                    # Do not release braking on a single clear frame, including
                    # across rolling control-horizon calls.
                    if not braking and hold_reason == 'waiting_for_plan':
                        hold_target = None
                        hold_reason = None
                        clear_since = None
                    elif not braking and np.linalg.norm(self.get_velocity()) <= self.config.landed_speed_tolerance:
                        clear_since = command_time if clear_since is None else clear_since
                        if command_time-clear_since >= self.config.guided_stable_time:
                            hold_target = None
                            hold_reason = None
                            clear_since = None
                    else:
                        clear_since = None
                if hold_target is not None and braking:
                    # Zero horizontal position error requests deceleration from
                    # PX4 without pulling back toward the trigger location.
                    # Keep the original Z and yaw; latch XY once actually slow.
                    hold_target[:2] = current[:2]
                    if np.linalg.norm(self.get_velocity()[:2]) <= .05:
                        braking = False
                self._waypoint_braking = braking
                self.waypoint_control_state = {
                    "plan_status": status,
                    "status": ("braking_without_pullback" if braking else
                               "holding_for_stable_clearance" if hold_target is not None else status),
                    "target_ned": target.tolist(),
                    "hold_ned": None if hold_target is None else hold_target.tolist(),
                    "velocity_ned": self.get_velocity().tolist(),
                    "speed_limit": speed_limit,
                    "control_dt_seconds": control_dt,
                    "planning_seconds": command_time-now,
                    "local_planning_seconds": planner_seconds,
                    "post_planning_validation_seconds": revalidation_seconds,
                    "clearance_stable_seconds": 0. if clear_since is None else command_time-clear_since,
                    "flight_status": self._flight_status(),
                }
                self._waypoint_hold_target = hold_target
                self._waypoint_hold_reason = hold_reason
                self._waypoint_clear_since = clear_since
                progress.update(self.waypoint_control_state['status'],
                                moving=can_move and hold_target is None)
                if hold_target is not None:
                    lead_speed = 0.0
                    lead_acceleration = 0.0
                    stable_since = None
                    self._waypoint_position_reference = hold_target.copy()
                    self.waypoint_control_state["command_ned"] = hold_target.tolist()
                    self.send_position(hold_target, command_yaw)
                else:
                    if reached and pass_through:
                        # Continue through straight subgoals without snapping
                        # the position reference to the arrival-radius boundary.
                        return now-started
                    acceleration = self.config.guided_max_horizontal_acceleration
                    offset = command-current
                    horizontal = float(np.linalg.norm(offset[:2]))
                    previous_reference = getattr(self, "_waypoint_position_reference", None)
                    along = (0. if previous_reference is None or horizontal <= 1e-9 else
                             float(np.dot(previous_reference[:2]-current[:2], offset[:2]/horizontal)))
                    along = float(np.clip(along, 0., horizontal))
                    desired_lead = speed_limit
                    if not pass_through:
                        # Brake the moving reference before IT reaches the end,
                        # rather than waiting for the lagging vehicle to arrive.
                        # The gain term tapers the final approach; the extra
                        # time allows the acceleration ramp to unwind.
                        remaining = horizontal-along
                        ramp_speed = acceleration*acceleration/lead_jerk
                        desired_lead = min(desired_lead,
                                           self.config.guided_position_kp*remaining,
                                           math.sqrt(ramp_speed*ramp_speed + 2.*acceleration*remaining)-ramp_speed)
                    # Integrate real elapsed control time in small substeps.
                    # Slow map/planner calls must not turn 8 wall seconds into
                    # only a few tenths of a second of reference motion.
                    reference_distance = 0.0
                    steps = max(1, int(math.ceil(control_dt/period-1e-9)))
                    step_dt = control_dt/steps
                    for _ in range(steps):
                        error = desired_lead-lead_speed
                        # Taper acceleration before reaching cruise, rather than
                        # stepping immediately between zero and maximum acceleration.
                        requested_acceleration = math.copysign(
                            min(acceleration, math.sqrt(2.*lead_jerk*abs(error))), error
                        ) if abs(error) > 1e-9 else 0.0
                        lead_acceleration += float(np.clip(requested_acceleration-lead_acceleration,
                                                           -lead_jerk*step_dt, lead_jerk*step_dt))
                        next_speed = lead_speed + lead_acceleration*step_dt
                        if (desired_lead-lead_speed)*(desired_lead-next_speed) <= 0:
                            next_speed = desired_lead
                        # Emergency stops / tighter safety caps are never delayed.
                        lead_speed = max(0., min(speed_limit, next_speed))
                        reference_distance += lead_speed*step_dt
                    # Only shorten the currently certified chord. Blending old
                    # and new avoidance points could cut across an obstacle.
                    fraction = 1.0
                    if horizontal > 1e-9:
                        # The safety cap bounds lead, independently of the
                        # reference's deceleration. Shrinking lead with its
                        # speed would pull the target backwards while braking.
                        fraction = min(fraction, speed_limit /
                                       (self.config.guided_position_kp*horizontal))
                    if abs(offset[2]) > 1e-9:
                        fraction = min(fraction, self.config.guided_max_z_speed /
                                       (self.config.guided_z_kp*abs(offset[2])))
                    reference = current + offset*fraction
                    if horizontal > 1e-9:
                        # Continue the reference along this certified chord,
                        # instead of jumping to current position + full lead.
                        # Reproject only its scalar distance: never blend across
                        # a corner or an unchecked obstacle-avoidance chord.
                        distance = min(horizontal*fraction,
                                       along + reference_distance)
                        reference = current + offset*(distance/horizontal)
                    else:
                        # A height-only waypoint also needs a moving reference,
                        # not a fresh fixed lead above/below the measured pose.
                        previous_z = current[2] if previous_reference is None else previous_reference[2]
                        reference[2] = np.clip(reference[2],
                                               previous_z-self.config.guided_max_z_speed*control_dt,
                                               previous_z+self.config.guided_max_z_speed*control_dt)
                        reference[2] = np.clip(reference[2], min(current[2], command[2]), max(current[2], command[2]))
                    # Do not jump to the final target merely because the UAV
                    # entered its arrival radius. Finish the shaped reference
                    # and verify low measured speed before a stopping arrival.
                    reference_finished = (np.array_equal(command, target)
                                          and np.linalg.norm(reference-target) <=
                                          min(1e-3, .25*acceleration*period*period))
                    if reference_finished:
                        reference = target.copy()
                    self._waypoint_position_reference = reference.copy()
                    self.waypoint_control_state.update(
                        command_ned=reference.tolist(),
                        lead_speed=lead_speed, lead_acceleration=lead_acceleration,
                    )
                    self.send_position(reference, command_yaw)
                    if (not pass_through and reached and reference_finished
                            and np.linalg.norm(self.get_velocity()) <= .05):
                        stable_since = now if stable_since is None else stable_since
                        if now-stable_since >= self.config.guided_stable_time:
                            self._waypoint_lead_speed = 0.0
                            self._waypoint_lead_acceleration = 0.0
                            progress.finish()
                            return now-started
                    else:
                        stable_since = None
                self._waypoint_lead_speed = lead_speed
                self._waypoint_lead_acceleration = lead_acceleration
                time.sleep(max(0., period-(time.monotonic()-now)))
        except BaseException:
            progress.finish()
            try:
                self.hover()
            except Exception as hold_error:
                print(f"--- 航点异常后位置保持失败：{hold_error}；保留原始异常 ----")
            raise

    def align_yaw(self, yaw=0.0, max_yaw_rate=None, timeout=None, health_check=None):
        """锁住当前位置，通过限速的位置模式航向设定点原地转向。"""
        self._require_offboard()

        target_yaw = self._wrap_angle(float(yaw))
        yaw_kp = float(self.config.guided_yaw_kp)
        max_yaw_rate = float(
            self.config.guided_max_yaw_rate
            if max_yaw_rate is None
            else max_yaw_rate
        )
        tolerance = float(self.config.align_yaw_tolerance)
        stable_time = float(self.config.align_yaw_stable_time)
        timeout = float(self.config.align_yaw_timeout if timeout is None else timeout)
        update_rate = float(self.config.guided_update_rate)

        for name, value in {
            "yaw_kp": yaw_kp,
            "max_yaw_rate": max_yaw_rate,
            "tolerance": tolerance,
            "stable_time": stable_time,
            "timeout": timeout,
            "update_rate": update_rate,
        }.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")

        started_at = time.monotonic()
        deadline = started_at + timeout
        period = 1.0 / update_rate
        next_update = started_at
        previous_command_at = started_at
        stable_since = None
        current_yaw = self._wrap_angle(float(self.get_euler()[2]))
        yaw_error = self._wrap_angle(target_yaw - current_yaw)
        hold_position = self.get_position().copy()
        command_yaw = current_yaw

        while True:
            if health_check is not None:
                health_check()
            self.raise_if_collision()
            now = time.monotonic()
            if now >= deadline:
                self.hover()
                raise TimeoutError(
                    f"Yaw alignment timed out after {timeout:.1f} s: "
                    f"current={math.degrees(current_yaw):.1f} deg, "
                    f"target={math.degrees(target_yaw):.1f} deg, "
                    f"error={math.degrees(yaw_error):+.1f} deg"
                )

            current_yaw = self._wrap_angle(float(self.get_euler()[2]))
            yaw_error = self._wrap_angle(target_yaw - current_yaw)

            if abs(yaw_error) <= tolerance:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= stable_time:
                    self.hover()
                    return now - started_at
            else:
                stable_since = None

            command_error = self._wrap_angle(target_yaw-command_yaw)
            yaw_rate = float(np.clip(yaw_kp*command_error, -max_yaw_rate, max_yaw_rate))
            # Ramp the heading setpoint while keeping XYZ fixed. Actual yaw
            # feedback still determines arrival and timeout.
            command_dt = max(period, now-previous_command_at)
            yaw_step = float(np.clip(yaw_rate*command_dt, -abs(command_error), abs(command_error)))
            command_yaw = self._wrap_angle(command_yaw + yaw_step)
            previous_command_at = now
            self.send_position(hold_position, command_yaw)

            next_update += period
            sleep_time = next_update - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_update = time.monotonic()

    def raise_if_collision(self, include_ground=False):
        """
        停止当前设定点任务，并在报告碰撞时提高警报。
        地面接触默认会被忽略，所以正常起飞和着陆不会中止任务。
        静态场景和其他碰撞仍会被报告。
        """
        self._require_connection()

        target_id = 0
        object_name = ""

        # UE crash data distinguishes ground (-2), static scene objects (-1),
        # no collision (0), and another vehicle (positive CopterID).
        if self.ue is not None:
            ue_data = self.ue.getUE4Data(
                int(self.config.copter_id))
            if ue_data != 0:
                target_id = int(ue_data.CrashType)
                object_name = str(getattr(ue_data, "CrashedName", ""))

        # The short 12-byte UE crash message is a second UE-side fallback.
        if target_id == 0 and bool(
            getattr(self.ue, "isVehicleCrash", False)
        ):
            target_id = int(getattr(self.ue, "isVehicleCrashID", -10))

        # Fall back to the MAV controller for vehicle-to-vehicle collision
        # messages when detailed UE data is unavailable.
        if target_id == 0 and bool(
            getattr(self.mav, "isVehicleCrash", False)
        ):
            target_id = int(getattr(self.mav, "isVehicleCrashID", -10))

        if target_id == 0:
            return
        if target_id == -2 and not include_ground:
            return

        position = self.get_position()

        # Stop pursuing the old waypoint before handing control to the caller.
        # Capture BEFORE hover overwrites the position command that preceded impact.
        context = {
            "timestamp": time.time(),
            "position_ned": position.tolist(),
            "velocity_ned": self.get_velocity().tolist(),
            "euler_rpy": self.get_euler().tolist(),
            "last_position_command": deepcopy(getattr(self, "_last_position_command", None)),
            "waypoint_controller": deepcopy(getattr(self, "waypoint_control_state", None)),
            "landing_commanded": self._landing_commanded,
            "ground_down": self._ground_down,
            "flight_status": self._flight_status(),
        }
        if self.offboard_enabled:
            yaw = float(self.get_euler()[2])
            try:
                self.send_position(position, yaw)
            except Exception as error:
                context["hold_error"] = str(error)
                print(f"--- 碰撞后位置保持失败：{error} ----")

        print(
            "WARNING: 检测到碰撞 原地降落. "
            f"target_id={target_id}, position_ned={position.tolist()}, "
            f"object={object_name or 'unknown'}"
        )
        print(f"--- 碰撞前控制证据：{context} ----")
        raise CollisionDetectedError(target_id, position, object_name, context)

    def _flight_status(self):
        """Read available SDK/MAVLink state; do not infer actual mode from our flag."""
        messages = getattr(getattr(self.mav, "the_connection", None), "messages", {})
        heartbeat = messages.get("HEARTBEAT")
        battery = messages.get("BATTERY_STATUS")
        sender = getattr(self.mav, "t2", None)
        received = getattr(heartbeat, "_timestamp", None)
        return {
            "armed": bool(getattr(self.mav, "isArmed", False)),
            "offboard_requested": self.offboard_enabled,
            "offboard_sender_alive": None if sender is None else sender.is_alive(),
            "custom_mode": None if heartbeat is None else int(heartbeat.custom_mode),
            "base_mode": None if heartbeat is None else int(heartbeat.base_mode),
            "system_status": None if heartbeat is None else int(heartbeat.system_status),
            "heartbeat_age_seconds": None if received is None else time.time()-float(received),
            "battery_remaining_percent": None if battery is None else int(battery.battery_remaining),
            "battery_voltages_mv": None if battery is None else [int(v) for v in battery.voltages],
        }

    def send_velocity(self, velocity, yaw_rate=0.0):
        """发送 NED 速度指令。

        Args:
            velocity: [vn, ve, vd]，单位 m/s。
            yaw_rate: 偏航角速度，单位 rad/s。
        """
        self._require_offboard()

        vn, ve, vd = self._vector3(velocity, "velocity")

        self.mav.SendVelNED(
            float(vn),
            float(ve),
            float(vd),
            float(yaw_rate),
        )

    def hover(self):
        """保持调用时的当前位置。"""
        self._require_offboard()
        worker = getattr(self, '_waypoint_planning_worker', None)
        if worker is not None:
            worker.invalidate()
        progress = getattr(self, '_motion_progress_display', None)
        if progress is not None:
            progress.finish()

        position = self.get_position()
        yaw = float(self.get_euler()[2])

        self.send_position(position, yaw)
        self._waypoint_lead_speed = 0.0
        self._waypoint_lead_acceleration = 0.0
        self._waypoint_position_reference = None
        self._waypoint_command_time = None

    def _ue_landing_diagnostics(self):
        """Read the existing UE stream, independently of PX4's movable home."""
        ue = getattr(self, 'ue', None)
        if ue is None:
            return None
        packet = ue.getUE4Data(int(self.config.copter_id))
        if packet is None or isinstance(packet, (int, float)):
            return None
        stamp = float(getattr(packet, 'runnedTime', float('nan')))
        if not math.isfinite(stamp):
            return None
        now = time.monotonic()
        previous = getattr(self, '_ue_landing_sample', None)
        # Some simulator builds keep runnedTime constant. CopyData replaces
        # PosE on each received packet, even while the vehicle is stationary.
        packet_token = getattr(packet, 'PosE', None)
        changed = previous is None or packet_token is not previous[0]
        if changed:
            self._ue_landing_sample = (packet_token, now)
        age = 0.0 if changed else now-previous[1]
        def vector(name):
            value = np.asarray(getattr(packet, name, []), dtype=float)
            return value.tolist() if value.size and np.isfinite(value).all() else None
        return dict(timestamp=stamp, observed_age_seconds=age,
                    position=vector('PosE'), velocity=vector('VelE'),
                    motor_rpms=vector('MotorRPMS'), rays_raw=vector('ray'),
                    crash_type=int(getattr(packet, 'CrashType', 0)),
                    object_name=str(getattr(packet, 'CrashedName', '')))

    def _landing_height_diagnostics(self):
        """Keep independent altitude sources with ages; do not certify touchdown."""
        messages = getattr(getattr(self.mav, "the_connection", None), "messages", {})
        result = {}
        fields = {
            "GLOBAL_POSITION_INT": ("alt", "relative_alt", "lat", "lon"),
            "HOME_POSITION": ("altitude", "x", "y", "z"),
            "ALTITUDE": ("altitude_local", "altitude_relative", "altitude_terrain", "bottom_clearance"),
            "DISTANCE_SENSOR": ("current_distance", "min_distance", "max_distance", "orientation", "id"),
        }
        for name, names in fields.items():
            message = messages.get(name)
            if message is None:
                continue
            item = {}
            for key in names:
                value = getattr(message, key, None)
                item[key] = float(value) if value is not None and math.isfinite(float(value)) else None
            stamp = getattr(message, "_timestamp", None)
            item["age_seconds"] = (time.time()-float(stamp) if stamp is not None
                                   and math.isfinite(float(stamp)) else None)
            result[name] = item
        result["units"] = "MAVLink native: GLOBAL_POSITION_INT/HOME altitude mm; DISTANCE_SENSOR cm; ALTITUDE m"
        result["sdk_point_mode"] = bool(getattr(self.mav, "isInPointMode", False))
        result['ue_vehicle'] = self._ue_landing_diagnostics()
        baseline = getattr(self, '_landing_ground_sources', {}).get('ue_vehicle')
        current_ue = result['ue_vehicle']
        if (baseline and current_ue and baseline.get('position') is not None
                and current_ue.get('position') is not None):
            result['ue_position_change_from_takeoff_ground'] = (
                np.asarray(current_ue['position'])-np.asarray(baseline['position'])).tolist()
        result["sim_truth_received"] = bool(getattr(self.mav, "hasTrueDataRec", False))
        if result["sim_truth_received"]:
            position = np.asarray(self.mav.truePosNED, dtype=float)
            result["sim_position_world_ned"] = position.tolist() if np.isfinite(position).all() else None
            # SDK truth clock is not assumed to share the PX4 NED origin/timebase.
            stamp = float(getattr(self.mav, "trueTimeStmp", 0.))
            result["sim_timestamp_raw"] = stamp if math.isfinite(stamp) else None
        return result

    def land(self):
        """Use SDK land command; retire position streaming after AUTO.LAND acknowledgement."""
        self._require_connection()
        worker = getattr(self, '_waypoint_planning_worker', None)
        if worker is not None:
            worker.invalidate()
        if self._landing_commanded:
            self._wait_for_landing_handoff()
            return
        position = self.get_position()
        if getattr(self.mav, "the_connection", None) is not None:
            # Diagnostic telemetry only; does not change flight mode or setpoints.
            self.mav.SendMavCmdLong(511, 245, 200000)
        self.landing_confirmation = None
        self._landing_start_evidence = self._landing_height_diagnostics()
        self._landing_requested_at = time.time()
        self._landing_commanded = True
        self.mav.sendMavLand(float(position[0]), float(position[1]), 0.0)
        print("--- 已发送 PX4 降落指令 sendMavLand，等待 AUTO.LAND 接管 ----")
        self._wait_for_landing_handoff()

    def _wait_for_landing_handoff(self):
        """Complete mode handoff before callers start slow shutdown/checkpoints."""
        connection = getattr(self.mav, "the_connection", None)
        if connection is None:
            return  # Direct SDK mode exposes no MAVLink heartbeat cache.
        deadline = time.monotonic() + min(5., float(self.config.landing_timeout))
        while time.monotonic() < deadline:
            heartbeat = getattr(connection, "messages", {}).get("HEARTBEAT")
            stamp = getattr(heartbeat, "_timestamp", float('nan'))
            if (math.isfinite(stamp) and self._landing_requested_at is not None
                    and stamp > self._landing_requested_at and 0 <= time.time()-stamp < 2.):
                mode = int(heartbeat.custom_mode)
                if (mode >> 16) & 255 == 4 and (mode >> 24) & 255 == 6:
                    self._handoff_landing_stream(heartbeat)
                    return
            time.sleep(.05)
        # Permit an explicit retry; never claim a completed handoff on timeout.
        self._landing_commanded = False
        raise TimeoutError("PX4 did not acknowledge AUTO.LAND within 5 seconds; "
                           f"flight_status={self._flight_status()}")

    def _handoff_landing_stream(self, heartbeat):
        """Stop only the SDK sender, without a second mode or disarm command."""
        requested = self._landing_requested_at
        if not self.offboard_enabled or requested is None or heartbeat is None:
            return
        if heartbeat._timestamp <= requested:
            return
        mode = int(heartbeat.custom_mode)
        if (mode >> 16) & 255 != 4 or (mode >> 24) & 255 != 6:
            return
        sender = getattr(self.mav, "t2", None)
        if sender is None or not hasattr(self.mav, "isInOffboard"):
            return
        # SDK endOffboard also sends NAV_GUIDED_ENABLE(False). Avoid changing
        # the acknowledged landing mode; its sender exits on this flag alone.
        self.mav.isInOffboard = False
        sender.join(timeout=1.0)
        if sender.is_alive():
            raise TimeoutError("AUTO.LAND acknowledged but Offboard sender did not stop")
        self.offboard_enabled = False
        print("--- PX4 AUTO.LAND 已接管，停止旧悬停位置指令发送 ----")

    def wait_until_landed(self, timeout=None):
        """Confirm ground proximity and motion, checking PX4 landing telemetry."""
        self._require_connection()

        if timeout is None:
            timeout = float(self.config.landing_timeout)

        height_tolerance = float(self.config.landed_height_tolerance)
        speed_tolerance = float(self.config.landed_speed_tolerance)
        stable_time = float(self.config.landed_stable_time)
        deadline = time.monotonic() + float(timeout)
        stable_since = None
        first_sample = None
        last_flight_status = None
        conflict_since = None
        conflict_sample = None

        while time.monotonic() < deadline:
            connection = getattr(self.mav, "the_connection", None)
            messages = getattr(connection, "messages", {})
            now = time.time()

            def fresh(name):
                message = messages.get(name)
                stamp = getattr(message, "_timestamp", float('nan'))
                return message if math.isfinite(stamp) and 0 <= now-stamp < 2.0 else None

            heartbeat, local = fresh("HEARTBEAT"), fresh("LOCAL_POSITION_NED")
            self._handoff_landing_stream(heartbeat)
            extended = fresh("EXTENDED_SYS_STATE")
            requested = self._landing_requested_at
            if extended is not None and requested is not None and extended._timestamp < requested:
                extended = None
            landed_state = None if extended is None else int(extended.landed_state)
            telemetry_fresh = connection is None or (heartbeat is not None and local is not None)
            if local is not None:
                # Position and speed from one immutable telemetry packet.
                position = np.array([local.x, local.y, local.z], dtype=float)
                velocity = np.array([local.vx, local.vy, local.vz], dtype=float)
            else:
                position = self.get_position()
                velocity = self.get_velocity()
            height_error = abs(float(position[2]) - self._ground_down)
            speed = float(np.linalg.norm(velocity))
            position_finite = bool(np.isfinite(position).all())
            velocity_finite = bool(np.isfinite(velocity).all())
            finite = position_finite and velocity_finite and math.isfinite(height_error)

            # A stopped/disarmed simulator can freeze above the floor. Even
            # ON_GROUND must not bypass the measured ground-height check.
            ground_evidence = height_error <= height_tolerance and landed_state in (None, 0, 1)
            sample = None if local is None else local._timestamp
            if finite and telemetry_fresh and ground_evidence and speed <= speed_tolerance:
                if stable_since is None:
                    stable_since = time.monotonic()
                    first_sample = sample
            else:
                stable_since = None
            stable_for = 0. if stable_since is None else time.monotonic()-stable_since
            advancing = connection is None or (sample is not None and first_sample is not None and sample > first_sample)
            confirmed = stable_since is not None and stable_for >= stable_time and advancing
            self.landing_confirmation = {
                "confirmed": confirmed, "source": "stable_ground_height_and_speed",
                "position_ned": position.tolist() if position_finite else None,
                "velocity_ned": velocity.tolist() if velocity_finite else None,
                "ground_down": float(self._ground_down) if math.isfinite(self._ground_down) else None,
                "height_error_m": height_error if math.isfinite(height_error) else None,
                "speed_mps": speed if math.isfinite(speed) else None,
                "height_tolerance_m": height_tolerance, "speed_tolerance_mps": speed_tolerance,
                "stable_seconds": stable_for, "required_stable_seconds": stable_time,
                "armed_reported": bool(self.mav.isArmed),
                "px4_landed_state": landed_state, "telemetry_fresh": telemetry_fresh,
                "extended_state_age_seconds": None if extended is None else now-extended._timestamp,
                "flight_status": self._flight_status(),
                "height_sources": self._landing_height_diagnostics(),
                "landing_start_height_sources": getattr(self, "_landing_start_evidence", None),
                "takeoff_ground_height_sources": getattr(self, "_landing_ground_sources", None),
            }
            # A fresh PX4-disarmed/on-ground state can conflict with height.
            # Fail promptly rather than wait for stopped motors to descend;
            # this is NOT touchdown evidence and cannot enable another mission.
            disarmed_conflict = (finite and telemetry_fresh and heartbeat is not None
                and not (int(heartbeat.base_mode) & 128) and not bool(self.mav.isArmed)
                and landed_state == 1 and height_error > height_tolerance
                and speed <= speed_tolerance)
            if disarmed_conflict:
                if conflict_since is None:
                    conflict_since, conflict_sample = time.monotonic(), sample
                if (time.monotonic()-conflict_since >= stable_time and sample is not None
                        and conflict_sample is not None and sample > conflict_sample):
                    self.landing_confirmation['failure_reason'] = 'disarmed_height_conflict'
                    return False
            else:
                conflict_since = None
            flight = self.landing_confirmation["flight_status"]
            state_key = (flight["custom_mode"], flight["armed"], landed_state)
            if state_key != last_flight_status:
                last_flight_status = state_key
                print(f"--- PX4 降落反馈：mode={flight['custom_mode']}，armed={flight['armed']}，"
                      f"landed_state={landed_state}，距起飞地面={height_error:.2f} m ----")
            if confirmed:
                if not bool(self.mav.isArmed):
                    self.armed = False
                return True

            time.sleep(0.1)

        return False

    def get_position(self):
        """获取无人机位置。

        Returns:
            包含 ``[north, east, down]`` 的 NumPy 数组，单位为米。
        """
        self._require_connection()
        return self._copy_vector(self.mav.uavPosNED)

    def get_velocity(self):
        """获取无人机速度。

        Returns:
            包含 ``[vn, ve, vd]`` 的 NumPy 数组，单位为 m/s。
        """
        self._require_connection()
        return self._copy_vector(self.mav.uavVelNED)

    def get_euler(self):
        """获取无人机欧拉角。

        Returns:
            包含 ``[roll, pitch, yaw]`` 的 NumPy 数组，单位为 rad。
        """
        self._require_connection()
        return self._copy_vector(self.mav.uavAngEular)


    def get_angular_velocity(self):
        """获取机体系 FRD 角速度 [roll_rate, pitch_rate, yaw_rate]。"""
        self._require_connection()
        return self._copy_vector(self.mav.uavAngRate)

    def get_drone_state(self, timestamp=None, *, pose_messages=None):
        """对当前 PX4 状态做一次完整快照。

        MAVLink 模式按完整消息读取，使用 ATTITUDE 接收时间，避免轮询
        延迟以及旧 ATTITUDE_QUATERNION 混入当前姿态。位置消息的源时间
        差单独交给同步器检查；不把尚未同步的位置外推为精确测量。
        """
        self._require_connection()
        from sensors.empirical_pose_covariance import position_covariance_from_odometry

        before = time.time()
        messages = getattr(getattr(self.mav, "the_connection", None), "messages", {})
        if pose_messages is not None:
            messages = pose_messages
        attitude, local_position = messages.get("ATTITUDE"), messages.get("LOCAL_POSITION_NED")
        pose_sync_error = 0.0
        if attitude is not None and local_position is not None:
            # Cached message objects remain intact when the receiver publishes
            # their replacements; SDK component arrays are updated in-place.
            euler = np.array([attitude.roll, attitude.pitch, attitude.yaw], dtype=float)
            angular_velocity = np.array([attitude.rollspeed, attitude.pitchspeed, attitude.yawspeed], dtype=float)
            position = np.array([local_position.x, local_position.y, local_position.z], dtype=float)
            velocity = np.array([local_position.vx, local_position.vy, local_position.vz], dtype=float)
            boot_timestamp = float(attitude.time_boot_ms) / 1000.0
            pose_sync_error = abs(float(attitude.time_boot_ms) - float(local_position.time_boot_ms)) / 1000.0
            received_at = float(attitude._timestamp)
        else:
            # Direct SDK/UDP mode has no MAVLink message cache.
            position, velocity = self.get_position(), self.get_velocity()
            euler, angular_velocity = self.get_euler(), self.get_angular_velocity()
            boot_timestamp = float(getattr(self.mav, "uavTimeStmp", 0.0))
            received_at = before
        after = time.time()
        sample_timestamp = (
            received_at if timestamp is None else float(timestamp)
        )
        return DroneState(
            timestamp=sample_timestamp,
            position_xyz=position,
            quaternion=self._euler_to_quaternion(euler),
            linear_velocity=velocity,
            angular_velocity=angular_velocity,
            euler_rpy=euler,
            px4_boot_timestamp=boot_timestamp,
            pose_sync_error=pose_sync_error,
            position_covariance_ned=position_covariance_from_odometry(
                messages.get("ODOMETRY"),
                after,
            ),
        )

    @staticmethod
    def _euler_to_quaternion(euler):
        """将 NED/FRD 的 ZYX 欧拉角转换为 [w, x, y, z]。"""
        roll, pitch, yaw = np.asarray(euler, dtype=float)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return np.asarray(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=float,
        )

    def stop_waypoint_planning(self):
        worker = getattr(self, '_waypoint_planning_worker', None)
        if worker is not None:
            worker.close()
            self._waypoint_planning_worker = None

    def finish_flight_session(self):
        """Confirm touchdown and reset Offboard state while retaining SDK connections."""
        self._require_connection()
        self.stop_waypoint_planning()
        if not self.wait_until_landed():
            raise TimeoutError("Landing not confirmed; another flight cannot be started; "
                               f"evidence={getattr(self, 'landing_confirmation', None)}")
        if bool(self.mav.isArmed):
            self.disarm(force=True)
        if self.offboard_enabled:
            self.mav.endOffboard()
        self.offboard_enabled = False
        self.armed = False
        self._landing_commanded = False
        self._landing_requested_at = None
        self._shutdown_disarmed = False

    def close(self, disarm=False):
        """关闭控制接口。

        Args:
            disarm: 是否同时强制上锁。无人机未落地时不要设为 True。
        """
        self.stop_waypoint_planning()
        if self.mav is None:
            if self.ue is not None and not self.ue.stopFlagUE4:
                self.ue.endUE4MsgRec()
            return

        actually_armed = bool(self.armed or self.mav.isArmed)
        if actually_armed:
            if not disarm:
                raise RuntimeError(
                    "Vehicle is still armed; closing PX4MavCtrlV4 would force "
                    "a disarm. Land first and use close(disarm=True)."
                )

            if not self.wait_until_landed():
                if (self.landing_confirmation or {}).get('failure_reason') == 'disarmed_height_conflict':
                    print('--- PX4 已停桨，但高度与落地状态冲突；关闭通信，未确认落地，禁止继续任务 ----')
                    self.armed = False
                else:
                    raise TimeoutError(
                        "Landing was not confirmed before timeout. To prevent an "
                        "in-air motor stop, disarm and communication shutdown were "
                        f"not performed. evidence={self.landing_confirmation}"
                    )

            # The kinematic fallback may confirm touchdown before PX4's own
            # automatic-disarm delay has expired.
            if bool(self.mav.isArmed):
                self.disarm(force=True)
            else:
                self.armed = False
                self._shutdown_disarmed = True

        if self.offboard_enabled:
            self.mav.endOffboard()
            self.offboard_enabled = False

        self.mav.stopRun()

        if self.ue is not None and not self.ue.stopFlagUE4:
            self.ue.endUE4MsgRec()

        self.connected = False
        shutdown_events = []
        if self._landing_commanded and (self.landing_confirmation or {}).get('confirmed'):
            shutdown_events.append("无人机降落")
        if self._shutdown_disarmed:
            shutdown_events.append("无人机上锁")
        shutdown_events.append("多旋翼控制接口已关闭")
        print(f"--- {' → '.join(shutdown_events)} ----")

    def _require_connection(self):
        if not self.connected or self.mav is None:
            raise RuntimeError("尚未连接无人机，请先调用 connect()")

    def _require_offboard(self):
        self._require_connection()

        if not self.offboard_enabled:
            raise RuntimeError(
                "尚未进入 Offboard 模式，请先调用 enable_offboard()"
            )

    @staticmethod
    def _vector3(value: Sequence[float], name: str):
        vector = np.asarray(value, dtype=float)

        if vector.size != 3:
            raise ValueError(f"{name} 必须包含三个元素")

        return vector.reshape(3)

    @staticmethod
    def _copy_vector(value):
        return np.asarray(value[:3], dtype=float).copy()

    @staticmethod
    def _wrap_angle(angle):
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
