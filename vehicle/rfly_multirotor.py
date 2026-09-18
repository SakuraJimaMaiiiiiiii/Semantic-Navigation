"""RflySim/PX4 多旋翼统一控制接口。"""

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

import PX4MavCtrlV4 as PX4MavCtrl
import ReqCopterSim
import UE4CtrlAPI


ROUTE_SOURCE_LABELS = {
    "ego_clear": "EGO B样条生成的直达轨迹",
    "ego_avoid": "EGO A*种子优化后的三维绕障轨迹",
    "depth_guard_clear": "深度估计生成的直线路径",
    "avoid": "A* 生成的绕障路径",
    "clear": "A* 地图验证后的直线路径",
    "sensor_clear_fallback": "深度与地图安全兜底直线路径",
}

DIRECT_ROUTE_STATUSES = {
    "ego_clear",
    "depth_guard_clear",
    "clear",
    "sensor_clear_fallback",
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


class CollisionDetectedError(RuntimeError):
    """Raised when RflySim reports a collision during a flight task."""

    def __init__(self, target_id, position, object_name=""):
        self.target_id = int(target_id)
        self.position = np.asarray(position, dtype=float).copy()
        self.object_name = str(object_name)
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
        self._shutdown_disarmed = False

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
            print("--- 进入 Offboard 模式…… ----")
            self.mav.initOffboard()
            time.sleep(0.5)
            self.offboard_enabled = True

            # RflySimSDK 的 initOffboard() 在 MAVLink 模式下会自动发送
            # 解锁命令，因此应同步真实状态，避免再次解锁。
            self.armed = bool(self.mav.isArmed)

        if arm and not self.armed:
            self.arm()

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

    def takeoff(self, height, yaw=0.0):
        """通过 Offboard 位置控制起飞。

        Args:
            height: 相对起飞点高度，单位 m，必须为正数。
            yaw: 偏航角，单位 rad。
        """
        height = float(height)

        if height <= 0:
            raise ValueError("起飞高度必须大于 0")

        self._require_offboard()

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

    def move_velocity_guided(
        self,
        waypoint,
        yaw=0.0,
        face_direction=True,
        max_forward_speed=None,
        max_z_speed=None,
        timeout=None,
        local_planner=None,
        pass_through=False,
    ):
        """使用位置外环生成机体系前向速度和 yaw 角速度飞向目标点。

        位置误差只用于计算运动方向、速度大小和到达条件。飞行过程中持续
        发送 ``SendVelFRD``，使前视相机通过弧形转向自然面向运动方向；
        到达后切换到当前位置悬停。

        Args:
            waypoint: 目标 ``[north, east, down]``，采用本地 NED，单位 m。
            yaw: ``face_direction=False`` 时使用的固定目标 yaw，单位 rad。
            face_direction: 是否根据当前位置到目标点的方向实时计算机头朝向。
            max_forward_speed: 最大机头前向速度，单位 m/s。
            max_z_speed: 最大升降速度，单位 m/s。
            timeout: 最长导航时间，单位秒。
            local_planner: 可选局部避障规划器。不可安全移动时将悬停等待。
            pass_through: 是否把目标作为连续通过点，不悬停、不等待稳定。

        Returns:
            实际导航耗时，单位秒。
        """
        self._require_offboard()
        target = self._vector3(waypoint, "waypoint")

        max_forward_speed = float(
            self.config.guided_max_forward_speed
            if max_forward_speed is None
            else max_forward_speed
        )
        max_horizontal_acceleration = float(
            self.config.guided_max_horizontal_acceleration
        )
        max_z_speed = float(
            self.config.guided_max_z_speed
            if max_z_speed is None
            else max_z_speed
        )
        position_kp = float(self.config.guided_position_kp)
        z_kp = float(self.config.guided_z_kp)
        z_kd = float(self.config.guided_z_kd)
        max_z_acceleration = float(self.config.guided_max_z_acceleration)
        z_target_filter_time_constant = float(
            self.config.guided_z_target_filter_time_constant
        )
        yaw_kp = float(self.config.guided_yaw_kp)
        max_yaw_rate = float(self.config.guided_max_yaw_rate)
        yaw_stop_angle = float(self.config.guided_yaw_stop_angle)
        xy_tolerance = float(self.config.guided_xy_tolerance)
        z_tolerance = float(self.config.guided_z_tolerance)
        stable_time = float(self.config.guided_stable_time)
        pass_through_radius = float(
            self.config.guided_pass_through_radius
        )
        timeout = float(
            self.config.guided_timeout if timeout is None else timeout
        )
        update_rate = float(self.config.guided_update_rate)

        positive_values = {
            "max_forward_speed": max_forward_speed,
            "max_horizontal_acceleration": max_horizontal_acceleration,
            "max_z_speed": max_z_speed,
            "position_kp": position_kp,
            "z_kp": z_kp,
            "z_kd": z_kd,
            "max_z_acceleration": max_z_acceleration,
            "z_target_filter_time_constant": z_target_filter_time_constant,
            "yaw_kp": yaw_kp,
            "max_yaw_rate": max_yaw_rate,
            "yaw_stop_angle": yaw_stop_angle,
            "xy_tolerance": xy_tolerance,
            "z_tolerance": z_tolerance,
            "stable_time": stable_time,
            "pass_through_radius": pass_through_radius,
            "timeout": timeout,
            "update_rate": update_rate,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        if yaw_stop_angle > math.pi:
            raise ValueError("guided_yaw_stop_angle must not exceed pi")
        if pass_through_radius <= xy_tolerance:
            raise ValueError(
                "guided_pass_through_radius must exceed "
                "guided_xy_tolerance"
            )

        fixed_yaw = self._wrap_angle(float(yaw))
        period = 1.0 / update_rate
        started_at = time.monotonic()
        deadline = started_at + timeout
        next_update = started_at
        stable_since = None
        last_plan_status = None
        filtered_down_target = None
        commanded_down_speed = 0.0
        last_vertical_update = started_at
        last_motion_velocity = None
        last_motion_update = None
        filtered_acceleration = np.zeros(3, dtype=float)
        commanded_horizontal_velocity = None
        commanded_forward_speed = None
        last_horizontal_update = started_at

        while True:
            # 航点导航已经发生在起飞之后，此时地面接触也属于异常，
            # 不再像起飞和降落阶段那样忽略 CrashType=-2。
            self.raise_if_collision(include_ground=True)
            now = time.monotonic()
            if now >= deadline:
                self.hover()
                position = self.get_position()
                final_delta = target - position
                print(
                    "导航结果：未到达，"
                    f"目标NED={target.round(3).tolist()}，"
                    f"实际NED={position.round(3).tolist()}，"
                    f"水平误差={np.linalg.norm(final_delta[:2]):.2f} m，"
                    f"高度误差={abs(float(final_delta[2])):.2f} m"
                )
                raise TimeoutError(
                    f"Velocity-guided move timed out after {timeout:.1f} s"
                )

            position = self.get_position()
            current_velocity = self.get_velocity()
            delta = target - position
            distance_xy = float(np.linalg.norm(delta[:2]))
            down_error = float(delta[2])
            current_yaw = self._wrap_angle(float(self.get_euler()[2]))

            if (
                pass_through
                and distance_xy <= pass_through_radius
                and abs(down_error) <= z_tolerance
            ):
                elapsed = now - started_at
                print(
                    "导航结果：已连续通过中间航点，"
                    f"目标NED={target.round(3).tolist()}，"
                    f"实际NED={position.round(3).tolist()}，"
                    f"水平误差={distance_xy:.2f} m，"
                    f"耗时={elapsed:.1f} s"
                )
                return elapsed

            active_target = target
            planner_speed_limit = max_forward_speed
            using_local_target = False
            using_direct_heading = False
            if local_planner is not None:
                motion_state_updater = getattr(
                    local_planner,
                    "set_motion_state",
                    None,
                )
                if callable(motion_state_updater):
                    if (
                        last_motion_velocity is not None
                        and last_motion_update is not None
                    ):
                        motion_dt = max(1e-3, now - last_motion_update)
                        raw_acceleration = (
                            current_velocity - last_motion_velocity
                        ) / motion_dt
                        configured_acceleration = float(getattr(
                            getattr(local_planner, "config", None),
                            "ego_max_acceleration",
                            1.2,
                        ))
                        raw_acceleration = np.clip(
                            raw_acceleration,
                            -2.0 * configured_acceleration,
                            2.0 * configured_acceleration,
                        )
                        filtered_acceleration += 0.2 * (
                            raw_acceleration - filtered_acceleration
                        )
                    motion_state_updater(
                        current_velocity,
                        filtered_acceleration,
                    )
                    last_motion_velocity = current_velocity.copy()
                    last_motion_update = now
                plan = local_planner.plan(position, target)
                if plan.status != last_plan_status:
                    obstacle_text = (
                        "inf"
                        if not math.isfinite(plan.obstacle_distance)
                        else f"{plan.obstacle_distance:.2f} m"
                    )
                    speed_text = (
                        "vehicle-limit"
                        if not math.isfinite(plan.speed_limit)
                        else f"{plan.speed_limit:.2f} m/s"
                    )
                    map_distance = float(getattr(
                        local_planner,
                        "last_map_obstacle_distance",
                        math.inf,
                    ))
                    forward_clearance = float(getattr(
                        local_planner,
                        "last_forward_clearance",
                        math.inf,
                    ))
                    map_text = (
                        "inf"
                        if not math.isfinite(map_distance)
                        else f"{map_distance:.2f} m"
                    )
                    forward_text = (
                        "inf"
                        if not math.isfinite(forward_clearance)
                        else f"{forward_clearance:.2f} m"
                    )
                    route_source = ROUTE_SOURCE_LABELS.get(
                        plan.status,
                        "停止：当前没有可执行路径",
                    )
                    detail = (
                        f"状态={plan.status} | 障碍={obstacle_text} | "
                        f"地图={map_text} | 前向深度={forward_text} | "
                        f"限速={speed_text}"
                    )
                    width = max(len(route_source), len(detail)) + 4
                    border = "-" * width
                    print(
                        f"\n{border}\n"
                        f"  当前路径来源：{route_source}\n"
                        f"  {detail}\n"
                        f"{border}"
                    )
                    last_plan_status = plan.status
                if not plan.can_move:
                    self.send_body_velocity([0.0, 0.0, 0.0])
                    commanded_horizontal_velocity = np.zeros(2, dtype=float)
                    commanded_forward_speed = 0.0
                    last_horizontal_update = now
                    commanded_down_speed = 0.0
                    filtered_down_target = float(position[2])
                    last_vertical_update = now
                    next_update += period
                    sleep_time = next_update - time.monotonic()
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
                    else:
                        next_update = time.monotonic()
                    continue
                active_target = np.asarray(
                    plan.local_target_ned,
                    dtype=float,
                )
                using_local_target = True
                using_direct_heading = plan.status in DIRECT_ROUTE_STATUSES
                trajectory_target = getattr(
                    local_planner,
                    "tracking_target",
                    None,
                )
                if callable(trajectory_target):
                    moving_target = trajectory_target(now=now)
                    if moving_target is not None:
                        active_target = np.asarray(
                            moving_target,
                            dtype=float,
                        )
                planner_speed_limit = min(
                    max_forward_speed,
                    float(plan.speed_limit),
                )

            active_delta = active_target - position
            active_distance_xy = float(np.linalg.norm(active_delta[:2]))
            active_target_tolerance = (
                min(xy_tolerance, 0.05)
                if using_local_target
                else xy_tolerance
            )
            heading_delta = delta if using_direct_heading else active_delta
            heading_distance_xy = (
                distance_xy if using_direct_heading else active_distance_xy
            )

            if (
                face_direction
                and heading_distance_xy > active_target_tolerance
            ):
                desired_yaw = math.atan2(
                    float(heading_delta[1]),
                    float(heading_delta[0]),
                )
            elif face_direction:
                desired_yaw = current_yaw
            else:
                desired_yaw = fixed_yaw

            # 每一轮都重新计算最短航向误差，结果始终位于 [-pi, pi)。
            yaw_error = self._wrap_angle(desired_yaw - current_yaw)
            yaw_rate = float(
                np.clip(yaw_kp * yaw_error, -max_yaw_rate, max_yaw_rate)
            )

            if pass_through:
                # 中间返航点是路径采样点，不是停车点。保持规划器给出的
                # 通过速度，避免每隔一个航段先抬头刹车、再低头加速。
                requested_horizontal_speed = planner_speed_limit
            else:
                requested_horizontal_speed = min(
                    planner_speed_limit,
                    position_kp * (
                        distance_xy
                        if using_local_target
                        else active_distance_xy
                    ),
                )
            if face_direction and abs(yaw_error) >= yaw_stop_angle:
                requested_horizontal_speed = 0.0
            elif face_direction:
                # 航向偏差增大时连续减速，避免边大角度转向边向侧面飞。
                requested_horizontal_speed *= max(
                    0.0,
                    math.cos(yaw_error),
                )
            if active_distance_xy <= active_target_tolerance:
                requested_horizontal_speed = 0.0

            horizontal_dt = max(1e-3, now - last_horizontal_update)
            max_horizontal_step = (
                max_horizontal_acceleration * horizontal_dt
            )
            horizontal_direction = np.zeros(2, dtype=float)
            if active_distance_xy > active_target_tolerance:
                horizontal_direction = active_delta[:2] / active_distance_xy
            if face_direction:
                if commanded_forward_speed is None:
                    commanded_forward_speed = float(
                        np.linalg.norm(current_velocity[:2])
                    )
                horizontal_speed = float(np.clip(
                    requested_horizontal_speed,
                    commanded_forward_speed - max_horizontal_step,
                    commanded_forward_speed + max_horizontal_step,
                ))
                commanded_forward_speed = horizontal_speed
            else:
                requested_horizontal_velocity = (
                    requested_horizontal_speed * horizontal_direction
                )
                if commanded_horizontal_velocity is None:
                    commanded_horizontal_velocity = np.asarray(
                        current_velocity[:2],
                        dtype=float,
                    ).copy()
                horizontal_change = (
                    requested_horizontal_velocity
                    - commanded_horizontal_velocity
                )
                change_norm = float(np.linalg.norm(horizontal_change))
                if change_norm > max_horizontal_step:
                    horizontal_change *= (
                        max_horizontal_step / max(change_norm, 1e-9)
                    )
                commanded_horizontal_velocity += horizontal_change
            last_horizontal_update = now

            # 直达航段的高度基准必须固定在最终目标，不跟随
            # 5 Hz 重规划时不断重建的 B 样条前视点。绕障航段仍使用
            # 三维轨迹高度，但先低通滤波，避免拓扑切换直接变成
            # 升降速度阶跃。
            raw_down_target = float(
                target[2] if using_direct_heading else active_target[2]
            )
            vertical_dt = max(1e-3, now - last_vertical_update)
            if filtered_down_target is None:
                filtered_down_target = float(position[2])
            target_alpha = vertical_dt / (
                z_target_filter_time_constant + vertical_dt
            )
            filtered_down_target += target_alpha * (
                raw_down_target - filtered_down_target
            )
            active_down_error = filtered_down_target - float(position[2])

            vertical_velocity = float(current_velocity[2])
            requested_down_speed = float(
                np.clip(
                    z_kp * active_down_error - z_kd * vertical_velocity,
                    -max_z_speed,
                    max_z_speed,
                )
            )
            if abs(raw_down_target - float(position[2])) <= z_tolerance:
                # 进入高度保持区后立即交给 PX4 的速度环刹停，
                # 不再因前视点小幅变化反复翻转指令方向。
                requested_down_speed = 0.0

            max_z_speed_step = max_z_acceleration * vertical_dt
            down_speed = float(np.clip(
                requested_down_speed,
                commanded_down_speed - max_z_speed_step,
                commanded_down_speed + max_z_speed_step,
            ))
            commanded_down_speed = down_speed
            last_vertical_update = now

            arrived = (
                distance_xy <= xy_tolerance
                and abs(down_error) <= z_tolerance
            )
            if arrived:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= stable_time:
                    self.hover()
                    elapsed = now - started_at
                    position = self.get_position()
                    final_delta = target - position
                    print(
                        "导航结果：已到达，"
                        f"目标NED={target.round(3).tolist()}，"
                        f"实际NED={position.round(3).tolist()}，"
                        f"水平误差={np.linalg.norm(final_delta[:2]):.2f} m，"
                        f"高度误差={abs(float(final_delta[2])):.2f} m，"
                        f"耗时={elapsed:.1f} s"
                    )
                    return elapsed
            else:
                stable_since = None

            if face_direction:
                self.send_body_velocity(
                    [horizontal_speed, 0.0, down_speed],
                    yaw_rate=yaw_rate,
                )
            else:
                self.send_velocity(
                    [
                        commanded_horizontal_velocity[0],
                        commanded_horizontal_velocity[1],
                        down_speed,
                    ],
                    yaw_rate=yaw_rate,
                )

            next_update += period
            sleep_time = next_update - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_update = time.monotonic()

    def align_yaw(self, yaw=0.0, max_yaw_rate=None):
        """通过最短 yaw 误差和角速度控制原地调整机头方向。"""
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
        timeout = float(self.config.align_yaw_timeout)
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
        stable_since = None
        current_yaw = self._wrap_angle(float(self.get_euler()[2]))
        yaw_error = self._wrap_angle(target_yaw - current_yaw)

        while True:
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

            yaw_rate = float(
                np.clip(yaw_kp * yaw_error, -max_yaw_rate, max_yaw_rate)
            )
            self.send_body_velocity([0.0, 0.0, 0.0], yaw_rate=yaw_rate)

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
        if self.offboard_enabled:
            yaw = float(self.get_euler()[2])
            self.send_position(position, yaw)

        print(
            "WARNING: 检测到碰撞 原地降落. "
            f"target_id={target_id}, position_ned={position.tolist()}, "
            f"object={object_name or 'unknown'}"
        )
        raise CollisionDetectedError(target_id, position, object_name)

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

    def send_body_velocity(self, velocity, yaw_rate=0.0):
        """发送机体系 FRD 速度指令。

        Args:
            velocity: ``[forward, right, down]``，采用机体系 FRD，单位为
                m/s。正值分别表示向前、向右和向下。
            yaw_rate: 偏航角速度，单位为 rad/s。
        """
        self._require_offboard()

        vx, vy, vz = self._vector3(velocity, "velocity")

        self.mav.SendVelFRD(
            float(vx),
            float(vy),
            float(vz),
            float(yaw_rate),
        )

    def hover(self):
        """保持调用时的当前位置。"""
        self._require_offboard()

        position = self.get_position()
        yaw = float(self.get_euler()[2])

        self.send_position(position, yaw)

    def land(self):
        """在当前位置执行 PX4 自动降落。"""
        self._require_connection()

        position = self.get_position()

        self._landing_commanded = True
        self.mav.sendMavLand(
            float(position[0]),
            float(position[1]),
            0.0,
        )

    def wait_until_landed(self, timeout=None):
        """Wait until PX4 auto-disarms or touchdown is stably detected.

        PX4MavCtrlV4 does not expose MAV_LANDED_STATE, so the fallback check
        uses local-NED height and total speed for a continuous stable period.
        """
        self._require_connection()

        if timeout is None:
            timeout = float(self.config.landing_timeout)

        height_tolerance = float(self.config.landed_height_tolerance)
        speed_tolerance = float(self.config.landed_speed_tolerance)
        stable_time = float(self.config.landed_stable_time)
        deadline = time.monotonic() + float(timeout)
        stable_since = None

        while time.monotonic() < deadline:
            # Automatic post-landing disarm is the strongest signal exposed by
            # this SDK and takes precedence over the kinematic fallback.
            if not bool(self.mav.isArmed):
                self.armed = False
                return True

            position = self.get_position()
            velocity = self.get_velocity()
            height_error = abs(float(position[2]) - self._ground_down)
            speed = float(np.linalg.norm(velocity))

            if height_error <= height_tolerance and speed <= speed_tolerance:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= stable_time:
                    return True
            else:
                stable_since = None

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

    def get_quaternion(self):
        """获取机体 FRD 到世界 NED 的姿态四元数 [w, x, y, z]。

        若当前 PX4 数据流没有发送 ``ATTITUDE_QUATERNION``，则使用最新
        roll、pitch、yaw 计算四元数，避免返回 SDK 的初始全零值。
        """
        self._require_connection()
        quaternion = np.asarray(
            getattr(self.mav, "uavAngQuatern", [0.0] * 4),
            dtype=float,
        ).reshape(-1)
        if quaternion.size == 4:
            norm = float(np.linalg.norm(quaternion))
            if math.isfinite(norm) and norm > 1e-6:
                return (quaternion / norm).copy()
        return self._euler_to_quaternion(self.get_euler())

    def get_angular_velocity(self):
        """获取机体系 FRD 角速度 [roll_rate, pitch_rate, yaw_rate]。"""
        self._require_connection()
        return self._copy_vector(self.mav.uavAngRate)

    def get_drone_state(self, timestamp=None):
        """对当前 PX4 状态做一次完整快照。

        ``timestamp`` 默认采用读取前后的主机 Unix 时间中点。RflySim 图像
        时间戳同样使用 Unix 秒，因此该状态可以进入时间同步缓存。
        """
        self._require_connection()
        from sensors.empirical_pose_covariance import position_covariance_from_odometry

        before = time.time()
        position = self.get_position()
        velocity = self.get_velocity()
        euler = self.get_euler()
        angular_velocity = self.get_angular_velocity()
        quaternion = self.get_quaternion()
        after = time.time()
        sample_timestamp = (
            0.5 * (before + after) if timestamp is None else float(timestamp)
        )
        return DroneState(
            timestamp=sample_timestamp,
            position_xyz=position,
            quaternion=quaternion,
            linear_velocity=velocity,
            angular_velocity=angular_velocity,
            euler_rpy=euler,
            px4_boot_timestamp=float(
                getattr(self.mav, "uavTimeStmp", 0.0)
            ),
            position_covariance_ned=position_covariance_from_odometry(
                getattr(getattr(self.mav, "the_connection", None), "messages", {}).get("ODOMETRY"),
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

    def close(self, disarm=False):
        """关闭控制接口。

        Args:
            disarm: 是否同时强制上锁。无人机未落地时不要设为 True。
        """
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
                raise TimeoutError(
                    "Landing was not confirmed before timeout. To prevent an "
                    "in-air motor stop, disarm and communication shutdown were "
                    "not performed."
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
        if self._landing_commanded:
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
