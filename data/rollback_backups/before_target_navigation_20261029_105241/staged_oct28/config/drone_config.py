"""RflySim 无人机配置。"""

from dataclasses import dataclass


@dataclass
class Config:
    """RflySim/PX4 无人机运行配置。

    Attributes:
        copter_id: CopterSim 中的无人机编号，从 1 开始。
        reset_initial_pose: 连接时是否强制重设水平位置和航向。PX4 已完成
            EKF 初始化后不建议启用，否则可能触发位置估计错误。
        initial_xy_yaw: 初始 ``(north, east, yaw)``。前两项采用 NED
            坐标系，单位为米；yaw 单位为度。仅在
            ``reset_initial_pose=True`` 时生效。
        ekf_timeout: 等待 PX4 EKF 三维定位完成的最长时间，单位为秒。
        arm_timeout: 等待 PX4 确认解锁的最长时间，单位为秒。
    """

    copter_id: int = 1
    reset_initial_pose: bool = False
    initial_xy_yaw: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ekf_timeout: float = 30.0
    arm_timeout: float = 5.0
    landing_timeout: float = 30.0
    landed_height_tolerance: float = 0.35
    landed_speed_tolerance: float = 0.20
    landed_stable_time: float = 2.0
    guided_max_forward_speed: float = 0.30  # 水平引导上限，m/s；位置模式用于参考点推进
    guided_max_horizontal_acceleration: float = 0.12  # 水平参考速度变化率，m/s²；降低起停俯仰
    guided_position_lead_jerk: float = 0.20  # 位置目标前伸量的加速度变化率，m/s³；非飞控物理限幅
    guided_max_z_speed: float = 0.12  # 垂直位置参考推进上限，单位 m/s
    guided_position_kp: float = 0.6  # 水平位置参考前伸增益（限速/增益得到前伸距离）
    guided_z_kp: float = 0.6  # 垂直位置参考前伸增益
    guided_yaw_kp: float = 1.2  # 航向误差到 yaw 角速度的比例增益
    guided_max_yaw_rate: float = 0.3490658504  # 最大 yaw 角速度，单位 rad/s（20 deg/s）
    guided_xy_tolerance: float = 0.05  # 最终到点水平容差，单位 m
    guided_z_tolerance: float = 0.15  # 到达目标的 NED 高度容差，单位 m
    guided_stable_time: float = 0.5  # 连续满足位置容差后确认到达的时间，单位秒
    guided_timeout: float = 90.0  # 单次航点控制允许的最长时间，单位秒
    guided_update_rate: float = 20.0  # 位置与航向设定点的更新频率，单位 Hz
    align_yaw_tolerance: float = 0.0349065850  # 机头对准容差，单位 rad（2 deg）
    align_yaw_stable_time: float = 0.3  # 连续满足角度容差后确认对准的时间，单位秒
    align_yaw_timeout: float = 15.0  # 原地调整机头方向的最长时间，单位秒
