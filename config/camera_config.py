"""RflySim RGB-D 相机接口配置。"""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CameraConfig:
    """一组对齐的 RGB、深度和仿真分割相机运行参数。"""

    copter_id: int = 1  # 相机绑定的无人机编号，必须与 JSON 中的 TargetCopter 一致
    sdk_root: Path = Path(r"D:\PX4PSP\RflySimAPIs\RflySimSDK")  # RflySim SDK 根目录
    sensor_json: Path = (
        PROJECT_ROOT / "config" / "sensors" / "VisionSensors_RGBD_Config_v0.json"
    )  # RGB-D 视觉传感器 JSON 配置文件
    ue_window_id: int = 0  # 接收传感器请求的 RflySim3D 窗口编号
    first_frame_timeout: float = 10.0  # 等待第一帧完整 RGB-D 数据的超时时间
    poll_interval: float = 0.01  # 等待图像时检查数据状态的时间间隔
    max_rgb_depth_skew: float = 0.02  # RGB 与深度允许的最大时间差，单位 s
    state_sample_rate: float = 100.0  # 无人机状态历史缓存的采样频率，单位 Hz
    state_history_seconds: float = 5.0  # 状态历史缓存保留时间，单位 s
    max_state_sync_error: float = 0.03  # 图像与最近原始位姿样本的最大时间差，单位 s
    # 暂定误差模型，非PX4实时协方差；应与同步仿真真值比较后标定。
    pose_position_std_ned: tuple[float, float, float] = (0.1, 0.1, 0.15)  # m
    pose_rotation_std_body_deg: tuple[float, float, float] = (0.5, 0.5, 1.0)  # 局部小角度，非欧拉角协方差
    pose_timestamp_std: float = 0.01  # 剩余时间偏差的标准差假设，s
    pose_covariance_profile: Path | None = None  # 显式选择本机/本场景的误差标定 JSON
    depth_unit_scale: float = 0.001  # uint16 原始深度到米的换算比例
    depth_min: float = 0.3  # 接受和显示的最小深度，单位 m
    depth_max: float = 30.0  # 接受和显示的最大深度，单位 m
    show_rgb: bool = False  # 是否在飞行过程中实时显示 RGB 图像窗口
    show_depth: bool = False  # 是否在飞行过程中实时显示深度伪彩色图窗口
    # False 时不向 UE 请求 TypeID=4；True 时请求并显示，重启后生效。
    show_segmentation: bool = False
    depth_display_mode: str = "grayscale"  # 深度显示方式：color 为红蓝伪彩色，grayscale 为黑白
    preview_fps: float = 30.0  # RGB-D 预览窗口的目标刷新频率，单位 Hz
    suppress_sdk_output: bool = True  # 是否隐藏 VisionCaptureApi 启动时的普通提示信息
