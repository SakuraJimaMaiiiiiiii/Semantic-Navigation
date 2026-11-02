"""RflySim Type20 body-frame LiDAR, started alongside vision in every DemoRuntime session."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LidarConfig:
    point_axes: str = "flu"  # 本机 Type20 回波样本验证为 FLU，读取后统一转 FRD
    seq_id: int = 3
    max_range: float = 30.0
    min_range: float = 0.6
    max_points: int = 3000
    update_rate: float = 5.0
    stale_timeout: float = 3.0
    camera_max_skew: float = 0.05  # 仅匹配图像时刻附近 50 ms 内的雷达帧
    semantic_min_points: int = 6  # 同一物体掩膜内部的一致点数
    alignment_absolute_gate: float = 0.3  # 相机光轴深度一致性门槛（m）
    alignment_relative_gate: float = 0.02

    def sensor(self, copter_id):
        # Zero mounting transform avoids confusing sensor/body origins in Type20.
        # 360 degrees horizontal, 32 vertical samples. SDK decodes XYZ to metres.
        return dict(SeqID=self.seq_id, TypeID=20, TargetCopter=copter_id,
                    TargetMountType=0, DataWidth=900, DataHeight=32, DataCheckFreq=10,
                    SendProtocol=[0, 0, 0, 0, 0, 0, 0, 0], CameraFOV=90,
                    EularOrQuat=0, SensorPosXYZ=[0, 0, 0], SensorAngQuat=[0,0,0,0],
                    SensorAngEular=[0,0,0],
                    otherParams=[self.max_range, .05, -180, 180, -20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
