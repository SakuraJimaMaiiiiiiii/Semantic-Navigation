"""飞行状态与 RGB-D HDF5 数据记录配置。"""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RecordingConfig:
    """运行时同步数据记录参数。"""

    enabled: bool = False  # 按需开启原始 RGB-D 录制，避免常规运行持续占用磁盘
    # 独立使用记录器时的默认目录；DemoRuntime 会改为本次运行目录。
    output_dir: Path = PROJECT_ROOT / "data" / "recordings"
    file_prefix: str = "flight_rgbd"  # 每次记录文件的名称前缀
    record_rate: float = 30.0  # 目标记录频率，单位 Hz，不应高于相机帧率
    observation_timeout: float = 1.0  # 等待一组同步数据的超时时间，单位 s
    flush_interval: int = 10  # 每记录多少帧将 HDF5 缓冲区刷新到磁盘
    hdf5_compression: str = "lzf"  # 在线优先用快速 lzf；也可设为 gzip 或 none
    hdf5_compression_level: int = 4  # gzip 压缩等级，范围0～9
