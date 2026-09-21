"""局部三维占据栅格配置。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OccupancyGridConfig:
    """深度图局部建图参数。"""

    enabled: bool = True  # 是否运行局部三维占据栅格
    size_ned: tuple[float, float, float] = (10.0, 10.0, 6.0)  # N/E/D范围，m
    resolution: float = 0.2  # 体素边长，m
    update_rate: float = 5.0  # 地图更新频率，Hz
    vehicle_free_radius: float = 0.0  # 启用时，仅补足当前机身覆盖的未观测体素
    vehicle_free_half_height: float = 0.25
    depth_stride: int = 8  # 深度图采样步长，越大计算越快但点云越稀疏
    min_depth: float = 0.3  # 建图使用的最小深度，m
    max_depth: float = 8.0  # 建图使用的最大深度，m
    depth_is_range: bool = False  # False表示Z轴深度，True表示射线距离
    hit_log_odds: float = 0.85  # 深度端点对占据概率的增量
    miss_log_odds: float = 0.4  # 射线穿过体素对占据概率的减量
    min_log_odds: float = -2.0  # 空闲证据下限
    max_log_odds: float = 3.5  # 占据证据上限
    occupied_threshold: float = 1.2  # 约需连续两次命中才判为占据，减少单帧假障碍
    free_threshold: float = -0.2  # 判定为空闲的log-odds阈值
    inflation_radius: float = 0.3  # 障碍膨胀半径，m；缩小以保留更多可通行空间
    forward_safety_roi_width: float = 0.5  # 前向急停区域占图像宽度的比例
    forward_safety_roi_height: float = 0.5  # 前向急停区域占图像高度的比例
    forward_safety_percentile: float = 5.0  # 用近距离分位数抑制单像素噪声
