"""车库全局稀疏三维占据地图配置。"""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class GlobalSparseMapConfig:
    """固定在世界NED坐标系中的稀疏占据地图参数。"""

    enabled: bool = False  # 试验阶段关闭全局地图；返航改用实时局部规划
    resolution: float = 0.3  # 全局体素边长，m；通常比局部地图更粗
    update_rate: float = 3.0  # 全局地图更新频率，Hz
    vehicle_free_radius: float = 0.0  # 启用时，仅补足当前机身覆盖的未观测体素
    vehicle_free_half_height: float = 0.25
    depth_stride: int = 12  # 深度图采样步长，控制全局建图计算量
    min_depth: float = 0.3  # 使用的最小深度，m
    max_depth: float = 10.0  # 使用的最大深度，m
    depth_is_range: bool = False  # False表示Z轴深度，True表示射线距离
    ray_step: float = 0.3  # 空闲空间射线采样间距，m
    hit_log_odds: float = 0.85  # 深度端点的占据证据
    miss_log_odds: float = 0.4  # 射线经过位置的空闲证据
    min_log_odds: float = -2.0  # 空闲证据下限
    max_log_odds: float = 3.5  # 占据证据上限
    occupied_threshold: float = 0.7  # 判定占据的log-odds阈值
    free_threshold: float = -0.2  # 判定空闲的log-odds阈值
    max_voxels: int = 2_000_000  # 防止异常深度导致内存无限增长

    loop_closure_enabled: bool = True  # 是否启用关键帧点云回环校正
    keyframe_translation: float = 0.8  # 平移超过该距离时创建关键帧，m
    keyframe_yaw_degrees: float = 15.0  # 偏航变化超过该角度时创建关键帧
    keyframe_max_interval: float = 3.0  # 即使悬停也定期保留关键帧，s
    max_keyframes: int = 500  # 本次运行最多保存的回环关键帧数
    loop_min_separation: int = 15  # 回环帧与当前帧至少间隔的关键帧数
    loop_search_radius: float = 2.0  # 在估计位置附近搜索历史关键帧，m
    loop_max_candidates: int = 3  # 每次最多进行ICP验证的候选数量
    icp_voxel_size: float = 0.35  # ICP点云降采样尺寸，m
    icp_max_correspondence: float = 0.7  # ICP最大点对应距离，m
    icp_min_fitness: float = 0.35  # 接受回环所需的最小重叠比例
    icp_max_rmse: float = 0.35  # 接受回环所允许的最大配准误差，m
    loop_max_translation_correction: float = 2.0  # 单次最大平移校正，m
    loop_max_yaw_correction_degrees: float = 35.0  # 单次最大偏航校正
    pose_graph_iterations: int = 8  # 位姿图高斯牛顿迭代次数
    loop_edge_weight: float = 4.0  # 回环约束相对连续运动约束的权重

    visualize: bool = True  # 是否打开全局稀疏地图Open3D窗口
    show_free_voxels: bool = False  # 空闲体素较多，默认只显示红色障碍
    visualization_point_size: float = 2.0  # 显示点大小
    visualization_min_spacing: float = 0.45  # 显示抽稀间距，m
    max_visualized_occupied_voxels: int = 20000  # 障碍显示上限
    max_visualized_free_voxels: int = 10000  # 空闲体素显示上限

    save_on_close: bool = True  # 任务结束时是否保存全局地图
    export_foxglove_mcap_on_close: bool = True
    dense_export_subdivisions: int = 4  # 0.3 m voxel -> about 0.1 m surface spacing
    dense_export_max_points: int = 1_500_000  # Export-only memory/file-size guard
    # 独立使用地图类时的默认目录；DemoRuntime 会改为本次运行目录。
    output_dir: Path = PROJECT_ROOT / "data" / "maps"
    file_prefix: str = "garage_global_sparse_map"  # 地图文件名前缀
