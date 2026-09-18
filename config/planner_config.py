"""局部主动避障配置。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerConfig:
    """EGO/二维 A* 局部规划及独立安全停车参数。"""

    # ego_native=C++ 核心（不可用时回退 Python）；ego=纯 Python 三维 B 样条。
    navigation_mode: str = "ego_native"
    # 返航航段的局部规划模式：
    # ego_native/ego=三维B样条；astar=二维栅格；depth_guard=仅深度保护。
    return_navigation_mode: str = "ego_native"
    replan_rate: float = 5.0  # 局部路径重规划频率，Hz
    map_timeout: float = 0.6  # 地图连续多久未更新后悬停，s
    planning_lookahead: float = 3.0  # 每次规划朝最终目标看的距离，m
    local_target_distance: float = 1.2  # 跟踪路径上的短期航点距离，m
    minimum_forward_progress: float = 0.1  # 绕障航点沿目标方向的最小推进量，m
    map_boundary_margin: float = 0.1  # 局部航点距当前地图边界的最小余量，m
    reachable_goal_tolerance: float = 1.2  # 可达替代终点距期望终点上限，m
    emergency_stop_distance: float = 0.6  # 前向深度小于此距离立即悬停，m
    map_emergency_stop_distance: float = 0.35  # 保留近距离地图障碍的硬急停保护
    slowdown_distance: float = 1.0  # 障碍进入此距离后限制速度，m
    cruise_max_speed: float = 0.6  # 兼顾视觉稳定性的安全巡航速度，m/s
    avoidance_max_speed: float = 0.35  # 避障期间最大前向速度，m/s
    sensor_clear_fallback_speed: float = 0.4  # A*假阻塞时的低速直行速度
    fallback_min_clearance: float = 1.5  # 直行兜底所需的前向深度，m
    fallback_min_map_clearance: float = 0.4  # 保留地图兜底硬间距，避免贴近真实障碍直行
    vertical_target_step: float = 0.5  # 每次局部目标允许调整的最大高度，m
    no_path_confirmation_count: int = 3  # 连续失败多少次才确认无路
    clear_path_confirmation_count: int = 3  # 绕障后连续直线安全多少帧才恢复直飞
    avoidance_reuse_min_distance: float = 0.35  # 旧绕障航点距机体大于此值才复用，m
    unknown_is_occupied: bool = False  # 初期允许穿过未观测格；严格模式可设True
    vertical_band: float = 0.2  # 投影到二维规划面时检查的上下半高，m
    safety_corridor_half_width: float = 0.25  # 目标方向窄走廊半宽，m
    allow_diagonal: bool = True
    max_search_nodes: int = 60_000  # 本地二维/三维种子搜索共用的节点上限
    global_return_inflation_radius: float = 0.3  # 全局返航障碍膨胀半径，m
    global_return_trace_radius: float = 0.45  # 实际去程轨迹形成的已知安全走廊半径，m
    # NED 的 Down 轴向下为正；缩小上方范围可过滤天花板钢筋的二维投影。
    global_return_above_height: float = 0.25  # 全局返航检查飞行平面上方高度，m
    global_return_below_height: float = 0.9  # 向下聚合多层体素，补全柱墙轮廓，m
    global_return_max_search_nodes: int = 300_000

    # C++/Python EGO 风格局部规划共用参数。这里不依赖 ROS；A* 只负责为
    # 碰撞段提供拓扑种子，最终执行路径由三次均匀 B 样条生成。
    ego_control_point_spacing: float = 0.35
    ego_trajectory_lookahead_time: float = 0.8
    ego_clearance: float = 0.45
    ego_knot_interval: float = 0.35
    ego_max_acceleration: float = 1.2
    ego_feasibility_tolerance: float = 0.05
    ego_optimization_iterations: int = 200
    ego_lbfgs_memory: int = 16
    ego_rebound_max_restarts: int = 6
    # 仅供未编译 C++ 扩展时的旧 Python 梯度下降回退使用。
    ego_optimization_step: float = 0.035
    ego_smoothness_weight: float = 1.0
    ego_collision_weight: float = 5.0
    ego_feasibility_weight: float = 0.2
    ego_fitness_weight: float = 0.15
    ego_collision_check_step: float = 0.08

    def __post_init__(self) -> None:
        valid_navigation_modes = {
            "depth_guard",
            "astar",
            "ego",
            "ego_native",
        }
        for name in ("navigation_mode", "return_navigation_mode"):
            value = getattr(self, name)
            if value not in valid_navigation_modes:
                raise ValueError(
                    f"{name} must be 'depth_guard', 'astar', 'ego', "
                    "or 'ego_native'."
                )
        positive = {
            "replan_rate": self.replan_rate,
            "map_timeout": self.map_timeout,
            "planning_lookahead": self.planning_lookahead,
            "local_target_distance": self.local_target_distance,
            "minimum_forward_progress": self.minimum_forward_progress,
            "map_boundary_margin": self.map_boundary_margin,
            "reachable_goal_tolerance": self.reachable_goal_tolerance,
            "emergency_stop_distance": self.emergency_stop_distance,
            "map_emergency_stop_distance": (
                self.map_emergency_stop_distance
            ),
            "slowdown_distance": self.slowdown_distance,
            "cruise_max_speed": self.cruise_max_speed,
            "avoidance_max_speed": self.avoidance_max_speed,
            "sensor_clear_fallback_speed": self.sensor_clear_fallback_speed,
            "fallback_min_clearance": self.fallback_min_clearance,
            "fallback_min_map_clearance": (
                self.fallback_min_map_clearance
            ),
            "vertical_target_step": self.vertical_target_step,
            "avoidance_reuse_min_distance": (
                self.avoidance_reuse_min_distance
            ),
            "vertical_band": self.vertical_band,
            "safety_corridor_half_width": self.safety_corridor_half_width,
            "max_search_nodes": self.max_search_nodes,
            "global_return_inflation_radius": (
                self.global_return_inflation_radius
            ),
            "global_return_trace_radius": self.global_return_trace_radius,
            "global_return_above_height": self.global_return_above_height,
            "global_return_below_height": self.global_return_below_height,
            "global_return_max_search_nodes": (
                self.global_return_max_search_nodes
            ),
            "ego_control_point_spacing": self.ego_control_point_spacing,
            "ego_trajectory_lookahead_time": (
                self.ego_trajectory_lookahead_time
            ),
            "ego_clearance": self.ego_clearance,
            "ego_knot_interval": self.ego_knot_interval,
            "ego_max_acceleration": self.ego_max_acceleration,
            "ego_feasibility_tolerance": (
                self.ego_feasibility_tolerance
            ),
            "ego_optimization_iterations": (
                self.ego_optimization_iterations
            ),
            "ego_lbfgs_memory": self.ego_lbfgs_memory,
            "ego_rebound_max_restarts": self.ego_rebound_max_restarts,
            "ego_optimization_step": self.ego_optimization_step,
            "ego_smoothness_weight": self.ego_smoothness_weight,
            "ego_collision_weight": self.ego_collision_weight,
            "ego_feasibility_weight": self.ego_feasibility_weight,
            "ego_fitness_weight": self.ego_fitness_weight,
            "ego_collision_check_step": self.ego_collision_check_step,
        }
        for name, value in positive.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")
        if self.slowdown_distance <= self.emergency_stop_distance:
            raise ValueError(
                "slowdown_distance must exceed emergency_stop_distance."
            )
        if (
            self.map_emergency_stop_distance
            >= self.fallback_min_map_clearance
        ):
            raise ValueError(
                "map_emergency_stop_distance must be less than "
                "fallback_min_map_clearance."
            )
        if self.avoidance_max_speed > self.cruise_max_speed:
            raise ValueError(
                "avoidance_max_speed must not exceed cruise_max_speed."
            )
        if self.sensor_clear_fallback_speed > self.cruise_max_speed:
            raise ValueError(
                "sensor_clear_fallback_speed must not exceed "
                "cruise_max_speed."
            )
        if int(self.no_path_confirmation_count) < 1:
            raise ValueError(
                "no_path_confirmation_count must be at least one."
            )
        if int(self.clear_path_confirmation_count) < 1:
            raise ValueError(
                "clear_path_confirmation_count must be at least one."
            )
