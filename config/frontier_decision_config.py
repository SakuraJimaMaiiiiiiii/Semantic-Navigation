"""Decision-only exploration options; motion and safety settings live elsewhere."""
from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class FrontierDecisionConfig:
    min_frontier_size: int = 5  # 原始自由格上的 8 邻接前沿，小于该格数过滤
    candidates_per_cluster: int = 4
    cluster_max_span: float = 6.0  # 长边界拆为连通的局部片段，避免质心落在房间中央
    max_clusters_per_update: int = 24  # 分批轮转评估，另保留当前目标所在簇
    gain_weight: float = 2.0
    path_weight: float = 1.0
    yaw_weight: float = 0.5
    revisit_weight: float = 0.8
    path_normalization_m: float = 20.0
    forward_cone_degrees: float = 30.0  # 当前行进方向左右各 30° 优先直行
    revisit_radius: float = 3.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            zero_allowed = field.name.endswith('_weight')
            if not math.isfinite(value) or value < 0 or (value == 0 and not zero_allowed):
                raise ValueError(f"decision.{field.name} has an invalid value")
        for name in ('min_frontier_size', 'candidates_per_cluster', 'max_clusters_per_update'):
            if not isinstance(getattr(self,name),int) or isinstance(getattr(self,name),bool):
                raise ValueError(f"decision.{name} must be an integer")
        if self.forward_cone_degrees > 90:
            raise ValueError("forward_cone_degrees must be <= 90")
        if not 3 <= self.candidates_per_cluster <= 5:
            raise ValueError('candidates_per_cluster must be between 3 and 5')
