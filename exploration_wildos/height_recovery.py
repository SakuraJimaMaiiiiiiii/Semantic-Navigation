"""Bounded graph-plane recovery using fully observed 3-D body clearance."""
import numpy as np
from navigation.route import observed_body_segment_rejection
from planner.local_avoidance_planner import LocalPlan
from .vehicle_constraints import expanded_boxes, segment_hits_vehicle

MAX_RECOVERY_METRES = .5
RECOVERY_START_METRES = .03
RECOVERY_ARRIVAL_METRES = .02


def height_detail(current, altitude, tolerance):
    return (f'当前 NED z={float(current[2]):.3f} m，图高度={altitude:.3f} m，'
            f'偏差={abs(float(current[2])-altitude):.3f} m，容差={tolerance:.3f} m')


def vertical_clear(snapshot, current, target, config, objects):
    return vertical_rejection(snapshot,current,target,config,objects) is None


def vertical_rejection(snapshot, current, target, config, objects):
    reason = observed_body_segment_rejection(snapshot,current,target,config)
    if reason is not None:
        return dict(missing_map='尚无局部地图',invalid_position='位置无效',outside_map='恢复通道超出局部地图',
                    occupied_body_volume='实际机体扫掠范围存在占用体素',
                    unknown_body_volume='实际机体扫掠范围尚未完整观测')[reason]
    for identity,low,high in expanded_boxes(objects,config):
        if segment_hits_vehicle(current,target,low,high):
            return '恢复通道位于车辆禁飞区域：'+str(identity)
    return None


class HeightRecoveryGuard:
    """Only the fixed, locally observed recovery chord is eligible for execution."""
    def __init__(self, navigator, target):
        self.navigator = navigator
        self.target = np.asarray(target).copy()
        self.config = navigator.local_planner.config

    def check_execution_health(self):
        self.navigator._check_health()

    def plan(self, current, target):
        nav = self.navigator
        current = np.asarray(current,dtype=float)
        projected = current.copy();projected[2] = self.target[2]
        # Used both by the planner worker and before EVERY position command.
        if (np.linalg.norm(current-self.target) > MAX_RECOVERY_METRES
                or np.linalg.norm(current[:2]-self.target[:2]) > .05
                or not np.allclose(target,self.target)
                or not vertical_clear(nav.local_map.execution_snapshot,current,projected,
                                      nav.config,nav.objects_provider())):
            return LocalPlan('height_recovery_blocked',None,np.empty((0,3)),0.)
        # Use the existing executable status whitelist. The task event and UI
        # phase identify this as height recovery; unknown status strings hold.
        return LocalPlan('clear',projected,np.array([current,projected]),
                         min(.15,float(self.config.cruise_max_speed)))

    def revalidate_plan(self, plan, current, *, certificate=None):
        return self.plan(current,self.target)
