import json
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from vehicle.rfly_multirotor import RflyMultirotorInterface
from navigation.mission import _FlightGuard
from config.planner_config import PlannerConfig
from planner.local_avoidance_planner import LocalPlan


def test_height_diagnostics_preserve_sdk_and_packet_sources_separately():
    vehicle=object.__new__(RflyMultirotorInterface)
    vehicle.connected=True
    vehicle.ue=None
    packet=NS(x=-2.4,y=22.6,z=-1.74,vx=0.,vy=0.,vz=-.02,time_boot_ms=2000,_timestamp=1234.)
    vehicle.mav=NS(uavPosNED=[-2.4,22.6,-1.75],uavVelNED=[0.,0.,-.02],
                   pos=[-2.4,22.6,-1.506],isInPointMode=False,
                   the_connection=NS(messages={'LOCAL_POSITION_NED':packet}))
    vehicle._last_position_command=dict(position_ned=[-2.4,22.6,-1.506],yaw=0.,monotonic_time=1.)
    evidence=vehicle.height_tracking_diagnostics()
    assert evidence['position_ned'][2]==-1.75
    assert evidence['local_position_ned']['z']==-1.74
    assert evidence['sdk_position_setpoint_ned'][2]==-1.506
    assert evidence['last_position_command']['position_ned'][2]==-1.506
    json.dumps(evidence,allow_nan=False)


def test_height_recovery_event_and_failure_keep_same_tracking_evidence():
    evidence={'position_ned':[0.,0.,-1.75],'sdk_position_setpoint_ned':[0.,0.,-1.5]}
    nav=NS(local_planner=NS(config=PlannerConfig()),_event=Mock(),
           vehicle=NS(height_tracking_diagnostics=Mock(return_value=evidence)))
    guard=_FlightGuard(nav)
    guard.capture_failure=Mock()
    guard._diagnostic_context=('local','world',np.zeros(3),np.ones(3))
    guard.notify_execution(LocalPlan('height_recovery_blocked',None,np.empty((0,3)),0.))
    assert nav._event.call_args.kwargs['height_tracking']==evidence
    assert guard.capture_failure.call_args.args[1]['height_tracking']==evidence


def test_optional_diagnostic_error_cannot_interrupt_recovery():
    nav=NS(local_planner=NS(config=PlannerConfig()),_event=Mock(),
           vehicle=NS(height_tracking_diagnostics=Mock(side_effect=RuntimeError('no UE telemetry'))))
    guard=_FlightGuard(nav)
    plan=LocalPlan('height_recovery',np.array([0.,0.,-1.5]),np.empty((0,3)),.15)
    assert guard.notify_execution(plan).can_move
    assert nav._event.call_args.kwargs['height_tracking_error']=='no UE telemetry'
