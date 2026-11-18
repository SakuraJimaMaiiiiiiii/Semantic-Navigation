from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
from config.semantic_navigation_config import SemanticNavigationConfig
from navigation.visual_verification import verify_saved_target


def setup(period=6.,matches=True,freeze=False):
    now=[0.]
    current=np.array([0.,0.,-1.5])
    record=dict(runtime_instance_id='vehicle_01',position_world=[1.5,0.,-.5],
                bbox_3d=dict(min=[1.2,-.5,-1.],max=[3.,.5,0.]))
    nav=NS(config=SemanticNavigationConfig(),clock=lambda:now[0],deadline=100.,
           _event=Mock(),_check_health=Mock(),_align=Mock())
    nav.vehicle=NS(get_position=lambda:current.copy(),wait=lambda t:now.__setitem__(0,now[0]+t))
    def frame(**kwargs):
        index=0 if freeze else int((now[0]+1e-8)//period)
        fresh=dict(record,last_seen=float(index))
        return NS(timestamp=float(index),completed_at=index*period,
                  inference_seconds=period,records=[fresh])
    nav._frame=frame
    nav._verification_candidates=lambda f,c:f.records if matches else []
    return nav,record,current,now


def test_slow_sam3_has_time_for_three_distinct_confirmations():
    nav,record,goal,now=setup()
    assert verify_saved_target(nav,record,goal) is not None
    assert 18.-1e-7 <= now[0] < 30.
    assert nav._align.call_count==1


def test_repeated_frame_never_counts_as_multiple_confirmations():
    nav,record,goal,now=setup(period=1.,freeze=True)
    assert verify_saved_target(nav,record,goal) is None
    event=nav._event.call_args.kwargs
    assert event['accepted_frames']==0 and event['observed_frames']==0
    assert event['reason']=='insufficient_fresh_frames'
    assert now[0] <= 8.01


def test_no_match_uses_bounded_yaw_views_without_translation():
    nav,record,goal,now=setup(period=1.,matches=False)
    assert verify_saved_target(nav,record,goal) is None
    assert nav._align.call_count==3
    assert now[0] <= 45.
    assert nav._event.call_args.kwargs['reason']=='target_not_reobserved'
    np.testing.assert_array_equal(nav.vehicle.get_position(),goal)


def test_wrong_geometry_remains_unconfirmed():
    nav,record,goal,now=setup(period=1.)
    wrong=dict(record,bbox_3d=dict(min=[3.,-.5,-1.],max=[4.,.5,0.]))
    nav._verification_candidates=lambda f,c:[dict(wrong,last_seen=f.timestamp)]
    assert verify_saved_target(nav,record,goal) is None
    assert nav._event.call_args.kwargs['matched_frames'] > 0
    assert nav._event.call_args.kwargs['accepted_frames']==0


def test_detector_failure_and_mission_deadline_remain_authoritative():
    nav,record,goal,now=setup(period=20.)
    nav.deadline=12.
    assert verify_saved_target(nav,record,goal) is None
    assert now[0] <= 12.01
    nav._check_health=Mock(side_effect=RuntimeError('detector failed'))
    now[0]=0.
    with pytest.raises(RuntimeError,match='detector failed'):
        verify_saved_target(nav,record,goal)


def test_very_slow_frames_respect_hard_cap():
    nav,record,goal,now=setup(period=20.,matches=False)
    assert verify_saved_target(nav,record,goal) is None
    assert now[0] <= 45.01
    assert nav._event.call_args.kwargs['accepted_frames']==0
