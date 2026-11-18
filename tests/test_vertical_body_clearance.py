from types import SimpleNamespace as NS
import numpy as np
import pytest
from navigation.route import observed_body_segment_rejection


@pytest.mark.parametrize('current,origin,hits',[
    ([-2.39633536,22.60044479,-1.61584318],[-7.4,17.4,-4.6],[[-2.4,23.,-2.]]),
    ([-2.33514094,22.64440918,-1.75088239],[-7.4,17.6,-4.8],[[-2.4,23.,-2.2],[-2.2,23.,-2.2]]),
    ([-2.32931185,22.64071465,-1.75019181],[-7.4,17.6,-4.8],[[-2.4,23.,-2.2],[-2.2,23.,-2.2]]),
])
def test_nov17_side_voxels_do_not_block_vertical_recovery(current,origin,hits):
    snapshot=NS(states=np.zeros((50,50,30),dtype=np.int8),resolution=.2,origin_ned=np.array(origin))
    config=NS(clearance_radius=.35,vertical_half_extent=.25)
    for hit in hits:
        cell=np.floor((np.array(hit)-snapshot.origin_ned)/.2+1e-8).astype(int)
        snapshot.states[tuple(cell)]=1
    current=np.array(current)
    target=current.copy();target[2]=-1.5057213306427002
    assert observed_body_segment_rejection(snapshot,current,target,config) is None
    # Moving the actual disk into the same voxel must retain the stop.
    current[1] += .08;target[1] += .08
    assert observed_body_segment_rejection(snapshot,current,target,config)=='occupied_body_volume'


@pytest.mark.parametrize('state,reason',[(1,'occupied_body_volume'),(-1,'unknown_body_volume')])
def test_actual_diagonal_overlap_is_checked_even_near_cell_corner(state,reason):
    snapshot=NS(states=np.zeros((15,15,20),dtype=np.int8),resolution=.2,
                origin_ned=np.array([-1.,-1.,-2.5]))
    config=NS(clearance_radius=.35,vertical_half_extent=.25)
    snapshot.states[3,3,5]=state
    current=np.array([.001,.001,-1.6]);target=current.copy();target[2]=-1.5
    # The old centre-offset mask omitted (-2,-2), although this voxel's
    # nearest corner lies only ~0.284 m from a 0.35 m body disk.
    assert observed_body_segment_rejection(snapshot,current,target,config)==reason


def test_vertical_sweep_still_rejects_out_of_map_space():
    snapshot=NS(states=np.zeros((15,15,20),dtype=np.int8),resolution=.2,
                origin_ned=np.array([-1.,-1.,-2.5]))
    config=NS(clearance_radius=.35,vertical_half_extent=.25)
    assert observed_body_segment_rejection(snapshot,[-.9,0.,-1.5],[-.9,0.,-1.4],config)=='outside_map'
