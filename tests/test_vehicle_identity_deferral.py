"""Unseen portions of a car need evidence before becoming another identity."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from test_vehicle_split_evidence import module


def observation(points, appearance=(1., 0.)):
    return SimpleNamespace(position_world=points.mean(0), bbox_3d_min=points.min(0),
        bbox_3d_max=points.max(0), measurement_covariance=np.eye(3)*.001,
        appearance=np.asarray(appearance), color_scores={}, surface_points_world=points)


def surfaces(angle=0.):
    front = np.array([[0., y, z] for y in np.linspace(-1, 1, 21) for z in np.linspace(-1, 0, 11)])
    side = np.array([[x, 1., z] for x in np.linspace(0, 1.5, 16) for z in np.linspace(-1, 0, 11)])
    fragment = np.array([[x, 1., z] for x in np.linspace(3., 4.5, 16) for z in np.linspace(-1, 0, 11)])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    parent = np.vstack([front, side])
    for points in (parent, fragment):
        points[:, :2] = points[:, :2] @ rotation.T
    return parent, fragment


def prepared(angle=0.):
    tracker = module.VehicleInstanceTracker()
    parent, fragment = surfaces(angle)
    for stamp in (1., 2., 3.):
        tracker.update(['car'], [.9], [observation(parent)], stamp)
    return tracker, parent, fragment


@pytest.mark.parametrize('angle', [0., np.deg2rad(31), np.deg2rad(102)])
def test_disjoint_side_continuation_stays_unresolved_at_any_heading(angle):
    tracker, _, fragment = prepared(angle)
    for stamp in (4., 5., 6.):
        assert tracker.update(['car'], [.9], [observation(fragment, (.65, .35))], stamp) == [None]
    assert list(tracker.tracks) == [1]
    assert not tracker.redirects  # A plausible continuation is not proof of identity.
    assert tracker._pending[0].possible_parents == {1}
    assert 1 in tracker.connection_hypotheses()
    # A static unresolved fragment must survive a long revisit gap.
    tracker.update([], [], [], 600.)
    assert 1 in tracker.connection_hypotheses()


def test_repeated_split_masks_do_not_count_as_independent_cars():
    tracker, parent, fragment = prepared()
    for stamp in range(4, 11):
        assert tracker.update(['car', 'car'], [.9, .9],
            [observation(parent), observation(fragment)], float(stamp)) == [1, None]
    assert len(tracker.tracks) == 1
    assert tracker._pending[0].separate_counts.get(1, 0) == 0


def test_joint_views_resolve_fragment_without_a_second_published_id():
    tracker, parent, fragment = prepared()
    for stamp in (4., 5., 6.):
        tracker.update(['car'], [.9], [observation(fragment)], stamp)
    for stamp in (7., 8., 9.):
        assert tracker.update(['car'], [.9], [observation(np.vstack([parent, fragment]))], stamp) == [1]
    assert not tracker._pending
    assert tracker.tracks[1].surface.observed_patch(fragment)
    assert not tracker.connection_hypotheses()


def test_same_paint_neighbor_is_published_separately():
    tracker, parent, _ = prepared()
    neighbor = parent + [0., 3.5, 0.]
    for stamp in (4., 5., 6.):
        tracker.update(['car', 'car'], [.9, .9], [observation(parent), observation(neighbor)], stamp)
    assert len(tracker.tracks) == 2
    assert not tracker.redirects


def side_and_end(angle=0.):
    side = np.array([[x, 1.5, z] for x in np.linspace(0., 4.4, 45)
                     for z in np.linspace(-1., 0., 11)])
    end = np.vstack([
        [[4.3, y, z] for y in np.linspace(0., 1., 11) for z in np.linspace(-1.2, -.35, 10)],
        [[x, 0., z] for x in np.linspace(3.3, 4.3, 11) for z in np.linspace(-1.2, -.35, 10)],
    ])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for points in (side, end):
        points[:, :2] = points[:, :2] @ rotation.T
    return side, end


@pytest.mark.parametrize('angle', [0., np.deg2rad(31), np.deg2rad(102)])
@pytest.mark.parametrize('end_first', [False, True])
def test_disjoint_end_corner_and_long_side_need_a_connection_view(angle, end_first):
    # The Nov 16 run saw a long side, then a roughly square end/corner patch
    # across an unobserved gap. Neither containment nor a thin-face gate applies.
    side, end = side_and_end(angle)
    first, second = (end, side) if end_first else (side, end)
    tracker = module.VehicleInstanceTracker()
    for stamp in (1., 2., 3.):
        tracker.update(['car'], [.9], [observation(first)], stamp)
    parent = tracker.tracks[1]
    anchor, surface = parent.state.copy(), parent.surface.supported_points().copy()
    for stamp in (4., 5., 6., 7.):
        tracker.update(['car'], [.9], [observation(second, (.84, .16))], stamp)
    assert list(tracker.tracks) == [1]
    assert len(tracker._pending) == 1 and tracker._pending[0].possible_parents == {1}
    assert 1 in tracker.connection_hypotheses()
    assert np.array_equal(parent.state, anchor)
    assert np.array_equal(parent.surface.supported_points(), surface)
    assert not tracker.redirects


@pytest.mark.parametrize('shift', [(6., 0., 0.), (0., 4., 0.)])
def test_same_paint_end_of_a_neighbor_is_not_held_against_another_car(shift):
    side, end = side_and_end()
    tracker = module.VehicleInstanceTracker()
    for stamp in (1., 2., 3.):
        tracker.update(['car'], [.9], [observation(side)], stamp)
    for stamp in (4., 5., 6.):
        tracker.update(['car'], [.9], [observation(end + shift, (.84, .16))], stamp)
    assert len(tracker.tracks) == 2
    assert not tracker.redirects


def test_end_corner_is_only_imported_after_repeated_joint_views():
    side, end = side_and_end()
    tracker = module.VehicleInstanceTracker()
    for stamp in (1., 2., 3.):
        tracker.update(['car'], [.9], [observation(side)], stamp)
    for stamp in (4., 5., 6.):
        tracker.update(['car'], [.9], [observation(end, (.84, .16))], stamp)
    assert not tracker.tracks[1].surface.observed_patch(end)
    for stamp in (7., 8., 9.):
        tracker.update(['car'], [.9], [observation(np.vstack([side, end]), (.95, .05))], stamp)
    assert list(tracker.tracks) == [1]
    assert not tracker._pending
    assert tracker.tracks[1].surface.observed_patch(end)


def test_two_overlapping_current_proposals_can_repair_old_published_split():
    tracker = module.VehicleInstanceTracker(process_noise_std=0., viewpoint_noise_std=0.)
    side, _ = side_and_end()
    # Historical masks left two estimated anchors on the same measured face.
    for identity, x in [(1, 1.5), (2, 2.5)]:
        obs = observation(side)
        anchor = np.array([x, 1.5, -.5])
        track = module.VehicleTrack(identity, 'car', anchor, np.eye(3)*.001,
            obs.bbox_3d_min.copy(), obs.bbox_3d_max.copy(), obs.appearance, .9, 6, 0.,
            appearance_gallery=[obs.appearance])
        track.surface.update(side); track.surface.update(side)
        tracker.tracks[identity] = track
    tracker._next_id = 3
    tracker._distinct_pairs.add((1, 2))
    for stamp in (1., 2., 3.):
        observations = [observation(side), observation(side)]
        observations[0].position_world = np.array([1.5, 1.5, -.5])
        observations[1].position_world = np.array([2.5, 1.5, -.5])
        tracker.update(['car', 'car'], [.9, .9], observations, stamp)
    assert list(tracker.tracks) == [1]
    assert tracker.redirects == {2: 1}


def test_independent_full_bodies_release_a_previously_ambiguous_neighbor():
    tracker = module.VehicleInstanceTracker()
    parent, _ = surfaces()
    side = np.array([[x, 1., z] for x in np.linspace(0., 3.5, 36) for z in np.linspace(-1., 0., 11)])
    parent = np.vstack([parent, side])
    fragment = np.array([[x, 1., z] for x in np.linspace(4., 5., 11) for z in np.linspace(-1., 0., 11)])
    for stamp in (1., 2., 3.):
        tracker.update(['car'], [.9], [observation(parent)], stamp)
    for stamp in (4., 5., 6.):
        tracker.update(['car'], [.9], [observation(fragment)], stamp)
    assert len(tracker.tracks) == 1 and tracker._pending[0].possible_parents == {1}
    for stamp in (7., 8., 9., 10.):
        tracker.update(['car', 'car'], [.9, .9], [observation(parent), observation(parent+[4.,0.,0.])], stamp)
    assert len(tracker.tracks) == 2
    assert not tracker.redirects


def test_connection_review_requires_joint_visibility_and_a_free_route():
    from navigation.vehicle_inspection import needs_inspection, observation_route
    record = dict(class_name='car', position_status='multi_surface_estimate',
        position_covariance=np.eye(3)*.001, bbox_3d={'min':[-2., -1., -1.], 'max':[2., 1., 0.]},
        connection_targets=[{'min':[3., .8, -1.], 'max':[4.5, 1., 0.], 'last_seen':10.}])
    assert needs_inspection(record, .4)
    grid = SimpleNamespace(free={0}, cell=lambda p: 0, path=Mock(side_effect=lambda a,b:np.array([a,b])))
    with patch('navigation.object_inspection.ObjectInspectionMemory._visible', return_value=False):
        assert observation_route(grid, np.array([0., 5., -1.5]), record, -1.5, 1.2, []) is None
        grid.path.assert_not_called()
    with patch('navigation.object_inspection.ObjectInspectionMemory._visible', return_value=True):
        assert observation_route(grid, np.array([0., 5., -1.5]), record, -1.5, 1.2, []) is not None
    grid.free.clear()
    assert observation_route(grid, np.array([0., 5., -1.5]), record, -1.5, 1.2, []) is None


def test_new_fragment_gets_one_review_even_after_prior_body_inspection():
    from navigation.frontier_explorer import FrontierExplorer
    navigator = object.__new__(FrontierExplorer)
    navigator._inspected_vehicle_ids = {'vehicle_01'}
    navigator.vehicle = Mock()
    navigator.vehicle.get_position.return_value = np.array([4., 0., -1.5])
    record = dict(class_name='car', runtime_instance_id='vehicle_01', position_world=[0.,0.,0.],
        position_status='multi_surface_estimate', position_covariance=np.eye(3)*.001, last_seen=1.,
        bbox_3d={'min':[-2.,-1.,-1.], 'max':[2.,1.,0.]}, connection_targets=[{
            'min':[3.,.8,-1.], 'max':[4.5,1.,0.], 'last_seen':20.}])
    navigator._frame = lambda: SimpleNamespace(timestamp=20., object_inspection_candidates=[record])
    navigator.config = SimpleNamespace(perception_timeout=5., clearance_radius=.4, standoff_distance=1.2)
    navigator.exploration_config = SimpleNamespace(information_radius=3., candidate_radius_max=2.5,
        vehicle_inspection_timeout=45., vehicle_inspection_views=2)
    navigator.clock=lambda:20.; navigator.deadline=100.; navigator._motion_progress=None; navigator.altitude=-1.5
    navigator._check_health=Mock(); navigator._checkpoint=Mock()
    navigator._grid=Mock(return_value=SimpleNamespace(free=set(), cell=lambda p:0))
    navigator._inspect_nearby_vehicle()
    navigator._inspect_nearby_vehicle()
    assert navigator._grid.call_count == 1
    assert len(navigator._connection_review_tokens['vehicle_01']) == 1
    assert navigator.deadline == 100.
    navigator.vehicle.move_waypoint_guided.assert_not_called()


def review_explorer(targets):
    from navigation.frontier_explorer import FrontierExplorer
    navigator = object.__new__(FrontierExplorer)
    navigator._inspected_vehicle_ids = {'vehicle_01'}
    navigator.vehicle = Mock()
    navigator.vehicle.get_position.return_value = np.array([4., 0., -1.5])
    record = dict(class_name='car', runtime_instance_id='vehicle_01', position_world=[0., 0., 0.],
        position_status='partial_surface', position_covariance=np.eye(3)*.001, last_seen=20.,
        bbox_3d={'min':[-2., -1., -1.], 'max':[2., 1., 0.]}, connection_targets=targets)
    navigator._frame = lambda: SimpleNamespace(timestamp=20., object_inspection_candidates=[record])
    navigator.config = SimpleNamespace(perception_timeout=5., clearance_radius=.4, standoff_distance=1.2)
    navigator.exploration_config = SimpleNamespace(information_radius=3., candidate_radius_max=2.5,
        vehicle_inspection_timeout=45., vehicle_inspection_views=2)
    navigator.clock = lambda:20.
    navigator.deadline = 100.
    navigator._motion_progress = None
    navigator.altitude = -1.5
    navigator._check_health = Mock()
    navigator._checkpoint = Mock()
    navigator._grid = Mock()
    return navigator, record


def target(key, stamp=20.):
    return dict(min=[3., .8, -1.], max=[4.5, 1., 0.], last_seen=stamp, hypothesis_key=key)


def test_old_unreviewed_region_is_not_starved_by_latest_already_reviewed_region():
    navigator, _ = review_explorer([target('recent'), target('older', 18.)])
    navigator._connection_review_tokens = {'vehicle_01': {'recent'}}
    with patch('navigation.vehicle_inspection.observation_route', return_value=None) as route:
        navigator._inspect_nearby_vehicle()
        assert route.call_args.args[2]['connection_targets'][0]['hypothesis_key'] == 'older'
        navigator._inspect_nearby_vehicle()
    assert route.call_count == 1


def test_same_fragment_new_timestamp_and_box_jitter_do_not_reopen_review():
    navigator, record = review_explorer([target('one')])
    with patch('navigation.vehicle_inspection.observation_route', return_value=None) as route:
        navigator._inspect_nearby_vehicle()
        record['connection_targets'] = [dict(target('one', 21.), min=[3.1, .8, -1.])]
        navigator._inspect_nearby_vehicle()
    assert route.call_count == 1


def test_distinct_connection_regions_share_one_total_time_budget():
    navigator, record = review_explorer([target('one')])
    navigator.exploration_config.vehicle_inspection_views = 4
    now, deadlines = [20.], []
    navigator.clock = lambda:now[0]
    def no_route(*args):
        deadlines.append(navigator.deadline)
        now[0] += 40. if len(deadlines) == 1 else 6.
        return None
    with patch('navigation.vehicle_inspection.observation_route', side_effect=no_route) as route:
        navigator._inspect_nearby_vehicle()
        record['connection_targets'] = [target('two')]
        navigator._inspect_nearby_vehicle()
        record['connection_targets'] = [target('three')]
        navigator._inspect_nearby_vehicle()
    assert route.call_count == 2 and deadlines == [65., 65.]
    assert navigator.deadline == 100.


def test_parent_merge_does_not_reset_connection_budget():
    navigator, record = review_explorer([target('new')])
    record['merged_from'] = ['vehicle_08']
    navigator._inspected_vehicle_ids = {'vehicle_08'}
    navigator._connection_review_usage = {'vehicle_08': 45.}
    with patch('navigation.vehicle_inspection.observation_route', return_value=None) as route:
        navigator._inspect_nearby_vehicle()
    route.assert_not_called()
