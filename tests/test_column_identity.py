"""Identity comes from repeated measured surfaces, never recorded object IDs."""
import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from mapping.semantic_instance_mapper import SemanticInstanceMapper, SemanticObservation3D, _SemanticLandmark
from mapping.explore_v3.adapters import SemanticInstanceMapper as V3Mapper
from mapping.semantic_voxels import SemanticVoxelEvidence
from mapping.loop_semantic_replay import replay
from config.exploration_config import FrontierExplorationConfig
from navigation.scene_memory import SceneMemory, load_bundle
from navigation.target import SemanticTargetRequest, select_targets


def faces():
    face = np.array([[0., y, z] for y in np.linspace(0., 1., 11)
                     for z in np.linspace(-2.8, 0., 29)])
    side = np.array([[x, 1., z] for x in np.linspace(0., 1., 11)
                     for z in np.linspace(-2.8, 0., 29)])
    return face, side


def observation(points):
    return SemanticObservation3D(points.mean(0), points.min(0), points.max(0),
                                  np.eye(3)*.0001, surface_points_world=points)


def seed_split(mapper, shift=None):
    a, b = faces()
    points = np.vstack([a, b])
    for identity in (1, 2):
        cloud = points if identity == 1 or shift is None else points+shift
        obs = observation(cloud)
        track = _SemanticLandmark(.9, obs.position_world.copy(), np.eye(3)*.0001,
                                  obs.bbox_3d_min.copy(), obs.bbox_3d_max.copy(), 6, 0.)
        track.surface.update(cloud); track.surface.update(cloud)
        mapper._landmarks['column', identity] = track
    mapper._next_instance_id['column'] = 3
    mapper._distinct_landmark_pairs.add(frozenset({('column', 1), ('column', 2)}))
    return points


@pytest.mark.parametrize('mapper_type', [SemanticInstanceMapper, V3Mapper])
def test_overlapping_proposals_do_not_mint_two_column_ids(mapper_type):
    mapper = mapper_type()
    a, b = faces()
    a = np.vstack([a, b])
    for stamp in (1., 2., 3., 4.):
        ids = mapper.update(['column', 'column'], [.9, .8], [observation(a), observation(a)], stamp)
        assert ids == [1, None]
    records = mapper.navigation_records(3)
    assert len(records) == 1 and records[0]['observation_count'] == 4


@pytest.mark.parametrize('mapper_type', [SemanticInstanceMapper, V3Mapper])
@pytest.mark.parametrize('angle', [0., .7, 1.8])
def test_connecting_corner_keeps_identity_when_surface_centroid_changes(mapper_type, angle):
    mapper = mapper_type()
    a, b = faces()
    basis = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for points in (a, b):
        points[:, :2] = points[:, :2] @ basis.T
    # Most of the new view is a different face. Its centroid is not the pillar center.
    corner = np.vstack([a.reshape(11, 29, 3)[-4:].reshape(-1, 3), b])
    for stamp, cloud in enumerate([a]*3+[corner]*4, 1):
        assert mapper.update(['column'], [.9], [observation(cloud)], float(stamp)) == [1]
    assert len(mapper.navigation_records(3)) == 1


@pytest.mark.parametrize('mapper_type', [SemanticInstanceMapper, V3Mapper])
def test_repeated_shared_views_repair_historical_same_frame_split(mapper_type, tmp_path):
    mapper = mapper_type()
    points = seed_split(mapper)
    mapper.semantic_voxels = SemanticVoxelEvidence(.3, 1000)
    for stamp in (-2., -1.):
        mapper.semantic_voxels.update([('column_01', points), ('column_02', points)], stamp)
    memory = SceneMemory('test', np.zeros(3), 0., FrontierExplorationConfig())
    memory.add_place('place_0000', np.zeros(3))
    memory.observe(SimpleNamespace(timestamp=0., records=mapper.navigation_records(3)), 'place_0000')
    for stamp in (1., 2.):
        mapper.update(['column'], [.9], [observation(points)], stamp)
        assert len(mapper._landmarks) == 2
    candidates = mapper.object_inspection_records(mapper.navigation_records(3))
    assert all(r['identity_status'] == 'needs_connection_view' for r in mapper.navigation_records(3))
    parent = next(r for r in candidates if r['runtime_instance_id'] == 'column_01')
    assert parent['connection_targets'][0]['target_id'] == 'column_02'
    from navigation.vehicle_inspection import needs_inspection, observation_route
    from test_vehicle_inspection import grid
    assert needs_inspection(parent, .4)
    assert observation_route(grid(), np.array([3., 3., -1.5]), parent, -1.5, 1.2, []) is not None
    mapper.update(['column'], [.9], [observation(points)], 3.)
    records = mapper.navigation_records(3)
    assert len(records) == 1 and records[0]['merged_from'] == ['column_02']
    assert records[0]['identity_status'] == 'observed'
    assert records[0]['last_seen'] == 3.
    assert mapper.update(['column'], [.9], [observation(points)], 4.) == [1]
    assert mapper._canonical_landmark_key(('column', 2)) == ('column', 1)
    assert not any(needs_inspection(r, .4) for r in mapper.object_inspection_records(mapper.navigation_records(3)))
    evidence = mapper.semantic_voxels.snapshot(records)
    assert len(evidence.indices) and set(evidence.instance_ids) == {'column_01'}
    memory.observe(SimpleNamespace(timestamp=4., records=mapper.navigation_records(3), semantic_voxels=evidence))
    assert set(memory.records) == {'column_01'}
    assert 'column_02' not in {node['id'] for node in memory.graph()['nodes']}
    selected = select_targets(list(memory.records.values()), SemanticTargetRequest(instance_id='column_02'),
                               np.zeros(3), 4., check_age=False)
    assert [r['runtime_instance_id'] for r in selected] == ['column_01']
    snapshot = SimpleNamespace(indices=evidence.indices, log_odds=np.full(len(evidence.indices), 2.),
                               resolution=.3, free_threshold=-.2, occupied_threshold=.7, timestamp=4.)
    memory.exploration['returned_home'] = True
    memory.save_bundle(tmp_path, snapshot)
    assert [r['runtime_instance_id'] for r in load_bundle(tmp_path, 'test')['records']] == ['column_01']


@pytest.mark.parametrize('mapper_type', [SemanticInstanceMapper, V3Mapper])
def test_nearby_independent_columns_remain_separate(mapper_type):
    mapper = mapper_type()
    points = seed_split(mapper, np.array([1.4, 0., 0.]))
    for stamp in (1., 2., 3., 4.):
        assert mapper.update(['column', 'column'], [.9, .9],
                             [observation(points), observation(points+[1.4, 0., 0.])], stamp) == [1, 2]
    assert len(mapper.navigation_records(3)) == 2
    assert not mapper._landmark_redirects


@pytest.mark.parametrize('mapper_type', [SemanticInstanceMapper, V3Mapper])
def test_missing_or_repeated_frames_cannot_confirm_old_split(mapper_type):
    mapper = mapper_type()
    points = seed_split(mapper)
    mapper.update(['column'], [.9], [observation(points)], 1.)
    for _ in range(3):
        mapper.update(['column'], [.9], [observation(points)], 1.)
    mapper.update([], [], [], 20.)
    mapper.update(['column'], [.9], [observation(points)], 21.)
    assert len(mapper._landmarks) == 2


@pytest.mark.parametrize('mapper_type', [SemanticInstanceMapper, V3Mapper])
def test_upper_and_lower_parts_merge_only_with_fresh_continuous_column_view(mapper_type):
    mapper = mapper_type()
    face, _ = faces()
    for identity, points in [(1, face[face[:, 2] <= -1.4]), (2, face[face[:, 2] >= -1.4])]:
        obs = observation(points)
        track = _SemanticLandmark(.9, obs.position_world.copy(), np.eye(3)*.001,
                                  obs.bbox_3d_min.copy(), obs.bbox_3d_max.copy(), 6, 0.)
        track.surface.update(points); track.surface.update(points)
        mapper._landmarks['column', identity] = track
    mapper._next_instance_id['column'] = 3
    for stamp in (1., 2., 3.):
        mapper.update(['column'], [.9], [observation(face)], stamp)
    assert mapper._canonical_landmark_key(('column', 2)) == ('column', 1)
    assert not mapper.navigation_records(3)  # A continuous plane alone is still not a confirmed pillar.
    _, side = faces()
    for stamp in (4., 5., 6.):
        mapper.update(['column'], [.9], [observation(np.vstack([face, side]))], stamp)
    assert len(mapper.navigation_records(3)) == 1


@pytest.mark.parametrize('mapper_type', [SemanticInstanceMapper, V3Mapper])
def test_similar_boxes_without_surface_evidence_cannot_merge(mapper_type):
    mapper = mapper_type()
    points = seed_split(mapper)
    for track in mapper._landmarks.values():
        track.surface.cells.clear()
    for stamp in (1., 2., 3., 4.):
        obs = replace(observation(points), surface_points_world=None)
        mapper.update(['column'], [.9], [obs], stamp)
    assert not mapper._landmark_redirects


@pytest.mark.parametrize('mapper_type', [SemanticInstanceMapper, V3Mapper])
def test_static_split_can_be_confirmed_across_revisits(mapper_type):
    mapper = mapper_type()
    points = seed_split(mapper)
    for stamp in (1., 30., 60.):
        mapper.update(['column'], [.9], [observation(points)], stamp)
    assert len(mapper.navigation_records(3)) == 1


@pytest.mark.parametrize('mapper_type', [SemanticInstanceMapper, V3Mapper])
def test_short_visible_patch_keeps_id_of_a_model_seen_from_all_sides(mapper_type):
    mapper = mapper_type()
    a, b = faces()
    whole = np.vstack([a, b, a+[1., 0., 0.], b+[0., -1., 0.]])
    for stamp in (1., 2., 3.):
        mapper.update(['column'], [.9], [observation(whole)], stamp)
    patch = a[(a[:, 1] <= .4) & (a[:, 2] >= -1.3)]
    for stamp in (4., 5., 6.):
        assert mapper.update(['column'], [.9], [observation(patch)], stamp) == [1]
    assert len(mapper.navigation_records(3)) == 1


@pytest.mark.parametrize('mapper_type', [SemanticInstanceMapper, V3Mapper])
def test_one_shared_face_of_different_multiface_models_repairs_split(mapper_type):
    mapper = mapper_type()
    a, b = faces()
    top = np.array([[x, y, 0.] for x in np.linspace(0., 1., 11) for y in np.linspace(0., 1., 11)])
    clouds = [np.vstack([a, b+[0., -1., 0.], a+[1., 0., 0.]]), np.vstack([a, b, top])]
    for identity, points in enumerate(clouds, 1):
        obs = observation(points)
        track = _SemanticLandmark(.9, obs.position_world.copy(), np.eye(3)*.001,
                                  obs.bbox_3d_min.copy(), obs.bbox_3d_max.copy(), 6, 0.)
        track.surface.update(points); track.surface.update(points)
        mapper._landmarks['column', identity] = track
    mapper._next_instance_id['column'] = 3
    mapper._distinct_landmark_pairs.add(frozenset({('column', 1), ('column', 2)}))
    for stamp in (1., 2., 3.):
        mapper.update(['column'], [.9], [observation(a)], stamp)
    assert len(mapper.navigation_records(3)) == 1


@pytest.mark.parametrize('mapper_type', [SemanticInstanceMapper, V3Mapper])
def test_loop_replay_ignores_recorded_ids_and_keeps_live_column_geometry(tmp_path, mapper_type):
    mapper = mapper_type()
    a, b = faces()
    packets = []
    for stamp, points in enumerate([a]*3+[np.vstack([a, b])]*4, 1):
        obs = observation(points)
        mapper.update(['column'], [.9], [obs], float(stamp))
        measurement = dict(position_world=obs.position_world.tolist(), bbox_3d_min=obs.bbox_3d_min.tolist(),
                           bbox_3d_max=obs.bbox_3d_max.tolist(), measurement_covariance=obs.measurement_covariance.tolist(),
                           surface_points_world=points.tolist())
        packets.append(dict(timestamp=float(stamp), samples=[dict(class_name='column', confidence=.9,
                            source_instance_id=99+stamp, measurement=measurement)]))
    path = tmp_path/'observations.jsonl'
    path.write_text(''.join(json.dumps(p)+'\n' for p in packets), encoding='utf-8')
    records, _, _ = replay(path, [1., 7.], [np.eye(4), np.eye(4)], .3, 1000, mapper_type=mapper_type)
    assert len(records) == 1 and records[0]['runtime_instance_id'] == 'column_01'
    assert records[0]['bbox_3d'] == mapper.navigation_records(3)[0]['bbox_3d']
