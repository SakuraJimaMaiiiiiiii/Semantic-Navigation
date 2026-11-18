import numpy as np
import pytest
from test_column_identity import faces, observation
from mapping.semantic_instance_mapper import SemanticInstanceMapper
from mapping.explore_v3.adapters import SemanticInstanceMapper as V3Mapper
from navigation.vehicle_inspection import needs_inspection


@pytest.mark.parametrize('mapper_type', [SemanticInstanceMapper, V3Mapper])
@pytest.mark.parametrize('angle', [0., .7, 1.8])
def test_real_two_face_column_confirms_at_any_heading(mapper_type, angle):
    a, b = faces()
    points = np.vstack([a, b])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points[:, :2] = points[:, :2] @ rotation.T
    mapper = mapper_type()
    for stamp in range(1, 6):
        mapper.update(['column'], [.95], [observation(points)], float(stamp))
    assert len(mapper.navigation_records(3)) == 1
    assert mapper.confirmed_instance_id('column', 1, 3) == 1


@pytest.mark.parametrize('mapper_type', [SemanticInstanceMapper, V3Mapper])
@pytest.mark.parametrize('kind', ['flat_wall', 'thin_strip', 'large_wall_corner'])
def test_high_confidence_wall_geometry_never_publishes_as_column(mapper_type, kind):
    a, b = faces()
    points = a if kind == 'flat_wall' else a * [1., .05, 1.] if kind == 'thin_strip' else np.vstack([a, b])*[3., 3., 1.]
    mapper = mapper_type()
    for stamp in range(1, 12):
        mapper.update(['column'], [.99], [observation(points)], float(stamp))
    assert not mapper.navigation_records(3)
    assert mapper.confirmed_instance_id('column', 1, 3) is None


@pytest.mark.parametrize('mapper_type', [SemanticInstanceMapper, V3Mapper])
def test_single_face_stays_pending_and_requests_second_view(mapper_type):
    mapper = mapper_type()
    a, b = faces()
    for stamp in (1., 2., 3.):
        mapper.update(['column'], [.9], [observation(a)], stamp)
    assert not mapper.navigation_records(3)
    pending = mapper.object_inspection_records([])
    assert len(pending) == 1 and needs_inspection(pending[0], .4)
    assert pending[0]['geometry_status'] == 'needs_second_face'
    for stamp in (4., 5., 6.):
        mapper.update(['column'], [.9], [observation(np.vstack([a, b]))], stamp)
    assert [r['runtime_instance_id'] for r in mapper.navigation_records(3)] == ['column_01']
