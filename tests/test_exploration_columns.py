"""Default exploration must publish structural columns, not only detect masks."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from config.exploration_config import FrontierExplorationConfig
from detections.object_detect import DetectorConfig, SemanticMaskProcessor
from mapping.semantic_instance_mapper import SemanticInstanceMapper, SemanticObservation3D
from mapping.explore_v3.adapters import SemanticInstanceMapper as V3Mapper
from mapping.semantic_voxels import SemanticVoxelEvidence
from navigation.scene_memory import SceneMemory
from navigation.map_view import render_map


def measurement(x, height):
    points = np.array([[x + .5, y, z] for y in np.linspace(.05, .55, 6)
                       for z in np.linspace(-height, 0., 29)])
    points = np.vstack([points, [[a, .55, z] for a in np.linspace(x, x+.5, 6)
                                 for z in np.linspace(-height, 0., 29)]])
    low, high = np.array([x, 0., -height]), np.array([x + .6, .6, 0.])
    return SemanticObservation3D(
        (low + high) / 2, low, high, np.eye(3) * .01,
        surface_points_world=points,
    )


@pytest.mark.parametrize('mapper_type', [SemanticInstanceMapper, V3Mapper])
def test_default_columns_reach_semantic_bundle_and_map_with_stable_ids(tmp_path, mapper_type):
    config = DetectorConfig(backend='sam3')
    processor = SemanticMaskProcessor(config, mapper_type=mapper_type)
    assert 'parking garage structural column' in config.class_prompts
    assert processor._is_semantic_landmark('parking garage structural column')
    mapper = processor.mapper
    mapper.semantic_voxels = SemanticVoxelEvidence(.3, 1000)
    names = ['parking garage structural column'] * 2
    observations = [measurement(0., 2.8), measurement(5., .8)]
    for stamp in (1., 2., 3., 4.):
        mapper.update(names, [.9, .9], observations, stamp)
    records = mapper.navigation_records(3)
    assert len(records) == 1
    assert records[0]['runtime_instance_id'] == 'column_01'
    assert records[0]['class_name'] == 'column'
    assert records[0]['is_obstacle'] and records[0]['is_static']
    assert records[0]['observation_count'] == 4
    evidence = mapper.semantic_voxels.snapshot(records)
    assert len(evidence.indices) > 0
    snapshot = SimpleNamespace(
        indices=evidence.indices, log_odds=np.ones(len(evidence.indices)) * 2.,
        resolution=.3, occupied_threshold=.7, free_threshold=-.2, timestamp=4.,
    )
    memory = SceneMemory('columns-test', np.zeros(3), 0., FrontierExplorationConfig())
    memory.records = {record['runtime_instance_id']:record for record in records}
    memory.semantic_voxels = evidence
    memory.exploration['returned_home'] = True
    memory.save_bundle(tmp_path, snapshot)
    render_map(tmp_path / 'map.html', snapshot, memory)
    semantics = json.loads((tmp_path / 'semantic_map.json').read_text(encoding='utf-8'))
    graph = json.loads((tmp_path / 'semantic_graph.json').read_text(encoding='utf-8'))
    assert [r['runtime_instance_id'] for r in semantics['objects']] == ['column_01']
    assert any(node['id'] == 'column_01' for node in graph['nodes'])
    assert 'column_01' in (tmp_path / 'map.html').read_text(encoding='utf-8')
    with np.load(tmp_path / 'private/navigation_cache.npz', allow_pickle=False) as data:
        assert 'column_01' in data['semantic_instance_ids']
        assert np.any(data['semantic_labels'] > 0)
