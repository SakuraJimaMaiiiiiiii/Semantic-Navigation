import copy
import pytest
from exploration_wildos.identity_overrides import merge_identities


def records():
    return [dict(runtime_instance_id=f'vehicle_{i:02}',position_world=[i,0,0],
                 bbox_3d=dict(min=[i-1,0,-1],max=[i+1,2,0]),merged_from=[],observation_count=5)
            for i in (2,5,8)]


def test_confirmed_identity_does_not_average_or_mutate_localization():
    before=records();source=copy.deepcopy(before)
    spec=dict(session_id='one',redirects={'vehicle_08':'vehicle_02','vehicle_07':'vehicle_05'})
    merged=merge_identities(source,spec,'one');by_id={r['runtime_instance_id']:r for r in merged}
    assert len(merged)==2 and source==before
    assert by_id['vehicle_02']['position_world']==before[0]['position_world']
    assert by_id['vehicle_02']['bbox_3d']==before[0]['bbox_3d']
    assert by_id['vehicle_02']['planning_bbox_3d']['max']==before[2]['bbox_3d']['max']
    assert by_id['vehicle_02']['merged_from']==['vehicle_08']
    assert by_id['vehicle_05']['merged_from']==['vehicle_07']
    assert merge_identities(merged,spec,'one')==merged


def test_confirmations_cannot_leak_to_another_run():
    with pytest.raises(ValueError,match='different session'):
        merge_identities(records(),dict(session_id='one',redirects={'vehicle_08':'vehicle_02'}),'two')


def test_live_number_confirmation_follows_final_numbering_without_merging_another_car():
    source = records()
    for record, final_id, live_id in zip(source, (5, 6, 7), (5, 7, 8)):
        record.update(runtime_instance_id=f'vehicle_{final_id:02d}',
                      source_runtime_instance_id=f'vehicle_{live_id:02d}')
    spec = dict(session_id='one', source_namespace='live_detection',
                redirects={'vehicle_08': 'vehicle_05'})
    merged = merge_identities(source, spec, 'one')
    by_id = {r['runtime_instance_id']: r for r in merged}
    assert set(by_id) == {'vehicle_05', 'vehicle_06'}
    assert by_id['vehicle_06'] == source[1]  # Live 07 is a different car.
    assert by_id['vehicle_05']['merged_from'] == ['vehicle_07']
    assert by_id['vehicle_05']['source_merged_from'] == ['vehicle_08']
    assert by_id['vehicle_05']['position_world'] == source[0]['position_world']
    assert merge_identities(merged, spec, 'one') == merged


def test_missing_live_number_does_not_match_an_unrelated_final_number():
    source = records()
    for record, identity in zip(source, (2, 5, 10)):
        record['source_runtime_instance_id'] = f'vehicle_{identity:02d}'
    spec = dict(session_id='one', source_namespace='live_detection',
                redirects={'vehicle_08': 'vehicle_05'})
    merged = merge_identities(source, spec, 'one')
    assert len(merged) == 3
    assert merged[2] == source[2]  # Final 08 came from live 10, not live 08.


def test_column_confirmation_uses_live_ids_and_preserves_vehicle_confirmation(tmp_path):
    import json
    from exploration_wildos.identity_overrides import corrected_records
    source = records()
    for identity, original in [(9, 14), (23, 70)]:
        source.append(dict(runtime_instance_id=f'column_{identity:02d}',
                           source_runtime_instance_id=f'column_{original:02d}', class_name='column',
                           position_world=[4., 23.6, -1.3],
                           bbox_3d=dict(min=[3.5, 23., -2.6], max=[4.6, 24.2, 0.])))
    vehicle = dict(session_id='one', redirects={'vehicle_08': 'vehicle_05'})
    columns = dict(session_id='one', source_namespace='live_detection',
                   redirects={'column_70': 'column_14'})
    for name, spec in [('vehicle_identity_overrides.json', vehicle), ('semantic_identity_overrides.json', columns)]:
        (tmp_path/name).write_text(json.dumps(spec), encoding='utf-8')
    merged = corrected_records(source, tmp_path, 'one')
    by_id = {r['runtime_instance_id']: r for r in merged}
    assert set(by_id) == {'vehicle_02', 'vehicle_05', 'column_09'}
    assert by_id['column_09']['merged_from'] == ['column_23']
    assert by_id['column_09']['identity_evidence'] == 'user_confirmed_same_column'
    assert by_id['vehicle_05']['merged_from'] == ['vehicle_08']
    assert corrected_records(merged, tmp_path, 'one') == merged


def test_old_vehicle_number_resolves_to_canonical_target():
    import numpy as np
    from navigation.target import SemanticTargetRequest, select_targets
    source=records()
    for record in source:
        record.update(class_name='car',confidence=1.,last_seen=1.)
    merged=merge_identities(source,dict(session_id='one',redirects={'vehicle_08':'vehicle_02'}),'one')
    matches=select_targets(merged,SemanticTargetRequest(instance_id='vehicle_08'),np.zeros(3),1.,check_age=False)
    assert [r['runtime_instance_id'] for r in matches]==['vehicle_02']
