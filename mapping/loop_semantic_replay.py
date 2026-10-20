"""Replay individual semantic measurements in the optimized geometry frame."""
import json
from itertools import product
from collections import Counter, defaultdict
import numpy as np
from .semantic_instance_mapper import SemanticInstanceMapper, SemanticObservation3D, instance_class_name
from .semantic_voxels import SemanticVoxelEvidence


def correction_at(timestamp, timestamps, corrections):
    stamps = np.asarray(timestamps, float)
    if len(stamps) != len(corrections) or len(stamps) == 0 or np.any(np.diff(stamps) <= 0):
        raise ValueError('Correction timestamps must be strictly increasing')
    upper = int(np.searchsorted(stamps, timestamp))
    if upper == 0:
        return corrections[0].copy()
    if upper == len(stamps):
        return corrections[-1].copy()
    lower = upper-1
    fraction = (timestamp-stamps[lower])/(stamps[upper]-stamps[lower])
    first, last = corrections[lower], corrections[upper]
    yaw0 = np.arctan2(first[1,0],first[0,0])
    yaw1 = np.arctan2(last[1,0],last[0,0])
    delta = np.arctan2(np.sin(yaw1-yaw0),np.cos(yaw1-yaw0))
    yaw = yaw0 + fraction*delta
    result = np.eye(4)
    result[:3,:3] = [[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]]
    result[:3,3] = (1-fraction)*first[:3,3]+fraction*last[:3,3]
    return result


def transform_measurement(data, transform):
    rotation, shift = transform[:3,:3], transform[:3,3]
    corners = np.array(list(product(*zip(data['bbox_3d_min'],data['bbox_3d_max']))))
    corners = corners @ rotation.T + shift
    points = data.get('surface_points_world')
    if points is not None:
        points = np.asarray(points).reshape(-1,3) @ rotation.T + shift
    return SemanticObservation3D(
        rotation @ np.asarray(data['position_world']) + shift,
        corners.min(axis=0), corners.max(axis=0),
        rotation @ np.asarray(data['measurement_covariance']) @ rotation.T,
        appearance=None if data.get('appearance') is None else np.asarray(data['appearance']),
        color_scores=data.get('color_scores'), surface_points_world=points)


def replay(path, timestamps, corrections, resolution, max_voxels, *, live=False, check_cancelled=None):
    mapper = SemanticInstanceMapper(sign_min_observations=10)
    mapper.semantic_voxels = SemanticVoxelEvidence(resolution, max_voxels)
    last_stamp = -float('inf')
    last_packet = None
    skipped_duplicates = 0
    count = 0
    source_links = defaultdict(Counter)
    with open(path, encoding='utf-8') as stream:
        # Freeze the readable prefix: an active writer must not extend this job forever.
        import os
        end_offset = os.fstat(stream.fileno()).st_size
        line_number = 0
        while stream.tell() < end_offset:
            if check_cancelled is not None:
                check_cancelled()
            line = stream.readline()
            line_number += 1
            if stream.tell() > end_offset:
                break
            if live and not line.endswith("\n"):
                break  # The writer may still be appending this record.
            packet = json.loads(line)
            if 'samples' not in packet:
                raise ValueError('Old fused-record history cannot be used for observation replay')
            stamp = float(packet['timestamp'])
            if not np.isfinite(stamp) or stamp < last_stamp:
                raise ValueError(f'Semantic timestamp invalid or moved backwards at line {line_number}: {stamp}, previous={last_stamp}')
            if stamp == last_stamp:
                if packet != last_packet:
                    raise ValueError(f'Conflicting semantic records at line {line_number}, timestamp={stamp}')
                skipped_duplicates += 1
                continue
            last_stamp = stamp
            last_packet = packet
            transform = correction_at(stamp, timestamps, corrections)
            samples = packet['samples']
            replay_ids = mapper.update([s['class_name'] for s in samples], [s['confidence'] for s in samples],
                          [transform_measurement(s['measurement'], transform) for s in samples], stamp)
            for sample, identity in zip(samples, replay_ids):
                source_id = sample.get('source_instance_id')
                if identity is not None and source_id is not None:
                    category = instance_class_name(sample['class_name'])
                    source_links[f'{category}_{int(identity):02d}'][f'{category}_{int(source_id):02d}'] += 1
            count += len(samples)
    records = mapper.navigation_records(3)
    for record in records:
        links = Counter()
        for alias in [record['runtime_instance_id'], *record.get('merged_from', [])]:
            links.update(source_links[alias])
        record['source_runtime_instance_ids'] = sorted(links)
        record['source_identity_observations'] = dict(links)
    if skipped_duplicates:
        print(f'--- 语义回放已跳过 {skipped_duplicates} 条完全相同的重复记录 ----')
    return records, mapper.semantic_voxels.snapshot(records), count


def number_exported_vehicles(records, evidence):
    """Compact IDs once per published revision, preserving source provenance."""
    from copy import deepcopy
    from dataclasses import replace
    records = deepcopy(records)
    vehicles = sorted((r for r in records if r['runtime_instance_id'].startswith('vehicle_')),
                      key=lambda r:int(r['runtime_instance_id'].split('_')[-1]))
    mapping = {r['runtime_instance_id']:f'vehicle_{i:02d}' for i,r in enumerate(vehicles,1)}
    for record in vehicles:
        old = record['runtime_instance_id']
        record['source_replay_instance_id'] = old
        record['source_merged_from'] = record.get('merged_from', [])
        # Old IDs belong to another namespace; keeping them as aliases could
        # redirect a new vehicle_03 to the former vehicle_03 incorrectly.
        record['merged_from'] = []
        record['instance_id'] = record['runtime_instance_id'] = mapping[old]
        record['video_instance_id'] = None
    evidence = replace(evidence, instance_ids=np.asarray(
        [mapping.get(str(identity), str(identity)) for identity in evidence.instance_ids], dtype=str))
    return records, evidence, mapping


def export_bundle(run, output, snapshot, timestamps, corrections, *, source=None, live=False, on_publish=None, check_cancelled=None):
    """Publish one complete geometry/semantics/HTML revision by directory rename."""
    from pathlib import Path
    import uuid
    from config.exploration_config import FrontierExplorationConfig
    from navigation.scene_memory import SceneMemory, write_json
    from navigation.map_view import render_map
    if source is None:
        source = json.loads((Path(run)/'exploration/semantic_map.json').read_text())
    records, evidence, count = replay(Path(run)/'maps/semantic_observations.jsonl',
        timestamps, corrections, snapshot.resolution, max(1, len(snapshot.indices)*2), live=live, check_cancelled=check_cancelled)
    if check_cancelled is not None:
        check_cancelled()
    if live:
        # Do not compact IDs repeatedly while the map is growing.
        vehicle_ids = {r["runtime_instance_id"]:r["runtime_instance_id"] for r in records}
    else:
        records, evidence, vehicle_ids = number_exported_vehicles(records, evidence)
    revision = 'revision_' + uuid.uuid4().hex[:12]
    root = Path(output)
    stage = root / ('.'+revision+'.tmp')
    scene = SceneMemory(revision, source['home_ned'], source['initial_yaw_rad'], FrontierExplorationConfig())
    scene.records = {r['runtime_instance_id']:r for r in records}
    scene.semantic_voxels = evidence
    scene.flight_altitude = float(np.median(snapshot.trajectory_ned[:,2]))
    scene.exploration = dict(state='live_loop_corrected' if live else 'offline_loop_corrected', returned_home=False,
                             coordinate_frame='loop_corrected_ned', revision=revision)
    for i, position in enumerate(snapshot.trajectory_ned):
        scene.add_place(f'keyframe_{i:04}', position)
    if check_cancelled is not None:
        check_cancelled()
    scene.save_bundle(stage, snapshot)
    write_json(stage/'vehicle_id_mapping.json', {
        'source_namespace':'semantic_replay_before_numbering',
        'target_namespace':revision, 'old_to_new':vehicle_ids})
    write_json(stage/'source_identity_mapping.json', {
        'source_namespace': 'live_detection', 'target_namespace': revision,
        'replay_to_source': {r['runtime_instance_id']:r['source_identity_observations'] for r in records},
    })
    descriptor = json.loads((stage/'semantic_map.json').read_text())
    descriptor['coordinate_frame'] = 'loop_corrected_ned'
    descriptor['observation_replay_count'] = count
    descriptor['source_run'] = Path(run).name
    write_json(stage/'semantic_map.json', descriptor)
    manifest = json.loads((stage/'bundle.json').read_text())
    manifest['coordinate_frame'] = 'loop_corrected_ned'
    manifest['navigation_enabled'] = False
    write_json(stage/'bundle.json', manifest)
    if not live:
        render_map(stage/'map.html', snapshot, scene,
                   state='OFFLINE: loop corrected geometry + semantics; PX4 navigation separate')
    destination = root / revision
    if check_cancelled is not None:
        check_cancelled()
    stage.replace(destination)
    write_json(root/'latest.json', {'revision':revision, 'directory':revision,
                                  'coordinate_frame':'loop_corrected_ned'})
    if on_publish is not None:
        on_publish(snapshot, scene)
    return destination, count
