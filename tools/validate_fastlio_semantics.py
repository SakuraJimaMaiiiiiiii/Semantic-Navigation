"""Replay semantic measurements against recorded LIO poses without changing navigation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from localization.fastlio_semantics import FastLioSemantics
from navigation.scene_memory import write_json


def validate(run, output):
    if output.exists():
        raise ValueError('Use a new output file; existing validation is preserved')
    graph_output = output.with_name(output.stem + '_graph.json')
    if graph_output.exists():
        raise ValueError('Graph output already exists')
    bridge = FastLioSemantics()
    events = []
    with h5py.File(run/'fastlio/inputs.h5','r') as raw:
        clocks = {float(g.attrs['source_timestamp']):float(g.attrs['shared_timestamp'])
                  for g in raw['scans'].values()}
    anchor = None
    for line in (run/'fastlio/shadow_poses.jsonl').open():
        r = json.loads(line)
        if r['status'] != 'tracking':
            continue
        px4 = np.asarray(r['px4_pose'])
        pose = np.asarray(r['pose'])
        reference = np.eye(4)
        reference[:3,:3] = Rotation.from_quat(px4[[4,5,6,3]]).as_matrix()
        reference[:3,3] = px4[:3]
        if anchor is None:
            anchor = reference @ np.linalg.inv(pose)
        correction = anchor @ pose @ np.linalg.inv(reference)
        events.append((clocks[r['timestamp']], 0, correction))
    for line in (run/'maps/semantic_observations.jsonl').open(encoding='utf-8'):
        packet = json.loads(line)
        events.append((float(packet['timestamp']),1,packet))
    for stamp, kind, value in sorted(events, key=lambda event:(event[0],event[1])):
        if kind == 0:
            bridge.add_pose(stamp,value)
        else:
            bridge.submit(value)
    bridge.drain(final=True)
    write_json(output,dict(source=str(run), frame='lio_aligned_to_initial_px4_ned',
                           controls_navigation=False, synchronization=dict(bridge.counts),
                           objects=bridge.records))
    graph = bridge.scene_graph.snapshot(bridge.records, run.name)
    # Offline output is outside fastlio/, so reference the original evidence explicitly.
    for node in graph['nodes']:
        if node['type'] == 'observation':
            node['attributes']['evidence_ref']['path'] = str((run/'maps/semantic_observations.jsonl').resolve())
    write_json(graph_output, graph)
    print(json.dumps(dict(synchronization=dict(bridge.counts),objects=len(bridge.records)),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    validate(args.run,args.output)
