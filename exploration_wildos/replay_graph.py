"""Replay ordered, explicitly aligned snapshots without running a flight session.

Manifest paths are relative to its parent. All snapshots must share frame_key.
Optional semantic_map entries enable the same vehicle constraints as the main UI.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from .graph_export import export_graph
from .incremental_graph import update_graph
from .map_adapter import load_map
from .semantic_graph import attach_objects
from .static_graph import GraphConfig, validate_graph
from .vehicle_constraints import constrain_adapter, validate_vehicles


def replay(manifest_path, output, initial_state=None):
    manifest_path,output=Path(manifest_path),Path(output)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if output.exists():
        raise ValueError('Output already exists; choose a new replay directory')
    if not manifest.get('snapshots') or not manifest.get('frame_key'):
        raise ValueError('Manifest requires frame_key and a nonempty snapshots list')
    options=GraphConfig(**manifest.get('options',{}))
    state=json.loads(Path(initial_state).read_text()) if initial_state else None
    output.parent.mkdir(parents=True,exist_ok=True)
    summaries=[]
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage=Path(temp)/'replay';stage.mkdir()
        for index,entry in enumerate(manifest['snapshots']):
            if entry.get('frame_key',manifest['frame_key'])!=manifest['frame_key']:
                raise ValueError('Snapshot coordinate frame changed; align it or start a new epoch')
            adapter=load_map(manifest_path.parent/entry['map'],altitude=entry.get('altitude'),
                clearance=entry.get('clearance'),vertical_half_extent=entry.get('vertical_half_extent'))
            objects=[];meta=None;states=None;constraints=None
            if entry.get('semantic_map'):
                semantic_path=manifest_path.parent/entry['semantic_map']
                semantic_bytes=semantic_path.read_bytes();meta=json.loads(semantic_bytes)
                if meta.get('coordinate_frame')!='world_NED':
                    raise ValueError('Vehicle objects must use world_NED coordinates')
                if adapter.metadata['source_sha256'] not in [meta[k]['sha256'] for k in ('occupancy_2d','occupancy_3d')]:
                    raise ValueError('Semantic and occupancy snapshots do not match')
                archive=semantic_path.parent/meta['occupancy_3d']['file']
                if hashlib.sha256(archive.read_bytes()).hexdigest()!=meta['occupancy_3d']['sha256']:
                    raise ValueError('Semantic 3-D archive checksum mismatch')
                with np.load(archive,allow_pickle=False) as data:
                    states={tuple(c):'occupied' if v>=data['occupied_threshold'] else 'free' if v<=data['free_threshold'] else 'unknown'
                            for c,v in zip(data['indices'],data['log_odds'])}
                scratch={'nodes':[]};attach_objects(scratch,meta,states,adapter.grid.resolution,max_links=0)
                objects=scratch['semantic_nodes'];adapter,constraints=constrain_adapter(adapter,objects)
                adapter.metadata['semantic_source_sha256']=hashlib.sha256(semantic_bytes).hexdigest()
            graph,state=update_graph(adapter,frame_key=manifest['frame_key'],state=state,options=options,start=manifest.get('start_ned'))
            if meta:
                attach_objects(graph,meta,states,adapter.grid.resolution)
                graph['semantic_source_sha256']=adapter.metadata['semantic_source_sha256']
                graph['vehicle_constraints']=constraints
            validation=validate_graph(adapter,graph)
            validation['vehicle_geometry']=validate_vehicles(graph,objects,adapter.grid.config)
            validation['passed'] &= validation['vehicle_geometry']['passed']
            if not validation['passed']:
                raise ValueError(f'Snapshot {index} failed geometry checks')
            state['graph']=graph
            export_graph(stage/f'snapshot_{index:04d}',adapter,graph,validation)
            summaries.append(dict(snapshot=index,source_sha256=adapter.metadata['source_sha256'],
                nodes=len(graph['nodes']),edges=len(graph['edges']),frontiers=len(graph['frontiers']),
                uncovered=len(graph['diagnostics']['uncovered_safe_cells']),events=graph['incremental']))
        (stage/'incremental_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        (stage/'replay_summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
        (stage/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        stage.rename(output)
    return summaries


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial-state',type=Path)
    args=parser.parse_args(argv)
    try:
        result=replay(args.manifest,args.output,args.initial_state)
    except (ValueError,OSError,KeyError) as error:
        parser.error(str(error))
    print(json.dumps(dict(output=str(args.output.resolve()),snapshots=len(result)),ensure_ascii=False))


if __name__=='__main__':
    main()
