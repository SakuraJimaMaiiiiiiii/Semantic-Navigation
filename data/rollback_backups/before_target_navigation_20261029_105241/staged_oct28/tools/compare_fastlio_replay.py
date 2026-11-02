"""Compare paired LIO/PX4 poses and identical scan surfaces; PX4 is NOT ground truth."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def downsample(points, size=.1):
    _, keep = np.unique(np.floor(points/size).astype(np.int64), axis=0, return_index=True)
    return points[keep]


def surface_spread(points):
    # Local plane residual is a roughness proxy, not a calibrated map error.
    if len(points) < 30:
        return None
    tree = cKDTree(points)
    queries = points[np.linspace(0, len(points)-1, min(2000, len(points)), dtype=int)]
    distance, indices = tree.query(queries, k=12)
    local = points[indices]
    local -= local.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', local, local)/12
    spread = np.sqrt(np.maximum(0, np.linalg.eigvalsh(cov)[:, 0]))
    valid = distance[:, -1] <= .6
    return dict(points=len(points), valid_neighborhoods=int(valid.sum()),
                plane_residual_median_m=float(np.median(spread[valid])) if valid.any() else None,
                plane_residual_p90_m=float(np.percentile(spread[valid],90)) if valid.any() else None)


def compare(source, replay, semantic_map=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = [json.loads(line) for line in (replay/'trajectory.jsonl').read_text().splitlines()]
    records = [r for r in records if 'pose' in r]
    if not records:
        raise ValueError('No LIO poses to compare')
    poses = np.asarray([r['pose'] for r in records])
    px4 = np.asarray([r['px4_pose'] for r in records])
    quats = px4[:, 3:7]  # recorder stores wxyz
    rotations = Rotation.from_quat(quats[:, [1,2,3,0]]).as_matrix()
    rotation = rotations[0] @ poses[0,:3,:3].T
    shift = px4[0,:3] - rotation @ poses[0,:3,3]
    aligned = poses[:,:3,3] @ rotation.T + shift
    error = np.linalg.norm(aligned-px4[:,:3], axis=1)
    stamps = np.asarray([r['timestamp'] for r in records])
    clouds, lio_clouds = [], []
    with h5py.File(source, 'r') as f:
        for i,r in enumerate(records):
            points = f['scans'][r['scan_key']]['points_body'][:]
            clouds.append(points @ rotations[i].T + px4[i,:3])
            world = points @ poses[i,:3,:3].T + poses[i,:3,3]
            lio_clouds.append(world @ rotation.T + shift)
            if (i+1) % 100 == 0:
                clouds = [downsample(np.vstack(clouds))]
                lio_clouds = [downsample(np.vstack(lio_clouds))]
    px4_cloud = downsample(np.vstack(clouds))
    lio_cloud = downsample(np.vstack(lio_clouds))
    np.savez_compressed(replay/'comparison_maps.npz', px4=px4_cloud, lio_aligned=lio_cloud,
                        rotation_lio_to_px4=rotation, translation_lio_to_px4=shift)
    report = dict(reference='PX4 estimate, NOT ground truth', alignment='first matched pose SE(3), no scale or full-trajectory fitting',
                  matched_poses=len(records), duration_s=float(stamps[-1]-stamps[0]),
                  disagreement_rmse_m=float(np.sqrt(np.mean(error**2))),
                  disagreement_median_m=float(np.median(error)), disagreement_p95_m=float(np.percentile(error,95)),
                  disagreement_max_m=float(error.max()), endpoint_disagreement_m=float(error[-1]),
                  endpoint_displacement_px4_m=float(np.linalg.norm(px4[-1,:3]-px4[0,:3])),
                  endpoint_displacement_lio_m=float(np.linalg.norm(aligned[-1]-aligned[0])),
                  endpoint_note='Displacements are not closure errors unless start/end physically coincide.',
                  scan_timing_verified=False, online_navigation_validated=False)
    # Late return near initial pose, avoiding the initial stationary period.
    late = np.flatnonzero(stamps-stamps[0] > (stamps[-1]-stamps[0])*.5)
    if len(late):
        k = late[np.argmin(np.linalg.norm(px4[late,:3]-px4[0,:3], axis=1))]
        report['nearest_late_return'] = dict(timestamp=float(stamps[k]),
            px4_distance_to_start_m=float(np.linalg.norm(px4[k,:3]-px4[0,:3])),
            lio_distance_to_start_m=float(np.linalg.norm(aligned[k]-aligned[0])),
            paired_disagreement_m=float(error[k]))
    objects = json.loads(semantic_map.read_text())['objects'] if semantic_map is not None else []
    regions = {}
    for obj in objects:
        if not obj['instance_id'].startswith('vehicle_'):
            continue
        bounds = obj['bbox_3d'];lo=np.array(bounds['min'])-.3;hi=np.array(bounds['max'])+.3
        regions[obj['instance_id']] = {
            name:surface_spread(cloud[np.all((cloud>=lo)&(cloud<=hi), axis=1)])
            for name,cloud in [('px4',px4_cloud),('lio',lio_cloud)]}
    report['vehicle_roi_surface_proxy'] = regions
    # Axis-aligned outer-wall strips are visualization ROIs, not fitted ground truth.
    walls = {}
    for name,axis in [('north',0),('east',1)]:
        for side, percentile in [('low',2),('high',98)]:
            plane = float(np.percentile(px4_cloud[:,axis], percentile))
            walls[f'{name}_{side}'] = dict(coordinate=plane, half_width_m=.35,
                surfaces={label:surface_spread(cloud[(np.abs(cloud[:,axis]-plane)<=.35)
                    & (cloud[:,2]>-2.5) & (cloud[:,2]<-.2)])
                    for label,cloud in [('px4',px4_cloud),('lio',lio_cloud)]})
    report['outer_wall_strip_surface_proxy'] = walls
    report['surface_proxy_caveat'] = 'Local plane residual is not ground-truth accuracy; partial surfaces, sampling, and wrong associations affect it.'
    (replay/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1,2,figsize=(13,6))
    axes[0].plot(px4[:,0],px4[:,1],label='PX4');axes[0].plot(aligned[:,0],aligned[:,1],label='LIO (first-pose aligned)')
    axes[0].axis('equal');axes[0].legend();axes[0].set(xlabel='North (m)',ylabel='East (m)',title='Trajectory: PX4 is not ground truth')
    axes[1].plot(stamps-stamps[0],error);axes[1].set(xlabel='Time (s)',ylabel='Position disagreement (m)',title='Fixed initial alignment; no scale fitting')
    fig.tight_layout();fig.savefig(replay/'trajectory_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,15))
    for ax,cloud,title in zip(axes,[px4_cloud,lio_cloud],['PX4 same scans','FAST-LIO same scans']):
        select=cloud[(cloud[:,2]>-2.5)&(cloud[:,2]<-.2)]
        ax.scatter(select[:,0],select[:,1],s=.08,c=select[:,2],cmap='viridis',vmin=-2.5,vmax=-.2)
        ax.set_title(title);ax.set_aspect('equal', adjustable='box');ax.set(xlabel='North (m)',ylabel='East (m)')
    axes[1].set_xlim(axes[0].get_xlim());axes[1].set_ylim(axes[0].get_ylim())
    fig.tight_layout();fig.savefig(replay/'map_comparison.png',dpi=180);plt.close(fig)
    if objects:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        groups = [('vehicle_01',), ('vehicle_02',), ('vehicle_06', 'vehicle_10')]
        for col, group in enumerate(groups):
            selected = [o for o in objects if o['instance_id'] in group]
            if not selected:
                continue
            lo = np.min([o['bbox_3d']['min'] for o in selected], axis=0)-.4
            hi = np.max([o['bbox_3d']['max'] for o in selected], axis=0)+.4
            for row, (cloud, title) in enumerate([(px4_cloud, 'PX4'), (lio_cloud, 'LIO')]):
                points = cloud[np.all((cloud>=lo)&(cloud<=hi), axis=1)]
                ax = axes[row, col]
                ax.scatter(points[:,0], points[:,1], s=1, c=points[:,2], cmap='viridis', vmin=lo[2], vmax=hi[2])
                ax.set(xlim=(lo[0],hi[0]), ylim=(lo[1],hi[1]), title=title+' '+ '/'.join(group))
                ax.set_aspect('equal', adjustable='box')
        fig.tight_layout();fig.savefig(replay/'vehicle_detail.png',dpi=180);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='vehicle_roi_surface_proxy'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('replay',type=Path)
    parser.add_argument('--semantic-map',type=Path,help='Optional vehicle ROIs; short calibration flights do not require semantic detection')
    args=parser.parse_args();compare(args.source,args.replay,args.semantic_map)
