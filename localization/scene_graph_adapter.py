"""Additive LIO object/observation graph; no planning or association decisions."""
from copy import deepcopy


class SceneGraphAdapter:
    FRAME = 'lio_aligned_to_initial_px4_ned'

    def __init__(self):
        self.observations = {}
        self.revision = 0

    def observe(self, timestamp, samples, measurements, identities, transform, category):
        if not (len(samples) == len(measurements) == len(identities)):
            raise ValueError('Semantic observation/identity lengths differ')
        for index, (sample, measurement, identity) in enumerate(zip(samples, measurements, identities)):
            key = f'observation_{float(timestamp).hex()}_{index}'
            if key in self.observations:
                continue
            source = sample.get('source_instance_id')
            self.observations[key] = {
                'id': key, 'type': 'observation',
                'attributes': {
                    'timestamp': float(timestamp), 'class_name': sample['class_name'],
                    'confidence': float(sample['confidence']),
                    'position_world': measurement.position_world.tolist(),
                    'covariance': measurement.measurement_covariance.tolist(),
                    'surface_point_count': (0 if measurement.surface_points_world is None
                                            else len(measurement.surface_points_world)),
                    'assigned_instance_id': (None if identity is None else
                                             f"{category(sample['class_name'])}_{int(identity):02d}"),
                    'source_instance_id': (None if source is None else
                                           f"{category(sample['class_name'])}_{int(source):02d}"),
                    'source_identity_scope': 'px4_runtime_not_lio',
                    'evidence_ref': {
                        'path': '../maps/semantic_observations.jsonl',
                        'timestamp': float(timestamp), 'sample_index': index,
                    },
                    'px4_to_lio_map': transform.tolist(),
                }}
        self.revision += 1

    def snapshot(self, records, map_id):
        ids = {r['instance_id'] for r in records}
        if len(ids) != len(records):
            raise ValueError('Duplicate LIO object ID')
        redirects = {}
        for record in records:
            for alias in record.get('merged_from', []):
                if alias in redirects and redirects[alias] != record['instance_id']:
                    raise ValueError('Ambiguous instance redirect')
                redirects[alias] = record['instance_id']

        def canonical(identity):
            seen = set()
            while identity in redirects and redirects[identity] != identity:
                if identity in seen:
                    raise ValueError('Cyclic instance redirects')
                seen.add(identity)
                identity = redirects[identity]
            return identity

        nodes = [dict(id=r['instance_id'], type='object',
                      attributes=deepcopy({k:v for k,v in r.items() if k != 'instance_id'}))
                 for r in records]
        edges = []
        for observation in self.observations.values():
            node = deepcopy(observation)
            attrs = node['attributes']
            original = attrs['assigned_instance_id']
            target = canonical(original)
            if target in ids:
                attrs['association_status'] = 'associated'
                attrs['canonical_instance_id'] = target
                edges.append(dict(id=f"{node['id']}:measurement_of:{target}",
                                  source=node['id'], target=target, relation='measurement_of'))
            else:
                attrs['association_status'] = 'unassigned' if original is None else 'not_exported'
                attrs['canonical_instance_id'] = None
            nodes.append(node)
        return dict(schema_version=1, map_id=map_id, graph_revision=self.revision,
                    representation='lio_object_observation_graph', frame=self.FRAME,
                    identity_scope='independent_lio_instances', controls_navigation=False,
                    directed=True, nodes=nodes, edges=edges,
                    layers=['object', 'observation'],
                    timestamp=max((o['attributes']['timestamp'] for o in self.observations.values()), default=None))
