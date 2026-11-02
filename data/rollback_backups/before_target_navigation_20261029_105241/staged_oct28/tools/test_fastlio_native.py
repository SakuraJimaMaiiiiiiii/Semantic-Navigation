"""Run after compilation in the same Python environment; no UE or flight control."""
from pathlib import Path
import sys
import unittest
import tempfile
import json
import h5py
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from localization import _native_fastlio


def cloud():
    a=np.linspace(-3,3,25);b=np.linspace(-1,2,16)
    return np.asarray([[5.,y,z,0.] for y in a for z in b]+
                      [[x,5.,z,0.] for x in a for z in b]+
                      [[x,y,2.,0.] for x in a for y in a])


def imu(start,end,accel=0.):
    t=np.arange(start,end+.0001,.01)
    return np.column_stack([t,np.full(len(t),accel),np.zeros(len(t)),
                            np.full(len(t),-9.81),np.zeros((len(t),3))])


class NativeTests(unittest.TestCase):
    def test_recording_format_replays_without_ros(self):
        from tools.replay_fastlio import replay
        from tools.audit_fastlio_inputs import audit
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'inputs.h5'
            samples=imu(.01,2.)
            with h5py.File(source,'w') as f:
                f.attrs['scan_model']='snapshot'
                f.attrs['clock_alignment']='synthetic exact shared clock'
                f.create_dataset('imu',data=np.column_stack((samples,np.zeros((len(samples),3)))))
                scans=f.create_group('scans')
                for i in range(1,21):
                    group=scans.create_group(f'{i:08d}')
                    group.attrs['source_timestamp']=round(i*.1,4)
                    group.create_dataset('points_body',data=cloud()[:,:3])
                    group.create_dataset('px4_pose',data=[0.,0.,0.,1.,0.,0.,0.])
            self.assertTrue(audit(source)['static_start_verified'])
            replay(source,Path(tmp)/'output',None)
            report=json.loads((Path(tmp)/'output/report.json').read_text())
            self.assertGreater(report['status_counts'].get('tracking',0),0)
            self.assertTrue((Path(tmp)/'output/map.npz').is_file())

    def test_static_and_translation(self):
        engine=_native_fastlio.Estimator(instantaneous=True)
        points=cloud()
        self.assertEqual(engine.process(1.,points,imu(.01,1.))['status'],'initializing')
        last=1.
        result=None
        for i in range(1,31):
            now=round(1.+i*.1,4)
            elapsed=max(0.,now-2.)
            x=.5*.1*elapsed**2
            observed=points.copy();observed[:,0]-=x
            samples=imu(last+.01,now, .1 if now>2 else 0.)
            result=engine.process(now,observed,samples);last=now
            self.assertIn(result['status'],['map_initialized','tracking'])
            self.assertTrue(np.isfinite(result['pose']).all())
            self.assertLess(np.linalg.norm(result['pose'][:3,3]-[x,0.,0.]),.12)
        self.assertGreater(result['effective_points'],100)
        self.assertLess(result['mean_residual_m'],.05)
        with self.assertRaises(ValueError):
            engine.process(last,points,imu(last+.01,last+.1))

    def test_point_time_contract(self):
        engine=_native_fastlio.Estimator(instantaneous=True)
        points=cloud();points[:,3]=.1
        with self.assertRaisesRegex(ValueError,'point times'):
            engine.process(1.,points,imu(.01,1.))


if __name__=='__main__':
    print('Native FAST-LIO:',_native_fastlio.ALGORITHM_VERSION,_native_fastlio.UPSTREAM_COMMIT)
    unittest.main()
