"""Validation wrapper around the actual native FAST-LIO filter (no PX4 fallback)."""
import numpy as np


class FastLioBackend:
    def __init__(self, *, instantaneous):
        from localization import _native_fastlio
        self.estimator = _native_fastlio.Estimator(instantaneous=instantaneous)
        self.initial_imu = []
        self.initialized = False

    def process(self, stamp, points, imu):
        imu = np.asarray(imu, dtype=float)
        if not self.initialized:
            self.initial_imu.extend(imu.tolist())
            samples = np.asarray(self.initial_imu)
            if samples[-1,0]-samples[0,0] < 1.:
                return {'status':'waiting_for_static_imu', 'timestamp':stamp}
            if (not np.isfinite(samples).all() or np.any(np.diff(samples[:,0])<=0)
                    or np.max(np.diff(samples[:,0]))>.1):
                raise ValueError('Initial IMU timing invalid')
            if (np.max(np.linalg.norm(samples[:,4:7],axis=1)) > .1
                    or np.max(np.std(samples[:,1:4],axis=0)) > .2
                    or abs(np.linalg.norm(np.mean(samples[:,1:4],axis=0))-9.81)>1.):
                raise ValueError('FAST-LIO initialization needs one second of stationary raw IMU')
            imu=samples
            self.initial_imu=[]
            self.initialized=True
        return self.estimator.process(float(stamp),np.asarray(points,dtype=float),imu)
