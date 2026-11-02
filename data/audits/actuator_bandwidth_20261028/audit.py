from pathlib import Path
import json
import h5py
import numpy as np
from scipy.signal import welch
run=Path('data/runs/run_20261028_140435_896686')
record=next((run/'recordings').glob('*.h5'))
with h5py.File(record,'r') as f:
    stamp=f['timestamps/timestamp'][:]
    euler=f['state/euler_rpy'][:]
    position=f['state/position_xyz'][:]
    velocity=f['state/linear_velocity'][:]
    angular=f['state/angular_velocity'][:]
t=stamp-stamp[0]
air=position[0,2]-position[:,2]>.7
pitch=np.rad2deg(euler[:,1]);speed=np.linalg.norm(velocity[:,:2],axis=1)
window=(t>=100)&(t<=140)
x=t[window]; sampling=np.arange(100.,140.,.2)
pitch_uniform=np.interp(sampling,x,pitch[window]);speed_uniform=np.interp(sampling,x,speed[window])
f,power=welch(pitch_uniform,fs=5.,nperseg=100,detrend='linear')
search=(f>=.1)&(f<=1.5);peak=float(f[search][np.argmax(power[search])])
summary=dict(run=run.name,samples=len(t),duration_seconds=float(t[-1]),
    telemetry_interval_seconds=dict(median=float(np.median(np.diff(stamp))),p95=float(np.percentile(np.diff(stamp),95)),max=float(np.max(np.diff(stamp)))),
    airborne_pitch_abs_degrees=dict(median=float(np.median(abs(pitch[air]))),p95=float(np.percentile(abs(pitch[air]),95)),max=float(np.max(abs(pitch[air])))),
    straight_segment=dict(start_seconds=100,end_seconds=140,samples=int(window.sum()),largest_record_gap_seconds=float(np.max(np.diff(x))),yaw_range_degrees=float(np.ptp(np.rad2deg(np.unwrap(euler[window,2])))),pitch_peak_to_peak_degrees=float(np.ptp(pitch[window])),horizontal_speed_p10_mps=float(np.percentile(speed[window],10)),horizontal_speed_p90_mps=float(np.percentile(speed[window],90)),pitch_oscillation_spectrum_peak_hz=peak),
    unavailable=['position setpoint timestamps/values throughout flight','attitude/rate setpoints','actuator outputs or motor RPM','saturation/control-allocation status'],
    conclusion='Output oscillation and slow outer-reference updates are observable; actuator closed-loop bandwidth is NOT identifiable from these signals alone.')
out=Path('data/audits/actuator_bandwidth_20261028')
(out/'summary.json').write_text(json.dumps(summary,indent=2))
np.savez_compressed(out/'flight_signals.npz',t=t,pitch_deg=pitch,speed=speed,pitch_rate_deg_s=np.rad2deg(angular[:,1]),air=air)
print(json.dumps(summary,ensure_ascii=False,indent=2))
