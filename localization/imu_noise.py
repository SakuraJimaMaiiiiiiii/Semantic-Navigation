"""Stationary, uniformly sampled IMU Allan diagnostics (Kalibr noise convention)."""
import numpy as np


def overlapping_allan(values, dt):
    values = np.asarray(values, float)
    if values.ndim != 2 or len(values) < 20 or not np.isfinite(values).all() or dt <= 0:
        raise ValueError('Allan analysis needs finite NxD samples and positive dt')
    sizes = np.unique(np.geomspace(1, max(1, len(values)//40), 40).astype(int))
    integral = np.vstack((np.zeros(values.shape[1]), np.cumsum(values, axis=0)))
    deviations = []
    for m in sizes:
        delta = (integral[2*m:]-2*integral[m:-m]+integral[:-2*m])/m
        deviations.append(np.sqrt(.5*np.mean(delta*delta, axis=0)))
    return sizes*dt, np.asarray(deviations)


def noise_fit(tau, deviation, slope):
    """Require a resolved slope over >=4x tau; no extrapolation from one point."""
    candidates = []
    for start in range(len(tau)):
        for end in range(start+3, len(tau)):
            if tau[end]/tau[start] < 4.:
                continue
            if slope < 0 and tau[end] > 2.:
                continue
            if slope > 0 and tau[start] < 1.:
                continue
            y = deviation[start:end+1]
            if np.any(y <= 0):
                continue
            x = np.log(tau[start:end+1])
            fitted_slope, intercept = np.polyfit(x, np.log(y), 1)
            residual = float(np.sqrt(np.mean((np.log(y)-fitted_slope*x-intercept)**2)))
            if abs(fitted_slope-slope) > .12 or residual > .1:
                continue
            coefficient = np.median(y / tau[start:end+1]**slope)
            if slope > 0:
                coefficient *= np.sqrt(3.)  # sigma_b * sqrt(tau/3)
            candidates.append((abs(fitted_slope-slope)+residual, {
                'value': float(coefficient), 'slope': float(fitted_slope),
                'tau_range_s': [float(tau[start]), float(tau[end])],
                'log_fit_rmse': residual}))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def analyze_stationary(samples, confirmed_stationary=False):
    """Input is source timestamp, ax/ay/az, gx/gy/gz. Never rewrites config."""
    samples = np.asarray(samples, float)
    if samples.ndim != 2 or samples.shape[1] != 7 or len(samples) < 20 or not np.isfinite(samples).all():
        raise ValueError('Expected at least 20 finite timestamp/accel/gyro rows')
    dt = np.diff(samples[:, 0])
    if np.any(dt <= 0):
        raise ValueError('IMU source clock is not strictly increasing')
    median = float(np.median(dt))
    jitter = float(np.percentile(abs(dt-median)/median, 95))
    result = {'samples': len(samples), 'duration_s': float(np.ptp(samples[:, 0])),
              'effective_rate_hz': float((len(samples)-1)/np.ptp(samples[:, 0])),
              'interval_median_s': median, 'interval_p95_s': float(np.percentile(dt, 95)),
              'max_gap_s': float(dt.max()), 'relative_interval_jitter_p95': jitter,
              'mean_accel_mps2': samples[:, 1:4].mean(axis=0).tolist(),
              'mean_gyro_rad_s': samples[:, 4:].mean(axis=0).tolist(),
              'apply_calibration': False, 'reasons': [], 'candidate_parameters': None}
    if not confirmed_stationary: result['reasons'].append('stationarity_not_confirmed')
    if result['duration_s'] < 600.: result['reasons'].append('stationary_duration_under_600s')
    if jitter > .2 or dt.max() > 2.*median:
        result['reasons'].append('irregular_sampling_not_supported_by_standard_allan')
    if np.max(np.std(samples[:, 1:4], axis=0)) > .1 or np.max(np.linalg.norm(samples[:, 4:], axis=1)) > .03:
        result['reasons'].append('motion_or_large_noise_requires_review')
    if result['reasons']:
        result['status'] = 'insufficient_calibration_data'
        return result
    tau, adev = overlapping_allan(samples[:, 1:], median)
    result['allan_tau_s'], result['allan_deviation'] = tau.tolist(), adev.tolist()
    white = [noise_fit(tau, adev[:, i], -.5) for i in range(6)]
    walk = [noise_fit(tau, adev[:, i], .5) for i in range(6)]
    result['white_noise_by_axis'], result['bias_random_walk_by_axis'] = white, walk
    if any(fit is None for fit in white+walk):
        result['reasons'].append('not_all_six_axes_resolve_both_noise_slopes')
        result['status'] = 'partial_noise_model'
    else:
        result['status'] = 'candidate_requires_vio_replay'
        result['candidate_parameters'] = {
            'accelerometer_noise': max(fit['value'] for fit in white[:3]),
            'gyroscope_noise': max(fit['value'] for fit in white[3:]),
            'accelerometer_random_walk': max(fit['value'] for fit in walk[:3]),
            'gyroscope_random_walk': max(fit['value'] for fit in walk[3:])}
    return result
