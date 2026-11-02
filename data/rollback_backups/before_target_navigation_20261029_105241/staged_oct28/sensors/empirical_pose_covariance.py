"""PX4 缺失协方差的 Python 层补充；绝不把运动方差当成估计误差。"""

import json
from pathlib import Path

import numpy as np

from .pose_uncertainty import covariance_matrix


CONVENTION = "position_ned_m__right_rotation_body_rad"


def pose_error(position, quaternion, truth_position, truth_quaternion):
    """真值减估计；Log(q_est^-1 * q_true) 为估计机体系右扰动。"""
    def unit(value):
        q = np.asarray(value, dtype=float)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
            raise ValueError("Invalid wxyz quaternion")
        return q / np.linalg.norm(q)

    q, t = unit(quaternion), unit(truth_quaternion)
    w = q @ t
    v = q[0] * t[1:] - t[0] * q[1:] - np.cross(q[1:], t[1:])
    if w < 0:
        w, v = -w, -v
    length = np.linalg.norm(v)
    angle = 2 * np.arctan2(length, w)
    rotation = v * (angle / length if length > 1e-10 else 2.)
    delta = np.asarray(truth_position, dtype=float) - np.asarray(position, dtype=float)
    error = np.concatenate((delta, rotation))
    if error.shape != (6,) or not np.isfinite(error).all():
        raise ValueError("Invalid pose error")
    return error


def estimate_profile(errors):
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 2 or errors.shape[1] != 6 or len(errors) < 100:
        raise ValueError("At least 100 valid paired pose errors are required")
    if not np.isfinite(errors).all():
        raise ValueError("Pose errors must be finite")
    bias = errors.mean(axis=0)
    covariance = np.cov(errors, rowvar=False, ddof=1)
    # 不校正状态偏置时，使用带偏置二阶矩，避免去均值后过度自信。
    effective = covariance + np.outer(bias, bias)
    return {"schema_version": 1, "convention": CONVENTION,
            "sample_count": len(errors), "bias": bias.tolist(),
            "covariance": covariance.tolist(),
            "effective_covariance": effective.tolist()}


def load_profile(path):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1 or profile.get("convention") != CONVENTION:
        raise ValueError("Unsupported pose covariance profile convention")
    if profile.get("sample_count", 0) < 100:
        raise ValueError("Insufficient calibration samples")
    return covariance_matrix(profile["effective_covariance"], 6)


def position_covariance_from_odometry(message, now, max_age=0.5):
    """只取当前旧版固件真正实现的 NED 位置对角线；不解释零姿态块。"""
    if message is None or getattr(message, "frame_id", None) != 1:
        return None
    stamp = float(getattr(message, "_timestamp", 0))
    if not np.isfinite(stamp) or not 0 <= now - stamp <= max_age:
        return None
    if getattr(message, "estimator_type", None) != 8:
        return None
    values = np.asarray(getattr(message, "pose_covariance", []), dtype=float)
    if values.shape != (21,):
        return None
    diagonal = values[[0, 6, 11]]
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        return None
    return np.diag(diagonal)


def combine_pose_covariance(base, position_covariance, calibrated=False):
    result = np.array(base, dtype=float, copy=True)
    if position_covariance is None:
        return result
    position = covariance_matrix(position_covariance, 3)
    if calibrated:
        # 只增加不足的位置方差，保留实测跨项并保证半正定；不是独立噪声相加。
        extra = np.maximum(np.diag(position) - np.diag(result)[:3], 0)
        result[:3, :3] += np.diag(extra)
    else:
        result[:3, :3] = position
    return result
