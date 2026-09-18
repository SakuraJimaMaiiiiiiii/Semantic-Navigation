"""位姿误差传播：平移在NED，旋转为局部右乘小角度（rad）。"""

import numpy as np


def skew(vector):
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def covariance_matrix(value, size):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"covariance must be finite with shape ({size}, {size})")
    if not np.allclose(matrix, matrix.T, atol=1e-9):
        raise ValueError("covariance must be symmetric")
    matrix = (matrix + matrix.T) * 0.5
    if np.linalg.eigvalsh(matrix).min() < -1e-9:
        raise ValueError("covariance must be positive semidefinite")
    return matrix


def camera_pose_covariance(rotation_world_body, transform_body_camera,
                           body_covariance, velocity_world, angular_velocity_body,
                           timestamp_std=0.0):
    """机体6x6协方差转到相机；保留平移/旋转交叉项及时间误差相关性。

    body_covariance顺序为[delta_t_NED, delta_theta_body]；返回顺序为
    [delta_t_camera_origin_NED, delta_theta_camera]。外参视为已标定常量。
    """
    covariance = covariance_matrix(body_covariance, 6)
    if not np.isfinite(timestamp_std) or timestamp_std < 0:
        raise ValueError("timestamp_std must be finite and non-negative")
    rate = np.concatenate((velocity_world, angular_velocity_body)).astype(float)
    if rate.shape != (6,) or not np.all(np.isfinite(rate)):
        raise ValueError("velocity and angular velocity must be finite 3-vectors")
    covariance = covariance + timestamp_std**2 * np.outer(rate, rate)
    rotation = np.asarray(rotation_world_body, dtype=float)
    extrinsics = np.asarray(transform_body_camera, dtype=float)
    jacobian = np.zeros((6, 6))
    jacobian[:3, :3] = np.eye(3)
    jacobian[:3, 3:] = -rotation @ skew(extrinsics[:3, 3])
    jacobian[3:, 3:] = extrinsics[:3, :3].T
    result = jacobian @ covariance @ jacobian.T
    return (result + result.T) * 0.5


def world_point_pose_jacobian(rotation_world_camera, point_camera):
    return np.hstack((np.eye(3), -rotation_world_camera @ skew(point_camera)))


def world_point_covariance(rotation_world_camera, point_camera,
                           point_covariance_camera, pose_covariance=None):
    rotation = np.asarray(rotation_world_camera, dtype=float)
    result = rotation @ point_covariance_camera @ rotation.T
    if pose_covariance is not None:
        jacobian = world_point_pose_jacobian(rotation, point_camera)
        result += jacobian @ covariance_matrix(pose_covariance, 6) @ jacobian.T
    return (result + result.T) * 0.5
