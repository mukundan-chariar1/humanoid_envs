import numpy as np
from typing import Tuple

def quaternion_to_matrix(quaternions: np.array) -> np.array:
    r, i, j, k = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_rotation_6d(matrix: np.array) -> np.array:
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))


def quaternion_to_rotation_6d(quaternion: np.array) -> np.array:
    return matrix_to_rotation_6d(quaternion_to_matrix(quaternion))

def axis_angle_to_quaternion(axis_angle: np.array) -> np.array:
    # Compute the angle (magnitude of the vector)
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    eps = 1e-8

    sin_half_angles_over_angles=np.where(np.abs(angles)<eps, 0.5-(angles**2)/48, np.sin(half_angles)/np.clip(angles, eps, None))

    # Compute the quaternion
    quaternions = np.concatenate(
        [np.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1
    )

    quaternions=quaternions / (np.linalg.norm(quaternions, axis=-1, keepdims=True) + eps)

    return quaternions

def axis_angle_to_matrix(axis_angle: np.array) -> np.array:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def axis_angle_to_rotation_6d(axis_angle: np.array):
    return quaternion_to_rotation_6d(axis_angle_to_quaternion(axis_angle))

def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to axis-angle representation.

    Args:
        quaternion: A quaternion array of shape (..., 4).

    Returns:
        axis_angle: An axis-angle array of shape (..., 3).
    """
    # Normalize the quaternion to ensure it's a unit quaternion
    quaternion = quaternion / (np.linalg.norm(quaternion, axis=-1, keepdims=True)+1e-8)

    # Extract the real part (scalar) and imaginary part (vector)
    w = quaternion[..., 0]  # Scalar part
    xyz = quaternion[..., 1:]  # Vector part

    # Compute the angle
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))  # Clip to avoid numerical instability

    # Compute the axis
    norm_xyz = np.linalg.norm(xyz, axis=-1, keepdims=True)
    axis = np.where(norm_xyz > 1e-8, xyz / norm_xyz, np.array([1.0, 0.0, 0.0]))

    # Combine axis and angle into axis-angle representation
    axis_angle = axis * angle[..., None]

    return axis_angle