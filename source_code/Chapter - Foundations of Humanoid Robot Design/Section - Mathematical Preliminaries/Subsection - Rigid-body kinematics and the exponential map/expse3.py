import numpy as np
from math import sin, cos
from typing import Union

ArrayLike = Union[np.ndarray, list, tuple]

def hat(omega: ArrayLike) -> np.ndarray:
    """Convert 3D vector to skew-symmetric matrix."""
    omega = np.asarray(omega)
    return np.array([[0, -omega[2], omega[1]],
                     [omega[2], 0, -omega[0]],
                     [-omega[1], omega[0], 0]])

def exp_se3(xi: ArrayLike) -> np.ndarray:
    """Exponential map from se(3) to SE(3): converts 6D twist to 4x4 transformation matrix."""
    xi = np.asarray(xi)
    if xi.shape != (6,):
        raise ValueError("Input must be a 6D vector")
    
    v = xi[:3]
    omega = xi[3:]
    theta = np.linalg.norm(omega)
    
    if theta < 1e-8:
        # Series expansion for numerical stability when theta is small
        omega_hat = hat(omega)
        omega_hat_sq = omega_hat @ omega_hat
        R = np.eye(3) + omega_hat + 0.5 * omega_hat_sq
        J = np.eye(3) + 0.5 * omega_hat + (1.0/6.0) * omega_hat_sq
    else:
        u = omega / theta
        u_hat = hat(u)
        u_hat_sq = u_hat @ u_hat
        R = np.eye(3) + sin(theta) * u_hat + (1 - cos(theta)) * u_hat_sq
        J = (np.eye(3) * theta + (1 - cos(theta)) * u_hat + (theta - sin(theta)) * u_hat_sq) / theta
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = J @ v
    return T