import numpy as np
from typing import List, Tuple

def hat_se3(xi: np.ndarray) -> np.ndarray:
    """Convert 6D twist vector to 4x4 se(3) matrix."""
    omega, v = xi[:3], xi[3:]
    W = np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])
    return np.block([[W, v.reshape(3, 1)], [np.zeros((1, 3)), 0]])

def exp_se3(xi: np.ndarray) -> np.ndarray:
    """Exponential map from se(3) to SE(3) using Rodrigues' formula."""
    omega = xi[:3]
    theta = np.linalg.norm(omega)
    
    if theta < 1e-8:
        # First-order approximation for small angles
        W = hat_se3(np.hstack([omega, np.zeros(3)]))[:3, :3]
        R = np.eye(3) + W
        V = np.eye(3) + 0.5 * W
    else:
        # Full Rodrigues expansion
        W = hat_se3(np.hstack([omega / theta, np.zeros(3)]))[:3, :3]
        W2 = W @ W
        R = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * W2
        V = np.eye(3) + (1 - np.cos(theta)) / theta * W + (theta - np.sin(theta)) / theta * W2

    t = V @ xi[3:]
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = t
    return X

def propagate(Xhat: np.ndarray, xi_meas: np.ndarray, dt: float) -> np.ndarray:
    """Propagate state using measured twist for dt time."""
    return Xhat @ exp_se3(xi_meas * dt)

def innovation(Xhat: np.ndarray, y: np.ndarray, g_bar: np.ndarray) -> np.ndarray:
    """Compute right-invariant innovation."""
    Xinv = np.linalg.inv(Xhat)
    pred = Xinv[:3, :3] @ g_bar + Xinv[:3, 3]
    return pred - y

def update(Xhat: np.ndarray, y_list: List[np.ndarray], 
           gbar_list: List[np.ndarray], K: np.ndarray) -> np.ndarray:
    """Update state estimate using measurements and gain matrix."""
    if len(y_list) != len(gbar_list):
        raise ValueError("Measurement lists must have equal length")
        
    xi_corr = np.zeros(6)
    for y, g in zip(y_list, gbar_list):
        e = innovation(Xhat, y, g)
        # Rotational innovation via cross product in body frame
        xi_corr[:3] += np.cross(Xhat[:3, :3].T @ y, g)
        xi_corr[3:] += e  # Translational innovation
        
    return exp_se3(K @ xi_corr) @ Xhat