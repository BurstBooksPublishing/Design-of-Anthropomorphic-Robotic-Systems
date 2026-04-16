import numpy as np
from typing import List, Tuple

def hat_omega(omega: np.ndarray) -> np.ndarray:
    """Convert 3D vector to 3x3 skew-symmetric matrix."""
    return np.array([[0, -omega[2], omega[1]],
                     [omega[2], 0, -omega[0]],
                     [-omega[1], omega[0], 0]])

def hat_xi(xi: np.ndarray) -> np.ndarray:
    """Convert 6D twist to 4x4 se(3) matrix."""
    omega, v = xi[:3], xi[3:]
    M = np.zeros((4, 4))
    M[:3, :3] = hat_omega(omega)
    M[:3, 3] = v
    return M

def exp_se3(xi: np.ndarray, theta: float) -> np.ndarray:
    """Compute SE(3) transformation from twist and angle."""
    omega, v = xi[:3], xi[3:]
    wnorm = np.linalg.norm(omega)
    
    if wnorm < 1e-12:  # Pure translation
        R = np.eye(3)
        p = v * theta
    else:
        w = omega / wnorm
        w_hat = hat_omega(w)
        theta_w = wnorm * theta
        
        # Rodrigues' rotation formula
        R = (np.eye(3) + np.sin(theta_w) * w_hat + 
             (1 - np.cos(theta_w)) * (w_hat @ w_hat))
        
        # Compute linear component
        A = (np.eye(3) * theta + 
             ((1 - np.cos(theta_w)) / wnorm) * w_hat +
             ((theta_w - np.sin(theta_w)) / wnorm) * (w_hat @ w_hat))
        p = A @ v
    
    g = np.eye(4)
    g[:3, :3] = R
    g[:3, 3] = p
    return g

def adjoint(g: np.ndarray) -> np.ndarray:
    """Compute 6x6 adjoint matrix from 4x4 transformation."""
    R = g[:3, :3]
    p = g[:3, 3]
    p_hat = hat_omega(p)
    
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = p_hat @ R
    return Ad

def forward_kinematics(twists: List[np.ndarray], 
                      thetas: List[float], 
                      g0: np.ndarray) -> np.ndarray:
    """Compute end-effector pose from joint twists and angles."""
    g = np.eye(4)
    for xi, th in zip(twists, thetas):
        g = g @ exp_se3(xi, th)
    return g @ g0

def spatial_jacobian(twists: List[np.ndarray], 
                    thetas: List[float]) -> np.ndarray:
    """Compute spatial manipulator Jacobian."""
    n = len(twists)
    J = np.zeros((6, n))
    g = np.eye(4)
    
    for i, (xi, th) in enumerate(zip(twists, thetas)):
        J[:, i] = adjoint(g) @ xi
        g = g @ exp_se3(xi, th)
    
    return J