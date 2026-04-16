import numpy as np
from typing import Callable, Tuple

def se3_log(T: np.ndarray) -> np.ndarray:
    """
    Compute the logarithm map from SE(3) to se(3).
    
    Args:
        T: 4x4 transformation matrix
        
    Returns:
        6-element array representing twist coordinates
    """
    R = T[:3, :3]
    t = T[:3, 3]
    
    # Rotation part - axis-angle representation
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    
    if np.isclose(cos_theta, 1):
        # No rotation case
        omega = np.zeros(3)
        v = t
    elif np.isclose(cos_theta, -1):
        # 180-degree rotation case
        omega = np.pi * np.array([R[0,0]+1, R[1,1]+1, R[2,2]+1])
        omega = np.sqrt(omega/2)
        # Handle sign ambiguity
        if omega[0] < 0: omega = -omega
        v = np.linalg.inv(np.eye(3) - R) @ t
    else:
        # General case
        theta = np.arccos(cos_theta)
        sin_theta = np.sin(theta)
        
        # Skew-symmetric matrix to vector conversion
        skew_omega = (R - R.T) / (2 * sin_theta) * theta
        omega = np.array([skew_omega[2,1], skew_omega[0,2], skew_omega[1,0]])
        
        # Translation part
        A = np.eye(3) - R
        B = np.outer(omega, omega) * (1 - cos_theta) / theta**2
        C = np.cross(np.eye(3), omega) * sin_theta / theta**2
        v = np.linalg.inv(A + B + C) @ t * theta
    
    return np.concatenate([omega, v])

def closure_residual(q: np.ndarray, 
                    link_a: int, 
                    link_b: int, 
                    T_ab_inv: np.ndarray, 
                    FK: Callable) -> np.ndarray:
    """
    Compute closure constraint residual for two links.
    
    Args:
        q: Joint configuration vector
        link_a, link_b: Link indices
        T_ab_inv: Inverse of desired relative transform
        FK: Forward kinematics function
        
    Returns:
        6D residual vector in se(3)
    """
    T_a = FK(q, link_a)            # 4x4 transform of body a
    T_b = FK(q, link_b)            # 4x4 transform of body b
    X = np.linalg.inv(T_a) @ T_b @ T_ab_inv
    xi = se3_log(X)                # 6-vector logarithm map
    return xi

def closure_jacobian_fd(q: np.ndarray, 
                       link_a: int, 
                       link_b: int, 
                       T_ab_inv: np.ndarray, 
                       FK: Callable, 
                       eps: float = 1e-6) -> np.ndarray:
    """
    Compute Jacobian of closure constraint using finite differences.
    
    Args:
        q: Joint configuration vector
        link_a, link_b: Link indices
        T_ab_inv: Inverse of desired relative transform
        FK: Forward kinematics function
        eps: Finite difference step size
        
    Returns:
        6xn Jacobian matrix
    """
    n = q.size
    r0 = closure_residual(q, link_a, link_b, T_ab_inv, FK)
    J = np.zeros((6, n))
    
    for i in range(n):
        dq = np.zeros_like(q)
        dq[i] = eps
        r = closure_residual(q + dq, link_a, link_b, T_ab_inv, FK)
        J[:, i] = (r - r0) / eps    # finite-difference column
        
    return J