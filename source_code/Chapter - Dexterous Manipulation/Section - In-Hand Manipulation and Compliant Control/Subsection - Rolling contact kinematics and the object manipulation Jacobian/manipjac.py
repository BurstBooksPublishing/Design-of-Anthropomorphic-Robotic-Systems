]

import numpy as np
from typing import List, Union

ArrayLike = Union[np.ndarray, List[np.ndarray]]

def hat(v: np.ndarray) -> np.ndarray:
    """so(3) hat operator: maps R^3 to so(3) skew-symmetric matrix"""
    v = np.asarray(v).flatten()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def adjoint(T: np.ndarray) -> np.ndarray:
    """Compute Adjoint matrix Ad_T (6x6) from transformation matrix T (4x4)"""
    T = np.asarray(T)
    R = T[:3, :3]
    p = T[:3, 3]
    
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = hat(p) @ R
    Ad[3:, 3:] = R
    return Ad

def manipulation_jacobian(T_o_ci_list: ArrayLike, Jf_list: ArrayLike) -> np.ndarray:
    """Compute manipulation Jacobian from object transforms and finger Jacobians"""
    # Convert to arrays and validate input
    T_list = [np.asarray(T) for T in T_o_ci_list]
    J_list = [np.asarray(J) for J in Jf_list]
    
    if len(T_list) != len(J_list):
        raise ValueError("T_o_ci_list and Jf_list must have same length")
    
    # Stack adjoint matrices and Jacobians
    A_blocks = [adjoint(T) for T in T_list]  # List of 6x6 matrices
    A_o = np.vstack(A_blocks)                # (6n_c x 6)
    J_c = np.vstack(J_list)                  # (6n_c x n)
    
    # Compute left pseudoinverse using SVD for numerical stability
    U, s, Vt = np.linalg.svd(A_o, full_matrices=False)
    tol = max(A_o.shape) * np.finfo(float).eps * s[0]  # Threshold for singular values
    s_inv = np.where(s > tol, 1/s, 0)
    A_pinv = (Vt.T * s_inv) @ U.T            # (6 x 6n_c)
    
    return A_pinv @ J_c                      # (6 x n)