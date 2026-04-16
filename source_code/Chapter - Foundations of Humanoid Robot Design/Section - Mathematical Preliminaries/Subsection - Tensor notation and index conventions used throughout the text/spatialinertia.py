import numpy as np
from typing import Union, Tuple

ArrayLike = Union[np.ndarray, list, tuple]

def hat(v: ArrayLike) -> np.ndarray:
    """Convert 3-vector to 3x3 skew-symmetric matrix."""
    v = np.asarray(v)
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def adjoint(R: ArrayLike, p: ArrayLike) -> np.ndarray:
    """Compute 6x6 adjoint matrix for transformation T = [R p; 0 1]."""
    R = np.asarray(R)
    p = np.asarray(p)
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = hat(p) @ R
    return Ad

def transform_inertia(M_A: ArrayLike, R: ArrayLike, p: ArrayLike) -> np.ndarray:
    """Transform inertia matrix using adjoint representation."""
    M_A = np.asarray(M_A)
    R = np.asarray(R)
    p = np.asarray(p)
    
    Ad = adjoint(R, p)
    Ad_inv = np.linalg.inv(Ad)
    return Ad_inv.T @ M_A @ Ad_inv