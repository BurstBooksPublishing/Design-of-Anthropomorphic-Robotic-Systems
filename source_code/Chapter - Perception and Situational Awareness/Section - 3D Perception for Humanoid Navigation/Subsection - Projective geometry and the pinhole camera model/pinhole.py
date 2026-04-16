import numpy as np
from typing import Tuple, Union

ArrayLike = Union[np.ndarray, list, tuple]

def project(K: np.ndarray, R: np.ndarray, t: np.ndarray, Xw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D world point to 2D pixel coordinates.
    
    Args:
        K: Camera intrinsic matrix (3x3)
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
        Xw: World point (3x1)
        
    Returns:
        Tuple of (pixel coordinates, camera coordinates)
    """
    Xc = R @ Xw + t                      # Transform to camera coordinates
    x = Xc[:2] / Xc[2]                   # Perspective division
    u = K[:2,:2] @ x + K[:2,2]           # Apply intrinsic parameters
    return u, Xc

def backproject_ray(K: np.ndarray, R: np.ndarray, t: np.ndarray, uv_h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backproject pixel ray to 3D world coordinates.
    
    Args:
        K: Camera intrinsic matrix (3x3)
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
        uv_h: Homogeneous pixel coordinates (3x1)
        
    Returns:
        Tuple of (camera center, ray direction in world frame)
    """
    Kinv = np.linalg.inv(K)
    dir_cam = Kinv @ uv_h                # Ray direction in camera frame
    dir_world = R.T @ dir_cam            # Transform to world frame
    C = -R.T @ t                         # Camera center in world frame
    return C, dir_world / np.linalg.norm(dir_world)  # Normalize direction

def jacobian_pixel_wrt_world(K: np.ndarray, R: np.ndarray, t: np.ndarray, Xw: np.ndarray) -> np.ndarray:
    """
    Compute Jacobian of pixel coordinates with respect to world coordinates.
    
    Args:
        K: Camera intrinsic matrix (3x3)
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
        Xw: World point (3x1)
        
    Returns:
        Jacobian matrix (2x3)
    """
    _, Xc = project(K, R, t, Xw)
    Xc1, Xc2, Xc3 = Xc.ravel()
    
    # Compute camera Jacobian
    J_cam = (1.0 / Xc3) * np.array([
        [K[0,0], 0, -K[0,0]*Xc1/Xc3],
        [0, K[1,1], -K[1,1]*Xc2/Xc3]
    ])
    
    J = J_cam @ R                         # Chain rule to world coordinates
    return J