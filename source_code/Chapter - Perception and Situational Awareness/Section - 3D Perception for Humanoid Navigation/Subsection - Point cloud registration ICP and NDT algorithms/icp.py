import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional

def icp_point2point(P: np.ndarray, Q: np.ndarray, max_iters: int = 50, tol: float = 1e-6) -> np.ndarray:
    """
    Perform point-to-point ICP alignment between two point clouds.
    
    Args:
        P: Source point cloud (N, 3)
        Q: Target point cloud (M, 3)
        max_iters: Maximum number of iterations
        tol: Convergence tolerance
        
    Returns:
        T: 4x4 transformation matrix mapping P to Q
    """
    # Validate inputs
    if P.ndim != 2 or Q.ndim != 2 or P.shape[1] != 3 or Q.shape[1] != 3:
        raise ValueError("Point clouds must be (N, 3) arrays")
    
    if P.shape[0] == 0 or Q.shape[0] == 0:
        return np.eye(4)
    
    T = np.eye(4)
    Q_tree = cKDTree(Q)
    prev_err = np.inf
    
    for it in range(max_iters):
        # Transform source points
        P_trans = (T[:3, :3] @ P.T).T + T[:3, 3]
        
        # Find nearest neighbors
        _, idx = Q_tree.query(P_trans)
        Q_corr = Q[idx]
        
        # Compute centroids
        p_mean = P_trans.mean(axis=0)
        q_mean = Q_corr.mean(axis=0)
        
        # Center point clouds
        Pc = (P_trans - p_mean).T
        Qc = (Q_corr - q_mean).T
        
        # Compute optimal rotation using SVD
        S = Pc @ Qc.T
        U, _, Vt = np.linalg.svd(S)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = q_mean - R @ p_mean
        
        # Update transformation
        T_new = np.eye(4)
        T_new[:3, :3] = R
        T_new[:3, 3] = t
        T = T_new @ T
        
        # Check convergence
        err = np.mean(np.sum((P_trans - Q_corr)**2, axis=1))
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return T