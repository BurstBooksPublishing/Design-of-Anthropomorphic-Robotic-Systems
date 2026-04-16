import numpy as np
from typing import List, Tuple, Union

def dyn_consistent_pinv(J: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Compute dynamically consistent pseudoinverse J^# = M^-1 * J^T * (J * M^-1 * J^T)^-1
    """
    # Regularized inverse of M to avoid numerical instability
    M_inv = np.linalg.inv(M + 1e-12 * np.eye(M.shape[0]))
    B = J @ M_inv @ J.T
    
    # Check if B is invertible, use pseudo-inverse if not
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        B_inv = np.linalg.pinv(B, rcond=1e-12)
    
    return M_inv @ J.T @ B_inv

def hierarchical_accel(M: np.ndarray, 
                      qdot: np.ndarray, 
                      tasks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Compute hierarchical acceleration using null-space projection method.
    
    Args:
        M: Mass matrix (nxn)
        qdot: Joint velocity vector (n)
        tasks: List of (Jacobian, desired acceleration, Jdot*qdot) tuples
    
    Returns:
        Joint acceleration vector (n)
    """
    n = M.shape[0]
    qdd = np.zeros(n)
    N = np.eye(n)
    
    for J, xdd_star, Jdot_qdot in tasks:
        # Project Jacobian into current null-space
        J_projected = J @ N
        
        # Check if projected Jacobian has sufficient rank
        if np.linalg.matrix_rank(J_projected) < min(J_projected.shape):
            continue
            
        Jbar = dyn_consistent_pinv(J_projected, M)
        
        # Compute residual in reduced space
        res = xdd_star - Jdot_qdot - J @ qdd
        
        # Compute null-space-respecting increment
        delta = N @ (Jbar @ res)
        qdd = qdd + delta
        
        # Update null-space projector
        N = N - Jbar @ J_projected
        
        # Symmetrize and re-orthogonalize null-space projector for numerical stability
        N = (N + N.T) / 2
        N = N - N @ N.T + np.eye(n)  # Re-normalize
    
    return qdd