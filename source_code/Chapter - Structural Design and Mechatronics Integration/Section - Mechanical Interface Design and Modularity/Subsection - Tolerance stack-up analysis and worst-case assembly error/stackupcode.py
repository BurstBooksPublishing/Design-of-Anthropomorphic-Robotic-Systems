import numpy as np
from typing import List, Tuple, Optional

def adjoint(T: np.ndarray) -> np.ndarray:
    """Compute the adjoint matrix of an SE(3) transformation matrix."""
    R = T[:3, :3]
    p = T[:3, 3]
    
    # Skew-symmetric matrix of translation vector
    p_hat = np.array([[0, -p[2], p[1]],
                      [p[2], 0, -p[0]],
                      [-p[1], p[0], 0]])
    
    J = np.zeros((6, 6))
    J[:3, :3] = R
    J[3:6, 3:6] = R
    J[3:6, :3] = p_hat @ R
    return J

def worst_case_stackup(G_list: List[np.ndarray], 
                      delta_list: List[np.ndarray], 
                      W: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """
    Compute worst-case stackup for a sequence of SE(3) transformations.
    
    Args:
        G_list: List of N SE(3) transformation matrices (4x4)
        delta_list: List of N delta vectors (6x1) representing uncertainties
        W: Weight matrix (6x6) for weighted norm computation
        
    Returns:
        Tuple of (absolute sum vector, worst-case norm)
    """
    if W is None:
        W = np.eye(6)
        
    N = len(G_list)
    if N != len(delta_list):
        raise ValueError("G_list and delta_list must have the same length")
    
    # Precompute suffix products G_{i+1:N}
    suffix = [np.eye(4) for _ in range(N + 1)]
    for i in range(N - 1, -1, -1):
        suffix[i] = G_list[i] @ suffix[i + 1]
    
    # Compute Jacobians J_i = Ad_{G_{i+1:N}}
    J_list = [adjoint(suffix[i + 1]) for i in range(N)]
    
    # Elementwise absolute accumulation
    abs_sum = np.zeros(6)
    for J_i, d_i in zip(J_list, delta_list):
        abs_sum += np.abs(J_i) @ np.abs(d_i)
    
    # Weighted norm computation
    worst_norm = np.linalg.norm(np.sqrt(W) @ abs_sum)
    return abs_sum, worst_norm