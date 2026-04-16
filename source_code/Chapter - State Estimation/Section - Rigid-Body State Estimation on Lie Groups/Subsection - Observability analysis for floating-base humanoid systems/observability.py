import numpy as np
from typing import List, Tuple

def empirical_observability(A_list: List[np.ndarray], 
                          H_list: List[np.ndarray], 
                          horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the empirical observability matrix and its singular values.
    
    Args:
        A_list: List of discrete-time state Jacobians A_k
        H_list: List of output Jacobians H_k
        horizon: Time horizon for observability analysis
        
    Returns:
        Tuple of (observability_matrix, singular_values)
    """
    if len(A_list) < horizon or len(H_list) < horizon:
        raise ValueError("Insufficient matrices for specified horizon")
    
    n = A_list[0].shape[0]  # State dimension
    O = []
    Ak = np.eye(n)  # Initialize with identity matrix
    
    for k in range(horizon):
        O.append(H_list[k] @ Ak)  # Stack H_k * A^{k}
        Ak = A_list[k] @ Ak       # Update A^{k+1} = A_k * A^k
    
    Omat = np.vstack(O)
    singular_values = np.linalg.svd(Omat, compute_uv=False)
    
    return Omat, singular_values