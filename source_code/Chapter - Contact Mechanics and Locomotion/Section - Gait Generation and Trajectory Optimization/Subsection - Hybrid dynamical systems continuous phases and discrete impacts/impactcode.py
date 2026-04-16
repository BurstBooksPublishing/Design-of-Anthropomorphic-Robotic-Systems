import numpy as np
from typing import Tuple, Optional

def plastic_reset(M: np.ndarray, Jc: np.ndarray, v_minus: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity update and impulse for plastic impact.
    
    Args:
        M: (n,n) inertia matrix
        Jc: (k,n) contact Jacobian
        v_minus: (n,) pre-impact velocity
        
    Returns:
        v_plus: (n,) post-impact velocity
        Lambda: (k,) impulse vector
    """
    # Compute impulse using least squares for numerical stability
    W = Jc @ np.linalg.solve(M, Jc.T)  # Contact space inertia (k,k)
    Lambda = -np.linalg.solve(W, Jc @ v_minus)  # Impulse vector (k,)
    v_plus = v_minus + np.linalg.solve(M, Jc.T @ Lambda)  # Post-impact velocity (n,)
    
    return v_plus, Lambda

def saltation_matrix(DR: np.ndarray, f_minus: np.ndarray, f_plus: np.ndarray, grad_g: np.ndarray, 
                   epsilon: float = 1e-12) -> np.ndarray:
    """
    Compute saltation matrix for hybrid system transitions.
    
    Args:
        DR: Jacobian of reset map at x*
        f_minus: vector field before transition
        f_plus: vector field after transition
        grad_g: gradient of switching function
        epsilon: numerical tolerance for denominator check
        
    Returns:
        Saltation matrix
    """
    denom = grad_g @ f_minus
    
    # Check for degenerate case
    if abs(denom) < epsilon:
        raise ValueError(f"Degenerate case: gradient orthogonal to flow field (denominator = {denom})")
    
    # Compute saltation matrix
    return DR + np.outer((f_plus - DR @ f_minus), grad_g) / denom