import numpy as np
from typing import Tuple, Union

def hunt_crossley(nodes: Union[int, slice, np.ndarray], 
                  k: float, 
                  b: float, 
                  alpha: float, 
                  delta: np.ndarray, 
                  delta_dot: np.ndarray, 
                  n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute normal contact forces using Hunt-Crossley model with nonlinear stiffness and damping.
    
    Args:
        nodes: Indices or slice for selecting contact points
        k: Stiffness coefficient
        b: Damping coefficient
        alpha: Nonlinear exponent (0 < alpha < 1 for Hertzian contact)
        delta: Penetration depths (N,)
        delta_dot: Penetration velocity (N,)
        n: Contact normal vectors (N,3) or (3,)
        
    Returns:
        f_vec: Contact forces (N,3)
        K_n: Normal stiffness (N,)
        B_n: Normal damping (N,)
    """
    # Ensure vectorized inputs
    delta = np.atleast_1d(delta)
    delta_dot = np.atleast_1d(delta_dot)
    
    # Hunt-Crossley contact model: f_n = k*Î´^Î± + b*Î´^Î±*Î´_dot
    # Only active when penetration Î´ > 0
    active = delta > 0
    delta_active = np.where(active, delta, 0.0)
    
    # Normal force magnitude with nonlinear stiffness and velocity-dependent damping
    f_n = np.where(active, k * delta_active**alpha + b * delta_active**alpha * delta_dot, 0.0)
    
    # Incremental stiffness and damping coefficients for Î´ > 0
    K_n = np.where(active, alpha * (k + b * delta_dot) * delta_active**(alpha - 1), 0.0)
    B_n = np.where(active, b * delta_active**alpha, 0.0)
    
    # Compute spatial force vectors: f = f_n * n
    if n.ndim == 1:
        n = n[None, :]  # Broadcast single normal to all contacts
    f_vec = f_n[:, None] * n
    
    return f_vec, K_n, B_n