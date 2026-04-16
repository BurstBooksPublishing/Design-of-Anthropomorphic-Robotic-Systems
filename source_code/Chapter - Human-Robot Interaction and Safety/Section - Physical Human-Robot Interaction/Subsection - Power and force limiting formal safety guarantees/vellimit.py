import numpy as np
from typing import Tuple

def velocity_limit(M: np.ndarray, J: np.ndarray, n: np.ndarray, E_max: float) -> Tuple[float, float]:
    """
    Calculate effective mass and maximum velocity limit for a given constraint.
    
    Args:
        M: (n,n) inertia matrix
        J: (6,n) Jacobian matrix
        n: (3,) unit normal vector in task frame
        E_max: maximum allowed kinetic energy
    
    Returns:
        Tuple of (effective_mass, velocity_limit)
    """
    # Extract translational Jacobian (first 3 rows)
    Jt = J[:3, :]
    
    # Compute operational-space inertia matrix (translational part only)
    M_inv = np.linalg.inv(M)
    Lambda = np.linalg.inv(Jt @ M_inv @ Jt.T)  # 3x3 translational inertia
    
    # Calculate effective mass along constraint normal
    m_eff = float(n.T @ Lambda @ n)
    
    # Compute maximum velocity bound (ensure non-negative under sqrt)
    v_max = np.sqrt(max(0.0, 2.0 * E_max / m_eff)) if m_eff > 0 else np.inf
    
    return m_eff, v_max