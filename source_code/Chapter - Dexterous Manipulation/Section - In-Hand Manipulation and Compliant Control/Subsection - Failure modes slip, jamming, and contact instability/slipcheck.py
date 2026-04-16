import numpy as np
from typing import Tuple, Union

def slip_check(f: np.ndarray, mu: float) -> Tuple[bool, float, float]:
    """
    Check if contact is slipping based on friction cone constraint.
    
    Args:
        f: Contact force vector [fx, fy, fz]
        mu: Coefficient of friction
        
    Returns:
        Tuple of (slipping, normal_force, tangential_force)
    """
    fn = max(0.0, f[2])  # Normal force (assuming z-axis is normal)
    ft = np.linalg.norm(f[:2])  # Tangential force magnitude
    slipping = (ft >= mu * fn) and (fn > 1e-6)
    return slipping, fn, ft

def contact_stability_eigs(M: np.ndarray, 
                          D: np.ndarray, 
                          K: np.ndarray, 
                          Jc: np.ndarray, 
                          Kc: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalues of closed-loop system with contact constraints.
    
    Args:
        M: Mass matrix (nq x nq)
        D: Damping matrix (nq x nq)
        K: Stiffness matrix (nq x nq)
        Jc: Contact Jacobian (nc x nq)
        Kc: Contact stiffness (nc x nc)
        
    Returns:
        Eigenvalues of the closed-loop system matrix
    """
    # Compute effective stiffness with contact constraints
    Keff = K - Jc.T @ Kc @ Jc
    
    # Build closed-loop system matrix (2*nq x 2*nq)
    nq = M.shape[0]
    A = np.block([
        [np.zeros((nq, nq)), np.eye(nq)],
        [-np.linalg.solve(M, Keff), -np.linalg.solve(M, D)]
    ])
    
    return np.linalg.eigvals(A)