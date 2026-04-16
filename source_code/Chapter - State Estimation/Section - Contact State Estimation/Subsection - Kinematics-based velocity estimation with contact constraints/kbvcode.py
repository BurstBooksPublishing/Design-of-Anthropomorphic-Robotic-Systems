import numpy as np
from typing import Tuple, Optional

def estimate_base_velocity(
    Jcb: np.ndarray,
    Jcq: np.ndarray,
    qdot: np.ndarray,
    omega_gyro: np.ndarray,
    W: Optional[np.ndarray] = None,
    Rw: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Estimate base velocity using contact constraints and gyro measurements.
    
    Args:
        Jcb: Contact Jacobian w.r.t. base (m x 6)
        Jcq: Contact Jacobian w.r.t. joints (m x n)
        qdot: Joint velocities (n,)
        omega_gyro: Gyro measurement (3,)
        W: Contact constraint weights (m x m), default: 1e6 * I
        Rw: Gyro covariance (3 x 3), default: 1e-3 * I
    
    Returns:
        v_b: Base twist [omega; linear velocity] (6,)
    """
    # Validate input dimensions
    m, n = Jcb.shape[0], Jcq.shape[1]
    if Jcq.shape[0] != m or len(qdot) != n or len(omega_gyro) != 3:
        raise ValueError("Inconsistent input dimensions")
    
    # Initialize default weights if not provided
    if W is None:
        W = np.eye(m) * 1e6
    if Rw is None:
        Rw = np.eye(3) * 1e-3
    
    # Form constraint system: Jcb * v_b = -Jcq * qdot
    A = Jcb
    b = -Jcq @ qdot
    
    # Selection matrix for angular velocity component
    S = np.hstack([np.eye(3), np.zeros((3, 3))])
    
    # Solve regularized normal equations
    # Weighted contact constraints + gyro measurement prior
    M = A.T @ W @ A + S.T @ np.linalg.inv(Rw) @ S
    rhs = A.T @ W @ b + S.T @ np.linalg.inv(Rw) @ omega_gyro
    
    return np.linalg.solve(M, rhs)

# Example usage:
# v_b = estimate_base_velocity(Jcb, Jcq, qdot, omega_gyro)