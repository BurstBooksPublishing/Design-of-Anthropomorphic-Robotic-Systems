import numpy as np
from typing import Callable, Tuple

def compute_task_space_dynamics(
    q: np.ndarray,
    qdot: np.ndarray,
    M_of: Callable[[np.ndarray], np.ndarray],
    J_of: Callable[[np.ndarray], np.ndarray],
    C_of: Callable[[np.ndarray, np.ndarray], np.ndarray],
    g_of: Callable[[np.ndarray], np.ndarray],
    dJ_dt: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute task-space dynamics components for operational space control.
    
    Returns:
        Lambda: Operational space inertia matrix (m,m)
        Jsharp: Dynamically consistent generalized inverse (n,m)
        mu: Task-space Coriolis and centrifugal bias (m,)
        p: Task-space gravity bias (m,)
    """
    # Compute joint-space dynamics components
    M = M_of(q)                  # (n,n) SPD inertia matrix
    J = J_of(q)                  # (m,n) task Jacobian
    Cdot = C_of(q, qdot) @ qdot  # (n,) Coriolis/centrifugal vector
    gq = g_of(q)                 # (n,) gravity vector
    
    # Compute operational space inertia (with numerical stability)
    J_M_inv = np.linalg.solve(M, J.T).T  # J @ M^(-1)
    Lambda = np.linalg.inv(J @ J_M_inv.T)  # (m,m) operational inertia
    
    # Dynamically consistent generalized inverse
    Jsharp = J_M_inv.T @ Lambda  # (n,m)
    
    # Task-space bias terms
    mu = Lambda @ (J_M_inv @ Cdot) - Lambda @ (dJ_dt(q, qdot) @ qdot)  # Coriolis/centrifugal
    p = Lambda @ (J_M_inv @ gq)  # gravity bias
    
    return Lambda, Jsharp, mu, p

# Example usage:
# Lambda, Jsharp, mu, p = compute_task_space_dynamics(q, qdot, M_of, J_of, C_of, g_of, dJ_dt)