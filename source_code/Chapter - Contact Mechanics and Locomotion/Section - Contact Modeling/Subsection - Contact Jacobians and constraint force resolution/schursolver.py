import numpy as np
from scipy.linalg import solve, inv
from typing import Tuple, Optional

def solve_contact_forces(M: np.ndarray, Jc: np.ndarray, h: np.ndarray, 
                        S: np.ndarray, tau: np.ndarray, Jdot_qdot: np.ndarray,
                        friction_cone_projection: Optional[callable] = None) -> np.ndarray:
    """
    Solve for contact forces using constrained dynamics equations.
    
    Args:
        M: (n,n) symmetric positive definite mass matrix
        Jc: (k,n) contact Jacobian matrix
        h: (n,) bias force vector
        S: (m,n) selection matrix for actuated joints
        tau: (m,) actuator torque vector
        Jdot_qdot: (k,) velocity product terms for contact constraints
        friction_cone_projection: Optional function to project forces into friction cone
    
    Returns:
        (k,) contact force vector
    """
    # Compute mass matrix inverse (use factorization for better performance)
    Minv = inv(M)
    
    # Compute contact space inertia matrix: W = Jc * Minv * Jc^T
    W = Jc @ Minv @ Jc.T
    
    # Compute right-hand side of constraint equation
    generalized_forces = h - S.T @ tau
    rhs = -(Jc @ (Minv @ generalized_forces) + Jdot_qdot)
    
    # Solve linear system W * lambda = rhs for contact forces
    lambda_contact = solve(W, rhs, assume_a='sym')
    
    # Optional: Project contact forces into friction cone
    if friction_cone_projection is not None:
        lambda_contact = friction_cone_projection(lambda_contact)
    
    return lambda_contact