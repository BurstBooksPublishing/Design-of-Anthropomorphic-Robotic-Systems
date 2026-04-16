import numpy as np
from typing import Tuple

def dyn_consistent_projector(J: np.ndarray, M: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute dynamically consistent projector matrices for constrained dynamics.
    
    Args:
        J: Constraint Jacobian matrix (m x n)
        M: Mass matrix (n x n), symmetric positive definite
        eps: Regularization parameter for numerical stability
        
    Returns:
        Tuple of (Jbar, N, Lambda) where:
        - Jbar: Generalized inverse of J (n x m)
        - N: Null space projector (n x n)
        - Lambda: Constraint force scaling matrix (m x m)
    """
    # Compute M^{-1} using solve for numerical stability
    Minv = np.linalg.solve(M, np.eye(M.shape[0]))
    
    # Compute J * M^{-1} * J^T + eps*I
    JMInvJT = J @ Minv @ J.T
    regularized_matrix = JMInvJT + eps * np.eye(JMInvJT.shape[0])
    
    # Compute Lambda = inv(J * M^{-1} * J^T + eps*I)
    Lambda = np.linalg.solve(regularized_matrix, np.eye(regularized_matrix.shape[0]))
    
    # Compute generalized inverse and null space projector
    Jbar = Minv @ J.T @ Lambda
    N = np.eye(M.shape[0]) - Jbar @ J
    
    return Jbar, N, Lambda