import numpy as np
from typing import Callable, Tuple

def newton_project(
    q0: np.ndarray, 
    h: Callable[[np.ndarray], np.ndarray], 
    Dh: Callable[[np.ndarray], np.ndarray], 
    tol: float = 1e-8, 
    maxit: int = 20
) -> np.ndarray:
    """
    Project point q0 onto manifold defined by h(q) = 0 using Newton's method.
    
    Args:
        q0: Initial point in R^n
        h: Constraint function R^n -> R^k
        Dh: Jacobian function R^n -> R^{kÃ—n}
        tol: Convergence tolerance
        maxit: Maximum iterations
    
    Returns:
        Projected point on manifold
    """
    q = q0.copy().astype(float)
    
    for _ in range(maxit):
        r = h(q)
        residual_norm = np.linalg.norm(r)
        if residual_norm < tol:
            break
            
        J = Dh(q)
        # Solve for minimum-norm correction in range space
        try:
            delta = np.linalg.lstsq(J @ J.T, r, rcond=None)[0]
            q = q - J.T @ delta
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if J*J.T is singular
            delta = np.linalg.pinv(J) @ r
            q = q - delta
            
    return q

def sample_tangent(
    q: np.ndarray, 
    Dh: Callable[[np.ndarray], np.ndarray], 
    sampler_Rd: Callable[[int], np.ndarray]
) -> np.ndarray:
    """
    Sample a point in the tangent space of the manifold at q.
    
    Args:
        q: Point on manifold
        Dh: Jacobian of constraint function
        sampler_Rd: Function that samples from R^d
    
    Returns:
        Tangent space sample
    """
    J = Dh(q)
    
    # Compute nullspace basis via SVD
    try:
        U, S, Vt = np.linalg.svd(J, full_matrices=True)
        rank = np.sum(S > 1e-12)
        d = J.shape[1] - rank
        N = Vt.T[:, -d:] if d > 0 else np.empty((J.shape[1], 0))
    except np.linalg.LinAlgError:
        # Fallback rank estimation
        rank = np.linalg.matrix_rank(J, tol=1e-12)
        d = J.shape[1] - rank
        _, _, Vt = np.linalg.svd(J, full_matrices=True)
        N = Vt.T[:, -d:] if d > 0 else np.empty((J.shape[1], 0))
    
    if d == 0:
        return q.copy()
        
    xi = sampler_Rd(d)
    return q + N @ xi