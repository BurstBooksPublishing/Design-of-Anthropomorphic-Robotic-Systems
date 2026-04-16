import numpy as np
from typing import Optional, Tuple

def projected_gradient_lcp(N: np.ndarray, 
                          b: np.ndarray, 
                          steps: int = 200, 
                          alpha: float = 1e-3,
                          tol: float = 1e-6,
                          verbose: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Solve LCP: min 0.5 x^T N x + b^T x s.t. x >= 0 using projected gradient descent.
    
    Returns:
        Tuple of (solution_vector, convergence_info)
    """
    # Validate inputs
    if N.ndim != 2 or N.shape[0] != N.shape[1]:
        raise ValueError("N must be a square matrix")
    if b.ndim != 1 or len(b) != N.shape[0]:
        raise ValueError("b must be a 1D array compatible with N")
    if steps <= 0 or alpha <= 0:
        raise ValueError("steps and alpha must be positive")
    
    x = np.zeros_like(b, dtype=float)
    converged = False
    residual_norm = 0.0
    
    # Compute step size using spectral radius for better convergence
    try:
        L = np.linalg.norm(N, 2)  # Spectral norm
        tau = 1.0 / (L + 1e-12)
    except np.linalg.LinAlgError:
        tau = alpha  # Fallback to provided alpha
    
    for k in range(steps):
        grad = N @ x + b
        x_new = np.maximum(0.0, x - tau * grad)
        
        # Check convergence using projected gradient condition
        residual = x - x_new
        residual_norm = np.linalg.norm(residual)
        
        if residual_norm < tol:
            converged = True
            break
            
        x = x_new
    
    if verbose:
        print(f"{'Converged' if converged else 'Not converged'} after {k+1} iterations. "
              f"Residual norm: {residual_norm:.2e}")
    
    convergence_info = {
        'converged': converged,
        'iterations': k + 1,
        'residual_norm': residual_norm,
        'step_size': tau
    }
    
    return x, convergence_info