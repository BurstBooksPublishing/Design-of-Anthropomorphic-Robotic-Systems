import numpy as np
from typing import Optional, Tuple

def pgs_lcp(A: np.ndarray, b: np.ndarray, maxiter: int = 1000, tol: float = 1e-8) -> Tuple[np.ndarray, bool]:
    """
    Solve Linear Complementarity Problem using Projected Gauss-Seidel method.
    
    Solves: find x >= 0 such that Ax + b >= 0 and x^T(Ax + b) = 0
    """
    n = b.size
    x = np.zeros(n)
    
    # Precompute diagonal elements for efficiency
    diag = np.diag(A)
    
    # Check for zero diagonal elements which would cause division by zero
    if np.any(np.abs(diag) < 1e-12):
        raise ValueError("Matrix A must have non-zero diagonal elements")
    
    converged = False
    for it in range(maxiter):
        x_old = x.copy()
        
        for i in range(n):
            # Compute residual excluding current variable's contribution
            r = b[i] + A[i, :] @ x - A[i, i] * x[i]
            xi = -r / diag[i]  # Unconstrained update
            x[i] = max(0.0, xi)  # Project onto non-negative orthant
        
        # Check convergence using infinity norm
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            converged = True
            break
    
    return x, converged