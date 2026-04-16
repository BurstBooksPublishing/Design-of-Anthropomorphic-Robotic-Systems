import numpy as np
import cvxpy as cp
from typing import Union, Optional

def project_gradient(
    g: np.ndarray, 
    a: np.ndarray, 
    b: float, 
    eps: float = 1e-8
) -> np.ndarray:
    """
    Project gradient onto feasible set defined by linear constraint.
    
    Solves: min_x 0.5*||x-g||^2  s.t. a^T x <= b
    """
    n = g.shape[0]
    x = cp.Variable(n)
    obj = cp.Minimize(0.5 * cp.sum_squares(x - g))
    constraints = [a.T @ x <= b]
    prob = cp.Problem(obj, constraints)
    
    # Solve with numerical robustness settings
    prob.solve(
        solver=cp.OSQP, 
        eps_abs=eps, 
        eps_rel=eps,
        max_iter=10000,
        verbose=False
    )
    
    if x.value is None:
        raise RuntimeError("Optimization failed to converge")
        
    return np.asarray(x.value).ravel()