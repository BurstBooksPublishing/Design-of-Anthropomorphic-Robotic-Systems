import cvxpy as cp
import numpy as np
from typing import Union, Optional

def compute_safe_arbitration(
    Lf_h: float,
    Lg_h: float,
    u_h_scalar: float,
    u_r_scalar: float,
    h_val: float,
    gamma: float,
    alpha_pref: float,
    prev_alpha: float,
    solver_kwargs: Optional[dict] = None
) -> tuple[float, float]:
    """
    Compute safe arbitration parameter with barrier function constraints.
    
    Returns:
        tuple of (alpha_value, alpha_smoothed)
    """
    if solver_kwargs is None:
        solver_kwargs = {"warm_start": True}
    
    # Define optimization variable
    alpha = cp.Variable()
    
    # Safety constraint: Lf_h + Lg_h*((1-alpha)*u_h + alpha*u_r) >= -gamma*h
    safety_ineq = Lf_h + Lg_h*((1-alpha)*u_h_scalar + alpha*u_r_scalar) + gamma*h_val
    constraints = [safety_ineq >= 0, alpha >= 0, alpha <= 1]
    
    # Minimize deviation from human preference
    objective = cp.Minimize(cp.square(alpha - alpha_pref))
    problem = cp.Problem(objective, constraints)
    
    # Solve optimization
    problem.solve(solver=cp.OSQP, **solver_kwargs)
    
    if alpha.value is None:
        raise RuntimeError("Optimization failed to converge")
    
    alpha_val = float(alpha.value)
    alpha_smoothed = 0.95 * prev_alpha + 0.05 * alpha_val
    
    return alpha_val, alpha_smoothed