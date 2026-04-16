import cvxpy as cp
import numpy as np
from typing import Tuple, Optional

def solve_force_distribution(G: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Solve the force distribution problem: find alpha such that G @ alpha = W with alpha >= 0.
    
    Args:
        G: (6, p) numpy array of generators
        W: (6,) required wrench
        
    Returns:
        alpha_val: (p,) optimal force distribution coefficients
        
    Raises:
        RuntimeError: if no feasible solution exists
    """
    # Define optimization variable with non-negativity constraint
    alpha = cp.Variable(G.shape[1], nonneg=True)
    
    # Set up equality constraints for dynamics feasibility
    constraints = [G @ alpha == W]
    
    # Form and solve the feasibility problem
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.ECOS)
    
    # Check solution status and return result
    if prob.status == cp.OPTIMAL:
        return alpha.value
    else:
        raise RuntimeError('No feasible contact wrench (W_req not in CWC)')

# Example usage:
# alpha_val = solve_force_distribution(G, W)