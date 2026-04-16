import cvxpy as cp
import numpy as np
from typing import Tuple, Optional

def solve_contact_forces(
    G: np.ndarray,
    w_des: np.ndarray,
    mu: float,
    n: int,
    solver: str = cp.OSQP
) -> Tuple[Optional[np.ndarray], float]:
    """
    Solve for contact forces that satisfy wrench balance with friction constraints.
    
    Args:
        G: Grasp matrix mapping contact forces to wrench space, shape (6, 3*n)
        w_des: Desired wrench to balance, shape (6,)
        mu: Friction coefficient
        n: Number of contact points
        solver: Convex optimization solver to use
    
    Returns:
        Tuple of (contact forces, optimal value) or (None, inf) if infeasible
    """
    # Decision variable: concatenated contact forces [f1_x, f1_y, f1_z, ..., fn_x, fn_y, fn_z]
    f = cp.Variable(3 * n)
    
    # Objective: minimize sum of squares of forces (regularization)
    objective = cp.Minimize(cp.sum_squares(f))
    
    # Constraint 1: wrench balance equation
    constraints = [G @ f == w_des]
    
    # Constraint 2: friction cone and normal force constraints for each contact
    for i in range(n):
        fi = f[3*i:3*i+3]  # Force vector for contact i
        ft = fi[:2]        # Tangential components
        fn = fi[2]         # Normal component (assuming z-axis is normal)
        constraints += [
            cp.norm(ft, 2) <= mu * fn,  # Friction cone constraint
            fn >= 0                     # Normal force must be non-negative
        ]
    
    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)
    
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return f.value, prob.value
    else:
        return None, np.inf