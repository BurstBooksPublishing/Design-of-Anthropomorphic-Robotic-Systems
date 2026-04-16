import cvxpy as cp
import numpy as np
from typing import Tuple, Optional

def solve_contact_optimization(
    w_des: np.ndarray,
    G: np.ndarray, 
    Jc: np.ndarray,
    R: np.ndarray,
    tau_pre: np.ndarray,
    A_f: np.ndarray,
    b_f: np.ndarray,
    tau_max: float = 30.0,
    solver: str = cp.OSQP
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], cp.Problem]:
    """
    Solve contact force and actuator torque optimization problem.
    
    Minimizes actuator effort while satisfying wrench matching, equilibrium,
    friction constraints, and torque limits.
    """
    
    # Decision variables
    f = cp.Variable(G.shape[1])        # contact forces
    tau_a = cp.Variable(R.shape[1])    # actuator torques
    
    # Problem constraints
    constraints = [
        G @ f == w_des,                                    # wrench matching
        R @ tau_a + tau_pre + Jc.T @ f == 0,              # force equilibrium
        A_f @ f <= b_f,                                   # friction cone
        cp.abs(tau_a) <= tau_max                          # torque limits
    ]
    
    # Objective: minimize actuator effort
    obj = cp.Minimize(cp.sum_squares(tau_a))
    prob = cp.Problem(obj, constraints)
    
    # Solve optimization
    prob.solve(solver=solver)
    
    # Return solution if feasible
    if prob.status == cp.OPTIMAL:
        return f.value, tau_a.value, prob
    else:
        return None, None, prob