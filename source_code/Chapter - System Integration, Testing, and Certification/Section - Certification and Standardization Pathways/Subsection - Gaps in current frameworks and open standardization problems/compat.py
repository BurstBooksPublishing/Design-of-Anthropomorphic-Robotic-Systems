import cvxpy as cp
import numpy as np
from typing import Tuple, Optional

def check_feasibility() -> bool:
    """
    Check feasibility of system parameter constraints using convex optimization.
    Returns True if constraints are feasible, False otherwise.
    """
    # Decision variable: [tau_limit, latency_ctrl, frame_rot_z]
    x = cp.Variable(3)
    
    # Actuator torque bounds: -100 <= tau <= 100
    A1 = np.array([[1, 0, 0], [-1, 0, 0]])
    b1 = np.array([100.0, 0.0])
    
    # Timing constraints: 0.001 <= latency <= 0.01
    A2 = np.array([[0, 1, 0], [0, -1, 0]])
    b2 = np.array([0.01, -0.001])
    
    # Frame rotation fixed to zero
    A3 = np.array([[0, 0, 1]])
    b3 = np.array([0.0])
    
    # Combine all constraints
    constraints = [
        A1 @ x <= b1,
        A2 @ x <= b2,
        A3 @ x == b3
    ]
    
    # Solve feasibility problem
    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve()
        return prob.status == cp.OPTIMAL
    except cp.error.SolverError:
        return False

# Check if the problem is feasible
is_feasible = check_feasibility()