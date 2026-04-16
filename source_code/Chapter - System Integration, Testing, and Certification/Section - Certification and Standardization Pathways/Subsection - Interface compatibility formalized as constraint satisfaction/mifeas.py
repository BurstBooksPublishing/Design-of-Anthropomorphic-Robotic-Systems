import cvxpy as cp
import numpy as np
from typing import Tuple, Optional

def solve_connector_alignment() -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
    """
    Solves the connector alignment feasibility problem with mechanical and electrical constraints.
    Returns feasibility status and optimal values if feasible.
    """
    # Continuous variables: translational error (x,y,z) and voltage
    dx = cp.Variable(3)    # translational error in 3D space
    V = cp.Variable()      # voltage at connector
    
    # Binary variable: protocol match indicator
    q = cp.Variable(boolean=True)
    
    # System parameters and tolerances
    t_max = 0.002          # mechanical tolerance: 2 mm
    V_nom = 12.0           # nominal voltage
    V_tol = 0.5            # voltage tolerance
    
    # Constraint definitions
    constraints = [
        cp.norm(dx, 2) <= t_max,                    # mechanical alignment constraint
        V >= V_nom - V_tol,                         # voltage lower bound
        V <= V_nom + V_tol,                         # voltage upper bound
        q == 1                                      # protocol match requirement
    ]
    
    # Feasibility problem formulation
    prob = cp.Problem(cp.Minimize(0), constraints)
    
    # Solve using MIP-capable solver
    prob.solve(solver=cp.ECOS_BB)
    
    # Return feasibility status and solution values
    if prob.status == cp.OPTIMAL:
        return True, dx.value, V.value
    else:
        return False, None, None

# Execute the solver
is_feasible, translation_error, voltage = solve_connector_alignment()