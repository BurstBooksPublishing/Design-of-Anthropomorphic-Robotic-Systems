import cvxpy as cp
import numpy as np

def solve_feasibility_problem(
    N: int,
    Aint: np.ndarray,
    bint: np.ndarray,
    Ad: np.ndarray,
    Bd: np.ndarray,
    x0: np.ndarray,
    tau_max: float,
    angle_limit: float
) -> cp.Problem:
    """
    Solve feasibility problem for wrench and torque sequence optimization.
    
    Args:
        N: Prediction horizon
        Aint: Interface constraint matrix (m, 4*N)
        bint: Interface constraint vector (m,)
        Ad: Discrete dynamics A matrix
        Bd: Discrete dynamics B matrix
        x0: Initial state vector
        tau_max: Maximum allowable torque
        angle_limit: Safety angle limit for state constraint
    """
    
    # Decision variables: wrench (3xN) and torque (1xN) sequences
    w = cp.Variable((3, N))    
    tau = cp.Variable((1, N))  
    
    # Build constraints iteratively over prediction horizon
    constraints = []
    state = x0.copy()
    
    for k in range(N):
        # Interface polyhedral constraints: Aint * [w_k; tau_k] <= bint
        constraints.append(Aint @ cp.vstack([w[:, k], tau[:, k]]) <= bint)
        
        # Torque magnitude limits: |tau_k|_âˆž <= tau_max
        constraints.append(cp.norm(tau[:, k], 'inf') <= tau_max)
        
        # State evolution: x_{k+1} = Ad * x_k + Bd * tau_k
        state = Ad @ state + Bd @ tau[:, k]
        
        # Safety constraint: angle limits on first state component
        constraints.append(state[0] >= -angle_limit)
        constraints.append(state[0] <= angle_limit)
    
    # Formulate and solve feasibility problem (minimize 0 subject to constraints)
    problem = cp.Problem(cp.Minimize(0), constraints)
    problem.solve(solver=cp.GUROBI)
    
    return problem

# Example usage:
# problem = solve_feasibility_problem(N, Aint, bint, Ad, Bd, x0, tau_max, angle_limit)
# if problem.status == cp.OPTIMAL:
#     print("Feasible solution found")