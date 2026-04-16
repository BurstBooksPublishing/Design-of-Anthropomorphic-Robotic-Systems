import cvxpy as cp
import numpy as np

def wrench_allocation_solver(J: np.ndarray, w_req: np.ndarray, tau_max: float) -> tuple[bool, np.ndarray]:
    """
    Solve the wrench allocation problem: min ||Ï„||â‚ subject to Jáµ€Ï„ = w_req and |Ï„| â‰¤ Ï„_max
    
    Returns:
        tuple of (is_feasible, torques)
    """
    n = J.shape[1]
    tau = cp.Variable(n)
    
    constraints = [
        J.T @ tau == w_req,           # Wrench feasibility constraint
        cp.abs(tau) <= tau_max        # Actuator torque limits
    ]
    
    prob = cp.Problem(cp.Minimize(cp.norm(tau, 1)), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    
    is_feasible = prob.status == cp.OPTIMAL
    torques = tau.value if is_feasible else np.zeros(n)
    
    return is_feasible, torques

# Example usage
if __name__ == "__main__":
    # System parameters
    n_actuators = 12
    J = np.random.randn(6, n_actuators)  # Jacobian matrix (6 x n)
    w_req = np.array([10., 0., 0., 0., 0., 0.])  # Desired wrench [Fx,Fy,Fz,Mx,My,Mz]
    tau_max = 40.0  # Torque limit per actuator
    
    # Solve allocation problem
    feasible, torques = wrench_allocation_solver(J, w_req, tau_max)
    
    # Output results
    print(f"Feasible: {feasible}")
    if feasible:
        print(f"Allocated torques: {torques}")
        print(f"Resulting wrench: {J.T @ torques}")