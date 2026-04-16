import numpy as np
import cvxpy as cp
from typing import Tuple, Optional

def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load regressor and torque data from files."""
    Y = np.load('Y.npy')
    tau = np.load('tau.npy')
    return Y, tau

def build_pseudo_inertia_matrix(pi_var: cp.Variable) -> cp.Expression:
    """
    Construct 4x4 pseudo-inertia matrix from 10 inertial parameters.
    Parameters: [m, h_x, h_y, h_z, s11, s12, s13, s22, s23, s33]
    """
    m = pi_var[0]
    h = cp.vstack([pi_var[1], pi_var[2], pi_var[3]])
    Sigma = cp.vstack([
        cp.hstack([pi_var[4],  pi_var[5],  pi_var[6]]),
        cp.hstack([pi_var[5],  pi_var[7],  pi_var[8]]),
        cp.hstack([pi_var[6],  pi_var[8],  pi_var[9]])
    ])
    P = cp.vstack([
        cp.hstack([Sigma, h]),
        cp.hstack([h.T, cp.reshape(m, (1, 1))])
    ])
    return P

def solve_inertial_parameter_estimation(Y: np.ndarray, tau: np.ndarray) -> Optional[np.ndarray]:
    """
    Estimate inertial parameters using convex optimization with physical consistency constraints.
    """
    n_samples, n_params = Y.shape
    if n_params != 10:
        raise ValueError(f"Expected 10 parameters, got {n_params}")
    
    pi = cp.Variable(n_params)
    
    # Objective: minimize squared error between predicted and actual torques
    objective = cp.Minimize(cp.sum_squares(Y @ pi - tau))
    
    # Constraints: positive mass and positive semi-definite pseudo-inertia matrix
    constraints = [
        pi[0] >= 1e-6,  # Mass must be positive
        build_pseudo_inertia_matrix(pi) >> 0  # Pseudo-inertia matrix must be PSD
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)
    
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return None
        
    return pi.value

def main() -> None:
    """Main execution function."""
    try:
        Y, tau = load_data()
        pi_hat = solve_inertial_parameter_estimation(Y, tau)
        
        if pi_hat is not None:
            print("Estimated inertial parameters:", pi_hat)
        else:
            print("Optimization failed to converge")
            
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()