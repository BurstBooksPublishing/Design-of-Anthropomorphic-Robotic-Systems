import numpy as np
from scipy.optimize import minimize
from scipy.linalg import svd, inv

def solve_tension_optimization(R, tau, tmin):
    """
    Solve minimum-norm tension optimization: min ||t||â‚‚ s.t. R*t = tau, t â‰¥ tmin
    """
    # Validate input dimensions
    if R.shape[0] != len(tau) or R.shape[1] != len(tmin):
        raise ValueError("Dimension mismatch in inputs")
    
    # Compute left pseudoinverse and nullspace projector
    RRt_inv = inv(R @ R.T)
    Rp = R.T @ RRt_inv                    # Left pseudoinverse
    Pnull = np.eye(R.shape[1]) - Rp @ R   # Nullspace projector
    
    # Particular solution (minimum norm)
    t_part = Rp @ tau
    
    # Orthonormal basis for nullspace using SVD
    U, S, _ = svd(Pnull)
    rank = np.sum(S > 1e-9)
    N = U[:, :rank]                       # Nullspace basis
    
    # Reparameterize in nullspace: t = t_part + N*y, solve for y
    b = tmin - t_part                     # Constraint offset
    
    # Objective: min 0.5 * ||y||Â² (minimizes ||t||â‚‚)
    def objective(y):
        return 0.5 * np.dot(y, y)
    
    # Constraints: N*y >= b (equivalent to t >= tmin)
    constraints = [{'type': 'ineq', 'fun': lambda y, i=i: (N @ y)[i] - b[i]} 
                   for i in range(len(b))]
    
    # Initial guess: satisfy constraints if possible
    y0 = np.zeros(rank)
    
    # Solve QP
    result = minimize(objective, y0, constraints=constraints, method='SLSQP')
    
    if not result.success:
        raise RuntimeError("Optimization failed to converge")
    
    # Compute final solution
    t = t_part + N @ result.x
    return t

# Problem data
R = np.array([[0.02, -0.01, 0.015],
              [-0.01, 0.03, 0.005]])
tau = np.array([1.0, 0.5])
tmin = np.array([5.0, 5.0, 5.0])

# Solve
t = solve_tension_optimization(R, tau, tmin)