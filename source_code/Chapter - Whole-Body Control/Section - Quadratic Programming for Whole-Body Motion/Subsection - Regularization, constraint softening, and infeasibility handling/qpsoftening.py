import cvxpy as cvx
import numpy as np

def solve_regularized_qp(H, g, A, b, C, d, eps=1e-5, rho_e=1e3, rho_i=1e2):
    """
    Solve a regularized quadratic program with slack variables for constraints.
    
    Args:
        H: Hessian matrix (n x n)
        g: Linear term vector (n x 1)
        A: Equality constraint matrix (m_eq x n)
        b: Equality constraint vector (m_eq x 1)
        C: Inequality constraint matrix (m_ineq x n)
        d: Inequality constraint vector (m_ineq x 1)
    """
    n = H.shape[0]
    
    # Decision variables and slacks
    x = cvx.Variable(n)
    s_e = cvx.Variable(A.shape[0])  # Equality slacks (free variables)
    s_i = cvx.Variable(C.shape[0])  # Inequality slacks (non-negative)
    
    # Regularized quadratic objective with l1 penalties on slacks
    obj = 0.5 * cvx.quad_form(x, H + eps * np.eye(n)) + g.T @ x \
          + rho_e * cvx.norm1(s_e) + rho_i * cvx.sum(s_i)
    
    # Constraints with slack variables
    constraints = [
        A @ x == b + s_e,      # Equality constraints with slack
        C @ x <= d + s_i,      # Inequality constraints with slack
        s_i >= 0               # Non-negativity on inequality slacks
    ]
    
    # Solve with OSQP solver using warm start
    prob = cvx.Problem(cvx.Minimize(obj), constraints)
    prob.solve(solver=cvx.OSQP, warm_start=True)
    
    return x.value, prob.status, prob.value

# Example usage:
# x_opt, status, opt_val = solve_regularized_qp(H, g, A, b, C, d)