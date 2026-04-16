import numpy as np
import scipy.sparse as sp
from osqp import OSQP

# Problem dimensions: n=gen_accel, m=actuators, k=contact_wrenches, p=slacks
n, m, k, p = 12, 12, 6, 2

# Objective: minimize acceleration, torque, contact, and slack norms
H_diag = np.hstack([
    np.full(n, 1.0),      # acceleration regularization
    np.full(m, 1e-2),     # torque regularization
    np.full(k, 1e-3),     # contact regularization
    np.full(p, 1e6)       # slack penalty
])
H = sp.diags(H_diag, format='csc')
g = np.zeros(n + m + k + p)

# Dynamics constraint: M*ddq - S^T*tau - Jc^T*f = -h
M = sp.eye(n, format='csc')  # mass matrix
S = sp.eye(m, n, format='csc')  # actuator selection
Jc = sp.random(k, n, density=0.3, format='csc')  # contact Jacobian
Aeq = sp.hstack([M, -S.T, -Jc.T, sp.csc_matrix((n, p))], format='csc')
beq = np.zeros(n)  # bias term

# Inequality constraints: friction pyramid and torque limits
Af = sp.random(8, k, density=0.5, format='csc')  # friction polyhedral model
Aineq = sp.hstack([
    sp.csc_matrix((8, n)),
    sp.csc_matrix((8, m)),
    Af,
    sp.csc_matrix((8, p))
], format='csc')
bineq = np.zeros(8)

# Slack variable non-negativity
A_slack = sp.hstack([
    sp.csc_matrix((p, n + m + k)),
    sp.eye(p, format='csc')
], format='csc')

# Combined constraints
A = sp.vstack([Aeq, Aineq, A_slack], format='csc')
l = np.hstack([beq, np.full(Aineq.shape[0], -np.inf), np.zeros(p)])
u = np.hstack([beq, bineq, np.full(p, np.inf)])

# Solve QP
solver = OSQP()
solver.setup(P=H, q=g, A=A, l=l, u=u, verbose=False)
result = solver.solve()

# Extract solution
x_opt = result.x  # [ddq, tau, f, s]