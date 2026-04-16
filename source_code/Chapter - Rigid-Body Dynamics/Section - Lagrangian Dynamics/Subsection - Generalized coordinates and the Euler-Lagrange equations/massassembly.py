import numpy as np

def assemble_M(J_list, I_list):
    # J_list: list of J_i (6xn), I_list: list of I_i (6x6)
    n = J_list[0].shape[1]
    M = np.zeros((n,n))
    for J,I in zip(J_list, I_list):
        M += J.T @ I @ J   # per-link contribution
    return M

def coriolis_vector(M, q, qdot, eps=1e-8):
    # Christoffel-based finite-differencing for robustness in examples
    n = M.shape[0]; Cqdot = np.zeros(n)
    dM = np.zeros((n,n,n))
    for k in range(n):
        dq = np.zeros(n); dq[k] = eps
        # user supplies function mass(q); here M is passed, so numeric diff omitted in sample
        # in real code, compute M(q+dq) and M(q-dq)
    # placeholder: use skew-symmetry property or robot-specific regressor
    return Cqdot

def eom_residual(J_list, I_list, q, qdot, qddot, tau, g, Jc=None, lam=None):
    M = assemble_M(J_list, I_list)
    Cqdot = coriolis_vector(M, q, qdot)
    g_vec = g(q)                       # user-provided gravity vector function
    rhs = M @ qddot + Cqdot + g_vec
    if Jc is not None and lam is not None:
        rhs -= Jc.T @ lam
    return rhs - tau                   # residual should be zero for consistent dynamics