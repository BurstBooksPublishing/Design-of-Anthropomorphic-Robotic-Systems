import numpy as np
# Assemble KKT matrix and rhs (M: (n+6,n+6), Jc: (m,n+6))
KKT = np.block([[M, Jc.T],
                [Jc, np.zeros((Jc.shape[0], Jc.shape[0]))]])
rhs = np.concatenate([S_T_tau - Cdot - g, -Jc_dot_dotq])  # vectorize RHS
# Direct solve (use a symmetric indefinite solver in production)
sol = np.linalg.solve(KKT, rhs)
ddq = sol[:M.shape[0]]          # generalized accelerations
lambda_c = sol[M.shape[0]:]     # contact multipliers (wrenches)