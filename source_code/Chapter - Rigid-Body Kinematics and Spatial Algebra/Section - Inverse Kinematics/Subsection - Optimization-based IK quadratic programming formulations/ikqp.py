import numpy as np, osqp, scipy.sparse as sp
# inputs: J (p x n), e (p), W (p x p), R (n x n), Aineq, bineq, Aeq, beq
H = J.T @ W @ J + R                          # Hessian (dense)
g = -J.T @ W @ e                             # gradient
P = sp.csc_matrix((H + H.T)/2)               # symmetric sparse Hessian
q = g.astype(np.float64)
A_blocks = []
l = []
u = []
if Aeq is not None:                           # equality -> two-sided
    A_blocks.append(sp.csc_matrix(Aeq))
    l.extend(beq); u.extend(beq)
if Aineq is not None:
    A_blocks.append(sp.csc_matrix(Aineq))
    l.extend([-np.inf]*Aineq.shape[0]); u.extend(bineq)
A = sp.vstack(A_blocks).tocsc()
# setup and solve
prob = osqp.OSQP()
prob.setup(P=P, q=q, A=A, l=np.array(l), u=np.array(u), verbose=False)
res = prob.solve()
dq = res.x                                   # \Deltaq solution