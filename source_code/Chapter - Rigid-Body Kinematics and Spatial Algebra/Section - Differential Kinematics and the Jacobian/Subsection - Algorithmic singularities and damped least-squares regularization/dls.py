import numpy as np
# J1,J2: numpy arrays; v: task velocity
def dls_pinv(J, lam):
    U,s,Vt = np.linalg.svd(J, full_matrices=False)
    # compute gains sigma/(sigma^2+lam^2)
    g = s / (s*s + lam*lam)
    return (Vt.T * g) @ U.T  # V diag(g) U^T

def adaptive_lambda(svals, eps=1e-2, k=1e-2):
    smin = svals.min()
    return max(0.0, k*(eps - smin)) if smin < eps else 0.0

# example usage: compute projected Jacobian singular values
U1,s1,Vt1 = np.linalg.svd(J1, full_matrices=False)
lam = adaptive_lambda(s1)
J1_dls = dls_pinv(J1, lam)         # DLS pseudoinverse
N1_dls = np.eye(J1.shape[1]) - J1_dls @ J1
Jproj = J2 @ N1_dls
sv_proj = np.linalg.svd(Jproj, compute_uv=False)  # diagnostic singular values