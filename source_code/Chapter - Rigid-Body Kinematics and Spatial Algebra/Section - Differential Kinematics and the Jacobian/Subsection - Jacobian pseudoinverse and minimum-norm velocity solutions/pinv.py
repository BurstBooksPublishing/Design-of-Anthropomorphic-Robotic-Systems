import numpy as np

def pinv_svd(J, tol=1e-9):
    U,s,Vt = np.linalg.svd(J, full_matrices=False)
    s_inv = np.array([1/si if si>tol else 0.0 for si in s])
    return Vt.T @ np.diag(s_inv) @ U.T  # Moore-Penrose

def dyn_consistent_pinv(J, M, lam=0.0):
    # J: m x n, M: n x n SPD, lam damping scalar
    JWJt = J @ np.linalg.solve(M, J.T)  # J M^{-1} J^T
    JWJt += lam*lam*np.eye(JWJt.shape[0])  # optional damping
    inv = np.linalg.inv(JWJt)
    return np.linalg.solve(M, J.T) @ inv  # W^{-1}=M^{-1}

# Example usage (J shape 6x7): qdot = pinv_svd(J) @ v