import numpy as np
# Inputs: A (n,n), Jc (m,n), b (n), g (n), S (r,n), qdot (n), tau (r)
A = ...      # mass matrix
Jc = ...     # contact Jacobian
b = ...      # bias vector
g = ...      # gravity vector
S = ...      # actuation selection
qdot = ...   # generalized velocity
tau = ...    # actuator torques

# Precompute Schur complement and its inverse
W = Jc @ np.linalg.solve(A, Jc.T)           # Jc A^{-1} Jc^T
Winv = np.linalg.inv(W)                     # assume invertible

# Dynamically consistent null-space projector
N = np.eye(A.shape[0]) - np.linalg.solve(A, Jc.T @ (Winv @ Jc))

# Contact force lambda via (3)
JcAdag = Jc @ np.linalg.solve(A, np.eye(A.shape[0]))
Jdot_qdot = ...      # compute \dot Jc * qdot externally
rhs = -Jdot_qdot + Jc @ np.linalg.solve(A, S.T @ tau - b - g)
lambda_c = Winv @ rhs

# Reduced dynamics (optionally form basis Z from N)
# Compute Z via QR on N (columns spanning range(N))
Z, _ = np.linalg.qr(N)                       # Z has n columns; select first n-m
Z = Z[:, :A.shape[0]-Jc.shape[0]]
Az = Z.T @ A @ Z
bz = Z.T @ (b + A @ (np.zeros_like(qdot)))  # include Zdot if available
Szg = Z.T @ S.T

# Az, bz, Szg form reduced model Az zddot + bz = Szg tau