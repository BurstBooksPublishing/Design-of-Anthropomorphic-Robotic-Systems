import numpy as np
from typing import List, Tuple

# Rodrigues rotation formula: so(3) vector to SO(3) matrix
def rodrigues(w: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

# Exponential map from se(3) to SE(3): twist (v, omega) -> transformation matrix
def exp_se3(v: np.ndarray, omega: np.ndarray) -> np.ndarray:
    R = rodrigues(omega)
    theta = np.linalg.norm(omega)
    if theta < 1e-8:
        V = np.eye(3)
    else:
        k = omega / theta
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]])
        V = np.eye(3) + (1 - np.cos(theta)) / theta * K + (theta - np.sin(theta)) / theta * (K @ K)
    t = V @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

# Forward kinematics using Product of Exponentials formula
def fkine_poe(S_hat: List[np.ndarray], M: np.ndarray, q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for i, qi in enumerate(q):
        w_skew = S_hat[i][:3, :3]
        v = S_hat[i][:3, 3]
        omega = np.array([w_skew[2, 1], w_skew[0, 2], w_skew[1, 0]])
        T = T @ exp_se3(v * qi, omega * qi)
    return T @ M

# Project 3D point to image plane using camera intrinsic matrix
def project(x: np.ndarray, K: np.ndarray) -> np.ndarray:
    X = x[:3]
    z = X[2]
    if z == 0:
        raise ValueError("Depth cannot be zero.")
    u = K @ X
    return u[:2] / z

# Compute residuals and Jacobian for bundle adjustment
def residual_and_jac(
    S_hat: List[np.ndarray],
    M: np.ndarray,
    q: np.ndarray,
    landmarks: List[np.ndarray],
    meas: List[np.ndarray],
    K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    res_list = []
    J_list = []

    for p, u_meas in zip(landmarks, meas):
        T_fk = fkine_poe(S_hat, M, q)
        X_hom = np.hstack([p, 1])
        X = (T_fk @ X_hom)[:3]
        u_proj = project(X, K)
        r = u_proj - u_meas
        res_list.append(r)

        eps = 1e-6
        J_cols = []
        for i in range(len(q)):
            dq = np.zeros_like(q)
            dq[i] = eps
            T_fk_pert = fkine_poe(S_hat, M, q + dq)
            X_pert = (T_fk_pert @ X_hom)[:3]
            Jx = (X_pert - X) / eps
            dproj = np.array([
                [K[0, 0] / X[2], 0, -K[0, 0] * X[0] / (X[2] ** 2)],
                [0, K[1, 1] / X[2], -K[1, 1] * X[1] / (X[2] ** 2)]
            ])
            J_cols.append(dproj @ Jx)
        J_list.append(np.column_stack(J_cols))

    return np.concatenate(res_list), np.vstack(J_list)

# Gauss-Newton step for optimization
# Usage:
# res, J = residual_and_jac(...)
# delta = np.linalg.solve(J.T @ J + reg * np.eye(J.shape[1]), -J.T @ res)
# q = retract(q, delta)