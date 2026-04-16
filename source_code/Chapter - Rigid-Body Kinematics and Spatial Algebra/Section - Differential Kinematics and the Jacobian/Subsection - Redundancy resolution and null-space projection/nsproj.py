import numpy as np
def damped_dyn_pinv(J, M, damping):
    # J: m x n, M: n x n SPD, damping: scalar >0
    A = J @ np.linalg.inv(M) @ J.T
    # damped inverse (m x m)
    A_damped = A + (damping**2) * np.eye(A.shape[0])
    Ainv = np.linalg.solve(A_damped, np.eye(A.shape[0]))
    # dynamically consistent pseudoinverse (n x m)
    return np.linalg.inv(M) @ J.T @ Ainv

def nullspace_velocity(J, M, v_des, grad_h, damping=1e-2):
    Jdag = damped_dyn_pinv(J, M, damping)
    Ndyn = np.eye(J.shape[1]) - Jdag @ J
    qdot_task = Jdag @ v_des           # accomplish task
    qdot_null = Ndyn @ grad_h         # secondary objective in null-space
    return qdot_task + qdot_null
# Usage: qdot = nullspace_velocity(J, M, v_des, grad_h, damping=0.01)