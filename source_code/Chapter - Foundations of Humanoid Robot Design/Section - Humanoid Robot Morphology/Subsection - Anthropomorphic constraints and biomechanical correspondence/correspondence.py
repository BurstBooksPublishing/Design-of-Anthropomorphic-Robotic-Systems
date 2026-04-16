import numpy as np

# fk_h, fk_r: user-provided functions returning SE3 (R, p) tuples
# inertia_h, inertia_r: return 6x6 spatial inertia matrices
def pose_error(se3_a, se3_b):
    R_a, p_a = se3_a
    R_b, p_b = se3_b
    R_err = R_a.T @ R_b
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
    pos_err = np.linalg.norm(p_a - p_b)
    return angle**2 + pos_err**2

def evaluate(phi, theta, qsamples):
    Ck = 0.0
    Cd = 0.0
    for qh in qsamples:
        qr = phi(qh)  # mapping
        Ck += pose_error(fk_r(qr), fk_h(qh))
        I_h = inertia_h(qh)
        I_r = inertia_r(qr, theta)
        Cd += np.linalg.norm(I_r - I_h, 'fro')**2
    N = len(qsamples)
    return Ck / N, Cd / N

# usage: sample qsamples on Q_H; call evaluate(phi, theta, qsamples)