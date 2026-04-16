import casadi as ca
import numpy as np

# Problem parameters
m = 75.0
g = ca.DM([0, 0, -9.81])
N = 20  # Number of intervals per phase
h = ca.SX.sym('h')  # Time step (can be optimized)

# Symbolic states and controls at k, c (midpoint), and k+1
p_k = ca.SX.sym('p_k', 3)
ell_k = ca.SX.sym('ell_k', 3)
kap_k = ca.SX.sym('kap_k', 3)

p_k1 = ca.SX.sym('p_k1', 3)
ell_k1 = ca.SX.sym('ell_k1', 3)
kap_k1 = ca.SX.sym('kap_k1', 3)

p_c = ca.SX.sym('p_c', 3)
ell_c = ca.SX.sym('ell_c', 3)
kap_c = ca.SX.sym('kap_c', 3)

# Contact forces at k, c, and k+1
f_k = ca.SX.sym('f_k', 3)
f_c = ca.SX.sym('f_c', 3)
f_k1 = ca.SX.sym('f_k1', 3)

# Dynamics: dp = ell/m, dell = mg + f, dkap = cross(r, f)
def f(p, ell, kap, f):
    dp = ell / m
    dell = m * g + f
    dkap = ca.cross(ca.SX.zeros(3), f)  # Placeholder for actual moment arm
    return ca.vertcat(dp, dell, dkap)

# Evaluate dynamics at k, c, and k+1
fk = f(p_k, ell_k, kap_k, f_k)
fc = f(p_c, ell_c, kap_c, f_c)
fk1 = f(p_k1, ell_k1, kap_k1, f_k1)

# Hermite-Simpson collocation defect
state_diff = ca.vertcat(p_k1 - p_k, ell_k1 - ell_k, kap_k1 - kap_k)
integral_approx = (h / 6) * (fk + 4 * fc + fk1)
defect = state_diff - integral_approx

# Midpoint consistency conditions
pos_mid = p_c - (p_k + p_k1) / 2 - (h / 8) * (fk[:3] - fk1[:3])
vel_mid = ell_c - (ell_k + ell_k1) / 2 - (h / 8) * (fk[3:6] - fk1[3:6])
ang_mid = kap_c - (kap_k + kap_k1) / 2 - (h / 8) * (fk[6:9] - fk1[6:9])
midpoint_consistency = ca.vertcat(pos_mid, vel_mid, ang_mid)