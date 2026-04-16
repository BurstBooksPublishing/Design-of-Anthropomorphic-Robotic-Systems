import numpy as np
from scipy import signal

# Physical parameters (example ankle values)
Jm = 5e-3          # Motor inertia
N = 50.0           # Gear ratio
Jmr = Jm / N**2    # Reflected motor inertia
JL = 2.5e-2        # Load inertia
bm = 0.1           # Motor damping
bL = 0.2           # Load damping
k = 1200.0         # Spring constant

# Plant transfer function: from motor torque to spring torque
# Denominator: (Jm_r s^2 + bm s)(JL s^2 + bL s) + k*(Jm_r + JL)s^2 + k*(bm + bL)s
Pm = [Jmr, bm, 0.0]
PL = [JL, bL, 0.0]
den = np.polymul(Pm, PL)
den[-3] += k * (Jmr + JL)  # s^2 term
den[-2] += k * (bm + bL)   # s^1 term

# Numerator: k * (JL s^2 + bL s)
num = k * np.array([JL, bL, 0.0, 0.0])

G = signal.TransferFunction(num, den)

# PD Controller: C(s) = Kp + Kd*s
Kp = 100.0
Kd = 0.01
C = signal.TransferFunction([Kd, Kp], [1.0])

# Closed-loop system
L = signal.series(C, G)
T = signal.feedback(L, 1)

# Frequency response
w, mag, phase = signal.bode(T, np.logspace(0, 3, 1000))

# Bandwidth calculation (-3 dB from low-frequency gain)
low_gain = mag[0]
bw_idx = np.where(mag <= low_gain - 3.0)[0]
bw = w[bw_idx[0]] / (2 * np.pi) if bw_idx.size > 0 else None

print("Estimated closed-loop bandwidth (Hz):", bw)