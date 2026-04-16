import numpy as np
import control as ctl
from typing import Tuple

def compute_control_system_metrics(J: float, b: float, Kp: float, Kd: float, tauf: float) -> Tuple[float, float, float, float]:
    """
    Compute control system performance metrics for a motor with filtered PD controller.
    
    Returns:
        wc: Gain crossover frequency
        PM: Phase margin (radians)
        tau_max: Maximum delay margin
        jitter_LHS: Jitter bound
    """
    # Define plant and controller transfer functions
    P = ctl.TransferFunction([1.0], [J, b])                    # Plant: 1/(Js+b)
    C = ctl.TransferFunction([Kd, Kp], [tauf, 1.0])           # Filtered PD controller
    L = C * P                                                 # Open-loop transfer function
    
    # Frequency analysis
    w = np.logspace(-1, 3, 10000)                            # Frequency grid
    mag, phase, _ = ctl.bode(L, w, plot=False)
    
    # Find gain crossover frequency and phase margin
    idx = np.argmin(np.abs(mag - 1.0))                       # Find where |L(jw)| = 1
    wc = w[idx]                                              # Gain crossover frequency
    PM = np.pi + phase[idx]                                  # Phase margin in radians
    tau_max = PM / wc                                        # Maximum allowable time delay
    
    # Compute complementary sensitivity function and H-infinity norm
    T = ctl.feedback(L, 1)                                   # T = L/(1+L)
    T_inf = ctl.nsigma(T, w)[0].max()                        # H-infinity norm approximation
    
    # Conservative jitter bound calculation
    omega_max = 200.0
    jitter_LHS = T_inf * (2 * np.abs(np.sin(0.5 * omega_max * tauf)))  # Jitter bound
    
    return wc, PM, tau_max, jitter_LHS

# System parameters
J, b = 0.02, 0.1          # Motor inertia and damping
Kp, Kd, tauf = 100.0, 1.0, 0.01  # Controller gains and filter time constant

# Compute and display results
wc, PM, tau_max, jitter_LHS = compute_control_system_metrics(J, b, Kp, Kd, tauf)
print(f"wc={wc:.4f}, PM={PM:.4f}, tau_max={tau_max:.6f}, jitter_LHS={jitter_LHS:.6f}")