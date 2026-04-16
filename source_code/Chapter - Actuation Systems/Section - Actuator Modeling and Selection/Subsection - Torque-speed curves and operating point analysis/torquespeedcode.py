import numpy as np
from scipy.optimize import bisect
from typing import Tuple

# Motor/transmission parameters (SI units)
K_t = 0.07      # Torque constant (Nm/A)
K_e = 0.07      # Back EMF constant (V*s/rad)
R = 0.2         # Winding resistance (Ohm)
V = 48.0        # Supply voltage (Volt)
N = 100.0       # Gear ratio
eta = 0.9       # Gearbox efficiency
b_m = 0.0       # Motor viscous damping coefficient
I_max = 40.0    # Maximum current (A)

# Derived parameters
omega_o0 = V / (N * K_e)  # No-load output speed
kappa = (R / (eta * N**2 * K_t * K_e)) + (b_m / (eta * N * K_e * K_t))  # Speed-torque slope
tau_stall = eta * N * K_t * V / R  # Stall torque
tau_max = eta * N * K_t * I_max    # Maximum available torque

def tau_motor(omega_o: float) -> float:
    """Calculate motor torque at given output speed."""
    return (omega_o0 - omega_o) / kappa

# Viscous load model: tau_load = tau_const + B*omega_o
tau_const = 78.5    # Constant gravity torque (Nm)
B = 5.0             # Viscous damping coefficient (Nm/(rad/s))

def tau_load(omega_o: float) -> float:
    """Calculate load torque at given output speed."""
    return tau_const + B * omega_o

def find_operating_point() -> Tuple[float, float]:
    """Find operating point where motor and load torques balance."""
    omega_min, omega_max = 0.0, omega_o0
    try:
        omega_op = bisect(lambda w: tau_load(w) - tau_motor(w), omega_min, omega_max)
        tau_op = tau_load(omega_op)
        return omega_op, tau_op
    except ValueError as e:
        raise RuntimeError(f"Failed to find operating point: {e}")

# Calculate and display results
if __name__ == "__main__":
    omega_op, tau_op = find_operating_point()
    
    print(f"omega_o0={omega_o0:.3f} rad/s, tau_stall={tau_stall:.1f} Nm")
    print(f"operating omega={omega_op:.3f} rad/s, tau={tau_op:.3f} Nm")