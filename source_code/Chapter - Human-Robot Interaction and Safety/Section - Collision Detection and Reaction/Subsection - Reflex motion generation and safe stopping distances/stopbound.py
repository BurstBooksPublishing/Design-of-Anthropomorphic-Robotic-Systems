import numpy as np

def calculate_min_deceleration(tau_max: float, lambda_max: float, sigma_min_J: float) -> float:
    """Calculate minimum deceleration based on system dynamics parameters."""
    gamma_lower = sigma_min_J / lambda_max
    return tau_max * gamma_lower

def calculate_stopping_distance(v0: float, t_delay: float, a_min: float) -> float:
    """Calculate conservative stopping distance considering delay and deceleration."""
    return v0 * t_delay + v0**2 / (2.0 * a_min)

# System parameters
tau_max = 60.0          # Nm - maximum torque
lambda_max = 40.0       # max eigenvalue of inertia matrix M
sigma_min_J = 0.15      # min singular value of Jacobian matrix J
v0 = 1.2                # m/s - initial task-space speed
t_delay = 0.03          # s - detection and actuation delay

# Calculate safety parameters
a_min = calculate_min_deceleration(tau_max, lambda_max, sigma_min_J)
s_stop = calculate_stopping_distance(v0, t_delay, a_min)

print(f"a_min = {a_min:.3f} m/s^2  # minimum deceleration")
print(f"s_stop = {s_stop:.3f} m    # conservative stopping distance")