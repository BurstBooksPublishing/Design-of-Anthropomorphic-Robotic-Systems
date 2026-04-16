import numpy as np
from typing import Tuple

def compute_safety_distance(
    mass: float = 5.0,
    initial_velocity: float = 1.0,
    response_time: float = 0.05,
    max_acceleration: float = 8.0,
    contact_safety_margin: float = 0.02
) -> Tuple[float, float]:
    """
    Compute stopping distance and required separation for collision avoidance.
    
    Args:
        mass: System mass in kg
        initial_velocity: Initial velocity in m/s
        response_time: Detection+compute+actuation time in seconds
        max_acceleration: Maximum achievable acceleration in m/s^2
        contact_safety_margin: Additional safety margin in meters
    
    Returns:
        Tuple of (stopping_distance, required_separation)
    """
    # Compute worst-case stopping distance (equation 1)
    stopping_distance = (initial_velocity * response_time + 
                        initial_velocity**2 / (2 * max_acceleration))
    
    required_separation = stopping_distance + contact_safety_margin
    return stopping_distance, required_separation

def is_safe_approach(
    measured_separation: float,
    required_separation: float
) -> bool:
    """Check if measured separation is sufficient for safe operation."""
    return measured_separation > required_separation

# Configuration parameters
M_EQ = 5.0          # kg
V0 = 1.0            # m/s
T_R = 0.05          # s (detection+compute+actuate)
A_MAX = 8.0         # m/s^2
D_CONTACT_SAFE = 0.02  # m

# Compute safety distances
d_stop, d_sep_required = compute_safety_distance(
    mass=M_EQ,
    initial_velocity=V0,
    response_time=T_R,
    max_acceleration=A_MAX,
    contact_safety_margin=D_CONTACT_SAFE
)

print(f"d_stop = {d_stop:.4f}")           
print(f"required separation > {d_sep_required:.4f}")

# Safety verification with measured separation
measured_sep = 0.15
is_safe = is_safe_approach(measured_sep, d_sep_required)
print(f"safe: {is_safe}")