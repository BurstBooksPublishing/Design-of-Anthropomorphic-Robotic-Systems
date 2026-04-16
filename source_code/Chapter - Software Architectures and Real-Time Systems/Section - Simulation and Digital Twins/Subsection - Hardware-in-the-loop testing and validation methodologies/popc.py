import numpy as np
from typing import Callable, Any

def passive_torque_controller(
    compute_command: Callable[[], np.ndarray],
    read_velocity_sensor: Callable[[], np.ndarray],
    send_to_actuator: Callable[[np.ndarray], None],
    dt: float,
    N: int,
    alpha: float = 1e-3
) -> None:
    """
    Passivity-based torque controller with energy dissipation.
    
    Args:
        compute_command: Function returning torque command vector
        read_velocity_sensor: Function returning angular velocity vector
        send_to_actuator: Function sending torque command to actuators
        dt: Control period (seconds)
        N: Number of control iterations
        alpha: Dissipation gain (tunable parameter)
    """
    E = 0.0  # Stored energy increment
    
    for _ in range(N):
        u = compute_command()      # Commanded torque vector
        y = read_velocity_sensor() # Measured angular velocity vector
        
        E += float(np.dot(u, y)) * dt  # Energy increment: âˆ«uáµ€y dt
        
        if E > 0:  # Passivity violation detected
            d = alpha * E / (1.0 + alpha)  # Energy-proportional damping
            u = u - d * y                  # Inject damping term
            E -= d * float(np.dot(y, y)) * dt  # Update stored energy
            
        send_to_actuator(u)