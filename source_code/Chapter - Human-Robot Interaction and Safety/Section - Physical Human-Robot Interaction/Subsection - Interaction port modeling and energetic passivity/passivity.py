import numpy as np
from typing import Tuple

# System matrices (symmetric positive definite)
M_X = np.diag([2.0, 2.0, 1.5, 0.5, 0.5, 0.5])    # Task inertia matrix
KP = np.diag([50.0, 50.0, 40.0, 10.0, 10.0, 10.0])  # Stiffness matrix
D = np.diag([5.0, 5.0, 4.0, 1.0, 1.0, 1.0])       # Damping matrix

def storage(x_err: np.ndarray, x_dot: np.ndarray) -> float:
    """Compute storage function (Lyapunov function) value."""
    kinetic_energy = 0.5 * x_dot.dot(M_X).dot(x_dot)
    potential_energy = 0.5 * x_err.dot(KP).dot(x_err)
    return kinetic_energy + potential_energy

def storage_dot(x_err: np.ndarray, x_dot: np.ndarray, x_ddot: np.ndarray, 
                w_h: np.ndarray) -> float:
    """
    Compute time derivative of storage function.
    Uses passivity condition: Vdot <= w_h^T * x_dot
    """
    # Direct computation from passivity inequality (when w_h=0, Vdot <= 0)
    if x_ddot is not None:
        # Alternative computation using system dynamics
        dynamics_residual = M_X.dot(x_ddot) + D.dot(x_dot) + KP.dot(x_err) - w_h
        Vdot = -x_dot.dot(dynamics_residual) - x_dot.dot(D).dot(x_dot)
    else:
        Vdot = x_dot.dot(w_h) - x_dot.dot(D).dot(x_dot)
    return Vdot

# Example usage with test values
x_err = np.array([0.01, -0.02, 0.0, 0.0, 0.0, 0.0])
x_dot = np.array([0.05, 0.0, -0.02, 0.0, 0.0, 0.0])
w_h = np.zeros(6)  # No external input

# Verify passivity: Vdot <= w_h^T * x_dot (should be <= 0 when w_h=0)
storage_value = storage(x_err, x_dot)
storage_dot_value = storage_dot(x_err, x_dot, None, w_h)

print(f"Storage function: {storage_value:.6f}")
print(f"Storage function time derivative: {storage_dot_value:.6f}")