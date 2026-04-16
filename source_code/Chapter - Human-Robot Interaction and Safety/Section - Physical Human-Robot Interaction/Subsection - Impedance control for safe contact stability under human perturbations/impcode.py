import numpy as np
from typing import Tuple, Optional

def cartesian_impedance_step(
    q: np.ndarray,           # Joint positions (n,)
    qd: np.ndarray,          # Joint velocities (n,)
    x: np.ndarray,           # Task positions (m,)
    xd: np.ndarray,          # Task velocities (m,)
    x_des: np.ndarray,       # Desired task positions (m,)
    xd_des: np.ndarray,      # Desired task velocities (m,)
    K_e: np.ndarray,         # Stiffness matrix (m,m)
    D_e: np.ndarray,         # Damping matrix (m,m)
    M: np.ndarray,           # Inertia matrix (n,n)
    C: np.ndarray,           # Coriolis matrix (n,n)
    g: np.ndarray,           # Gravity vector (n,)
    J: np.ndarray,           # Jacobian matrix (m,n)
    N: np.ndarray,           # Null-space projector (n,n)
    tau0: np.ndarray,        # Null-space torque (n,)
    f_h: np.ndarray          # Human force (m,)
) -> np.ndarray:             # Commanded joint torques (n,)
    """
    Compute joint torques for Cartesian impedance control with gravity compensation.
    """
    # Calculate desired Cartesian force from position/velocity errors
    f_c = K_e @ (x_des - x) + D_e @ (xd_des - xd) + f_h
    
    # Compute joint-space torque: task-space forces + null-space torque
    tau = J.T @ f_c + N @ tau0
    
    # Add gravity compensation
    tau += g
    
    return tau