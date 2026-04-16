import numpy as np
from typing import Tuple

def compute_dynamics_consistent_projection(M: np.ndarray, 
                                         J: np.ndarray, 
                                         qdot: np.ndarray, 
                                         Fr: np.ndarray,
                                         Kq: np.ndarray, 
                                         Dq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
    """
    Compute dynamically consistent projection matrices and power decomposition.
    
    Returns:
        Lambda: task space inertia matrix
        Jd: dynamically consistent generalized inverse
        N: null space projector
        power_task: task space power
        qnull: null space velocity
        tau_null: null space torque
        power_null: null space power
    """
    # Compute task space inertia matrix (generalized inverse)
    Lambda = np.linalg.inv(J @ np.linalg.inv(M) @ J.T)
    
    # Compute dynamically consistent generalized inverse
    Jd = np.linalg.inv(M) @ J.T @ Lambda
    
    # Compute null space projector
    N = np.eye(M.shape[0]) - Jd @ J
    
    # Compute task space power
    power_task = (J @ qdot).T @ Fr
    
    # Compute null space velocity and torque
    qnull = N @ qdot
    tau_null = N.T @ (-Kq @ qnull - Dq @ qnull)
    
    # Compute null space power
    power_null = qdot.T @ tau_null
    
    # Verify power conservation
    total_power = power_task + power_null
    joint_torque_power = qdot.T @ (J.T @ Fr + tau_null)
    
    assert np.allclose(total_power, joint_torque_power), "Power decomposition inconsistency"
    assert np.allclose(qdot.T @ (J.T @ Fr), power_task), "Task power computation error"
    
    return Lambda, Jd, N, power_task, qnull, tau_null, power_null