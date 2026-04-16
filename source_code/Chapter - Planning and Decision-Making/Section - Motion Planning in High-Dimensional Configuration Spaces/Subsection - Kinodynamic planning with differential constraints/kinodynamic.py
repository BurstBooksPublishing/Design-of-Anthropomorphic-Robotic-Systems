import numpy as np
from typing import List, Tuple, Callable, Union, Optional

def propagate(
    x0: np.ndarray, 
    u_seq: List[np.ndarray], 
    dt: float, 
    dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray], 
    collision_check: Callable[[np.ndarray], bool]
) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
    """
    Propagate system dynamics using RK4 integration with collision checking.
    
    Returns:
        Tuple of (final_state or None if collision, trajectory_list)
    """
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    
    for u in u_seq:
        # RK4 integration step
        k1 = dynamics(x, u)
        k2 = dynamics(x + 0.5 * dt * k1, u)
        k3 = dynamics(x + 0.5 * dt * k2, u)
        k4 = dynamics(x + dt * k3, u)
        x += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Check for collisions after state update
        if collision_check(x):
            return None, traj
            
        traj.append(x.copy())
        
    return x, traj