import numpy as np
from typing import Dict, Union

def shaped_reward(s: Dict[str, np.ndarray], 
                  a: np.ndarray, 
                  s_next: Dict[str, np.ndarray], 
                  gamma: float, 
                  k_p: float) -> float:
    """
    Compute shaped reward for locomotion tasks with potential-based reward shaping.
    
    Args:
        s: Current state dictionary containing 'com' and other state variables
        a: Action vector (torques)
        s_next: Next state dictionary containing 'com', 'goal', 'tau', 'contact_slip'
        gamma: Discount factor for potential shaping
        k_p: Potential shaping coefficient
    
    Returns:
        float: Combined reward value
    """
    # Base task reward components
    pos_err = s_next['com'][:2] - s_next['goal'][:2]
    r_base = -np.dot(pos_err, pos_err)  # Distance penalty to goal
    r_reg = -0.01 * np.dot(s_next['tau'], s_next['tau'])  # Control effort penalty
    
    # Contact penalty
    slip = -1.0 if s_next.get('contact_slip', False) else 0.0
    
    # Potential-based shaping (forward progress encouragement)
    phi_s = k_p * s['com'][0]  # Current potential
    phi_s_n = k_p * s_next['com'][0]  # Next potential
    shaping = gamma * phi_s_n - phi_s  # Potential difference
    
    return r_base + r_reg + slip + shaping