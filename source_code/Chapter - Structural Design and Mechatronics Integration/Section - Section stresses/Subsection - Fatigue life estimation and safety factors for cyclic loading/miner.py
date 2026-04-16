import numpy as np
from typing import List, Tuple, Union

def basquin_cycles_to_failure(sigma_a: float, sigma_f_prime: float, b: float) -> float:
    """
    Calculate cycles to failure using Basquin equation.
    
    Args:
        sigma_a: Stress amplitude
        sigma_f_prime: Fatigue strength coefficient
        b: Fatigue strength exponent
    
    Returns:
        Cycles to failure
    """
    if sigma_a <= 0:
        return float('inf')
    return 0.5 * (sigma_a / sigma_f_prime) ** (1.0 / b)

def miner_damage(cycles: List[Tuple[float, float, float]], 
                sigma_f_prime: float, 
                b: float, 
                sigma_u: float, 
                Kf: float = 1.0) -> float:
    """
    Calculate cumulative damage using Miner's rule with Goodman mean-stress correction.
    
    Args:
        cycles: List of (sigma_a, sigma_m, count) tuples
        sigma_f_prime: Fatigue strength coefficient
        b: Fatigue strength exponent
        sigma_u: Ultimate tensile strength
        Kf: Fatigue stress concentration factor
    
    Returns:
        Cumulative damage ratio
    """
    D = 0.0
    for sigma_a, sigma_m, n in cycles:
        # Avoid division by zero and tensile overload
        if sigma_m >= sigma_u:
            return float('inf')
            
        # Modified Goodman mean-stress correction
        sigma_a_eff = (Kf * sigma_a) / (1.0 - sigma_m / sigma_u)
        
        # Ensure effective stress amplitude is positive
        if sigma_a_eff <= 0:
            continue
            
        N = basquin_cycles_to_failure(sigma_a_eff, sigma_f_prime, b)
        # Handle infinite life case
        if np.isinf(N) or N <= 0:
            continue
            
        D += n / N
        
    return D