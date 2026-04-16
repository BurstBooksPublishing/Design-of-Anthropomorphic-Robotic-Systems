import math
from typing import Union

def allowed_peak_force(
    A: float, 
    p_lim: float, 
    U_lim: float, 
    E: float, 
    h: float, 
    alpha: float = 0.5
) -> float:
    """
    Calculate the maximum allowable peak force based on pressure and energy constraints.
    
    Args:
        A: Contact area (mÂ²)
        p_lim: Region pressure limit (Pa)
        U_lim: Energy density limit (J/mÂ³)
        E: Tissue Young's modulus (Pa)
        h: Tissue thickness (m)
        alpha: Dimensionless parameter for energy-based calculation
    
    Returns:
        Maximum allowable peak force (N)
    
    Raises:
        ValueError: If any parameter is non-positive or alpha is zero
    """
    # Validate inputs
    if any(param <= 0 for param in [A, p_lim, U_lim, E, h]):
        raise ValueError("All parameters must be positive")
    if alpha <= 0:
        raise ValueError("Alpha must be positive")
    
    # Calculate force limits from pressure and energy constraints
    F_from_p = A * p_lim                                    # Pressure-based limit
    F_from_U = math.sqrt((U_lim * A**2 * E * h) / alpha)   # Energy-based limit
    
    # Return conservative (minimum) allowable force
    return min(F_from_p, F_from_U)