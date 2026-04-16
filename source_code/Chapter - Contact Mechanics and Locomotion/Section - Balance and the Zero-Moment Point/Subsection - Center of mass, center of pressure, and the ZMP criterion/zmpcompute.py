import numpy as np
from typing import Union, Tuple

def compute_zmp(
    rG: np.ndarray, 
    aG: np.ndarray, 
    zG: float, 
    m: float, 
    dotH: np.ndarray, 
    g: float = 9.81
) -> np.ndarray:
    """
    Compute the Zero Moment Point (ZMP) position.
    
    Args:
        rG: Center of mass position (3D)
        aG: Center of mass acceleration (3D)  
        zG: ZMP height
        m: Mass
        dotH: Angular momentum rate (3D)
        g: Gravitational acceleration
        
    Returns:
        ZMP position vector with z=0 (3D)
        
    Raises:
        ValueError: If normal force is non-positive (lift-off condition)
    """
    # Calculate normal (z-direction) force component
    Fz = m * (aG[2] + g)
    
    # Check for valid contact (positive normal force)
    if Fz <= 0:
        raise ValueError("Contact lift-off or invalid normal force (Fz <= 0)")
    
    # Compute ZMP coordinates using moment equilibrium
    px = rG[0] - (dotH[1] + zG * m * aG[0]) / Fz
    py = rG[1] + (dotH[0] - zG * m * aG[1]) / Fz
    
    return np.array([px, py, 0.0])