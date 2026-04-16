import numpy as np
from typing import Dict, Any, List, Tuple, Union
from shapely.geometry import Point, Polygon

def leg_feasible(p: Dict[str, Any], 
                 q_samples: np.ndarray, 
                 support_poly: List[Tuple[float, float]], 
                 tau_max: np.ndarray) -> bool:
    """
    Check if all leg configurations are feasible given constraints.
    
    Args:
        p: Dictionary containing robot physical parameters (lengths, masses)
        q_samples: Array of joint configurations to test
        support_poly: Support polygon vertices as list of (x,y) tuples
        tau_max: Maximum allowable joint torques for each joint
    
    Returns:
        True if all configurations are feasible, False otherwise
    """
    # Create Shapely polygon for efficient point-in-polygon tests
    support_polygon = Polygon(support_poly)
    
    for q in q_samples:
        # Compute 6xn spatial Jacobian matrix
        J = compute_jacobian_leg(q, p)
        
        # Calculate 3D center of mass position
        com = compute_COM(q, p)
        
        # Project COM to ground plane (ZMP calculation)
        zmp = project_COM_to_plane(com)  # Returns (x,y) coordinates
        
        # Check ZMP constraint - must be inside support polygon
        if not Point(zmp).within(support_polygon):
            return False
            
        # Calculate worst-case joint torques from gravitational forces
        w = np.array([0, 0, -total_weight(p)])  # Vertical force only
        tau = np.abs(J[:3, :].T @ w)           # Torque magnitudes
        
        # Check torque limits for all joints
        if np.any(tau > tau_max):
            return False
            
    return True

# Note: compute_jacobian_leg, compute_COM, project_COM_to_plane, 
# and total_weight are model-specific functions that must be implemented