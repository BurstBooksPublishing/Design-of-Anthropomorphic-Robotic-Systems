import numpy as np
from typing import Tuple, Union

class CapturePointDynamics:
    """Computes capture point dynamics for bipedal locomotion stability analysis."""
    
    def __init__(self, gravity: float = 9.81, com_height: float = 1.0):
        self.g = gravity
        self.z_c = com_height
        self.omega = np.sqrt(gravity / com_height)  # Natural frequency
    
    def compute_capture_point(self, com_pos: float, com_vel: float) -> float:
        """Calculate instantaneous capture point from CoM state."""
        return com_pos + com_vel / self.omega
    
    def is_zero_step_capturable(self, capture_point: float, support_polygon: Tuple[float, float]) -> bool:
        """Check if robot can stop without stepping."""
        return support_polygon[0] <= capture_point <= support_polygon[1]
    
    def simulate_capture_point_trajectory(self, 
                                        initial_capture_point: float, 
                                        support_polygon: Tuple[float, float],
                                        duration: float = 1.0, 
                                        dt: float = 0.001) -> np.ndarray:
        """
        Simulate capture point dynamics with constant CoP at support edge.
        Uses Euler integration for \dot{xi} = omega * (xi - p).
        """
        steps = int(duration / dt)
        trajectory = np.empty(steps)
        
        # Select CoP at support polygon edge based on capture point direction
        cop = support_polygon[1] if initial_capture_point > 0 else support_polygon[0]
        
        capture_point = initial_capture_point
        for i in range(steps):
            capture_point += dt * self.omega * (capture_point - cop)
            trajectory[i] = capture_point
            
        return trajectory

# Usage example
if __name__ == "__main__":
    # Initialize dynamics model
    cp_dynamics = CapturePointDynamics()
    
    # Initial CoM state (sagittal scalar example)
    com_pos_0 = 0.0    # m
    com_vel_0 = 0.5    # m/s
    
    # Compute capture point
    xi_0 = cp_dynamics.compute_capture_point(com_pos_0, com_vel_0)
    
    # Support polygon bounds
    support_polygon = (-0.10, 0.10)
    
    # Check zero-step capturability
    zero_step_capturable = cp_dynamics.is_zero_step_capturable(xi_0, support_polygon)
    
    # Simulate trajectory
    trajectory = cp_dynamics.simulate_capture_point_trajectory(
        initial_capture_point=xi_0,
        support_polygon=support_polygon,
        duration=1.0,
        dt=0.001
    )