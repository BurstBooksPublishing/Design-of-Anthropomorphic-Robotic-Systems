import numpy as np
from typing import Tuple

class BeamStressCalculator:
    """Calculates stresses in a rotating beam with payload."""
    
    def __init__(self, length: float, link_mass: float, payload_mass: float, 
                 payload_distance: float, radius: float):
        self.L = length
        self.m_link = link_mass
        self.m_payload = payload_mass
        self.d = payload_distance
        self.r = radius
        
        # Cross-sectional properties
        self.A = np.pi * self.r**2
        self.I = np.pi * self.r**4 / 4
        self.J = 2 * self.I
    
    def calculate_inertial_moments(self, linear_accel: float, 
                                 angular_accel: float) -> Tuple[float, float, float]:
        """Calculate inertial moments from payload and link rotation."""
        I_c = (1/12) * self.m_link * self.L**2
        M_payload = self.m_payload * linear_accel * self.d
        M_link_rot = I_c * angular_accel
        M_total = M_payload + M_link_rot
        return M_payload, M_link_rot, M_total
    
    def calculate_stresses(self, M_total: float, N_total: float = 0.0) -> Tuple[float, float, float]:
        """Calculate von Mises stress components."""
        sigma_bend = M_total * self.r / self.I
        sigma_axial = N_total / self.A
        tau = 0.0  # No torsion in this example
        sigma_vm = np.sqrt(sigma_axial**2 + sigma_bend**2 + 3*tau**2)
        return sigma_bend, sigma_axial, sigma_vm

def main():
    # Geometry and masses (SI units)
    calculator = BeamStressCalculator(
        length=0.30,
        link_mass=1.5,
        payload_mass=2.0,
        payload_distance=0.30,
        radius=0.02
    )
    
    # Parametric inputs
    a = 5.0        # linear acceleration (m/s^2)
    alpha = 20.0   # angular acceleration (rad/s^2)
    
    # Calculate inertial moments
    M_payload, M_link_rot, M_total = calculator.calculate_inertial_moments(a, alpha)
    
    # Calculate stresses
    sigma_bend, sigma_axial, sigma_vm = calculator.calculate_stresses(M_total)
    
    # Output results
    print(f"M_total (Nâ‹…m): {M_total:.2f}")
    print(f"sigma_vm (Pa): {sigma_vm:.0f}")
    print(f"safety factor (yield/sigma): {250e6/sigma_vm:.1f}")  # for Al alloy

if __name__ == "__main__":
    main()