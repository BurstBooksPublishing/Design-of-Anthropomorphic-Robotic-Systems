import numpy as np
from typing import Tuple, Union

def compute_minimal_energy_solution(
    stiffness: np.ndarray, 
    geometry_ratios: np.ndarray, 
    joint_angle: float
) -> Tuple[np.ndarray, float]:
    """
    Compute minimal energy solution for spring system under zero torque.
    
    For quadratic energy functions, the optimal spring angles are analytically
    determined as the product of geometry ratios and joint angle.
    """
    # Analytical solution for minimal energy configuration
    optimal_angles = geometry_ratios * joint_angle
    
    # Energy calculation (zero for optimal solution in quadratic case)
    energy = 0.5 * np.sum(stiffness * (optimal_angles - geometry_ratios * joint_angle)**2)
    
    return optimal_angles, energy

def main() -> None:
    # System parameters for 2-spring configuration
    stiffness = np.array([5000.0, 5000.0])        # Spring stiffness (Nm/rad)
    geometry_ratios = np.array([0.04, 0.04])      # Geometry ratios (rad^-1)
    joint_angle = 0.3                             # Joint angle (rad)
    
    # Compute minimal energy solution
    optimal_angles, energy = compute_minimal_energy_solution(
        stiffness, geometry_ratios, joint_angle
    )
    
    # Output results
    print(f"Optimal spring angles: {optimal_angles}")
    print(f"Minimal energy: {energy}")

if __name__ == "__main__":
    main()