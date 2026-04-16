import numpy as np
from typing import Tuple

def compute_kinematic_mappings() -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity and force coupling matrices for tendon-driven robotic systems.
    
    Returns:
        Tuple containing:
        - Jc: Velocity coupling matrix mapping actuator rates to joint rates
        - Ct: Force coupling matrix mapping actuator torques to joint torques
    """
    # System geometry: moment-arms (m x n) and spool radii (m x ma)
    R = np.array([[0.012, 0.010, 0.0],
                  [0.0,   0.009, 0.011]])
    S = np.array([[0.015],
                  [0.020]])

    # Velocity coupling: pseudoinverse for underdetermined system
    Jc = np.linalg.pinv(R) @ S
    
    # Force coupling: transpose of velocity coupling for energy conservation
    Ct = R.T @ np.linalg.pinv(S.T)
    
    return Jc, Ct

def main() -> None:
    """Execute kinematic mapping computations and display results."""
    Jc, Ct = compute_kinematic_mappings()
    
    print("J_c =\n", Jc)
    print("C(q) =\n", Ct)

if __name__ == "__main__":
    main()