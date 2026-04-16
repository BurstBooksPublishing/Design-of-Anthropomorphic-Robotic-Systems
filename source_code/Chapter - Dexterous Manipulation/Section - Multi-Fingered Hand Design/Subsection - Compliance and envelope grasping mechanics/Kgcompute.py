import numpy as np
from typing import Tuple

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from 3D vector."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def contact_frame(theta: float) -> np.ndarray:
    """Generate contact frame rotation matrix (tangent, normal, axis)."""
    tangent = np.array([-np.sin(theta), np.cos(theta), 0.0])
    normal = np.array([np.cos(theta), np.sin(theta), 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    return np.column_stack((tangent, normal, axis))

def compute_grasp_stiffness_matrix(
    radius: float, 
    thetas: np.ndarray, 
    k_t: float, 
    k_n: float, 
    k_o: float
) -> np.ndarray:
    """Compute 6x6 grasp stiffness matrix for circular grasp."""
    Kg = np.zeros((6, 6))
    contact_stiffness = np.diag([k_t, k_n, k_o])
    
    for theta in thetas:
        # Contact point position
        r_i = np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
        
        # Jacobian matrix (translation + rotation part)
        J = np.hstack((np.eye(3), skew_symmetric(r_i)))
        
        # Contact frame transformation
        R = contact_frame(theta)
        
        # Accumulate stiffness contribution
        Kg += J.T @ R @ contact_stiffness @ R.T @ J
    
    return Kg

def main() -> None:
    # Configuration parameters
    radius = 0.03
    thetas = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    k_t, k_n, k_o = 1e4, 1e6, 1e4  # Stiffness coefficients (tangential, normal, orientation)
    
    # Compute grasp stiffness matrix
    Kg = compute_grasp_stiffness_matrix(radius, thetas, k_t, k_n, k_o)
    
    # Spectral analysis
    eigvals = np.linalg.eigvalsh(Kg)
    is_positive_definite = np.all(eigvals > 1e-8)
    
    print("eigenvalues:", np.round(eigvals, 6))
    print("positive_definite:", is_positive_definite)

if __name__ == "__main__":
    main()