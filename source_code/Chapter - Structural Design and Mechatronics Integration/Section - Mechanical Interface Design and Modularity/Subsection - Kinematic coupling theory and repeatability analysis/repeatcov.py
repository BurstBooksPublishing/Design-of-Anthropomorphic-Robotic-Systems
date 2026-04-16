import numpy as np
from typing import Tuple

def compute_sensitivity_and_repeatability(
    A: np.ndarray, 
    B: np.ndarray, 
    Sigma_p: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute sensitivity map and RMS repeatability metrics from Jacobians and parameter covariance.
    
    Args:
        A: m x 6 constraint Jacobian
        B: m x k parameter Jacobian  
        Sigma_p: k x k parameter covariance matrix
        
    Returns:
        Tuple of (sensitivity_map, twist_covariance, trans_rms, rot_rms)
    """
    # Compute pseudoinverse robustly via SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_inv = np.diag(np.where(S > 1e-12, 1/S, 0.0))
    A_pinv = Vt.T @ S_inv @ U.T
    
    # Compute sensitivity map and propagated covariance
    J = -A_pinv @ B                            # sensitivity map xi = J delta_p
    Sigma_xi = J @ Sigma_p @ J.T               # propagated twist covariance
    
    # Extract RMS translational and rotational repeatability
    trans_rms = np.sqrt(np.trace(Sigma_xi[:3, :3]))  # meters
    rot_rms = np.sqrt(np.trace(Sigma_xi[3:, 3:]))    # radians
    
    return J, Sigma_xi, trans_rms, rot_rms