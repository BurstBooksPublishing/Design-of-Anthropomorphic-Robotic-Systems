import numpy as np
from typing import Union

def iv_two_stage(y: np.ndarray, Phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Two-stage instrumental variables estimator for linear regression.
    
    Solves y = Phi * theta + epsilon using instruments Z to address endogeneity.
    """
    # Validate input dimensions
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    N, P = Phi.shape
    M = Z.shape[1]
    
    # First-stage: project regressors onto instruments
    ZZ = Z.T @ Z
    # Check condition number to ensure numerical stability
    cond_num = np.linalg.cond(ZZ)
    if cond_num > 1e12:
        raise np.linalg.LinAlgError(f"Ill-conditioned matrix (cond={cond_num:.2e})")
    
    Pi = np.linalg.solve(ZZ, Z.T @ Phi)     # Projection coefficients (MxP)
    Phi_hat = Z @ Pi                        # Predicted regressors (NxP)
    
    # Second-stage: IV estimator using predicted regressors
    A = Phi_hat.T @ Phi
    b = Phi_hat.T @ y
    theta_hat = np.linalg.solve(A, b)
    
    return theta_hat.reshape(-1)