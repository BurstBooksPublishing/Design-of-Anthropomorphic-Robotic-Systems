import numpy as np
from typing import Union, Optional

def expected_fisher(
    J: np.ndarray, 
    lam: np.ndarray, 
    pvec: np.ndarray, 
    alpha: float = 1.0, 
    beta: float = 0.0
) -> np.ndarray:
    """
    Compute expected Fisher information matrix for Poisson-Gaussian noise model.
    
    Args:
        J: Jacobian matrix of shape (M, n) where M is number of channels, n is parameter dimension
        lam: Expected photon counts per channel, shape (M,)
        pvec: Detection probabilities per channel, shape (M,)
        alpha: Poisson noise scaling factor
        beta: Gaussian noise baseline
    
    Returns:
        Expected Fisher information matrix of shape (n, n)
    """
    # Validate input shapes
    if not (J.ndim == 2 and lam.ndim == 1 and pvec.ndim == 1):
        raise ValueError("J must be 2D, lam and pvec must be 1D")
    if not (J.shape[0] == len(lam) == len(pvec)):
        raise ValueError("Dimension mismatch: J.shape[0] must equal len(lam) and len(pvec)")
    
    # Per-channel noise variance: alpha*lambda + beta
    var = alpha * lam + beta
    
    # Avoid division by zero
    var = np.maximum(var, 1e-12)
    
    # Compute Fisher information contribution for each channel
    # Fi = (pvec/var) * J^T * J for each channel, summed over all channels
    weights = pvec / var  # (M,)
    Fi = np.einsum('i,ij,ik->jk', weights, J, J)
    
    return Fi

# Example usage
if __name__ == "__main__":
    M, n = 100, 6
    J = np.random.randn(M, n)
    lam = np.maximum(1.0, 10.0 * np.random.rand(M))
    pvec = 0.6 * np.ones(M)
    Iexp = expected_fisher(J, lam, pvec, alpha=1.0, beta=2.0)