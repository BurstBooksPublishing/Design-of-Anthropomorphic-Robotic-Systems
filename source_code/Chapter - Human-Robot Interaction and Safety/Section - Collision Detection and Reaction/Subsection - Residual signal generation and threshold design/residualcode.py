import numpy as np
from scipy.stats import chi2
from typing import Tuple, Union

def mahalanobis_energy(r: np.ndarray, Sigma: np.ndarray) -> float:
    """
    Compute Mahalanobis distance squared (energy) between residual and zero mean.
    
    Args:
        r: Residual vector of shape (n,)
        Sigma: Covariance matrix of shape (n, n)
        
    Returns:
        Mahalanobis distance squared (scalar)
    """
    # Add regularization for numerical stability
    Sigma_reg = Sigma + 1e-9 * np.eye(Sigma.shape[0])
    inv_Sigma = np.linalg.inv(Sigma_reg)
    return float(r.T @ inv_Sigma @ r)

def chi2_threshold(degrees_of_freedom: int, false_alarm_rate: float) -> float:
    """
    Compute chi-squared threshold for given false alarm rate.
    
    Args:
        degrees_of_freedom: Number of dimensions
        false_alarm_rate: Probability of false alarm (0 < P_FA < 1)
        
    Returns:
        Chi-squared threshold value
    """
    if not (0 < false_alarm_rate < 1):
        raise ValueError("False alarm rate must be between 0 and 1")
    return chi2.ppf(1.0 - false_alarm_rate, df=degrees_of_freedom)

def collision_detection(r: np.ndarray, 
                       Sigma: np.ndarray, 
                       false_alarm_rate: float = 1e-4) -> Tuple[float, float, bool]:
    """
    Perform collision detection using Mahalanobis distance and chi-squared test.
    
    Args:
        r: Residual vector
        Sigma: Covariance matrix
        false_alarm_rate: Desired false alarm rate
        
    Returns:
        Tuple of (test_statistic, threshold, collision_detected)
    """
    if r.shape[0] != Sigma.shape[0]:
        raise ValueError("Residual vector and covariance matrix dimension mismatch")
    
    test_statistic = mahalanobis_energy(r, Sigma)
    threshold = chi2_threshold(Sigma.shape[0], false_alarm_rate)
    collision_detected = test_statistic > threshold
    
    return test_statistic, threshold, collision_detected

# Example usage
if __name__ == "__main__":
    r = np.array([0.1, -0.05, 0.0, 0.02, 0.01, -0.03])  # Residual wrench
    Sigma = np.diag([0.01] * 6)                          # Covariance matrix
    T, gamma, collision = collision_detection(r, Sigma, 1e-4)