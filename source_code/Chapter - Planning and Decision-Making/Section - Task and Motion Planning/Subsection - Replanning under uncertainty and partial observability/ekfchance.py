import numpy as np
from scipy import stats
from typing import Tuple, Callable

def predict(mu: np.ndarray, 
           Sigma: np.ndarray, 
           u: np.ndarray,
           f: Callable[[np.ndarray, np.ndarray], np.ndarray], 
           F: np.ndarray, 
           W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict step of Extended Kalman Filter
    """
    mu_pred = f(mu, u)                # Nonlinear state prediction
    Sigma_pred = F @ Sigma @ F.T + W  # Predicted covariance with process noise
    return mu_pred, Sigma_pred

def update(Sigma_pred: np.ndarray, 
          H: np.ndarray, 
          R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update step of Extended Kalman Filter
    """
    S = H @ Sigma_pred @ H.T + R      # Innovation covariance
    K = Sigma_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
    Sigma_upd = (np.eye(Sigma_pred.shape[0]) - K @ H) @ Sigma_pred  # Updated covariance
    return K, Sigma_upd

def chance_satisfied(a: np.ndarray, 
                    mu: np.ndarray, 
                    Sigma: np.ndarray, 
                    alpha: float, 
                    b_value: float) -> bool:
    """
    Check if probabilistic constraint is satisfied with confidence level (1-alpha)
    """
    margin = np.sqrt(a.T @ Sigma @ a) * stats.norm.ppf(1 - alpha)  # Safety margin
    return (a.T @ mu + margin) <= b_value  # Chance constraint evaluation