import numpy as np
from scipy.stats import chi2
from typing import Tuple, Union

def log_likelihood_ratio(y: np.ndarray, 
                        w_hat: np.ndarray, 
                        Sigma: np.ndarray, 
                        Sigma_nc: np.ndarray) -> float:
    """
    Compute log likelihood ratio for contact detection.
    """
    r = y - w_hat
    
    # Compute quadratic forms using Cholesky decomposition for numerical stability
    L_Sigma = np.linalg.cholesky(Sigma)
    L_Sigma_nc = np.linalg.cholesky(Sigma_nc)
    
    # Solve linear systems efficiently
    Sigma_inv_r = np.linalg.solve(L_Sigma.T, np.linalg.solve(L_Sigma, r))
    Sigma_nc_inv_y = np.linalg.solve(L_Sigma_nc.T, np.linalg.solve(L_Sigma_nc, y))
    
    term1 = -0.5 * r.T @ Sigma_inv_r
    term2 = 0.5 * y.T @ Sigma_nc_inv_y
    term3 = 0.5 * (2 * (np.sum(np.log(np.diag(L_Sigma_nc))) - np.sum(np.log(np.diag(L_Sigma)))))
    
    return term1 + term2 + term3

class ContactFilter:
    def __init__(self, m: int, alpha_FA: float = 1e-3, p11: float = 0.999, p01: float = 0.01):
        """
        Initialize contact filter.
        
        Args:
            m: Dimension of measurement space
            alpha_FA: False alarm probability
            p11: Probability of staying in contact state
            p01: Probability of transitioning from no-contact to contact
        """
        self.logodds = 0.0
        self.p11, self.p01 = p11, p01
        self.gamma = chi2.ppf(1 - alpha_FA, df=m)  # Mahalanobis threshold
        
    def update(self, y: np.ndarray, 
               w_hat: np.ndarray, 
               Sigma: np.ndarray, 
               Sigma_nc: np.ndarray) -> Tuple[bool, float]:
        """
        Update contact state estimate based on new measurement.
        
        Returns:
            contact: Boolean contact decision
            logodds: Current log-odds of contact
        """
        llr = log_likelihood_ratio(y, w_hat, Sigma, Sigma_nc)
        
        # Incorporate transition prior via additive log-odds term
        prior_odds = np.log(self.p11 / (1 - self.p11))
        self.logodds = self.logodds + llr + prior_odds
        
        # MAP decision using both log-odds and Mahalanobis distance
        r = y - w_hat
        L_Sigma = np.linalg.cholesky(Sigma)
        mahalanobis_dist_sq = np.sum(np.linalg.solve(L_Sigma, r) ** 2)
        contact = (self.logodds > 0) or (mahalanobis_dist_sq > self.gamma)
        
        return contact, self.logodds