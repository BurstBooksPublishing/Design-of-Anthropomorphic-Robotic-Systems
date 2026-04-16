import numpy as np
from typing import Tuple, Union

def bayes_update(b_theta: np.ndarray, 
                P_y_given_theta: np.ndarray, 
                y_idx: int) -> np.ndarray:
    """
    Compute Bayesian posterior distribution using Bayes' rule.
    
    Args:
        b_theta: Prior probability vector over discrete theta values
        P_y_given_theta: Likelihood matrix of shape (n_theta, n_y)
        y_idx: Index of observed outcome
        
    Returns:
        Posterior probability vector over theta
    """
    # Extract likelihood for observed outcome and compute unnormalized posterior
    likelihood = P_y_given_theta[:, y_idx]
    post_unnormalized = b_theta * likelihood
    
    # Normalize to obtain valid probability distribution
    posterior = post_unnormalized / post_unnormalized.sum()
    return posterior

def expected_kl(true_idx: int, 
               P_y_given_theta: np.ndarray) -> float:
    """
    Compute expected KL divergence between true model and average model.
    
    Args:
        true_idx: Index of true theta value
        P_y_given_theta: Likelihood matrix of shape (n_theta, n_y)
        
    Returns:
        Expected KL divergence
    """
    # Get true likelihood and average likelihood across all theta values
    p_true = P_y_given_theta[true_idx]
    p_avg = P_y_given_theta.mean(axis=0)
    
    # Compute KL divergence with numerical stability
    kl = np.sum(p_true * (np.log(p_true + 1e-12) - np.log(p_avg + 1e-12)))
    return kl