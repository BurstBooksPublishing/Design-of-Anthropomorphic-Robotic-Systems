import numpy as np
import scipy.special
from typing import List, Tuple, Callable

def posterior_score(xi: np.ndarray, mus: List[np.ndarray], prior: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute posterior distribution over goals given trajectory xi.
    
    Args:
        xi: Observed trajectory (T x d)
        mus: List of goal trajectories, each (T x d)
        prior: Prior probabilities for each goal
        sigma: Covariance scalar for Gaussian likelihood
    
    Returns:
        Posterior distribution over goals
    """
    # Compute squared distances for each goal trajectory
    diffs = np.array([np.sum((xi - mu) ** 2) for mu in mus])
    
    # Calculate log-likelihood and log-posterior
    loglik = -0.5 * diffs / (sigma ** 2)
    logpost = loglik + np.log(prior)
    
    # Normalize using log-sum-exp for numerical stability
    logpost -= scipy.special.logsumexp(logpost)
    
    return np.exp(logpost)

def select_most_legible_trajectory(
    generate_candidates: Callable[[], List[np.ndarray]],
    mus: List[np.ndarray], 
    prior: np.ndarray, 
    sigma: float,
    true_goal_idx: int = 0
) -> Tuple[np.ndarray, float]:
    """
    Select the most legible trajectory from candidates.
    
    Args:
        generate_candidates: Function that generates candidate trajectories
        mus: Goal trajectories
        prior: Prior probabilities
        sigma: Covariance parameter
        true_goal_idx: Index of the true goal for legibility evaluation
    
    Returns:
        Tuple of (best_trajectory, best_score)
    """
    candidates = generate_candidates()
    
    # Evaluate legibility for each candidate
    scores = [posterior_score(c, mus, prior, sigma)[true_goal_idx] for c in candidates]
    
    # Select maximally legible candidate
    best_idx = np.argmax(scores)
    
    return candidates[best_idx], scores[best_idx]