import numpy as np
from typing import Callable, List, Union

def bayes_update(prior: np.ndarray, likelihood_y_given_theta: np.ndarray) -> np.ndarray:
    """Compute posterior distribution using Bayes' rule."""
    unnorm = likelihood_y_given_theta * prior  # elementwise multiplication
    return unnorm / np.sum(unnorm)

def expected_information_gain(
    prior: np.ndarray, 
    action: Union[int, str], 
    sample_y: Callable, 
    likelihood_fn: Callable, 
    num_samples: int = 50
) -> float:
    """
    Estimate expected information gain (mutual information) for an action.
    
    Args:
        prior: Prior distribution over theta
        action: Action to evaluate
        sample_y: Function to sample observation given action
        likelihood_fn: Function returning p(y|theta,a) for given action and observation
        num_samples: Number of samples for Monte Carlo estimation
    """
    H_post = 0.0
    for _ in range(num_samples):
        y = sample_y(action)
        L = likelihood_fn(action, y)
        post = bayes_update(prior, L)
        # Add small epsilon to prevent log(0)
        H_post += -np.sum(post * np.log(post + 1e-12))
    H_post /= num_samples
    
    H_prior = -np.sum(prior * np.log(prior + 1e-12))
    return H_prior - H_post

def select_optimal_action(
    actions: List[Union[int, str]], 
    prior: np.ndarray, 
    sample_y: Callable, 
    likelihood_fn: Callable, 
    num_samples: int = 50
) -> Union[int, str]:
    """Select action that maximizes expected information gain."""
    gains = [
        expected_information_gain(prior, action, sample_y, likelihood_fn, num_samples)
        for action in actions
    ]
    return actions[np.argmax(gains)]