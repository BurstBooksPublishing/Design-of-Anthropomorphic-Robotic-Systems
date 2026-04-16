import numpy as np
from scipy.stats import chi2

def slip_detector(f_t, f_n, mu_hat, sigma_mu, Rf, v_t, sigma_v, alpha=0.01):
    """
    Detect slip based on friction model residual and kinematic consistency.
    
    Args:
        f_t: Tangential force vector (2,)
        f_n: Normal force (scalar)
        mu_hat: Friction coefficient estimate
        sigma_mu: Uncertainty in friction coefficient
        Rf: Force covariance matrix (3x3 for [f_tx, f_ty, f_n])
        v_t: Tangential velocity (scalar)
        sigma_v: Velocity measurement uncertainty
        alpha: Significance level for slip detection
    
    Returns:
        slip_prob: Probability of slip occurrence
        trigger: Boolean slip detection flag
    """
    # Compute friction model residual
    ft_norm = np.linalg.norm(f_t)
    r = ft_norm - mu_hat * f_n
    
    # Compute residual gradient for uncertainty propagation
    if ft_norm > 1e-6:
        dr_dft = f_t / ft_norm
    else:
        dr_dft = np.array([0.0, 0.0])
    
    dr_df = np.hstack([dr_dft, -mu_hat])
    
    # Propagate uncertainty through linearized model
    sigma_r2 = dr_df @ Rf @ dr_df + (f_n**2) * (sigma_mu**2)
    
    # Normalize residuals and compute Mahalanobis distance
    z = np.array([r / np.sqrt(sigma_r2), v_t / sigma_v])
    D2 = z @ z
    
    # Compute detection threshold and slip probability
    threshold = chi2.ppf(1.0 - alpha, 2)
    slip_prob = 1.0 / (1.0 + np.exp(-0.8 * (np.sqrt(D2) - np.sqrt(threshold))))
    trigger = D2 > threshold
    
    return slip_prob, trigger