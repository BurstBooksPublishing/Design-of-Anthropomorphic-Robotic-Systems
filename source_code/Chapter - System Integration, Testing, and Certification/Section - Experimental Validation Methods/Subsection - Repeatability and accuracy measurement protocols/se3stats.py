\begin{lstlisting}[language=Python,caption={Compute aggregated benchmark and bootstrap CI},label={lst:benchmark}]

import numpy as np
from typing import List, Dict, Tuple, Union

def normalize(value: float, p10: float, p90: float, invert: bool = False) -> float:
    """Normalize value using percentile-based mapping. Maps p10->0, p90->1, then clips to [0,1]."""
    normalized = (value - p10) / (p90 - p10 + 1e-12)
    clipped = np.clip(normalized, 0.0, 1.0)
    return 1.0 - clipped if invert else clipped

def compute_performance_scores(trials: List[Dict[str, float]]) -> np.ndarray:
    """Calculate normalized performance scores for each trial using weighted metrics."""
    weights = np.array([0.4, 0.4, 0.2])
    scores = []
    
    for trial in trials:
        cot_score = normalize(trial['cot'], p10=0.3, p90=1.2, invert=True)
        recovery_score = normalize(trial['recovery_frac'], p10=0.0, p90=1.0)
        foot_var_score = normalize(trial['foot_var'], p10=0.01, p90=0.1, invert=True)
        scores.append(weights.dot([cot_score, recovery_score, foot_var_score]))
    
    return np.array(scores)

def bootstrap_confidence_interval(
    data: np.ndarray, 
    n_bootstrap: int = 2000, 
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    alpha = 1.0 - confidence_level
    percentiles = [100 * alpha/2, 100 * (1 - alpha/2)]
    
    bootstrap_means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True)) 
        for _ in range(n_bootstrap)
    ])
    
    return tuple(np.percentile(bootstrap_means, percentiles))

# Main processing pipeline
trials: List[Dict[str, float]] = [...]  # Loaded experimental traces

# Calculate performance scores
P_vals = compute_performance_scores(trials)
P_mean = P_vals.mean()

# Compute confidence interval
ci_lower, ci_upper = bootstrap_confidence_interval(P_vals, n_bootstrap=2000)

# Output: P_mean, (ci_lower, ci_upper)