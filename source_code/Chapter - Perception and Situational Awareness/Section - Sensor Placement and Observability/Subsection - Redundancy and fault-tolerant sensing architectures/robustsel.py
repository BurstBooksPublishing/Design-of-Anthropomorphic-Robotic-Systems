import numpy as np
from typing import List, Dict, Tuple, Any, Optional

# Global variables (should be passed as parameters in production)
n: int = 0  # dimension of state space
sensors: Dict[Any, Dict[str, np.ndarray]] = {}  # sensor data
m: int = 0  # number of sensors to select
candidate_pool: List[Any] = []  # available sensors
failures_sample: List[Any] = []  # possible failure scenarios

def fim(sensor_set: List[Any]) -> np.ndarray:
    """Compute Fisher Information Matrix for given sensor set."""
    I = np.zeros((n, n))
    for s in sensor_set:
        J, R = sensors[s]['J'], sensors[s]['R']
        I += J.T @ np.linalg.inv(R) @ J
    return I

def worst_logdet(selected: List[Any], failures: List[Any]) -> float:
    """Compute worst-case log determinant after single sensor failures."""
    worst = np.inf
    for f in failures:
        # Simulate failure by removing sensor f
        T = [s for s in selected if s != f]
        I = fim(T)
        # Add regularization to prevent log(0)
        det_val = np.linalg.det(I) + 1e-12
        worst = min(worst, np.log(max(det_val, 1e-12)))
    return worst

def greedy_sensor_selection() -> List[Any]:
    """Select m sensors using greedy worst-case logdet optimization."""
    selected = []
    for _ in range(m):
        best = None
        best_val = -np.inf
        for cand in candidate_pool:
            if cand in selected:
                continue
            val = worst_logdet(selected + [cand], failures_sample)
            if val > best_val:
                best_val = val
                best = cand
        if best is None:
            break
        selected.append(best)
    return selected

# Execute selection
selected_sensors = greedy_sensor_selection()