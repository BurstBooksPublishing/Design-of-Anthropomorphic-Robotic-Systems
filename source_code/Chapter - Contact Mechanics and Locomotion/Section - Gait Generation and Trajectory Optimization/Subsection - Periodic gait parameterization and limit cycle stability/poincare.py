import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Optional

def flow(x0: np.ndarray, params: dict, Tmax: float = 10.0) -> np.ndarray:
    """
    Integrate closed-loop dynamics until next section crossing.
    Returns post-impact state x_plus.
    """
    # Integrate hybrid dynamics with event detection
    sol = solve_ivp(
        lambda t, x: closed_loop(x, params), 
        [0, Tmax], 
        x0, 
        events=section_event,
        rtol=1e-9,
        atol=1e-12
    )
    
    # Handle case where no event is detected
    if len(sol.t_events[0]) == 0:
        raise RuntimeError("No section crossing detected within integration window")
    
    x_minus = sol.y[:, -1]
    return impact_map(x_minus, params)

def poincare_map(x0: np.ndarray, params: dict) -> np.ndarray:
    """Poincare map: flow from section crossing to next section crossing."""
    return flow(x0, params)

def jacobian_poincare(x0: np.ndarray, params: dict, eps: float = 1e-6) -> np.ndarray:
    """
    Compute Jacobian of Poincare map using finite differences.
    """
    n = x0.size
    P0 = poincare_map(x0, params)
    J = np.zeros((n, n))
    
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J[:, i] = (poincare_map(x0 + dx, params) - P0) / eps
    
    return J

# Note: The following functions must be defined elsewhere:
# - closed_loop(x, params): returns time derivative dx/dt
# - section_event(t, x): event function that is zero at section crossing
# - impact_map(x_minus, params): maps pre-impact to post-impact state
# - Tmax: maximum integration time (should be passed as parameter)