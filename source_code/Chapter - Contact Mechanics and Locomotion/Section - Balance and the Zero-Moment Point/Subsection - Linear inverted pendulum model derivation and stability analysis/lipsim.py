import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Callable

# Physical parameters
g: float = 9.81  # gravitational acceleration (m/s^2)
h: float = 0.9   # height (m)
omega: float = np.sqrt(g/h)  # natural frequency

def zmp(t: float) -> float:
    """Zero Moment Point trajectory: step function at t=0.2s"""
    return 0.0 if t < 0.2 else 0.05

def lip_ode(t: float, z: np.ndarray) -> np.ndarray:
    """Linear Inverted Pendulum dynamics: x'' = Ï‰Â²(x - zmp(t))"""
    x, xd = z
    xdd = omega**2 * (x - zmp(t))
    return np.array([xd, xdd])

def simulate_lip() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate LIP dynamics and compute capture point trajectory"""
    z0 = np.array([0.02, 0.5])  # initial state [position, velocity]
    t_span = (0.0, 1.0)         # simulation time span
    
    # Solve ODE with high temporal resolution
    sol = solve_ivp(lip_ode, t_span, z0, max_step=1e-3)
    
    if not sol.success:
        raise RuntimeError("ODE integration failed")
    
    x, xd = sol.y[0], sol.y[1]
    # Capture point: x + x'/Ï‰ (predicted final position)
    xi = x + xd/omega
    
    return x, xd, sol.t, xi

# Execute simulation
x, xd, t, xi = simulate_lip()