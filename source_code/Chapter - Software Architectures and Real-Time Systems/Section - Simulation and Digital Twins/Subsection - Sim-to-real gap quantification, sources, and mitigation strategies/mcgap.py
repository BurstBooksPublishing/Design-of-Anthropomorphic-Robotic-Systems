import numpy as np
from typing import Callable, Tuple, Any

def simulate(f: Callable, x0: np.ndarray, policy: Callable, theta: Any, 
             dt: float, T: float) -> Tuple[np.ndarray, float]:
    """
    Simulate discrete-time system using forward Euler integration.
    """
    x = x0.copy()
    traj = [x.copy()]
    cost = 0.0
    
    for _ in range(int(T/dt)):
        u = policy(x)                           # state-feedback control
        x = x + dt * f(x, u, theta)            # forward Euler integration
        cost += dt * l(x, u)                   # accumulate instantaneous cost
        traj.append(x.copy())
    
    return np.array(traj), cost

def f(x: np.ndarray, u: np.ndarray, theta: Any) -> np.ndarray:
    """Linear system dynamics with parameter-dependent matrices."""
    return A(theta) @ x + B(theta) @ u

def l(x: np.ndarray, u: np.ndarray) -> float:
    """Quadratic cost function."""
    return float(x.T @ Q @ x + u.T @ R @ u)

def estimate_gap(x0: np.ndarray, policy: Callable, theta_real: np.ndarray, 
                 theta_sim: np.ndarray, n_samples: int = 100) -> float:
    """
    Estimate performance gap between real and simulated parameters using Monte Carlo.
    """
    costs_real = []
    costs_sim = []
    
    for _ in range(n_samples):
        # Sample perturbed parameters around simulator model
        theta_i = theta_sim + np.random.normal(scale=0.01, size=theta_sim.shape)
        
        _, c_sim = simulate(f, x0, policy, theta_i, dt=0.001, T=2.0)
        _, c_real = simulate(f, x0, policy, theta_real, dt=0.001, T=2.0)
        
        costs_sim.append(c_sim)
        costs_real.append(c_real)
    
    return float(np.mean(np.abs(np.array(costs_real) - np.array(costs_sim))))

# Placeholder functions that must be defined by user
def A(theta): raise NotImplementedError("Define system matrix A")
def B(theta): raise NotImplementedError("Define system matrix B")
Q = np.eye(1)  # Default identity cost matrix
R = np.eye(1)  # Default identity cost matrix