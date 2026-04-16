import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Callable

# Physical parameters
C = np.array([200.0, 400.0, 50.0])  # Heat capacities (J/K): winding, housing, electronics
G = np.array([[ 5.0, -4.0, -1.0],   # Conductance matrix (W/K)
              [-4.0, 10.0, -6.0],
              [-1.0, -6.0,  7.0]])
T_AMB = 293.15                      # Ambient temperature (K)
R0_WINDING = 0.1                    # Winding resistance at T_AMB (ohm)
ALPHA_CU = 3.9e-3                   # Copper temperature coefficient (1/K)
I_RMS = 30.0                        # Operating RMS current (A)

def compute_losses(temperature: np.ndarray) -> np.ndarray:
    """Calculate power losses for each thermal node."""
    r_winding = R0_WINDING * (1 + ALPHA_CU * (temperature[0] - T_AMB))  # Temperature-dependent resistance
    p_winding = I_RMS**2 * r_winding                                    # Joule heating in winding
    p_electronics = 50.0                                                # Constant electronics loss
    return np.array([p_winding, 0.0, p_electronics])

def system_rhs(t: float, temperature: np.ndarray) -> np.ndarray:
    """Compute time derivative of temperature vector."""
    power = compute_losses(temperature)
    # dT/dt = C^(-1) * (-G * T + power + G * T_amb)
    conductance_term = G @ (T_AMB - temperature)
    return (conductance_term + power) / C

# Initial conditions and simulation
initial_temperature = np.full(3, T_AMB)  # All nodes start at ambient
solution = solve_ivp(
    system_rhs,
    [0, 600],                         # Time span (seconds)
    initial_temperature,
    rtol=1e-6,
    atol=1e-8,
    method='RK45'                     # Robust explicit solver
)