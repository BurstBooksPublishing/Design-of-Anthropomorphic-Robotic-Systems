import numpy as np
from scipy.linalg import expm, eig

def thermal_network_analysis():
    # Network parameters
    C = np.diag([10., 50., 200.])  # Heat capacities (J/K)
    K = np.array([[ 2., -2.,  0.],   # Conductance matrix (W/K)
                  [-2.,  3., -1.],
                  [ 0., -1.,  6.]])
    P = np.array([20., 0., 0.])      # Internal power generation (W)
    g_amb = np.array([0., 0., 5.])   # Conductance to ambient (W/K)
    T_a = 25.                        # Ambient temperature (Â°C)
    
    # Steady-state temperature calculation
    w = P + g_amb * T_a              # Total heat input vector
    T_inf = np.linalg.solve(K, w)    # Steady-state temperatures
    
    # Modal analysis for transient behavior
    A = -np.linalg.solve(C, K)       # System matrix: A = -C^(-1)K
    eigvals, eigvecs = eig(A)
    # Filter out near-zero eigenvalues to avoid division by zero
    valid_eigenvals = eigvals[np.abs(eigvals) > 1e-12]
    time_constants = -1.0 / np.real(valid_eigenvals)  # Thermal time constants
    
    # Transient temperature calculation
    T0 = np.array([25., 25., 25.])   # Initial temperatures (Â°C)
    t = 60.                          # Time for transient analysis (s)
    delta_T = T0 - T_inf             # Temperature difference from steady-state
    Tt = T_inf + expm(A * t) @ delta_T  # Transient solution
    
    return T_inf, time_constants, Tt

# Execute analysis
T_inf, time_constants, Tt = thermal_network_analysis()