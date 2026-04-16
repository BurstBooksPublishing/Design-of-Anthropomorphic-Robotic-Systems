import numpy as np

def calculate_actuator_performance() -> None:
    """Calculate and compare hydraulic and tendon-driven actuator performance metrics."""
    
    # Physical parameters (SI units)
    tau_req: float = 150.0            # Required torque
    m_h: float = 4.0                  # Hydraulic actuator mass
    omega_v: float = 120.0            # Valve cutoff frequency
    K_h: float = 1.0e4                # Hydraulic stiffness
    I_load: float = 0.05              # Load inertia
    
    m_t: float = 2.0                  # Tendon actuator mass
    K_t: float = 1.0e6                # Tendon stiffness
    r: float = 0.03                   # Pulley radius
    I_ref: float = 0.02               # Reference inertia
    
    # Performance calculations
    D_h: float = tau_req / m_h                           # Hydraulic torque-to-mass ratio
    omega_h: float = min(omega_v, np.sqrt(K_h / I_load)) # Hydraulic bandwidth limit
    K_tau: float = K_t * r**2                            # Effective rotational stiffness
    D_t: float = tau_req / m_t                           # Tendon torque-to-mass ratio
    omega_t: float = np.sqrt(K_tau / I_ref)              # Tendon bandwidth
    
    # Output results
    print(f"Hydraulic: D={D_h:.1f} Nm/kg, omega_c_bound={omega_h:.1f} rad/s")
    print(f"Tendon:   D={D_t:.1f} Nm/kg, omega_c_bound={omega_t:.1f} rad/s")

if __name__ == "__main__":
    calculate_actuator_performance()