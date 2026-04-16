\begin{lstlisting}[language=Python,caption={Compute minimal redundancy and admissible velocity given E_lim and safety distance.},label={lst:pl_calc}]

import math

def calculate_safety_parameters(M, E_lim, a_max, T_r, d_safe, lambda_ch, PFD_req=1e-5, T_check=3600.0):
    """
    Calculate safety parameters for mechanical system design.
    
    Args:
        M: effective mass (kg)
        E_lim: impact energy limit (J)
        a_max: maximum deceleration (m/s^2)
        T_r: reaction time (s)
        d_safe: allowable stopping distance (m)
        lambda_ch: failure rate (failures per hour)
        PFD_req: required probability of failure on demand
        T_check: demand interval (s)
    
    Returns:
        tuple: (v_max, v_req, n) where n is minimum redundancy
    """
    # Convert failure rate to per second
    lambda_ch_per_sec = lambda_ch / 3600.0
    
    # Maximum velocity from impact energy constraint
    v_max = math.sqrt(2 * E_lim / M)
    
    # Maximum velocity from stopping distance constraint
    # Solving quadratic: d = v*T_r + v^2/(2*a_max) <= d_safe
    A = 1.0 / (2 * a_max)
    B = T_r
    C = -d_safe
    discriminant = B * B - 4 * A * C
    
    if discriminant < 0:
        raise ValueError("No real solution for velocity constraint")
    
    v_req = (-B + math.sqrt(discriminant)) / (2 * A)
    
    # Calculate minimum redundancy for PFD target
    p = lambda_ch_per_sec * T_check
    n = 1
    
    while (p ** n) / math.factorial(n) > PFD_req:
        n += 1
        if n > 100:  # Prevent infinite loop
            raise ValueError("Cannot achieve required PFD with reasonable redundancy")
    
    return v_max, v_req, n

# Example usage with provided values
if __name__ == "__main__":
    # Input parameters
    M = 3.5         # kg, effective mass
    E_lim = 10.0    # J, impact energy limit
    a_max = 15.0    # m/s^2, max decel
    T_r = 0.08      # s, reaction time
    d_safe = 0.25   # m, allowable stopping distance
    lambda_ch = 1e-6  # per hour -> convert to per second
    
    # Calculate safety parameters
    v_max, v_req, n = calculate_safety_parameters(
        M, E_lim, a_max, T_r, d_safe, lambda_ch
    )
    
    # Output results
    print(f"Maximum velocity (energy constraint): {v_max:.3f} m/s")
    print(f"Maximum velocity (stopping distance): {v_req:.3f} m/s")
    print(f"Minimum redundancy required: {n}")