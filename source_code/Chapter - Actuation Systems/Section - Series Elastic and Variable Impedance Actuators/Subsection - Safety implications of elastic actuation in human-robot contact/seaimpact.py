import numpy as np

def peak_force_bound(m, v, k):
    """
    Calculate conservative bound for peak impact force.
    
    Args:
        m: mass (kg)
        v: impact velocity (m/s)  
        k: stiffness (N/m)
        
    Returns:
        Peak force bound (N)
    """
    return v * np.sqrt(m * k)

def simulate_impact(m, v, k, c, dt=1e-4, T=0.5):
    """
    Simulate impact with spring-damper system.
    
    Args:
        m: mass (kg)
        v: impact velocity (m/s)
        k: stiffness (N/m)  
        c: damping coefficient (N*s/m)
        dt: time step (s)
        T: maximum simulation time (s)
        
    Returns:
        Maximum impact force (N)
    """
    # Initial conditions: relative position and velocity
    x = 0.0
    xdot = -v  # Negative velocity towards contact surface
    
    F_hist = []
    t = 0.0
    
    while t < T:
        # Calculate contact force (spring + damper)
        F = -k * x - c * xdot
        F_hist.append(F)
        
        # Update dynamics: F = ma => a = F/m
        xddot = F / m
        
        # Numerical integration (velocity-Verlet)
        xdot += xddot * dt
        x += xdot * dt
        
        # Exit conditions: rebound completed
        if x > 0 and xdot > 0:
            break
            
        t += dt
    
    return max(F_hist) if F_hist else 0.0

# Example usage
m, v, k, c = 2.0, 1.0, 500.0, 20.0
print("bound", peak_force_bound(m, v, k))
print("simulated Fmax", simulate_impact(m, v, k, c))