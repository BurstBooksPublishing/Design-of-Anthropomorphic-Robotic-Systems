import cvxpy as cp
import numpy as np

def create_momentum_optimization_problem(r, m, g, Dt, cL, cU, fL, fU):
    """
    Create a convex optimization problem for momentum-based contact force planning.
    
    Parameters:
        r: contact point position (3,)
        m: mass
        g: gravity vector (3,)
        Dt: time step
        cL, cU: CoM bounds (3,)
        fL, fU: force bounds (3,)
    """
    # Decision variables
    c = cp.Variable(3, name="com")           # Center of mass position
    f = cp.Variable(3, name="force")         # Contact force
    z = cp.Variable((3, 3), name="products") # Bilinear products (r_j - c_j) * f_k
    
    # Linear momentum target (gravity compensation)
    p_target = m * g * Dt
    
    # Angular momentum target (assuming zero desired angular velocity change)
    L_target = np.zeros(3)
    
    # Constraints
    constraints = []
    
    # CoM bounds
    constraints.append(c >= cL)
    constraints.append(c <= cU)
    
    # Force bounds
    constraints.append(f >= fL)
    constraints.append(f <= fU)
    
    # McCormick envelopes for bilinear terms z[j,k] = (r[j] - c[j]) * f[k]
    for j in range(3):
        xL = r[j] - cU[j]  # Lower bound for (r[j] - c[j])
        xU = r[j] - cL[j]  # Upper bound for (r[j] - c[j])
        
        for k in range(3):
            yL, yU = fL[k], fU[k]
            
            # McCormick inequalities for bounding z[j,k]
            constraints.append(z[j, k] >= xL * f[k] + (r[j] - c[j]) * yL - xL * yL)
            constraints.append(z[j, k] >= xU * f[k] + (r[j] - c[j]) * yU - xU * yU)
            constraints.append(z[j, k] <= xU * f[k] + (r[j] - c[j]) * yL - xU * yL)
            constraints.append(z[j, k] <= xL * f[k] + (r[j] - c[j]) * yU - xL * yU)
    
    # Friction polytope (assuming isotropic friction cone: |f_x|, |f_y| <= mu * f_z)
    mu = 0.5  # Friction coefficient
    constraints.append(cp.abs(f[0]) <= mu * f[2])
    constraints.append(cp.abs(f[1]) <= mu * f[2])
    
    # Moment calculation: L = r Ã— f, computed using bilinear variables
    moment_constraints = [
        L_target[0] == r[1] * f[2] - r[2] * f[1] - (z[1, 2] - z[2, 1]),
        L_target[1] == r[2] * f[0] - r[0] * f[2] - (z[2, 0] - z[0, 2]),
        L_target[2] == r[0] * f[1] - r[1] * f[0] - (z[0, 1] - z[1, 0])
    ]
    constraints.extend(moment_constraints)
    
    # Objective: minimize deviation from target linear momentum
    objective = cp.Minimize(cp.sum_squares(f - p_target))
    
    # Create problem
    problem = cp.Problem(objective, constraints)
    return problem, c, f