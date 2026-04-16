import cvxpy as cp
import numpy as np

def solve_contact_forces(Aeq, b, mu=0.6):
    """
    Solve for contact forces satisfying equilibrium and friction constraints.
    
    Args:
        Aeq: (6 x 6) centroidal wrench map matrix
        b: (6,) desired net wrench vector
        mu: friction coefficient
    
    Returns:
        tuple: (fL, fR) contact force vectors or (None, None) if infeasible
    """
    # Decision variables for left and right foot contact forces
    fL = cp.Variable(3, name="fL")  # [normal, tangent1, tangent2]
    fR = cp.Variable(3, name="fR")
    F = cp.vstack([fL, fR])
    
    # Objective: minimize squared forces (regularization)
    H = cp.eye(6)
    obj = cp.Minimize(0.5 * cp.quad_form(F, H))
    
    # Constraints: equilibrium, friction cones, and unilateral normal forces
    cons = [
        Aeq @ F == b,                                    # Linear equilibrium
        cp.norm(fL[1:3]) <= mu * fL[0], fL[0] >= 0,     # Left foot friction cone
        cp.norm(fR[1:3]) <= mu * fR[0], fR[0] >= 0      # Right foot friction cone
    ]
    
    # Solve the second-order cone program
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.MOSEK, verbose=False)
    
    # Check solution status and return results
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return None, None
        
    return fL.value, fR.value

# Example usage with placeholder data
if __name__ == "__main__":
    # Replace with actual system parameters
    Aeq = np.random.randn(6, 6)
    b = np.random.randn(6)
    fL_sol, fR_sol = solve_contact_forces(Aeq, b)