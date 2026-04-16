from gurobipy import Model, GRB, quicksum
import math

# Problem data (assumed to be defined externally)
# I: set of contact points
# N: number of time steps
# mass: mass of the object
# mu: friction coefficient
# M: big-M constant for linearization
# Delta_t: time step duration
# g_x, g_y, g_z: gravity components

def create_contact_model(I, N, mass, mu, M, Delta_t, g):
    """
    Create and solve a contact sequence optimization model.
    
    Returns:
        model: optimized Gurobi model
    """
    m = Model('contact_sequence')

    # Contact variables
    z = m.addVars(I, N, vtype=GRB.BINARY, name='z')            # contact activation
    fx = m.addVars(I, N, lb=-GRB.INFINITY, name='fx')          # friction force x-component
    fy = m.addVars(I, N, lb=-GRB.INFINITY, name='fy')          # friction force y-component
    fz = m.addVars(I, N, lb=0.0, name='fz')                    # normal force (>= 0)

    # Center of mass variables
    cx = m.addVars(N+1, lb=-GRB.INFINITY, name='cx')
    cy = m.addVars(N+1, lb=-GRB.INFINITY, name='cy')
    cz = m.addVars(N+1, lb=-GRB.INFINITY, name='cz')

    # Dynamics constraints (discrete integration)
    for k in range(N):
        # x-direction momentum balance
        m.addConstr(mass * (cx[k+1] - cx[k]) == Delta_t * (
            quicksum(fx[i, k] for i in I) + mass * g[0]))
        
        # y-direction momentum balance
        m.addConstr(mass * (cy[k+1] - cy[k]) == Delta_t * (
            quicksum(fy[i, k] for i in I) + mass * g[1]))
        
        # z-direction momentum balance
        m.addConstr(mass * (cz[k+1] - cz[k]) == Delta_t * (
            quicksum(fz[i, k] for i in I) + mass * g[2]))

    # Friction constraints with big-M formulation
    for k in range(N):
        for i in I:
            # Coulomb friction polyhedral approximation (4 linear constraints)
            m.addConstr(fx[i, k] <=  mu * fz[i, k] + M * (1 - z[i, k]))
            m.addConstr(-fx[i, k] <= mu * fz[i, k] + M * (1 - z[i, k]))
            m.addConstr(fy[i, k] <=  mu * fz[i, k] + M * (1 - z[i, k]))
            m.addConstr(-fy[i, k] <= mu * fz[i, k] + M * (1 - z[i, k]))
            
            # Force is zero when contact is inactive
            m.addConstr(fz[i, k] <= M * z[i, k])

    # Set objective (example: minimize total force magnitude)
    m.setObjective(
        quicksum(fx[i, k]*fx[i, k] + fy[i, k]*fy[i, k] + fz[i, k]*fz[i, k] 
                for i in I for k in range(N)),
        GRB.MINIMIZE
    )

    m.optimize()
    return m

# Example usage:
# model = create_contact_model(I, N, mass, mu, M, Delta_t, [g_x, g_y, g_z])