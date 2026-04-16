import cvxpy as cp
import numpy as np

# Design variables: actuator torque limits and series stiffness (continuous relaxation)
tau_max = cp.Variable(name='tau_max')  # peak actuator torque (Nm)
k_se = cp.Variable(name='k_se')        # series elastic stiffness (Nm/rad)

# Normative constants (from standards and ID)
F_max = 40.0              # N, max allowed contact force (ISO/TS 15066)
a_tau = 0.5               # N per Nm mapping (contact model)
c_geom = 0.0              # offset

# Functional safety mapping: diagnostic margin mapped to tau_max upper bound
tau_safety_upper = 100.0  # Nm from SIL analysis

# EMC envelope modeled conservatively as bound on switching frequency energy proxy
emc_energy_upper = 1.0
emc_proxy = 0.01 * tau_max + 0.001 * k_se

# Define constraints (convex)
constraints = [
    a_tau * tau_max + c_geom <= F_max,      # contact force constraint
    tau_max >= 0.0,                         # non-negative torque
    tau_max <= tau_safety_upper,            # SIL-derived limit
    emc_proxy <= emc_energy_upper           # EMC proxy constraint
]

# Formulate and solve feasibility problem
prob = cp.Problem(cp.Minimize(0), constraints)
print("Solving feasibility...")
status = prob.solve()

print(f"Problem status: {prob.status}")
if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
    print(f"Feasible solution found:")
    print(f"  tau_max = {tau_max.value:.4f} Nm")
    print(f"  k_se = {k_se.value:.4f} Nm/rad")
else:
    print("No feasible solution exists")