import cvxpy as cp
import numpy as np

def motor_sizing_optimization(theta_ddot, theta_dot, tau_ext, Hdiag, Cvec, gvec):
    """
    Optimize motor sizing considering torque-speed constraints and thermal limits.
    
    Args:
        theta_ddot: Joint acceleration samples (N,)
        theta_dot: Joint velocity samples (N,)
        tau_ext: External torque samples (N,)
        Hdiag: Joint inertia diagonal elements (N,) or scalar
        Cvec: Coriolis/centrifugal torque (N,)
        gvec: Gravity torque (N,)
    
    Returns:
        tuple: (optimal_continuous_torque, problem_status, motor_mass)
    """
    N = len(theta_dot)
    r = 40.0                      # gear ratio
    tau_stall = 1.0               # stall torque (Nm)
    omega_nl = 2 * np.pi * 100.0  # no-load speed (rad/s)
    
    # Compute required joint torque for each sample
    tau_joint_req = Hdiag * theta_ddot + Cvec + gvec - tau_ext
    tau_m_samples = tau_joint_req / r  # motor torque requirements
    
    # Continuous torque bound variable (RMS constraint)
    tauc = cp.Variable(nonneg=True)
    
    # Torque-speed constraint: tau_m <= tau_stall * (1 - omega_m/omega_nl)
    constraints = []
    for i in range(N):
        omegam = r * theta_dot[i]
        constraints.append(tau_m_samples[i] <= tau_stall * (1 - omegam / omega_nl))
    
    # Thermal constraint: RMS(tau_m) <= tauc
    constraints.append(cp.norm(tau_m_samples, 2) <= cp.sqrt(N) * tauc)
    
    # Objective: minimize motor mass (linear in continuous torque)
    alpha = 0.12  # kg/Nm
    beta = 0.2    # base mass
    objective = cp.Minimize(alpha * tauc + beta)
    
    # Solve optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)
    
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(f"Optimization failed with status: {problem.status}")
    
    motor_mass = alpha * tauc.value + beta
    return tauc.value, problem.status, motor_mass