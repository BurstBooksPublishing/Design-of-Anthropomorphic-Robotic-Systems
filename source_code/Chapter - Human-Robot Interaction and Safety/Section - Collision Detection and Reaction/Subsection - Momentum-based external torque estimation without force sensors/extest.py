import numpy as np
from typing import Dict, Union, Tuple

def momentum_observer(
    x: Dict[str, np.ndarray],
    model: object,
    tau_meas: np.ndarray,
    Lambda: Union[float, np.ndarray],
    dt: float,
    h_prev: np.ndarray,
    w_g: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute external force and torque estimates using momentum observer.
    
    Args:
        x: State dictionary containing q, q_dot, q_ddot_est
        model: Model object providing A, M, C, g, S matrices
        tau_meas: Measured/estimated actuator torques
        Lambda: Observer gain (scalar or 6x6 diagonal matrix)
        dt: Time step
        h_prev: Previous centroidal momentum
        w_g: Gravitational wrench
    
    Returns:
        w_ext_hat: Estimated external wrench (6D)
        tau_ext_hat: Estimated external joint torques (n_a)
    """
    q, q_dot, q_ddot_est = x['q'], x['q_dot'], x['q_ddot_est']
    
    # Compute centroidal momentum and its estimate
    h = model.A(q) @ q_dot
    h_hat = h_prev + dt * (Lambda @ (h - h_prev))
    
    # Estimate external wrench from momentum derivative
    w_ext_hat = (h_hat - h_prev) / dt - w_g
    
    # Estimate external joint torques using rigid-body dynamics
    tau_ext_hat = tau_meas - (model.S @ (model.M(q) @ q_ddot_est + 
                                        model.C(q, q_dot) @ q_dot + 
                                        model.g(q)))
    
    return w_ext_hat, tau_ext_hat