)

import numpy as np
import control as ctrl

def passive_reference_test() -> bool:
    """
    Test if the system satisfies passivity (positive realness) condition.
    A system is passive if its frequency response has non-negative real part.
    """
    # Physical parameters
    Jm, bm, ks = 0.02, 0.01, 1000.0  # motor inertia, damping, spring stiffness
    
    # Define Laplace variable for transfer function construction
    s = ctrl.TransferFunction.s
    
    # Plant transfer function: maps velocity to spring torque (linearized dynamics)
    plant = ks / (Jm * s**2 + bm * s)
    
    # Virtual impedance controller parameters
    Kv, Bv = 50.0, 4.0  # virtual stiffness and damping
    virtual_impedance = Bv + Kv / s  # virtual impedance transfer function
    
    # Combined system impedance (plant and controller in series)
    total_impedance = ctrl.minreal(plant * virtual_impedance)
    
    # Frequency range for analysis (logarithmic spacing)
    frequencies = np.logspace(-1, 3, 1000)
    
    # Evaluate impedance at each frequency point
    impedance_response = np.array([
        ctrl.evalfr(total_impedance, 1j * omega) for omega in frequencies
    ])
    
    # Passivity test: real part must be non-negative (with numerical tolerance)
    tolerance = 1e-6
    is_passive = np.all(np.real(impedance_response) >= -tolerance)
    
    return is_passive

if __name__ == "__main__":
    result = passive_reference_test()
    print(f"PR test: {result}")