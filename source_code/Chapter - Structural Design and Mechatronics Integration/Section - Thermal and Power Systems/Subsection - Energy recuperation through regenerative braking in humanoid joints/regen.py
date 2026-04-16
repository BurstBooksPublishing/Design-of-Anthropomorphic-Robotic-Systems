import numpy as np
from typing import Tuple

def compute_regenerative_energy(
    omega: np.ndarray, 
    Vbus: np.ndarray, 
    dt: np.ndarray,
    keN: float = 0.8,        # V*s/rad (reflected)
    Rm: float = 0.2,         # Ohm
    kt_over_N: float = 0.12, # Nm/A (reflected torque per current)
    eta_c: float = 0.9,      # converter efficiency (scalar)
    I_max: float = 30.0,     # A
    Vb: float = 48.0         # battery voltage
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute regenerative energy recovery from motor trajectory data.
    
    Returns:
        Tuple of (current, bus_power, recovered_energy)
    """
    # Compute current with back-EMF and voltage constraints
    i = np.maximum((keN * omega - Vbus) / Rm, 0.0)
    i = np.minimum(i, I_max)  # Limit by maximum current rating
    
    # Calculate power into battery considering converter efficiency
    Pbus = Vbus * i * eta_c
    
    # Integrate power over time to get total recovered energy
    E_rec = np.sum(Pbus * dt)
    
    return i, Pbus, E_rec

def main():
    # Load trajectory data
    omega = np.load('omega.npy')   # rad/s (motor side)
    Vbus = np.load('Vbus.npy')     # V
    dt = np.load('dt.npy')         # s
    
    # Validate input arrays have compatible dimensions
    if not (omega.shape == Vbus.shape == dt.shape):
        raise ValueError("Input arrays must have identical dimensions")
    
    _, _, E_rec = compute_regenerative_energy(omega, Vbus, dt)
    print(f'Recovered energy (J): {E_rec:.2f}')

if __name__ == "__main__":
    main()