import numpy as np
from typing import Tuple

def calculate_ripple_voltage(
    switching_frequency: float = 20e3,
    series_resistance: float = 20e-3,
    esr: float = 10e-3,
    capacitance: float = 470e-6,
    inductance: float = 0.0,
    switching_current: float = 2.0
) -> Tuple[float, float]:
    """
    Calculate RMS ripple voltage for RC/RLC filter.
    
    Returns:
        Tuple of (frequency, ripple_voltage_rms)
    """
    # Frequency range: 1 kHz to 1 MHz
    frequencies = np.logspace(3, 6, 301)
    angular_freq = 2 * np.pi * frequencies
    
    # Capacitor impedance with ESR
    capacitor_impedance = esr + 1.0 / (1j * angular_freq * capacitance)
    
    # Total impedance (RC or RLC)
    if inductance > 0:
        inductor_impedance = 1j * angular_freq * inductance
        total_impedance = 1.0 / (1.0/series_resistance + 1.0/capacitor_impedance + 1.0/inductor_impedance)
    else:
        # Simple RC filter
        total_impedance = 1.0 / (1.0/series_resistance + 1.0/capacitor_impedance)
    
    # Ripple voltage magnitude
    ripple_voltage = np.abs(switching_current * total_impedance)
    
    # Find closest frequency to switching frequency
    target_idx = np.argmin(np.abs(frequencies - switching_frequency))
    
    return frequencies[target_idx], ripple_voltage[target_idx]

def main() -> None:
    frequency, ripple_rms = calculate_ripple_voltage()
    print(f"f_sw {frequency:.0f} Hz  V_ripple_rms {ripple_rms:.6f} V")

if __name__ == "__main__":
    main()