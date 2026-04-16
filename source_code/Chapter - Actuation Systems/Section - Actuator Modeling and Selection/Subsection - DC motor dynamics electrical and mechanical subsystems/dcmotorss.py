import numpy as np
from scipy import signal

class MotorSystem:
    """DC motor with gearbox system model."""
    
    def __init__(self):
        # Motor parameters (SI units)
        self.kt = 0.05      # Torque constant (Nm/A)
        self.ke = 0.05      # Back EMF constant (V/(rad/s))
        self.R = 0.2        # Armature resistance (Ohm)
        self.L = 1e-3       # Armature inductance (H)
        self.Jm = 6e-6      # Motor inertia (kg*m^2)
        self.Bm = 1e-4      # Motor damping (N*m*s)
        self.n = 200.0      # Gearbox ratio
        self.eta = 0.9      # Gearbox efficiency
        
        # State-space representation: x=[i; omega_m], input V, output omega_j
        self.A = np.array([[-self.R/self.L, -self.ke/self.L],
                          [self.kt/self.Jm, -self.Bm/self.Jm]])
        self.B = np.array([[1.0/self.L], [0.0]])
        self.C = np.array([[0.0, self.eta/self.n]])  # Include efficiency in output
        self.D = np.array([[0.0]])
        
        self.sys = signal.StateSpace(self.A, self.B, self.C, self.D)
        self.tf = signal.TransferFunction(*signal.ss2tf(self.A, self.B, self.C, self.D))
    
    def analyze_system(self):
        """Compute system characteristics for design validation."""
        poles = np.linalg.eigvals(self.A)
        # DC gain calculation with proper coefficient extraction
        dc_gain = self.tf.num[0][-1]/self.tf.den[0][-1] if self.tf.den[0][-1] != 0 else 0
        
        return poles, dc_gain

def main():
    motor = MotorSystem()
    poles, dc_gain = motor.analyze_system()
    
    print(f"Poles: {poles}")
    print(f"DC gain (rad/s per V): {dc_gain:.6f}")

if __name__ == "__main__":
    main()