import numpy as np

def compute_system_eigenvalues():
    """
    Compute eigenvalues of a hydraulic system linearized state-space model.
    System states: [pressure, velocity, spool_position]
    """
    # Physical parameters (SI units)
    A = 5e-4        # Piston area (mÂ²)
    m = 8.0         # Mass (kg)
    beta = 8e8      # Bulk modulus (Pa)
    Vt = 1e-3       # Volume (mÂ³)
    B = 400.0       # Damping coefficient (N*s/m)
    Kq = 0.05       # Flow gain (mÂ³/s/Pa)
    Kp = 1e-9       # Pressure gain (mÂ³/s/Pa)
    tau_s = 0.002   # Spool time constant (s)
    Ku = 1.0        # Spool gain (dimensionless)

    # State matrix coefficients
    a11 = -(beta / Vt) * Kp
    a12 = -(beta / Vt) * A
    a13 = (beta / Vt) * Kq
    a21 = A / m
    a22 = -B / m
    a23 = 0.0
    a31 = 0.0
    a32 = 0.0
    a33 = -1.0 / tau_s

    # Construct state matrix
    A_mat = np.array([
        [a11, a12, a13],
        [a21, a22, a23],
        [a31, a32, a33]
    ])

    # Compute and return eigenvalues
    return np.linalg.eigvals(A_mat)

if __name__ == "__main__":
    eigenvalues = compute_system_eigenvalues()
    print("eigenvalues:", eigenvalues)