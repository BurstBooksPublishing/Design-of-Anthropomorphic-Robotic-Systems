import numpy as np
from typing import Tuple

def poly_friction_constraints(N: int, mu: float, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate polyhedral friction cone constraints for N contacts.
    
    Returns A, b for A @ f_vec <= b where f_vec = [f1x,f1y,f1z, f2x,f2y,f2z, ...]
    in contact frames (z-axis aligned with contact normal).
    """
    # Generate m edge vectors of the friction polygon
    angles = 2 * np.pi * np.arange(m) / m
    U = np.column_stack([np.cos(angles), np.sin(angles)])  # m x 2
    
    # Normalization factor for inscribed circle radius
    cos_alpha = np.cos(np.pi / m)
    
    # Build constraint matrix
    rows = []
    for i in range(N):
        base = np.zeros(3 * N)
        for u in U:
            row = base.copy()
            # Constraint: u^T f_t - mu*cos(alpha)*f_n <= 0
            row[3*i:3*i+2] = u                    # tangential components
            row[3*i+2] = -mu * cos_alpha         # normal component coefficient
            rows.append(row)
    
    A = np.vstack(rows)
    b = np.zeros(A.shape[0])
    return A, b

# Example usage
A, b = poly_friction_constraints(2, 0.7, 8)