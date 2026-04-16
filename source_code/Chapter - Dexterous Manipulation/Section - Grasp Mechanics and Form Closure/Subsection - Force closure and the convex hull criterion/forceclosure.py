import numpy as np
from scipy.optimize import linprog
from typing import Union, Optional

def force_closure_test(V: np.ndarray, eps: float = 1e-3) -> bool:
    """
    Test if a grasp achieves force closure using linear programming.
    
    Args:
        V: (6, M) array of primitive wrenches (columns)
        eps: small positive interior margin for lambda bounds
        
    Returns:
        bool: True if force closure is achieved
    """
    if V.ndim != 2 or V.shape[0] != 6:
        raise ValueError("V must be a 6xM matrix")
    
    M = V.shape[1]
    if M == 0:
        return False
    
    # Constraints: V @ lambda = 0 (force/torque balance)
    # Additional constraint: sum(lambda) = 1 (convex combination)
    A_eq = np.vstack([V, np.ones(M)])
    b_eq = np.zeros(7)
    b_eq[-1] = 1.0
    
    # Bounds: lambda_k >= eps (strictly positive coefficients)
    bounds = [(eps, None) for _ in range(M)]
    
    # Solve feasibility problem (zero objective)
    try:
        res = linprog(
            c=np.zeros(M), 
            A_eq=A_eq, 
            b_eq=b_eq, 
            bounds=bounds, 
            method='highs'
        )
        return res.success
    except Exception:
        return False