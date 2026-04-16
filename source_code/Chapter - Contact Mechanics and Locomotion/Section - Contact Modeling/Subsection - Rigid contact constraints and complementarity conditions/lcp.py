import numpy as np
from typing import Tuple, Optional

def solve_lcp(A: np.ndarray, c: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Solve Linear Complementarity Problem: find p >= 0, w = A*p + c >= 0, p^T*w = 0
    Uses direct enumeration for small problems; suitable only for demonstration.
    """
    m = A.shape[0]
    tolerance = 1e-12
    
    # Enumerate all possible active set combinations
    for mask in range(1 << m):
        # Determine active constraints (where p_i > 0)
        active = [i for i in range(m) if (mask >> i) & 1]
        
        if not active:
            # All variables inactive: p = 0, w = c
            p = np.zeros(m)
            w = A @ p + c
            if np.all(w >= -tolerance):
                return (p, w)
        else:
            # Solve for active variables: A_active * p_active = -c_active
            try:
                A_active = A[np.ix_(active, active)]
                c_active = c[active]
                p_active = np.linalg.solve(A_active, -c_active)
                
                # Check if solution is strictly positive for active variables
                if np.all(p_active > 0):
                    p = np.zeros(m)
                    p[active] = p_active
                    w = A @ p + c
                    
                    # Verify complementarity conditions
                    if np.all(w >= -tolerance):
                        return (p, w)
            except np.linalg.LinAlgError:
                # Skip singular subproblems
                continue
    
    return None

# Problem data
A = np.array([[0.8, 0.1],
              [0.1, 0.6]])
c = np.array([-0.05, 0.02])

# Solve LCP
solution = solve_lcp(A, c)

if solution is not None:
    p, w = solution
    print(f"p = {p}, w = {w}")
else:
    print("No solution found")