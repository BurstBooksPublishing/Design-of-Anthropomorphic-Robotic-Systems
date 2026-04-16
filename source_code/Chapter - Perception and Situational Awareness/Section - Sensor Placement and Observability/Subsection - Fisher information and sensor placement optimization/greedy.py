import numpy as np
from typing import List, Tuple, Optional

def greedy_logdet_optimization(I0_chol: np.ndarray, Ai_list: List[np.ndarray], k: int) -> Tuple[List[int], float]:
    """
    Greedy optimization for log-determinant maximization using Cholesky updates.
    
    Args:
        I0_chol: Lower triangular Cholesky factor of initial matrix I0 (I0 = L @ L.T)
        Ai_list: List of positive semi-definite matrices to select from
        k: Number of matrices to select
    
    Returns:
        Tuple of (selected indices, final log-determinant value)
    """
    if k <= 0 or k > len(Ai_list):
        raise ValueError("Invalid budget k")
    
    selected = []
    L = I0_chol.copy()
    logdet = 2 * np.sum(np.log(np.diag(L)))
    
    for _ in range(k):
        best_gain = -np.inf
        best_idx = None
        
        for j, A in enumerate(Ai_list):
            if j in selected:
                continue
                
            # Compute M = L^{-1} @ A for logdet calculation
            M = np.linalg.solve(L, A)
            S = M @ M.T
            
            # Calculate gain = logdet(I + L^{-1} @ A @ L^{-T}) = logdet(I + S)
            eigenvals = np.linalg.eigvalsh(S)
            gain = np.sum(np.log1p(eigenvals))
            
            if gain > best_gain:
                best_gain = gain
                best_idx = j
        
        if best_idx is None:
            break
            
        selected.append(best_idx)
        
        # Update Cholesky factorization with selected matrix
        current_matrix = L @ L.T + Ai_list[best_idx]
        L = np.linalg.cholesky(current_matrix)
        logdet += best_gain
    
    return selected, logdet