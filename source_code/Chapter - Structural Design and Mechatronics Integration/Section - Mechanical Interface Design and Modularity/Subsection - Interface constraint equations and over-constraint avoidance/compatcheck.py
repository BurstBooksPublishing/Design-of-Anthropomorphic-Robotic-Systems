import numpy as np
from typing import Dict, List, Tuple

def check_compatibility(A: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> Dict[str, object]:
    """
    Check if system Ax = b is compatible (has solution) using SVD.
    
    Args:
        A: (m x n) constraint matrix
        b: (m,) residual vector
        tol: tolerance for rank determination
        
    Returns:
        Dictionary with rank, residual norm, and compatibility status
    """
    # Perform SVD decomposition
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    rank = np.sum(s > tol)
    
    # Project b onto column space of A
    U_r = U[:, :rank]
    proj = U_r @ (U_r.T @ b)
    resid_norm = np.linalg.norm(b - proj)
    
    return {
        'rank': rank, 
        'residual_norm': resid_norm, 
        'compatible': resid_norm < tol
    }

def independent_rows(A: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """
    Find indices of linearly independent rows using QR decomposition with pivoting.
    
    Args:
        A: (m x n) matrix
        tol: tolerance for rank determination
        
    Returns:
        Array of indices corresponding to linearly independent rows
    """
    # QR decomposition with column pivoting on transpose
    Q, R, piv = np.linalg.qr(A.T, mode='economic', pivoting=True)
    diag = np.abs(np.diag(R))
    rank = np.sum(diag > tol)
    
    # Return indices of first 'rank' pivot elements
    return piv[:rank]