import numpy as np
from typing import Tuple, Optional

def matrix_rank(A: np.ndarray, tol: float = 1e-9) -> int:
    """Compute the rank of matrix A using SVD with specified tolerance."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return np.sum(S > tol)

def is_range_equal(J: np.ndarray, s_basis: np.ndarray, tol: float = 1e-9) -> bool:
    """Check if range of J equals range of s_basis by rank test."""
    combined_rank = matrix_rank(np.hstack([J, s_basis]), tol)
    s_rank = matrix_rank(s_basis, tol)
    J_rank = matrix_rank(J, tol)
    return combined_rank == s_rank == J_rank

def orthogonal_complement(basis: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Return orthonormal basis for orthogonal complement of basis in R^6."""
    U, S, Vt = np.linalg.svd(basis, full_matrices=True)
    rank = np.sum(S > tol)
    # Return the right singular vectors corresponding to zero singular values
    return Vt.T[:, rank:]

# Example: single-axis hinge about x-axis at origin
s = np.array([[1, 0, 0, 0, 0, 0]]).T  # 6x1 screw
J = s.copy()  # ideal actuator map

print("Span match:", is_range_equal(J, s))
print("Reaction basis dim:", orthogonal_complement(s).shape[1])