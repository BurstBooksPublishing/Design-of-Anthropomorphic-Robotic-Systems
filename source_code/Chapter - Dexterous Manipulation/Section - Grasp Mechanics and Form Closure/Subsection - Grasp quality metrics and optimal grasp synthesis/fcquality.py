import numpy as np
import cvxpy as cp
from typing import List, Tuple, Union

def compute_grasp_matrix(contacts: List[Tuple[np.ndarray, List[np.ndarray]]]) -> np.ndarray:
    """
    Build grasp matrix G from contact points and friction cone directions.
    Each contact contributes force components and moments about origin.
    """
    wrench_generators = []
    
    for r, rays in contacts:  # r: contact position, rays: list of force directions
        for d in rays:
            f = d / np.linalg.norm(d)  # normalize force direction
            w = np.hstack((f, np.cross(r, f)))  # [force; moment] wrench
            wrench_generators.append(w)
    
    return np.column_stack(wrench_generators) if wrench_generators else np.zeros((6, 0))

def sample_unit_sphere(dim: int, M: int) -> np.ndarray:
    """Generate M uniformly distributed samples on S^(dim-1)."""
    samples = np.random.randn(M, dim)
    return samples / np.linalg.norm(samples, axis=1, keepdims=True)

def compute_approximate_qlimit(contacts: List[Tuple[np.ndarray, List[np.ndarray]]], 
                              M: int = 400) -> float:
    """
    Compute approximate quality limit using random sampling and LP optimization.
    Solves: max r s.t. r <= max_i <u_j, w_i> for all sampled u_j.
    """
    # Build wrench matrix
    W = compute_grasp_matrix(contacts)  # 6 x N
    if W.shape[1] == 0:
        return 0.0
    
    # Sample directions on unit sphere in 6D
    U = sample_unit_sphere(6, M)
    
    # Setup optimization problem
    r = cp.Variable()
    constraints = [r <= cp.max(W.T @ uj) for uj in U]  # <uj, W[:,i]> for all i
    prob = cp.Problem(cp.Maximize(r), constraints)
    
    # Solve and return result
    prob.solve(solver=cp.CVXOPT, verbose=False)
    return r.value if r.value is not None else 0.0

# Example usage:
# contacts = [(np.array([1,0,0]), [np.array([0,1,0])]), ...]  # (position, [directions])
# qlimit = compute_approximate_qlimit(contacts)