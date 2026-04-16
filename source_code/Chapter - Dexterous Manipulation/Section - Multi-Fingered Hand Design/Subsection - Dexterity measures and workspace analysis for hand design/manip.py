import numpy as np
from scipy.spatial import ConvexHull
from typing import Callable, Union

def manipulability(jacobian: np.ndarray) -> float:
    """Calculate manipulability measure as the product of singular values."""
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return float(np.prod(singular_values))

def min_singular(jacobian: np.ndarray) -> float:
    """Calculate minimum singular value of the Jacobian."""
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return float(singular_values[-1])

def sample_workspace(forward_kinematics: Callable, q_samples: np.ndarray) -> float:
    """
    Estimate workspace volume using convex hull of sampled end-effector positions.
    
    Args:
        forward_kinematics: Function mapping joint angles to end-effector positions
        q_samples: Array of joint angle samples (N x n)
    """
    points = np.vstack([forward_kinematics(q).reshape(-1, 3) for q in q_samples])
    hull = ConvexHull(points)
    return float(hull.volume)