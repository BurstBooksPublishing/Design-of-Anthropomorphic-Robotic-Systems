import numpy as np
from typing import Callable, List, Union

def tendon_length(q: np.ndarray, node_fn: Callable) -> np.ndarray:
    """
    Compute tendon lengths from joint coordinates.
    
    Args:
        q: Joint coordinate vector
        node_fn: Function that returns list of node position lists for each tendon
        
    Returns:
        Array of tendon lengths
    """
    lengths = []
    nodes_all = node_fn(q)
    
    for nodes in nodes_all:
        segment_lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
        lengths.append(np.sum(segment_lengths))
        
    return np.array(lengths)

def numerical_Jt(q: np.ndarray, node_fn: Callable, eps: float = 1e-6) -> np.ndarray:
    """
    Compute tendon Jacobian matrix using finite differences.
    
    Args:
        q: Joint coordinate vector
        node_fn: Function that returns list of node position lists for each tendon
        eps: Finite difference step size
        
    Returns:
        Jacobian matrix (tendons Ã— joints)
    """
    m = len(tendon_length(q, node_fn))
    n = q.size
    J = np.zeros((m, n))
    l0 = tendon_length(q, node_fn)
    
    for i in range(n):
        dq = np.zeros_like(q)
        dq[i] = eps
        lp = tendon_length(q + dq, node_fn)
        J[:, i] = (lp - l0) / eps
        
    return J