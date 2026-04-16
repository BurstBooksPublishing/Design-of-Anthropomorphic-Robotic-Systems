import numpy as np
from typing import List, Tuple, Union

def adjoint(g: np.ndarray) -> np.ndarray:
    """Compute the 6x6 adjoint matrix from a 4x4 homogeneous transformation.
    
    Args:
        g: 4x4 homogeneous transformation matrix (configuration -> origin)
        
    Returns:
        6x6 adjoint matrix
    """
    R = g[:3, :3]
    p = g[:3, 3]
    
    # Create skew-symmetric matrix from translation vector
    p_hat = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])
    
    # Construct 6x6 adjoint matrix
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = p_hat @ R
    Ad[3:, 3:] = R
    
    return Ad

def assemble_G(g_oc_list: List[np.ndarray], S_list: List[np.ndarray]) -> np.ndarray:
    """Assemble the grasp matrix G from object-to-contact transformations and contact signatures.
    
    Args:
        g_oc_list: List of 4x4 homogeneous transformations (object -> contact frame)
        S_list: List of contact signature matrices
        
    Returns:
        Assembled grasp matrix with columns corresponding to each contact
    """
    if len(g_oc_list) != len(S_list):
        raise ValueError("g_oc_list and S_list must have the same length")
    
    blocks = []
    for g_oc, S in zip(g_oc_list, S_list):
        Ad = adjoint(g_oc)
        # Compute Ad^{-T} * S for wrench transformation
        blocks.append(np.linalg.inv(Ad).T @ S)
    
    return np.hstack(blocks)