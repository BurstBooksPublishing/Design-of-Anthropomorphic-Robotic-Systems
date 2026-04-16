)

import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from typing import Tuple, Optional

def update_log_odds(
    logodds: np.ndarray, 
    endpoint_mask: np.ndarray, 
    free_mask: np.ndarray, 
    L_occ: float = 0.85, 
    L_free: float = -0.4, 
    l0: float = 0.0
) -> np.ndarray:
    """
    Update log-odds values for occupancy grid mapping.
    
    Args:
        logodds: 3D array of current log-odds values
        endpoint_mask: Boolean mask indicating voxel endpoints of sensor measurements
        free_mask: Boolean mask indicating free voxels along sensor rays
        L_occ: Log-odds increment for occupied voxels
        L_free: Log-odds decrement for free voxels
        l0: Initial log-odds value
    
    Returns:
        Updated log-odds array with values clamped for numerical stability
    """
    # Update free voxel log-odds
    logodds[free_mask] += L_free - l0
    
    # Update occupied voxel log-odds
    logodds[endpoint_mask] += L_occ - l0
    
    # Clamp values to prevent numerical instability
    np.clip(logodds, -20.0, 20.0, out=logodds)
    return logodds

def make_tsdf(
    occupancy_mask: np.ndarray, 
    voxel_size: float, 
    truncation: float
) -> np.ndarray:
    """
    Generate Truncated Signed Distance Function from occupancy mask.
    
    Args:
        occupancy_mask: Boolean array where True indicates occupied voxels
        voxel_size: Physical size of each voxel
        truncation: Maximum distance value for truncation
    
    Returns:
        TSDF array with values truncated to [-truncation, truncation]
    """
    # Compute distance from free space to occupied voxels
    dist_occ = edt(~occupancy_mask) * voxel_size
    
    # Compute distance from occupied voxels to free space
    dist_free = edt(occupancy_mask) * voxel_size
    
    # Calculate signed distance field
    sdf = dist_occ - dist_free
    
    # Apply truncation to limit influence of distant voxels
    tsdf = np.clip(sdf, -truncation, truncation)
    return tsdf