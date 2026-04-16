import numpy as np
from typing import Tuple, Optional

def block_sad_disparity(Ileft: np.ndarray, Iright: np.ndarray, max_disp: int = 64, win: int = 5) -> np.ndarray:
    """
    Compute disparity map using block matching with Sum of Absolute Differences (SAD).
    
    Args:
        Ileft: Left image as grayscale float array
        Iright: Right image as grayscale float array
        max_disp: Maximum disparity to search for
        win: Window size for block matching (must be odd)
        
    Returns:
        Disparity map as int32 array
    """
    if win % 2 == 0:
        raise ValueError("Window size must be odd")
        
    if Ileft.shape != Iright.shape:
        raise ValueError("Left and right images must have the same shape")
        
    h, w = Ileft.shape
    disp = np.zeros((h, w), dtype=np.int32)
    pad = win // 2
    
    # Pad images to handle boundary conditions
    L = np.pad(Ileft, pad, mode='reflect')
    R = np.pad(Iright, pad, mode='reflect')
    
    # Pre-allocate block memory for efficiency
    blockL = np.empty((win, win), dtype=Ileft.dtype)
    
    for y in range(pad, h + pad):
        for x in range(pad, w + pad):
            best_cost = np.inf
            best_disp = 0
            
            # Extract left block once per pixel
            blockL[:] = L[y-pad:y+pad+1, x-pad:x+pad+1]
            
            # Search for best matching disparity
            for d in range(min(max_disp, x - pad + 1)):  # Ensure we don't go out of bounds
                xr = x - d
                if xr - pad < 0:
                    break
                    
                blockR = R[y-pad:y+pad+1, xr-pad:xr+pad+1]
                cost = np.sum(np.abs(blockL - blockR))  # SAD matching cost
                
                if cost < best_cost:
                    best_cost = cost
                    best_disp = d
                    
            disp[y-pad, x-pad] = best_disp
            
    return disp

def disparity_to_depth(disp: np.ndarray, f_px: float, B_m: float) -> np.ndarray:
    """
    Convert disparity map to depth map using stereo geometry.
    
    Args:
        disp: Disparity map
        f_px: Focal length in pixels
        B_m: Baseline in meters
        
    Returns:
        Depth map in meters
    """
    # Handle division by zero and invalid disparities
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (f_px * B_m) / disp.astype(np.float32)
        
    # Set invalid disparities (zero or negative) to infinity
    Z[disp <= 0] = np.inf
    return Z