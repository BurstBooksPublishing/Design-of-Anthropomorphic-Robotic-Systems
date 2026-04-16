import numpy as np

def solve_pnp(depth: np.ndarray, mask: np.ndarray, K: np.ndarray, model_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve PnP problem using SVD-based Procrustes alignment.
    
    Args:
        depth: (H,W) depth map
        mask: (H,W) boolean mask of valid pixels
        K: (3,3) camera intrinsic matrix
        model_pts: (3,m) model points in object coordinates
    
    Returns:
        R: (3,3) rotation matrix (camera <- object)
        t: (3,) translation vector (camera <- object)
    """
    # Backproject masked depth to 3D points
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError("No valid pixels in mask")
    
    zs = depth[ys, xs].astype(np.float32)
    pix = np.stack([xs, ys, np.ones_like(xs)], axis=0)  # Homogeneous pixel coordinates
    invK = np.linalg.inv(K)
    pts_cam = invK @ pix * zs  # 3 x N measured points
    
    # Center points for Procrustes alignment
    X = model_pts - model_pts.mean(axis=1, keepdims=True)
    Y = pts_cam - pts_cam.mean(axis=1, keepdims=True)
    
    # SVD for optimal rotation
    U, _, Vt = np.linalg.svd(Y @ X.T)
    R = U @ Vt
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    
    # Compute translation
    t = pts_cam.mean(axis=1) - R @ model_pts.mean(axis=1)
    
    return R, t