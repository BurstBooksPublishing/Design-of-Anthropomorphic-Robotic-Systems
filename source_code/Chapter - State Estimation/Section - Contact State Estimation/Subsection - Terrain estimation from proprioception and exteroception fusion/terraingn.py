import numpy as np
from typing import List, Tuple

def estimate_plane(
    pts_y: np.ndarray, 
    feet_p: np.ndarray, 
    normals_u: np.ndarray, 
    R_list: List[np.ndarray], 
    Wp: np.ndarray, 
    Wy: np.ndarray, 
    Wu: np.ndarray, 
    n_init: np.ndarray, 
    d_init: float, 
    iters: int = 10
) -> Tuple[np.ndarray, float]:
    """
    Estimate plane parameters (normal vector n and distance d) using Gauss-Newton optimization.
    
    The plane equation is defined as: nÂ·x + d = 0, where ||n|| = 1
    """
    n = n_init.copy()
    d = float(d_init)
    
    for _ in range(iters):
        # Build tangent basis S (two orthonormal vectors spanning tangent plane at n)
        t1 = np.array([-n[1], n[0], 0.0])
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.array([0.0, -n[2], n[1]])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        S = np.column_stack([t1, t2])  # 3x2 matrix
        
        # Assemble Jacobian and residuals
        J_blocks = []
        r_blocks = []
        W_blocks = []
        
        # Point-to-plane constraints (feet points)
        for p, w in zip(feet_p, Wp):
            r = n.dot(p) + d
            J = np.hstack([p.dot(S), 1.0])
            J_blocks.append(J)
            r_blocks.append(r)
            W_blocks.append(w)
            
        # Point-to-plane constraints (y points)
        for y, w in zip(pts_y, Wy):
            r = n.dot(y) + d
            J = np.hstack([y.dot(S), 1.0])
            J_blocks.append(J)
            r_blocks.append(r)
            W_blocks.append(w)
            
        # Normal direction constraints
        for u, R, W in zip(normals_u, R_list, Wu):
            r = u - R.dot(n)  # 3x1 residual
            J = np.hstack([-R.dot(S), np.zeros((3, 1))])  # 3x3 block
            # Convert block into stacked scalar rows
            for k in range(3):
                J_blocks.append(J[k, :])
                r_blocks.append(r[k])
                W_blocks.append(W[k, k])
                
        J = np.vstack(J_blocks)
        r = np.array(r_blocks)
        W = np.diag(W_blocks)
        
        # Gauss-Newton step
        H = J.T.dot(W).dot(J)
        g = J.T.dot(W).dot(r)
        try:
            delta = -np.linalg.solve(H, g)  # 3x1: [dtheta(2); dd]
        except np.linalg.LinAlgError:
            # Handle singular matrix case
            delta = -np.linalg.pinv(H).dot(g)
            
        dtheta = delta[:2]
        dd = delta[2]
        
        # Retract to manifold (update n and d)
        n = n + S.dot(dtheta)
        n /= np.linalg.norm(n)  # Enforce unit norm constraint
        d += dd
        
    return n, d