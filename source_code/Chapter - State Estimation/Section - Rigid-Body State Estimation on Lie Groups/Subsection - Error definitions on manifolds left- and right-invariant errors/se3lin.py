import numpy as np
from scipy.linalg import expm, logm

def hat_se3(xi):
    """
    Convert 6D twist vector to 4x4 se(3) matrix representation.
    xi: 6-vector [angular_velocity; linear_velocity]
    """
    w, v = xi[:3], xi[3:]
    wx = np.array([[0, -w[2], w[1]],
                   [w[2], 0, -w[0]],
                   [-w[1], w[0], 0]])
    return np.block([[wx, v.reshape(3, 1)],
                     [np.zeros((1, 3)), 0]])

def adj_ad(xi):
    """
    Compute adjoint representation of Lie algebra element (ad_xi).
    Returns 6x6 matrix for SE(3) algebra operations.
    """
    w, v = xi[:3], xi[3:]
    wx = np.array([[0, -w[2], w[1]],
                   [w[2], 0, -w[0]],
                   [-w[1], w[0], 0]])
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.block([[wx, np.zeros((3, 3))],
                     [vx, wx]])

# Pose matrices (4x4 homogeneous transformation)
X = np.eye(4)              # True pose
X_hat = np.eye(4)          # Estimated pose
etaL = np.linalg.inv(X) @ X_hat  # Left-invariant error

# Body twists (6-vectors: [angular; linear])
xi = np.array([0.1, 0.2, 0.0, 0.01, 0.0, 0.0])       # True twist
xi_hat = np.array([0.1, 0.19, 0.01, 0.012, 0.0, 0.0]) # Estimated twist

# Linearized error dynamics: e_dot = A @ e + b
A = -adj_ad(xi)     # System matrix (6x6)
b = xi_hat - xi     # Input/innovation vector (6x1)