import numpy as np
from scipy.linalg import expm, norm

def hat(omega):
    """Convert 3D vector to 3x3 skew-symmetric matrix."""
    omega = np.asarray(omega)
    if omega.shape != (3,):
        raise ValueError("Input must be a 3D vector")
    
    return np.array([[0, -omega[2], omega[1]],
                     [omega[2], 0, -omega[0]],
                     [-omega[1], omega[0], 0]])

def vee(S):
    """Convert 3x3 skew-symmetric matrix to 3D vector."""
    S = np.asarray(S)
    if S.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix")
    
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

def exp_so3(omega):
    """Compute SO(3) exponential map using Rodrigues' formula."""
    omega = np.asarray(omega)
    if omega.shape != (3,):
        raise ValueError("Input must be a 3D vector")
    
    return expm(hat(omega))

def left_jacobian(omega):
    """Compute the left Jacobian for SO(3)."""
    omega = np.asarray(omega)
    if omega.shape != (3,):
        raise ValueError("Input must be a 3D vector")
    
    theta = norm(omega)
    
    if theta < 1e-8:
        return np.eye(3)
    
    W = hat(omega)
    W2 = W @ W
    theta2 = theta * theta
    theta3 = theta2 * theta
    
    return (np.eye(3) + 
            (1 - np.cos(theta)) / theta2 * W + 
            (theta - np.sin(theta)) / theta3 * W2)

def exp_se3(xi):
    """Compute SE(3) exponential map from 6D twist."""
    xi = np.asarray(xi)
    if xi.shape != (6,):
        raise ValueError("Input must be a 6D vector")
    
    omega, v = xi[:3], xi[3:]
    R = exp_so3(omega)
    J = left_jacobian(omega)
    
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = J @ v
    
    return X