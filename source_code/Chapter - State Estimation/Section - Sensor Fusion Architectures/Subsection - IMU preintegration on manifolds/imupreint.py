import numpy as np

def hat(v):
    """Skew-symmetric matrix representation of a 3D vector."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def Exp(phi):
    """Exponential map from so(3) to SO(3) using Rodrigues' formula."""
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        return np.eye(3) + hat(phi)
    k = phi / theta
    K = hat(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

class Preintegrator:
    """Preintegrates IMU measurements between two keyframes, handling bias correction."""
    
    def __init__(self, b_omega, b_acc, P0=None):
        self.bw = np.array(b_omega)
        self.ba = np.array(b_acc)
        self.DR = np.eye(3)
        self.Dv = np.zeros(3)
        self.Dp = np.zeros(3)
        self.P = P0 if P0 is not None else np.zeros((9, 9))
        # Jacobians w.r.t. biases
        self.J_rb = np.zeros((3, 6))
        self.J_vb = np.zeros((3, 6))
        self.J_pb = np.zeros((3, 6))

    def propagate(self, omega_m, acc_m, dt, Q):
        """Propagate dynamics and covariance over a time step with measurement inputs."""
        omega_m = np.array(omega_m)
        acc_m = np.array(acc_m)
        Q = np.array(Q)

        # Compute rotation increment and its Jacobian
        phi = (omega_m - self.bw) * dt
        R_inc = Exp(phi)

        # State transition and noise Jacobians (simplified)
        F = np.eye(9)
        G = np.zeros((9, 6))

        # Update preintegrated measurements
        self.Dp += self.Dv * dt + 0.5 * self.DR @ (acc_m - self.ba) * dt**2
        self.Dv += self.DR @ (acc_m - self.ba) * dt
        self.DR = self.DR @ R_inc

        # Covariance propagation
        self.P = F @ self.P @ F.T + G @ Q @ G.T

        # Bias Jacobian updates (accumulate sensitivity to biases)
        # In full implementation, compute exact terms for J_rb, J_vb, J_pb