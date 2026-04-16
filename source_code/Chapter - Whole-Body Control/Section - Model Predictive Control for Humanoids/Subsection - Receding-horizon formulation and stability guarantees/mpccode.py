import numpy as np
import scipy.linalg as la
from osqp import OSQP
from scipy.sparse import csc_matrix

class LIPMMPCController:
    def __init__(self, N=30):
        # System matrices for discretized Linear Inverted Pendulum Model
        self.A = np.array([[1., 0.02], [0., 1.]])
        self.B = np.array([[0.0002], [0.02]])
        self.Q = np.diag([10., 1.])
        self.R = np.array([[1.]])
        self.N = N  # Prediction horizon
        
        # Compute terminal cost matrix from Discrete Algebraic Riccati Equation
        self.P = la.solve_discrete_are(self.A, self.B, self.Q, self.R)
        
        # Initialize OSQP solver
        self.solver = OSQP()
        self.problem_setup = False
    
    def _build_condensed_qp(self, x0, zmp_bounds):
        """
        Build condensed QP formulation for MPC problem.
        x0: current state [pos, vel]
        zmp_bounds: [min_zmp, max_zmp] constraints
        """
        n, m = self.B.shape
        N = self.N
        
        # Build Hessian matrix (cost weighting)
        H_blocks = [self.R + self.B.T @ self.Q @ self.B] * (N-1) + [self.R + self.B.T @ self.P @ self.B]
        H_diag = la.block_diag(*H_blocks)
        
        # Build gradient vector (tracking cost)
        q = np.zeros(N * m)
        
        # Build constraint matrices for ZMP constraints
        # ZMP = C*A^k*B*u + C*A^(k+1)*x0, where C = [1, 0] for position output
        A_ineq_list = []
        l_list = []
        u_list = []
        
        for k in range(N):
            # ZMP constraint coefficient matrix
            C = np.array([[1., 0.]])  # Output matrix for position
            CAkB = C @ (la.matrix_power(self.A, k) @ self.B)
            row = np.zeros((1, N))
            row[0, :k+1] = CAkB[0, 0]  # Only need first k+1 elements
            A_ineq_list.append(row)
            l_list.append(zmp_bounds[0] - (C @ la.matrix_power(self.A, k+1) @ x0)[0])
            u_list.append(zmp_bounds[1] - (C @ la.matrix_power(self.A, k+1) @ x0)[0])
        
        A_ineq = np.vstack(A_ineq_list)
        l = np.array(l_list)
        u = np.array(u_list)
        
        return H_diag, q, A_ineq, l, u
    
    def solve_mpc(self, x0, zmp_bounds=[-0.1, 0.1]):
        """
        Solve MPC QP and return first control input.
        x0: current state [position, velocity]
        zmp_bounds: ZMP constraint bounds
        """
        H, q, A_ineq, l, u = self._build_condensed_qp(x0, zmp_bounds)
        
        # Convert to sparse matrices for OSQP
        H_sparse = csc_matrix(H)
        A_sparse = csc_matrix(A_ineq)
        
        if not self.problem_setup:
            # Setup OSQP problem
            self.solver.setup(H_sparse, q, A_sparse, l, u, verbose=False)
            self.problem_setup = True
        else:
            # Update problem data (warm start)
            self.solver.update(Hx=H_sparse.data, Ax=A_sparse.data, q=q, l=l, u=u)
        
        # Solve QP
        result = self.solver.solve()
        
        if result.info.status != 'solved':
            raise RuntimeError(f"OSQP failed to solve: {result.info.status}")
        
        # Return first control input
        return result.x[0]

# Usage example:
# controller = LIPMMPCController(N=30)
# u_opt = controller.solve_mpc(x0=np.array([0.0, 0.1]))