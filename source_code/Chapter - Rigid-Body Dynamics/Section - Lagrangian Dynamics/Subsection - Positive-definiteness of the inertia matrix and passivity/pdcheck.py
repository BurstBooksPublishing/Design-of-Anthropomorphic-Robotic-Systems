import numpy as np
# Inputs: list of 6x6 spatial inertias I_list, list of 6xn Jacobians J_list
# Compute M = sum J_i^T I_i J_i and check eigenvalues
def mass_matrix(I_list, J_list):
    M = np.zeros((J_list[0].shape[1],)*2)
    for I,J in zip(I_list, J_list):
        M += J.T @ I @ J
    return M

# Example usage (placeholders represent realistic link inertias and Jacobians)
I_list = [np.eye(6)*m for m in [1.2,0.8,0.6,0.5,0.4,0.3,0.2]]  # spatial inertias
J_list = [np.random.randn(6,7) for _ in range(7)]             # numeric Jacobians
M = mass_matrix(I_list, J_list)
eig = np.linalg.eigvalsh(M)                                   # symmetric eigenvalues
print('min eigenvalue:', eig[0])                               # PD if >0