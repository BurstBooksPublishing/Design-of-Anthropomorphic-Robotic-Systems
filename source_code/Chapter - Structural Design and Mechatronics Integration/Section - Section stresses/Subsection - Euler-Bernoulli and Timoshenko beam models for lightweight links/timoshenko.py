import numpy as np
from scipy.linalg import eigh

# Beam geometry and material properties
L = 0.30          # Length (m)
b = 0.03          # Width (m)
h = 0.02          # Height (m)
A = b * h         # Cross-sectional area
I = b * h**3 / 12 # Second moment of area
E = 70e9          # Young's modulus (Pa)
nu = 0.33         # Poisson's ratio
G = E / (2 * (1 + nu))  # Shear modulus
rho = 2700.0      # Density (kg/m^3)
k_shear = 5 / 6   # Shear correction factor

# Discretization
ne = 10                    # Number of elements
nn = ne + 1                # Number of nodes
dof = 2 * nn               # Total degrees of freedom (2 per node: transverse displacement and rotation)
Le = L / ne                # Element length

def element_matrices(length):
    """Compute Timoshenko beam element stiffness and mass matrices."""
    # Bending stiffness matrix
    k_b = (E * I / length**3) * np.array([
        [12,      6*length,  -12,      6*length],
        [6*length, 4*length**2, -6*length, 2*length**2],
        [-12,    -6*length,   12,     -6*length],
        [6*length, 2*length**2, -6*length, 4*length**2]
    ])
    
    # Shear stiffness matrix
    k_s = (k_shear * G * A / length) * np.array([
        [1,  length/2, -1,  length/2],
        [length/2, length**2/4, -length/2, length**2/4],
        [-1, -length/2,  1, -length/2],
        [length/2, length**2/4, -length/2, length**2/4]
    ])

    # Transverse inertia mass matrix
    m_t = (rho * A * length / 420) * np.array([
        [156,    22*length,   54,   -13*length],
        [22*length, 4*length**2, 13*length, -3*length**2],
        [54,     13*length,  156,   -22*length],
        [-13*length, -3*length**2, -22*length, 4*length**2]
    ])

    # Rotary inertia mass matrix
    m_r = (rho * I * length / 30) * np.array([
        [36,   3*length, -36,   3*length],
        [3*length, 4*length**2, -3*length, -length**2],
        [-36, -3*length,  36,  -3*length],
        [3*length, -length**2, -3*length, 4*length**2]
    ])

    return k_b + k_s, m_t + m_r

# Assemble global stiffness and mass matrices
K = np.zeros((dof, dof))
M = np.zeros((dof, dof))
ke, me = element_matrices(Le)

for e in range(ne):
    idx = 2 * e
    K[idx:idx+4, idx:idx+4] += ke
    M[idx:idx+4, idx:idx+4] += me

# Apply cantilever boundary conditions (fixed at x=0: w=0, phi=0)
free_dofs = slice(2, dof)
K_free = K[free_dofs, free_dofs]
M_free = M[free_dofs, free_dofs]

# Solve generalized eigenvalue problem for the first 6 modes
eigenvalues, _ = eigh(K_free, M_free, subset_by_index=[0, 5])
frequencies = np.sqrt(np.maximum(eigenvalues, 0.0)) / (2 * np.pi)

# Output first three natural frequencies
print("First three Timoshenko frequencies (Hz):", frequencies[:3])