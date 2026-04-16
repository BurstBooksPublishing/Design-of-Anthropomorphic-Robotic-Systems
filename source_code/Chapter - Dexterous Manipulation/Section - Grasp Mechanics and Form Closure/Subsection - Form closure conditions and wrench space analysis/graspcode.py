import numpy as np
from scipy.optimize import linprog
from typing import Tuple, List

def unit_wrench(n: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Compute unit wrench from normal vector and position vector."""
    return np.hstack((n, np.cross(r, n)))

def compute_form_closure_matrix(contact_positions: np.ndarray, 
                               contact_normals: np.ndarray) -> np.ndarray:
    """Build the form closure matrix G from contact positions and normals."""
    return np.column_stack([
        unit_wrench(contact_normals[i], contact_positions[i]) 
        for i in range(contact_positions.shape[0])
    ])

def generate_friction_cone_edges(normal: np.ndarray, 
                                position: np.ndarray,
                                k_edges: int = 8, 
                                angle: float = np.deg2rad(25.0)) -> List[np.ndarray]:
    """Generate k edge vectors approximating the friction cone."""
    # Build orthonormal basis with normal as first vector
    t1 = np.array([1.0, 0, 0])
    if abs(np.dot(t1, normal)) > 0.9:
        t1 = np.array([0, 1.0, 0])
    t1 -= np.dot(t1, normal) * normal
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    
    # Generate k vectors on cone boundary
    edges = []
    for k in range(k_edges):
        phi = 2 * np.pi * k / k_edges
        u = (np.cos(angle) * normal + 
             np.sin(angle) * (np.cos(phi) * t1 + np.sin(phi) * t2))
        u /= np.linalg.norm(u)
        edges.append(unit_wrench(u, position))
    return edges

def test_force_closure(W: np.ndarray, eps: float = 1e-6) -> bool:
    """Test force closure using linear programming."""
    m = W.shape[1]
    c = np.zeros(m)
    A_eq = np.vstack([W, np.ones(m)])
    b_eq = np.zeros(W.shape[0] + 1)
    b_eq[-1] = 1.0
    bounds = [(eps, None)] * m
    
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return result.success

# Define contact positions and compute inward-facing normals
r = np.array([
    [0.05, 0.05, 0.02],
    [-0.05, 0.04, -0.01],
    [0.03, -0.06, 0.02],
    [-0.04, -0.03, -0.03]
])
n = -r / np.linalg.norm(r, axis=1)[:, None]

# Compute form closure matrix and check rank
G = compute_form_closure_matrix(r, n)
rank_G = np.linalg.matrix_rank(G)
print("rank(G) =", rank_G)

# Generate friction cone approximation
k_edges = 8
angle = np.deg2rad(25.0)
W_edges = []

for i in range(r.shape[0]):
    W_edges.extend(generate_friction_cone_edges(n[i], r[i], k_edges, angle))

W_friction = np.vstack(W_edges).T

# Test force closure
is_force_closure = test_force_closure(W_friction)
print("force-closure test success:", is_force_closure)