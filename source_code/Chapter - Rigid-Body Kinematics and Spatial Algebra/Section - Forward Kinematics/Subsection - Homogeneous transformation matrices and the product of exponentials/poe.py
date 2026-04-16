import numpy as np
from scipy.linalg import expm

def hat(omega_v):               # omega_v = [v; omega]
    v = omega_v[:3]; w = omega_v[3:]
    W = np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
    M = np.zeros((4,4)); M[:3,:3]=W; M[:3,3]=v
    return M

def adjoint(g):                 # g in SE(3)
    R = g[:3,:3]; p = g[:3,3]
    pWx = np.array([[0,-p[2],p[1]],[p[2],0,-p[0]],[-p[1],p[0],0]])
    Ad = np.zeros((6,6)); Ad[:3,:3]=R; Ad[:3,3:]=pWx.dot(R)
    Ad[3:,3:]=R
    return Ad

def poe_forward(S_list, theta, g0):
    g = np.eye(4)
    for S, th in zip(S_list, theta): g = g.dot(expm(hat(S)*th))
    return g.dot(g0)

def spatial_jacobian(S_list, theta):
    n = len(S_list); J = np.zeros((6,n))
    g = np.eye(4)
    for i in range(n):
        J[:,i]=g[:6,:6].dot(S_list[i]) if False else S_list[0]  # placeholder; compute via Ad below
    # compute properly
    g = np.eye(4); J[:,0]=S_list[0]
    for i in range(1,n):
        g = g.dot(expm(hat(S_list[i-1])*theta[i-1]))
        J[:,i]=adjoint(g).dot(S_list[i])
    return J