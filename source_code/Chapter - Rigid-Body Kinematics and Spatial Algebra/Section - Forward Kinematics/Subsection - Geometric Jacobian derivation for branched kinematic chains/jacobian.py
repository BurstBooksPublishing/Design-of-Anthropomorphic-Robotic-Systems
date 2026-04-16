import numpy as np

def adjoint(T):
    R = T[:3,:3]; p = T[:3,3]
    Ad = np.zeros((6,6))
    Ad[:3,:3] = R
    Ad[3:,:3] = hat(p) @ R
    Ad[3:,3:] = R
    return Ad

def hat(v):                      # 3-vector to skew-symmetric
    return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

def jacobian_for_end(T_root_to, S_list, path_indices, idx_e):
    # T_root_to: list of transforms T_root->i for all i and end-effector
    # S_list: list of 6x1 twist axes S_i expressed in joint frames
    # path_indices: ordered indices from root to end-effector idx_e
    T_root_e = T_root_to[idx_e]
    J = np.zeros((6, len(path_indices)))
    for k, i in enumerate(path_indices):
        T_i = T_root_to[i]
        T_i_to_e = T_root_e @ np.linalg.inv(T_i)   # adjoint argument
        J[:,k] = adjoint(T_i_to_e) @ S_list[i]     # column contribution
    return J