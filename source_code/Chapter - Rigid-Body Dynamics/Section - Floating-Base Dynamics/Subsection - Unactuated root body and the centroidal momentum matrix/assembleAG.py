import numpy as np
# Inputs: list of link inertias I_list (6x6), jacobians J_list (6 x ndof), transforms Tc_list (4x4)
def adjoint_T(T):
    R = T[:3,:3]; p = T[:3,3]
    Ad = np.block([[R, np.zeros((3,3))],
                   [skew(p)@R, R]])  # 6x6 adjoint
    return Ad
def skew(v): return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
def assemble_AG(I_list, J_list, Tc_list):
    ndof = J_list[0].shape[1]
    AG = np.zeros((6, ndof))
    for I, J, Tc in zip(I_list, J_list, Tc_list):
        AdT = adjoint_T(Tc).T     # Ad_{T_c^i}^T
        AG += AdT @ I @ J        # accumulate
    return AG
# Use in control: hG = AG @ qdot; dot_hG = AG @ qdd + A_dot @ qdot  (A_dot via finite diff)