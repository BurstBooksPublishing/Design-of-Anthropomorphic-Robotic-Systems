import numpy as np
# T_d: 4x4 desired pose; params: dict with L2,L3,d
def closed_form_ik(T_d, params):
    R_d, p_d = T_d[:3,:3], T_d[:3,3]
    d = params['d']                     # wrist->ee in ee frame
    p_w = p_d - R_d.dot(d)             # wrist center
    rx,ry,rz = p_w                     # base origin at 0
    q1 = np.arctan2(ry, rx)            # base yaw
    # rotate into shoulder plane
    c,s = np.cos(-q1), np.sin(-q1)
    Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    rprime = Rz.dot(p_w)
    xprime, zprime = rprime[0], rprime[2]
    L2, L3 = params['L2'], params['L3']
    D = (xprime**2+zprime**2 - L2**2 - L3**2)/(2*L2*L3)
    if abs(D) > 1.0: return []          # unreachable
    q3_options = [np.arctan2(np.sqrt(1-D**2), D), np.arctan2(-np.sqrt(1-D**2), D)]
    solutions = []
    for q3 in q3_options:
        q2 = np.arctan2(zprime, xprime) - np.arctan2(L3*np.sin(q3), L2+L3*np.cos(q3))
        # compute R_w and extract wrist Euler (Z-Y-X) as example
        R_13 = forward_R_1_to_3(q1,q2,q3,params)
        R_36 = R_13.T.dot(R_d)
        # Z-Y-X extraction
        if abs(R_36[2,0]) != 1:
            q5 = np.arctan2(np.sqrt(1 - R_36[2,0]**2), R_36[2,0])
            q4 = np.arctan2(R_36[1,0], R_36[0,0])
            q6 = np.arctan2(R_36[2,1], -R_36[2,2])
            solutions.append([q1,q2,q3,q4,q5,q6])
            q5_alt = np.arctan2(-np.sqrt(1 - R_36[2,0]**2), R_36[2,0])
            q4_alt = np.arctan2(-R_36[1,0], -R_36[0,0])
            q6_alt = np.arctan2(-R_36[2,1], R_36[2,2])
            solutions.append([q1,q2,q3,q4_alt,q5_alt,q6_alt])
        else:
            # singularity
            q6 = 0
            if R_36[2,0] == -1:
                q5 = np.pi/2
                q4 = q6 + np.arctan2(R_36[0,1], R_36[0,2])
            else:
                q5 = -np.pi/2
                q4 = -q6 + np.arctan2(-R_36[0,1], -R_36[0,2])
            solutions.append([q1,q2,q3,q4,q5,q6])
    return solutions