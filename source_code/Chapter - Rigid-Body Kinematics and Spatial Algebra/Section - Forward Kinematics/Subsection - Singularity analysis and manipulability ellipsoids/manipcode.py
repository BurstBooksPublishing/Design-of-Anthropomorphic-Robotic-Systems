import numpy as np
# J: m x n Jacobian (replace with robot-specific spatial Jacobian)
J = np.loadtxt("J_sample.csv", delimiter=",")  # load precomputed Jacobian
# optional weighting for SE(3) tasks
alpha = 0.1
W = np.diag([1,1,1,alpha,alpha,alpha]) if J.shape[0]==6 else np.eye(J.shape[0])
Jw = np.sqrt(W) @ J  # weighted Jacobian
U, S, Vt = np.linalg.svd(Jw, full_matrices=False)
axes_lengths = S  # singular values are principal axis lengths for unit joint norm
yoshikawa = np.prod(S)           # product of singular values
condition = S.max() / S.min()    # condition number
# brief outputs
print("axes:", axes_lengths)      # principal axes lengths
print("Yoshikawa:", yoshikawa)
print("condition number:", condition)
# compute principal directions in task space: columns of U