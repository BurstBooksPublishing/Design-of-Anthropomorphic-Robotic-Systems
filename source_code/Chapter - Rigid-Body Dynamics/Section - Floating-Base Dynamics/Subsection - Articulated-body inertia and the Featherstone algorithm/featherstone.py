import numpy as np
# Inputs: tree nodes with I[i], X_parent[i], S[i], v[i], tau[i], fext[i], parent index p[i]
# Outputs: qdd[i], a[i]
def aba(nodes, root):
    # upward pass
    IA, pA, U, D, u = {}, {}, {}, {}, {}
    order = reversed(nodes.postorder(root))
    for i in order:
        IA[i] = nodes.I[i].copy()                    # spatial inertia
        pA[i] = crm(nodes.v[i]) @ nodes.I[i] @ nodes.v[i] - nodes.fext[i]
        for j in nodes.children(i):
            X = nodes.X[j]                          # child->i transform
            IA[i] += X.T @ IA[j] @ X
            pA[i] += X.T @ pA[j]
        U[i] = IA[i] @ nodes.S[i]
        D[i] = nodes.S[i].T @ U[i]
        u[i] = nodes.tau[i] - nodes.S[i].T @ pA[i]
        # rank-1 elimination
        invD = np.linalg.inv(D[i])                  # scalar for 1-DOF
        IA[i] -= U[i] @ (invD @ U[i].T)
        pA[i] += U[i] @ (invD @ u[i])
    # root acceleration
    a = {}
    a[root] = np.linalg.solve(IA[root], -pA[root])
    # downward pass
    for i in nodes.preorder(root):
        if i==root: continue
        parent = nodes.parent(i)
        qdd = np.linalg.solve(D[i], (u[i] - U[i].T @ a[parent]))  # joint accel
        a[i] = nodes.X[i] @ a[parent] + nodes.S[i] @ qdd + nodes.c[i]
        nodes.qdd[i] = qdd
    nodes.a = a