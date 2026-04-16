def tree_rnea(root, children, parent, X, S, qd, qdd, I, f_ext, v0, a0):
    # forward: compute v,a
    v = dict(); a = dict()
    v[root]=v0; a[root]=a0
    for i in topological_order(root, children):
        p=parent[i]
        if p is None: continue
        v[i]=X[p,i].dot(v[p]) + S[i]*qd[i]     # spatial velocity
        a[i]=X[p,i].dot(a[p]) + S[i]*qdd[i] + S_dot(i,qd)*qd[i] # spatial acc
    # backward: accumulate forces and torques
    f = {i: np.zeros(6) for i in v}
    tau = {}
    for i in reversed(topological_order(root, children)):
        f_i = I[i].dot(a[i]) + crm(v[i]).T.dot(I[i].dot(v[i])) - f_ext.get(i,0)
        for j in children[i]:
            f_i += X[j,i].T.dot(f[j])   # accumulate child forces
        f[i]=f_i
        tau[i]=S[i].T.dot(f_i)         # joint torque
    return tau, f