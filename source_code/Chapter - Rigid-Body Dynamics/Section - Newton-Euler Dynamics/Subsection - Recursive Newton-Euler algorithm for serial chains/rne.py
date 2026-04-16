import numpy as np

def crm(v):  # motion cross-product operator (6x6)
    # v: 6-vector [w; v]
    # returns 6x6 matrix such that crm(v)*u = v x u
    w = v[:3]; vlin = v[3:]
    W = hat(w); V = hat(vlin)
    return np.block([[W, np.zeros((3,3))],[V, W]])

def crf(v):  # force cross-product operator (6x6)
    return -crm(v).T

def rne_serial(chain, q, qd, qdd, g): 
    n = len(chain)
    v = [np.zeros(6) for _ in range(n+1)]
    a = [np.zeros(6) for _ in range(n+1)]
    f = [np.zeros(6) for _ in range(n+1)]
    tau = np.zeros(n)
    a[0] = -g  # base spatial accel
    # outward pass
    for i in range(n):
        X = chain[i]['X'](q[i])    # 6x6 adjoint from parent->child
        S = chain[i]['S']         # 6-vector screw in child frame
        v[i+1] = X @ v[i] + S*qd[i]
        a[i+1] = X @ a[i] + S*qdd[i] + crm(v[i+1]) @ (S*qd[i])
    # inward pass
    for i in reversed(range(n)):
        I = chain[i]['I']         # 6x6 spatial inertia
        b = I @ a[i+1] + crf(v[i+1]) @ (I @ v[i+1])  # bias force
        f[i] = b
        for j in chain[i].get('children',[]):       # add mapped child wrenches
            f[i] += chain[j]['X'].T @ f[j]
        tau[i] = chain[i]['S'].T @ f[i]
    return tau