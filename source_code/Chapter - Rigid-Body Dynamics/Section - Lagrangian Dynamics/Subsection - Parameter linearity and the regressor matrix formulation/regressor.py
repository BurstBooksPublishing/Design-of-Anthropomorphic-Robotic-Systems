import numpy as np
# q, qd, qdd: joint states; links: list of link objects with methods below
# link.J(q): spatial Jacobian (6xn); link.Phi[k]: 6x6 basis inertia matrices
def assemble_regressor(q,qd,qdd,links):
    n = q.size
    Y = np.zeros((n, 10*len(links)))
    for i,link in enumerate(links):
        J = link.J(q)            # spatial Jacobian for link i
        # for each basis inertia element, compute contribution columns
        for k in range(10):
            Phi = link.Phi[k]    # basis spatial inertia (6x6)
            # inertia-term contribution: J^T Phi J * qdd  (coefficient wrt pi_k)
            Mcoeff = J.T @ Phi @ J
            # Coriolis/Centrifugal terms require Christoffel-like extraction;
            # here we form a quadratic term basis via finite differences or analytic expressions.
            col = Mcoeff @ qdd + coriolis_coeff(J,Phi,q,qd) + gravity_coeff(J,Phi,q)
            Y[:,10*i+k] = col      # place column into global regressor
    return Y