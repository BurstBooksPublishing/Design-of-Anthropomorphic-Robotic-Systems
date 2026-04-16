import sympy as sp
n=3
q=sp.symbols('q0:%d'%n); qdot=sp.symbols('qdot0:%d'%n)
m=sp.symbols('m0:%d'%n); l=sp.symbols('l0:%d'%n); I=sp.symbols('I0:%d'%n)
g0=sp.symbols('g0')
# kinematics: positions of link COMs in plane (recursive) -- placeholder expressions
# build kinetic energy T and potential V for planar revolute chain
T=0; V=0
x=0; y=0; th=0
for i in range(n):
    th += q[i]
    # COM position (assume COM at l_i/2 along link)
    xi = x + (l[i]/2)*sp.cos(th); yi = y + (l[i]/2)*sp.sin(th)
    # velocity via Jacobian
    Ji = sp.Matrix([sp.diff(xi, qj) for qj in q]).T.row_join(sp.Matrix([sp.diff(yi, qj) for qj in q]).T)
    vi = Ji*sp.Matrix(qdot)
    T += sp.Rational(1,2)*m[i]*(vi.dot(vi)) + sp.Rational(1,2)*I[i]*sum(qdot)**2  # planar inertia approx
    V += m[i]*g0*yi
    # advance link end
    x += l[i]*sp.cos(th); y += l[i]*sp.sin(th)
# mass matrix
M = sp.Matrix([[sp.diff(sp.diff(T, qdot[i]), qdot[j]) for j in range(n)] for i in range(n)])
# Christoffel and Coriolis
Gamma = [[[sp.Rational(1,2)*(sp.diff(M[i,j], q[k])+sp.diff(M[i,k], q[j])-sp.diff(M[j,k], q[i]))
            for k in range(n)] for j in range(n)] for i in range(n)]
C = sp.Matrix([[sum(Gamma[i][j][k]*qdot[k] for k in range(n)) for j in range(n)] for i in range(n)])
g_vec = sp.Matrix([sp.diff(V, qi) for qi in q])
# outputs: M, C, g_vec