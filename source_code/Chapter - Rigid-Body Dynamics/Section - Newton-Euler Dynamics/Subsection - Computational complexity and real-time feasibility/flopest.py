def rnea_flops(n, c_per_joint=200):
    # returns estimated FLOPs for RNEA on n joints
    return c_per_joint * n

def contact_schur_flops(n, m, c1=200, c2=1.0/3):
    # c2*m^3 approximates dense factorization FLOPs; c1*n linear term
    return c1*n + c2*(m**3) + 10*(m**2)  # small quadratic assembly term

# example usage
n = 30; m = 4
print(rnea_flops(n))               # per-tick RNEA FLOPs
print(contact_schur_flops(n, m))   # per-tick constrained dynamics FLOPs