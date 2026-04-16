import numpy as np
def pairwise_poe_to_dh(z1_p, z1_dir, z2_p, z2_dir, tol=1e-9):
    # z1_p,z2_p: points on axes; z?_dir: unit direction vectors
    # compute shortest common normal between two lines
    u, v = z1_dir, z2_dir
    w0 = z1_p - z2_p
    denom = 1 - (u.dot(v))**2
    if abs(denom) < tol:
        return None, "parallel_or_coincident"  # non-unique common normal
    s = ( (v.dot(w0))*(u.dot(v)) - u.dot(w0) ) / denom
    t = ( (u.dot(w0))*(u.dot(v)) - v.dot(w0) ) / denom
    pa = z1_p + s*u
    pb = z2_p + t*v
    a = np.linalg.norm(pb - pa)                         # DH a
    n = (pb - pa) / max(a, tol)                        # common normal dir
    alpha = np.arctan2(np.dot(np.cross(u, v), n), np.dot(u, v)) # signed angle
    # choose d, theta by projecting origins (requires global frame choices)
    # here return geometric values; caller computes signed d/theta per frame choices
    return dict(a=a, alpha=alpha, point_on_z1=pa, point_on_z2=pb), "ok"