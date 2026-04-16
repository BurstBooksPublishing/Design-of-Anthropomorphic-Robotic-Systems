def mobility(N, f_list, lam=6, ground_included=True):
    """
    Calculate the mobility (degrees of freedom) of a mechanical system.
    
    Args:
        N: Number of bodies in the system
        f_list: List of joint degrees of freedom (length J)
        lam: Spatial degrees of freedom (6 for 3D, 3 for 2D)
        ground_included: Whether ground body is included in N
    
    Returns:
        Mobility value (degrees of freedom)
    """
    J = len(f_list)
    
    if ground_included:
        # Grounded mechanism: mobility = sum of joint freedoms - constraints
        return lam * (N - 1 - J) + sum(f_list)
    else:
        # Free-floating mechanism: add global rigid body modes
        return lam + sum(f_list)

# Example usage:
# f = [1, 1, 1, 3, 1, 1, 1, 3]  # 8 joints with varying DoFs
# mobility(9, f, ground_included=True)  # 9 bodies including ground